"""
finetune_cross_encoder.py  (v2)  -  RUN ON A GPU (Colab/Kaggle) - plain PyTorch loop, no Trainer-API drift
--------------------------------------------------------------------------------------------------------------
Two fine-tuning tasks on PUBLIC human-graded data (see load_external.py / export_hf_datasets.py):

  --task regression  (reference_answer, student_answer) -> normalised score in [0,1]   (Mohler)
        start from any encoder, e.g. microsoft/deberta-v3-base or cross-encoder/nli-deberta-v3-base
  --task nli3        premise = student answer, hypothesis = reference answer ->
                     {entailment=correct, contradiction=contradictory, neutral=incorrect}      (SciEntsBank)
        start from an NLI checkpoint (cross-encoder/nli-deberta-v3-base): the pretrained 3-way head is REUSED, so this
        domain-adapts the contradiction detector that replaces the hand-written fact-binding rules.

Validation is QUESTION-GROUPED (dev questions never appear in train).  The best dev epoch is saved to --out.

  # regression on Mohler
  python src/training/finetune_cross_encoder.py --task regression --data data/external/mohler_open_ended.csv \
      --model microsoft/deberta-v3-base --epochs 4 --lr 2e-5 --out models/mohler_ce
  # NLI domain adaptation on SciEntsBank
  python src/training/finetune_cross_encoder.py --task nli3 --data data/external/scientsbank_train.csv \
      --model cross-encoder/nli-deberta-v3-base --epochs 2 --lr 1e-5 --out models/sciq_nli
  python ... --dry-run     # data pipeline only, no torch/model needed

After training, point the benchmark at it:  G2_NLI_MODEL=models/sciq_nli python src/evaluation/run_benchmark.py --graders nli_bi
HONESTY: this file could only be syntax-checked and dry-run in the sandbox that wrote it (no torch there).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
from load_external import load_mohler, load_scientsbank   # noqa: E402

LABEL3 = ["correct", "contradictory", "incorrect"]
NLI_NAME_FOR = {"correct": "entail", "contradictory": "contra", "incorrect": "neutral"}


def build_examples(rows, task):
    """-> list of (text_a, text_b, target, group)"""
    out = []
    for r in rows:
        if task == "regression":
            out.append((r["reference_answer"], r["student_answer"], float(r["human_score"]) / float(r["max_marks"]), r["qid"]))
        else:
            out.append((r["student_answer"], r["reference_answer"], r["label3"], r["qid"]))
    return out


def group_split(examples, dev_frac=0.15, seed=0):
    groups = sorted({e[3] for e in examples})
    random.Random(seed).shuffle(groups)
    dev_g = set(groups[:max(1, int(len(groups) * dev_frac))])
    return [e for e in examples if e[3] not in dev_g], [e for e in examples if e[3] in dev_g]


def load_rows(task, path):
    return load_mohler(path) if task == "regression" else load_scientsbank(path, "train")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["regression", "nli3"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="cross-encoder/nli-deberta-v3-base")
    ap.add_argument("--out", default="models/finetuned")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    ex = build_examples(load_rows(a.task, a.data), a.task)
    if a.max_examples:
        random.Random(a.seed).shuffle(ex)
        ex = ex[:a.max_examples]
    train, dev = group_split(ex, seed=a.seed)
    print(f"{len(train)} train / {len(dev)} dev examples; dev questions are disjoint from train questions")
    print("example:", train[0][:3])
    if a.dry_run:
        return {"train": len(train), "dev": len(dev)}

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.task == "regression":
        model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=1, ignore_mismatched_sizes=True)
        to_target = lambda t: float(t)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(a.model)
        id2 = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
        idx = {lab: next(i for i, n in id2.items() if NLI_NAME_FOR[lab] in n) for lab in LABEL3}
        to_target = lambda t: idx[t]
        inv = {v: k for k, v in idx.items()}
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(train) + a.bs - 1) // a.bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")

    def batches(data, shuffle):
        order = list(range(len(data)))
        if shuffle:
            random.shuffle(order)
        for i in range(0, len(order), a.bs):
            b = [data[j] for j in order[i:i + a.bs]]
            enc = tok([x[0] for x in b], [x[1] for x in b], truncation=True, max_length=a.max_len, padding=True, return_tensors="pt").to(device)
            y = torch.tensor([to_target(x[2]) for x in b], dtype=torch.float if a.task == "regression" else torch.long).to(device)
            yield enc, y

    def evaluate():
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for enc, y in batches(dev, False):
                lg = model(**enc).logits
                P += (lg.squeeze(-1) if a.task == "regression" else lg.argmax(-1)).float().cpu().tolist()
                Y += y.cpu().tolist()
        model.train()
        if a.task == "regression":
            from eval_utils import qwk
            P = np.clip(P, 0, 1)
            return {"pearson": float(np.corrcoef(P, Y)[0, 1]), "QWK": qwk(np.array(Y) * 10, P * 10, [10] * len(Y)),
                    "MAE_frac": float(np.mean(np.abs(P - np.array(Y))))}, float(np.corrcoef(P, Y)[0, 1])
        from sklearn.metrics import f1_score
        f1 = f1_score(Y, P, average="macro")
        return {"macro_F1": float(f1)}, float(f1)

    model.train()
    best = -1e9
    for ep in range(a.epochs):
        for step, (enc, y) in enumerate(batches(train, True)):
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                lg = model(**enc).logits
                loss = torch.nn.functional.mse_loss(lg.squeeze(-1).float(), y) if a.task == "regression" else torch.nn.functional.cross_entropy(lg.float(), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            if step % 50 == 0:
                print(f"epoch {ep + 1} step {step} loss {loss.item():.4f}", flush=True)
        m, score = evaluate()
        print(f"== epoch {ep + 1} dev: {json.dumps(m)}", flush=True)
        if score > best:
            best = score
            Path(a.out).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(a.out); tok.save_pretrained(a.out)
            json.dump({"task": a.task, "base": a.model, "epoch": ep + 1, "dev": m}, open(Path(a.out) / "finetune_info.json", "w"), indent=2)
            print(f"   saved best -> {a.out}")
    return {"best_dev_score": best}


if __name__ == "__main__":
    main()
