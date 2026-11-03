"""
eval_utils.py  (v2)
--------------------
Shared, dependency-light evaluation helpers that fix the statistical weaknesses of the
Week 2-6 evaluation:

  * QWK (quadratic weighted kappa) - the standard agreement metric in short-answer scoring,
    which the original reports lacked.
  * Errors normalised by max marks (a 1-mark miss on a 5-mark question is not the same as on
    a 10-mark question, so "within +/-1 mark" was not comparable across questions).
  * CLUSTER bootstrap: answers to the same question are not independent, so the bootstrap
    resamples whole QUESTIONS.  (The Week 6 bootstrap resampled rows, giving intervals that
    are too narrow.)
  * Risk-coverage / selective prediction: how much error is avoided as a function of how many
    answers are sent to a human - instead of one post-hoc threshold.
  * Leave-one-question-out split conformal intervals for the score.
"""

from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr


# --------------------------------------------------------------------------- point metrics
def qwk(y_true: Sequence[float], y_pred: Sequence[float], max_marks: Sequence[float],
        levels: int = 10) -> float:
    """
    Quadratic weighted kappa on scores normalised to the fraction of max marks and binned
    into `levels`+1 ordinal categories (0..levels).  Binning on the FRACTION makes 5- and
    10-mark questions comparable.  Returns nan if undefined.
    """
    yt = np.rint(np.asarray(y_true, float) / np.asarray(max_marks, float) * levels).astype(int)
    yp = np.rint(np.asarray(y_pred, float) / np.asarray(max_marks, float) * levels).astype(int)
    k = levels + 1
    O = np.zeros((k, k))
    for a, b in zip(yt, yp):
        O[np.clip(a, 0, levels), np.clip(b, 0, levels)] += 1
    if O.sum() == 0:
        return float("nan")
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], float) / (k - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    denom = (W * E).sum()
    return float("nan") if denom == 0 else float(1 - (W * O).sum() / denom)


def metrics(y_true, y_pred, max_marks) -> Dict[str, float]:
    y, p, m = (np.asarray(v, float) for v in (y_true, y_pred, max_marks))
    err = np.abs(y - p)
    ok = len(set(y)) > 1 and len(set(p)) > 1
    return {
        "n": int(len(y)),
        "MAE": round(float(err.mean()), 3),
        "nMAE_pct_of_max": round(float((err / m).mean() * 100), 2),
        "RMSE": round(float(np.sqrt(((y - p) ** 2).mean())), 3),
        "pearson_r": round(float(pearsonr(y, p)[0]), 3) if ok else None,
        "spearman_rho": round(float(spearmanr(y, p)[0]), 3) if ok else None,
        "QWK": round(qwk(y, p, m), 3),
        "within_10pct_of_max": round(float((err / m <= 0.10 + 1e-9).mean()), 3),
        "within_1_mark": round(float((err <= 1.0).mean()), 3),
    }


# ------------------------------------------------------------------------ cluster bootstrap
def cluster_bootstrap(values: Sequence[float], groups: Sequence[str], stat: Callable = np.mean,
                      n_boot: int = 5000, seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    """Percentile CI of stat(values) resampling whole groups (e.g. questions) with replacement."""
    values = np.asarray(values, float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    by_g = [values[groups == g] for g in uniq]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        out.append(stat(np.concatenate([by_g[i] for i in pick])))
    lo, hi = np.percentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": round(float(stat(values)), 3), "ci_low": round(float(lo), 3),
            "ci_high": round(float(hi), 3), "n_groups": int(len(uniq))}


def paired_cluster_bootstrap(err_a: Sequence[float], err_b: Sequence[float], groups: Sequence[str],
                             n_boot: int = 5000, seed: int = 0) -> Dict[str, object]:
    """
    CI for mean(err_a) - mean(err_b), resampling questions.  Positive => B has LOWER error
    (B better).  `excludes_zero` is the only thing that licenses the word "better".
    """
    d = np.asarray(err_a, float) - np.asarray(err_b, float)
    r = cluster_bootstrap(d, groups, n_boot=n_boot, seed=seed)
    r["excludes_zero"] = bool(r["ci_low"] > 0 or r["ci_high"] < 0)
    return r


# -------------------------------------------------------------------- selective prediction
def risk_coverage(errors: Sequence[float], confidence: Sequence[float]) -> Dict[str, object]:
    """
    Sort answers by confidence (high first) and report mean error of the retained set at each
    coverage level.  AURC = area under that curve (lower is better).  `random_aurc` is the
    expected AURC of an uninformative confidence (= overall mean error), so
    `aurc < random_aurc` is evidence the confidence is useful.
    """
    e = np.asarray(errors, float)
    c = np.asarray(confidence, float)
    order = np.argsort(-c, kind="stable")
    cum = np.cumsum(e[order]) / np.arange(1, len(e) + 1)
    cov = np.arange(1, len(e) + 1) / len(e)
    aurc = float(np.trapezoid(cum, cov)) if hasattr(np, "trapezoid") else float(np.trapz(cum, cov))
    return {"coverage": cov.tolist(), "risk": cum.tolist(), "aurc": round(aurc, 4),
            "random_aurc": round(float(e.mean()), 4),
            "oracle_aurc": round(_oracle_aurc(e), 4)}


def _oracle_aurc(e: np.ndarray) -> float:
    s = np.sort(e)
    cum = np.cumsum(s) / np.arange(1, len(s) + 1)
    cov = np.arange(1, len(s) + 1) / len(s)
    return float(np.trapezoid(cum, cov)) if hasattr(np, "trapezoid") else float(np.trapz(cum, cov))


def risk_at_coverage(rc: Dict[str, object], coverage: float) -> float:
    cov = np.asarray(rc["coverage"])
    return float(np.asarray(rc["risk"])[min(len(cov) - 1, int(np.searchsorted(cov, coverage)))])


# ------------------------------------------------------------------------------ conformal
def loqo_conformal(errors_norm: Sequence[float], groups: Sequence[str], alpha: float = 0.10) -> Dict[str, object]:
    """
    Leave-one-question-out split conformal for |score error| / max_marks.
    For each question, the half-width is the finite-sample-corrected (1-alpha) quantile of the
    normalised absolute errors of ALL OTHER questions.  Returns per-row half-widths (as a
    fraction of max marks) and the empirical coverage on the held-out question.
    """
    e = np.asarray(errors_norm, float)
    g = np.asarray(groups)
    half = np.zeros(len(e))
    for q in np.unique(g):
        rest = e[g != q]
        k = int(np.ceil((len(rest) + 1) * (1 - alpha)))
        half[g == q] = np.sort(rest)[min(k, len(rest)) - 1]
    covered = e <= half + 1e-12
    return {"half_width_frac": half.tolist(), "empirical_coverage": round(float(covered.mean()), 3),
            "target_coverage": 1 - alpha, "mean_half_width_frac": round(float(half.mean()), 3)}
