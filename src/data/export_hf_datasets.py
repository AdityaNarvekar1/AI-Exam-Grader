"""
export_hf_datasets.py  (v2)  -  RUN THIS ON YOUR MACHINE (needs internet + `pip install datasets`)
-----------------------------------------------------------------------------------------------------
Downloads the two public short-answer datasets from the Hugging Face Hub and writes plain CSVs into
data/external/ so that nothing else in the project needs the `datasets` library.

    pip install datasets
    python src/data/export_hf_datasets.py            # both
    python src/data/export_hf_datasets.py --only mohler

It PRINTS the structure it found (splits, columns, first row).  If anything differs from the dataset
card (names/configs change), paste that printed block back to Claude - the loaders in load_external.py
are the only thing that needs adjusting.
"""
import argparse
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "data" / "external"


def dump(ds_dict, prefix):
    for split, ds in ds_dict.items():
        p = OUT / f"{prefix}_{split}.csv"
        ds.to_csv(str(p), index=False)
        print(f"  {split}: {len(ds)} rows, columns={ds.column_names} -> {p}")
        print(f"  first row: {ds[0]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["mohler", "scientsbank", "beetle"])
    a = ap.parse_args()
    from datasets import load_dataset
    OUT.mkdir(parents=True, exist_ok=True)
    if a.only in (None, "mohler"):
        print("== Mohler (Viswa45/ASAG, config 'cleaned')")
        try:
            dump(load_dataset("Viswa45/ASAG", "cleaned"), "mohler")
        except Exception as e:
            print("  config 'cleaned' failed (", e, ") - trying default config")
            dump(load_dataset("Viswa45/ASAG"), "mohler")
    if a.only in (None, "scientsbank"):
        print("== SciEntsBank (nkazi/SciEntsBank)")
        dump(load_dataset("nkazi/SciEntsBank"), "scientsbank")
    if a.only == "beetle":
        print("== Beetle (nkazi/Beetle)")
        dump(load_dataset("nkazi/Beetle"), "beetle")


if __name__ == "__main__":
    main()
