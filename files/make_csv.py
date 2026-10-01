"""Build a standard CSV (image_path, grade, label[, split]) from a dataset's own label file.

Examples
  python make_csv.py --images data/aptos/train_images --labels data/aptos/train.csv \
      --id-col id_code --grade-col diagnosis --ext .png --threshold 2 --split --out csv/aptos.csv

  python make_csv.py --images data/idrid/images --labels data/idrid/labels.csv \
      --id-col "Image name" --grade-col "Retinopathy grade" --ext .jpg --threshold 2 --out csv/external.csv

label = 1 if grade >= threshold ("referable" DR when threshold=2), else 0.
CHECK each dataset's grading scheme before using it as an external set: grade definitions differ
between datasets, and a mismatch will look like "domain shift" when it is really label noise.
"""
import argparse, os
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--id-col", required=True)
    ap.add_argument("--grade-col", required=True)
    ap.add_argument("--ext", default="", help="file extension to append to ids, e.g. .png")
    ap.add_argument("--threshold", type=int, default=2)
    ap.add_argument("--split", action="store_true", help="add stratified train/val/test split (70/15/15)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.labels)
    out = pd.DataFrame({
        "image_path": df[a.id_col].astype(str).map(lambda s: os.path.join(a.images, s + a.ext)),
        "grade": df[a.grade_col].astype(int),
    })
    exists = out["image_path"].map(os.path.exists)
    if (~exists).any():
        print(f"Warning: dropping {(~exists).sum()} rows with missing image files")
    out = out[exists].reset_index(drop=True)
    out["label"] = (out["grade"] >= a.threshold).astype(int)

    if a.split:
        tr, rest = train_test_split(out.index, test_size=0.30, stratify=out["label"], random_state=a.seed)
        va, te = train_test_split(rest, test_size=0.50, stratify=out.loc[rest, "label"], random_state=a.seed)
        out["split"] = ""
        out.loc[tr, "split"] = "train"
        out.loc[va, "split"] = "val"
        out.loc[te, "split"] = "test"

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"Wrote {len(out)} rows to {a.out}; positive rate = {out['label'].mean():.3f}")
    if a.split:
        print(out.groupby("split")["label"].agg(["count", "mean"]))


if __name__ == "__main__":
    main()
