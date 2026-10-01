"""Build a standard CSV (image_path, grade, label[, split]) from a dataset's own label file(s).

Examples
  python make_csv.py --images data/aptos/train_images --labels data/aptos/train.csv \
      --id-col id_code --grade-col diagnosis --ext .png --threshold 2 --dedup --split --out csv/aptos.csv

  # IDRiD: flatten first with prepare_idrid.py (its train/test folders reuse file names)
  python make_csv.py --images data/idrid_flat/images --labels data/idrid_flat/labels.csv \
      --id-col id --grade-col grade --ext .jpg --threshold 2 --dedup --out csv/external.csv

Several --labels files are concatenated (for datasets that split their grades over several CSVs but share one
image folder with unique names).

label = 1 if grade >= threshold ("referable" DR when threshold=2), else 0.
CHECK each dataset's grading scheme before using it as an external set: grade definitions differ
between datasets, and a mismatch will look like "domain shift" when it is really label noise.

--dedup removes duplicate images (identical perceptual hash). APTOS 2019 contains duplicates under different ids;
left in, they end up on both sides of the train/test split and inflate the internal score. Duplicate groups whose
grades disagree are dropped entirely, since we cannot tell which label is right.
"""
import argparse, os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def read_labels(paths, id_col, grade_col):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]  # IDRiD CSVs have empty trailing columns
    df.columns = df.columns.str.strip()
    return df.dropna(subset=[id_col, grade_col])


def dedup(out):
    import imagehash
    hashes = [str(imagehash.phash(Image.open(p))) for p in tqdm(out["image_path"], desc="phash")]
    out = out.assign(phash=hashes)
    n_grades = out.groupby("phash")["grade"].transform("nunique")
    n_conflict = int((n_grades > 1).sum())
    kept = out[n_grades == 1].drop_duplicates("phash")
    n_dupes = len(out) - n_conflict - len(kept)
    print(f"dedup: removed {n_dupes} duplicate copies and {n_conflict} images in groups with conflicting grades")
    return kept.drop(columns="phash").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True, nargs="+", help="one or more label CSVs, concatenated")
    ap.add_argument("--id-col", required=True)
    ap.add_argument("--grade-col", required=True)
    ap.add_argument("--ext", default="", help="file extension to append to ids, e.g. .png")
    ap.add_argument("--threshold", type=int, default=2)
    ap.add_argument("--dedup", action="store_true", help="drop duplicate images by perceptual hash")
    ap.add_argument("--split", action="store_true", help="add stratified train/val/test split (70/15/15)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = read_labels(a.labels, a.id_col, a.grade_col)
    out = pd.DataFrame({
        "image_path": df[a.id_col].astype(str).str.strip().map(lambda s: os.path.join(a.images, s + a.ext)),
        "grade": df[a.grade_col].astype(int),
    })
    exists = out["image_path"].map(os.path.exists)
    if (~exists).any():
        print(f"Warning: dropping {(~exists).sum()} rows with missing image files")
    out = out[exists].reset_index(drop=True)
    if a.dedup:
        out = dedup(out)
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
