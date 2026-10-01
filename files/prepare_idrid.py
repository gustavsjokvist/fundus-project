"""Flatten IDRiD 'Disease Grading' into one image folder + one label CSV, ready for make_csv.py.

IDRiD reuses file names between its training and testing sets (IDRiD_001.jpg exists in both, as different
images), so the two cannot share a folder as-is. This pairs each grading CSV with its image folder by the
word train/test in their paths and prefixes ids with the subset name.

  python prepare_idrid.py --root "data/IDRiD/B. Disease Grading" --out data/idrid_flat
  -> data/idrid_flat/images/{train,test}_IDRiD_xxx.jpg and data/idrid_flat/labels.csv (id, grade)
"""
import argparse, glob, os, shutil
import pandas as pd


def subset_of(path):
    p = path.lower()
    hits = [s for s in ("train", "test") if s in p]
    return hits[0] if len(hits) == 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing the IDRiD grading CSVs and images (searched recursively)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    csvs = {}
    for p in glob.glob(os.path.join(a.root, "**", "*.csv"), recursive=True):
        cols = pd.read_csv(p, nrows=0).columns.str.strip()
        if "Retinopathy grade" in cols and subset_of(os.path.relpath(p, a.root)):
            csvs[subset_of(os.path.relpath(p, a.root))] = p
    jpgs = glob.glob(os.path.join(a.root, "**", "*.jpg"), recursive=True)
    # the full IDRiD download also has Localization/Segmentation folders with train/test subfolders and the same
    # file names; use only the Disease Grading images when they can be told apart
    grading = [p for p in jpgs if "grading" in os.path.relpath(p, a.root).lower()]
    images = {}
    for p in grading or jpgs:
        s = subset_of(os.path.relpath(os.path.dirname(p), a.root))
        if s:
            images.setdefault(s, {})[os.path.splitext(os.path.basename(p))[0]] = p
    print("label files:", csvs)
    print("images found:", {s: len(v) for s, v in images.items()})
    if set(csvs) != {"train", "test"}:
        raise SystemExit("could not find one train and one test grading CSV under --root")

    os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
    rows = []
    for subset, csv in csvs.items():
        df = pd.read_csv(csv)
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=["Image name", "Retinopathy grade"])
        for name, grade in zip(df["Image name"].astype(str).str.strip(), df["Retinopathy grade"].astype(int)):
            src = images.get(subset, {}).get(name)
            if src is None:
                print(f"missing image for {subset}/{name}")
                continue
            new_id = f"{subset}_{name}"
            dst = os.path.join(a.out, "images", new_id + ".jpg")
            if not os.path.exists(dst):
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError:  # e.g. Windows without symlink rights
                    shutil.copy(src, dst)
            rows.append((new_id, grade))
    out = pd.DataFrame(rows, columns=["id", "grade"])
    out.to_csv(os.path.join(a.out, "labels.csv"), index=False)
    print(f"wrote {len(out)} labels to {a.out}/labels.csv; grade counts:", out["grade"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
