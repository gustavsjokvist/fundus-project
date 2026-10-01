"""One-time preprocessing: crop black borders, pad to square, resize, cache as JPEG, and compute image-quality metrics.

Examples (run the training dataset first: it fits the quality normalisation on its train split)
  python preprocess.py --csv csv/aptos.csv    --cache-dir cache/aptos    --stats csv/quality_stats.json --out csv/aptos_cached.csv
  python preprocess.py --csv csv/external.csv --cache-dir cache/external --stats csv/quality_stats.json --out csv/external_cached.csv

Quality metrics (computed on the cached 512px image, so they are comparable across camera resolutions):
  sharpness   variance of the Laplacian inside the field of view (low = blurry), stored as log
  brightness  mean grey level inside the field of view
  clip_frac   fraction of field-of-view pixels that are under- or over-exposed
  fov_frac    field-of-view area / image area (low = field cut off or badly framed)
quality = mean of signed z-scores clipped to +-3 (higher = better), normalised with statistics from the TRAIN split only.
"""
import argparse, json, os
from multiprocessing import Pool
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
from data import crop_black_borders, fov_mask, pad_to_square

METRICS = ["log_sharpness", "brightness", "clip_frac", "fov_frac"]


def quality_metrics(img: Image.Image) -> dict:
    grey = np.asarray(img.convert("L"), dtype=np.float32)
    field = ndimage.binary_fill_holes(fov_mask(img))  # fill: dark regions inside the field still count
    inner = ndimage.binary_erosion(field, iterations=8)  # keep the sharp field edge out of the Laplacian
    if inner.sum() < 100:
        inner = field if field.any() else np.ones_like(field)
    lap = ndimage.laplace(grey)
    vals = grey[inner]
    return {
        "log_sharpness": float(np.log(lap[inner].var() + 1e-6)),
        "brightness": float(vals.mean()),
        "clip_frac": float(((vals < 25) | (vals > 240)).mean()),
        "fov_frac": float(field.mean()),
    }


def process(job):
    src, dst, size = job
    img = pad_to_square(crop_black_borders(Image.open(src).convert("RGB"))).resize((size, size), Image.BICUBIC)
    img.save(dst, quality=95)
    return quality_metrics(img)


def quality_score(df, stats):
    # clip so a metric with tiny train variance cannot dominate the score
    z = {m: ((df[m] - stats[m]["mean"]) / stats[m]["std"]).clip(-3, 3) for m in METRICS}
    # sign convention: higher = better. Brightness is bad in both directions.
    return (z["log_sharpness"] - z["brightness"].abs() - z["clip_frac"] + z["fov_frac"]) / 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--stats", required=True, help="quality normalisation JSON; fitted on the train split if missing")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    os.makedirs(a.cache_dir, exist_ok=True)
    df["raw_path"] = df["image_path"]
    df["image_path"] = [os.path.join(a.cache_dir, os.path.splitext(os.path.basename(p))[0] + ".jpg")
                        for p in df["raw_path"]]
    jobs = list(zip(df["raw_path"], df["image_path"], [a.size] * len(df)))
    with Pool(a.workers) as pool:
        rows = list(tqdm(pool.imap(process, jobs, chunksize=8), total=len(jobs), desc="preprocess"))
    df = pd.concat([df, pd.DataFrame(rows)], axis=1)

    if os.path.exists(a.stats):
        with open(a.stats) as f:
            stats = json.load(f)
        print(f"loaded quality normalisation from {a.stats}")
    else:
        if "split" not in df.columns:
            raise SystemExit(f"{a.stats} not found and {a.csv} has no split column: run the training dataset first")
        tr = df[df["split"] == "train"]
        stats = {m: {"mean": float(tr[m].mean()), "std": float(tr[m].std() + 1e-8)} for m in METRICS}
        os.makedirs(os.path.dirname(a.stats) or ".", exist_ok=True)
        with open(a.stats, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"fitted quality normalisation on {len(tr)} train images -> {a.stats}")
    df["quality"] = quality_score(df, stats)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows to {a.out}")
    print(df[METRICS + ["quality"]].describe().loc[["mean", "std", "min", "max"]].round(3))


if __name__ == "__main__":
    main()
