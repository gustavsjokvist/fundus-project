"""Write per-image logits for a CSV (optionally one split) so evaluate.py can run without a GPU."""
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data import FundusDataset
from train import build_model, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split", default=None, help="train/val/test; omit to use every row (external sets)")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model()
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    model.to(device)
    ds = FundusDataset(a.csv, a.split, a.size, False)
    logits, labels = predict(model, DataLoader(ds, batch_size=a.bs, num_workers=a.workers), device)
    out = ds.df[["image_path"]].copy()
    out["label"] = labels.astype(int)
    out["logit"] = logits
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out)} predictions to {a.out}")


if __name__ == "__main__":
    main()
