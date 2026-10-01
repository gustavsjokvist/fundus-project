"""Fine-tune an ImageNet-pretrained ResNet-50 for binary referable-DR classification.
Model selection uses validation AUROC only. The test split and the external set are never touched here."""
import argparse, os, random
import numpy as np
import torch, torch.nn as nn
import torchvision
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import FundusDataset


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_model():
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).squeeze(1).float().cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/resnet50")
    a = ap.parse_args()

    seed_all(a.seed)
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    tr = DataLoader(FundusDataset(a.csv, "train", a.size, True), batch_size=a.bs, shuffle=True,
                    num_workers=a.workers, drop_last=True)
    va = DataLoader(FundusDataset(a.csv, "val", a.size, False), batch_size=a.bs, num_workers=a.workers)

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    lossf = nn.BCEWithLogitsLoss()
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = -1.0
    for ep in range(a.epochs):
        model.train()
        run = 0.0
        for x, y in tqdm(tr, desc=f"epoch {ep+1}/{a.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = lossf(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            run += loss.item()
        sched.step()
        lg, lb = predict(model, va, device)
        auc = roc_auc_score(lb, lg)
        print(f"epoch {ep+1}: train loss {run/len(tr):.4f}  val AUROC {auc:.4f}")
        if auc > best:
            best = auc
            torch.save(model.state_dict(), os.path.join(a.out, "best.pt"))
    print(f"best val AUROC {best:.4f}; checkpoint saved to {a.out}/best.pt")


if __name__ == "__main__":
    main()
