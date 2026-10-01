"""Clinical-style evaluation of a binary classifier under domain shift. Needs only the prediction CSVs.

Protocol
  1. Fit temperature scaling on the VAL predictions (calibration).
  2. Pick an operating threshold on VAL that reaches the target sensitivity (default 90%).
  3. Freeze both, and report on internal TEST and EXTERNAL data:
     AUROC, sensitivity/specificity at the frozen threshold, specificity at fixed 90% sensitivity
     (re-picked on that dataset), ECE before/after temperature scaling, with bootstrap 95% CIs.
The gap between 'frozen threshold' and 're-picked threshold' on the external set is the practical cost of shift.

Quality gate (if the prediction CSVs have a 'quality' column from preprocess.py)
  4. Freeze a quality cutoff at the --abstain-frac quantile of VAL quality; images below it are rejected
     ("please retake the photo"). Report coverage and metrics on kept vs rejected images.
  5. Sweep the abstention fraction and compare against a confidence baseline that rejects the images whose
     logit is closest to the operating threshold.
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import roc_auc_score, roc_curve


def nll(logits, y, T):
    p = np.clip(expit(logits / T), 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_temperature(logits, y):
    return float(minimize_scalar(lambda t: nll(logits, y, t), bounds=(0.05, 20), method="bounded").x)


def ece(probs, y, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - probs[m].mean())
    return float(total)


def threshold_for_sensitivity(scores, y, target):
    """Highest threshold whose sensitivity is >= target (positives are score >= thr)."""
    pos = np.sort(scores[y == 1])
    k = int(np.floor((1 - target) * len(pos)))  # allow this many positives below the threshold
    return float(pos[k])


def sens_spec(scores, y, thr):
    pred = scores >= thr
    sens = pred[y == 1].mean() if (y == 1).any() else np.nan
    spec = (~pred)[y == 0].mean() if (y == 0).any() else np.nan
    return float(sens), float(spec)


def spec_at_sens(scores, y, target):
    return sens_spec(scores, y, threshold_for_sensitivity(scores, y, target))[1]


def bootstrap(fn, scores, y, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(fn(scores[i], y[i]))
    vals = np.array(vals, dtype=float)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def safe_auc(y, scores):
    return float(roc_auc_score(y, scores)) if 0 < y.sum() < len(y) else float("nan")


def fmt(v, ci):
    return f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def load(path):
    d = pd.read_csv(path)
    return d["logit"].to_numpy(float), d["label"].to_numpy(int)


def quality_gate(val, sets, thr, a):
    """val/sets hold DataFrames with logit, label, quality. Returns results dict and saves the sweep figure."""
    q_cut = float(np.quantile(val["quality"], a.abstain_frac))
    out = {"abstain_frac_on_val": a.abstain_frac, "quality_cutoff": q_cut, "sets": {}}
    print(f"\n== Quality gate: reject quality < {q_cut:.3f} (the {a.abstain_frac:.0%} quantile on val) ==")
    for name, d in sets.items():
        lg, y, q = d["logit"].to_numpy(float), d["label"].to_numpy(int), d["quality"].to_numpy(float)
        res = {"coverage": float((q >= q_cut).mean())}
        for part, m in [("kept", q >= q_cut), ("rejected", q < q_cut)]:
            if not m.any():
                res[part] = {"n": 0}
                continue
            sens, spec = sens_spec(lg[m], y[m], thr)
            res[part] = {"n": int(m.sum()), "prevalence": float(y[m].mean()), "auroc": safe_auc(y[m], lg[m]),
                         "frozen_threshold_sensitivity": sens, "frozen_threshold_specificity": spec}
        k = q >= q_cut
        res["kept"]["frozen_threshold_sensitivity_ci"] = bootstrap(lambda s, t: sens_spec(s, t, thr)[0], lg[k], y[k], a.boot)
        out["sets"][name] = res
        kp, rj = res["kept"], res["rejected"]
        print(f"{name}: coverage {res['coverage']:.1%}; kept n={kp['n']} AUROC {kp['auroc']:.3f} "
              f"sens {kp['frozen_threshold_sensitivity']:.3f} spec {kp['frozen_threshold_specificity']:.3f}"
              + (f" | rejected n={rj['n']} sens {rj['frozen_threshold_sensitivity']:.3f} "
                 f"spec {rj['frozen_threshold_specificity']:.3f}" if rj["n"] else " | rejected n=0"))

    # sweep: cutoffs frozen on val, applied to each set; x-axis is the fraction actually rejected on that set
    fracs = np.linspace(0, 0.30, 16)
    conf_val = np.abs(val["logit"].to_numpy(float) - thr)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sweep = {}
    for name, d in sets.items():
        lg, y = d["logit"].to_numpy(float), d["label"].to_numpy(int)
        crit = {"quality": (d["quality"].to_numpy(float), val["quality"].to_numpy(float)),
                "confidence": (np.abs(lg - thr), conf_val)}
        for crit_name, (score, score_val) in crit.items():
            xs, sens_l, spec_l = [], [], []
            for f in fracs:
                keep = score >= np.quantile(score_val, f) if f > 0 else np.ones(len(y), bool)
                if (y[keep] == 1).any() and (y[keep] == 0).any():
                    sens, spec = sens_spec(lg[keep], y[keep], thr)
                    xs.append(1 - keep.mean()); sens_l.append(sens); spec_l.append(spec)
            sweep[f"{name}/{crit_name}"] = {"rejected": xs, "sensitivity": sens_l, "specificity": spec_l}
            ls = "-" if name == "external" else "--"
            mk = "o" if crit_name == "quality" else "s"
            axes[0].plot(xs, sens_l, ls, marker=mk, ms=3, label=f"{name}, {crit_name} gate")
            axes[1].plot(xs, spec_l, ls, marker=mk, ms=3, label=f"{name}, {crit_name} gate")
    axes[0].axhline(a.target_sens, color="k", lw=0.8, ls=":")
    axes[0].set(xlabel="fraction of images rejected", ylabel="sensitivity @ frozen threshold", title="Sensitivity vs abstention")
    axes[1].set(xlabel="fraction of images rejected", ylabel="specificity @ frozen threshold", title="Specificity vs abstention")
    axes[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "quality_gate.png"), dpi=150); plt.close(fig)
    out["sweep"] = sweep
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--internal", required=True, help="predictions on the held-out internal test split")
    ap.add_argument("--external", required=True)
    ap.add_argument("--target-sens", type=float, default=0.90)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--abstain-frac", type=float, default=0.10, help="quality gate: fraction of VAL images to reject")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    lv, yv = load(a.val)
    T = fit_temperature(lv, yv)
    thr = threshold_for_sensitivity(lv, yv, a.target_sens)  # monotone in logits, so same on calibrated scale
    print(f"temperature T = {T:.3f}; frozen logit threshold = {thr:.3f} (target sensitivity {a.target_sens:.0%} on val)")

    sets = {"internal": load(a.internal), "external": load(a.external)}
    results = {"temperature": T, "threshold_logit": thr, "target_sensitivity": a.target_sens, "sets": {}}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for name, (lg, y) in sets.items():
        pr_raw, pr_cal = expit(lg), expit(lg / T)
        sens, spec = sens_spec(lg, y, thr)
        # CIs for the frozen-threshold numbers
        sens_ci = bootstrap(lambda s, t: sens_spec(s, t, thr)[0], lg, y, a.boot)
        spec_ci = bootstrap(lambda s, t: sens_spec(s, t, thr)[1], lg, y, a.boot)
        auc = roc_auc_score(y, lg)
        auc_ci = bootstrap(lambda s, t: roc_auc_score(t, s), lg, y, a.boot)
        s90 = spec_at_sens(lg, y, a.target_sens)
        s90_ci = bootstrap(lambda s, t: spec_at_sens(s, t, a.target_sens), lg, y, a.boot)
        e_raw = ece(pr_raw, y); e_cal = ece(pr_cal, y)
        res = {
            "n": int(len(y)), "prevalence": float(y.mean()),
            "auroc": [auc, *auc_ci],
            "frozen_threshold_sensitivity": [sens, *sens_ci],
            "frozen_threshold_specificity": [spec, *spec_ci],
            "specificity_at_target_sens_repicked": [s90, *s90_ci],
            "ece_raw": e_raw, "ece_temp_scaled": e_cal,
        }
        results["sets"][name] = res
        print(f"\n== {name} (n={len(y)}, prevalence {y.mean():.2f}) ==")
        print(f"AUROC                                 {fmt(auc, auc_ci)}")
        print(f"Sensitivity @ frozen threshold        {fmt(sens, sens_ci)}")
        print(f"Specificity @ frozen threshold        {fmt(spec, spec_ci)}")
        print(f"Specificity @ {a.target_sens:.0%} sens (re-picked)   {fmt(s90, s90_ci)}")
        print(f"ECE raw -> temperature scaled         {e_raw:.3f} -> {e_cal:.3f}")

        fpr, tpr, _ = roc_curve(y, lg)
        axes[0].plot(fpr, tpr, label=f"{name} (AUROC {auc:.3f})")
        for lab, pr, ls in [("raw", pr_raw, "--"), ("temp-scaled", pr_cal, "-")]:
            edges = np.linspace(0, 1, 11)
            idx = np.clip(np.digitize(pr, edges) - 1, 0, 9)
            xs, ys = [], []
            for b in range(10):
                m = idx == b
                if m.sum() >= 5:
                    xs.append(pr[m].mean()); ys.append(y[m].mean())
            axes[1].plot(xs, ys, ls, marker="o", ms=3, label=f"{name} {lab}")

    axes[0].plot([0, 1], [0, 1], "k:", lw=0.8); axes[0].set(xlabel="1 - specificity", ylabel="sensitivity", title="ROC")
    axes[0].legend()
    axes[1].plot([0, 1], [0, 1], "k:", lw=0.8); axes[1].set(xlabel="predicted probability", ylabel="observed frequency", title="Reliability")
    axes[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "roc_and_calibration.png"), dpi=150); plt.close(fig)

    dfs = {"val": pd.read_csv(a.val), "internal": pd.read_csv(a.internal), "external": pd.read_csv(a.external)}
    if all("quality" in d.columns for d in dfs.values()):
        results["quality_gate"] = quality_gate(dfs["val"], {k: dfs[k] for k in sets}, thr, a)
    else:
        print("\n(no 'quality' column in the prediction CSVs; skipping the quality gate - run preprocess.py first)")
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics.json and figures to {a.out}/")


if __name__ == "__main__":
    main()
