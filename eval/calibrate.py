"""
calibrate.py - turn raw detector scores into honest probabilities.

Raw scores cannot be shown to users: the best cut-off moves a lot between
clean and phone audio, and between heads. This fits a small logistic model
(see detectors/calibration.py) on VALIDATION clips, separately for wideband
(clean and compressed clips) and phone-band audio because the detector
behaves so differently on each.
The verdict detects which one a clip is.

It can combine several heads. Each head is scored first with eval.run_eval
into its own folder, on the same clips:

    python -m eval.run_eval --manifest data/train_manifest_sg_orpheus.csv --split val --out results/val_orig
    python -m eval.run_eval --manifest data/train_manifest_sg_orpheus.csv --split val --out results/val_ft --head models/finetuned/x.safetensors
    (the same on the test manifest, into results/test_orig and results/test_ft)

    python -m eval.calibrate --heads original models/finetuned/x.safetensors \
        --fit results/val_orig results/val_ft --test results/test_orig results/test_ft \
        --report-dir results/calibration

No leakage: weights come from validation only, so accuracy on --test at the
50% cut-off is a fair estimate, unlike the EER in run_eval.py, which picks the
best threshold on the test set itself.
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from detectors.calibration import combine, logit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "detectors" / "calibration.json"
CHANNELS = {"wideband": ["clean", "compressed"], "phone": ["phone"]}  # audio channel -> eval conditions
BAND_CUTOFF = 0.0002  # energy share above 3.8 kHz; below = phone-band (see eval_results.md)
BANDS = [(0.3, "LOW"), (0.7, "MEDIUM"), (1.01, "HIGH")]  # ASSUMPTION: band edges
SPREAD_WARN, SPREAD_LIMIT = 90, 98  # percentiles of validation disagreement between windows


def load(folders):
    """Join each head's scores.csv on clip + condition. Returns the first table
    plus raw (one column per head) and windows (per clip, a windows x heads table)."""
    tables = [pd.read_csv(ROOT / f / "scores.csv", keep_default_na=False).set_index(["path", "condition"])
              for f in folders]
    index = tables[0].index
    tables = [t.loc[index] for t in tables]  # same order in every table
    raw = np.column_stack([t.fake_score_raw.to_numpy(dtype=float) for t in tables])
    windows = [np.array(row).T for row in zip(*[t.window_scores.map(json.loads) for t in tables])]
    return tables[0].reset_index(), raw, windows


def band_of(p):
    return next(name for edge, name in BANDS if p < edge)


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    i = np.argmin(np.abs((1 - tpr) - fpr))
    return 100 * (fpr[i] + 1 - tpr[i]) / 2


def ece(y, p, bins=10):
    """Expected calibration error: average gap between predicted and actual fake rate."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return total


def evaluate(condition, name, y, p):
    pred = p >= 0.5
    row = {"condition": condition, "fakes": name, "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
           "EER_%": round(eer(y, p), 2), "AUC": round(roc_auc_score(y, p), 3),
           "brier": round(brier_score_loss(y, p), 3), "ECE": round(ece(y, p), 3),
           "acc_%@0.5": round(100 * (pred == y).mean(), 1),
           "fakes_missed_%": round(100 * (~pred[y == 1]).mean(), 1),
           "real_flagged_%": round(100 * pred[y == 0].mean(), 1)}
    bands = pd.Series([band_of(x) for x in p])
    for label, value in (("real", 0), ("fake", 1)):
        share = bands[y == value].value_counts(normalize=True)
        for band in ("LOW", "MEDIUM", "HIGH"):
            row[f"{label}_{band}_%"] = round(100 * share.get(band, 0))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", nargs="+", required=True,
                    help="'original' or a fine-tuned head path, one per --fit folder")
    ap.add_argument("--fit", nargs="+", required=True, help="folders with validation scores.csv, one per head")
    ap.add_argument("--test", nargs="+", required=True, help="folders with test scores.csv, one per head")
    ap.add_argument("--report-dir", default="results/calibration")
    args = ap.parse_args()
    if not len(args.heads) == len(args.fit) == len(args.test):
        raise SystemExit("--heads, --fit and --test need one entry per head")
    fit_all, fit_raw, fit_windows = load(args.fit)
    test_all, test_raw, _ = load(args.test)
    report_dir = ROOT / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    params, report = {}, []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, (channel, conditions) in zip(axes, CHANNELS.items()):
        m = fit_all.condition.isin(conditions).to_numpy()
        y_fit = (fit_all.label[m] == "fake").astype(int).to_numpy()
        # class_weight: treat real and fake as equally likely, whatever the mix of validation clips.
        lr = LogisticRegression(C=1.0, class_weight="balanced").fit(logit(fit_raw[m]), y_fit)
        p = {"weights": [round(float(w), 4) for w in lr.coef_[0]], "bias": round(float(lr.intercept_[0]), 4),
             "n_real": int((y_fit == 0).sum()), "n_fake": int((y_fit == 1).sum())}
        # How much windows of one clip disagree, on the same scale as the final chance.
        spreads = [float(combine(w, p).std()) for w, keep in zip(fit_windows, m) if keep and len(w) > 1]
        p["spread_warn"] = round(float(np.percentile(spreads, SPREAD_WARN)), 3)
        p["spread_limit"] = round(float(np.percentile(spreads, SPREAD_LIMIT)), 3)
        params[channel] = p

        t = test_all[test_all.condition.isin(conditions)]
        y = (t.label == "fake").astype(int).to_numpy()
        chance = combine(test_raw[t.index], p)
        fakes = t.generator.where(t.generator != "", t.source).to_numpy()
        for condition in conditions:
            c = (t.condition == condition).to_numpy()
            report.append(evaluate(condition, "ALL", y[c], chance[c]))
            for name in sorted(set(fakes[c & (y == 1)])):
                keep = c & ((y == 0) | (fakes == name))
                report.append(evaluate(condition, name, y[keep], chance[keep]))

        # Reliability chart: in each bin, predicted vs actual share of fakes.
        bins = np.clip(np.digitize(chance, np.linspace(0, 1, 11)) - 1, 0, 9)
        mids = [chance[bins == i].mean() for i in range(10) if (bins == i).any()]
        actual = [y[bins == i].mean() for i in range(10) if (bins == i).any()]
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
        ax.plot(mids, actual, "o-", color="#2a9d8f", label="calibrated")
        ax.set_title(f"{channel} audio (test clips)")
        ax.set_xlabel("predicted chance of fake")
        ax.set_ylabel("actual share of fakes")
        ax.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "calibration.png", dpi=150)
    plt.close()

    OUT_JSON.write_text(json.dumps({
        "method": "P(fake) = sigmoid(sum(w * logit(raw per head)) + bias), fitted per channel on validation",
        "heads": [None if h == "original" else h for h in args.heads],
        "band_cutoff": BAND_CUTOFF,
        "bands": {name: edge for edge, name in BANDS},
        "prior_note": "assumes real and fake equally likely; real calls are mostly genuine",
        **params}, indent=2))
    table = pd.DataFrame(report)
    table.to_csv(report_dir / "calibration_report.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n" + table.to_string(index=False))
    print(f"\nWeights: {json.dumps(params)}")
    print(f"Saved {OUT_JSON.relative_to(ROOT)} and {args.report_dir}/calibration_report.csv, calibration.png")


if __name__ == "__main__":
    main()
