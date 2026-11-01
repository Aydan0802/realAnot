"""
run_eval.py - test the detector on clean, compressed and phone audio.

For every clip in data/manifest.csv:
  1. score the clean clip
  2. make a compressed copy, as a clip arrives by messaging app or the web,
     and a phone-quality copy (audio/phone.py, seeded per clip so both are
     repeatable), and score those too
Then report results separately by condition (clean / compressed / phone), by
fake source (each TTS generator on its own), and by codec. Never just one number.

Outputs in results/eval/:
    scores.csv        one row per clip per condition
    metrics.csv       EER, AUC, accuracy, false accepts / rejects per group
    score_stats.json  mean and spread of scores, real vs fake, per condition
                      (feeds simulator/scam_sim.py later)
    hist_scores.png, roc.png

Run from the project root:
    python -m eval.run_eval              # all clips
    python -m eval.run_eval --limit 20   # quick check
"""
import argparse
import json
import zlib
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from audio.phone import PhoneConfig, compress_clip, degrade
from detectors.forensics_xlsr import load_audio, score_clip

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def score_all(manifest, head=None):
    rows = []
    for i, clip in enumerate(manifest.itertuples(index=False)):
        wav = load_audio(ROOT / clip.path)
        seed = zlib.crc32(clip.path.encode())  # same clip -> same phone copy every run
        phone_wav, phone_info = degrade(wav, seed=seed, config=PhoneConfig())
        clip_wav, clip_info = compress_clip(wav, seed=seed)
        for condition, audio, info in [("clean", wav, {}), ("compressed", clip_wav, clip_info),
                                       ("phone", phone_wav, phone_info)]:
            r = score_clip(audio, head)
            rows.append({**clip._asdict(), "condition": condition,
                         "codec": info.get("codec"), "snr_db": info.get("snr_db"),
                         "packet_loss": info.get("packet_loss"),
                         "fake_score_raw": r["fake_score_raw"], "window_spread": r["window_spread"],
                         "window_scores": json.dumps(r["window_scores"]),
                         "n_windows": r["n_windows"], "duration_s": r["duration_s"],
                         "inconclusive": r["inconclusive"]})
        if (i + 1) % 25 == 0:
            print(f"  scored {i + 1}/{len(manifest)} clips")
    return pd.DataFrame(rows)


def metrics(group, threshold):
    """EER and AUC need no threshold. Accuracy / FAR / FRR use `threshold`.
    FAR = fakes accepted as real; FRR = real voices wrongly flagged as fake."""
    y = (group.label == "fake").astype(int).to_numpy()
    s = group.fake_score_raw.to_numpy()
    out = {"n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())}
    if out["n_real"] == 0 or out["n_fake"] == 0:
        return out
    fpr, tpr, thr = roc_curve(y, s)
    i = np.argmin(np.abs((1 - tpr) - fpr))
    pred = s >= threshold
    out.update({
        "EER_%": round(100 * (fpr[i] + 1 - tpr[i]) / 2, 2),
        "EER_threshold": round(float(thr[i]), 3),
        "AUC": round(roc_auc_score(y, s), 3),
        f"acc_%@{threshold}": round(100 * (pred == y).mean(), 1),
        f"FAR_%@{threshold}": round(100 * (~pred[y == 1]).mean(), 1),
        f"FRR_%@{threshold}": round(100 * pred[y == 0].mean(), 1),
    })
    return out


def build_metrics(scores, threshold):
    """Rows: each condition overall, each fake generator/source against all
    real clips, and (compressed and phone) each codec, by the first one applied."""
    table = []
    for condition, cond in scores.groupby("condition"):
        real = cond[cond.label == "real"]
        table.append({"condition": condition, "group": "ALL", **metrics(cond, threshold)})
        fakes = cond[cond.label == "fake"]
        for name, fk in fakes.groupby(fakes.generator.where(fakes.generator != "", fakes.source)):
            table.append({"condition": condition, "group": f"fake={name}",
                          **metrics(pd.concat([real, fk]), threshold)})
        if condition != "clean":
            for codec, cg in cond.groupby(cond.codec.str.split("+").str[0]):
                table.append({"condition": condition, "group": f"codec={codec}",
                              **metrics(cg, threshold)})
    return pd.DataFrame(table)


def score_stats(scores):
    """Mean and spread of raw scores, used later by the simulator."""
    stats = {}
    for condition, cond in scores.groupby("condition"):
        stats[condition] = {
            label: {"mean": round(float(g.fake_score_raw.mean()), 4),
                    "std": round(float(g.fake_score_raw.std()), 4), "n": int(len(g))}
            for label, g in cond.groupby("label")}
    return stats


def charts(scores, out):
    fig, axes = plt.subplots(1, scores.condition.nunique(), figsize=(16, 4), sharey=True)
    for ax, (condition, cond) in zip(axes, scores.groupby("condition")):
        for label, color in [("real", "#2a9d8f"), ("fake", "#e76f51")]:
            ax.hist(cond[cond.label == label].fake_score_raw, bins=25, range=(0, 1),
                    alpha=0.65, color=color, label=label)
        ax.set_title(f"{condition} audio")
        ax.set_xlabel("raw fake score (0 = real, 1 = synthetic)")
        ax.legend()
    axes[0].set_ylabel("number of clips")
    plt.tight_layout()
    plt.savefig(out / "hist_scores.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 5))
    for condition, cond in scores.groupby("condition"):
        fpr, tpr, _ = roc_curve((cond.label == "fake").astype(int), cond.fake_score_raw)
        auc = roc_auc_score((cond.label == "fake").astype(int), cond.fake_score_raw)
        plt.plot(fpr, tpr, label=f"{condition} (AUC {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("real voices wrongly flagged (false positive rate)")
    plt.ylabel("fakes caught (true positive rate)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "roc.png", dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--limit", type=int, default=0, help="score only N random clips (0 = all)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="cut-off for accuracy/FAR/FRR (the authors' default is 0.5)")
    ap.add_argument("--rescore", action="store_true", help="ignore saved scores.csv and score again")
    ap.add_argument("--head", default=None, help="fine-tuned head (.safetensors); default = original model")
    ap.add_argument("--split", default=None, help="only clips with this value in the manifest's split column")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    scores_path = out / "scores.csv"
    if scores_path.exists() and not args.rescore and not args.limit:
        scores = pd.read_csv(scores_path, keep_default_na=False)
        print(f"Re-using {scores_path} (pass --rescore to score again)")
    else:
        manifest = pd.read_csv(ROOT / args.manifest, keep_default_na=False)
        if args.split:  # e.g. the validation clips of data/train_manifest.csv
            manifest = manifest[(manifest.split == args.split) & (manifest.version == "clean")]
            manifest = manifest.drop(columns=["split", "version"])
        if args.limit:
            manifest = manifest.sample(n=min(args.limit, len(manifest)), random_state=0)
        print(f"Scoring {len(manifest)} clips, clean + phone"
              f"{' with head ' + args.head if args.head else ''} ...")
        scores = score_all(manifest, args.head)
        scores.to_csv(scores_path, index=False)

    table = build_metrics(scores, args.threshold)
    table.to_csv(out / "metrics.csv", index=False)
    (out / "score_stats.json").write_text(json.dumps(score_stats(scores), indent=2))
    charts(scores, out)

    pd.set_option("display.width", 200)
    print("\n" + table.to_string(index=False))
    n_inc = scores.groupby("condition").inconclusive.apply(lambda s: s.astype(str).eq("True").sum())
    print(f"\nInconclusive clips per condition: {n_inc.to_dict()}")
    print(f"Saved results to {out}")
    print("Note: EER uses the best threshold for THIS test set, so it is optimistic. "
          "A threshold must be chosen on separate data before quoting accuracy.")


if __name__ == "__main__":
    main()
