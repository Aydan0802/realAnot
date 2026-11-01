"""
splice_test.py - does the "part of this clip sounds synthetic" rule work?

Scam videos often drop a few seconds of cloned audio into real footage, so the
clip's average score can look fine. detectors/verdict.py therefore warns when
enough 5 s windows sound synthetic. This script measures that rule:

    real only   real + real + real      a warning here is a false alarm
    spliced     real + FAKE + real      a warning here is a catch

Each pair of test clips is built from different recordings, clean and through
a simulated phone call. Real pieces come from different speakers, so this
tests the windows, not a believable edit. The script scores every clip once,
saves the window chances, then tries each rule setting on those saved scores.

Choose the rule on validation clips, then run the test clips ONCE to report it:
    python -m eval.splice_test --manifest data/train_manifest_sg_orpheus.csv --split val --out results/splice_val
    python -m eval.splice_test --manifest data/manifest_sg_orpheus.csv --out results/splice_test
"""
import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from audio.phone import PhoneConfig, degrade
from detectors.calibration import combine
from detectors.forensics_xlsr import load_audio, score_clip
from detectors.verdict import CALIBRATION, PART_MIN_SHARE, PART_MIN_WINDOWS, PART_WINDOW, phone_band_share

ROOT = Path(__file__).resolve().parent.parent
GRID = [(w, n, s) for w in (0.5, 0.6, 0.7, 0.8, 0.9) for n in (1, 2, 3) for s in (0.1, 0.2, 0.3)]


def make_trials(manifest, n, seed=0):
    """n pairs of clips, sharing their outer real pieces: real only, then spliced."""
    rng = np.random.default_rng(seed)
    real = manifest[manifest.label == "real"].path.to_numpy()
    fake = manifest[manifest.label == "fake"]
    trials = []
    for i in range(n):
        before, middle, after = rng.choice(real, 3, replace=False)
        row = fake.iloc[int(rng.integers(len(fake)))]
        trials.append(("real only", "", [before, middle, after]))
        trials.append(("spliced", row.generator, [before, row.path, after]))
    return trials


def score_trial(kind, generator, paths, cal):
    wav = np.concatenate([load_audio(ROOT / p) for p in paths])
    phone, _ = degrade(wav, seed=zlib.crc32("".join(paths).encode()), config=PhoneConfig())
    rows = []
    for condition, audio in [("clean", wav), ("phone", phone)]:
        params = cal["phone" if phone_band_share(audio) < cal["band_cutoff"] else "wideband"]
        dets = [score_clip(audio, ROOT / head if head else None) for head in cal["heads"]]
        rows.append({"kind": kind, "generator": generator, "condition": condition,
                     "chance": float(combine([d["fake_score_raw"] for d in dets], params)),
                     "windows": json.dumps(combine(np.array([d["window_scores"] for d in dets]).T,
                                                   params).round(4).tolist())})
    return rows


def warned(windows, part_window, min_windows, min_share):
    """The same rule as detectors/verdict.py."""
    n_flagged = sum(w >= part_window for w in windows)
    return n_flagged >= min_windows and n_flagged / len(windows) >= min_share


def rule_table(scores, high_edge):
    """For each rule setting: real clips wrongly warned about, spliced clips caught.
    A clip already in the HIGH band counts as flagged, whatever the rule says."""
    windows = scores.windows.map(json.loads)
    rows = []
    for part_window, min_windows, min_share in GRID:
        flagged = (scores.chance >= high_edge) | windows.map(lambda w: warned(w, part_window, min_windows, min_share))
        row = {"window_chance": part_window, "min_windows": min_windows, "min_share": min_share}
        for condition in ("clean", "phone"):
            kept = flagged[scores.condition == condition]
            kinds = scores.kind[scores.condition == condition]
            row[f"{condition}_false_alarm_%"] = round(100 * kept[kinds == "real only"].mean(), 1)
            row[f"{condition}_caught_%"] = round(100 * kept[kinds == "spliced"].mean(), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default=None, help="only clips with this value in the manifest's split column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100, help="pairs of clips to build")
    ap.add_argument("--rescore", action="store_true", help="ignore saved scores.csv and score again")
    args = ap.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    cal = json.loads(CALIBRATION.read_text())
    if (out / "scores.csv").exists() and not args.rescore:
        print(f"Re-using {out / 'scores.csv'} (pass --rescore to score again)")
        scores = pd.read_csv(out / "scores.csv", keep_default_na=False)
    else:
        manifest = pd.read_csv(ROOT / args.manifest, keep_default_na=False)
        if args.split:
            manifest = manifest[(manifest.split == args.split) & (manifest.version == "clean")]
        trials = make_trials(manifest, args.n)
        rows = []
        for i, trial in enumerate(trials):
            rows += score_trial(*trial, cal)
            if (i + 1) % 25 == 0:
                print(f"  scored {i + 1}/{len(trials)} clips")
        scores = pd.DataFrame(rows)
        scores.to_csv(out / "scores.csv", index=False)

    table = rule_table(scores, cal["bands"]["MEDIUM"])
    table.to_csv(out / "rules.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n" + table.to_string(index=False))
    current = table[(table.window_chance == PART_WINDOW) & (table.min_windows == PART_MIN_WINDOWS)
                    & (table.min_share == PART_MIN_SHARE)]
    print(f"\nCurrent rule in detectors/verdict.py:\n{current.to_string(index=False)}")
    print(f"Saved {out / 'rules.csv'}")


if __name__ == "__main__":
    main()
