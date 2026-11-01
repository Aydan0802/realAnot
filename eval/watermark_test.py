"""
watermark_test.py - does the AudioSeal watermark survive a phone call?

We watermark some of our TTS clips (pretending the TTS provider cooperates),
then check how often the mark is found:
  - on the watermarked clip, clean and after phone degradation (per codec)
  - on real voices and un-watermarked fakes (false alarms should be ~0)
The phone detection rate is the realistic "watermark coverage" for the simulator.

Run from the project root:
    python -m eval.watermark_test --n 50
"""
import argparse
import zlib
from pathlib import Path

import pandas as pd

from audio.phone import CODECS, PhoneConfig, degrade
from detectors import watermark
from detectors.forensics_xlsr import load_audio

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "watermark"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="clips per set")
    args = ap.parse_args()
    manifest = pd.read_csv(ROOT / "data" / "manifest.csv", keep_default_na=False)
    fakes = manifest[manifest.label == "fake"].sample(n=args.n, random_state=0)
    reals = manifest[manifest.label == "real"].sample(n=args.n, random_state=0)

    rows = []
    for set_name, clips, mark in [("watermarked fake", fakes, True),
                                  ("un-watermarked fake", fakes, False),
                                  ("real voice", reals, False)]:
        for clip in clips.itertuples():
            wav = load_audio(ROOT / clip.path)
            if mark:
                wav = watermark.add_watermark(wav)
            seed = zlib.crc32(clip.path.encode())
            versions = {"clean": wav}
            for codec in sorted(CODECS):
                versions[f"phone-{codec}"], _ = degrade(wav, seed=seed, config=PhoneConfig(codec=codec))
            for condition, audio in versions.items():
                score = watermark.detect(audio)
                rows.append({"set": set_name, "condition": condition,
                             "detected": score >= watermark.DETECT_THRESHOLD, "score": score})
        print(f"  done: {set_name}")

    df = pd.DataFrame(rows)
    summary = (df.groupby(["set", "condition"]).agg(n=("detected", "size"), detected_pct=("detected", "mean"),
                                                   mean_score=("score", "mean")).reset_index())
    summary["detected_pct"] = (100 * summary.detected_pct).round(1)
    summary["mean_score"] = summary.mean_score.round(3)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
