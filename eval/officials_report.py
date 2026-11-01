"""
officials_report.py - test the detector on REAL deepfakes of Singapore officials.

Evaluation only: collect
publicly circulated deepfake clips, puts the files in
    data/fake_public/sg_officials/raw/
and adds one row per file to docs/sg_officials_sources.csv
(file, url, date_published, official_impersonated, platform, what_it_claims, notes).
The clips are git-ignored and never used for training.

For each clip this script:
  1. extracts the audio (works on video files too) as 16 kHz mono
  2. scores it with the original detector and, if present, the fine-tuned heads,
     as received AND after a simulated phone call (GSM)
  3. reports the average score, the most suspicious window (and where it is),
     the share of 5 s windows flagged, and the full verdict (band + calibrated
     likelihood); every window's score goes to timeline.csv

Only a handful of clips exist, so read this per clip, not as an accuracy
figure. Many of these videos mix cloned audio into real footage, so some
windows may be genuine speech: look at the most suspicious window too.

Run from the project root:
    python -m eval.officials_report
"""
import csv
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pandas as pd

from audio.phone import PhoneConfig, degrade
from detectors.forensics_xlsr import HOP_S, SR, WINDOW_S, load_audio, score_clip
from detectors.verdict import analyse

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "fake_public" / "sg_officials" / "raw"
WAV_DIR = RAW.parent
SOURCES = ROOT / "docs" / "sg_officials_sources.csv"
FINETUNED = {"fine-tuned 1": ROOT / "models" / "finetuned" / "head.safetensors",      # audiobook data only
             "fine-tuned 2": ROOT / "models" / "finetuned" / "head_sg.safetensors",  # + Singaporean speech
             "best so far": ROOT / "models" / "finetuned" / "final_slow.safetensors"}  # + Orpheus fakes
OUT = ROOT / "results" / "sg_officials"
FLAG_AT = 0.5  # a window counts as "flagged" when its raw fake score is at least this


def extract_audio(src, dst):
    """Any audio/video file -> 16 kHz mono WAV, using the bundled ffmpeg."""
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(dst)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"could not read audio from {src.name}: {done.stderr[-300:]}")


def window_starts(n_samples, n_windows):
    """Start time (s) of each 5 s window, matching make_windows in the detector."""
    n, hop = int(WINDOW_S * SR), int(HOP_S * SR)
    starts = list(range(0, max(n_samples - n, 0) + 1, hop))
    if n_samples > n and starts[-1] + n < n_samples:
        starts.append(n_samples - n)
    return [s / SR for s in starts[:n_windows]]


def main():
    with open(SOURCES, encoding="utf-8") as f:
        sources = [r for r in csv.DictReader(f) if r["file"].strip()]
    if not sources:
        raise SystemExit(f"No clips listed in {SOURCES.relative_to(ROOT)}. Add the files to "
                         f"{RAW.relative_to(ROOT)} and one row per file to the CSV.")
    models = {"original": None, **{name: path for name, path in FINETUNED.items() if path.exists()}}
    cal = json.loads((ROOT / "detectors" / "calibration.json").read_text())
    cal_model = "+".join(Path(h).stem if h else "original" for h in cal["heads"])
    rows, timeline = [], []

    for src in sources:
        raw_path = RAW / src["file"]
        if not raw_path.exists():
            print(f"  missing file, skipped: {raw_path.relative_to(ROOT)}")
            continue
        who = src["official_impersonated"].strip().lower().replace(" ", "_") or "unknown"
        wav_path = WAV_DIR / f"sgofficial-{who}__{raw_path.stem}.wav"
        if not wav_path.exists():
            extract_audio(raw_path, wav_path)
        wav = load_audio(wav_path)
        phone, _ = degrade(wav, seed=0, config=PhoneConfig(codec="gsm"))

        for condition, audio in [("as received", wav), ("phone (GSM)", phone)]:
            for model_name, head in models.items():
                r = score_clip(audio, head)
                windows = np.array(r["window_scores"])
                starts = window_starts(len(audio), len(windows))
                # Every window's score, so they can be lined up with the video
                # (these clips often mix the deepfake with narration or real footage).
                timeline += [{"file": src["file"], "condition": condition, "model": model_name,
                              "start_s": t, "fake_score": round(float(s), 3)}
                             for t, s in zip(starts, windows)]
                peak = starts[int(windows.argmax())]
                rows.append({"file": src["file"], "official": src["official_impersonated"],
                             "condition": condition, "model": model_name,
                             "raw_score": round(r["fake_score_raw"], 3),
                             "max_window": round(float(windows.max()), 3),
                             "max_at": f"{int(peak // 60)}:{peak % 60:04.1f}",
                             "share_windows_flagged": f"{100 * (windows >= FLAG_AT).mean():.0f}%",
                             "n_windows": r["n_windows"], "seconds": r["duration_s"]})
            # The app's full verdict uses whichever model detectors/calibration.json was
            # fitted for, so it is the same for every model row of this clip + condition.
            v = analyse(audio)
            app = f"{v['band']} {v['likelihood_pct']:.0f}% ({cal_model})"
            for row in rows[-len(models):]:
                row["app_verdict"] = app

    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "report.csv", index=False)
    pd.DataFrame(timeline).to_csv(OUT / "timeline.csv", index=False)
    pd.set_option("display.width", 220)
    print("\n" + table.fillna("").to_string(index=False))
    print(f"\nSaved {OUT.relative_to(ROOT)}/report.csv. Read per clip: a few real-world clips are "
          "evidence of behaviour, not an accuracy figure.")


if __name__ == "__main__":
    main()
