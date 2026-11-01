"""
prepare_sg.py - real SINGAPOREAN English voices for testing.

Source: IMDA National Speech Corpus (Singapore Open Data Licence), via the
MERaLiON Multitask National Speech Corpus test splits on Hugging Face:
  - Part 1: Singaporeans reading prompted sentences          -> data/real/nsc_read/
  - Part 3: natural conversation, with Singlish particles     -> data/real/nsc_conv/
These are consenting corpus speakers; we only test on them, never clone them.

Why: a detector trained mostly on Western English might wrongly flag
Singaporean accents or Singlish as "fake". That would hit exactly the
customers this project is for, so we measure it.

Writes data/manifest_sg.csv: these Singaporean real clips plus the existing
test fakes from data/manifest.csv (the fake side for EER).

Download the two parquet shards first (see README.md), then run:
    python -m data_prep.prepare_sg --per-part 150
"""
import argparse
import csv
import io
from pathlib import Path

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
MNSC = ROOT / "downloads" / "mnsc"
PARTS = {"nsc_read": "ASR-PART1-Test", "nsc_conv": "ASR-PART3-Test"}
SR = 16000
NOTE = "IMDA National Speech Corpus (via MERaLiON MNSC), Singapore Open Data Licence v1.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-part", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []

    for source, part in PARTS.items():
        shards = sorted((MNSC / part).glob("*.parquet"))
        if not shards:
            raise SystemExit(f"No parquet files in {MNSC / part}. Download them first (see README.md).")
        table = pq.read_table(shards[0], columns=["context"])
        out_dir = ROOT / "data" / "real" / source
        out_dir.mkdir(parents=True, exist_ok=True)
        kept = 0
        for i in rng.permutation(table.num_rows):  # random but repeatable order
            audio = table.column("context")[int(i)].as_py()
            wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
            wav = wav.mean(axis=1)
            if sr != SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
            seconds = len(wav) / SR
            if seconds < 3.0:
                continue
            wav = wav[:15 * SR]  # long conversation turns: keep the first 15 s
            name = f"{part.lower()}-{int(i)}"
            path = out_dir / f"{name}__{int(i)}.wav"
            sf.write(path, wav, SR)
            # The corpus split has no speaker IDs, so each clip is its own "speaker".
            rows.append({"path": path.relative_to(ROOT).as_posix(), "label": "real", "source": source,
                         "generator": "", "speaker_id": name, "consent_note": NOTE})
            kept += 1
            if kept == args.per_part:
                break
        print(f"  {source}: {kept} clips")

    with open(ROOT / "data" / "manifest.csv", encoding="utf-8") as f:
        fakes = [r for r in csv.DictReader(f) if r["label"] == "fake"]
    with open(ROOT / "data" / "manifest_sg.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows + fakes)
    print(f"Wrote data/manifest_sg.csv: {len(rows)} Singaporean real clips + {len(fakes)} test fakes")


if __name__ == "__main__":
    main()
