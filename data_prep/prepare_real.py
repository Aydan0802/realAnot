"""
prepare_real.py - pick real voice clips from LibriSpeech test-clean.

LibriSpeech (CC BY 4.0) is public audiobook speech from volunteers, so no
biometric data is collected from anyone. We take a few clips per speaker
(3 to 15 s long) so that no single voice dominates the test set, convert them
to 16 kHz WAV, and save their transcripts so the TTS script can read the
SAME sentences with synthetic voices. That way real and fake clips have the
same content, and the detector cannot tell them apart by the words used.

Download first (see README.md), then run from the project root:
    python -m data_prep.prepare_real --per-speaker 5
"""
import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "downloads" / "librispeech" / "LibriSpeech" / "test-clean"
OUT = ROOT / "data" / "real" / "librispeech"
SR = 16000


def read_transcripts(src=SRC):
    """Return {utt_id: text} from every *.trans.txt file."""
    texts = {}
    for trans in sorted(src.glob("*/*/*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            utt_id, text = line.split(" ", 1)
            texts[utt_id] = text
    return texts


def to_sentence_case(text):
    """LibriSpeech transcripts are ALL CAPS; TTS reads normal text better."""
    text = text.lower()
    return text[0].upper() + text[1:] + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-speaker", type=int, default=5)
    ap.add_argument("--min-s", type=float, default=3.0)
    ap.add_argument("--max-s", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"{SRC} not found. Download LibriSpeech test-clean first (see README.md).")
    texts = read_transcripts()
    rng = np.random.default_rng(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    for speaker_dir in sorted(p for p in SRC.iterdir() if p.is_dir()):
        files = sorted(speaker_dir.glob("*/*.flac"))
        rng.shuffle(files)  # random but repeatable choice of clips
        kept = 0
        for flac in files:
            info = sf.info(str(flac))
            if not args.min_s <= info.duration <= args.max_s:
                continue
            wav, sr = sf.read(str(flac), dtype="float32")
            if sr != SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
            speaker = f"libri{speaker_dir.name}"
            sf.write(OUT / f"{speaker}__{flac.stem}.wav", wav, SR)
            rows.append({"utt_id": flac.stem, "speaker_id": speaker,
                         "text": to_sentence_case(texts[flac.stem]),
                         "seconds": round(info.duration, 2)})
            kept += 1
            if kept == args.per_speaker:
                break

    with open(OUT / "transcripts.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["utt_id", "speaker_id", "text", "seconds"])
        writer.writeheader()
        writer.writerows(rows)
    speakers = len({r["speaker_id"] for r in rows})
    print(f"Saved {len(rows)} real clips from {speakers} speakers to {OUT}")


if __name__ == "__main__":
    main()
