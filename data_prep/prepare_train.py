"""
prepare_train.py - build a TRAINING set that never overlaps the test set.

  - Real voices: LibriSpeech dev-clean, whose 40 speakers are all different
    from the test set's (test-clean).
  - Fake voices: the same sentences read by 9 Piper voices that are NOT used
    in the test set. SpeechT5 is left out on purpose, so the test set can
    show whether the fine-tuned model still works on a generator it never
    trained on.
  - Split by person: 32 real speakers + 7 TTS voices for training,
    8 speakers + 2 voices for validation (choosing the best epoch and
    calibrating). The test set is only used at the very end.
  - Every TRAINING clip also gets 2 phone-quality copies with different
    random codecs (the "presentation" idea from the research).

Run from the project root (needs dev-clean in downloads/librispeech/):
    python -m data_prep.prepare_train --per-speaker 15
"""
import argparse
import csv
import zlib
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from audio.phone import PhoneConfig, degrade
from data_prep.make_tts import Piper
from data_prep.prepare_real import read_transcripts, to_sentence_case

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "downloads" / "librispeech" / "LibriSpeech" / "dev-clean"
OUT = ROOT / "data" / "train"
SR = 16000

TRAIN_VOICES = {
    "alba": "en/en_GB/alba/medium/en_GB-alba-medium.onnx",
    "northern_english_male": "en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium.onnx",
    "hfc_female": "en/en_US/hfc_female/medium/en_US-hfc_female-medium.onnx",
    "hfc_male": "en/en_US/hfc_male/medium/en_US-hfc_male-medium.onnx",
    "kristin": "en/en_US/kristin/medium/en_US-kristin-medium.onnx",
    "norman": "en/en_US/norman/medium/en_US-norman-medium.onnx",
    "ljspeech": "en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx",
}
VAL_VOICES = {
    "joe": "en/en_US/joe/medium/en_US-joe-medium.onnx",
    "jenny_dioco": "en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx",
}
PHONE_COPIES = 2


def pick_real_clips(per_speaker, rng):
    """Up to `per_speaker` clips of 3 to 15 s from each dev-clean speaker."""
    texts = read_transcripts(SRC)
    speakers = sorted(p for p in SRC.iterdir() if p.is_dir())
    rows = []
    for i, speaker_dir in enumerate(speakers):
        split = "val" if i % 5 == 0 else "train"  # every 5th speaker -> validation
        files = sorted(speaker_dir.glob("*/*.flac"))
        rng.shuffle(files)
        kept = 0
        for flac in files:
            if not 3.0 <= sf.info(str(flac)).duration <= 15.0:
                continue
            rows.append({"flac": flac, "speaker": f"libri{speaker_dir.name}", "split": split,
                         "text": to_sentence_case(texts[flac.stem])})
            kept += 1
            if kept == per_speaker:
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-speaker", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found. Download LibriSpeech dev-clean first (see README.md).")

    rng = np.random.default_rng(args.seed)
    clips = pick_real_clips(args.per_speaker, rng)
    tts = Piper({**TRAIN_VOICES, **VAL_VOICES})
    voice_lists = {"train": list(TRAIN_VOICES), "val": list(VAL_VOICES)}
    manifest = []

    def add(path, label, source, generator, speaker, split, version, note):
        manifest.append({"path": path.relative_to(ROOT).as_posix(), "label": label, "source": source,
                         "generator": generator, "speaker_id": speaker, "consent_note": note,
                         "split": split, "version": version})

    for i, clip in enumerate(clips):
        # Real clip
        real_path = OUT / "real" / f"{clip['speaker']}__{clip['flac'].stem}.wav"
        real_path.parent.mkdir(parents=True, exist_ok=True)
        if not real_path.exists():
            wav, sr = sf.read(str(clip["flac"]), dtype="float32")
            sf.write(real_path, librosa.resample(wav, orig_sr=sr, target_sr=SR) if sr != SR else wav, SR)
        add(real_path, "real", "librispeech", "", clip["speaker"], clip["split"], "clean",
            "public dataset, CC BY 4.0 (volunteer audiobook readers)")

        # The same sentence read by a Piper voice from the matching split
        voices = voice_lists[clip["split"]]
        voice = voices[i % len(voices)]
        fake_path = OUT / "fake" / "piper" / f"piper-{voice}__{clip['flac'].stem}.wav"
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        if not fake_path.exists():
            wav = tts.say(clip["text"], voice)
            sf.write(fake_path, wav / (np.abs(wav).max() + 1e-8) * 0.9, SR)
        add(fake_path, "fake", "tts", "piper", f"piper-{voice}", clip["split"], "clean",
            "synthetic stock voice (Piper); no real person cloned")

        # Phone-quality copies, training split only (validation makes its own, like the test)
        if clip["split"] == "train":
            for src_path, label, gen, speaker in [(real_path, "real", "", clip["speaker"]),
                                                  (fake_path, "fake", "piper", f"piper-{voice}")]:
                wav, _ = sf.read(src_path, dtype="float32")
                for k in range(1, PHONE_COPIES + 1):
                    out = OUT / "phone" / label / f"{src_path.stem}__p{k}.wav"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    if not out.exists():
                        seed = zlib.crc32(f"{src_path.name}#{k}".encode())
                        sf.write(out, degrade(wav, seed=seed, config=PhoneConfig())[0], SR)
                    add(out, label, "librispeech" if label == "real" else "tts", gen, speaker,
                        "train", f"phone{k}", "phone copy of the clip above")
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(clips)} sentences done")

    with open(ROOT / "data" / "train_manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    counts = {}
    for r in manifest:
        key = (r["split"], r["label"], r["version"])
        counts[key] = counts.get(key, 0) + 1
    print(f"Wrote data/train_manifest.csv ({len(manifest)} files)")
    for key, n in sorted(counts.items()):
        print(f"  {key[0]:<5} {key[1]:<4} {key[2]:<6} {n}")


if __name__ == "__main__":
    main()
