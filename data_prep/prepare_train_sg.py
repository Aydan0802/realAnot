"""
prepare_train_sg.py - add real SINGAPOREAN voices to the training set.

Why: the original detector and calibration were tuned on Western audiobook
speech and flagged over half of real Singlish conversation clips as "likely
AI" (docs/eval_results.md, Run 3).

  - Real: IMDA National Speech Corpus Part 2 (read) and Part 6 (conversation).
    The TEST set uses Parts 1 and 3, which were recorded separately.
  - Fake: the same Singaporean sentences read by the Piper TRAINING voices,
    so real and fake stay balanced and have the same words (incl. Singlish
    particles like "lah", "ah").
  - 80% train (+ 2 phone copies each), 20% validation. The corpus split has
    no speaker IDs, so this split is by clip.

Caveat: there are no Singapore-ACCENTED fakes here (no open Singlish TTS
fits our limits yet), so the model could learn "Singaporean accent = real".

Writes data/train_manifest_sg.csv = data/train_manifest.csv + the new rows.
Run from the project root (after data_prep.prepare_train):
    python -m data_prep.prepare_train_sg --per-part 240
"""
import argparse
import csv
import io
import re
import zlib

import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from audio.phone import PhoneConfig, degrade
from data_prep.make_tts import Piper
from data_prep.prepare_train import OUT, PHONE_COPIES, ROOT, SR, TRAIN_VOICES, VAL_VOICES

PARTS = {"nsc_read_p2": "ASR-PART2-Test", "nsc_conv_p6": "ASR-PART6-Test"}
NOTE = "IMDA National Speech Corpus (via MERaLiON MNSC), Singapore Open Data Licence v1.0"


def clean_text(text):
    """Turn a corpus transcript into text a TTS voice can read."""
    text = re.sub(r"<[^>]*>:?", " ", text)         # <Speaker1>:, <UNK> and other tags
    text = re.sub(r"[\[\]()]", " ", text)          # [ah] -> ah (keep Singlish particles)
    text = re.sub(r"[#~_*]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return (text[:1].upper() + text[1:] + ".") if text else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-part", type=int, default=240)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    tts = Piper({**TRAIN_VOICES, **VAL_VOICES})
    voice_lists = {"train": list(TRAIN_VOICES), "val": list(VAL_VOICES)}
    rows = []

    def add(path, label, source, gen, speaker, split, version, note):
        rows.append({"path": path.relative_to(ROOT).as_posix(), "label": label, "source": source,
                     "generator": gen, "speaker_id": speaker, "consent_note": note,
                     "split": split, "version": version})

    for source, part in PARTS.items():
        shard = sorted((ROOT / "downloads" / "mnsc" / part).glob("*.parquet"))[0]
        table = pq.read_table(shard, columns=["context", "answer"])
        kept = 0
        for i in rng.permutation(table.num_rows):
            text = clean_text(table.column("answer")[int(i)].as_py())
            if len(text.split()) < 4:
                continue
            audio = table.column("context")[int(i)].as_py()
            wav, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
            wav = wav.mean(axis=1)
            if sr != SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
            if len(wav) < 3 * SR:
                continue
            wav = wav[:15 * SR]
            split = "val" if kept % 5 == 0 else "train"
            name = f"{part.lower()}-{int(i)}"

            real_path = OUT / "sg_real" / f"{name}__{int(i)}.wav"
            real_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(real_path, wav, SR)
            add(real_path, "real", source, "", name, split, "clean", NOTE)

            voices = voice_lists[split]
            voice = voices[kept % len(voices)]
            fake_path = OUT / "sg_fake" / f"piper-{voice}__{name}.wav"
            fake_path.parent.mkdir(parents=True, exist_ok=True)
            if not fake_path.exists():
                fake = tts.say(text[:400], voice)  # very long turns: first ~400 characters
                sf.write(fake_path, fake / (np.abs(fake).max() + 1e-8) * 0.9, SR)
            add(fake_path, "fake", "tts", "piper", f"piper-{voice}", split, "clean",
                "synthetic stock voice (Piper) reading a Singaporean transcript")

            if split == "train":
                for src, label, gen, speaker in [(real_path, "real", "", name),
                                                 (fake_path, "fake", "piper", f"piper-{voice}")]:
                    clip, _ = sf.read(src, dtype="float32")
                    for k in range(1, PHONE_COPIES + 1):
                        out = OUT / "phone" / label / f"{src.stem}__p{k}.wav"
                        if not out.exists():
                            seed = zlib.crc32(f"{src.name}#{k}".encode())
                            sf.write(out, degrade(clip, seed=seed, config=PhoneConfig())[0], SR)
                        add(out, label, source if label == "real" else "tts", gen, speaker,
                            "train", f"phone{k}", "phone copy of the clip above")
            kept += 1
            if kept % 50 == 0:
                print(f"  {source}: {kept}/{args.per_part}")
            if kept == args.per_part:
                break

    with open(ROOT / "data" / "train_manifest.csv", encoding="utf-8") as f:
        base = list(csv.DictReader(f))
    with open(ROOT / "data" / "train_manifest_sg.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(base[0]))
        writer.writeheader()
        writer.writerows(base + rows)
    print(f"Wrote data/train_manifest_sg.csv: {len(base)} original + {len(rows)} Singapore rows")


if __name__ == "__main__":
    main()
