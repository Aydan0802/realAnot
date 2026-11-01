"""
make_orpheus.py - Singapore-ACCENTED fakes from a newer, codec-based TTS.

Why: every fake we had came from older TTS designs (Piper, SpeechT5) with
Western accents. The detector caught those well but lost most REAL deepfakes
of Singapore officials over a phone line (docs/eval_results.md,
Run 5), and could have learned "Singaporean accent = real".

Orpheus 3B fine-tuned on Singlish (data_prep/make_tts.py: Orpheus) reads the
SAME Singaporean sentences as the real clips, so real and fake differ only in
being real or synthetic. Voices are kept apart between splits (the fine-tune
blurs the base voices a little, so this separation is approximate):
    train: default voice, leo, leah, zac, jess     (+ 2 phone copies each)
    val:   tara
    test:  mia, dan, zoe  (never trained on)

Sentences: train/val = the NSC Part 2/6 clips already in the training set;
test = the NSC Part 1/3 test clips (data/manifest_sg.csv).

Writes:
    data/train_manifest_sg_orpheus.csv = data/train_manifest_sg.csv + Orpheus rows
    data/manifest_sg_orpheus.csv       = data/manifest_sg.csv + Orpheus test fakes
Run from the project root (about 4 hours on an RTX 4050; rerun to resume):
    python -m data_prep.make_orpheus
"""
import argparse
import csv
import zlib

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

from audio.phone import PhoneConfig, degrade
from data_prep.make_tts import Orpheus
from data_prep.prepare_train import OUT, PHONE_COPIES, ROOT, SR
from data_prep.prepare_train_sg import clean_text

VOICES = {"train": ["", "leo", "leah", "zac", "jess"], "val": ["tara"], "test": ["mia", "dan", "zoe"]}
NOTE = "synthetic voice (Orpheus 3B Singlish fine-tune) reading a Singaporean transcript; no real person cloned"


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def transcript(name, cache={}):
    """'asr-part2-test-123' -> cleaned text of row 123 of NSC Part 2, from the
    same parquet shard the real clip was cut from."""
    part, index = name.rsplit("-", 1)
    part = part.upper().replace("-TEST", "-Test")  # folder name, e.g. ASR-PART2-Test
    if part not in cache:
        shard = sorted((ROOT / "downloads" / "mnsc" / part).glob("*.parquet"))[0]
        cache[part] = pq.read_table(shard, columns=["answer"]).column("answer")
    return clean_text(cache[part][int(index)].as_py())


def jobs():
    """(split, sentence id, text) for every clip to make."""
    todo = []
    for r in read_csv(ROOT / "data" / "train_manifest_sg.csv"):
        if r["generator"] == "piper" and r["version"] == "clean" and "/sg_fake/" in r["path"]:
            name = r["path"].rsplit("__", 1)[1][:-4]
            todo.append((r["split"], name))
    for r in read_csv(ROOT / "data" / "manifest_sg.csv"):
        if r["label"] == "real":
            todo.append(("test", r["speaker_id"]))
    return [(split, name, shorten(transcript(name))) for split, name in todo]


def shorten(text, max_chars=150):
    """Stop long turns at a word boundary: ~150 characters is about 10 s of
    speech, plenty for the detector's 5 s windows. Orpheus makes about a third
    of a second of audio per second, so long clips are the main cost."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (0 = all)")
    ap.add_argument("--manifests-only", action="store_true",
                    help="make no audio, just write the manifests from the clips that exist")
    args = ap.parse_args()
    torch.manual_seed(0)

    todo = jobs()[:args.limit or None]
    plan = []  # (split, name, text, voice, out path)
    counters = {s: 0 for s in VOICES}
    for split, name, text in todo:
        voice = VOICES[split][counters[split] % len(VOICES[split])]
        counters[split] += 1
        folder = ROOT / "data" / "fake_tts" / "orpheus" if split == "test" else OUT / "sg_fake_orpheus"
        plan.append((split, name, text, voice, folder / f"orpheus-{voice or 'default'}__{name}.wav"))

    missing = [p for p in plan if not p[4].exists()]
    print(f"{len(plan)} clips planned, {len(missing)} still to make")
    if missing and not args.manifests_only:
        tts = Orpheus()
        # Validation first, then test, then training, so that a run cut short still
        # leaves something usable. Within that: one voice per batch, and similar
        # lengths together so no one waits for a long sentence.
        order = {"val": 0, "test": 1, "train": 2}
        missing.sort(key=lambda p: (order[p[0]], p[3], len(p[2])))
        batches = [missing[i:i + args.batch] for i in range(0, len(missing), args.batch)]
        batches = [[p for p in b if p[3] == v] for b in batches for v in dict.fromkeys(p[3] for p in b)]
        done = 0
        for group in batches:
            for p, wav in zip(group, tts.say_batch([p[2] for p in group], voice=group[0][3] or None)):
                if wav is None or len(wav) < SR:  # no usable audio: skipped, counted below
                    continue
                p[4].parent.mkdir(parents=True, exist_ok=True)
                sf.write(p[4], wav / (np.abs(wav).max() + 1e-8) * 0.9, SR)
            done += len(group)
            print(f"  {done}/{len(missing)}", flush=True)

    train_rows, test_rows, skipped = [], [], 0
    for split, name, text, voice, path in plan:
        if not path.exists():
            skipped += 1
            continue
        row = {"path": path.relative_to(ROOT).as_posix(), "label": "fake", "source": "tts",
               "generator": "orpheus", "speaker_id": f"orpheus-{voice or 'default'}", "consent_note": NOTE}
        if split == "test":
            test_rows.append(row)
            continue
        train_rows.append({**row, "split": split, "version": "clean"})
        if split == "train":
            clip, _ = sf.read(path, dtype="float32")
            for k in range(1, PHONE_COPIES + 1):
                out = OUT / "phone" / "fake" / f"{path.stem}__p{k}.wav"
                if not out.exists():
                    seed = zlib.crc32(f"{path.name}#{k}".encode())
                    sf.write(out, degrade(clip, seed=seed, config=PhoneConfig())[0], SR)
                train_rows.append({**row, "path": out.relative_to(ROOT).as_posix(), "split": "train",
                                   "version": f"phone{k}", "consent_note": "phone copy of the clip above"})

    write_csv(ROOT / "data" / "train_manifest_sg_orpheus.csv",
              read_csv(ROOT / "data" / "train_manifest_sg.csv") + train_rows)
    write_csv(ROOT / "data" / "manifest_sg_orpheus.csv", read_csv(ROOT / "data" / "manifest_sg.csv") + test_rows)
    print(f"Orpheus clips: {len(train_rows)} training rows (incl. phone copies), {len(test_rows)} test fakes, "
          f"{skipped} skipped (no usable audio)")


if __name__ == "__main__":
    main()
