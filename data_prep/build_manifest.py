"""
build_manifest.py - list every test clip in data/manifest.csv.

Folder conventions (file names are always <speaker>__<clip id>.wav):
    data/real/<source>/...            real voices      (e.g. source = librispeech)
    data/fake_tts/<generator>/...     our own TTS fakes (e.g. generator = piper)
    data/fake_public/<source>/...     fakes from public deepfake datasets

Run from the project root:
    python -m data_prep.build_manifest
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Where each source's audio came from, and why we are allowed to use it.
CONSENT_NOTES = {
    "librispeech": "public dataset, CC BY 4.0 (volunteer audiobook readers)",
    "piper": "synthetic stock voice (Piper); no real person cloned",
    "speecht5": "synthetic stock voice (SpeechT5 + CMU Arctic x-vector); no real person cloned",
}


def main():
    rows = []
    for label, folder in [("real", "real"), ("fake", "fake_tts"), ("fake", "fake_public")]:
        for wav in sorted((DATA / folder).glob("*/*.wav")):
            group = wav.parent.name
            if group == "sg_officials":  # real-world deepfakes: separate report (eval/officials_report.py)
                continue
            rows.append({
                "path": wav.relative_to(ROOT).as_posix(),
                "label": label,
                "source": group if folder != "fake_tts" else "tts",
                "generator": group if folder == "fake_tts" else "",
                "speaker_id": wav.stem.split("__")[0],
                "consent_note": CONSENT_NOTES.get(group, "see CITATIONS.md"),
            })
    out = DATA / "manifest.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} clips to {out}")
    counts = {}
    for r in rows:
        key = (r["label"], r["source"], r["generator"] or "-")
        counts[key] = counts.get(key, 0) + 1
    for (label, source, gen), n in sorted(counts.items()):
        print(f"  {label:<5} {source:<12} {gen:<10} {n}")


if __name__ == "__main__":
    main()
