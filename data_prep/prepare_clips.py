"""
prepare_clips.py - training data aimed at real-world voice clips.

Adds two things to data/train_manifest_sg_orpheus.csv:

1. Vocoded fakes. A real clip from a public dataset goes through a neural
   vocoder or codec that rebuilds the sound. The speaker and the words stay
   the same and only the machine traces change, so the head cannot lean on
   voices or accents. Four tools: EnCodec, DAC, UnivNet and SNAC. SpeechT5's
   HiFi-GAN is left out so that SpeechT5 stays an unseen generator.
   Each real clip goes through one tool and stays in its own split.

2. Compressed copies. Each training clip gets 2 copies, as clips arrive by
   messaging app or the web (compress_clip in audio/phone.py).

Live calls are the later stage, so each clip keeps only 1 phone copy.

Writes:
    data/train/vocoded/<tool>/<clip>.wav     data/train/vocoded/phone/<clip>.wav
    data/train/compressed/<copy>/<generator>/<clip>.wav
    data/train_manifest_clips.csv

Run from the project root (rerun to resume, finished files are skipped):
    python -m data_prep.prepare_clips
    python -m data_prep.prepare_clips --check 2   # try 2 clips per tool, print how close they are
"""
import argparse
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from snac import SNAC
from transformers import DacModel, EncodecModel, UnivNetFeatureExtractor, UnivNetModel

from audio.phone import PhoneConfig, compress_clip, degrade
from detectors.forensics_xlsr import SR, load_audio

ROOT = Path(__file__).resolve().parent.parent
DL = ROOT / "downloads"
IN = ROOT / "data" / "train_manifest_sg_orpheus.csv"
OUT = ROOT / "data" / "train_manifest_clips.csv"
TOOLS = ["dac", "encodec", "snac", "univnet"]
HIGH_SR = 24000  # EnCodec, SNAC and UnivNet work at 24 kHz
CONSENT = "public dataset speech rebuilt by a neural vocoder; same speaker and words, no new voice"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to_high(wav):
    return librosa.resample(wav, orig_sr=SR, target_sr=HIGH_SR)


def from_high(audio):
    return librosa.resample(audio.squeeze().float().cpu().numpy(), orig_sr=HIGH_SR, target_sr=SR)


def load_tools():
    """Tool name -> function from 16 kHz audio to rebuilt 16 kHz audio."""
    encodec = EncodecModel.from_pretrained(DL / "encodec_24khz", use_safetensors=True).to(DEVICE).eval()
    dac = DacModel.from_pretrained(DL / "dac_16khz", use_safetensors=True).to(DEVICE).eval()
    univnet = UnivNetModel.from_pretrained(DL / "univnet_dev").to(DEVICE).eval()
    features = UnivNetFeatureExtractor.from_pretrained(DL / "univnet_dev")
    # SNAC only ships a pickle (.bin), so load it with weights_only=True and a strict key check.
    snac = SNAC.from_config(str(DL / "snac_24khz" / "config.json"))
    snac.load_state_dict(torch.load(DL / "snac_24khz" / "pytorch_model.bin", map_location="cpu",
                                    weights_only=True), strict=True)
    snac = snac.to(DEVICE).eval()

    def batch(wav):
        return torch.from_numpy(wav)[None, None].to(DEVICE)

    def run_univnet(wav):
        inputs = features(to_high(wav), sampling_rate=HIGH_SR, truncation=False, pad_end=True, return_tensors="pt")
        return from_high(univnet(**{k: v.to(DEVICE) for k, v in inputs.items()}).waveforms)

    return {"dac": lambda wav: dac(batch(wav)).audio_values.squeeze().float().cpu().numpy(),
            "encodec": lambda wav: from_high(encodec(batch(to_high(wav)), bandwidth=6.0).audio_values),
            "snac": lambda wav: from_high(snac(batch(to_high(wav)))[0]),
            "univnet": run_univnet}


def rebuild(tool, wav):
    """Run a tool, keep the clip's length, and stop the peak clipping when written."""
    with torch.no_grad():
        out = tool(wav)
    out = np.pad(out, (0, max(0, len(wav) - len(out))))[:len(wav)]
    return (out * min(1.0, 0.99 / (np.abs(out).max() + 1e-8))).astype(np.float32)


def tool_for(stem):
    return TOOLS[zlib.crc32(stem.encode()) % len(TOOLS)]


def check(tools, real, n):
    """How close each tool stays to the original: a low correlation means it is broken."""
    for name, tool in tools.items():
        for path in real.path[:n]:
            wav = load_audio(ROOT / path)
            out = rebuild(tool, wav)
            print(f"{name:8s} {len(wav) / SR:5.1f} s  correlation {np.corrcoef(wav, out)[0, 1]:.3f}  "
                  f"loudness ratio {out.std() / wav.std():.2f}")


def make_copy(job):
    source, target, make = job
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, make(load_audio(source), zlib.crc32(target.as_posix().encode())), SR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, default=0, help="try N clips per tool, then stop")
    args = ap.parse_args()

    manifest = pd.read_csv(IN, keep_default_na=False)
    real = manifest[(manifest.label == "real") & (manifest.version == "clean")]
    tools = load_tools()
    if args.check:
        return check(tools, real, args.check)

    vocoded = []
    for i, clip in enumerate(real.itertuples(index=False)):
        stem, name = Path(clip.path).stem, tool_for(Path(clip.path).stem)
        target = f"data/train/vocoded/{name}/{stem}.wav"
        if not (ROOT / target).exists():
            (ROOT / target).parent.mkdir(parents=True, exist_ok=True)
            sf.write(ROOT / target, rebuild(tools[name], load_audio(ROOT / clip.path)), SR)
        vocoded.append({**clip._asdict(), "path": target, "label": "fake", "source": "vocoded",
                        "generator": f"voc-{name}", "consent_note": CONSENT})
        if (i + 1) % 100 == 0:
            print(f"  vocoded {i + 1}/{len(real)}", flush=True)
    vocoded = pd.DataFrame(vocoded)
    del tools
    torch.cuda.empty_cache()

    # Copies of every training clip: compressed twice, and 1 phone copy for the vocoded ones.
    clean = pd.concat([manifest[manifest.version == "clean"], vocoded])
    jobs, rows = [], []
    for clip in clean[clean.split == "train"].itertuples(index=False):
        stem, folder = Path(clip.path).stem, clip.generator or "real"
        for version in ("comp1", "comp2"):
            target = f"data/train/compressed/{version}/{folder}/{stem}.wav"
            jobs.append((ROOT / clip.path, ROOT / target, lambda wav, seed: compress_clip(wav, seed=seed)[0]))
            rows.append({**clip._asdict(), "path": target, "version": version})
        if clip.source == "vocoded":
            target = f"data/train/vocoded/phone/{stem}.wav"
            jobs.append((ROOT / clip.path, ROOT / target,
                         lambda wav, seed: degrade(wav, seed=seed, config=PhoneConfig())[0]))
            rows.append({**clip._asdict(), "path": target, "version": "phone1"})
    print(f"Making {len(jobs)} copies ...", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, _ in enumerate(pool.map(make_copy, jobs)):
            if (i + 1) % 1000 == 0:
                print(f"  copies {i + 1}/{len(jobs)}", flush=True)

    kept = manifest[manifest.version != "phone2"]
    table = pd.concat([kept, vocoded, pd.DataFrame(rows)], ignore_index=True)
    table.to_csv(OUT, index=False)
    print(table.groupby(["split", "version"]).size().to_string())
    print(f"Wrote {OUT.relative_to(ROOT)}: {len(table)} rows")


if __name__ == "__main__":
    main()
