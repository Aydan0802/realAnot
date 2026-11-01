"""
forensics_xlsr.py - wrapper around the open-source forensics XLS-R + AASIST
voice deepfake detector (eliya/forensics_0.3B_xlsr_wild_deepfake_classifier,
CC-BY-NC-4.0, see CITATIONS.md).

What this wrapper adds on top of the authors' own inference.py:
  1. Scores the WHOLE clip with overlapping 5 s windows (hop 2.5 s) instead of
     only the centre 5 s. There are no random crops, so the same clip always
     gets the same score.
  2. Builds the model from the backbone's config only (no extra 1.2 GB
     download) and loads the weights from .safetensors. It stops with an error
     if any weight is missing or unexpected, except the 4 known training-only
     "projection.*" tensors.
  3. Reports how much the windows agree, so the app can say "inconclusive"
     instead of guessing.

The score is RAW: not yet calibrated. Turning it into a trustworthy
probability needs our own phone-quality validation clips (a later step).

Run (from the project root):
    python -m detectors.forensics_xlsr path/to/clip.wav [more.wav ...]
"""
import argparse
import functools
import importlib.util
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file
from transformers import Wav2Vec2Config, Wav2Vec2Model

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "forensics_xlsr"
WEIGHTS = MODEL_DIR / "checkpoint_epoch_4.safetensors"
BACKBONE = "facebook/wav2vec2-xls-r-300m"  # only its small config.json is downloaded

SR = 16000          # the model expects 16 kHz mono audio
WINDOW_S = 5.0      # the model was trained on 5 s clips
HOP_S = 2.5         # 50% overlap between windows
BATCH = 4           # windows per forward pass (fits a 6 GB GPU)

# Tensors in the checkpoint that the authors' model.py never uses (a
# training-only head). Any OTHER missing or unexpected tensor is an error.
KNOWN_UNUSED_PREFIXES = ("projection.",)

# ASSUMPTION placeholders until we calibrate on our own data:
MIN_SECONDS = 2.0          # shorter clips are too little evidence -> inconclusive
MAX_WINDOW_SPREAD = 0.25   # windows disagree more than this (std) -> inconclusive


def _load_author_classes():
    """Import the AASISTPool and Classifier layers from the authors' model.py.
    (We read that file: it only defines PyTorch layers. Its DeepfakeDetector
    class would download the backbone again, so we do not use it.)"""
    path = MODEL_DIR / "model.py"
    if not path.exists() or not WEIGHTS.exists():
        raise FileNotFoundError(
            f"Model files not found in {MODEL_DIR}. Download them first (see README.md).")
    spec = importlib.util.spec_from_file_location("forensics_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AASISTPool, module.Classifier


class ForensicsDetector(torch.nn.Module):
    """Same layout as the authors' DeepfakeDetector. The attribute names
    (wavlm / pool / classifier) must match the names in the checkpoint."""

    def __init__(self, wavlm=None):
        super().__init__()
        AASISTPool, Classifier = _load_author_classes()
        config = Wav2Vec2Config.from_pretrained(BACKBONE)
        # wavlm: an already loaded backbone to share. Otherwise build an empty
        # one; its weights come from the checkpoint.
        self.wavlm = Wav2Vec2Model(config) if wavlm is None else wavlm
        self.pool = AASISTPool(config.hidden_size, config.num_hidden_layers, (64, 32))
        self.classifier = Classifier(self.pool.output_dim, 160, 3)

    def forward(self, waveform):
        """(batch, samples) at 16 kHz -> one logit per clip. sigmoid(logit) = P(real)."""
        hidden = list(self.wavlm(waveform, output_hidden_states=True).hidden_states[1:])
        return self.classifier(self.pool(hidden)).squeeze(-1)


@functools.lru_cache(maxsize=1)
def _load_original():
    """Load the original detector once (loading takes several seconds)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ForensicsDetector()
    state = load_file(str(WEIGHTS))
    result = model.load_state_dict(state, strict=False)

    # strict=False would silently run on random weights if names did not
    # match, so check every mismatch ourselves.
    unexpected = [k for k in result.unexpected_keys if not k.startswith(KNOWN_UNUSED_PREFIXES)]
    if result.missing_keys or unexpected:
        raise RuntimeError(
            "Checkpoint does not match the model.\n"
            f"  missing ({len(result.missing_keys)}): {result.missing_keys[:10]}\n"
            f"  unexpected ({len(unexpected)}): {unexpected[:10]}")
    print(f"[forensics_xlsr] loaded {len(state) - len(result.unexpected_keys)} weight tensors "
          f"on {device}; skipped {len(result.unexpected_keys)} known unused 'projection.*' tensors")
    return model.eval().to(device), device


@functools.lru_cache(maxsize=4)
def get_model(head=None):
    """The detector with its original head, or with a fine-tuned head
    (train/finetune_head.py) that replaces the pool + classifier weights.
    Every head shares one backbone, so extra heads use little memory."""
    original, device = _load_original()
    if not head:
        return original, device
    model = ForensicsDetector(wavlm=original.wavlm)
    head_state = load_file(str(head))
    result = model.load_state_dict(head_state, strict=False)
    missing = [k for k in result.missing_keys if k.startswith(("pool.", "classifier."))]
    if missing or result.unexpected_keys:
        raise RuntimeError(f"Fine-tuned head does not match: missing {missing[:5]}, "
                           f"unexpected {result.unexpected_keys[:5]}")
    print(f"[forensics_xlsr] fine-tuned head loaded from {head} ({len(head_state)} tensors)")
    return model.eval().to(device), device


def load_audio(path):
    """Read an audio file as 16 kHz mono float32."""
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)  # stereo -> mono
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav


def make_windows(wav):
    """Cut the clip into 5 s windows, 2.5 s apart. The last window is lined up
    with the end of the clip so no audio is skipped. Clips shorter than 5 s
    are repeated to fill 5 s, the same way the authors handle them."""
    n, hop = int(WINDOW_S * SR), int(HOP_S * SR)
    if len(wav) < n:
        wav = np.tile(wav, int(np.ceil(n / len(wav))))[:n]
    starts = list(range(0, len(wav) - n + 1, hop))
    if starts[-1] + n < len(wav):
        starts.append(len(wav) - n)
    return np.stack([wav[s:s + n] for s in starts]).astype(np.float32)


def score_clip(wav, head=None):
    """Score one 16 kHz mono clip. Returns a dict (see the keys below).
    head: optional fine-tuned head path (None = the original model)."""
    if len(wav) == 0:
        raise ValueError("empty audio")
    model, device = get_model(str(head) if head else None)
    duration = len(wav) / SR
    wav = wav / (np.abs(wav).max() + 1e-8)  # peak-normalise, as in the authors' code
    windows = torch.from_numpy(make_windows(wav))

    fake_scores = []
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16,
                                                enabled=(device == "cuda")):
        for i in range(0, len(windows), BATCH):
            logits = model(windows[i:i + BATCH].to(device)).float()
            fake_scores.append(1.0 - torch.sigmoid(logits).cpu())  # P(fake) = 1 - P(real)
    scores = torch.cat(fake_scores).numpy()

    spread = float(scores.std())
    reasons = []
    if duration < MIN_SECONDS:
        reasons.append(f"clip is only {duration:.1f} s long")
    if spread > MAX_WINDOW_SPREAD:
        reasons.append(f"windows disagree (spread {spread:.2f})")
    return {
        "fake_score_raw": float(scores.mean()),   # 0 = sounds real, 1 = sounds synthetic
        "window_scores": [round(float(s), 4) for s in scores],
        "window_spread": spread,
        "duration_s": round(duration, 2),
        "n_windows": len(scores),
        "inconclusive": bool(reasons),
        "reasons": reasons,
    }


def main():
    ap = argparse.ArgumentParser(description="Score audio clips with the forensics XLS-R detector")
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()
    for path in args.files:
        start = time.perf_counter()
        r = score_clip(load_audio(path))
        status = "INCONCLUSIVE: " + "; ".join(r["reasons"]) if r["inconclusive"] else "ok"
        print(f"{path}: raw fake score {r['fake_score_raw']:.3f} over {r['n_windows']} window(s), "
              f"{r['duration_s']} s [{status}] ({time.perf_counter() - start:.1f} s)")


if __name__ == "__main__":
    main()
