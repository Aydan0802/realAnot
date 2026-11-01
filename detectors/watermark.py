"""
watermark.py - layer 1: look for an AudioSeal watermark.

AudioSeal (Meta, MIT licence) hides an inaudible mark in AI-generated audio,
and a matching detector finds it. Limits to keep in mind:
  - It only finds AudioSeal's own mark. Open-source voice cloning tools add
    no watermark, so "no watermark" means UNKNOWN, never "real".
  - A watermark means "AI-generated", not "scam" (banks, bots and
    accessibility tools use legitimate AI voices).
  - In our demo, only clips we watermarked ourselves will carry the mark.

Safety: AudioSeal's own loader uses torch.load(weights_only=False), which can
run code hidden in a checkpoint. PyTorch's safe mode (weights_only=True)
cannot read these files (it does not support the defaultdict they contain),
so we use a RESTRICTED unpickler instead: it refuses every Python class or
function except the short allow-list below. Pickle can only run code by
loading a class or function, so anything outside the list is blocked.
The list is exactly what the checkpoints contain (checked with
torch.serialization.get_unsafe_globals_in_checkpoint) plus PyTorch's own
tensor-rebuilding helpers.
"""
import functools
import pickle
import types
from _compat_pickle import IMPORT_MAPPING, NAME_MAPPING  # Python 2 -> 3 names (standard library)
from pathlib import Path

import numpy as np
import torch
from audioseal import AudioSeal, loader
from audioseal.libs.moshi.utils.compile import no_compile

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "downloads" / "audioseal"
SR = 16000
DETECT_THRESHOLD = 0.5  # AudioSeal's default: over half the audio carries the mark

ALLOWED = {
    ("collections", "OrderedDict"), ("collections", "defaultdict"), ("typing", "Any"),
    ("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_parameter"),
    ("omegaconf.dictconfig", "DictConfig"), ("omegaconf.listconfig", "ListConfig"),
    ("omegaconf.nodes", "AnyNode"), ("omegaconf.base", "Metadata"),
    ("omegaconf.base", "ContainerMetadata"),
    *(("builtins", n) for n in ("dict", "list", "tuple", "set", "int", "float", "str", "bool")),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Older pickles use Python 2 names (e.g. __builtin__.long); check the
        # Python 3 name using the same table pickle itself uses.
        checked = NAME_MAPPING.get((module, name)) or (IMPORT_MAPPING.get(module, module), name)
        if checked not in ALLOWED:
            raise pickle.UnpicklingError(f"blocked unsafe object in checkpoint: {module}.{name}")
        return super().find_class(module, name)


_restricted_pickle = types.SimpleNamespace(__name__="restricted_pickle", Unpickler=_RestrictedUnpickler,
                                           load=lambda f, **kw: _RestrictedUnpickler(f, **kw).load())


def _safe_load_checkpoint(model_path, device="cpu"):
    # weights_only must be False to pass our own pickle module; the
    # restricted unpickler is what keeps this safe.
    return torch.load(model_path, map_location=device, weights_only=False,
                      pickle_module=_restricted_pickle)


def _local_checkpoint(url_or_path, device="cpu"):
    """AudioSeal's model cards point to a download URL; use our local copy instead."""
    local = CKPT_DIR / Path(str(url_or_path)).name  # detector_base.pth / generator_base.pth
    if not local.exists():
        raise FileNotFoundError(f"{local} not found. Download it first (see README.md).")
    return _safe_load_checkpoint(local, device)


loader._safe_load_checkpoint = _safe_load_checkpoint  # replace the unsafe loader
loader.load_model_checkpoint = _local_checkpoint       # and never download outside the project


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@functools.lru_cache(maxsize=1)
def get_detector():
    # The model card supplies the settings; the weights come from CKPT_DIR.
    return AudioSeal.load_detector("audioseal_detector_16bits").eval().to(_device())


@functools.lru_cache(maxsize=1)
def get_generator():
    return AudioSeal.load_generator("audioseal_wm_16bits").eval().to(_device())


def _as_tensor(wav):
    return torch.from_numpy(np.asarray(wav, dtype=np.float32)).reshape(1, 1, -1).to(_device())


def detect(wav):
    """Share of the clip (0 to 1) that carries an AudioSeal watermark."""
    with torch.inference_mode():
        score, _message = get_detector().detect_watermark(_as_tensor(wav), sample_rate=SR)
    return float(score)


def add_watermark(wav):
    """Return a watermarked copy of a 16 kHz clip (simulates a cooperating TTS provider)."""
    # no_compile: AudioSeal would otherwise try torch.compile, which needs
    # Triton (not available on Windows).
    with torch.inference_mode(), no_compile():
        x = _as_tensor(wav)
        marked = x + get_generator().get_watermark(x, sample_rate=SR)
    return marked.squeeze().cpu().numpy()
