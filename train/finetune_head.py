"""
finetune_head.py - adapt the detector to phone-quality audio.

What is trained: ONLY the detector's "head" (the AASIST graph pooling and the
small classifier on top). The XLS-R backbone (300M parameters) stays frozen,
so the model keeps everything it learned from millions of clips; we only
re-teach how it reads those features, now including phone-quality audio.
This fits on a 6 GB laptop GPU and takes minutes, not days.

Training data: data/train_manifest.csv, split "train" (each clip clean plus
2 phone copies). Each step uses a random 5 s slice of each clip (random
slices are fine for TRAINING; scoring always uses fixed windows).

Validation after every epoch: the "val" speakers/voices, clean, compressed
and phone (seeded copies, made once), scored exactly like the real detector
(whole clip, 5 s windows). Epoch 0 is the original model. The epoch with the
best selection score (clean EER + compressed-clip EER on the worst fake
generator) is saved to models/finetuned/head.safetensors. The test set is NOT
touched here.

Run from the project root:
    python -m train.finetune_head --epochs 8
"""
import argparse
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from safetensors.torch import load_file, save_file
from sklearn.metrics import roc_curve

from audio.phone import PhoneConfig, compress_clip, degrade
from detectors.forensics_xlsr import (KNOWN_UNUSED_PREFIXES, SR, WEIGHTS, WINDOW_S,
                                      ForensicsDetector, load_audio, make_windows)

ROOT = Path(__file__).resolve().parent.parent
HEAD_OUT = ROOT / "models" / "finetuned" / "head.safetensors"
LOG_OUT = ROOT / "results" / "train" / "log.csv"
HEAD_PREFIXES = ("pool.", "classifier.")


def build_model(device):
    model = ForensicsDetector()
    result = model.load_state_dict(load_file(str(WEIGHTS)), strict=False)
    bad = [k for k in result.unexpected_keys if not k.startswith(KNOWN_UNUSED_PREFIXES)]
    if result.missing_keys or bad:
        raise RuntimeError(f"checkpoint mismatch: {result.missing_keys[:5]} {bad[:5]}")
    for p in model.wavlm.parameters():
        p.requires_grad = False  # freeze the backbone
    return model.to(device)


def forward(model, batch):
    """Frozen backbone in fp16 without gradients, then the head in fp32.
    Returns logits where sigmoid(logit) = P(real), as in the original model."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=batch.is_cuda):
        hidden = model.wavlm(batch, output_hidden_states=True).hidden_states[1:]
    return model.classifier(model.pool([h.float() for h in hidden])).squeeze(-1)


def random_slice(path, rng, n=int(WINDOW_S * SR)):
    """Read one random 5 s slice from disk, so the clips need not all fit in memory."""
    info = sf.info(path)
    if info.samplerate != SR or info.channels != 1:
        raise ValueError(f"{path} is not {SR} Hz mono")
    start = rng.integers(0, info.frames - n + 1) if info.frames > n else 0
    piece, _ = sf.read(path, start=start, frames=n, dtype="float32")
    if len(piece) < n:
        piece = np.tile(piece, int(np.ceil(n / len(piece))))[:n]
    return piece / (np.abs(piece).max() + 1e-8)  # peak-normalise, as at inference


def eer(labels_fake, scores):
    fpr, tpr, _ = roc_curve(labels_fake, scores)
    i = np.argmin(np.abs((1 - tpr) - fpr))
    return 100 * (fpr[i] + 1 - tpr[i]) / 2


@torch.no_grad()
def validate(model, val_sets, generators, device):
    """Validation EER for each condition, scoring whole clips in fixed windows.
    On degraded audio also one EER per fake generator: a good overall number
    can hide a blind spot for one kind of fake, which is what went wrong
    before (docs/eval_results.md, Run 6)."""
    model.eval()
    out = {}
    for condition, (wavs, labels) in val_sets.items():
        scores = []
        for wav in wavs:
            windows = torch.from_numpy(make_windows(wav / (np.abs(wav).max() + 1e-8))).to(device)
            scores.append(float((1 - torch.sigmoid(forward(model, windows))).mean()))
        out[condition] = eer(labels, scores)
        if condition != "clean":
            scores = np.array(scores)
            for name in sorted(set(generators) - {""}):
                keep = (generators == name) | (generators == "")  # this fake + all real clips
                out[f"{condition}_{name}"] = eer(labels[keep], scores[keep])
    return out


def selection_score(v):
    """Lower is better: clean EER plus compressed-clip EER on the WORST fake
    generator. Attached clips are the current stage, so phone EER is logged
    but not selected on."""
    worst = max(value for key, value in v.items() if key.startswith("compressed_"))
    return (v["clean"] + worst) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", default="data/train_manifest.csv")
    ap.add_argument("--out", default=str(HEAD_OUT.relative_to(ROOT)), help="where to save the best head")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="aim for 0.05/0.95 instead of 0/1 (at 0.1), so the head stays less overconfident")
    ap.add_argument("--hold-out", nargs="+", default=[],
                    help="leave these generators out of TRAINING but keep them in validation, "
                         "to measure how the settings cope with a kind of fake never trained on")
    args = ap.parse_args()
    head_out, log_out = ROOT / args.out, (ROOT / args.out).with_suffix(".log.csv")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    manifest = pd.read_csv(ROOT / args.manifest, keep_default_na=False)
    train = manifest[manifest.split == "train"]
    train = train[~train.generator.isin(args.hold_out)]
    val = manifest[(manifest.split == "val") & (manifest.version == "clean")]
    print(f"{len(train)} training files; loading {len(val)} validation clips into memory ...")
    train_paths = [ROOT / p for p in train.path]
    train_y = torch.tensor((train.label == "real").to_numpy(), dtype=torch.float32)  # 1 = real
    targets = train_y * (1 - args.label_smoothing) + args.label_smoothing / 2
    val_clean = [load_audio(ROOT / p) for p in val.path]
    val_phone = [degrade(w, seed=zlib.crc32(p.encode()), config=PhoneConfig())[0]
                 for w, p in zip(val_clean, val.path)]
    val_compressed = [compress_clip(w, seed=zlib.crc32(p.encode()))[0] for w, p in zip(val_clean, val.path)]
    val_labels = (val.label == "fake").astype(int).to_numpy()
    val_sets = {"clean": (val_clean, val_labels), "compressed": (val_compressed, val_labels),
                "phone": (val_phone, val_labels)}
    val_generators = val.generator.to_numpy()

    model = build_model(device)
    head = [p for n, p in model.named_parameters() if n.startswith(HEAD_PREFIXES)]
    print(f"Training {sum(p.numel() for p in head):,} head parameters; backbone frozen.")
    optimiser = torch.optim.AdamW(head, lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = int(np.ceil(len(train_paths) / args.batch))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs * steps_per_epoch)
    # If there are more fake than real clips, weight the real ones up so each
    # class counts equally (otherwise the head drifts towards saying "fake").
    n_real, n_fake = int(train_y.sum()), int(len(train_y) - train_y.sum())
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(n_fake / n_real, device=device))
    print(f"Training clips: {n_real} real, {n_fake} fake (real weighted x{n_fake / n_real:.2f})")

    def describe(v):
        def by_generator(condition):
            return "  ".join(f"{k[len(condition) + 1:]} {x:.2f}%" for k, x in v.items()
                             if k.startswith(condition + "_"))
        return (f"val EER clean {v['clean']:.2f}%  compressed {v['compressed']:.2f}% ({by_generator('compressed')})"
                f"  phone {v['phone']:.2f}% ({by_generator('phone')})")

    def log_row(epoch, train_loss, v, seconds):
        return {"epoch": epoch, "train_loss": train_loss, **{f"val_eer_{k}": round(x, 2) for k, x in v.items()},
                "selection": round(selection_score(v), 2), "seconds": seconds}

    v = validate(model, val_sets, val_generators, device)
    log = [log_row(0, None, v, 0)]
    print(f"epoch 0 (original model): {describe(v)}")
    best = baseline = selection_score(v)
    head_out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        model.wavlm.eval()  # keep the frozen backbone in inference mode (no dropout)
        order = rng.permutation(len(train_paths))
        losses = []
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            batch = torch.from_numpy(np.stack([random_slice(train_paths[j], rng) for j in idx])).float().to(device)
            loss = loss_fn(forward(model, batch), targets[idx].to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()
            losses.append(loss.item())

        v = validate(model, val_sets, val_generators, device)
        log.append(log_row(epoch, round(float(np.mean(losses)), 4), v, round(time.time() - start)))
        saved = ""
        if selection_score(v) < best:
            best = selection_score(v)
            save_file({k: t.contiguous().cpu() for k, t in model.state_dict().items()
                       if k.startswith(HEAD_PREFIXES)}, str(head_out))
            saved = "  <- best so far, saved"
        print(f"epoch {epoch}: loss {np.mean(losses):.4f}  {describe(v)}  ({time.time() - start:.0f} s){saved}")

    pd.DataFrame(log).to_csv(log_out, index=False)
    if best == baseline:
        print("No epoch beat the original model on validation; nothing saved.")
    else:
        print(f"Best head saved to {args.out}; log in {log_out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
