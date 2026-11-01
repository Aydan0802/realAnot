"""
phone.py - make a clean recording sound like it went through a phone call.

Research (see docs/detector_research.md) says detectors fail mostly
because they are tested on clean audio, while scam calls arrive through a
phone channel. This module simulates that channel:

  1. background noise at a random SNR (5 to 30 dB), added at the caller's end
  2. resample to 8 kHz (a narrowband phone line)
  3. band-pass 300 to 3400 Hz (the classic telephone band)
  4. codec round trip through ffmpeg: G.711 mu-law / A-law, GSM, AMR-NB or Opus
  5. packet loss: random 20 ms frames dropped to silence (a bad mobile signal)
  6. resample back to 16 kHz (what the detectors expect)
  7. optional MP3 re-encode (e.g. a recording forwarded on a messaging app)

The same seed always gives the same output, so experiments can be repeated.

Run (from the project root):
    python -m audio.phone in.wav out.wav --seed 0 --codec gsm
"""
import argparse
import os
import subprocess
import tempfile
from dataclasses import dataclass

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

PHONE_SR = 8000
OUT_SR = 16000
FRAME_S = 0.02  # phone audio travels in 20 ms packets

# codec name -> (file extension, ffmpeg encoder arguments)
CODECS = {
    "mulaw": ("wav", ["-c:a", "pcm_mulaw"]),                            # G.711, landlines
    "alaw": ("wav", ["-c:a", "pcm_alaw"]),                              # G.711, Europe/Asia
    "gsm": ("gsm", ["-c:a", "libgsm"]),                                 # old 2G mobile
    "amrnb": ("amr", ["-c:a", "libopencore_amrnb", "-b:a", "12.2k"]),   # 3G/4G mobile
    "opus": ("ogg", ["-c:a", "libopus", "-b:a", "16k"]),                # WhatsApp/VoIP calls
}

# Attached clips stay wideband (sound up to 8 kHz). ASSUMPTION: typical bitrates.
CLIP_CODECS = {
    "aac": ("m4a", ["-c:a", "aac", "-b:a", "64k"]),                     # web video, phone recordings
    "mp3": ("mp3", ["-c:a", "libmp3lame", "-b:a", "64k"]),              # forwarded audio files
    "opus": ("ogg", ["-c:a", "libopus", "-b:a", "24k"]),                # WhatsApp/Telegram voice notes
}


@dataclass
class PhoneConfig:
    codec: str | None = "random"                      # a key of CODECS, "random", or None
    snr_db: tuple[float, float] | None = (5.0, 30.0)  # noise SNR range, or None for no noise
    packet_loss: tuple[float, float] = (0.0, 0.05)    # share of 20 ms frames dropped
    mp3: bool = False


def _ffmpeg_round_trip(wav, sr, ext, encoder_args):
    """Encode wav with ffmpeg, decode it back, and return the decoded audio."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as tmp:
        src, enc, dec = (os.path.join(tmp, name) for name in ("in.wav", f"enc.{ext}", "out.wav"))
        sf.write(src, wav, sr, subtype="PCM_16")
        for cmd in ([ffmpeg, "-y", "-i", src, "-ar", str(sr), "-ac", "1", *encoder_args, enc],
                    [ffmpeg, "-y", "-i", enc, "-ar", str(sr), "-ac", "1", dec]):
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{done.stderr[-500:]}")
        out, _ = sf.read(dec, dtype="float32")
    # Some codecs add or drop a few samples; keep the original length.
    return np.pad(out, (0, max(0, len(wav) - len(out))))[:len(wav)]


def add_noise(wav, snr_db, rng):
    """Add white noise so that signal power / noise power = snr_db."""
    signal_power = np.mean(wav ** 2) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10))
    return wav + rng.normal(0, np.sqrt(noise_power), len(wav)).astype(np.float32)


def drop_packets(wav, sr, loss_rate, rng):
    """Silence random 20 ms frames. Real phones partly hide lost packets
    (concealment); silence is the simple worst case."""
    frame = int(FRAME_S * sr)
    out = wav.copy()
    for i in np.flatnonzero(rng.random(len(wav) // frame) < loss_rate):
        out[i * frame:(i + 1) * frame] = 0.0
    return out


def degrade(wav, sr=OUT_SR, seed=0, config=PhoneConfig()):
    """Return (phone-quality 16 kHz float32 audio, info about what was applied)."""
    rng = np.random.default_rng(seed)
    wav = np.asarray(wav, dtype=np.float32)
    info = {"codec": None, "snr_db": None, "packet_loss": 0.0}

    if config.snr_db is not None:
        info["snr_db"] = round(float(rng.uniform(*config.snr_db)), 1)
        wav = add_noise(wav, info["snr_db"], rng)

    wav = librosa.resample(wav, orig_sr=sr, target_sr=PHONE_SR)
    sos = butter(4, [300, 3400], btype="band", fs=PHONE_SR, output="sos")
    wav = sosfiltfilt(sos, wav).astype(np.float32)

    codec = str(rng.choice(sorted(CODECS))) if config.codec == "random" else config.codec
    if codec is not None:
        info["codec"] = codec
        wav = _ffmpeg_round_trip(wav, PHONE_SR, *CODECS[codec])

    info["packet_loss"] = round(float(rng.uniform(*config.packet_loss)), 3)
    if info["packet_loss"] > 0:
        wav = drop_packets(wav, PHONE_SR, info["packet_loss"], rng)

    wav = librosa.resample(wav, orig_sr=PHONE_SR, target_sr=OUT_SR)
    if config.mp3:
        wav = _ffmpeg_round_trip(wav, OUT_SR, "mp3", ["-c:a", "libmp3lame", "-b:a", "32k"])

    # Keep the peak below 1.0 so writing to a 16-bit file never clips.
    peak = np.abs(wav).max()
    if peak > 0.99:
        wav = wav * (0.99 / peak)
    return wav.astype(np.float32), info


def compress_clip(wav, sr=OUT_SR, seed=0):
    """Return (16 kHz float32 audio, info) for a clip as it arrives by messaging
    app or the web: still wideband, compressed once, and half the time
    compressed again, because forwarded clips are re-encoded by each app."""
    rng = np.random.default_rng(seed)
    wav = np.asarray(wav, dtype=np.float32)
    if sr != OUT_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=OUT_SR)
    names = [str(rng.choice(sorted(CLIP_CODECS)))]
    if rng.random() < 0.5:
        names.append(str(rng.choice(sorted(CLIP_CODECS))))
    for name in names:
        wav = _ffmpeg_round_trip(wav, OUT_SR, *CLIP_CODECS[name])
    peak = np.abs(wav).max()
    if peak > 0.99:
        wav = wav * (0.99 / peak)
    return wav.astype(np.float32), {"codec": "+".join(names)}


def main():
    ap = argparse.ArgumentParser(description="Make a clip sound like a phone call")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--codec", default="random", help=f"one of {sorted(CODECS)}, random, or none")
    ap.add_argument("--mp3", action="store_true")
    args = ap.parse_args()

    wav, sr = sf.read(args.input, dtype="float32", always_2d=True)
    codec = None if args.codec == "none" else args.codec
    out, info = degrade(wav.mean(axis=1), sr, args.seed, PhoneConfig(codec=codec, mp3=args.mp3))
    sf.write(args.output, out, OUT_SR)
    print(f"wrote {args.output}: {info}")


if __name__ == "__main__":
    main()
