"""Tests for audio/phone.py. Run from the project root:  python -m pytest tests"""
import numpy as np
import pytest

from audio.phone import CLIP_CODECS, CODECS, OUT_SR, PhoneConfig, compress_clip, degrade


def make_signal(seconds=3.0):
    """A synthetic stand-in for speech: tones inside AND outside the phone band."""
    t = np.arange(int(seconds * OUT_SR)) / OUT_SR
    return (0.3 * np.sin(2 * np.pi * 440 * t)        # inside the phone band
            + 0.3 * np.sin(2 * np.pi * 1200 * t)     # inside the phone band
            + 0.3 * np.sin(2 * np.pi * 6000 * t)     # above it; should be removed
            ).astype(np.float32)


def energy_share_above(wav, hz):
    spectrum = np.abs(np.fft.rfft(wav)) ** 2
    freqs = np.fft.rfftfreq(len(wav), 1 / OUT_SR)
    return spectrum[freqs > hz].sum() / spectrum.sum()


def test_same_seed_gives_same_output():
    a, info_a = degrade(make_signal(), seed=7)
    b, info_b = degrade(make_signal(), seed=7)
    assert info_a == info_b
    np.testing.assert_array_equal(a, b)


def test_different_seeds_give_different_output():
    a, _ = degrade(make_signal(), seed=1)
    b, _ = degrade(make_signal(), seed=2)
    assert not np.allclose(a, b)


def test_length_and_sample_rate_are_kept():
    wav = make_signal(seconds=2.5)
    out, _ = degrade(wav, seed=0)
    assert len(out) == len(wav)
    assert out.dtype == np.float32


def test_removes_sound_above_the_phone_band():
    out, _ = degrade(make_signal(), seed=0, config=PhoneConfig(codec="mulaw", snr_db=None))
    assert energy_share_above(make_signal(), 4000) > 0.25   # the input has plenty
    assert energy_share_above(out, 4000) < 0.01              # the phone line removes it


@pytest.mark.parametrize("codec", sorted(CODECS))
def test_every_codec_runs(codec):
    out, info = degrade(make_signal(), seed=0, config=PhoneConfig(codec=codec))
    assert info["codec"] == codec
    assert np.isfinite(out).all() and np.abs(out).max() > 0.01


def test_mp3_option_runs():
    out, _ = degrade(make_signal(), seed=0, config=PhoneConfig(codec=None, mp3=True))
    assert np.isfinite(out).all()


def test_packet_loss_silences_about_the_right_share():
    cfg = PhoneConfig(codec=None, snr_db=None, packet_loss=(0.3, 0.3))
    out, info = degrade(make_signal(seconds=10), seed=0, config=cfg)
    frame = int(0.02 * OUT_SR)
    frames = out[:len(out) // frame * frame].reshape(-1, frame)
    # Resampling back to 16 kHz smears sound into the edges of each silent
    # frame, so only look at the middle half of each frame.
    middles = frames[:, frame // 4: 3 * frame // 4]
    silent_share = (np.abs(middles).max(axis=1) < 0.02).mean()
    assert info["packet_loss"] == 0.3
    assert 0.2 < silent_share < 0.4


def test_noise_follows_the_snr_range():
    _, info = degrade(make_signal(), seed=3, config=PhoneConfig(codec=None))
    assert 5.0 <= info["snr_db"] <= 30.0


def test_compressed_clip_is_repeatable_and_keeps_its_length():
    a, info_a = compress_clip(make_signal(), seed=7)
    b, info_b = compress_clip(make_signal(), seed=7)
    assert info_a == info_b
    np.testing.assert_array_equal(a, b)
    assert len(a) == len(make_signal())


@pytest.mark.parametrize("seed", range(6))
def test_compressed_clip_stays_wideband(seed):
    out, info = compress_clip(make_signal(), seed=seed)
    assert set(info["codec"].split("+")) <= set(CLIP_CODECS)
    assert energy_share_above(out, 3800) > 0.0002   # the verdict must not mistake it for a phone call
