"""
make_tts.py - make synthetic (fake) voice clips with free, open-source TTS.

Each real LibriSpeech sentence is read again by ONE synthetic voice, so the
fake set has the same sentences as the real set. Two different TTS designs
are used so results can be reported per generator:
  - Piper   (VITS, end-to-end; voices from rhasspy/piper-voices)
  - SpeechT5 (Microsoft; acoustic model + separate HiFi-GAN vocoder,
              speaker voices from CMU Arctic x-vectors)
No real person's voice is cloned: these are the tools' own stock voices.

Run from the project root (after data_prep.prepare_real):
    python -m data_prep.make_tts
"""
import argparse
import csv
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
DL = ROOT / "downloads"
TRANSCRIPTS = ROOT / "data" / "real" / "librispeech" / "transcripts.csv"
OUT = ROOT / "data" / "fake_tts"
SR = 16000

PIPER_VOICES = {  # voice name -> .onnx file
    "lessac": "en/en_US/lessac/medium/en_US-lessac-medium.onnx",
    "amy": "en/en_US/amy/medium/en_US-amy-medium.onnx",
    "ryan": "en/en_US/ryan/medium/en_US-ryan-medium.onnx",
    "alan": "en/en_GB/alan/medium/en_GB-alan-medium.onnx",
}
SPEECHT5_VOICES = ["awb", "bdl", "clb", "jmk", "ksp", "rms", "slt"]  # CMU Arctic speakers


class Piper:
    def __init__(self, voices=PIPER_VOICES):
        from piper import PiperVoice
        self.voices = {name: PiperVoice.load(str(DL / "piper_voices" / path))
                       for name, path in voices.items()}

    def say(self, text, voice):
        chunks = list(self.voices[voice].synthesize(text))
        wav = np.concatenate([c.audio_float_array for c in chunks])
        return librosa.resample(wav, orig_sr=chunks[0].sample_rate, target_sr=SR)


class SpeechT5:
    def __init__(self):
        from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = SpeechT5Processor.from_pretrained(DL / "speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained(DL / "speecht5_tts").to(self.device).eval()
        self.vocoder = SpeechT5HifiGan.from_pretrained(DL / "speecht5_hifigan").to(self.device).eval()
        self.speakers = {}
        for name in SPEECHT5_VOICES:  # one x-vector (voice fingerprint) per CMU Arctic speaker
            first = sorted((DL / "xvectors" / "spkrec-xvect").glob(f"cmu_us_{name}_*.npy"))[0]
            vec = np.load(first, allow_pickle=False)
            self.speakers[name] = torch.from_numpy(vec).float().reshape(1, -1).to(self.device)

    def say(self, text, voice):
        ids = self.processor(text=text, return_tensors="pt")["input_ids"].to(self.device)
        with torch.inference_mode():
            wav = self.model.generate_speech(ids, self.speakers[voice], vocoder=self.vocoder)
        return wav.cpu().numpy()  # already 16 kHz


class Orpheus:
    """Orpheus 3B fine-tuned on Singlish (codec-based TTS, the newer design).

    It is a Llama language model that writes audio as tokens of the SNAC
    neural codec; SNAC then decodes the tokens into a 24 kHz waveform. This is
    how many current voice-cloning tools work, unlike Piper/SpeechT5.
    Loaded in 4-bit (the precision it was fine-tuned in) to fit a 6 GB GPU.
    """
    # Special token ids from the Orpheus / Unsloth recipe.
    START_HUMAN, END_HUMAN, END_TEXT = 128259, 128260, 128009
    START_SPEECH, END_SPEECH, PAD = 128257, 128258, 128263
    AUDIO_OFFSET = 128266  # audio token k is stored as id AUDIO_OFFSET + k

    def __init__(self):
        from snac import SNAC
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        path = DL / "orpheus_singlish"
        self.tok = AutoTokenizer.from_pretrained(path)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(path, quantization_config=quant,
                                                          device_map="cuda:0").eval()
        # SNAC only ships a pickle (.bin), so load it with weights_only=True
        # (plain tensors, no code) and a strict key check.
        snac_dir = DL / "snac_24khz"
        self.snac = SNAC.from_config(str(snac_dir / "config.json"))
        state = torch.load(snac_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
        self.snac.load_state_dict(state, strict=True)
        self.snac = self.snac.to("cuda").eval()

    def _prompt(self, text, voice):
        ids = self.tok(f"{voice}: {text}" if voice else text).input_ids
        return [self.START_HUMAN] + ids + [self.END_TEXT, self.END_HUMAN]

    def _decode(self, tokens):
        """Generated ids -> waveform: keep the codes after START_SPEECH, then
        undo Orpheus's 7-tokens-per-frame layout into SNAC's 3 code layers."""
        tokens = tokens.tolist()
        if self.START_SPEECH in tokens:
            tokens = tokens[len(tokens) - tokens[::-1].index(self.START_SPEECH):]
        codes = [t - self.AUDIO_OFFSET for t in tokens if t >= self.AUDIO_OFFSET]
        codes = codes[:len(codes) // 7 * 7]
        if not codes:
            return None
        f = np.array(codes).reshape(-1, 7) - np.arange(7) * 4096  # each slot has its own 4096-code range
        if f.min() < 0 or f.max() >= 4096:
            return None  # malformed output; the caller retries or skips
        layers = [f[:, [0]], f[:, [1, 4]], f[:, [2, 3, 5, 6]]]
        layers = [torch.from_numpy(l.reshape(1, -1)).long().to("cuda") for l in layers]
        with torch.inference_mode():
            return self.snac.decode(layers).squeeze().float().cpu().numpy()

    def say_batch(self, texts, voice=None, max_new_tokens=1400):
        """Read several sentences at once (faster on the GPU). Returns 16 kHz
        waveforms, or None where the model produced no usable audio."""
        prompts = [self._prompt(t, voice) for t in texts]
        width = max(map(len, prompts))
        ids = torch.tensor([[self.PAD] * (width - len(p)) + p for p in prompts], device="cuda")
        mask = (ids != self.PAD).long()  # left padding is ignored
        with torch.inference_mode():
            out = self.model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens,
                                      do_sample=True, temperature=0.6, top_p=0.95, repetition_penalty=1.1,
                                      eos_token_id=self.END_SPEECH, pad_token_id=self.PAD)
        wavs = []
        for row in out[:, width:]:
            wav = self._decode(row)
            wavs.append(None if wav is None else librosa.resample(wav, orig_sr=24000, target_sr=SR))
        return wavs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N sentences (0 = all)")
    args = ap.parse_args()

    if not TRANSCRIPTS.exists():
        raise SystemExit("Run `python -m data_prep.prepare_real` first.")
    with open(TRANSCRIPTS, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    generators = {"piper": (Piper(), list(PIPER_VOICES)), "speecht5": (SpeechT5(), SPEECHT5_VOICES)}
    counts = {name: 0 for name in generators}
    for i, row in enumerate(rows):
        # Alternate generators sentence by sentence, and rotate through each one's voices.
        gen_name = "piper" if i % 2 == 0 else "speecht5"
        tts, voices = generators[gen_name]
        voice = voices[(i // 2) % len(voices)]
        out_path = OUT / gen_name / f"{gen_name}-{voice}__{row['utt_id']}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not out_path.exists():  # lets you re-run after an interruption
            wav = tts.say(row["text"], voice)
            sf.write(out_path, wav / (np.abs(wav).max() + 1e-8) * 0.9, SR)
        counts[gen_name] += 1
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(rows)} done")
    print(f"Fake clips: {counts} in {OUT}")


if __name__ == "__main__":
    main()
