# Detector evaluation results

## Run 8: calibrated test of the best version so far (1 Nov 2026)

**What was run:** the original model and `final_slow` (trained on real clips, Piper and Orpheus, lr 3e-5, 4 epochs) scored on clean, compressed and phone copies. Their scores are combined by a logistic model fitted per channel on validation only (`detectors/calibration.json`). Outputs are in `results/final_test/`. This is not the final test: vocoded training is still to come, and that head will need its own test.

| Test set | Audio | Original EER | Combined EER | Fakes missed at 50% | Real flagged at 50% |
|---|---|---|---|---|---|
| Singapore (300 real, 498 fake) | clean | 7.92% | 0.00% | 0.2% | 0.0% |
| Singapore | compressed | 7.75% | 0.00% | 0.8% | 0.0% |
| Singapore | phone | 11.56% | 3.01% | 2.6% | 3.3% |
| LibriSpeech (200 real, 200 fake) | clean | 7.50% | 0.00% | 0.5% | 0.0% |
| LibriSpeech | compressed | 6.00% | 1.25% | 2.0% | 0.5% |
| LibriSpeech | phone | 16.25% | 5.50% | 2.0% | 13.0% |

Phone EER by generator, original then combined. Singapore: Orpheus 7.02% to 3.51%, Piper 26.67% to 3.17%, SpeechT5 (never trained on) 8.00% to 1.67%. LibriSpeech: Piper 25.25% to 6.75%, SpeechT5 8.00% to 2.75%.

**Real deepfakes, full verdict (6 clips):**

| Clip | As received | Phone (GSM) |
|---|---|---|
| Lee Hsien Loong, fake CGTN interview | HIGH 99% | HIGH 99% |
| Compilation of Lee and Wong deepfakes | HIGH 99% | HIGH 97% |
| Lawrence Wong deepfake, reposted 2025 | HIGH 99% | MEDIUM 50% |
| Lawrence Wong (DPM), Straits Times short | INCONCLUSIVE 87% | HIGH 80% |
| Straits Times report (mixed news material) | INCONCLUSIVE 8% | HIGH 74% |
| SPF footage of the 2026 deepfake Zoom call | INCONCLUSIVE 5% | LOW 14% |

**Honest reading:**
- The phone channel is much better than Run 5, where only 1 of 6 real deepfakes stayed clearly flagged. That run scored raw heads and this one uses the full verdict, so the two are not an exact comparison.
- The Zoom call is still a miss, and it is the most recent and most relevant attack. The audio has not been separated from SPF's captions and music.
- Real deepfakes helped steer this work (Run 5 led to adding Orpheus), so these 6 clips are no longer fully independent, although they were never trained on or used to set any weights.
- Only SpeechT5 is an unseen generator. Orpheus and Piper test clips use unseen voices from generators in training.
- On validation, the Orpheus voice held back (tara) had a 16.12% phone EER with `final_slow` alone, against 2.68% for the head alone on the test voices. Results depend heavily on the voice.
- False alarms on phone audio differ by speaker set: 3.3% for Singaporean voices and 13.0% for LibriSpeech. Clips were assumed equally likely real or fake, while real calls are mostly genuine.
- The window-spread limits are now tight because validation scores are so separated. From saved window scores, 0% to 7.5% of test clips exceed the limit depending on the condition, with compressed LibriSpeech the highest.
- The test clips were also used to report older versions, but no setting was chosen on them here.

## Runs 6 and 7: Orpheus fakes and training settings (31 Oct to 1 Nov 2026)

**Run 6:** Orpheus Singlish fakes showed the earlier heads had a blind spot on phone audio. At the 0.5 cut-off, fine-tuned 2 missed 64.1% of Orpheus fakes, the original model 15.5%, on 216 real and 206 Orpheus clips.

**Run 7:** head training on validation clips, phone EER. A head fitted to one generator got worse on others, and a lower learning rate only limited the damage. Training on both generators worked best: all generators, lr 3e-5, epoch 4 gave clean 0.00%, phone 7.17%, Orpheus 16.12%, Piper 1.62% (original: 3.78%, 23.54%, 19.20%, 25.69%). Label smoothing made little difference. The run holding out Piper at lr 3e-5 failed with an out-of-memory error, so it has no result.

## Run 5: real deepfakes of Singapore officials (30 Oct 2026)

**What was run:** `python -m eval.officials_report` on 6 publicly circulated deepfake clips (evaluation only; sources kept in the private project notes), each as downloaded and after a simulated GSM call. Three detector versions: original, fine-tuned 1 (audiobook data), fine-tuned 2 (plus Singaporean speech, Run 4's recipe on `data/train_manifest_sg.csv`; best validation epoch 4, phone EER 0.46%). Mean raw fake score over all 5 s windows (0 = real, 1 = fake):

| Clip | Length | As received: orig / ft1 / ft2 | Phone (GSM): orig / ft1 / ft2 |
|---|---|---|---|
| Lee Hsien Loong, fake CGTN crypto interview (his own Facebook post) | 14 s | 0.73 / 0.99 / **1.00** | 0.63 / 0.88 / **0.94** |
| Compilation of the Dec 2023 Lee and Wong deepfakes (with narration) | 177 s | 0.72 / 0.99 / **1.00** | 0.65 / 0.53 / 0.47 |
| Lawrence Wong deepfake, reposted Mar 2025 | 37 s | 0.67 / 0.95 / **0.94** | 0.35 / 0.06 / **0.00** |
| Lawrence Wong (DPM) deepfake, Straits Times short | 40 s | 0.45 / 0.56 / 0.81 | 0.36 / 0.31 / 0.26 |
| Straits Times report on the Lee deepfake (mixed news material) | 61 s | 0.22 / 0.30 / 0.27 | 0.56 / 0.29 / 0.22 |
| SPF footage of the 2026 deepfake Zoom call (Wong, Tharman, Indranee) | 66 s | 0.43 / 0.13 / 0.04 | 0.31 / 0.08 / 0.02 |

Most suspicious windows and every window's score: `results/sg_officials/report.csv` and `timeline.csv`.

**Honest reading:**
- **As received** (web audio, the voice-note/video case), the fine-tuned heads flag the four clips where the deepfake is most of the audio at 0.8 to 1.0, far more confidently than the original model (about 0.7). The two news-style clips are mixed, as expected: windows near 1.0 inside otherwise low clips (Straits Times report peaks at 0:50).
- **The SPF Zoom-call reel is mostly not flagged** by the fine-tuned heads (one window at 0:25 reaches 0.87 / 0.69). We have not listened through to separate the call audio from SPF's own captions/music, so this may be partly content, but it is the most recent and most relevant attack and should be treated as a likely miss until checked.
- **Through a phone line, the fine-tuned heads lose most of the real deepfakes.** Only the Lee CGTN clip stays clearly flagged. The Wong 2025 clip falls from 0.94 to 0.002 with fine-tuned 2, while the original model still gives 0.35. So the big phone gains in Run 4 (on Piper/SpeechT5 fakes) **do not carry over to real-world cloned voices**, and fine-tuning may have made the phone channel lean "real" for generators it never saw. This is the most important limitation so far.
- The GSM copies are double-compressed (web codec, then GSM), which is harsher than many real calls, and 6 clips are evidence of behaviour, not an accuracy figure.
- The app verdicts in the report still use the original model and its calibration (`calibration.json` has not been refitted for a fine-tuned head yet).
- These clips stay test-only: do not train on them or tune thresholds to them, or this real-world check stops being honest.

**What this means for next steps:** picking a head on validation alone (fine-tuned 2 wins there) would hide this. Training needs fakes closer to real cloning tools (newer, codec-based TTS, e.g. the Orpheus Singlish model, and Singapore-accented fakes) before the phone channel can be trusted; until then, phone-call verdicts should lean on the warning flow and the request itself ("officials never ask for transfers"), not on the voice score.

## Run 4: phone-domain fine-tuning, run 1 (30 Oct 2026)

**What was trained:** `python -m train.finetune_head --epochs 8`. Only the detector head (AASIST pooling + classifier, 487,655 parameters) was trained; the XLS-R backbone (300M) stayed frozen. Training data (`data/train_manifest.csv`): LibriSpeech **dev-clean** (32 speakers, none in the test set) and the same sentences read by **7 Piper voices not used in the test set**, each clip clean plus 2 phone copies (2,880 files). SpeechT5 was kept out of training entirely. Best epoch chosen on validation (8 other speakers, 2 other Piper voices): epoch 2.

Validation EER: clean 3.33% -> 0.00%, phone 27.08% -> 1.25%.

**Untouched test sets** (same clips and phone copies as Runs 1 and 3):

| Test set | Condition | Fakes | EER before | EER after | AUC before | AUC after |
|---|---|---|---|---|---|---|
| LibriSpeech | clean | all | 7.50% | 0.00% | 0.976 | 1.000 |
| LibriSpeech | clean | Piper (unseen voices) | 11.00% | 0.00% | 0.952 | 1.000 |
| LibriSpeech | phone | all | 16.25% | 5.75% | 0.915 | 0.986 |
| LibriSpeech | phone | Piper (unseen voices) | 25.25% | 3.75% | 0.848 | 0.994 |
| LibriSpeech | phone | **SpeechT5 (unseen generator)** | 8.00% | 7.25% | 0.982 | 0.978 |
| Singapore | clean | all | 15.83% | 0.00% | 0.921 | 1.000 |
| Singapore | phone | all | 16.58% | 2.42% | 0.896 | 0.998 |
| Singapore | phone | SpeechT5 (unseen generator) | 8.00% | 3.17% | 0.984 | 0.996 |

Real Singaporean voices now score very low (conversation, clean: mean raw score 0.004, was 0.192), so the Run 3 false-alarm problem is largely gone **even though no Singaporean speech was in training**.

**Honest reading:**
- The big gains are on Piper, a generator family the head trained on (different voices). On the generator it never saw (SpeechT5), phone EER barely moved on LibriSpeech (8.0% -> 7.3%) and improved on the Singapore set (8.0% -> 3.2%). So the head did not lose generality, but most of the gain is "seen family".
- Both test generators are older TTS designs. Real-world deepfakes (commercial voice cloning, codec-based TTS) are untested; the officials' clips (`eval/officials_report.py`) are the real-world check.
- Inconclusive phone clips rose from 3 to 27 because fine-tuned window scores are more extreme; the window-spread threshold needs re-tuning on validation.
- EER still uses the best threshold per test set; calibration (fitted on validation) is next.

## Run 3: real SINGAPOREAN voices, original model (30 Oct 2026)

**What was run:** `python -m eval.run_eval --manifest data/manifest_sg.csv --out results/eval_sg_base` (original model), then the Run 2 calibration applied to the real Singaporean clips.

**Test set:** 300 real Singaporean English clips from IMDA's National Speech Corpus (150 read, Part 1; 150 natural conversation with Singlish, Part 3), against the same 200 TTS fakes as before.

| Real Singaporean voices | Raw score mean | Calibrated HIGH (70%+ "synthetic") | MEDIUM |
|---|---|---|---|
| Read, clean | 0.093 | 2.7% | 7.3% |
| Read, phone | 0.232 | 8.0% | 16.0% |
| **Conversation, clean** | **0.192** | **54.7%** | 29.3% |
| Conversation, phone | 0.333 | 8.7% | 56.0% |

(For comparison, LibriSpeech audiobook voices: about 1% HIGH on clean audio.) Over 90% of these clips are detected as wideband, so the app would use the wideband calibration on them.

**What this shows:** the raw detector separates real from fake about as well as before (clean EER for SpeechT5 0%, Piper 28%), but **the calibration fitted on audiobook speech is badly wrong for natural Singaporean speech**: it would tell over half of real Singlish speakers that their voice is likely AI. This is exactly the harm the challenge warns about ("blocking legitimate transactions without review"), and the most important problem to fix.

**Plan:** add real Singaporean speech to fine-tuning and calibration, using different NSC parts (Part 2 read, Part 6 conversation) from the Part 1/3 test clips. **Caveat:** we have no Singapore-accented FAKES yet, so a model trained this way could learn "Singaporean accent = real" and miss Singapore-accented deepfakes. Singapore-accented fakes (gated SpeechT5-NSC models, or the 6.6 GB Orpheus Singlish model) are needed to check this.

## Run 2: calibration, watermark layer and the full verdict (30 Oct 2026)

### Calibration (`python -m eval.calibrate`)
Platt calibration, fitted separately for wideband and phone-band audio on half the real speakers and half the TTS voices, then tested on the other half (people and voices it never saw). Accuracy at the 50% cut-off is therefore an honest estimate.

| Audio | Scores | Accuracy at 50% | Fakes missed | Real voices flagged | ECE (calibration error) |
|---|---|---|---|---|---|
| Wideband | raw | 90.2% | 20.4% | 0.0% | 0.200 |
| Wideband | calibrated | 96.9% | 5.4% | 1.0% | 0.068 |
| Phone | raw | 88.1% | 18.3% | 6.0% | 0.158 |
| Phone | calibrated | 87.0% | 11.8% | 14.0% | 0.075 |

Test clips by band (calibrated):

| Audio | Clips | LOW (<30%) | MEDIUM | HIGH (70%+) |
|---|---|---|---|---|
| Wideband | real | 94% | 5% | 1% |
| Wideband | fake | 1% | 9% | 90% |
| Phone | real | 73% | 21% | 6% |
| Phone | fake | 8% | 12% | 81% |

Takeaways: calibration makes the percentages trustworthy (ECE more than halved) but cannot fix phone audio. On phone audio the 50% cut-off flags 14% of real voices, so only the HIGH band (6% of real voices, 81% of fakes) should trigger a pause; MEDIUM should only warn. Calibration assumes about half the clips are fake; in real calls most voices are genuine, so the app should combine this with context signals, not act on it alone. The held-out half here happened to be easier on wideband audio (AUC 0.998 vs 0.976 on all clips), so expect some variation.

Phone-band detection (used to pick the calibration): every phone copy had at most 0.008% of its energy above 3.8 kHz and every clean clip at least 0.049%; the cut-off is 0.02%.

### Watermark survival (`python -m eval.watermark_test --n 50`)
50 TTS clips watermarked with AudioSeal (as if the TTS provider cooperated), each tested clean and through every phone codec.

| Condition | Watermarked fakes detected | False alarms (real voices / unmarked fakes) |
|---|---|---|
| Clean | 100% | 0% / 0% |
| G.711 A-law | 30% | 0% / 0% |
| G.711 mu-law | 30% | 0% / 0% |
| GSM | 24% | 0% / 0% |
| AMR-NB (mobile) | 0% | 0% / 0% |
| Opus (WhatsApp/VoIP) | 0% | 0% / 0% |

Takeaways: a watermark is reliable proof on clean recordings (e.g. a voice note file) and never false-alarmed, but mobile and VoIP codecs erase it. On a phone call it is a rare bonus, not a safety net. Real-world coverage is even lower: (share of scam tools that watermark, close to zero for open-source cloning) x (survival through the call). The G.711/GSM clips averaged a score of about 0.4, just under the 0.5 threshold; a lower threshold might recover some, but would need its own false-alarm test.

### Full verdict (`python -m detectors.verdict`)
On sample clips, the flow behaved as designed: watermarked clip -> "AI-generated (watermark)"; SpeechT5 clean -> HIGH, high confidence; SpeechT5 through GSM -> HIGH (88%), medium confidence; real voice through GSM -> LOW (30%), medium confidence; Piper through GSM -> INCONCLUSIVE (48%, windows disagree).

**Known failure:** a clean Piper clip came out "Probably real, 9%, confidence high". The confidence rules (length, phone audio, window disagreement) cannot see a generator the model is blind to. This is the main reason the voice score must stay one signal among several.

## Run 1: forensics XLS-R baseline, clean vs phone (30 Oct 2026)

**What was run:** `python -m eval.run_eval` with the `detectors/forensics_xlsr.py` wrapper (whole clip, 5 s windows, 2.5 s hop, fp16 on an RTX 4050). Full outputs are in `results/eval/` (git-ignored; regenerate with the command).

**Test set (400 clips):**
- Real: 200 LibriSpeech test-clean clips, 5 per speaker, 40 speakers, 3 to 15 s.
- Fake: the same 200 sentences read by TTS. 100 Piper (VITS, 4 voices), 100 SpeechT5 + HiFi-GAN (7 voices).
- Phone copy of every clip (`audio/phone.py`, seeded per clip): white noise at SNR 5 to 30 dB, 8 kHz, 300 to 3400 Hz band-pass, one random codec (G.711 mu-law/A-law, GSM, AMR-NB, Opus), 0 to 5% packet loss.

**Results (raw scores, no calibration):**

| Condition | Fakes | EER | AUC | Fakes missed at 0.5 | Real voices flagged at 0.5 |
|---|---|---|---|---|---|
| Clean | all | 7.5% | 0.976 | 25.5% | 0.0% |
| Clean | Piper | 11.0% | 0.952 | 51.0% | 0.0% |
| Clean | SpeechT5 | 0.0% | 1.000 | 0.0% | 0.0% |
| Phone | all | 16.3% | 0.915 | 23.0% | 9.5% |
| Phone | Piper | 25.3% | 0.848 | 40.0% | 9.5% |
| Phone | SpeechT5 | 8.0% | 0.982 | 6.0% | 9.5% |

By phone codec (all fakes, about 40 real + 40 fake clips each, so these numbers are noisy):

| Codec | EER | Real voices flagged at 0.5 |
|---|---|---|
| AMR-NB | 9.9% | 7.5% |
| A-law | 15.1% | 3.1% |
| mu-law | 16.2% | 7.1% |
| GSM | 17.3% | 22.9% |
| Opus | 19.0% | 2.6% |

Inconclusive (under 2 s, or windows disagree): 5 clean, 3 phone.

**What this shows:**
1. **Phone audio roughly doubles the error** (EER 7.5% to 16.3%) and starts flagging real people as fake (0% to 9.5% at 0.5). This matches the research in `detector_research.md`.
2. **Results depend heavily on the generator.** SpeechT5 is caught perfectly on clean audio; Piper is much harder (EER 11% clean, 25% phone). Quoting one overall number would hide this.
3. **The raw score is not a probability.** Scores top out around 0.75 even for obvious fakes, and the best cut-off moves from about 0.11 (clean) to about 0.42 (phone). A fixed 0.5 threshold misses half the clean Piper fakes. Calibration on phone-quality data is needed before showing users a percentage.

**Limits:**
- EER uses the best threshold for this test set, so it is optimistic.
- Only two TTS generators, both older designs; no codec-based (newer) TTS yet.
- Real voices are read audiobook speech, not phone conversations. The detector may also have seen similar audiobook data in training (unknown).
- White noise and silent packet loss are simplifications of real phone lines.
- One phone copy per clip; the codec groups are small.
