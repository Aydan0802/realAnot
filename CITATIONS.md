# Citations

Every external model, dataset and library used in this project, with its licence. Add a row whenever something new is used.

## Models

| Name | Source | Licence | Used for |
|---|---|---|---|
| forensics_0.3B_xlsr_wild_deepfake_classifier | https://huggingface.co/eliya/forensics_0.3B_xlsr_wild_deepfake_classifier | CC-BY-NC-4.0 (non-commercial) | Main voice deepfake detector |
| wav2vec2-xls-r-300m (backbone config) | https://huggingface.co/facebook/wav2vec2-xls-r-300m | Apache-2.0 | Backbone architecture of the detector |

| AudioSeal (San Roman et al., 2024), detector_base + generator_base | https://github.com/facebookresearch/audioseal, https://huggingface.co/facebook/audioseal | MIT | Watermark layer (detect) and making watermarked demo clips (generate) |
| Piper TTS voices: lessac, amy, ryan (en_US), alan (en_GB) | https://huggingface.co/rhasspy/piper-voices | Per voice: lessac = Blizzard 2013 licence (non-commercial research); amy, alan = MycroftAI mimic3-voices (see repo); ryan = CC BY-NC-SA 4.0 | Making synthetic TEST clips |
| Piper TTS voices (training only): alba, jenny_dioco, northern_english_male (en_GB); hfc_female, hfc_male, joe, kristin, norman, ljspeech (en_US) | https://huggingface.co/rhasspy/piper-voices | alba CC BY 4.0; northern_english_male CC BY-SA 4.0; hfc_female/hfc_male CC BY-NC-SA 4.0; joe CC0; kristin, norman, ljspeech public domain; jenny_dioco see voice card | Making synthetic TRAINING/validation clips |
| Orpheus TTS 3B, Singlish fine-tune (Hyperbyte-TTS-Experimental), based on Canopy Labs Orpheus 3B via Unsloth | https://huggingface.co/Hyperbyte-TTS-Experimental/Orpheus_TTS_3B_unsloth_Singlish_fine_tuned | Apache-2.0 (card); the card does not say what speech it was fine-tuned on | Singapore-accented synthetic clips from a codec-based TTS: training (voices default/leo/leah/zac/jess), validation (tara), test (mia/dan/zoe) |
| SNAC 24 kHz neural audio codec (Siuzdak et al., 2024) | https://huggingface.co/hubertsiuzdak/snac_24khz | MIT (weights are `.bin`; loaded with `weights_only=True`, strict keys) | Turning Orpheus audio tokens into sound |
| EnCodec 24 kHz (Zeilinger et al., 2022) | https://huggingface.co/facebook/encodec_24khz | MIT upstream; the model card states no licence | Vocoded training fakes (`data_prep/prepare_clips.py`) |
| DAC 16 kHz (Kumar et al., 2023) | https://huggingface.co/descript/dac_16khz | MIT upstream; the model card states no licence | Vocoded training fakes (`data_prep/prepare_clips.py`) |
| UnivNet (Jang et al., 2021) | https://huggingface.co/dg845/univnet-dev | BSD-3-Clause | Vocoded training fakes (`data_prep/prepare_clips.py`) |
| Whisper base.en (Radford et al., 2022) | https://huggingface.co/openai/whisper-base.en | Apache-2.0 | Quality check only: transcribing Orpheus output to confirm it is intelligible |
| SpeechT5 TTS + HiFi-GAN vocoder | https://huggingface.co/microsoft/speecht5_tts, https://huggingface.co/microsoft/speecht5_hifigan | MIT (weights are `.bin`; loaded by transformers with `weights_only=True`) | Making synthetic test clips |

## Datasets

| Name | Source | Licence | Used for |
|---|---|---|---|
| LibriSpeech test-clean (Panayotov et al., 2015) | https://www.openslr.org/12 | CC BY 4.0 | Real voices: 200 clips, 5 per speaker, 40 speakers |
| IMDA National Speech Corpus (Koh et al., 2019), Part 1 and Part 3 test splits via MERaLiON Multitask National Speech Corpus v1 | https://www.imda.gov.sg/about-imda/emerging-technologies-and-research/artificial-intelligence/national-speech-corpus, https://huggingface.co/datasets/MERaLiON/Multitask-National-Speech-Corpus-v1 | Singapore Open Data Licence v1.0 (NSC); the MNSC card states no licence | Real Singaporean English voices: 150 read + 150 conversational test clips |
| LibriSpeech dev-clean | https://www.openslr.org/12 | CC BY 4.0 | Training/validation real voices (no speaker overlap with test-clean) |
| CMU Arctic x-vectors | https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors | MIT | Stock speaker voices for SpeechT5 |
| Publicly circulated deepfakes of Singapore officials (6 clips: SPF, Lee Hsien Loong's Facebook, The Straits Times and others) | one row per clip, with URL and date, kept in the private project notes | Copyright of the original posters; used for evaluation only, stored locally, never redistributed or trained on | Real-world deepfake test (`eval/officials_report.py`) |

## Libraries

| Name | Licence |
|---|---|
| PyTorch, torchaudio | BSD-3-Clause |
| Hugging Face transformers, huggingface_hub, safetensors | Apache-2.0 |
| librosa | ISC |
| soundfile | BSD-3-Clause |
| numpy, scipy, pandas, scikit-learn | BSD-3-Clause |
| matplotlib | Matplotlib licence (PSF-based) |
| pytest | MIT |
| piper-tts, onnxruntime | GPL-3.0 (piper-tts 1.8), MIT (onnxruntime) |
| sentencepiece | Apache-2.0 |
| gradio | Apache-2.0 |
| pyarrow | Apache-2.0 |
| imageio-ffmpeg | BSD-2-Clause; bundles an FFmpeg build (GPL, gyan.dev "essentials") used only as a separate program |
| yt-dlp (only used to download the officials' clips) | Unlicense |
| bitsandbytes (4-bit loading), accelerate | MIT, Apache-2.0 |
| snac | MIT |

## Research

See `docs/detector_research.md`.
