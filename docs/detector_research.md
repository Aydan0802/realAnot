# Voice deepfake detector research (30 Oct 2026)

All accuracy numbers are the authors' own claims unless marked "verified". Nothing here has been run yet except where stated.

## 1. What we verified in the forensics model we downloaded

Files in `models/forensics_xlsr/` (git-ignored): `checkpoint_epoch_4.safetensors` (1.2 GB), `model.py`, `inference.py`, `config.json`. We skipped the 3.8 GB pickled `.pt`.

| Check | Result |
|---|---|
| Backbone | XLS-R 300M, all 24 hidden layers stacked and fed to AASIST graph pooling (matches the model card) |
| Weights file | 589 tensors: `wavlm.*` 422, `pool.*` 141, `classifier.*` 22, **`projection.*` 4** |
| `projection.*` | Not defined in `model.py`, so `strict=False` silently drops them. Probably a training-only head. Our wrapper must list them as "expected unexpected" and fail on any other mismatch. |
| Extra download | `model.py` calls `Wav2Vec2Model.from_pretrained(...)`, which downloads another ~1.2 GB copy of XLS-R that the checkpoint then overwrites. The checkpoint already holds all backbone weights, so we can build from the config only (a few KB). |
| Default verdict | `inference.py` labels FAKE at 0.5, but the model card says 0.1 to 0.2 is usually best. A fixed threshold is not trustworthy; calibrate on our own phone-quality data. |
| Clip handling | Uses only the centre 5 s of a clip and ignores the rest. Short clips are repeat-padded. |
| Output | `sigmoid(logit) = P(real)`, so `fake_probability = 1 - sigmoid(logit)` |

## 2. What the newer research says

1. **Real-world drop is big.** Deepfake-Eval-2024 (deepfakes actually circulated in 2024, 52 languages) found open-source audio detectors lose about **48% AUC** compared with older benchmarks. Commercial and fine-tuned models do better but still fall short of human analysts. Lesson: never present lab EERs as real-world accuracy. ([arXiv 2503.02857](https://arxiv.org/abs/2503.02857))
2. **The channel matters more than the model size.** "On Deepfake Voice Detection: It's All in the Presentation" (2025) says detectors fail mainly because training data is raw audio, not audio "presented" through a channel such as a phone call. Simulating that channel improved accuracy by 39% (lab) and 57% (real-world benchmark), and the authors say better data beats bigger models. ([arXiv 2509.26471](https://arxiv.org/abs/2509.26471))
3. **Phone codecs and packet loss break detectors.** ADD-C (EUSIPCO 2025) simulates VoLTE/VoIP codecs with packet loss; three baseline detectors degraded significantly, and targeted augmentation helped. ([arXiv 2504.12423](https://arxiv.org/abs/2504.12423)) Odyssey 2026 work found RawBoost plus random quantization augmentation could beat training on real telephony data (20.1% relative EER reduction, source-tracing task). ([ISCA](https://www.isca-archive.org/odyssey_2026/klein26_odyssey.html))
4. **Modern TTS is codec-based, and vocoder-trained detectors miss it.** Newer voice cloning uses neural audio codecs plus language models. Codecfake (2024) reports a 41% EER reduction when training on codec fakes rather than vocoder fakes; CodecFake+ (2025) extends this. Our test set should include at least one codec-based open TTS as its own source. ([arXiv 2406.08112](https://arxiv.org/abs/2406.08112), [arXiv 2501.08238](https://arxiv.org/abs/2501.08238))
5. **XLS-R is still the standard backbone.** Most strong 2024 to 2026 systems keep XLS-R and change the back end: SLS (attention over all 24 layers), XLSR-Mamba, Nes2Net (lighter back end, best reported on ASVspoof 2021 DF and In-the-Wild), BreathNet (2026, adds breath cues). These all need full training, and Intel Labs found the head matters less than data and augmentation. ([XLSR-Mamba](https://arxiv.org/abs/2411.10027), [Nes2Net](https://arxiv.org/abs/2504.05657), [BreathNet](https://arxiv.org/abs/2602.13596))
6. **Fusing several SSL-based detectors helps out-of-domain.** Layer selection plus fusion of SSL countermeasures improved out-of-domain results (2025). ([arXiv 2509.12003](https://arxiv.org/abs/2509.12003))

## 3. Candidate second models

| Model | Backbone | Self-reported EER | Licence | Notes |
|---|---|---|---|---|
| DF Arena 500M (`SpeechAntiSpoofingBenchmarks/DF_Arena_500M_V_1`) | XLS-R 300M + layer attention + Conformer head ("RAPTOR"), 436M params | In-the-Wild 1.87%, ASVspoof5 13.43%, mean 5.09% over 24 sets | "other" (terms not stated, **check before use**) | Needs `trust_remote_code=True`, so read its code before running. 4.04 s input, tile-repeat, no random crop. |
| DF Arena 1B | larger | mean 3.66% over 24 sets | check | Probably too heavy for a 6 GB GPU; not checked. |
| XLS-R-SLS community reproduction (`sukhdeveyash/XLS-R-SLS-Deepfake-Detection`) | XLS-R 300M + SLS | not checked | check | Author recommends the earlier checkpoint because the later one overfits. |
| Saracasm wav2vec2 / garystafford | wav2vec2-base | weak out of domain | tested earlier; not used | Deliberately weak baseline only. |

Same-backbone models may make similar mistakes, so fusion gains need to be measured, not assumed.

## 4. Singapore: deepfakes of government officials (researched 30 Oct 2026)

| When | What happened | Source |
|---|---|---|
| Dec 2023 | Deepfake of then-PM Lee Hsien Loong in a fake CGTN interview promoting a crypto "investment platform"; a similar deepfake of Lawrence Wong (then DPM). Both used cloned voices on altered real footage. | [Malay Mail](https://www.malaymail.com/news/singapore/2023/12/29/singapore-pm-lee-urges-public-not-to-respond-to-deepfake-ai-videos-featuring-him-or-lawrence-wong-promoting-crypto/109709), [CSA advisory AD-2024-006](https://www.csa.gov.sg/alerts-and-advisories/advisories/ad-2024-006/), [AI Incident Database #985](https://incidentdatabase.ai/cite/985/) |
| Jun 2024 | Deepfake built from Lee's 2023 National Day Message: mouth movements re-synced to cloned audio, promoting an investment product. | [Mothership](https://mothership.sg/2024/06/lee-hsien-loong-deepfake-scam/) |
| Mar 2025 | PM Lawrence Wong warns of deepfakes using his image to sell crypto, get-rich schemes and PR services. | [Connected to India](https://www.connectedtoindia.com/deepfakes-in-singapore-pm-wong-warns-against-digital-scams-using-his-image-to-sell-cryptocurrency-or-get-rich-quick-schemes/) |
| May 2026 | Business professional loses about S$4.9 million after a WhatsApp approach from a fake "Secretary to the Cabinet" and a deepfake Zoom call showing President Tharman, PM Wong, Minister Indranee Rajah, MAS staff and foreign officials. Police exposed it by lip movements not matching speech, all speech coming from one account, and a distorted background. | [SPF release, 16 May 2026](https://www.police.gov.sg/Media-Hub/News/2026/05/20260516_footage_from_zoom_video_conference_involving_impersonation_of_senior_government_officials), [Malay Mail](https://malaymail.com/news/singapore/2026/05/17/singapore-police-detail-how-ai-deepfake-zoom-call-mimicking-pm-lawrence-wong-was-exposed-video/220263) |

Patterns that matter for our design:
- The scams pair a **cloned voice with real footage** of a trusted official, and the money request comes after the call (a fake "lawyer" in the 2026 case). Detecting the voice alone is not enough; the **request** is the giveaway.
- The official rule to build into the app's advice: government officials will never ask for money transfers or bank log-ins over a call, will not ask you to install apps from unofficial stores, and will not transfer your call to the police. Verify via the ScamShield Helpline 1799.
- A third-party site reports at least 98 government-official impersonation cases since Feb 2026 with losses over S$8.5 million; **not verified** against an official source, so do not quote it without checking SPF figures.

## 5. Singaporean and Asian-accented speech for testing

| Resource | What it is | Access | Use |
|---|---|---|---|
| IMDA National Speech Corpus, via `MERaLiON/Multitask-National-Speech-Corpus-v1` | Singaporean English: read speech (Part 1), conversation with Singlish (Part 3), more parts | Open on Hugging Face; underlying NSC under the Singapore Open Data Licence (the MNSC card itself states no licence) | **Using**: 150 read + 150 conversational real clips (`data_prep/prepare_sg.py`) |
| RADAR Challenge 2026 | Deepfake benchmark with a Singapore English condition, under compression/noise/reverb | No public download found | Cite only |
| SEA-Spoof (2025) | Real + spoof speech in Tamil, Hindi, Thai, Indonesian, Malay, Vietnamese (not English) | Gated (manual approval) | Possible later, needs your own HF account |
| IndieFake (2025) | Indian-accented English real + deepfake | Not checked | Possible later |
| SpeechT5 fine-tuned on NSC (`JET2001/speecht5_tts_imda_nsc_*`) | Singapore-accented TTS | Gated | Would give Singapore-accented fakes; needs your own HF account |
| Orpheus 3B fine-tuned on Singlish (`Hyperbyte-TTS-Experimental/Orpheus_TTS_3B_unsloth_Singlish_fine_tuned`) | Codec-based (newer type) Singlish TTS, Apache-2.0, 6.6 GB | Open | **Using** (`data_prep/make_orpheus.py`): loaded in 4-bit (2.7 GB GPU), about 30 tokens/s on an RTX 4050. Whisper transcripts of probe clips were intelligible Singlish. The base Orpheus voice names (leah, mia, leo, zac, ...) still change the voice, but less reliably after the fine-tune |

None of the Piper voices has a Southeast Asian accent (`reza_ibrahim` is Persian-accented English).

## 6. Sources
- Chandra et al., Deepfake-Eval-2024, [arXiv 2503.02857](https://arxiv.org/abs/2503.02857)
- On Deepfake Voice Detection: It's All in the Presentation, [arXiv 2509.26471](https://arxiv.org/abs/2509.26471)
- Shi et al., Benchmarking ADD robustness in real-world communication scenarios (ADD-C), [arXiv 2504.12423](https://arxiv.org/abs/2504.12423)
- Klein et al., Odyssey 2026, [ISCA archive](https://www.isca-archive.org/odyssey_2026/klein26_odyssey.html)
- Codecfake [arXiv 2406.08112](https://arxiv.org/abs/2406.08112); CodecFake+ [arXiv 2501.08238](https://arxiv.org/abs/2501.08238)
- XLSR-Mamba [arXiv 2411.10027](https://arxiv.org/abs/2411.10027); Nes2Net [arXiv 2504.05657](https://arxiv.org/abs/2504.05657); BreathNet [arXiv 2602.13596](https://arxiv.org/abs/2602.13596)
- Layer selection and fusion of SSL countermeasures, [arXiv 2509.12003](https://arxiv.org/abs/2509.12003)
- DF Arena 500M model card, [Hugging Face](https://huggingface.co/SpeechAntiSpoofingBenchmarks/DF_Arena_500M_V_1); Speech Anti-Spoofing Arena leaderboard (Hugging Face Space `SpeechAntiSpoofingBenchmarks/SpeechAntiSpoofingArena`)
