"""
demo.py - a simple local page: upload a voice clip, see the verdict.

Upload only (no microphone), so the demo never records anyone's voice.
Nothing is saved. Runs only on this computer (127.0.0.1).

Run from the project root, then open http://127.0.0.1:7860
    python -m app.demo
"""
import gradio as gr

from detectors.forensics_xlsr import load_audio
from detectors.verdict import analyse

BAND_COLOURS = {"LOW": "#2a9d8f", "MEDIUM": "#e9a23b", "HIGH": "#d1495b", "INCONCLUSIVE": "#6c757d"}


def check(path):
    if path is None:
        return "Upload an audio clip first."
    r = analyse(load_audio(path))
    colour = BAND_COLOURS.get(r["band"], "#333")
    lines = [f"<h2 style='color:{colour};margin:0'>{r['verdict']}</h2>", ""]
    if r["likelihood_pct"] is not None:
        lines += [f"**Chance the voice is synthetic:** {r['likelihood_pct']}%  ",
                  f"**Band:** {r['band']}  ",
                  f"**Confidence in this result:** {r['confidence']}  ",
                  f"**Audio type detected:** {r['channel']}", ""]
    lines += ["**Why:**"] + [f"- {reason}" for reason in r["reasons"]]
    lines += ["", "_This is one warning signal, not proof of a scam. If in doubt, hang up and "
                  "call the bank on the number printed on your card._"]
    return "\n".join(lines)


demo = gr.Interface(
    fn=check,
    inputs=gr.Audio(type="filepath", sources=["upload"], label="Voice clip (wav, mp3, flac)"),
    outputs=gr.Markdown(),
    title="Is this voice AI-generated?",
    description="Checks for an AI watermark, then runs a deepfake voice detector and gives a "
                "calibrated chance, a confidence level and the reasons.",
    flagging_mode="never",
    analytics_enabled=False,
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
