"""E-wallet complaint analyzer (Gradio demo).

Tab 1 "Analyze a review": PRIMARY complaint verdict (IndoBERT + optional star rule), the model's
class probabilities, and for complaints with 3+ words an approximate topic with a reliability note.
Tab 2 "Findings": Phase 7 figures with one-line captions and limitations.

Run locally:
    uv run python app/app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from demo_core import Analyzer, load_config

CFG = load_config(Path(__file__).with_name("config.yaml"))
ANALYZER = Analyzer(CFG)
STAR_CHOICES = ["blank", "1", "2", "3", "4", "5"]
LABEL_NAMES = {"negative": "negative", "neutral": "neutral", "positive": "positive"}


def analyze(text: str, star: str) -> tuple[str, dict[str, float], str]:
    """Gradio callback: verdict markdown, class probabilities, topic markdown."""
    if not text or not text.strip():
        return "Enter a review to analyze.", {}, ""
    s = None if star in (None, "", "blank") else int(star)
    r = ANALYZER.analyze(text, s)
    reason = {"star+model": "the star rating is 1–2 and the model reads the text as negative",
              "star rule": f"the star rating is {s} (1–2★ counts as a complaint), although the model reads the text as {r.model_label}",
              "model": "the model reads the text as negative" + (f" (despite {s}★)" if s and s >= 4 else ""),
              "-": f"the model reads the text as {r.model_label}" + ("" if s is None else f" and the star rating is {s}")}[r.flagged_by]
    verdict = f"## {'Complaint' if r.is_complaint else 'Not a complaint'}\nBecause {reason}."
    if s is None:
        verdict += "\n\n*No star given: the verdict uses the text model only.*"
    if r.is_complaint:
        if r.topic_name:
            kw = f"  \nKeywords: {', '.join(r.keywords)}" if r.keywords else ""
            topic = f"### Topic: {r.topic_name}\n{r.topic_note}{kw}  \n*Similarity {r.topic_similarity:.2f}*"
        else:
            topic = f"### Topic: none\n{r.topic_note}"
    else:
        topic = "### Topic\nTopics are assigned to complaints only."
    return verdict, {LABEL_NAMES.get(k, k): v for k, v in r.probs.items()}, topic


def build() -> gr.Blocks:
    """The two-tab interface."""
    fig_dir = Path(CFG["_dir"]) / CFG["findings"]["figures_dir"]
    with gr.Blocks(title="E-wallet complaint analyzer") as demo:
        gr.Markdown("# E-wallet complaint analyzer\nIndonesian Google Play reviews of GoPay, OVO, DANA and ShopeePay "
                    "(Jul–Oct 2026). Research demo; results are approximate.")
        with gr.Tab("Analyze a review"):
            with gr.Row():
                with gr.Column(scale=3):
                    text = gr.Textbox(lines=4, label="Review text (Indonesian)", placeholder="Tulis ulasan di sini...")
                    star = gr.Radio(STAR_CHOICES, value="blank", label="Star rating (optional)")
                    btn = gr.Button("Analyze", variant="primary")
                with gr.Column(scale=2):
                    verdict = gr.Markdown()
                    probs = gr.Label(num_top_classes=3, label="Text model: class probabilities")
                    topic = gr.Markdown()
            btn.click(analyze, inputs=[text, star], outputs=[verdict, probs, topic])
            text.submit(analyze, inputs=[text, star], outputs=[verdict, probs, topic])
            gr.Examples(examples=CFG["examples"], inputs=[text, star], outputs=[verdict, probs, topic], fn=analyze,
                        run_on_click=True, label="Examples (written for the demo, not real reviews)")
            gr.Markdown("**How it works.** Complaint = the text model (IndoBERT, fine-tuned) predicts *negative*, "
                        "or a star rating of 1–2 is given. Topics come from BERTopic and are approximate.")
        with gr.Tab("Findings"):
            for name, caption in CFG["findings"]["items"]:
                gr.Markdown(f"**{caption}**")
                gr.Image(str(fig_dir / name), show_label=False, container=False, buttons=["fullscreen"])
            gr.Markdown("### Limitations\n" + CFG["findings"]["limitations"])
    return demo


if __name__ == "__main__":
    build().launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
                   server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")), share=False)
