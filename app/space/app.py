"""Gradio demo for the Hugging Face Space: cyberbullying type classifier (fine-tuned DistilBERT)."""

import html
import re

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    import spaces  # available on Hugging Face ZeroGPU Spaces

    gpu = spaces.GPU(duration=15)  # ask for a GPU for at most 15 s per call
except ImportError:  # running locally without the package

    def gpu(fn):
        return fn


# CHANGE THIS to your Hugging Face username and model repo name
MODEL_REPO = "arnav-dev-01/cyberbullying-distilbert"
MAX_LEN = 96  # same as training

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO).eval().to(device)
id2label = {int(k): v for k, v in model.config.id2label.items()}


def clean_text(t: str) -> str:
    """Must match the training notebook's cleaning exactly."""
    t = html.unescape(str(t)).lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"^rt\s+", "", t.strip())
    t = t.replace("#", "")
    return re.sub(r"\s+", " ", t).strip()


@gpu
def classify(text):
    cleaned = clean_text(text or "")
    if not cleaned:
        raise gr.Error("Please enter some text (links and @mentions are ignored).")
    enc = tokenizer(
        cleaned, truncation=True, max_length=MAX_LEN, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0].cpu().tolist()
    return {id2label[i].replace("_", " "): p for i, p in enumerate(probs)}


with gr.Blocks(title="SafeText: Cyberbullying Type Detector") as demo:
    gr.Markdown(
        "# SafeText: Cyberbullying Type Detector\n"
        "NLP (Natural Language Processing) mini project, UPES Dehradun. A fine-tuned DistilBERT "
        "classifies a message into 6 categories: age, ethnicity, gender, religion, other cyberbullying, "
        "or not cyberbullying.\n\n"
        "**Educational demo only.** The model makes mistakes, especially between *other cyberbullying* "
        "and *not cyberbullying*. Do not use it for real moderation decisions."
    )
    box = gr.Textbox(
        label="Message", lines=4, placeholder="Type or paste a tweet-style message..."
    )
    btn = gr.Button("Classify", variant="primary")
    out = gr.Label(num_top_classes=6, label="Prediction")
    gr.Examples(
        [
            "Women belong in the kitchen, not in engineering.",
            "Had a great day at the park with friends!",
            "You are so ugly and dumb, nobody likes you.",
        ],
        inputs=box,
    )
    btn.click(classify, box, out)
    box.submit(classify, box, out)

if __name__ == "__main__":
    demo.launch()
