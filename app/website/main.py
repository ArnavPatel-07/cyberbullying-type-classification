"""FastAPI demo: cyberbullying type classifier (fine-tuned DistilBERT)."""

import html
import re
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"  # put config.json, model.safetensors, tokenizer files here
MAX_LEN = 96  # same as training

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
id2label = {int(k): v for k, v in model.config.id2label.items()}


def clean_text(t: str) -> str:
    """Must match the training notebook's cleaning exactly."""
    t = html.unescape(str(t)).lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"^rt\s+", "", t.strip())
    t = t.replace("#", "")
    return re.sub(r"\s+", " ", t).strip()


class Item(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


app = FastAPI(title="Cyberbullying Type Classifier")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(item: Item):
    cleaned = clean_text(item.text)
    if not cleaned:
        return {"error": "Text is empty after cleaning (only links/mentions?)."}
    enc = tokenizer(cleaned, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0].tolist()
    scores = sorted(
        ({"label": id2label[i], "score": round(p, 4)} for i, p in enumerate(probs)),
        key=lambda d: -d["score"],
    )
    return {"cleaned_text": cleaned, "prediction": scores[0]["label"], "scores": scores}


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
