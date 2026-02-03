import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from navarasa_mapping import map_to_navarasa
import json

MODEL_DIR = "model"
TOKENIZER_DIR = "tokenizer_base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load label mapping
with open("artifacts/label_mapping.json", "r", encoding="utf-8") as f:
    id2label = {int(k): v for k, v in json.load(f)["id2label"].items()}

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = torch.argmax(logits, dim=1).item()
    emotion = id2label[pred_id]
    navarasa = map_to_navarasa(emotion)
    return emotion, navarasa

# ---------------- DEMO ----------------
text = "My heart feels heavy with sorrow and longing."
emotion, navarasa = predict_emotion(text)

print("Text:", text)
print("Predicted Emotion:", emotion)
print("Mapped Navarasa:", navarasa)