import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_DIR = BASE_DIR / "model"

BASE_MODEL = "distilbert-base-multilingual-cased"

# Load label mapping
with open(ARTIFACTS_DIR / "label_mapping.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

id2label = {int(v): k for k, v in label_map["label2id"].items()}

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# ---- SAMPLE INPUT ----
text = "I feel calm and hopeful after a long time."

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128
)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred_id = torch.argmax(probs, dim=1).item()

print("Input Text:")
print(text)
print("\nPredicted Emotion:", id2label[pred_id])
print("Confidence:", round(probs[0][pred_id].item(), 3))