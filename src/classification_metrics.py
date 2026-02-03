import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report

# =====================
# CONFIG
# =====================
DATA_PATH = "data/primary_emotions_filtered.csv"
MODEL_DIR = "model"
BASE_MODEL = "distilbert-base-multilingual-cased"
TEXT_COL = "Poem"
LABEL_COL = "label_id"
MAX_LEN = 128

# =====================
# LOAD DATA
# =====================
df = pd.read_csv(DATA_PATH)

texts = df[TEXT_COL].astype(str).tolist()
labels = df[LABEL_COL].tolist()

# =====================
# LOAD MODEL
# =====================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("cpu")
model.to(device)
model.eval()

# =====================
# TOKENIZE
# =====================
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt"
)

# =====================
# PREDICTION
# =====================
with torch.no_grad():
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )
    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

# =====================
# METRICS
# =====================
accuracy = accuracy_score(labels, preds)
macro_f1 = f1_score(labels, preds, average="macro")
weighted_f1 = f1_score(labels, preds, average="weighted")

print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Weighted F1-score: {weighted_f1:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(labels, preds))