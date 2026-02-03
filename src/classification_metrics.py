import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

# ---------------- PATHS ----------------
DATA_PATH = "data/primary_emotions_clean.csv"
MODEL_DIR = "model"

TEXT_COL = "Poem"
LABEL_COL = "label_id"

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)
df = df[[TEXT_COL, LABEL_COL]].dropna()

texts = df[TEXT_COL].astype(str).tolist()
labels = df[LABEL_COL].astype(int).tolist()

# ---------------- LOAD MODEL ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

# ---------------- INFERENCE ----------------
preds = []

with torch.no_grad():
    for text in tqdm(texts, desc="Running inference"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)

        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        preds.append(pred)

# ---------------- METRICS ----------------
accuracy = accuracy_score(labels, preds)
macro_f1 = f1_score(labels, preds, average="macro")
weighted_f1 = f1_score(labels, preds, average="weighted")

print("\n=== Classification Metrics ===")
print(f"Accuracy      : {accuracy:.4f}")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Weighted F1   : {weighted_f1:.4f}")

print("\n=== Detailed Classification Report ===")
print(classification_report(labels, preds))