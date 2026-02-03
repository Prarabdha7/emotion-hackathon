# src/step4_evaluate_and_plots.py

import json
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
BASE_MODEL = "tokenizer_base"
DATA_PATH = "data/primary_emotions_filtered.csv"
LABEL_MAP_PATH = "artifacts/label_mapping.json"
MODEL_DIR = "model"

BATCH_SIZE = 16
MAX_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load data & label mapping
# -----------------------------
df = pd.read_csv(DATA_PATH)

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)

id2label = {int(k): v for k, v in mapping["id2label"].items()}
labels = sorted(id2label.keys())

# -----------------------------
# Dataset
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)

class EmotionDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

dataset = EmotionDataset(
    df["Poem"].astype(str).tolist(),
    df["label_id"].astype(int).tolist()
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE)

# -----------------------------
# Load model
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

# -----------------------------
# Run inference
# -----------------------------
all_preds = []
all_true = []

with torch.no_grad():
    for batch in tqdm(loader, desc="Running inference"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels_t = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_true.extend(labels_t.cpu().numpy())

# -----------------------------
# Classification report
# -----------------------------
print("\nClassification Report:")
print(
    classification_report(
        all_true,
        all_preds,
        target_names=[id2label[i] for i in labels],
        zero_division=0
    )
)

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(all_true, all_preds, labels=labels)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    xticklabels=[id2label[i] for i in labels],
    yticklabels=[id2label[i] for i in labels],
    cmap="Blues",
    fmt="d"
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix – Primary Emotion Classification")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# -----------------------------
# Emotion distribution plot
# -----------------------------
plt.figure(figsize=(10, 5))
df["Primary"].value_counts().plot(kind="bar")
plt.title("Emotion Distribution in Dataset")
plt.ylabel("Number of Samples")
plt.xlabel("Emotion")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()