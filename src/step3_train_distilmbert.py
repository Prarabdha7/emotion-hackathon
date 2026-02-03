# src/step3_train_distilmbert.py
# UPDATED: uses filtered dataset

import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from tqdm import tqdm

MODEL_NAME = "distilbert-base-multilingual-cased"
DATA_PATH = "data/primary_emotions_filtered.csv"
LABEL_MAP_PATH = "artifacts/label_mapping.json"
OUTPUT_DIR = "model"

BATCH_SIZE = 8
MAX_LEN = 128
EPOCHS = 3
LR = 2e-5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_PATH)

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)

num_labels = len(mapping["label2id"])

texts = df["Poem"].astype(str).tolist()
labels = df["label_id"].astype(int).tolist()

# -----------------------------
# Stratified split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    texts, labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# -----------------------------
# Dataset
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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

train_ds = EmotionDataset(X_train, y_train)
val_ds = EmotionDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# -----------------------------
# Model
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=num_labels
).to(DEVICE)

# -----------------------------
# Loss & optimizer
# -----------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

criterion = CrossEntropyLoss(weight=class_weights)
optimizer = AdamW(model.parameters(), lr=LR)

# -----------------------------
# Training
# -----------------------------
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels_t = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels_t)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Average loss: {total_loss / len(train_loader):.4f}")

# -----------------------------
# Validation
# -----------------------------
model.eval()
preds, true = [], []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels_t = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)

        preds.extend(predictions.cpu().numpy())
        true.extend(labels_t.cpu().numpy())

print("\nValidation Results:")
print(classification_report(true, preds, zero_division=0))

# -----------------------------
# Save model
# -----------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nModel saved to 'model/'")