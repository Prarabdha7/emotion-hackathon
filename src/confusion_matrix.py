import pandas as pd
import torch
import json
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix

# ---------------- CONFIG ----------------
DATA_PATH = "data/primary_emotions_clean.csv"
MODEL_DIR = "model"
ARTIFACTS_DIR = "artifacts"

TEXT_COL = "Poem"
LABEL_COL = "label_id"
BASE_MODEL = "distilbert-base-multilingual-cased"

# ---------------- LOAD LABEL MAP ----------------
with open(f"{ARTIFACTS_DIR}/label_mapping.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

id2label = {int(v): k for k, v in label_map["label2id"].items()}

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)
df = df[[TEXT_COL, LABEL_COL]].dropna()

texts = df[TEXT_COL].astype(str).tolist()
true_labels = df[LABEL_COL].astype(int).tolist()

# ---------------- LOAD MODEL ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

# ---------------- PREDICTIONS ----------------
pred_labels = []

with torch.no_grad():
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(device)

        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        pred_labels.append(pred)

# ---------------- CONFUSION MATRIX ----------------
cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    xticklabels=[id2label[i] for i in range(len(id2label))],
    yticklabels=[id2label[i] for i in range(len(id2label))],
    cmap="Blues",
    fmt="d"
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – Emotion Classification")
plt.tight_layout()

plt.savefig(f"{ARTIFACTS_DIR}/confusion_matrix.png", dpi=300)
plt.show()

print("✅ Confusion matrix saved to artifacts/confusion_matrix.png")