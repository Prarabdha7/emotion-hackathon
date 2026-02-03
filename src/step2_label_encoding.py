# src/step2_label_encoding.py
# UPDATED: works on filtered dataset

import pandas as pd
import json
import re
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/primary_emotions_filtered.csv"
LABEL_MAP_PATH = "artifacts/label_mapping.json"

df = pd.read_csv(DATA_PATH)

print("Filtered dataset shape:", df.shape)
print("Unique labels:", df["Primary"].nunique())

# -----------------------------
# Encode labels
# -----------------------------
le = LabelEncoder()
df["label_id"] = le.fit_transform(df["Primary"])

original_labels = list(le.classes_)

# -----------------------------
# Sanitize labels (for UI/config only)
# -----------------------------
def sanitize(label):
    label = label.lower()
    label = re.sub(r"[-\\s]+", "_", label)
    label = re.sub(r"[^a-z0-9_]", "", label)
    return label

label2id = {}
id2label = {}
id2sanitized = {}

for idx, label in enumerate(original_labels):
    label2id[label] = idx
    id2label[str(idx)] = label
    id2sanitized[str(idx)] = sanitize(label)

mapping = {
    "label2id": label2id,
    "id2label": id2label,
    "id2sanitized": id2sanitized
}

with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(mapping, f, indent=4, ensure_ascii=False)

df.to_csv(DATA_PATH, index=False)

print("\nLabel encoding complete.")
print("Saved mapping to:", LABEL_MAP_PATH)
print("Total labels:", len(label2id))