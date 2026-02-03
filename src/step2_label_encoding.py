# src/step2_label_encoding.py

import pandas as pd
import json
import re
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/primary_emotions_clean.csv"
LABEL_MAP_PATH = "artifacts/label_mapping.json"

df = pd.read_csv(DATA_PATH)

# -----------------------------
# Encode original labels (DO NOT MODIFY)
# -----------------------------
label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["Primary"])

original_labels = list(label_encoder.classes_)

# -----------------------------
# Create sanitized label names (for config/UI only)
# -----------------------------
def sanitize_label(label):
    label = label.lower()
    label = re.sub(r"[-\s]+", "_", label)
    label = re.sub(r"[^a-z0-9_]", "", label)
    return label

label2id = {}
id2label = {}
id2sanitized = {}

for idx, label in enumerate(original_labels):
    sanitized = sanitize_label(label)
    label2id[label] = idx
    id2label[str(idx)] = label           # JSON-safe
    id2sanitized[str(idx)] = sanitized   # clean version

# -----------------------------
# Save mapping
# -----------------------------
mapping = {
    "label2id": label2id,
    "id2label": id2label,
    "id2sanitized": id2sanitized
}

with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(mapping, f, indent=4, ensure_ascii=False)

# Save dataset
df.to_csv(DATA_PATH, index=False)

print("Total labels:", len(label2id))
print("Sample mapping:")
for i in range(5):
    print(f"{i}: {id2label[str(i)]} -> {id2sanitized[str(i)]}")

print("Step 2 completed cleanly.")