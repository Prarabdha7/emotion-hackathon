# src/step2_label_encoding.py

import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = "data/primary_emotions_clean.csv"
LABEL_MAP_PATH = "artifacts/label_mapping.json"

# -----------------------------
# Load cleaned dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Unique emotion labels:", df["Primary"].nunique())

# -----------------------------
# Encode labels
# -----------------------------
label_encoder = LabelEncoder()
df["label_id"] = label_encoder.fit_transform(df["Primary"])

# -----------------------------
# Create mappings
# -----------------------------
label2id = {label: int(idx) for idx, label in enumerate(label_encoder.classes_)}
id2label = {int(idx): label for idx, label in enumerate(label_encoder.classes_)}

# -----------------------------
# Save mapping to JSON
# -----------------------------
mapping = {
    "label2id": label2id,
    "id2label": id2label
}

with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(mapping, f, indent=4, ensure_ascii=False)

# -----------------------------
# Save updated dataset (optional but useful)
# -----------------------------
df.to_csv(DATA_PATH, index=False)

# -----------------------------
# Sanity checks
# -----------------------------
print("\nSample label mapping (first 10):")
for i, (k, v) in enumerate(label2id.items()):
    if i == 10:
        break
    print(f"{k} → {v}")

print("\nSaved label mapping to:", LABEL_MAP_PATH)
print("Step 2 completed successfully.")