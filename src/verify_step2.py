# src/verify_step2.py

import json
import pandas as pd

# Load files
with open("artifacts/label_mapping.json", "r", encoding="utf-8") as f:
    mapping = json.load(f)

df = pd.read_csv("data/primary_emotions_clean.csv")

label2id = mapping["label2id"]
id2label = {int(k): v for k, v in mapping["id2label"].items()}
id2sanitized = {int(k): v for k, v in mapping["id2sanitized"].items()}

# Pick a random row
row = df.sample(1).iloc[0]

primary_label = row["Primary"]
label_id = row["label_id"]

print("Primary label from CSV:", primary_label)
print("Label ID from CSV:", label_id)
print("Decoded label:", id2label[label_id])
print("Sanitized label:", id2sanitized[label_id])

assert primary_label == id2label[label_id], "❌ Label mismatch!"

print("\n✅ STEP 2 VERIFIED SUCCESSFULLY")