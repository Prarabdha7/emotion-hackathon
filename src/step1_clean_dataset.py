# src/step1_clean_dataset.py

import pandas as pd

# -----------------------------
# Load raw dataset
# -----------------------------
RAW_DATA_PATH = "data/Primary_Emotions_raw.csv"
CLEAN_DATA_PATH = "data/primary_emotions_clean.csv"

df = pd.read_csv(RAW_DATA_PATH)

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())

# -----------------------------
# Keep only required columns
# -----------------------------
# Expected columns:
# 'Poem'    -> input text
# 'Primary' -> target emotion
# 'Source'  -> optional (language / origin)

required_columns = ["Poem", "Primary", "Source"]
df = df[required_columns]

# -----------------------------
# Basic cleaning
# -----------------------------

# Drop rows with missing text or labels
df = df.dropna(subset=["Poem", "Primary"])

# Convert text to string (safety)
df["Poem"] = df["Poem"].astype(str)
df["Primary"] = df["Primary"].astype(str)

# Strip whitespace
df["Poem"] = df["Poem"].str.strip()
df["Primary"] = df["Primary"].str.strip()

# Remove empty poems
df = df[df["Poem"].str.len() > 5]

# -----------------------------
# Show class distribution (important sanity check)
# -----------------------------
print("\nTop 10 emotion labels:")
print(df["Primary"].value_counts().head(10))

print("\nTotal unique emotions:", df["Primary"].nunique())

# -----------------------------
# Save cleaned dataset
# -----------------------------
df.to_csv(CLEAN_DATA_PATH, index=False)

print("\nCleaned dataset saved to:", CLEAN_DATA_PATH)
print("Final shape:", df.shape)