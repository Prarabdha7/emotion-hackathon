# src/step2b_remove_rare_labels.py

import pandas as pd

DATA_PATH = "data/primary_emotions_clean.csv"
OUTPUT_PATH = "data/primary_emotions_filtered.csv"

MIN_SAMPLES = 20  # threshold

df = pd.read_csv(DATA_PATH)

print("Original dataset size:", df.shape)
print("Original number of labels:", df["Primary"].nunique())

# Count samples per label
label_counts = df["Primary"].value_counts()

# Keep labels with enough samples
valid_labels = label_counts[label_counts >= MIN_SAMPLES].index

# Filter dataset
df_filtered = df[df["Primary"].isin(valid_labels)].reset_index(drop=True)

print("Filtered dataset size:", df_filtered.shape)
print("Remaining labels:", df_filtered["Primary"].nunique())

# Save new dataset
df_filtered.to_csv(OUTPUT_PATH, index=False)

print("\nSaved filtered dataset to:", OUTPUT_PATH)