import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModel

# =====================
# CONFIG
# =====================
DATA_PATH = "data/primary_emotions_filtered.csv"
MODEL_DIR = "model"
BASE_MODEL = "distilbert-base-multilingual-cased"
SAMPLE_SIZE = 500   # reduce if slow
MAX_LEN = 128

# =====================
# LOAD DATA
# =====================
df = pd.read_csv(DATA_PATH)

# ⚠️ CHANGE COLUMN NAME IF NEEDED
TEXT_COL = "Poem"
LABEL_COL = "Primary"

df = df[[TEXT_COL, LABEL_COL]].dropna()

# Sample for visualization clarity
df = df.sample(min(SAMPLE_SIZE, len(df)), random_state=42)

texts = df[TEXT_COL].astype(str).tolist()
labels = df[LABEL_COL].tolist()

# =====================
# LOAD MODEL
# =====================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModel.from_pretrained(BASE_MODEL)

device = torch.device("cpu")
model.to(device)
model.eval()

# =====================
# TOKENIZE
# =====================
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt"
)

# =====================
# EXTRACT EMBEDDINGS
# =====================
with torch.no_grad():
    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )
    # CLS token embedding
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

# =====================
# t-SNE REDUCTION
# =====================
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    init="random",
    learning_rate="auto"
)

reduced_embeddings = tsne.fit_transform(embeddings)

# =====================
# VISUALIZATION
# =====================
label_ids, label_names = pd.factorize(pd.Series(labels))

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    reduced_embeddings[:, 0],
    reduced_embeddings[:, 1],
    c=label_ids,
    cmap="tab20",
    alpha=0.75
)

plt.title("Embedding Space Visualization (t-SNE)", fontsize=14)
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

plt.colorbar(scatter, ticks=range(len(label_names)))
plt.clim(-0.5, len(label_names) - 0.5)

plt.tight_layout()
plt.savefig("artifacts/embedding_tsne.png", dpi=300)
plt.show()

print("✅ t-SNE embedding plot saved to artifacts/embedding_tsne.png")