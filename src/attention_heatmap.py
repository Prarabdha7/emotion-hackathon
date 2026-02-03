# src/attention_heatmap.py
# Real transformer attention heatmap for PPT

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "distilbert-base-multilingual-cased"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Example sentence (you can change this)
TEXT = "The Lord is my shepherd; He makes me lie down in green pastures."

# -----------------------------
# Load tokenizer & model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    output_attentions=True
).to(DEVICE)

model.eval()

# -----------------------------
# Tokenize input
# -----------------------------
inputs = tokenizer(TEXT, return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

# -----------------------------
# Forward pass
# -----------------------------
with torch.no_grad():
    outputs = model(**inputs)

# outputs.attentions -> tuple(num_layers)
# each layer shape: (batch_size, num_heads, seq_len, seq_len)
attentions = outputs.attentions

# -----------------------------
# Average attention across layers & heads
# -----------------------------
attn_stack = torch.stack(attentions)          # (layers, batch, heads, seq, seq)
attn_mean = attn_stack.mean(dim=(0, 2))[0]    # (seq_len, seq_len)

attn = attn_mean.cpu().numpy()

# -----------------------------
# Convert token ids to tokens
# -----------------------------
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# -----------------------------
# Plot heatmap
# -----------------------------
plt.figure(figsize=(12, 6))
sns.heatmap(
    attn,
    xticklabels=tokens,
    yticklabels=tokens,
    cmap="YlOrRd"
)

plt.title("Transformer Attention Heatmap (DistilBERT)")
plt.xlabel("Attended Tokens")
plt.ylabel("Query Tokens")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()