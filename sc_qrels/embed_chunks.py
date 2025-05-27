# sc_qrels/embed_chunks.py

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Import your portable device selection logic
from utils import get_torch_device

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

MODEL_NAME = "BAAI/bge-large-en-v1.5"
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
OUT_PATH = Path("data/processed/chunk_embeddings.npz")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

DEVICE = get_torch_device()
print(f"🖥️  Using device: {DEVICE}")

# ------------------------------------------------------------------
# Load tokenizer and model
# ------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval().to(DEVICE)

# ------------------------------------------------------------------
# Pooling: Use mean pooling (alternative to CLS pooling)
# ------------------------------------------------------------------

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # (batch_size, seq_len, hidden)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled = (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)
    return pooled

# ------------------------------------------------------------------
# Load chunks
# ------------------------------------------------------------------

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

# ------------------------------------------------------------------
# Encode each chunk
# ------------------------------------------------------------------

ids, docids, embeddings = [], [], []

for chunk in tqdm(chunks, desc="📡 Embedding chunks"):
    text = chunk["text"]
    encoded = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        output = model(**encoded)
        emb = mean_pooling(output, encoded["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        embeddings.append(emb.cpu().numpy()[0])

    ids.append(chunk["chunk_id"])
    docids.append(chunk["docid"])

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

np.savez_compressed(OUT_PATH, ids=ids, docids=docids, embeddings=np.stack(embeddings))
print(f"✔ Saved {len(embeddings)} embeddings to {OUT_PATH}")
