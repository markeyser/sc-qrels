# sc_qrels/retrieve_topk_chunks.py

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from utils import get_torch_device

# Configuration
EMBED_PATH = Path("data/processed/chunk_embeddings.npz")
QUESTIONS_PATH = Path("data/processed/questions.json")
OUT_PATH = Path("data/processed/retrievals.jsonl")

MODEL_NAME = "BAAI/bge-large-en-v1.5"
TOP_K = 20

# Load device and model
device = get_torch_device()
print(f"🖥️  Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()

def mean_pooling(output, mask):
    token_emb = output[0]
    mask = mask.unsqueeze(-1).expand(token_emb.size()).float()
    return (token_emb * mask).sum(1) / mask.sum(1)

# Load chunk embeddings
chunk_data = np.load(EMBED_PATH)
chunk_ids = chunk_data["ids"]
chunk_docids = chunk_data["docids"]
chunk_vecs = chunk_data["embeddings"]  # shape: (num_chunks, 1024)
chunk_vecs = torch.tensor(chunk_vecs, dtype=torch.float32).to(device)

# Normalize chunk embeddings
chunk_vecs = torch.nn.functional.normalize(chunk_vecs, p=2, dim=1)

# Load questions
questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

# Output: one JSONL line per question
with OUT_PATH.open("w", encoding="utf-8") as fout:
    for q in tqdm(questions, desc="🔎 Retrieving"):
        qid, question = q["qid"], q["question"]

        # Tokenize and embed the question
        encoded = tokenizer(question, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            output = model(**encoded)
            q_vec = mean_pooling(output, encoded["attention_mask"])
            q_vec = torch.nn.functional.normalize(q_vec, p=2, dim=1)  # shape: (1, 1024)

        # Cosine similarity with all chunk vectors
        scores = torch.matmul(q_vec, chunk_vecs.T).squeeze(0)  # shape: (num_chunks,)
        topk = torch.topk(scores, k=TOP_K)

        for rank, (score, idx) in enumerate(zip(topk.values.tolist(), topk.indices.tolist())):
            record = {
                "qid": qid,
                "docid": chunk_docids[idx],
                "chunk_id": chunk_ids[idx],
                "rank": rank + 1,
                "score": float(score)
            }
            fout.write(json.dumps(record) + "\n")

print(f"\n✔ Top-{TOP_K} retrievals saved to: {OUT_PATH}")
