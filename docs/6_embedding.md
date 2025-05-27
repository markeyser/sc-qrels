# 📚 Embedding Document Chunks with `bge-large-en-v1.5`

This document describes the process of embedding pre-chunked passages using the `bge-large-en-v1.5` model. It includes the rationale for selecting this model, full details on how the embedding is computed using the `sc_qrels/embed_chunks.py` script, and how compatibility is achieved across different devices (Apple MPS, CUDA, CPU).



## 🧠 Model Card: `bge-large-en-v1.5`

- **Model Name:** `BAAI/bge-large-en-v1.5`  
- **Publisher:** Beijing Academy of Artificial Intelligence (BAAI)  
- **License:** MIT  
- **Language:** English  
- **Model Type:** BERT-style encoder  
- **Embedding Dimension:** 1024  
- **Maximum Sequence Length:** 512 tokens  
- **Pooling Strategy:** Mean pooling over token embeddings  
- **Training Objective:** Contrastive learning for dense retrieval  
- **Model Size:** ~1.34 GB  
- **Model Card:** [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)

### 🔍 Performance (MTEB Benchmark)

- **Average Score (56 tasks):** 64.23  
- **Retrieval (15 tasks):** 54.29  
- **Clustering:** 46.08  
- **Pair Classification:** 87.12  
- **Reranking:** 60.03  
- **STS (Semantic Similarity):** 83.11  
- **Summarization:** 31.61  
- **Classification:** 75.97  

*Source: Hugging Face [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)*



## ✅ Why Use This Model for Alice in Wonderland?

- **Open Source & Accessible:** Freely downloadable and modifiable with full weight access for domain adaptation (e.g., LoRA).
- **Excellent Retrieval Quality:** Outperforms many dense encoders in zero-shot dense retrieval benchmarks.
- **Language & Style Coverage:** Handles narrative text well, making it well-suited to literary sources like *Alice in Wonderland*.
- **Embedding Consistency:** Reliable and smooth representations using mean pooling over contextualized token embeddings.



## 🧩 Chunk Embedding Process

The embeddings are created using the script:

```
sc_qrels/embed_chunks.py
```

This script computes high-quality vector representations for each chunk of the document corpus.



## ⚙️ How It Works

### 1. **Input Data Format**

- File: `data/processed/chunks.jsonl`
- Each JSON line contains:
  - `docid`: Source document ID
  - `chunk_id`: Unique chunk identifier
  - `chunk_index`: Sequential index in the doc
  - `text`: The actual chunk content

### 2. **Device Compatibility (`get_torch_device`)**

From your `utils.py`, the `get_torch_device()` function smartly selects:

- Apple Metal (`mps`) for macOS
- CUDA for Nvidia GPUs
- CPU if no acceleration available

Used as:

```python
DEVICE = get_torch_device()
```

This enables maximum portability across environments.



### 3. **Model and Tokenizer Initialization**

```python
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "BAAI/bge-large-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
```



### 4. **Chunk Loop and Embedding**

```python
for chunk in chunks:
    encoded = tokenizer(chunk["text"], truncation=True, padding=True, return_tensors="pt")
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        output = model(**encoded)
        pooled = mean_pooling(output, encoded["attention_mask"])
        embedding = torch.nn.functional.normalize(pooled, p=2, dim=1)
```



### 5. **Mean Pooling Function**

```python
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * mask).sum(1) / mask.sum(1)
```



### 6. **Output Storage**

Final vectors are saved to:

```bash
data/processed/chunk_embeddings.npz
```

**Contents:**

- `ids`: list of chunk IDs
- `docids`: corresponding `docid` per chunk
- `embeddings`: 2D float32 NumPy array of shape `(num_chunks, 1024)`

```python
np.savez_compressed(OUT_PATH, ids=ids, docids=docids, embeddings=np.stack(embeddings))
```



## 🧪 Example Usage

```python
import numpy as np

# Load embeddings
data = np.load("data/processed/chunk_embeddings.npz")
ids = data["ids"]
embeddings = data["embeddings"]

# Access the first chunk's embedding
first_chunk_embedding = embeddings[0]
```



## 🔄 Next Step

With all chunk embeddings ready:

1. **Embed queries using the same model**
2. **Compute cosine similarity with all chunks**
3. **Select top-k similar chunks**
4. **Evaluate using your SC-Qrels relevance annotations**



## 📁 File Summary

| File | Purpose |
|||
| `sc_qrels/embed_chunks.py` | Embeds all chunks and writes them to `.npz` |
| `data/processed/chunks.jsonl` | Source text chunks from Alice |
| `data/processed/chunk_embeddings.npz` | Final NumPy file with embeddings |
| `utils.py` | Supplies `get_torch_device()` to auto-select the hardware backend |



## 📎 References

- 🤗 [BAAI/bge-large-en-v1.5 on Hugging Face](https://huggingface.co/BAAI/bge-large-en-v1.5)
- 📊 [MTEB Benchmark Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- 🧪 [FlagEmbedding GitHub](https://github.com/FlagOpen/FlagEmbedding)
