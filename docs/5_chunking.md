# Chunking Strategy for RAG Evaluation using SC-Qrels

This document describes the token-based chunking process implemented in `sc_qrels/chunk_documents.py` to prepare documents for retrieval-augmented generation (RAG) evaluation using the SC-Qrels framework. The chunking logic, tokenizer, embedding model constraints, and output format are described in full detail.



## 1. Rationale for Token-Based Chunking

In real-world RAG systems, large documents must be split into manageable **chunks** that can be embedded independently and retrieved efficiently at inference time. Chunking enables:

- Efficient retrieval over long documents
- Sentence or paragraph-level grounding
- Compatibility with transformer-based embedding models

The **SC-Qrels** evaluation framework assumes that chunking is performed **before** retrieval and that each chunk can be traced back to its original document (`docid`) for alignment with SME-provided span annotations.



## 2. Chosen Chunking Strategy: Sliding Window Over Tokens

We use a **fixed-size sliding window over tokenized text**, a common and effective baseline in RAG literature.

### ✅ Settings

| Parameter     | Value                         | Justification                                     |
|---------------|-------------------------------|---------------------------------------------------|
| `chunk_size`  | 512 tokens                    | Max supported by BERT-style models like BGE-large |
| `stride`      | 128 tokens                    | ~75% overlap ensures minimal semantic loss        |
| `tokenizer`   | BERT-style (`bge-large-en-v1.5`) | Matches embedding model spec                   |


This technique yields overlapping token sequences such as:

```
Chunk 1: tokens 0–511  
Chunk 2: tokens 128–639  
Chunk 3: tokens 256–767  
...
```

This overlap helps preserve context across boundaries and mitigates edge effects in retrieval performance.



## 3. Embedding Model Constraints

The chunking strategy must respect the tokenization and architecture of the **retriever model**.

### Selected Model: `BAAI/bge-large-en-v1.5`

- Architecture: BERT-style encoder
- Context limit: **512 tokens**
- Pooling: [CLS] embedding
- Training objective: contrastive learning for retrieval

### Implications

- **Token limit** defines `chunk_size = 512`
- **BERT tokenizer** must be used for compatibility
- Detokenized output is aligned to span-annotated text for SC-Qrels
- Large model with ~1.2B params, but supports offline usage and LoRA finetuning



## 4. Tokenizer Details

We use the tokenizer bundled with the model:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5", use_fast=True)
```

This guarantees:
- Consistent token boundaries
- Identical behavior to that used in model pretraining
- Clean detokenization via `tokenizer.decode(...)`



## 5. Implementation Logic

The script `sc_qrels/chunk_documents.py` performs:

1. **Load input documents** from `data/processed/documents/*.json`  
   Each file must have:
   ```json
   {
     "docid": "alice:ch01",
     "title": "...",
     "text": "full text of the chapter..."
   }
   ```

2. **Tokenize the full document text** using the model’s tokenizer

3. **Generate overlapping chunks** using a sliding window over tokens

4. **Detokenize each chunk** to recover readable text

5. **Save output to `data/processed/chunks.jsonl`** as JSON lines:

```json
{
  "docid": "alice:ch03",
  "chunk_id": "alice:ch03::chunk02",
  "chunk_index": 2,
  "text": "She was looking about for some way of escape, and wondering whether she could get away without being seen..."
}
```



## 6. Output Summary

- ✅ `chunks.jsonl` contains **249 token-based chunks** derived from all 12 chapters of *Alice in Wonderland*
- Each chunk:
  - Respects the 512-token limit of the embedding model
  - Can be embedded independently
  - Can be aligned later to SME span annotations
- No vector DB is required — the chunks can be scored using **in-memory cosine similarity**



## 7. Alignment with SC-Qrels

This chunking stage satisfies the SC-Qrels requirement that:
- All annotated spans (ground truth answers) must be traceable to the text of a specific chunk
- Chunking must be **agnostic** to annotation span locations
- Ground truth mapping from spans to chunks will be handled later by the **span-to-chunk alignment algorithm**



## 8. Next Steps

Once the `chunks.jsonl` file is ready, the next step is:

➡️ **Embed all chunks using `bge-large-en-v1.5` and store their vector representations**

This will support:
- Cosine similarity-based retrieval at inference time
- Ground truth alignment using SC-Qrels for IR evaluation
- End-to-end RAG pipeline simulation on a synthetic benchmark
