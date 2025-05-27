# Synthetic Benchmark Design for SC-Qrels Evaluation

This document outlines a methodology for constructing a **realistic, end-to-end synthetic benchmark** to support SC-Qrels evaluation. The benchmark is designed to capture chunking variability, span alignment, logical span grouping (AND/OR), and information retrieval (IR) evaluation using retriever outputs.

## Goal: Realistic Synthetic Evaluation Setup

The objective is to simulate a complete **retrieval pipeline** with the following characteristics:

* **Documents** are sufficiently long to support **multiple chunking strategies**.
* **Queries** are grounded in document content.
* **Answer spans** are annotated at a fine granularity (i.e., not entire paragraphs).
* Some queries involve **logical span combinations** (AND/OR).
* Evaluation supports standard IR metrics (e.g., Precision\@k, MRR\@k, AGS\@k).
* The setup enables fair comparison of chunking methods and retriever outputs.

## Recommended Approach

### 1. Use a Known Corpus as the Document Base

Select a **public domain**, **coherent text corpus**, such as:

* *Alice in Wonderland*
* *Sherlock Holmes*
* Wikipedia articles

Recommended specifications:

* 10–20 documents
* Length of 500–2000 tokens per document
* Format: Markdown or plain text

These documents allow for realistic chunking while maintaining domain control.

### 2. Generate Synthetic Queries and Annotated Spans

Define queries and spans manually or through prompt-based methods. Each annotation should include:

| Field                | Description                                                          |
| -- | -- |
| `qid`                | Query text (e.g., "What did the Queen say to Alice?")                |
| `docid`              | Document identifier (e.g., `alice:ch04`)                             |
| `spans`              | Character offsets in the full document corresponding to answer spans |
| `group_id` / `logic` | Logical groupings for complex queries (e.g., AND/OR conditions)      |

Target: 30–50 queries covering:

* Factual retrieval (`COMPLETE_SPAN`)
* OR logic (e.g., "List Alice's friends")
* AND logic (e.g., "Explain cause and consequence of...")

This step ensures annotated ground truth data that mimics subject matter expert (SME) input.



### 3. Apply Multiple Chunking Strategies

Implement and store chunks with associated metadata for each document using the following strategies:

* `SENT`: Sentence-based chunking
* `FIX-200`: Fixed-length (e.g., 200 tokens) with overlap
* `RTS-512`: Recursive semantic chunking (optional)

Each chunk should include:

* Offsets
* `chunk_id`



### 4. Align Annotated Spans to Chunks

Utilize the SC-Qrels algorithm to produce:

* Per-query aligned chunks (`qid, chunk_id`)
* Derived qrels per chunking method

This alignment allows for consistent evaluation across different chunking strategies.



### 5. Simulate Retriever Behavior

Leverage an embedding-based retriever, such as:

* `all-MiniLM-L6-v2`: Suitable for fast prototyping
* `bge-small-en-v1.5`: Higher retrieval quality

Steps:

1. Embed all chunks
2. Embed queries
3. Compute similarity scores
4. Rank chunks
5. Output results as a retriever file



### 6. Evaluate with IR Metrics

Use the following tools for evaluation:

* **Standard Metrics (via `pytrec_eval`)**

  * Mean Average Precision (MAP)
  * Normalized Discounted Cumulative Gain (nDCG\@k)
  * Precision\@k
  * Mean Reciprocal Rank (MRR\@k)

* **Custom Metrics**

  * AGS\@k (AND Group Success at k)

Metrics should be used to compare:

* Different chunking strategies
* Different retriever backends



## Summary: Steps for Constructing the Synthetic Benchmark

| Step | Task                                                      |
| - |  |
| 1    | Select 10–20 long-form public domain documents            |
| 2    | Create 30–50 queries with manually annotated answer spans |
| 3    | Include logical grouping (AND/OR) in some queries         |
| 4    | Apply multiple chunking strategies                        |
| 5    | Align spans to chunks using SC-Qrels                      |
| 6    | Simulate retrieval using embedding models                 |
| 7    | Evaluate using IR and custom metrics such as AGS\@k       |



