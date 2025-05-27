# Generating Synthetic Questions and Spans (`generate_synthetic_queries.py`)

This document describes the QA generation and span annotation phase of the SC-Qrels pipeline. It documents how synthetic questions and their corresponding answer spans are created from the *Alice in Wonderland* corpus using OpenAI language models. The outputs feed directly into downstream chunking and evaluation steps.



## 1. Purpose

The goal of this step is to simulate SME-generated questions and exact span annotations, fully automatically, with a critical constraint:

> **All answer spans must be exact substrings of the document text.**

This guarantees compatibility with SC-Qrels evaluation, where SME annotations must be mapped precisely to retrievable text chunks.



## 2. Script Overview

`sc_qrels/generate_synthetic_queries.py` performs the following:

1. Loads each *Alice* chapter from `data/processed/documents/*.json`
2. Sends the chapter to GPT-4o to generate 6–10 diverse natural-language questions
3. For each question, sends a second LLM call to extract:

   * The exact spans answering the question
   * The **logical relationship** among those spans (COMPLETE\_SPAN, AND, or OR)
4. Verifies that each span exists in the normalized source document
5. Stores results in:

   * `data/processed/questions.json`
   * `data/processed/annotations.json`



## 3. Output Files

### `questions.json`

```json
{
  "qid": "q_8a7442fa",
  "question": "What did Alice find when she fell down the rabbit hole?",
  "docid": "alice:ch01"
}
```

* `qid`: Unique ID
* `question`: Natural-language question
* `docid`: Document context

### `annotations.json`

```json
{
  "qid": "q_8a7442fa",
  "docid": "alice:ch01",
  "start": 845,
  "end": 916,
  "text": "she looked at the sides of the well, and noticed that they were filled with cupboards and book-shelves",
  "logic": "COMPLETE_SPAN",
  "group": "g1"
}
```

* `qid`: Matches the question
* `docid`: Document the span was found in
* `start` / `end`: Character offsets (normalized text)
* `text`: The exact matched span
* `logic`:

  * `COMPLETE_SPAN`: One span suffices
  * `AND`: All listed spans are needed together
  * `OR`: Any listed span is independently sufficient
* `group`: Defaulted to "g1"



## 4. Prompt Strategy

### A. Question Generation Prompt

* Instructs LLM to generate 6–10 diverse questions
* Ensures JSON object with `questions` key
* Includes 2 few-shot examples

### B. Answer + Logic Extraction Prompt

* Given a document and question, returns:

```json
{
  "answers": ["exact span 1", "exact span 2"],
  "logic": "AND"
}
```

* Defines:

  * `COMPLETE_SPAN` (one sufficient span)
  * `AND` (all required)
  * `OR` (any one suffices)
* Few-shot examples reinforce structure and logic



## 5. Temperature Strategy

* `TEMP_QUESTION = 0.7`: Encourages variation in question phrasing
* `TEMP_SPAN_AND_LOGIC = 0.0`: Forces deterministic extraction of spans and logic



## 6. Span Matching Logic (`locate_span()`)

To validate each span:

1. Normalize both document and extracted answer:

   * Standardize quotes and dashes
   * Collapse whitespace
   * Strip punctuation
2. Try multiple string variants

   * With/without quotes
   * Underscore adjustments
3. Use `.find()` (case-insensitive) to locate offset
4. Snap to word boundaries if enabled
5. Return `start`, `end`, and extracted slice



## 7. Validation with `sanity_check.py`

The script `sc_qrels/sanity_check.py` performs strict validation:

* Recomputes `text == norm[start:end].strip()`
* Validates that every question in `questions.json` has at least one matching span in `annotations.json`

Run:

```bash
poetry run python sc_qrels/sanity_check.py
```

Output:

```
✔ All sanity checks passed. Every question has at least one valid span.
```



## 8. Output Distribution Summary (`analyze_output_distribution.py`)

This script produces a Markdown report of:

* Questions and span counts per chapter
* Logic type breakdown per question
* Totals for:

  * Questions
  * Spans
  * `COMPLETE_SPAN`, `AND`, `OR`

Run:

```bash
poetry run python sc_qrels/analyze_output_distribution.py
```

It saves to:

```
data/processed/output_distribution.md
```

Example excerpt:

```markdown
### Spans by logic type
- **COMPLETE_SPAN**: 93
- **AND**: 132
- **OR**: 4

> Assertion: Every question here has at least one span.
```



## 9. Summary

This synthetic QA generator combines:

* Diverse question phrasing (creative LLM call)
* Accurate span alignment with offset preservation
* Logical structure encoding for complex answers

The result is a fully usable QA+annotation dataset for SC-Qrels, RAG reranking, span-level supervision, and other QA tasks requiring both semantic and character-level precision.
