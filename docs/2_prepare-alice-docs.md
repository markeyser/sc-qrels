# Preparing Long-Form Synthetic Documents from *Alice’s Adventures in Wonderland*

This document describes the preprocessing steps that transform the full text of *Alice’s Adventures in Wonderland* into a structured and normalized corpus of individual documents suitable for synthetic evaluation in retrieval-augmented generation (RAG) pipelines. The resulting corpus is stored in JSON format and used in subsequent steps including synthetic query creation, span-level annotation, chunking, and evaluation following the SC-Qrels framework.



## 1. Rationale

### 1.1 Choice of Corpus

*Alices’s Adventures in Wonderland* was selected as the base corpus for the following reasons:

- It is public domain and readily available in clean text format.
- Its content is familiar and memorable to most readers, making it highly suitable for illustrative examples in technical papers.
- The narrative contains a diverse set of scenes, characters, and topics that allow for natural query generation and span annotation.
- Each chapter is sufficiently long to simulate realistic document chunking, including overlapping, sentence-based, or recursive segmentation strategies.

### 1.2 Mapping Chapters to Documents

For SC-Qrels, each **chapter** is treated as a **document**. This aligns with the framework’s design principles (Section 3.2.1 of the SC-Qrels paper), where document-level IDs (`docid`) provide the context for chunking and SME relevance annotation.



## 2. Script Overview

The script `sc_qrels/prepare_alice_docs.py` performs the following steps:

1. **Load the full raw text** from `data/raw/alice_in_wonderland.txt`.
2. **Normalize and clean** the text to ensure deterministic character offsets and eliminate typographic artifacts.
3. **Extract the main body** of the book by removing Project Gutenberg’s licensing header and footer.
4. **Split the body into chapters** using regular expressions that match standard chapter headings (e.g., `CHAPTER I. Down the Rabbit-Hole`).
5. **Generate a JSON file per chapter**, assigning each a unique `docid` and storing its full text content.

Each chapter is saved under:

```
data/processed/documents/alice:chXX.json
```

Where `XX` is the zero-padded chapter number (e.g., `alice:ch01`, `alice:ch02`, ...).



## 3. Unicode Normalization and Text Cleaning

### 3.1 Motivation

The SC-Qrels framework relies on **precise character-level alignment** between the source documents, SME span annotations, and the chunks derived from those documents. This necessitates a normalized and encoding-consistent source text.

The following issues were identified in the raw *Alice* text from Project Gutenberg:
- Smart quotes (e.g., `“`, `’`) encoded as Unicode characters
- Em dashes, ellipses, and non-breaking spaces
- Zero-width spaces or BOMs
- Variable line endings (`\r`, `\r\n`)

These characters introduce subtle offset drift and inconsistencies across platforms or tools, particularly in span-based annotation tasks.

### 3.2 Audit Tool: `scan_unicode_characters.py`

The script `sc_qrels/scan_unicode_characters.py` scans the corpus and reports all non-ASCII characters, providing:
- Unicode code point
- Character
- Official Unicode name
- Up to 3 usage examples in local context

This audit serves as a justification for all cleaning decisions.

### 3.3 Cleaning Strategy (`normalize()`)

The following replacements are applied before splitting the chapters:

| Character                | Unicode             | Replacement | Rationale                             |
|--------------------------|---------------------|-------------|----------------------------------------|
| `“ ”`                    | U+201C / U+201D     | `"`         | Standard double quote                  |
| `‘ ’`                    | U+2018 / U+2019     | `'`         | Standard single quote                  |
| `—`                      | U+2014              | `--`        | ASCII-safe em dash                     |
| `…`                      | U+2026              | `...`       | Ellipsis                               |
| `•`                      | U+2022              | `*`         | Bullet from license footer             |
| `™`                      | U+2122              | removed     | Trademark metadata                     |
| `\ufeff`                 | U+FEFF              | removed     | Byte-order mark / zero-width space     |
| `\u00A0`                 | NBSP                | space       | Layout artifact                        |
| Tabs or multiple spaces  | various             | collapsed   | Normalize spacing                      |


The text is also normalized to Unicode NFC and line endings are converted to `\n`.



## 4. Output Format

Each chapter is saved in JSON format using the following structure:

```json
{
  "docid": "alice:ch01",
  "title": "CHAPTER I. Down the Rabbit-Hole",
  "text": "Alice was beginning to get very tired of sitting by her sister on the bank, ..."
}
```

- `docid`: Unique identifier required for span annotation, chunking, and evaluation.
- `title`: Human-readable chapter title (optional metadata).
- `text`: Full normalized chapter content.



## 5. Terminology Mapping

| Field    | Used in Paper | Required in Framework | Purpose                           | Comment                                      |
|----------|----------------|------------------------|-----------------------------------|----------------------------------------------|
| `docid`  | ✅ Yes         | ✅ Yes                 | Unique ID for document (chapter) | Used in all annotations and alignment        |
| `text`   | ✅ Yes         | ✅ Yes                 | Full normalized document text     | Character offsets are computed from this     |
| `title`  | ❌ No          | ❌ No                  | Human-readable chapter label      | Included for readability only — not used in SC-Qrels logic |

## 6. Alignment with SC-Qrels Principles

These normalization and splitting steps ensure:
- Accurate offset tracking between annotated spans and chunk boundaries
- Clean, consistent text inputs for retrieval and alignment evaluation
- Transparency in the cleaning process, backed by auditable logs
- Compatibility with labeling tools (e.g., Label Studio)

This preprocessing phase ensures the corpus is reliable for downstream tasks such as span-level annotation, span-to-chunk alignment, and retriever evaluation.



## 7. Next Step

Once the chapter-level documents are saved, the next step is to generate synthetic queries and annotate their corresponding answer spans within each document. These annotations will later be aligned to chunks during evaluation using the SC-Qrels span-to-chunk alignment algorithm.
