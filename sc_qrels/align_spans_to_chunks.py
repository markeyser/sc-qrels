# sc_qrels/align_spans_to_chunks.py
import json
from pathlib import Path
from collections import defaultdict
import argparse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

ANNOTATIONS_MERGED_FILE = PROCESSED_DATA_DIR / "annotations_merged_final.json"
CHUNK_MANIFESTS_DIR = PROCESSED_DATA_DIR / "chunk_manifests"
QRELS_OUTPUT_DIR = PROCESSED_DATA_DIR / "qrels"
QRELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Alignment Thresholds (from SC-Qrels Section 3.4.2)
COVERAGE_SME_THRESHOLD = 0.85 # 0.80
COVERAGE_CHUNK_THRESHOLD = 0.50 # 0.25

# Default relevance grade for aligned chunks
RELEVANCE_GRADE = 1
TREC_ITERATION_COLUMN = "0" # Standard for qrels

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def calculate_overlap_and_lengths(
    span_start: int, span_end: int, 
    chunk_start: int, chunk_end: int
) -> tuple[float, float, float]:
    """Calculates lengths of span, chunk, and their overlap."""
    l_span = float(span_end - span_start)
    l_chunk = float(chunk_end - chunk_start)

    overlap_start = max(span_start, chunk_start)
    overlap_end = min(span_end, chunk_end)
    l_overlap = float(max(0, overlap_end - overlap_start))
    
    return l_span, l_chunk, l_overlap

# ---------------------------------------------------------------------------
# Main Alignment Logic
# ---------------------------------------------------------------------------
def align_spans_to_strategy(chunk_manifest_path: Path):
    strategy_name = chunk_manifest_path.stem.replace("chunks_", "")
    print(f"\n--- Aligning Spans to Chunks for Strategy: {strategy_name} ---")
    print(f"Using chunk manifest: {chunk_manifest_path}")
    print(f"Using merged annotations: {ANNOTATIONS_MERGED_FILE}")

    # 1. Load Adjudicated Spans
    try:
        with open(ANNOTATIONS_MERGED_FILE, "r", encoding="utf-8") as f:
            all_merged_annotations = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: Merged annotations file not found: {ANNOTATIONS_MERGED_FILE}", file=sys.stderr)
        return
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Could not decode JSON from {ANNOTATIONS_MERGED_FILE}: {e}", file=sys.stderr)
        return
    
    if not all_merged_annotations:
        print("ℹ️ No annotations found in the merged file. No qrels will be generated.", file=sys.stderr)
        return

    # Group spans by docid for efficient lookup
    spans_by_docid = defaultdict(list)
    for ann in all_merged_annotations:
        # Ensure spans are valid (start < end) before adding
        if ann.get("start") is not None and ann.get("end") is not None and ann["start"] < ann["end"]:
            spans_by_docid[ann["docid"]].append(ann)
        else:
            print(f"  ⚠️ Skipping invalid or zero-length span in merged annotations: QID={ann.get('qid')}, DOCID={ann.get('docid')}, Start={ann.get('start')}, End={ann.get('end')}", file=sys.stderr)


    # 2. Load Chunks for the given strategy
    chunks_for_strategy_by_docid = defaultdict(list)
    total_chunks_loaded = 0
    try:
        with open(chunk_manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                # Ensure chunks are valid (start < end)
                if chunk.get("start") is not None and chunk.get("end") is not None and chunk["start"] < chunk["end"]:
                    chunks_for_strategy_by_docid[chunk["original_doc_id"]].append(chunk)
                    total_chunks_loaded += 1
                else:
                     print(f"  ⚠️ Skipping invalid or zero-length chunk in {chunk_manifest_path.name}: CHUNK_ID={chunk.get('chunk_id')}, DOCID={chunk.get('original_doc_id')}, Start={chunk.get('start')}, End={chunk.get('end')}", file=sys.stderr)
    except FileNotFoundError:
        print(f"❌ ERROR: Chunk manifest file not found: {chunk_manifest_path}", file=sys.stderr)
        return
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Could not decode JSON from {chunk_manifest_path}: {e}", file=sys.stderr)
        return
        
    if total_chunks_loaded == 0:
        print(f"ℹ️ No valid chunks loaded from {chunk_manifest_path.name}. No qrels will be generated.", file=sys.stderr)
        return
    
    print(f"  Loaded {len(all_merged_annotations)} merged annotations.")
    print(f"  Loaded {total_chunks_loaded} chunks for strategy {strategy_name}.")

    # 3. Perform Alignment
    # Store relevant (qid, chunk_id) pairs to ensure uniqueness in qrels output
    relevant_qid_chunk_id_pairs = set()
    alignments_count = 0

    # Iterate through docids that have annotations
    for docid, doc_spans in spans_by_docid.items():
        if docid not in chunks_for_strategy_by_docid:
            # This means a document had SME annotations but no chunks were generated for it by this strategy
            # This is possible if, e.g., a document was empty after normalization or too short for the chunker
            # print(f"  ℹ️ No chunks found for document {docid} in strategy {strategy_name}, though it has {len(doc_spans)} SME spans.", file=sys.stderr)
            continue

        doc_chunks_for_strategy = chunks_for_strategy_by_docid[docid]

        for sme_span in doc_spans:
            s_qid = sme_span["qid"]
            s_start = sme_span["start"]
            s_end = sme_span["end"]

            for chunk in doc_chunks_for_strategy:
                c_id = chunk["chunk_id"]
                c_start = chunk["start"]
                c_end = chunk["end"]

                l_span, l_chunk, l_overlap = calculate_overlap_and_lengths(s_start, s_end, c_start, c_end)

                if l_span == 0: # Should have been filtered, but double check
                    # print(f"  ⚠️ Skipping SME span with zero length: QID={s_qid} Start={s_start} End={s_end}", file=sys.stderr)
                    continue
                if l_chunk == 0: # Should have been filtered
                    # print(f"  ⚠️ Skipping chunk with zero length: CHUNK_ID={c_id} Start={c_start} End={c_end}", file=sys.stderr)
                    continue
                
                if l_overlap == 0: # No overlap, no need to calculate coverage
                    continue

                coverage_sme = l_overlap / l_span
                coverage_chunk = l_overlap / l_chunk

                if coverage_sme >= COVERAGE_SME_THRESHOLD and coverage_chunk >= COVERAGE_CHUNK_THRESHOLD:
                    relevant_qid_chunk_id_pairs.add((s_qid, c_id))
                    alignments_count +=1 # Counts each successful span-to-chunk alignment event
    
    print(f"  Found {alignments_count} individual span-to-chunk alignments.")
    print(f"  Resulting in {len(relevant_qid_chunk_id_pairs)} unique (qid, chunk_id) relevant pairs.")

    # 4. Output Derived Qrels
    if relevant_qid_chunk_id_pairs:
        qrels_file_path = QRELS_OUTPUT_DIR / f"qrels_{strategy_name}.txt"
        with open(qrels_file_path, "w", encoding="utf-8") as f:
            # Sort for consistent output, though trec_eval doesn't require it
            for qid, chunk_id in sorted(list(relevant_qid_chunk_id_pairs)):
                f.write(f"{qid}\t{TREC_ITERATION_COLUMN}\t{chunk_id}\t{RELEVANCE_GRADE}\n")
        print(f"✔ Saved derived qrels for {strategy_name} to: {qrels_file_path}")
    else:
        print(f"ℹ️ No relevant (qid, chunk_id) pairs found for strategy {strategy_name}. No qrels file generated.")


def run_all_strategies():
    manifest_files = sorted(CHUNK_MANIFESTS_DIR.glob("*.jsonl"))
    if not manifest_files:
        print(f"No chunk manifest files found in {CHUNK_MANIFESTS_DIR} to process.", file=sys.stderr)
        return
    
    for manifest_file in manifest_files:
        align_spans_to_strategy(manifest_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align SME spans to chunk manifests to produce derived qrels.")
    parser.add_argument(
        "--chunk_manifest", 
        type=str, 
        help="Path to a specific chunk manifest file to process (e.g., data/processed/chunk_manifests/chunks_SENT.jsonl). If not provided, all manifests in the directory will be processed."
    )
    args = parser.parse_args()

    if args.chunk_manifest:
        manifest_path = Path(args.chunk_manifest)
        if manifest_path.exists():
            align_spans_to_strategy(manifest_path)
        else:
            print(f"❌ ERROR: Specified chunk manifest file not found: {manifest_path}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Processing all chunk manifests found in default directory...")
        run_all_strategies()
    
    print("\n--- Span-to-Chunk Alignment Finished ---")