# tune_alignment_thresholds.py
import json
from pathlib import Path
from collections import defaultdict
import itertools # For cartesian product of thresholds
import re # For normalization if needed again
import numpy as np # For averaging
import sys

# --- Configuration ---
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# You'll need to filter these for your 10-topic development set
ANNOTATIONS_DEV_FILE = PROCESSED_DATA_DIR / "annotations_merged_final.json" # Path to your dev topics' spans
# Choose ONE chunk manifest for tuning
CHUNK_MANIFEST_DEV_FILE = PROCESSED_DATA_DIR / "chunk_manifests" / "chunks_SENT.jsonl" # Example

# --- Helper: Load and prepare dev data (spans and chunks for specific topics) ---
# This part will require you to have a list of QIDs for your development set.
# For simplicity, this example assumes ANNOTATIONS_DEV_FILE and CHUNK_MANIFEST_DEV_FILE 
# *only* contain data for the development topics.
# In a real scenario, you'd filter them.

def load_dev_spans(file_path):
    spans_by_docid_qid = defaultdict(list)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for ann in data:
            if ann["start"] < ann["end"]: # Valid span
                spans_by_docid_qid[(ann["docid"], ann["qid"])].append(ann)
    return spans_by_docid_qid

def load_dev_chunks(file_path):
    chunks_by_docid = defaultdict(list)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            if chunk["start"] < chunk["end"]: # Valid chunk
                chunks_by_docid[chunk["original_doc_id"]].append(chunk)
    return chunks_by_docid

# --- Calculate Overlap and Lengths (from align_spans_to_chunks.py) ---
def calculate_overlap_and_lengths(
    span_start: int, span_end: int, 
    chunk_start: int, chunk_end: int
) -> tuple[float, float, float]:
    l_span = float(span_end - span_start)
    l_chunk = float(chunk_end - chunk_start)
    overlap_start = max(span_start, chunk_start)
    overlap_end = min(span_end, chunk_end)
    l_overlap = float(max(0, overlap_end - overlap_start))
    return l_span, l_chunk, l_overlap

# --- Main Grid Search Logic ---
def find_best_thresholds():
    print(f"Loading development spans from: {ANNOTATIONS_DEV_FILE}")
    dev_spans_map = load_dev_spans(ANNOTATIONS_DEV_FILE)
    print(f"Loading development chunks from: {CHUNK_MANIFEST_DEV_FILE}")
    dev_chunks_map = load_dev_chunks(CHUNK_MANIFEST_DEV_FILE)

    if not dev_spans_map or not dev_chunks_map:
        print("Error: Could not load development data. Exiting.", file=sys.stderr)
        return

    sme_threshold_grid = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    chunk_threshold_grid = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]

    best_f_cover_overall = -1.0
    best_thresholds = (None, None)
    best_avg_sme_cov = 0
    best_avg_chunk_cov = 0
    best_num_alignments = 0

    print(f"\nStarting grid search over {len(sme_threshold_grid) * len(chunk_threshold_grid)} threshold combinations...")

    for thresh_sme in sme_threshold_grid:
        for thresh_chunk in chunk_threshold_grid:
            
            current_alignments_sme_coverages = []
            current_alignments_chunk_coverages = []
            num_aligned_pairs = 0

            for (docid, qid), doc_qid_spans in dev_spans_map.items():
                if docid not in dev_chunks_map:
                    continue
                
                doc_strategy_chunks = dev_chunks_map[docid]

                for sme_span in doc_qid_spans:
                    s_start, s_end = sme_span["start"], sme_span["end"]
                    for chunk in doc_strategy_chunks:
                        c_start, c_end = chunk["start"], chunk["end"]

                        l_span, l_chunk, l_overlap = calculate_overlap_and_lengths(
                            s_start, s_end, c_start, c_end
                        )

                        if l_span == 0 or l_chunk == 0 or l_overlap == 0:
                            continue

                        coverage_sme_actual = l_overlap / l_span
                        coverage_chunk_actual = l_overlap / l_chunk

                        if coverage_sme_actual >= thresh_sme and coverage_chunk_actual >= thresh_chunk:
                            # This pair aligns with the current thresholds
                            num_aligned_pairs += 1
                            current_alignments_sme_coverages.append(coverage_sme_actual)
                            current_alignments_chunk_coverages.append(coverage_chunk_actual)
            
            if not current_alignments_sme_coverages: # No alignments for this threshold pair
                avg_sme_cov = 0
                avg_chunk_cov = 0
                f_cover_overall = 0.0
            else:
                avg_sme_cov = np.mean(current_alignments_sme_coverages)
                avg_chunk_cov = np.mean(current_alignments_chunk_coverages)
                
                if (avg_sme_cov + avg_chunk_cov) == 0:
                    f_cover_overall = 0.0
                else:
                    f_cover_overall = 2 * (avg_sme_cov * avg_chunk_cov) / (avg_sme_cov + avg_chunk_cov)

            print(f"  Thresh_SME={thresh_sme:.2f}, Thresh_Chunk={thresh_chunk:.2f} -> "
                  f"NumAlignments={num_aligned_pairs}, AvgSMEcov={avg_sme_cov:.4f}, "
                  f"AvgChunkCov={avg_chunk_cov:.4f}, F_cover_overall={f_cover_overall:.4f}")

            if f_cover_overall > best_f_cover_overall:
                best_f_cover_overall = f_cover_overall
                best_thresholds = (thresh_sme, thresh_chunk)
                best_avg_sme_cov = avg_sme_cov
                best_avg_chunk_cov = avg_chunk_cov
                best_num_alignments = num_aligned_pairs
            # Tie-breaking: if F_cover is similar, prefer more alignments or a better balance
            elif f_cover_overall == best_f_cover_overall:
                if num_aligned_pairs > best_num_alignments : # Prefer more alignments if F_cover is same
                     best_thresholds = (thresh_sme, thresh_chunk)
                     best_avg_sme_cov = avg_sme_cov
                     best_avg_chunk_cov = avg_chunk_cov
                     best_num_alignments = num_aligned_pairs


    print("\n--- Grid Search Finished ---")
    if best_thresholds[0] is not None:
        print(f"Best Thresholds Found:")
        print(f"  SME Coverage Threshold : {best_thresholds[0]:.2f}")
        print(f"  Chunk Coverage Threshold: {best_thresholds[1]:.2f}")
        print(f"  Resulting F_cover_overall: {best_f_cover_overall:.4f}")
        print(f"  With Avg Actual SME Coverage : {best_avg_sme_cov:.4f}")
        print(f"  With Avg Actual Chunk Coverage: {best_avg_chunk_cov:.4f}")
        print(f"  Number of Aligned Span-Chunk Pairs: {best_num_alignments}")
    else:
        print("No suitable thresholds found (no alignments made).")

if __name__ == "__main__":
    find_best_thresholds()