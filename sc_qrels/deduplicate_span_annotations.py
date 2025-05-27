# deduplicate_span_annotations.py
import json
from pathlib import Path
from collections import defaultdict
import re
from typing import List, Dict, Tuple, Optional, Any
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Assuming this script is in the root of your project, adjust if it's in sc_qrels/
BASE_DIR = Path(__file__).resolve().parent.parent 
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
CHAPTER_DIR_FOR_DEDUP = PROCESSED_DATA_DIR / "documents" 

SME1_ANNOTATIONS_FILE = PROCESSED_DATA_DIR / "annotations_sme1_openai.json"
SME2_ANNOTATIONS_FILE = PROCESSED_DATA_DIR / "annotations_sme2_gemini.json"
MERGED_OUTPUT_FILE = PROCESSED_DATA_DIR / "annotations_merged_final.json"
CONFLICT_LOG_FILE = PROCESSED_DATA_DIR / "deduplication_conflicts.log"

IOU_THRESHOLD = 0.5

# Cache for normalized document text
normalized_doc_cache_dedup: Dict[str, str] = {}

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def normalize_text_for_dedup(text: str) -> str:
    """Applies the same normalization as used in generate_synthetic_queries.py"""
    text = text.replace('’', "'").replace('‘', "'")
    text = text.replace('”', '"').replace('“', '"')
    text = text.replace('—', '-').replace('–', '-')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_normalized_doc_text_for_dedup(docid: str) -> Optional[str]:
    """Loads and normalizes document text, using a cache."""
    if docid in normalized_doc_cache_dedup:
        return normalized_doc_cache_dedup[docid]

    doc_path = CHAPTER_DIR_FOR_DEDUP / f"{docid}.json"
    if not doc_path.exists():
        print(f"⚠️ Document file not found for deduplication: {doc_path}", file=sys.stderr)
        return None
    try:
        with open(doc_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        original_text = content.get("text")
        if not original_text:
            print(f"⚠️ Document {docid} has no text or text is empty.", file=sys.stderr)
            return None
        normalized_text = normalize_text_for_dedup(original_text)
        normalized_doc_cache_dedup[docid] = normalized_text
        return normalized_text
    except Exception as e:
        print(f"⚠️ Error loading or normalizing document {docid} for deduplication: {e}", file=sys.stderr)
        return None

def calculate_iou(span1_start: int, span1_end: int, span2_start: int, span2_end: int) -> float:
    overlap_start = max(span1_start, span2_start)
    overlap_end = min(span1_end, span2_end)

    overlap_length = float(max(0, overlap_end - overlap_start))
    if overlap_length == 0:
        return 0.0

    # Union length: (max_end - min_start)
    # This is not (len1 + len2 - overlap) because that double counts overlap if we just sum lengths.
    # The correct union for ranges is (max_end - min_start)
    union_length = float(max(span1_end, span2_end) - min(span1_start, span2_start))
    
    if union_length == 0: 
        # This implies both spans are identical zero-length points that somehow had overlap > 0,
        # which shouldn't happen if overlap_length > 0.
        # If they are identical points, IoU is 1.0. If one is zero-length, IoU should be 0 if no overlap.
        return 1.0 if overlap_length > 0 else 0.0

    return overlap_length / union_length

def log_conflict(conflict_logger, active_span: Dict, current_span: Dict, iou: float):
    conflict_logger.write(f"CONFLICT DETECTED (IoU >= {IOU_THRESHOLD} but attributes differ):\n")
    conflict_logger.write(f"  Active Span: QID={active_span['qid']}, DOCID={active_span['docid']}, SME={active_span.get('sme_id', 'N/A')}, "
                          f"Offsets=[{active_span['start']}-{active_span['end']}], "
                          f"Logic={active_span['logic']}, Group={active_span.get('group', 'N/A')}, "
                          f"Text='{active_span['text'][:50]}...'\n")
    conflict_logger.write(f"  Current Span: QID={current_span['qid']}, DOCID={current_span['docid']}, SME={current_span.get('sme_id', 'N/A')}, "
                          f"Offsets=[{current_span['start']}-{current_span['end']}], "
                          f"Logic={current_span['logic']}, Group={current_span.get('group', 'N/A')}, "
                          f"Text='{current_span['text'][:50]}...'\n")
    conflict_logger.write(f"  IoU: {iou:.4f}\n\n")

# ---------------------------------------------------------------------------
# Main Deduplication Logic
# ---------------------------------------------------------------------------
def deduplicate_annotations():
    print("--- Starting Span Deduplication and Merging ---")
    all_spans_from_smes: List[Dict[str, Any]] = [] # Explicit typing

    # Load SME1 annotations
    sme1_loaded_count = 0
    if SME1_ANNOTATIONS_FILE.exists():
        try:
            with open(SME1_ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
                sme1_data = json.load(f)
                for ann in sme1_data:
                    ann["sme_id"] = ann.get("sme_id", "SME1_OpenAI_Synth") # Ensure sme_id for tracking
                    all_spans_from_smes.append(ann)
                sme1_loaded_count = len(sme1_data)
            print(f"✔ Loaded {sme1_loaded_count} annotations from SME1 ({SME1_ANNOTATIONS_FILE})")
        except Exception as e:
            print(f"❌ Error loading SME1 annotations from {SME1_ANNOTATIONS_FILE}: {e}", file=sys.stderr)
    else:
        print(f"⚠️ SME1 annotation file not found: {SME1_ANNOTATIONS_FILE}", file=sys.stderr)

    # Load SME2 annotations
    sme2_loaded_count = 0
    if SME2_ANNOTATIONS_FILE.exists():
        try:
            with open(SME2_ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
                sme2_data = json.load(f)
                for ann in sme2_data:
                    ann["sme_id"] = ann.get("sme_id", "SME2_Gemini_Synth") # Ensure sme_id
                    all_spans_from_smes.append(ann)
                sme2_loaded_count = len(sme2_data)
            print(f"✔ Loaded {sme2_loaded_count} annotations from SME2 ({SME2_ANNOTATIONS_FILE})")
        except Exception as e:
            print(f"❌ Error loading SME2 annotations from {SME2_ANNOTATIONS_FILE}: {e}", file=sys.stderr)
    else:
        print(f"⚠️ SME2 annotation file not found: {SME2_ANNOTATIONS_FILE}", file=sys.stderr)


    if not all_spans_from_smes:
        print("No annotations loaded from any SME. Exiting deduplication.", file=sys.stderr)
        return

    print(f"Total annotations loaded from all SMEs: {len(all_spans_from_smes)}")

    # 1. Sorting by Paragraph (docid), Question (qid), and then Span Offsets
    # We sort by qid first, then docid within qid, then start, then end.
    # This ensures all spans for the same question and document are processed together.
    all_spans_from_smes.sort(key=lambda x: (x["qid"], x["docid"], x["start"], x["end"]))

    final_adjudicated_spans: List[Dict[str, Any]] = []
    
    # Open conflict log
    with open(CONFLICT_LOG_FILE, "w", encoding="utf-8") as conflict_logger:
        conflict_logger.write("--- Deduplication Conflict Log ---\n\n")

        # Group by (qid, docid) to process related spans together
        from itertools import groupby
        for key, group_iter in groupby(all_spans_from_smes, key=lambda x: (x["qid"], x["docid"])):
            qid, docid = key
            sorted_spans_for_qd_pair = list(group_iter) # Convert iterator to list
            
            if not sorted_spans_for_qd_pair:
                continue

            # Fetch normalized document text once per (qid, docid) group
            normalized_doc_text = get_normalized_doc_text_for_dedup(docid)
            if not normalized_doc_text:
                print(f"  ⚠️ Skipping group (qid={qid}, docid={docid}) due to missing/unreadable document text. Spans preserved as is.", file=sys.stderr)
                final_adjudicated_spans.extend(sorted_spans_for_qd_pair) # Preserve if doc unreadable
                continue

            merged_spans_for_this_qd_pair: List[Dict[str, Any]] = []
            
            # Iterative Merge Evaluation
            active_span = dict(sorted_spans_for_qd_pair[0]) # Start with the first span as active (make a copy)

            for i in range(1, len(sorted_spans_for_qd_pair)):
                current_span = sorted_spans_for_qd_pair[i]

                # Ignore zero-length or invalid spans
                if active_span["start"] >= active_span["end"]:
                    print(f"  ℹ️ Active span for (qid={qid}, docid={docid}) became zero-length or invalid. Replacing with current.", file=sys.stderr)
                    active_span = dict(current_span) # Current becomes active
                    if active_span["start"] >= active_span["end"]: # If new active is also bad, skip to next iteration
                        continue
                
                if current_span["start"] >= current_span["end"]:
                    print(f"  ℹ️ Current span for (qid={qid}, docid={docid}) is zero-length or invalid. Skipping merge attempt with it.", file=sys.stderr)
                    continue # Skip this current_span, keep active_span as is

                iou = calculate_iou(active_span["start"], active_span["end"], 
                                    current_span["start"], current_span["end"])

                # Check merge conditions
                logic_match = active_span["logic"] == current_span["logic"]
                group_match = active_span.get("group", "g1") == current_span.get("group", "g1") # Default to "g1" if missing

                if iou >= IOU_THRESHOLD and logic_match and group_match:
                    # Merge: update active_span's boundaries
                    active_span["start"] = min(active_span["start"], current_span["start"])
                    active_span["end"] = max(active_span["end"], current_span["end"])
                    # Track that a merge happened, e.g., by concatenating sme_ids (optional)
                    active_span["sme_id"] = f"{active_span.get('sme_id', '')}+{current_span.get('sme_id', '')}"
                    # The text of active_span will be re-extracted later if it's finalized
                    print(f"  Merging spans for (qid={qid}, docid={docid}). SME IDs: {active_span['sme_id']}. New range: [{active_span['start']}-{active_span['end']}]")
                else:
                    # No merge (either IoU too low or attributes differ)
                    # Finalize the current active_span
                    active_span["text"] = normalized_doc_text[active_span["start"]:active_span["end"]].strip()
                    if active_span["text"]: # Only add if text is not empty
                        # Create a clean version for final output, removing temporary sme_id concatenation
                        finalized_active_span = {k:v for k,v in active_span.items() if k != 'sme_id'} 
                        # Optionally, add a field indicating it's a result of merge or specific SMEs
                        # finalized_active_span['merged_from_smes'] = active_span.get('sme_id')
                        merged_spans_for_this_qd_pair.append(finalized_active_span)
                    else:
                        print(f"  ⚠️ Active span for (qid={qid}, docid={docid}) text became empty after processing. Discarding.", file=sys.stderr)
                    
                    # Log conflict if IoU was high but attributes differed
                    if iou >= IOU_THRESHOLD and not (logic_match and group_match):
                        log_conflict(conflict_logger, active_span, current_span, iou)
                        # If conflict, current_span also needs to be added if it's not merged
                        # (The active_span was already added. current_span becomes new active_span)

                    # current_span becomes the new active_span for the next iteration
                    active_span = dict(current_span) 
            
            # After the loop, finalize the last active_span for the group
            if active_span["start"] < active_span["end"]: # Ensure it's a valid span
                active_span["text"] = normalized_doc_text[active_span["start"]:active_span["end"]].strip()
                if active_span["text"]:
                    finalized_last_active_span = {k:v for k,v in active_span.items() if k != 'sme_id'}
                    merged_spans_for_this_qd_pair.append(finalized_last_active_span)
                else:
                    print(f"  ⚠️ Final active span for (qid={qid}, docid={docid}) text was empty. Discarding.", file=sys.stderr)
            
            final_adjudicated_spans.extend(merged_spans_for_this_qd_pair)

    # Save merged and deduplicated annotations
    if final_adjudicated_spans:
        with open(MERGED_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(final_adjudicated_spans, f, indent=2, ensure_ascii=False)
        print(f"\n✔ Deduplication complete. {len(final_adjudicated_spans)} merged/final spans saved to {MERGED_OUTPUT_FILE}")
        print(f"Conflict log saved to: {CONFLICT_LOG_FILE}")
    else:
        print("\nNo spans to save after deduplication process.")

if __name__ == "__main__":
    deduplicate_annotations()