# sc_qrels/sanity_check.py
import json
import pathlib
import re
import sys
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
DOCS_DIR = PROCESSED_DATA_DIR / "documents"

QUESTIONS_FILE = PROCESSED_DATA_DIR / "questions_sme1.json"
ANNOTATIONS_SME1_FILE = PROCESSED_DATA_DIR / "annotations_sme1_openai.json"
ANNOTATIONS_SME2_FILE = PROCESSED_DATA_DIR / "annotations_sme2_gemini.json"

# ─────────────────────────────────────────────────────────────────────────────
# Load Questions
# ─────────────────────────────────────────────────────────────────────────────
try:
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions_sme1 = json.load(f)
    print(f"✔ Loaded {len(questions_sme1)} questions from {QUESTIONS_FILE}")
except Exception as e:
    print(f"❌ Error loading questions from {QUESTIONS_FILE}: {e}", file=sys.stderr)
    sys.exit(1)

question_id_to_docid_map = {q["qid"]: q["docid"] for q in questions_sme1}
all_question_ids = set(question_id_to_docid_map.keys())

# ─────────────────────────────────────────────────────────────────────────────
# Normalize function (must match generate_synthetic_queries.py)
# ─────────────────────────────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("”", '"').replace("“", '"') # CORRECTED LINE
    text = text.replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", text).strip()

# ─────────────────────────────────────────────────────────────────────────────
# Function to Validate Annotations for a Single SME
# ─────────────────────────────────────────────────────────────────────────────
def validate_sme_annotations(annotations_file_path: pathlib.Path, sme_id_str: str, all_qids: set, qid_to_docid: dict):
    print(f"\n--- Validating annotations for {sme_id_str} from {annotations_file_path} ---")
    sme_violations = []
    sme_annotations_data = []
    sme_qids_with_spans = set()

    try:
        with open(annotations_file_path, "r", encoding="utf-8") as f:
            sme_annotations_data = json.load(f)
        print(f"  Loaded {len(sme_annotations_data)} annotations for {sme_id_str}.")
    except FileNotFoundError:
        print(f"  ❌ Annotation file not found: {annotations_file_path}", file=sys.stderr)
        sme_violations.append(("GLOBAL", str(annotations_file_path), "ANNOTATION_FILE_NOT_FOUND"))
        return sme_violations, sme_qids_with_spans
    except Exception as e:
        print(f"  ❌ Error loading annotations from {annotations_file_path}: {e}", file=sys.stderr)
        sme_violations.append(("GLOBAL", str(annotations_file_path), f"ANNOTATION_FILE_LOAD_ERROR: {e}"))
        return sme_violations, sme_qids_with_spans

    annotations_by_qid = defaultdict(list)
    for ann in sme_annotations_data:
        annotations_by_qid[ann["qid"]].append(ann)

    for qid in all_qids: 
        docid_for_q = qid_to_docid.get(qid)
        if not docid_for_q:
            print(f"  ⚠️ QID {qid} from reference questions not found in QID-to-DocID map. Skipping its validation for {sme_id_str}.", file=sys.stderr)
            continue

        if qid not in annotations_by_qid:
            sme_violations.append((qid, docid_for_q, f"{sme_id_str}:NO_SPAN_FOR_QUESTION"))
            continue 

        sme_qids_with_spans.add(qid) 

        doc_path = DOCS_DIR / f"{docid_for_q}.json"
        if not doc_path.exists():
            if not any(v[0] == qid and v[2] == "DOC_NOT_FOUND" for v in sme_violations):
                 sme_violations.append((qid, docid_for_q, "DOC_NOT_FOUND"))
            continue 

        try:
            doc = json.loads(doc_path.read_text(encoding="utf-8"))
            orig_doc_text = doc.get("text", "")
            if not orig_doc_text:
                 sme_violations.append((qid, docid_for_q, "DOC_EMPTY_OR_NO_TEXT_FIELD"))
                 continue
            norm_doc_text = normalize_text(orig_doc_text)
        except Exception as e:
            sme_violations.append((qid, docid_for_q, f"DOC_LOAD_OR_NORMALIZE_ERROR: {e}"))
            continue
            
        for a in annotations_by_qid[qid]:
            ann_text, start, end = a["text"], a["start"], a["end"]

            if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(norm_doc_text)):
                sme_violations.append((qid, docid_for_q, f"{sme_id_str}:INVALID_OFFSETS [start:{start}, end:{end}, len:{len(norm_doc_text)}]"))
                continue

            reconstructed_from_norm = norm_doc_text[start:end]
            reconstructed_from_norm_stripped = reconstructed_from_norm.strip() 

            if reconstructed_from_norm_stripped != ann_text:
                sme_violations.append((qid, docid_for_q, f"{sme_id_str}:TEXT_MISMATCH"))
                if len([v for v in sme_violations if v[2] == f"{sme_id_str}:TEXT_MISMATCH"]) < 5: # Print debug for first few mismatches
                    print(f"    DEBUG MISMATCH ({sme_id_str}) QID={qid} DOCID={docid_for_q}", file=sys.stderr)
                    print(f"      Annotation Text: '{ann_text}' (len {len(ann_text)})", file=sys.stderr)
                    print(f"      Reconstructed  : '{reconstructed_from_norm_stripped}' (len {len(reconstructed_from_norm_stripped)}) (from norm[{start}:{end}].strip())", file=sys.stderr)

    if not sme_violations:
        print(f"  ✔ All checks passed for {sme_id_str}.")
    else:
        # Filter out the global errors from the count of per-annotation issues
        per_annotation_issues = [v for v in sme_violations if v[0] != "GLOBAL"]
        print(f"  ❌ Found {len(per_annotation_issues)} specific annotation issues for {sme_id_str}.")
        
    return sme_violations, sme_qids_with_spans

# ─────────────────────────────────────────────────────────────────────────────
# Main Validation Logic
# ─────────────────────────────────────────────────────────────────────────────
all_overall_violations = []

sme1_violations, sme1_qids_covered = validate_sme_annotations(ANNOTATIONS_SME1_FILE, "SME1_OpenAI", all_question_ids, question_id_to_docid_map)
all_overall_violations.extend(sme1_violations)

sme2_violations, sme2_qids_covered = validate_sme_annotations(ANNOTATIONS_SME2_FILE, "SME2_Gemini", all_question_ids, question_id_to_docid_map)
all_overall_violations.extend(sme2_violations)

# ─────────────────────────────────────────────────────────────────────────────
# Additional Check: Question Coverage Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Question Coverage Summary ---")
# Ensure sme1_qids_covered and sme2_qids_covered are sets before set operations
sme1_qids_covered = set(sme1_qids_covered) if sme1_qids_covered is not None else set()
sme2_qids_covered = set(sme2_qids_covered) if sme2_qids_covered is not None else set()


qids_covered_by_both = sme1_qids_covered.intersection(sme2_qids_covered)
qids_only_sme1 = sme1_qids_covered - sme2_qids_covered
qids_only_sme2 = sme2_qids_covered - sme1_qids_covered
qids_covered_by_neither = all_question_ids - (sme1_qids_covered.union(sme2_qids_covered))

print(f"Total reference questions: {len(all_question_ids)}")
print(f"  Questions with valid spans from SME1: {len(sme1_qids_covered)}")
print(f"  Questions with valid spans from SME2: {len(sme2_qids_covered)}")
print(f"  Questions covered by BOTH SME1 and SME2: {len(qids_covered_by_both)}")
if qids_only_sme1:
    print(f"  Questions covered ONLY by SME1: {len(qids_only_sme1)} (e.g., {list(qids_only_sme1)[:3]})")
if qids_only_sme2:
    print(f"  Questions covered ONLY by SME2: {len(qids_only_sme2)} (e.g., {list(qids_only_sme2)[:3]})")
if qids_covered_by_neither: 
    print(f"  Questions with NO valid spans from EITHER SME: {len(qids_covered_by_neither)} (e.g., {list(qids_covered_by_neither)[:3]})")

# ─────────────────────────────────────────────────────────────────────────────
# Final Report
# ─────────────────────────────────────────────────────────────────────────────
# Filter out global file-level errors for the final count of actual annotation violations
specific_annotation_violations = [v for v in all_overall_violations if v[0] != "GLOBAL"]

if not specific_annotation_violations and not any(v[2] == "ANNOTATION_FILE_NOT_FOUND" for v in all_overall_violations):
    print("\n✔✔✔ All sanity checks passed for both SMEs' annotations relative to their own generation!")
    if any(v[2].endswith(":NO_SPAN_FOR_QUESTION") for v in all_overall_violations):
        print("    (Note: Some questions may not have spans from one or both SMEs, as detailed in coverage summary.)")
    sys.exit(0)
else:
    print(f"\n❌❌❌ Found {len(specific_annotation_violations)} specific annotation violation(s) and potentially file-level issues:", file=sys.stderr)
    
    # Report file-level issues first
    global_errors = [v for v in all_overall_violations if v[0] == "GLOBAL"]
    if global_errors:
        print("\n  Global File/Load Errors:", file=sys.stderr)
        for _, filepath, error_msg in global_errors:
            print(f"    - File: {filepath}, Error: {error_msg}", file=sys.stderr)

    # Group specific violations by QID for cleaner reporting
    violations_by_qid = defaultdict(list)
    for item in specific_annotation_violations:
        # Ensure item is a tuple of 3 elements before unpacking
        if isinstance(item, tuple) and len(item) == 3:
            qid, docid, error = item
            violations_by_qid[qid].append((docid, error))
        else:
            print(f"    Malformed violation item: {item}", file=sys.stderr)


    for qid, errors_for_qid in sorted(violations_by_qid.items()):
        print(f"\n  Violations for QID={qid}:", file=sys.stderr)
        for docid, error_msg in errors_for_qid:
            loc = f" in {docid}" if docid else ""
            print(f"    - {error_msg}{loc}", file=sys.stderr)
    sys.exit(1)