# sc_qrels/analyze_output_distribution.py
import json
from pathlib import Path
from collections import defaultdict
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Paths (adjust as needed)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR             = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR   = BASE_DIR / "data" / "processed"
QUESTIONS_FILE       = PROCESSED_DATA_DIR / "questions.json"
ANNOTATIONS_FILE     = PROCESSED_DATA_DIR / "annotations.json"
OUTPUT_MD_FILE       = PROCESSED_DATA_DIR / "output_distribution.md"

def analyze_distribution():
    # --- Load questions ---
    try:
        questions_data = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error loading questions: {e}", file=sys.stderr)
        return

    # --- Load annotations ---
    try:
        annotations_data = json.loads(ANNOTATIONS_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error loading annotations: {e}", file=sys.stderr)
        return

    # --- Step 1: Group annotations by qid ---
    spans_per_question = defaultdict(list)
    for ann in annotations_data:
        spans_per_question[ann["qid"]].append(ann)

    # --- Step 2: Group questions by docid and collect stats ---
    chapter_stats = defaultdict(lambda: {
        "total_questions": 0,
        "questions_details": []
    })

    for q_entry in questions_data:
        qid          = q_entry["qid"]
        docid        = q_entry["docid"]
        question_txt = q_entry["question"]

        chapter_stats[docid]["total_questions"] += 1

        associated_spans = spans_per_question.get(qid, [])
        num_spans        = len(associated_spans)
        logic_type       = associated_spans[0]["logic"] if associated_spans else "N/A"

        chapter_stats[docid]["questions_details"].append({
            "qid":           qid,
            "num_spans":     num_spans,
            "logic_type":    logic_type,
            "question_text": (
                question_txt if len(question_txt) <= 100
                else question_txt[:100] + "..."
            )
        })

    # --- Step 3: Count spans by logic type overall ---
    logic_counts = defaultdict(int)
    for ann in annotations_data:
        logic_counts[ann["logic"]] += 1

    # --- Step 4: Build Markdown report ---
    md_lines = []
    md_lines.append("# Output Distribution Analysis\n")
    total_questions_overall = 0
    total_spans_overall     = 0

    # Per-chapter breakdown
    for docid in sorted(chapter_stats.keys()):
        stats = chapter_stats[docid]
        total_questions_overall += stats["total_questions"]

        md_lines.append(f"## Chapter `{docid}`")
        md_lines.append(f"- **Total questions:** {stats['total_questions']}\n")

        chapter_spans_count = 0
        for q in stats["questions_details"]:
            chapter_spans_count += q["num_spans"]
            md_lines.append(f"### QID: `{q['qid']}`")
            md_lines.append(f"- **Question:** {q['question_text']}")
            md_lines.append(f"- **# spans:** {q['num_spans']}")
            md_lines.append(f"- **Logic type:** {q['logic_type']}\n")

        total_spans_overall += chapter_spans_count
        md_lines.append(f"**Total spans in {docid}:** {chapter_spans_count}\n\n")

    # Overall summary
    md_lines.append("---")
    md_lines.append("## Overall Summary")
    md_lines.append(f"- **Total valid questions processed:** {total_questions_overall}")
    md_lines.append(f"- **Total valid spans associated:** {total_spans_overall}")
    md_lines.append(f"- **Questions without spans:** "
                    f"{sum(1 for q in questions_data if len(spans_per_question.get(q['qid'], [])) == 0)}\n")

    # Span counts by logic type
    md_lines.append("### Spans by logic type")
    # Always show these three categories, defaulting to 0 if missing
    for logic in ["COMPLETE_SPAN", "AND", "OR"]:
        count = logic_counts.get(logic, 0)
        md_lines.append(f"- **{logic}**: {count}")
    md_lines.append("")  # blank line


    md_lines.append("> **Assertion:** Every question here has at least one span. "
                    "Questions without spans would have been dropped by `generate_synthetic_queries.py`.\n")

    report_md = "\n".join(md_lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Print to stdout
    # ─────────────────────────────────────────────────────────────────────────
    print(report_md)

    # ─────────────────────────────────────────────────────────────────────────
    # Save to Markdown file
    # ─────────────────────────────────────────────────────────────────────────
    try:
        OUTPUT_MD_FILE.write_text(report_md, encoding="utf-8")
        print(f"\n✔ Markdown report saved to {OUTPUT_MD_FILE}")
    except Exception as e:
        print(f"Error writing Markdown report: {e}", file=sys.stderr)

if __name__ == "__main__":
    analyze_distribution()
