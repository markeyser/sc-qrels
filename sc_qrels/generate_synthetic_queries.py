# sc_qrels/generate_synthetic_queries.py
"""Generate synthetic QA pairs and exact span annotations for each Alice chapter.
SME1 uses OpenAI. SME2 (optional pass) uses Google Gemini.
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
import re
import sys
import argparse # For command-line arguments
from pathlib import Path
from uuid import uuid4
from typing import List, Optional, Tuple, Dict

from openai import OpenAI
from google import genai as google_genai_client # Use your working import for Gemini

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHAPTER_DIR = Path("data/processed/documents")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- SME 1 (OpenAI) Configuration ---
QUESTIONS_SME1_PATH = OUTPUT_DIR / "questions_sme1.json"
ANNOTATIONS_SME1_OPENAI_PATH = OUTPUT_DIR / "annotations_sme1_openai.json"
MODEL_SME1_PRIMARY = "gpt-4o"
TEMP_SME1_QUESTION = 0.7
TEMP_SME1_SPAN_AND_LOGIC = 0.0

# --- SME 2 (Google Gemini) Configuration ---
ANNOTATIONS_SME2_GEMINI_PATH = OUTPUT_DIR / "annotations_sme2_gemini.json"
MODEL_SME2_GEMINI = "gemini-1.5-flash-latest" # Or your preferred "gemini-2.0-flash"

NUM_QUESTIONS_PER_CHAP_TARGET = (6, 10)
MAX_TOKENS_LLM_CALL_OPENAI = 15000

SNAP_SPANS_TO_WHOLE_WORDS = True

client_openai = OpenAI() 

try:
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("⚠️ GOOGLE_API_KEY not found in .env file. SME2 (Gemini) generation will fail if attempted.", file=sys.stderr)
        client_gemini_legacy = None
    else:
        client_gemini_legacy = google_genai_client.Client(api_key=google_api_key)
except Exception as e:
    print(f"⚠️ Error initializing Google Gemini client (using google.genai.Client): {e}. SME2 generation will fail.", file=sys.stderr)
    client_gemini_legacy = None

normalized_doc_cache: Dict[str, str] = {}

# ---------------------------------------------------------------------------
# Helper Function: Load Chapters
# ---------------------------------------------------------------------------
def load_chapters() -> List[dict]:
    chapters_path = sorted(CHAPTER_DIR.glob("alice:ch*.json"))
    if not chapters_path:
        print(f"⚠️ No chapter files found in {CHAPTER_DIR}. Please ensure they exist.", file=sys.stderr)
        return []
    loaded_chapters = []
    for fp in chapters_path:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                loaded_chapters.append(json.load(f))
        except Exception as e:
            print(f"⚠️ Error loading or parsing chapter file {fp}: {e}", file=sys.stderr)
    return loaded_chapters

# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------
QUESTION_GEN_FEWSHOTS = r'''
Example 1:
Document Snippet: "Alice was beginning to get very tired of sitting by her sister on the bank."
Output: { "questions": ["What was Alice doing before she saw the White Rabbit?"] }

Example 2:
Document Snippet: "Suddenly, a White Rabbit with pink eyes ran close by her, muttering \"Oh dear! Oh dear! I shall be late!\" The Rabbit actually took a watch out of its waistcoat-pocket."
Output: { "questions": ["What peculiar behavior did the White Rabbit exhibit while muttering?", "What did the Rabbit take from its pocket?"] }
'''

def build_question_generation_prompt(chapter: dict) -> str:
    low, high = NUM_QUESTIONS_PER_CHAP_TARGET
    return f"""
You are an expert at generating insightful and diverse questions from a given text.
Your task is to generate between {low} and {high} questions based on the document provided below.
Your entire output **must be a single JSON object** with a key named "questions", whose value is a JSON array of question strings.
Structure: {{ "questions": ["Question 1...", "Question 2..."] }}
Few-shot examples: {QUESTION_GEN_FEWSHOTS}
Now, generate questions for the following full document:
Title: {chapter['title']}
Document ID: {chapter['docid']}
Document: \"\"\"{chapter['text']}\"\"\"
"""

ANSWER_EXTRACT_LOGIC_FEWSHOTS = r'''
Example 1 (single-span):
Document:
"""Alice was beginning to get very tired of sitting by her sister on the bank."""
Question: "What was Alice doing before she saw the White Rabbit?"
Output:
{
  "answers": ["Alice was beginning to get very tired of sitting by her sister on the bank"],
  "logic": "COMPLETE_SPAN"
}

Example 2 (alternative answers → OR):
Document:
"""The Cheshire Cat said "We're all mad here" and then "But I don't want to go among mad people"."""
Question: "Which memorable lines did the Cheshire Cat say about madness?"
Output:
{
  "answers": ["We're all mad here","But I don't want to go among mad people"],
  "logic": "OR"
}

Example 3 (all-required answers → AND):
Document:
"""Alice found a small golden key and a low curtain hiding a tiny door."""
Question: "Name the two items Alice discovered that were significant for the tiny door." 
Output:
{
  "answers": ["a small golden key","a low curtain"],
  "logic": "AND"
}
Example 4 (clear alternative options → OR):
Document:
"""The path split left towards the dark wood and right towards the sunny meadow."""
Question: "Which directions could one take at the fork in the path?"
Output:
{
  "answers": ["left towards the dark wood", "right towards the sunny meadow"],
  "logic": "OR"
}
'''

def build_answer_extraction_logic_prompt(chapter: dict, question: str) -> str:
    return f"""
You are an expert annotator specializing in precise, verbatim text extraction and logical analysis.
Given the document and a specific question, your tasks are:
1.  Find and extract the **exact, continuous verbatim substring(s)** from the document that directly answer the question.
2.  Determine the **logical relationship** between these answer spans relative to the question. The logic can be 'COMPLETE_SPAN', 'OR', or 'AND'.
**CRITICAL INSTRUCTIONS FOR ANSWERS & LOGIC:**
-   **Answers**:
    1.  Each answer **must be an EXACT, continuous verbatim substring** copied character-for-character.
    2.  Do NOT paraphrase, summarize, reword, or add non-document text.
    3.  Do NOT cut words in half.
    4.  Dialogue only, unless narrative is integral.
    5.  No ellipses (...) unless in the source.
    6.  IMPORTANT: Do not truncate answers. Provide the full, complete verbatim span.
-   **Logic**:
    1.  `COMPLETE_SPAN`: Use if a single, self-contained answer span fully addresses the question.
    2.  `OR`: Use if multiple distinct answer spans are found, and **any ONE of these spans, if presented alone, would be a complete and sufficient answer to the question.**
    3.  `AND`: Use if multiple distinct answer spans are found, and **ALL of them are required together** to comprehensively answer the question.
Your entire output **must be a single JSON object**. This object must contain:
-   A key named "answers": a JSON array of extracted verbatim answer strings.
-   A key named "logic": a string, either "COMPLETE_SPAN", "OR", or "AND".
Return JSON ONLY (no markdown) in this structure: {{ "answers": ["exact span..."], "logic": "AND" }}
Here are a few examples: {ANSWER_EXTRACT_LOGIC_FEWSHOTS} 
Now, process:
Title: {chapter['title']}
Document ID: {chapter['docid']}
Question: "{question}"
Document: \"\"\"{chapter['text']}\"\"\"
"""

# ---------------------------------------------------------------------------
# LLM Call Helpers
# ---------------------------------------------------------------------------
def call_openai_llm_for_json(prompt_str: str, model: str, temperature: float, purpose: str) -> Optional[Dict]:
    try:
        messages = [{"role": "user", "content": prompt_str}]
        completion_params = {
            "model": model, "messages": messages, "max_tokens": MAX_TOKENS_LLM_CALL_OPENAI,
            "response_format": {"type": "json_object"}, "temperature": temperature
        }
        resp = client_openai.chat.completions.create(**completion_params)
        content = resp.choices[0].message.content.strip()
        loaded_json = json.loads(content)

        if not isinstance(loaded_json, dict):
            print(f"❌ OpenAI ({purpose}) response was not a JSON object: {content[:300]}...", file=sys.stderr)
            return None
        if purpose == "questions_sme1":
            key_to_check = "questions"
            if key_to_check not in loaded_json or not isinstance(loaded_json[key_to_check], list):
                print(f"❌ OpenAI ({purpose}) response missing '{key_to_check}' key or not a list: {content[:300]}...", file=sys.stderr)
                return None
        elif purpose == "answers_logic_sme1":
            if "answers" not in loaded_json or not isinstance(loaded_json["answers"], list):
                print(f"❌ OpenAI ({purpose}) response missing 'answers' key or not a list: {content[:300]}...", file=sys.stderr)
                return None
            if "logic" not in loaded_json or loaded_json["logic"] not in ["COMPLETE_SPAN", "OR", "AND"]:
                print(f"⚠️ OpenAI ({purpose}) response missing 'logic' key or invalid: '{loaded_json.get('logic', 'MISSING')}'. Defaulting later. Content: {content[:300]}...", file=sys.stderr)
        else:
            print(f"❌ Unknown purpose '{purpose}' for OpenAI LLM call.", file=sys.stderr)
            return None
        return loaded_json
    except json.JSONDecodeError as e:
        raw_content = "OpenAI Response content not available";
        if 'resp' in locals() and hasattr(resp, 'choices') and resp.choices and hasattr(resp.choices[0], 'message') and resp.choices[0].message: raw_content = resp.choices[0].message.content.strip()
        print(f"❌ JSONDecodeError in OpenAI ({purpose}) response: {e}. Raw content: {raw_content[:500]}...", file=sys.stderr); return None
    except Exception as e:
        print(f"❌ Error calling OpenAI LLM ({purpose}): {e}", file=sys.stderr); return None


def call_gemini_client_llm_for_json(prompt_str: str, model_name: str, purpose: str) -> Optional[Dict]:
    if not client_gemini_legacy:
        print("❌ Gemini client (google.genai.Client) not initialized. Cannot make SME2 call.", file=sys.stderr)
        return None
    try:
        response = client_gemini_legacy.models.generate_content(
            model=f"models/{model_name}", 
            contents=[prompt_str]
        )

        if not hasattr(response, 'text') or not response.text:
            error_detail = "Unknown error or empty response"
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason'):
                 error_detail = f"Block reason: {response.prompt_feedback.block_reason}"
            elif hasattr(response, 'candidates') and not response.candidates:
                 error_detail = "No candidates returned"
            print(f"❌ Gemini ({purpose}) response empty or missing text. Detail: {error_detail}", file=sys.stderr)
            return None

        content = response.text.strip()
        
        if content.startswith("```json"): content = content[len("```json"):].strip()
        if content.startswith("```"): content = content[len("```"):].strip()
        if content.endswith("```"): content = content[:-len("```")].strip()
            
        loaded_json = json.loads(content)

        if not isinstance(loaded_json, dict):
            print(f"❌ Gemini ({purpose}) response was not a JSON object after parsing: {content[:300]}...", file=sys.stderr)
            return None
        
        if purpose == "answers_logic_sme2": 
            if "answers" not in loaded_json or not isinstance(loaded_json["answers"], list):
                print(f"❌ Gemini ({purpose}) response missing 'answers' key or not a list: {content[:300]}...", file=sys.stderr)
                return None
            if "logic" not in loaded_json or loaded_json["logic"] not in ["COMPLETE_SPAN", "OR", "AND"]:
                print(f"⚠️ Gemini ({purpose}) response missing 'logic' key or invalid: '{loaded_json.get('logic', 'MISSING')}'. Defaulting later. Content: {content[:300]}...", file=sys.stderr)
        else:
            print(f"❌ Unknown purpose '{purpose}' for Gemini LLM call.", file=sys.stderr)
            return None
        return loaded_json

    except json.JSONDecodeError as e:
        raw_content = "Gemini Response content not available";
        if 'response' in locals() and hasattr(response, 'text'): raw_content = response.text.strip()
        print(f"❌ JSONDecodeError in Gemini ({purpose}) response: {e}. Raw content: {raw_content[:500]}...", file=sys.stderr); return None
    except Exception as e:
        print(f"❌ Error calling Gemini LLM ({purpose}): {e}", file=sys.stderr)
        import traceback; traceback.print_exc() 
        return None

# ---------------------------------------------------------------------------
# Span Location and Normalization
# ---------------------------------------------------------------------------
def get_normalized_doc_text(docid: str, original_text: str) -> str:
    if docid not in normalized_doc_cache:
        text = original_text; text = text.replace('’', "'").replace('‘', "'"); text = text.replace('”', '"').replace('“', '"'); text = text.replace('—', '-').replace('–', '-'); text = re.sub(r'\s+', ' ', text).strip()
        normalized_doc_cache[docid] = text
    return normalized_doc_cache[docid]

def locate_span(normalized_doc_text: str, snippet_from_llm: str) -> Optional[Tuple[int, int, str]]:
    if not snippet_from_llm: return None
    
    cleaned_snippet = snippet_from_llm.strip()
    cleaned_snippet = cleaned_snippet.replace('’', "'").replace('‘', "'") 
    cleaned_snippet = cleaned_snippet.replace('”', '"').replace('“', '"') 
    cleaned_snippet = cleaned_snippet.replace('—', '-').replace('–', '-') 
    cleaned_snippet = re.sub(r"\s*(\.\.\.|…|\.\.)$", "", cleaned_snippet).rstrip() 
    cleaned_snippet = re.sub(r'\s+', ' ', cleaned_snippet).strip() 

    temp_snippet_for_stripper = cleaned_snippet 
    punctuation_stripper = re.compile(r"^[,\.\"'\s]*(.*?)[,\.\"'\s]*$") 
    match = punctuation_stripper.match(temp_snippet_for_stripper)
    if match:
        stripped_core = match.group(1)
        if stripped_core is not None : # Check if group(1) actually matched something
             cleaned_snippet = stripped_core
    
    if not cleaned_snippet: return None 

    primary_variants = {cleaned_snippet} 
    if len(cleaned_snippet) > 1: 
        if (cleaned_snippet.startswith("'") and cleaned_snippet.endswith("'")) or \
           (cleaned_snippet.startswith('"') and cleaned_snippet.endswith('"')):
            primary_variants.add(cleaned_snippet[1:-1])
    if len(cleaned_snippet) > 2: 
        if (cleaned_snippet.startswith("_") and cleaned_snippet.endswith("_") and not cleaned_snippet.startswith("__")) or \
           (cleaned_snippet.startswith("*") and cleaned_snippet.endswith("*") and not cleaned_snippet.startswith("**")):
            primary_variants.add(cleaned_snippet[1:-1])

    search_variants_ordered = []
    for pv in sorted(list(primary_variants), key=len, reverse=True):
        if not pv: continue 
        search_variants_ordered.append(pv)
        if "_" in pv: search_variants_ordered.append(pv.replace("_", " "))
        no_underscore_variant = pv.replace("_", "")
        if no_underscore_variant != pv and no_underscore_variant != pv.replace("_", " "):
            search_variants_ordered.append(no_underscore_variant)

    seen_lower = set()
    final_search_variants = []
    for v in search_variants_ordered: # CORRECTED INDENTATION HERE
        v_lower = v.lower() 
        if v_lower not in seen_lower:
            final_search_variants.append(v)
            seen_lower.add(v_lower)
    
    normalized_doc_text_lower = normalized_doc_text.lower()
    for var_to_find in final_search_variants:
        if not var_to_find: continue
        try:
            start_idx = normalized_doc_text_lower.find(var_to_find.lower())
            if start_idx != -1:
                end_idx = start_idx + len(var_to_find)
                
                if SNAP_SPANS_TO_WHOLE_WORDS:
                    snap_start = start_idx
                    while snap_start > 0 and not normalized_doc_text[snap_start-1].isspace():
                        snap_start -= 1
                    
                    snap_end = end_idx
                    while snap_end < len(normalized_doc_text) and not normalized_doc_text[snap_end].isspace():
                        snap_end += 1
                    
                    start_idx, end_idx = snap_start, snap_end
                
                located_text = normalized_doc_text[start_idx:end_idx].strip() 
                if not located_text: 
                    continue
                return start_idx, end_idx, located_text
        except Exception: continue 
    return None
# ---------------------------------------------------------------------------
# Core Generation Logic for a Single SME
# ---------------------------------------------------------------------------
def generate_annotations_for_sme(
    sme_id_str: str,
    chapters_data: List[dict], 
    questions_to_process: List[Dict], 
    llm_call_func, 
    llm_model_name_for_answers: str, 
    llm_temp_for_answers: Optional[float], 
    output_annotations_path: Path
    ):
    
    sme_final_annotations = []
    chapters_lookup = {chap['docid']: chap for chap in chapters_data}

    for q_entry_idx, q_entry in enumerate(questions_to_process):
        qid = q_entry["qid"]
        q_text = q_entry["question"]
        docid = q_entry["docid"]

        print(f"    ➡️ {sme_id_str} - Processing Q {q_entry_idx+1}/{len(questions_to_process)} (QID: {qid}, DOCID: {docid})", flush=True)

        if docid not in chapters_lookup:
            print(f"    ⚠️ Document {docid} not found for QID {qid}. Skipping.", file=sys.stderr)
            continue
        
        chap = chapters_lookup[docid]
        original_chapter_text = chap["text"]
        normalized_chapter_text = get_normalized_doc_text(docid, original_chapter_text)

        answer_logic_prompt_str = build_answer_extraction_logic_prompt(chap, q_text)
        
        answer_logic_response_json = None
        if llm_call_func == call_openai_llm_for_json:
            purpose = "answers_logic_sme1"
            answer_logic_response_json = llm_call_func(
                answer_logic_prompt_str, 
                llm_model_name_for_answers, 
                llm_temp_for_answers,
                purpose
            )
        elif llm_call_func == call_gemini_client_llm_for_json:
            purpose = "answers_logic_sme2"
            answer_logic_response_json = llm_call_func(
                answer_logic_prompt_str, 
                llm_model_name_for_answers, # Pass Gemini model name
                purpose
            )
        else:
            print(f"    ❌ Unknown LLM call function for {sme_id_str}. Skipping.", file=sys.stderr)
            continue

        if not answer_logic_response_json: 
            print(f"    ❌ {sme_id_str} - Failed to get valid answer/logic structure for Q: '{q_text[:70]}...'.", file=sys.stderr)
            continue 

        extracted_answer_texts_from_llm = [str(a).strip() for a in answer_logic_response_json.get("answers",[]) if isinstance(a, str) and str(a).strip()]
        
        llm_determined_logic = answer_logic_response_json.get("logic")
        if llm_determined_logic not in ["COMPLETE_SPAN", "OR", "AND"]:
            print(f"    ⚠️ {sme_id_str} - LLM provided invalid or missing logic ('{llm_determined_logic}'). Defaulting for QID {qid}.", file=sys.stderr)
            llm_determined_logic = "COMPLETE_SPAN" if len(extracted_answer_texts_from_llm) == 1 else "OR"

        if not extracted_answer_texts_from_llm:
            print(f"    ⚠️ {sme_id_str} - No answer spans extracted by LLM for Q: '{q_text[:70]}...'.", file=sys.stderr)
            continue 
        
        located_spans_for_this_q_by_this_sme = 0
        for ans_text_single in extracted_answer_texts_from_llm:
            location_result = locate_span(normalized_chapter_text, ans_text_single)
            
            if not location_result:
                print(f"    ⚠️ {sme_id_str} - Local span not found in {docid} for LLM ans: '{ans_text_single[:80]}...' (Q: '{q_text[:60]}...')", file=sys.stderr)
                continue

            start, end, located_text_from_normalized = location_result
            
            sme_final_annotations.append({
                "qid": qid, "docid": docid, "start": start, "end": end, 
                "text": located_text_from_normalized, 
                "logic": llm_determined_logic, 
                "group": q_entry.get("group", "g1"), 
                "sme_id": sme_id_str 
            })
            located_spans_for_this_q_by_this_sme +=1
        
        if located_spans_for_this_q_by_this_sme > 0:
            print(f"    ✅ {sme_id_str} - Successfully located {located_spans_for_this_q_by_this_sme} span(s) with logic '{llm_determined_logic}' for Q: '{q_text[:60]}...'", file=sys.stderr)
        else:
             print(f"    ❌ {sme_id_str} - No valid local spans found by locate_span for Q: '{q_text[:60]}...' in {docid}.", file=sys.stderr)

    if sme_final_annotations:
        with open(output_annotations_path, "w", encoding="utf-8") as fa:
            json.dump(sme_final_annotations, fa, indent=2, ensure_ascii=False)
        print(f"\n✔ {sme_id_str}: Generated {len(sme_final_annotations)} spans. Saved to {output_annotations_path}")
    else:
        print(f"\nℹ️ {sme_id_str}: No annotations generated or saved to {output_annotations_path}")
    return sme_final_annotations

# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------
def main(run_sme1: bool, run_sme2: bool):
    global normalized_doc_cache
    normalized_doc_cache = {} 

    chapters_data = load_chapters() 
    if not chapters_data:
        print("❌ No chapters loaded. Exiting.", file=sys.stderr)
        return

    sme1_generated_questions_list = [] 

    if run_sme1:
        print(f"\n--- Starting SME1 (OpenAI: {MODEL_SME1_PRIMARY}) Annotation Generation ---")
        current_sme1_questions = [] 

        for chap_idx, chap in enumerate(chapters_data):
            docid = chap["docid"]
            print(f"\n🧠 SME1 - Processing {docid} (Chapter {chap_idx+1}/{len(chapters_data)}) …", flush=True)
            print(f"  ➡️ SME1 - Generating questions for {docid} (Temp: {TEMP_SME1_QUESTION}) …", flush=True)
            
            question_prompt = build_question_generation_prompt(chap)
            questions_response_json = call_openai_llm_for_json(question_prompt, MODEL_SME1_PRIMARY, TEMP_SME1_QUESTION, "questions_sme1")

            if not questions_response_json: 
                print(f"  ❌ SME1 - Failed to generate valid questions for {docid}. Skipping chapter.", file=sys.stderr)
                continue
            
            chapter_questions_text = [q for q in questions_response_json.get("questions", []) if isinstance(q, str) and q.strip()]
            if not chapter_questions_text:
                print(f"  ℹ️ SME1 - No questions generated or extracted for {docid}. Skipping chapter.", file=sys.stderr)
                continue
            print(f"  ✅ SME1 - Received {len(chapter_questions_text)} questions for {docid}.", file=sys.stderr)

            for q_text in chapter_questions_text:
                qid = f"q_{uuid4().hex[:8]}"
                current_sme1_questions.append({"qid": qid, "question": q_text, "docid": docid, "group": "g1"})
        
        sme1_generated_questions_list = current_sme1_questions

        if sme1_generated_questions_list:
            with open(QUESTIONS_SME1_PATH, "w", encoding="utf-8") as fq:
                json.dump(sme1_generated_questions_list, fq, indent=2, ensure_ascii=False)
            print(f"\n✔ SME1 - Generated {len(sme1_generated_questions_list)} total questions. Saved to {QUESTIONS_SME1_PATH}")
            
            print(f"\n  ➡️ SME1 - Generating annotations for these questions (Model: {MODEL_SME1_PRIMARY}, Temp: {TEMP_SME1_SPAN_AND_LOGIC})")
            generate_annotations_for_sme(
                sme_id_str="SME1_OpenAI",
                chapters_data=chapters_data,
                questions_to_process=sme1_generated_questions_list,
                llm_call_func=call_openai_llm_for_json,
                llm_model_name_for_answers=MODEL_SME1_PRIMARY,
                llm_temp_for_answers=TEMP_SME1_SPAN_AND_LOGIC,
                output_annotations_path=ANNOTATIONS_SME1_OPENAI_PATH
            )
        else:
            print("\nℹ️ SME1 - No questions were generated. Skipping annotation generation for SME1.")

    if run_sme2:
        if not client_gemini_legacy:
            print("❌ SME2 (Gemini) run requested, but Gemini client is not initialized. Skipping SME2.", file=sys.stderr)
            return

        questions_for_sme2 = []
        if sme1_generated_questions_list: 
            questions_for_sme2 = sme1_generated_questions_list
            print(f"ℹ️ Using {len(questions_for_sme2)} questions generated by SME1 in this session for SME2.")
        elif QUESTIONS_SME1_PATH.exists(): 
            print(f"ℹ️ SME1 questions not generated in this run. Loading from {QUESTIONS_SME1_PATH} for SME2.")
            try:
                with open(QUESTIONS_SME1_PATH, "r", encoding="utf-8") as f:
                    questions_for_sme2 = json.load(f)
                if not questions_for_sme2:
                     print(f"❌ Loaded questions file {QUESTIONS_SME1_PATH} is empty. Cannot run SME2.", file=sys.stderr)
                     return
            except Exception as e:
                print(f"❌ Error loading SME1 questions from {QUESTIONS_SME1_PATH} for SME2: {e}", file=sys.stderr)
                return
        else:
            print(f"❌ SME1 questions file ({QUESTIONS_SME1_PATH}) not found. Run SME1 first or ensure file exists to run SME2.", file=sys.stderr)
            return

        if not questions_for_sme2:
            print("❌ No questions available for SME2 pass. Exiting SME2 generation.", file=sys.stderr)
            return

        print(f"\n--- Starting SME2 (Google Gemini: {MODEL_SME2_GEMINI}) Annotation Generation ---")
        print(f"  ➡️ SME2 - Generating annotations for {len(questions_for_sme2)} questions from SME1.")
        
        generate_annotations_for_sme(
            sme_id_str="SME2_Gemini",
            chapters_data=chapters_data,
            questions_to_process=questions_for_sme2, 
            llm_call_func=call_gemini_client_llm_for_json,
            llm_model_name_for_answers=MODEL_SME2_GEMINI, 
            llm_temp_for_answers=None, 
            output_annotations_path=ANNOTATIONS_SME2_GEMINI_PATH
        )

    print("\n--- Script Finished ---")
    if run_sme1:
        print(f"SME1 output (OpenAI): {QUESTIONS_SME1_PATH}, {ANNOTATIONS_SME1_OPENAI_PATH}")
    if run_sme2:
        print(f"SME2 output (Gemini): {ANNOTATIONS_SME2_GEMINI_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic QA annotations using one or two SMEs.")
    parser.add_argument("--run_sme1", action="store_true", help="Run SME1 (OpenAI) question and annotation generation pass.")
    parser.add_argument("--run_sme2", action="store_true", help="Run SME2 (Google Gemini) annotation generation pass. Uses questions from SME1.")
    
    args = parser.parse_args()

    if not args.run_sme1 and not args.run_sme2:
        print("No SME pass selected. Use --run_sme1 and/or --run_sme2.")
        print("Example: python sc_qrels/generate_synthetic_queries.py --run_sme1 --run_sme2")
    else:
        normalized_doc_cache = {}
        main(run_sme1=args.run_sme1, run_sme2=args.run_sme2)