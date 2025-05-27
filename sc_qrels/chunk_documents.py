# sc_qrels/chunk_documents.py
import json
import re
from pathlib import Path
import nltk # For sentence tokenization
from transformers import AutoTokenizer # For token-based chunking
import sys

nltk.download('punkt_tab')

# Ensure NLTK's punkt tokenizer is downloaded (user might need to run this once manually or add to setup)
try:
    nltk.data.find('tokenizers/punkt')
except (LookupError, nltk.downloader.DownloadError): # Catch both if punkt not found or general download error
    print("NLTK 'punkt' tokenizer not found. Attempting to download...", file=sys.stderr)
    try:
        nltk.download('punkt', quiet=False) # Set quiet=False to see download progress/errors
        print("NLTK 'punkt' tokenizer downloaded successfully.", file=sys.stderr)
    except Exception as e:
        print(f"Error downloading NLTK 'punkt': {e}", file=sys.stderr)
        print("Please ensure NLTK's 'punkt' model is available. You might need to run: import nltk; nltk.download('punkt')", file=sys.stderr)
        # Depending on strictness, you might sys.exit(1) here if sentence tokenization is critical

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent # Assumes this script is in sc_qrels/
DOCS_DIR = BASE_DIR / "data" / "processed" / "documents"
CHUNK_OUTPUT_DIR = BASE_DIR / "data" / "processed" / "chunk_manifests"
CHUNK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Tokenizer for Token-Based Strategy ---
TOKENIZER_MODEL_NAME = "BAAI/bge-large-en-v1.5"
try:
    hf_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME, use_fast=True)
except Exception as e:
    print(f"Error loading HuggingFace tokenizer {TOKENIZER_MODEL_NAME}: {e}", file=sys.stderr)
    print("Token-based chunking will fail.", file=sys.stderr)
    hf_tokenizer = None


# ---------------------------------------------------------------------------
# Text Normalization Function (MUST BE IDENTICAL ACROSS ALL SCRIPTS)
# ---------------------------------------------------------------------------
def normalize_text_for_chunking(text: str) -> str:
    text = text.replace('’', "'").replace('‘', "'")
    text = text.replace('”', '"').replace('“', '"')
    text = text.replace('—', '-').replace('–', '-')
    text = re.sub(r'\s+', ' ', text).strip() # Collapses all whitespace (newlines, tabs, multiple spaces) to single space
    return text

# ---------------------------------------------------------------------------
# Chunking Strategy Implementations
# ---------------------------------------------------------------------------

# Strategy 1: Token-Based Sliding Window (BGE512T)
def chunk_strategy_token_window(normalized_doc_text: str, doc_id: str, 
                                tokenizer, chunk_size_tokens: int, stride_tokens: int):
    if not tokenizer:
        print(f"  [TokenWindow] Tokenizer not available for {doc_id}. Skipping.", file=sys.stderr)
        return []
        
    chunks = []
    # Tokenize the entire normalized document to get input_ids and offset_mappings
    # add_special_tokens=False is important for not adding CLS/SEP to the content itself
    tokenized_output = tokenizer.encode_plus(
        normalized_doc_text,
        add_special_tokens=False, 
        return_offsets_mapping=True,
        truncation=False # We handle chunking manually
    )
    
    all_token_ids = tokenized_output["input_ids"]
    all_offset_mapping = tokenized_output["offset_mapping"] # List of (char_start, char_end)

    if not all_token_ids: # Handle empty text after normalization/tokenization
        return []

    chunk_index = 0
    for i in range(0, len(all_token_ids), stride_tokens):
        current_chunk_token_ids = all_token_ids[i : i + chunk_size_tokens]
        current_chunk_offset_mapping = all_offset_mapping[i : i + chunk_size_tokens]

        if not current_chunk_token_ids: # Should not happen if loop condition is correct
            continue

        # Character start is from the first token of this chunk
        char_start = current_chunk_offset_mapping[0][0]
        # Character end is from the last token of this chunk
        char_end = current_chunk_offset_mapping[-1][1]
        
        # The text for the manifest is the slice from the *normalized document* using these char offsets
        chunk_text_from_normalized_doc = normalized_doc_text[char_start:char_end]

        if not chunk_text_from_normalized_doc.strip(): # Avoid empty or all-whitespace chunks
            continue

        chunks.append({
            "original_doc_id": doc_id,
            "chunk_id": f"TOKWIN{chunk_size_tokens}S{stride_tokens}-{doc_id}-{chunk_index:04d}",
            "start": char_start,
            "end": char_end,
            "text": chunk_text_from_normalized_doc
        })
        chunk_index += 1
        
        # Stop if this chunk already covers up to or beyond the end of the token list
        if (i + chunk_size_tokens) >= len(all_token_ids):
            break
            
    return chunks

# Strategy 2: Sentence-Based Chunking (SENT)
def chunk_strategy_sentences(normalized_doc_text: str, doc_id: str):
    chunks = []
    try:
        sentences = nltk.sent_tokenize(normalized_doc_text)
    except Exception as e:
        print(f"  [SentenceChunker] NLTK sent_tokenize error for {doc_id}: {e}. Skipping sentence chunking for this doc.", file=sys.stderr)
        return []

    current_search_offset = 0
    for i, sentence_text in enumerate(sentences):
        if not sentence_text.strip():
            continue
        
        try:
            # Find the sentence in the normalized text to get its accurate start offset
            # This is important because sent_tokenize might handle some punctuation slightly differently
            # or if there are identical sentences.
            char_start = normalized_doc_text.index(sentence_text, current_search_offset)
        except ValueError:
            # Fallback if sentence not found from current_search_offset (e.g. due to very subtle modifications by tokenizer)
            # This should be rare if normalized_doc_text is stable.
            char_start = normalized_doc_text.find(sentence_text)
            if char_start == -1:
                print(f"  [SentenceChunker] Warning: Could not reliably find sentence in {doc_id}: '{sentence_text[:50]}...'", file=sys.stderr)
                continue # Skip this problematic sentence
        
        char_end = char_start + len(sentence_text)
        
        chunks.append({
            "original_doc_id": doc_id,
            "chunk_id": f"SENT-{doc_id}-{i:04d}", # Using index as part of ID
            "start": char_start,
            "end": char_end,
            "text": sentence_text # Text is the sentence itself
        })
        current_search_offset = char_end # Ensure next search starts after this sentence
    return chunks

# Strategy 3: Fixed-Character Window (CHAR_WIN_500_OV50)
def chunk_strategy_char_window(normalized_doc_text: str, doc_id: str, 
                               window_size_chars: int, overlap_chars: int):
    chunks = []
    step = window_size_chars - overlap_chars
    if step <= 0:
        print(f"  [CharWindow] Error: Window size ({window_size_chars}) must be greater than overlap ({overlap_chars}). Skipping for {doc_id}.", file=sys.stderr)
        return []

    doc_len = len(normalized_doc_text)
    chunk_index = 0
    for char_start in range(0, doc_len, step):
        char_end = min(char_start + window_size_chars, doc_len)
        chunk_text = normalized_doc_text[char_start:char_end]

        if not chunk_text.strip():
            if char_end >= doc_len: break
            continue

        chunks.append({
            "original_doc_id": doc_id,
            "chunk_id": f"CHARWIN{window_size_chars}OV{overlap_chars}-{doc_id}-{chunk_index:04d}",
            "start": char_start,
            "end": char_end,
            "text": chunk_text
        })
        chunk_index += 1
        if char_end >= doc_len:
            break
    return chunks

# Strategy 4: Large Character Blocks (Non-Overlapping, CHAR_BLOCK_1000_NOV0)
def chunk_strategy_char_blocks(normalized_doc_text: str, doc_id: str, 
                               block_size_chars: int): # Overlap is 0
    # This is a special case of char_window with overlap=0
    return chunk_strategy_char_window(normalized_doc_text, doc_id, 
                                      window_size_chars=block_size_chars, 
                                      overlap_chars=0)


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------
def main():
    print("--- Starting Document Chunking for SC-Qrels ---")
    doc_files = sorted(DOCS_DIR.glob("alice:ch*.json"))
    if not doc_files:
        print(f"No document files found in {DOCS_DIR}. Exiting.", file=sys.stderr)
        return

    # Define strategies to run
    # Each entry: (strategy_name, function_to_call, args_for_function (excluding text, doc_id))
    strategies_to_run = [
        ("BGE512T_S128", chunk_strategy_token_window, (hf_tokenizer, 512, 128)), # Using global hf_tokenizer
        ("SENT", chunk_strategy_sentences, ()),
        ("CHARWIN500_OV50", chunk_strategy_char_window, (500, 50)),
        ("CHARBLOCK1000_NOV0", chunk_strategy_char_blocks, (1000,)), # Only block_size needed
    ]

    for strategy_name, chunk_func, func_args in strategies_to_run:
        print(f"\n-- Running Strategy: {strategy_name} --")
        all_chunks_for_strategy = []
        for doc_file_path in doc_files:
            try:
                with open(doc_file_path, "r", encoding="utf-8") as f:
                    doc_content = json.load(f)
                doc_id = doc_content.get("docid")
                original_text = doc_content.get("text")

                if not doc_id or not original_text:
                    print(f"  Skipping {doc_file_path}, missing 'docid' or 'text'.", file=sys.stderr)
                    continue
                
                print(f"  Processing document for {strategy_name}: {doc_id}")
                normalized_text = normalize_text_for_chunking(original_text)
                
                if not normalized_text: # Handle cases where normalization results in empty text
                    print(f"  Normalized text for {doc_id} is empty. Skipping chunking.", file=sys.stderr)
                    continue

                # Call the chunking function with normalized_text, doc_id, and its specific args
                chunks = chunk_func(normalized_text, doc_id, *func_args)
                all_chunks_for_strategy.extend(chunks)
                print(f"    Generated {len(chunks)} chunks for {doc_id} using {strategy_name}.")

            except Exception as e:
                print(f"  Error processing document {doc_file_path} for strategy {strategy_name}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()


        if all_chunks_for_strategy:
            output_path = CHUNK_OUTPUT_DIR / f"chunks_{strategy_name}.jsonl"
            with open(output_path, "w", encoding="utf-8") as fout:
                for chunk in all_chunks_for_strategy:
                    fout.write(json.dumps(chunk) + "\n")
            print(f"✔ Saved {strategy_name} manifest to {output_path} ({len(all_chunks_for_strategy)} total chunks)")
        else:
            print(f"ℹ️ No chunks generated for strategy {strategy_name}.")
            
    print("\n--- Document Chunking Finished ---")

if __name__ == "__main__":
    main()