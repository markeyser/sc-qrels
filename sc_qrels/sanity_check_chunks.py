# sc_qrels/sanity_check_chunks.py
import json
import pathlib
import re
import sys
import random

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
DOCS_DIR = PROCESSED_DATA_DIR / "documents"
CHUNK_MANIFESTS_DIR = PROCESSED_DATA_DIR / "chunk_manifests"

# Number of random chunks to sample per manifest for detailed check (in addition to first/last)
# Set to a very large number (or handle 'all') to check all chunks, but can be slow.
NUM_RANDOM_CHUNKS_TO_CHECK = 5 
CHECK_FIRST_N_CHUNKS = 2
CHECK_LAST_N_CHUNKS = 2


# ─────────────────────────────────────────────────────────────────────────────
# Normalize function (MUST BE IDENTICAL to chunk_documents.py and others)
# ─────────────────────────────────────────────────────────────────────────────
# Cache for normalized document text to avoid re-reading and re-normalizing the same doc multiple times
normalized_doc_cache_chunk_check: dict = {}

def normalize_text_for_chunk_check(text: str) -> str:
    text = text.replace('’', "'").replace('‘', "'")
    text = text.replace('”', '"').replace('“', '"')
    text = text.replace('—', '-').replace('–', '-')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_normalized_doc_for_check(doc_id: str) -> str | None:
    if doc_id in normalized_doc_cache_chunk_check:
        return normalized_doc_cache_chunk_check[doc_id]

    doc_path = DOCS_DIR / f"{doc_id}.json"
    if not doc_path.exists():
        print(f"    ERROR: Document file not found: {doc_path}", file=sys.stderr)
        return None
    try:
        with open(doc_path, "r", encoding="utf-8") as f:
            doc_content = json.load(f)
        original_text = doc_content.get("text")
        if not original_text:
            print(f"    ERROR: Document {doc_id} has no 'text' field or is empty.", file=sys.stderr)
            return None
        
        normalized_text = normalize_text_for_chunk_check(original_text)
        normalized_doc_cache_chunk_check[doc_id] = normalized_text
        return normalized_text
    except Exception as e:
        print(f"    ERROR: Could not load or normalize document {doc_id}: {e}", file=sys.stderr)
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Main Validation Logic
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("--- Starting Chunk Manifest Sanity Check ---")
    overall_violations = 0
    manifest_files = sorted(CHUNK_MANIFESTS_DIR.glob("*.jsonl"))

    if not manifest_files:
        print(f"No chunk manifest files found in {CHUNK_MANIFESTS_DIR}. Exiting.", file=sys.stderr)
        sys.exit(0) # Not an error, just nothing to check

    for manifest_file_path in manifest_files:
        print(f"\n📄 Validating manifest: {manifest_file_path.name}")
        violations_in_current_file = 0
        chunks_in_file = []
        try:
            with open(manifest_file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        chunk = json.loads(line)
                        chunks_in_file.append({"data": chunk, "line_num": line_num})
                    except json.JSONDecodeError:
                        print(f"  ERROR: Line {line_num}: Invalid JSON.", file=sys.stderr)
                        violations_in_current_file += 1
        except Exception as e:
            print(f"  ERROR: Could not read or process file {manifest_file_path}: {e}", file=sys.stderr)
            overall_violations +=1 # Count as a major violation
            continue
        
        print(f"  Loaded {len(chunks_in_file)} chunks from {manifest_file_path.name}.")
        if not chunks_in_file:
            continue

        # --- Determine which chunks to sample for detailed text check ---
        chunks_to_check_indices = set()
        # Add first N
        for i in range(min(CHECK_FIRST_N_CHUNKS, len(chunks_in_file))):
            chunks_to_check_indices.add(i)
        # Add last N
        for i in range(min(CHECK_LAST_N_CHUNKS, len(chunks_in_file))):
            chunks_to_check_indices.add(len(chunks_in_file) - 1 - i)
        # Add random N
        if len(chunks_in_file) > (CHECK_FIRST_N_CHUNKS + CHECK_LAST_N_CHUNKS):
            available_indices_for_random = list(set(range(len(chunks_in_file))) - chunks_to_check_indices)
            num_to_sample_randomly = min(NUM_RANDOM_CHUNKS_TO_CHECK, len(available_indices_for_random))
            if num_to_sample_randomly > 0:
                 chunks_to_check_indices.update(random.sample(available_indices_for_random, num_to_sample_randomly))
        
        print(f"  Performing detailed text check on {len(chunks_to_check_indices)} sampled chunks...")

        for chunk_idx, chunk_item in enumerate(chunks_in_file):
            chunk = chunk_item["data"]
            line_num = chunk_item["line_num"]
            
            # 1. Check for required fields
            required_fields = ["original_doc_id", "chunk_id", "start", "end", "text"]
            missing_fields = [field for field in required_fields if field not in chunk]
            if missing_fields:
                print(f"  ERROR: Line {line_num}, Chunk ID {chunk.get('chunk_id', 'N/A')}: Missing fields: {', '.join(missing_fields)}", file=sys.stderr)
                violations_in_current_file += 1
                continue # Skip further checks for this malformed chunk

            doc_id = chunk["original_doc_id"]
            chunk_id = chunk["chunk_id"]
            start_offset = chunk["start"]
            end_offset = chunk["end"]
            stored_chunk_text = chunk["text"]

            # Only do the expensive document loading and normalization for sampled chunks
            if chunk_idx in chunks_to_check_indices:
                normalized_doc = get_normalized_doc_for_check(doc_id)
                if normalized_doc is None: # Error already printed by get_normalized_doc_for_check
                    violations_in_current_file += 1
                    continue

                # 2. Offset bounds check
                if not (isinstance(start_offset, int) and isinstance(end_offset, int) and \
                        0 <= start_offset <= end_offset <= len(normalized_doc)):
                    print(f"  ERROR: Line {line_num}, Chunk ID {chunk_id}: Invalid offsets. "
                          f"Start: {start_offset}, End: {end_offset}, DocLen: {len(normalized_doc)}", file=sys.stderr)
                    violations_in_current_file += 1
                    continue
                
                # 3. Reconstruct text from normalized document and compare
                reconstructed_text = normalized_doc[start_offset:end_offset]
                # The text stored in the chunk manifest should be exactly this slice,
                # without extra stripping, as it represents the segment.
                if reconstructed_text != stored_chunk_text:
                    print(f"  ERROR: Line {line_num}, Chunk ID {chunk_id}: Text mismatch.", file=sys.stderr)
                    print(f"    Stored     : '{stored_chunk_text[:100]}...' (len {len(stored_chunk_text)})", file=sys.stderr)
                    print(f"    Reconstructed: '{reconstructed_text[:100]}...' (len {len(reconstructed_text)})", file=sys.stderr)
                    # For very detailed debugging of a mismatch:
                    # if violations_in_current_file < 3: # Limit extensive debug output
                    #     for i in range(min(len(stored_chunk_text), len(reconstructed_text))):
                    #         if stored_chunk_text[i] != reconstructed_text[i]:
                    #             print(f"      First diff at char {i}: Stored='{stored_chunk_text[i]}' (ord {ord(stored_chunk_text[i])}), Recon='{reconstructed_text[i]}' (ord {ord(reconstructed_text[i])})")
                    #             break
                    #     if len(stored_chunk_text) != len(reconstructed_text):
                    #         print(f"      Length difference is also an issue.")
                    violations_in_current_file += 1

        if violations_in_current_file == 0:
            print(f"  ✔ All checks passed for {manifest_file_path.name} (based on sampled chunks).")
        else:
            print(f"  ❌ Found {violations_in_current_file} issues in {manifest_file_path.name}.")
            overall_violations += violations_in_current_file
        
        # Clear cache for next manifest file if memory is a concern, 
        # or keep it if docids often repeat across manifests (less likely for chunk manifests of different strategies)
        # normalized_doc_cache_chunk_check.clear() 


    print("\n--- Chunk Manifest Sanity Check Finished ---")
    if overall_violations == 0:
        print("✔✔✔ All chunk manifests passed all checks (based on sampled chunks).")
        sys.exit(0)
    else:
        print(f"❌❌❌ Found {overall_violations} total issues across chunk manifests.")
        sys.exit(1)

if __name__ == "__main__":
    main()