# sc_qrels/generate_retriever_runs.py
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import os 
import sys # For utils.py path adjustment if needed
from typing import List, Optional, Dict, Tuple # CORRECTED: Added List and other common types


# Attempt to import get_torch_device from utils.py
try:
    from utils import get_torch_device
except ImportError:
    try:
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        from utils import get_torch_device
        print("Loaded get_torch_device from parent directory.")
    except ImportError:
        print("Error: utils.py with get_torch_device not found. Please ensure it's accessible.")
        print("Defaulting to CPU. For GPU/MPS, ensure utils.py is correctly placed.")
        def get_torch_device(): return torch.device("cpu")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

CHUNK_MANIFESTS_DIR = PROCESSED_DATA_DIR / "chunk_manifests"
QUESTIONS_FILE = PROCESSED_DATA_DIR / "questions_sme1.json" 
EMBEDDINGS_OUTPUT_DIR = PROCESSED_DATA_DIR / "strategy_embeddings" 
EMBEDDINGS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_FILES_OUTPUT_DIR = PROCESSED_DATA_DIR / "retriever_runs"
RUN_FILES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


MODEL_NAME = "BAAI/bge-large-en-v1.5" 
TOP_K = 20 
YOUR_RUN_NAME_PREFIX = "BGE_DenseRun" 

DEVICE = get_torch_device()
print(f"🖥️  Using device: {DEVICE}")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    embedding_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
except Exception as e:
    print(f"❌ Error loading HuggingFace model or tokenizer ({MODEL_NAME}): {e}", file=sys.stderr)
    print("   Ensure you have an internet connection or the model is cached.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    pooled = (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)
    return pooled

def embed_texts(texts_to_embed: List[str], batch_size: int = 32) -> torch.Tensor:
    """Embeds a list of texts in batches and returns a stacked tensor of embeddings."""
    all_embeddings_list = []
    for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="    Embedding texts", leave=False, ncols=80):
        batch_texts = texts_to_embed[i:i+batch_size]
        if not batch_texts: continue # Should not happen with correct loop logic

        encoded_input = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            model_output = embedding_model(**encoded_input)
            batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        all_embeddings_list.append(batch_embeddings.cpu()) 
    
    if not all_embeddings_list:
        # Attempt to get hidden size, default if model config not available
        hidden_size = getattr(embedding_model.config, 'hidden_size', 1024) # Default to BGE-large size
        return torch.empty((0, hidden_size), dtype=torch.float32)
        
    return torch.cat(all_embeddings_list, dim=0)

# ---------------------------------------------------------------------------
# Main Processing Logic
# ---------------------------------------------------------------------------
def main():
    print("--- Starting Retriever Run File Generation ---")

    try:
        with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
            questions = json.load(f)
        if not questions:
            print(f"No questions found in {QUESTIONS_FILE}. Exiting.", file=sys.stderr)
            return
        print(f"✔ Loaded {len(questions)} questions from {QUESTIONS_FILE}")
    except Exception as e:
        print(f"❌ Error loading questions from {QUESTIONS_FILE}: {e}", file=sys.stderr)
        return

    chunk_manifest_files = sorted(CHUNK_MANIFESTS_DIR.glob("chunks_*.jsonl"))
    if not chunk_manifest_files:
        print(f"No chunk manifest files found in {CHUNK_MANIFESTS_DIR}. Exiting.", file=sys.stderr)
        return
    
    print(f"Found {len(chunk_manifest_files)} chunk manifest strategies to process.")

    for manifest_path in chunk_manifest_files:
        strategy_name = manifest_path.stem.replace("chunks_", "")
        print(f"\n📄 Processing Strategy: {strategy_name} (from {manifest_path.name})")

        current_strategy_chunks = []
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    current_strategy_chunks.append(json.loads(line))
            if not current_strategy_chunks:
                print(f"  ℹ️ No chunks found in {manifest_path.name}. Skipping this strategy.", file=sys.stderr)
                continue
            print(f"  Loaded {len(current_strategy_chunks)} chunks for {strategy_name}.")
        except Exception as e:
            print(f"  ❌ Error loading chunks from {manifest_path.name}: {e}", file=sys.stderr)
            continue
            
        chunk_texts = [chunk['text'] for chunk in current_strategy_chunks]
        chunk_ids_for_strategy = [chunk['chunk_id'] for chunk in current_strategy_chunks]

        if not chunk_texts:
             print(f"  No text found in chunks for {strategy_name}. Skipping embedding and retrieval.", file=sys.stderr)
             continue
        
        print(f"  Embedding {len(chunk_texts)} chunks for {strategy_name}...")
        chunk_vecs_strategy = embed_texts(chunk_texts).to(DEVICE) 
        
        if chunk_vecs_strategy.numel() == 0: 
            print(f"  Embedding resulted in an empty tensor for {strategy_name}. Skipping retrieval.", file=sys.stderr)
            continue

        run_file_path = RUN_FILES_OUTPUT_DIR / f"run_{YOUR_RUN_NAME_PREFIX}_{strategy_name}.txt"
        run_name_for_trec = f"{YOUR_RUN_NAME_PREFIX}_{strategy_name}"

        with open(run_file_path, "w", encoding="utf-8") as fout:
            for q_data in tqdm(questions, desc=f"  Retrieving for {strategy_name}", leave=False, ncols=80):
                qid = q_data["qid"]
                question_text = q_data["question"]

                q_vec = embed_texts([question_text]).to(DEVICE)

                if q_vec.numel() == 0:
                    print(f"    ⚠️ Embedding for QID {qid} resulted in empty tensor. Skipping.", file=sys.stderr)
                    continue

                scores = torch.matmul(q_vec, chunk_vecs_strategy.T).squeeze(0) 
                
                actual_k = min(TOP_K, len(chunk_ids_for_strategy)) 
                if actual_k == 0 : continue

                top_k_scores, top_k_indices = torch.topk(scores, k=actual_k)

                for rank, (score_val, chunk_orig_idx) in enumerate(zip(top_k_scores.tolist(), top_k_indices.tolist())):
                    retrieved_chunk_id = chunk_ids_for_strategy[chunk_orig_idx]
                    fout.write(f"{qid}\tQ0\t{retrieved_chunk_id}\t{rank + 1}\t{score_val:.8f}\t{run_name_for_trec}\n")
        
        print(f"  ✔ Saved TREC run file for {strategy_name} to: {run_file_path}")

    print("\n--- Retriever Run File Generation Finished ---")

if __name__ == "__main__":
    main()