# sc_qrels/evaluate_retriever_runs.py
import json
from pathlib import Path
import pytrec_eval 
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Optional, Tuple # CORRECTED: Added Dict
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

QRELS_DIR = PROCESSED_DATA_DIR / "qrels"
RUN_FILES_DIR = PROCESSED_DATA_DIR / "retriever_runs"

METRICS_TO_COMPUTE = {
    'map',              
    'ndcg_cut_10',      
    'P_5',              
    'P_10',             
    'recall_100',       
    'recall_20',        
    'num_ret',          
    'num_rel',          
    'num_rel_ret',      
    'recip_rank'        
}
# ---------------------------------------------------------------------------
# Helper functions to load qrels and run files into dict format
# ---------------------------------------------------------------------------
def load_qrels_to_dict(file_path: Path) -> Dict[str, Dict[str, int]]:
    qrels_dict = defaultdict(dict)
    with open(file_path, 'r') as f_qrel:
        for line in f_qrel:
            try:
                qid, _, docid, rel = line.strip().split()
                qrels_dict[qid][docid] = int(rel)
            except ValueError:
                print(f"  ⚠️ Skipping malformed line in qrels file {file_path.name}: {line.strip()}", file=sys.stderr)
                continue
    return qrels_dict

def load_run_to_dict(file_path: Path) -> Dict[str, Dict[str, float]]:
    run_dict = defaultdict(dict)
    with open(file_path, 'r') as f_run:
        for line in f_run:
            try:
                qid, _, docid, rank, score, _ = line.strip().split()
                run_dict[qid][docid] = float(score)
            except ValueError:
                print(f"  ⚠️ Skipping malformed line in run file {file_path.name}: {line.strip()}", file=sys.stderr)
                continue
    return run_dict

# ---------------------------------------------------------------------------
# Main Evaluation Logic
# ---------------------------------------------------------------------------
def main():
    print("--- Starting Retriever Evaluation using pytrec_eval ---")

    qrels_files = sorted(QRELS_DIR.glob("qrels_*.txt"))
    # run_files = sorted(RUN_FILES_DIR.glob("run_*.txt")) # Not directly used for matching anymore

    if not qrels_files:
        print(f"No qrels files found in {QRELS_DIR}. Exiting.", file=sys.stderr)
        return
    
    results_summary = []

    for qrels_path in qrels_files:
        qrels_filename = qrels_path.name
        if not qrels_filename.startswith("qrels_") or not qrels_filename.endswith(".txt"):
            print(f"Skipping unexpected file in qrels directory: {qrels_filename}", file=sys.stderr)
            continue
        strategy_name_from_qrels = qrels_filename.replace("qrels_", "").replace(".txt", "")
        
        run_file_name_pattern = f"run_*_{strategy_name_from_qrels}.txt" 
        matching_run_files = list(RUN_FILES_DIR.glob(run_file_name_pattern))

        if not matching_run_files:
            print(f"  ⚠️ No matching run file found for qrels: {qrels_path.name} (pattern: {run_file_name_pattern}). Skipping.", file=sys.stderr)
            continue
        if len(matching_run_files) > 1:
            print(f"  ⚠️ Multiple matching run files found for qrels: {qrels_path.name}. Using first one: {matching_run_files[0].name}. Please check naming.", file=sys.stderr)
        
        run_path = matching_run_files[0]

        print(f"\n📄 Evaluating Strategy: {strategy_name_from_qrels}")
        print(f"  Qrels file: {qrels_path.name}")
        print(f"  Run file  : {run_path.name}")

        try:
            qrels = load_qrels_to_dict(qrels_path)
        except Exception as e:
            print(f"  ❌ Error loading qrels from {qrels_path} into dict: {e}", file=sys.stderr)
            continue
        
        if not qrels:
            print(f"  ℹ️ Qrels file {qrels_path.name} is empty or resulted in no valid judgments. Skipping.", file=sys.stderr)
            continue

        try:
            run = load_run_to_dict(run_path)
        except Exception as e:
            print(f"  ❌ Error loading run from {run_path} into dict: {e}", file=sys.stderr)
            continue
        
        if not run:
            print(f"  ℹ️ Run file {run_path.name} is empty or resulted in no retrievable items. Skipping.", file=sys.stderr)
            continue

        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, 
            METRICS_TO_COMPUTE 
        )

        results_per_query = evaluator.evaluate(run)
        
        if not results_per_query:
            print(f"  ℹ️ Evaluation produced no results for strategy {strategy_name_from_qrels}.", file=sys.stderr)
            continue

        aggregated_results_for_strategy = {}
        num_queries_evaluated = len(results_per_query)
        
        print(f"  Metrics for {strategy_name_from_qrels} (averaged over {num_queries_evaluated} queries with results):")
        
        current_strategy_summary = {"Strategy": strategy_name_from_qrels}

        for measure in sorted(list(METRICS_TO_COMPUTE)):
            metric_sum = 0.0
            query_count_for_measure = 0
            for qid_results in results_per_query.values():
                if measure in qid_results:
                    metric_sum += qid_results[measure]
                    query_count_for_measure += 1
            
            if query_count_for_measure > 0:
                if measure in ['num_ret', 'num_rel', 'num_rel_ret']:
                    aggregated_value = metric_sum 
                else: 
                    aggregated_value = metric_sum / query_count_for_measure
                
                print(f"    {measure:<15}: {aggregated_value:.4f}")
                current_strategy_summary[measure] = aggregated_value
            else:
                print(f"    {measure:<15}: N/A (no values found for this measure)")
                current_strategy_summary[measure] = "N/A"
        
        results_summary.append(current_strategy_summary)

    if results_summary and pd: # Check if pandas was successfully imported
        print("\n\n--- Overall Metrics Summary ---")
        df_summary = pd.DataFrame(results_summary)
        df_summary = df_summary.set_index("Strategy")
        
        cols_order = ['map', 'ndcg_cut_10', 'recip_rank', 'P_5', 'P_10', 'recall_20', 'recall_100', 'num_rel_ret', 'num_ret', 'num_rel']
        final_cols = [col for col in cols_order if col in df_summary.columns]
        final_cols.extend([col for col in df_summary.columns if col not in final_cols])
        
        df_summary = df_summary[final_cols]
        print(df_summary.to_string(float_format="%.4f"))
    elif results_summary:
        print("\n\n--- Overall Metrics Summary (raw list) ---")
        for res in results_summary:
            print(res)
    else:
        print("\n--- No results to summarize ---")


    print("\n--- Retriever Evaluation Finished ---")

if __name__ == "__main__":
    main()