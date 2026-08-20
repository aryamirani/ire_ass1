import polars as pl
import numpy as np
from pathlib import Path
import warnings
from typing import List, Dict, Tuple
from collections import Counter
import math

from src.evaluation.metrics import evaluate_ranking, bootstrap_confidence_intervals
from src.scripts.generate_submission import FastBM25

warnings.filterwarnings("ignore")

def run_evaluation(num_samples_eval: int = 5000):
    print("=" * 60)
    print("      OFFLINE EVALUATION HARNESS (MIND / EB-NeRD)")
    print("=" * 60)
    
    val_path = Path("data/processed/mind/val_interactions.parquet")
    articles_path = Path("data/processed/mind/articles.parquet")
    
    if not val_path.exists() or not articles_path.exists():
        print("Processed feature store not found. Please run build_pipeline.py first.")
        return

    print("1. Loading articles and building BM25 index...")
    articles_df = pl.read_parquet(articles_path)
    article_dict = {}
    for row in articles_df.select(["article_id", "title", "abstract"]).iter_rows():
        art_id, title, abstract = row[0], row[1] or "", row[2] or ""
        tokens = (title + " " + abstract).lower().split()
        article_dict[art_id] = tokens
        
    bm25 = FastBM25()
    bm25.fit(article_dict)
    print(f"BM25 index built with {len(article_dict)} articles.")

    print(f"2. Evaluating on validation impressions (sample size: {num_samples_eval})...")
    val_df = pl.read_parquet(val_path)
    if val_df.height > num_samples_eval:
        val_df = val_df.sample(num_samples_eval, seed=42)

    auc_all, mrr_all, ndcg5_all, ndcg10_all = [], [], [], []
    cold_auc, warm_auc = [], []
    cold_mrr, warm_mrr = [], []
    
    for row in val_df.select(["history", "impressions"]).iter_rows():
        hist_str = row[0] or ""
        impr_str = row[1] or ""
        
        hist_ids = hist_str.split() if hist_str else []
        impr_items = impr_str.split() if impr_str else []
        
        cand_ids = []
        y_true = []
        for item in impr_items:
            if "-" in item:
                aid, label = item.rsplit("-", 1)
                cand_ids.append(aid)
                y_true.append(int(label))
                
        if not cand_ids or len(y_true) < 2 or sum(y_true) == 0:
            continue
            
        y_true_arr = np.array(y_true, dtype=np.int32)
        
        # Construct user query
        query_tokens = []
        for hid in hist_ids[-10:]:
            if hid in article_dict:
                query_tokens.extend(article_dict[hid])
                
        y_scores = np.array(bm25.score_candidates(query_tokens, cand_ids), dtype=np.float32)
        
        metrics = evaluate_ranking(y_true_arr, y_scores)
        auc_all.append(metrics["AUC"])
        mrr_all.append(metrics["MRR"])
        ndcg5_all.append(metrics["nDCG@5"])
        ndcg10_all.append(metrics["nDCG@10"])
        
        # User slicing: Cold-start (<= 3 clicks) vs Warm (> 3 clicks)
        if len(hist_ids) <= 3:
            cold_auc.append(metrics["AUC"])
            cold_mrr.append(metrics["MRR"])
        else:
            warm_auc.append(metrics["AUC"])
            warm_mrr.append(metrics["MRR"])

    print("\n" + "=" * 60)
    print("                OVERALL BENCHMARK RESULTS")
    print("=" * 60)
    
    auc_m, auc_l, auc_u = bootstrap_confidence_intervals(auc_all)
    mrr_m, mrr_l, mrr_u = bootstrap_confidence_intervals(mrr_all)
    n5_m, n5_l, n5_u = bootstrap_confidence_intervals(ndcg5_all)
    n10_m, n10_l, n10_u = bootstrap_confidence_intervals(ndcg10_all)
    
    print(f"AUC:      {auc_m:.4f}  [95% CI: {auc_l:.4f} - {auc_u:.4f}]")
    print(f"MRR:      {mrr_m:.4f}  [95% CI: {mrr_l:.4f} - {mrr_u:.4f}]")
    print(f"nDCG@5:   {n5_m:.4f}  [95% CI: {n5_l:.4f} - {n5_u:.4f}]")
    print(f"nDCG@10:  {n10_m:.4f}  [95% CI: {n10_l:.4f} - {n10_u:.4f}]")
    
    print("\n" + "-" * 60)
    print("         USER SLICING ABLATION (Cold-Start vs. Warm)")
    print("-" * 60)
    if cold_auc:
        c_auc_m, c_auc_l, c_auc_u = bootstrap_confidence_intervals(cold_auc)
        c_mrr_m, c_mrr_l, c_mrr_u = bootstrap_confidence_intervals(cold_mrr)
        print(f"Cold-Start (<=3 clicks, N={len(cold_auc)}): AUC = {c_auc_m:.4f} [{c_auc_l:.4f}, {c_auc_u:.4f}] | MRR = {c_mrr_m:.4f}")
    if warm_auc:
        w_auc_m, w_auc_l, w_auc_u = bootstrap_confidence_intervals(warm_auc)
        w_mrr_m, w_mrr_l, w_mrr_u = bootstrap_confidence_intervals(warm_mrr)
        print(f"Warm Users (>3 clicks, N={len(warm_auc)}):   AUC = {w_auc_m:.4f} [{w_auc_l:.4f}, {w_auc_u:.4f}] | MRR = {w_mrr_m:.4f}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_evaluation()
