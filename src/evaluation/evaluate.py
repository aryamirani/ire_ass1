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

def evaluate_ebnerd(num_samples_eval: int = 5000):
    print("=" * 60)
    print("      OFFLINE EVALUATION HARNESS (EB-NeRD LARGE)")
    print("=" * 60)
    
    val_path = Path("data/processed/ebnerd/val_interactions.parquet")
    articles_path = Path("data/processed/ebnerd/articles.parquet")
    hist_path = Path("data/processed/ebnerd/user_history.parquet")
    
    if not val_path.exists() or not articles_path.exists():
        print("EB-NeRD processed feature store not found.")
        return

    print("1. Loading EB-NeRD articles and building BM25 index...")
    articles_df = pl.read_parquet(articles_path)
    article_dict = {}
    sub_col = "abstract" if "abstract" in articles_df.columns else "subtitle" if "subtitle" in articles_df.columns else None
    
    for row in articles_df.select(["article_id", "title", sub_col] if sub_col else ["article_id", "title"]).iter_rows():
        art_id = str(row[0])
        title = row[1] or ""
        sub = row[2] or "" if sub_col else ""
        tokens = (title + " " + sub).lower().split()
        article_dict[art_id] = tokens
        
    bm25 = FastBM25()
    bm25.fit(article_dict)
    print(f"BM25 index built with {len(article_dict)} articles.")

    print("2. Loading user history...")
    user_history_map = {}
    if hist_path.exists():
        hist_df = pl.read_parquet(hist_path)
        hist_col = "article_id_fixed" if "article_id_fixed" in hist_df.columns else "history" if "history" in hist_df.columns else hist_df.columns[1]
        for row in hist_df.select(["user_id", hist_col]).iter_rows():
            uid = str(row[0])
            raw_hist = row[1]
            if isinstance(raw_hist, list) or isinstance(raw_hist, np.ndarray):
                h_list = [str(x) for x in raw_hist]
            elif isinstance(raw_hist, str):
                h_list = [str(x) for x in raw_hist.split()]
            else:
                h_list = []
            user_history_map[uid] = h_list

    print(f"3. Evaluating on EB-NeRD validation impressions (sample: {num_samples_eval})...")
    val_df = pl.read_parquet(val_path)
    if val_df.height > num_samples_eval:
        val_df = val_df.sample(num_samples_eval, seed=42)

    auc_all, mrr_all, ndcg5_all, ndcg10_all = [], [], [], []
    cold_auc, warm_auc = [], []
    cold_mrr, warm_mrr = [], []
    
    inview_col = "article_ids_inview" if "article_ids_inview" in val_df.columns else "impressions"
    clicked_col = "article_ids_clicked" if "article_ids_clicked" in val_df.columns else None
    
    for row in val_df.select(["user_id", inview_col, clicked_col] if clicked_col else ["user_id", inview_col]).iter_rows():
        uid = str(row[0]) if row[0] is not None else ""
        raw_cands = row[1]
        raw_clicks = row[2] if clicked_col else []
        
        if isinstance(raw_cands, list) or isinstance(raw_cands, np.ndarray):
            cand_ids = [str(x) for x in raw_cands]
        elif isinstance(raw_cands, str):
            cand_ids = [str(x) for x in raw_cands.split()]
        else:
            continue
            
        if isinstance(raw_clicks, list) or isinstance(raw_clicks, np.ndarray):
            click_set = set(str(x) for x in raw_clicks)
        elif isinstance(raw_clicks, str):
            click_set = set(raw_clicks.split())
        else:
            click_set = set()
            
        y_true = [1 if cid in click_set else 0 for cid in cand_ids]
        if not cand_ids or len(y_true) < 2 or sum(y_true) == 0:
            continue
            
        y_true_arr = np.array(y_true, dtype=np.int32)
        history_ids = user_history_map.get(uid, [])
        query_tokens = []
        for hid in history_ids[-10:]:
            if hid in article_dict:
                query_tokens.extend(article_dict[hid])
                
        y_scores = np.array(bm25.score_candidates(query_tokens, cand_ids), dtype=np.float32)
        metrics = evaluate_ranking(y_true_arr, y_scores)
        
        auc_all.append(metrics["AUC"])
        mrr_all.append(metrics["MRR"])
        ndcg5_all.append(metrics["nDCG@5"])
        ndcg10_all.append(metrics["nDCG@10"])
        
        if len(history_ids) <= 3:
            cold_auc.append(metrics["AUC"])
            cold_mrr.append(metrics["MRR"])
        else:
            warm_auc.append(metrics["AUC"])
            warm_mrr.append(metrics["MRR"])

    print("\n" + "=" * 60)
    print("           EB-NeRD LARGE BENCHMARK RESULTS")
    print("=" * 60)
    if auc_all:
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
    evaluate_ebnerd()
