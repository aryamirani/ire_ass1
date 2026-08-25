import polars as pl
import numpy as np
from collections import Counter
from scipy.stats import rankdata
from pathlib import Path
import zipfile
import math
import os
import sys

class FastBM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs = Counter()
        self.doc_lens = {}
        self.doc_term_freqs = {}
        self.idf = {}
        self.avgdl = 0.0
        self.N = 0

    def fit(self, article_dict):
        self.N = len(article_dict)
        total_len = 0
        for aid, tokens in article_dict.items():
            dlen = len(tokens)
            self.doc_lens[aid] = dlen
            total_len += dlen
            tf = Counter(tokens)
            self.doc_term_freqs[aid] = tf
            for t in tf:
                self.doc_freqs[t] += 1
                
        self.avgdl = total_len / self.N if self.N > 0 else 1.0
        
        for t, df in self.doc_freqs.items():
            self.idf[t] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def score_candidates(self, query_tokens, candidate_ids):
        if not query_tokens:
            return [0.0] * len(candidate_ids)
            
        q_tf = Counter(query_tokens)
        scores = []
        k1 = self.k1
        b = self.b
        avgdl = self.avgdl
        
        for cid in candidate_ids:
            if cid not in self.doc_term_freqs:
                scores.append(0.0)
                continue
                
            tf_dict = self.doc_term_freqs[cid]
            dlen = self.doc_lens[cid]
            len_norm = k1 * (1.0 - b + b * (dlen / avgdl))
            
            score = 0.0
            for t, q_count in q_tf.items():
                if t in tf_dict:
                    f = tf_dict[t]
                    idf_val = self.idf.get(t, 0.0)
                    score += idf_val * ((f * (k1 + 1.0)) / (f + len_norm))
            scores.append(score)
            
        return scores

def generate_ebnerd_predictions(eb_test_path: Path, articles_path: Path, hist_path: Path, output_zip: Path):
    """
    Generates official predictions.txt for EB-NeRD Codabench (Competition 2469, Phase 2).
    """
    print(f"Generating EB-NeRD predictions from {eb_test_path}...")
    if not eb_test_path.exists() or not articles_path.exists():
        print(f"Error: EB-NeRD test files ({eb_test_path}) or articles ({articles_path}) not found.")
        return

    print("1. Loading EB-NeRD articles for BM25...")
    articles_df = pl.read_parquet(articles_path)
    article_dict = {}
    
    sub_col = "abstract" if "abstract" in articles_df.columns else "subtitle" if "subtitle" in articles_df.columns else None
    
    for row in articles_df.select(["article_id", "title", sub_col] if sub_col else ["article_id", "title"]).iter_rows():
        art_id = str(row[0])
        title = row[1] or ""
        sub = row[2] or "" if sub_col else ""
        tokens = (title + " " + sub).lower().split()
        article_dict[art_id] = tokens
        
    print(f"Fitting FastBM25 on {len(article_dict)} EB-NeRD articles...")
    bm25 = FastBM25()
    bm25.fit(article_dict)
    
    print("2. Loading user history mapping...")
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
            
    print(f"User history loaded for {len(user_history_map)} users.")

    pred_txt_path = Path("predictions.txt")
    print(f"3. Scoring test impressions to {pred_txt_path}...")
    
    test_df = pl.read_parquet(eb_test_path)
    inview_col = "article_ids_inview" if "article_ids_inview" in test_df.columns else "impressions"
    
    count = 0
    with open(pred_txt_path, "w", encoding="utf-8") as f_out:
        for row in test_df.select(["impression_id", "user_id", inview_col]).iter_rows():
            impr_id = row[0]
            uid = str(row[1]) if row[1] is not None else ""
            raw_cands = row[2]
            
            if isinstance(raw_cands, list) or isinstance(raw_cands, np.ndarray):
                cand_ids = [str(x) for x in raw_cands]
            elif isinstance(raw_cands, str):
                cand_ids = [str(x) for x in raw_cands.split()]
            else:
                continue
                
            if not cand_ids:
                continue
                
            history_ids = user_history_map.get(uid, [])
            query_tokens = []
            for hid in history_ids[-10:]:
                if hid in article_dict:
                    query_tokens.extend(article_dict[hid])
                    
            scores = bm25.score_candidates(query_tokens, cand_ids)
            scores_arr = np.array(scores, dtype=np.float32)
            
            ranks = rankdata(-scores_arr, method='ordinal').astype(int).tolist()
            rank_str = ",".join(map(str, ranks))
            f_out.write(f"{impr_id} [{rank_str}]\n")
            
            count += 1
            if count % 250000 == 0:
                print(f"Processed {count} EB-NeRD impressions...")
                
    print(f"Finished {count} impressions. Zipping submission to {output_zip}...")
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(pred_txt_path, arcname="predictions.txt")
        
    if pred_txt_path.exists():
        pred_txt_path.unlink()
        
    print(f"Successfully generated {output_zip} (ready for Codabench EB-NeRD upload)!")

def main():
    eb_articles_path = Path("data/processed/ebnerd/articles.parquet")
    eb_out_zip = Path("ebnerd_submission.zip")
    
    # Check Official Test Set first (impression ID 6451339)
    test_paths = [
        Path("data/raw/ebnerd/test/ebnerd_testset/test/behaviors.parquet"),
        Path("data/raw/ebnerd/test/test/behaviors.parquet"),
        Path("data/raw/ebnerd/test/behaviors.parquet"),
        Path("data/raw/ebnerd/large/validation/behaviors.parquet"),
    ]
    hist_paths = [
        Path("data/raw/ebnerd/test/ebnerd_testset/test/history.parquet"),
        Path("data/raw/ebnerd/test/test/history.parquet"),
        Path("data/raw/ebnerd/test/history.parquet"),
        Path("data/raw/ebnerd/large/validation/history.parquet"),
    ]
    
    eb_test_path = None
    eb_hist_path = None
    for p in test_paths:
        if p.exists():
            eb_test_path = p
            break
            
    for p in hist_paths:
        if p.exists():
            eb_hist_path = p
            break
            
    if eb_test_path and eb_hist_path and eb_articles_path.exists():
        print(f"Using test set: {eb_test_path}")
        print(f"Using history file: {eb_hist_path}")
        generate_ebnerd_predictions(eb_test_path, eb_articles_path, eb_hist_path, eb_out_zip)
    else:
        print(f"Required test files not found. (eb_test_path={eb_test_path}, eb_hist_path={eb_hist_path})")

if __name__ == "__main__":
    main()
