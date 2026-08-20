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
    """
    High-performance in-memory BM25 scorer optimized for candidate ranking.
    """
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
        
        # Precompute IDF
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

def generate_mind_predictions(raw_test_dir: Path, articles_path: Path, output_zip: Path):
    print(f"Generating MIND predictions from {raw_test_dir}...")
    test_beh_file = raw_test_dir / "behaviors.tsv"
    
    if not test_beh_file.exists():
        print(f"Error: {test_beh_file} not found.")
        return

    # 1. Load articles
    print("Loading articles for FastBM25...")
    articles_df = pl.read_parquet(articles_path)
    article_dict = {}
    
    for row in articles_df.select(["article_id", "title", "abstract"]).iter_rows():
        art_id, title, abstract = row[0], row[1] or "", row[2] or ""
        tokens = (title + " " + abstract).lower().split()
        article_dict[art_id] = tokens
        
    print(f"Fitting FastBM25 on {len(article_dict)} articles...")
    bm25 = FastBM25()
    bm25.fit(article_dict)
    
    pred_txt_path = Path("prediction.txt")
    print(f"Scoring test impressions and writing to {pred_txt_path}...")
    
    count = 0
    with open(test_beh_file, "r", encoding="utf-8") as f_in, open(pred_txt_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            impr_id = parts[0]
            history_str = parts[3]
            impr_str = parts[4]
            
            history_ids = history_str.split() if history_str else []
            candidate_ids = impr_str.split() if impr_str else []
            
            if not candidate_ids:
                continue
                
            # Construct user query tokens from history
            query_tokens = []
            for hid in history_ids[-10:]:
                if hid in article_dict:
                    query_tokens.extend(article_dict[hid])
                    
            scores = bm25.score_candidates(query_tokens, candidate_ids)
            scores_arr = np.array(scores, dtype=np.float32)
            
            # Rank descending: highest score gets rank 1
            ranks = rankdata(-scores_arr, method='ordinal').astype(int).tolist()
            rank_str = ",".join(map(str, ranks))
            f_out.write(f"{impr_id} [{rank_str}]\n")
            
            count += 1
            if count % 250000 == 0:
                print(f"Processed {count} impressions...")
                
    print(f"Finished {count} impressions. Zipping submission to {output_zip}...")
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(pred_txt_path, arcname="prediction.txt")
        
    if pred_txt_path.exists():
        pred_txt_path.unlink()
        
    print(f"Successfully generated {output_zip} (ready for Codabench upload)!")

def main():
    raw_mind_test = Path("data/raw/mind/test/MINDlarge_test")
    articles_path = Path("data/processed/mind/articles.parquet")
    out_zip = Path("mind_submission.zip")
    
    if raw_mind_test.exists() and articles_path.exists():
        generate_mind_predictions(raw_mind_test, articles_path, out_zip)
    else:
        print("MIND test files or articles.parquet not found yet.")

if __name__ == "__main__":
    main()
