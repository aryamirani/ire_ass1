import polars as pl
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict

class BM25CandidateGenerator:
    def __init__(self):
        self.bm25 = None
        self.article_ids = []
        
    def fit(self, articles_df: pl.DataFrame):
        """
        Builds the BM25 index over the article texts.
        Expects articles_df to have 'article_id', 'title', and 'abstract' columns.
        """
        print("Building BM25 index...")
        # Combine title and abstract for the document text
        # Fill nulls with empty string
        text_series = (
            articles_df.get_column("title").fill_null("") + " " + 
            articles_df.get_column("abstract").fill_null("")
        )
        
        # Optionally append the full article body if it exists (e.g., EB-NeRD dataset)
        if "body" in articles_df.columns:
            text_series = text_series + " " + articles_df.get_column("body").fill_null("")
            
        texts = text_series.to_list()
        
        self.article_ids = articles_df.get_column("article_id").to_list()
        
        # Tokenize (simple whitespace tokenization and lowercasing for BM25)
        tokenized_corpus = [doc.lower().split() for doc in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 index built.")

    def construct_query(self, click_history_titles: List[str]) -> List[str]:
        """
        Constructs a query from the user's click history.
        We concatenate the recent titles and tokenize.
        """
        query = " ".join(click_history_titles)
        return query.lower().split()

    def retrieve(self, query_tokens: List[str], top_k: int = 100) -> List[str]:
        """
        Retrieves top_k article IDs for the given query tokens.
        """
        if not query_tokens:
            return []
            
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        
        # Map indices back to article_ids
        return [self.article_ids[i] for i in top_k_indices]
        
    def evaluate_recall(self, retrieved_ids: List[str], ground_truth_ids: List[str], k_values=[50, 100, 200]) -> Dict[int, float]:
        """
        Calculates Recall@K.
        """
        results = {}
        for k in k_values:
            retrieved_at_k = set(retrieved_ids[:k])
            ground_truth_set = set(ground_truth_ids)
            
            if not ground_truth_set:
                results[k] = 0.0
                continue
                
            hits = len(retrieved_at_k.intersection(ground_truth_set))
            results[k] = hits / len(ground_truth_set)
            
        return results
