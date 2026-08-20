import numpy as np
import faiss
from typing import List, Dict
import polars as pl

class SemanticCandidateGenerator:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        # Inner Product index. Assuming embeddings are L2-normalized, IP is equivalent to Cosine Similarity.
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.article_ids = []
        
    def fit(self, articles_df: pl.DataFrame):
        """
        Builds the FAISS index over article embeddings.
        Expects articles_df to have 'article_id' and 'embedding' columns.
        Embeddings should be a list/array of floats.
        """
        print("Building FAISS index...")
        self.article_ids = articles_df.get_column("article_id").to_list()
        
        # Extract embeddings and convert to numpy matrix
        embeddings_list = articles_df.get_column("embedding").to_list()
        embeddings_matrix = np.vstack(embeddings_list).astype('float32')
        
        # L2 normalize for cosine similarity
        faiss.normalize_L2(embeddings_matrix)
        
        self.index.add(embeddings_matrix)
        print(f"FAISS index built with {self.index.ntotal} vectors.")

    def construct_user_profile(self, history_embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Creates a user representation by mean-pooling the embeddings of articles they've clicked.
        """
        if not history_embeddings:
            return np.zeros(self.embedding_dim, dtype='float32')
            
        history_matrix = np.vstack(history_embeddings)
        mean_embedding = np.mean(history_matrix, axis=0).astype('float32')
        
        return mean_embedding

    def retrieve(self, user_profile: np.ndarray, top_k: int = 100) -> List[str]:
        """
        Retrieves top_k nearest article IDs for a given user profile.
        """
        if np.all(user_profile == 0):
            return []
            
        # FAISS expects a 2D array
        query_vector = user_profile.reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(query_vector, top_k)
        
        # Map indices back to article_ids
        # indices is shape (1, top_k)
        return [self.article_ids[i] for i in indices[0] if i != -1]
        
    def evaluate_recall(self, retrieved_ids: List[str], ground_truth_ids: List[str], k_values=[50, 100, 200]) -> Dict[int, float]:
        """
        Calculates Recall@K. Same logic as Lexical retrieval.
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
