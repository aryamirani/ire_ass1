import numpy as np
from sklearn.metrics import roc_auc_score, ndcg_score
from typing import List, Dict, Tuple

def mrr_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    y_true: binary array of clicks (1 if clicked, 0 otherwise)
    y_score: predicted scores/probabilities
    """
    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order]
    
    # Find the rank of the first relevant item
    for i, rel in enumerate(y_true_sorted):
        if rel == 1:
            return 1.0 / (i + 1)
    return 0.0

def evaluate_ranking(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """
    Computes standard ranking metrics for a single impression/user.
    """
    # If all 0 or all 1, AUC is not defined
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_score)
    else:
        auc = 0.5
        
    mrr = mrr_score(y_true, y_score)
    
    # ndcg requires 2D array
    y_true_2d = np.asarray([y_true])
    y_score_2d = np.asarray([y_score])
    
    ndcg_5 = ndcg_score(y_true_2d, y_score_2d, k=5)
    ndcg_10 = ndcg_score(y_true_2d, y_score_2d, k=10)
    
    return {
        "AUC": auc,
        "MRR": mrr,
        "nDCG@5": ndcg_5,
        "nDCG@10": ndcg_10
    }

def calculate_diversity(recommended_embeddings: List[np.ndarray]) -> float:
    """
    Intra-list diversity based on pairwise cosine distance of embeddings.
    """
    if len(recommended_embeddings) < 2:
        return 0.0
        
    vecs = np.vstack(recommended_embeddings)
    # Cosine similarity matrix
    sim_matrix = np.dot(vecs, vecs.T)
    # Convert similarity to distance (1 - sim)
    dist_matrix = 1.0 - sim_matrix
    
    # Upper triangle excluding diagonal
    iu = np.triu_indices(len(vecs), k=1)
    avg_diversity = np.mean(dist_matrix[iu])
    return avg_diversity

def bootstrap_confidence_intervals(metric_values: List[float], num_samples=1000, ci=95) -> Tuple[float, float, float]:
    """
    Calculates the mean and the bootstrap confidence interval.
    Returns (mean, lower_bound, upper_bound)
    """
    values = np.array(metric_values)
    means = []
    
    for _ in range(num_samples):
        # Sample with replacement
        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
        
    p_lower = (100 - ci) / 2.0
    p_upper = 100 - p_lower
    
    lower_bound = np.percentile(means, p_lower)
    upper_bound = np.percentile(means, p_upper)
    
    return np.mean(values), lower_bound, upper_bound
