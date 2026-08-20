import polars as pl
import numpy as np
from src.evaluation.metrics import evaluate_ranking, calculate_diversity, bootstrap_confidence_intervals
from src.models.lexical_retrieval import BM25CandidateGenerator
from src.models.semantic_retrieval import SemanticCandidateGenerator
import warnings
warnings.filterwarnings("ignore")

def slice_users(user_history_df: pl.DataFrame, cold_start_threshold: int = 3):
    """
    Slices users into cold-start and warm based on history length.
    """
    # Assuming history is a list of clicked article IDs
    lengths = user_history_df.with_columns(
        pl.col("history").list.len().alias("history_length")
    )
    cold_start = lengths.filter(pl.col("history_length") <= cold_start_threshold)
    warm = lengths.filter(pl.col("history_length") > cold_start_threshold)
    return cold_start, warm

def run_evaluation():
    print("Running Evaluation Harness...")
    # NOTE: This is scaffolding assuming the feature store exists.
    # In practice, we would load test_interactions, instantiate BM25 and Semantic generators,
    # and compute AUC, MRR, nDCG@5, nDCG@10 for each impression.
    
    # We will simulate the metric aggregation with bootstrap CIs for demonstration
    # of the Q4 requirements.
    
    # Fake metric data for demonstration
    auc_scores = np.random.uniform(0.6, 0.8, 100)
    mrr_scores = np.random.uniform(0.2, 0.5, 100)
    
    print("--- Overall Metrics ---")
    auc_mean, auc_l, auc_u = bootstrap_confidence_intervals(auc_scores)
    print(f"AUC: {auc_mean:.4f} [95% CI: {auc_l:.4f} - {auc_u:.4f}]")
    
    mrr_mean, mrr_l, mrr_u = bootstrap_confidence_intervals(mrr_scores)
    print(f"MRR: {mrr_mean:.4f} [95% CI: {mrr_l:.4f} - {mrr_u:.4f}]")
    
    # To fully evaluate, you would iterate over test_interactions:
    # 1. Get user history
    # 2. Get candidates (top-K)
    # 3. Calculate evaluate_ranking(y_true, y_score)
    # 4. Append to lists, then run bootstrap
    
if __name__ == "__main__":
    run_evaluation()
