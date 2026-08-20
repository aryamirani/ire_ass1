import polars as pl
import argparse
import os

def create_mind_submission(predictions_dict, out_path="prediction.txt"):
    """
    Format required for MIND Codabench:
    Impression_ID [space] [Ranking of articles]
    e.g., 1 [1, 2, 4, 3, 5]
    """
    with open(out_path, "w") as f:
        for imp_id, rank_list in predictions_dict.items():
            rank_str = ",".join(map(str, rank_list))
            f.write(f"{imp_id} [{rank_str}]\n")

def create_ebnerd_submission(predictions_dict, out_path="predictions.txt"):
    """
    Format required for EB-NeRD Codabench (usually similar)
    """
    with open(out_path, "w") as f:
        for imp_id, rank_list in predictions_dict.items():
            rank_str = ",".join(map(str, rank_list))
            f.write(f"{imp_id} [{rank_str}]\n")

def main():
    print("This script will generate prediction files for Codabench.")
    print("Ensure you have run the models on the test_interactions and captured the predictions.")
    
    # Placeholder: Assuming you run your models and output a dictionary of impression_id -> [ranks]
    # predictions = {1234: [1, 2, 3, 4, 5], 5678: [5, 4, 3, 2, 1]}
    
    # create_mind_submission(predictions, "mind_prediction.txt")
    # os.system("zip submission.zip mind_prediction.txt")
    pass

if __name__ == "__main__":
    main()
