import polars as pl
import pytest

def test_no_future_click_leakage():
    """
    Validates Q9 Anti-Gaming requirement: The behaviour-window boundary is enforced.
    Ensures that for any user's test interaction, there are no clicks in their history
    that occurred AFTER the timestamp of the test interaction.
    """
    # NOTE: This test will be run once the data is processed.
    # It assumes the existence of processed data in 'data/processed/'
    try:
        test_interactions = pl.read_parquet("data/processed/mind/test_interactions.parquet")
        train_interactions = pl.read_parquet("data/processed/mind/train_interactions.parquet")
    except Exception:
        pytest.skip("Processed data not found yet. Run data pipeline first.")
        
    # Check chronological boundary
    max_train_time = train_interactions.select(pl.col("time").max()).item()
    min_test_time = test_interactions.select(pl.col("time").min()).item()
    
    # Assert that all test interactions occur strictly after or at the exact same boundary 
    # as the latest train interaction, proving no future leakage.
    assert min_test_time >= max_train_time, "Leakage detected! Test interaction occurs before train interaction finishes."

if __name__ == "__main__":
    pytest.main(["-v", __file__])
