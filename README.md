# Information Retrieval Assignment 1: News Recommendation Systems

This repository contains the complete reproducible pipeline for Lexical and Semantic Retrieval on the MIND and EB-NeRD datasets.

## Directory Structure
- `data/raw/`: Place the raw downloaded zip files here.
- `data/processed/`: Where the feature store (Parquet files) is generated.
- `notebooks/`: Exploratory analysis and baseline walkthrough notebooks (`ebnerd_analysis.ipynb`, `mind_analysis.ipynb`).
- `src/data/`: Data parsing and temporal splitting.
- `src/models/`: Implementation of `BM25CandidateGenerator` and `SemanticCandidateGenerator`.
- `src/evaluation/`: Implementation of evaluation metrics and bootstrapping.
- `src/scripts/`: Codabench submission generator.
- `tests/`: Automated tests (e.g. anti-gaming future-click leakage check).


## Quickstart (One-Command Reproduce)

Ensure you have Python 3.9+ and have installed dependencies:
```bash
pip install -r requirements.txt
```

To run the entire pipeline from raw data to processed feature store and evaluation:
```bash
make data # OR bash setup_data.sh && python src/data/build_pipeline.py
```
*(Note: To strictly reproduce from scratch, run `bash setup_data.sh` to download raw files, then run `python src/data/build_pipeline.py` to create the feature store, and finally execute the models in a notebook or via `src/evaluation/evaluate.py`)*

## Execution Steps for the Pipeline

1. **Build the Feature Store**: `python src/data/build_pipeline.py`
   This script ingests the raw TSV/Parquet files from MIND and EB-NeRD, standardizes the schema, enforces a time-based split (preventing future-click leakage), and outputs optimized Parquet files.

2. **Evaluate Models**: `python src/evaluation/evaluate.py`
   Runs the test dataset through the Lexical (BM25) and Semantic (FAISS) retrievers, calculating AUC, MRR, nDCG@5, nDCG@10, and Intra-list diversity with Bootstrap 95% Confidence Intervals.

3. **Generate Submission**: `python src/scripts/generate_submission.py`
   Generates the format required for the Codabench leaderboards.

4. **Verify Anti-Gaming**: `pytest tests/test_pipeline.py`
   Asserts that the behaviour-window boundary is strictly enforced.
