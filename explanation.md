# Information Retrieval & Extraction: Assignment 1 Guide

Welcome to the CS4.406 Information Retrieval & Extraction Assignment 1! This document explains the assignment from the very basics to the level required to complete it successfully. 

## 1. The High-Level Goal
Imagine you're visiting a news website. Out of thousands of articles available, which ones should the website show you right now? A **News Recommendation System** tries to solve this by ranking articles based on how likely you are to click on them. 

To achieve this, you use three main pieces of information:
1. **User's Click History:** What articles has this user read recently?
2. **Session Context:** Is it morning or evening? Are they browsing on mobile or desktop?
3. **Article Content:** What is the article about (title, abstract, text)?

This assignment specifically focuses on **Candidate Generation (Retrieval)**. When a user opens the app, you don't evaluate all 120,000+ articles. Instead, you quickly filter down to the top-$K$ (e.g., top 100) most relevant articles. You will implement two ways to retrieve these candidates:
- **Lexical Retrieval:** Based on exact word matches (BM25).
- **Semantic Retrieval:** Based on meaning and context (Embeddings & Vector Search).

You will run these systems on two real-world datasets:
- **MIND (Microsoft News Dataset):** English news dataset.
- **EB-NeRD (Ekstra Bladet):** Danish news dataset.

---

## 2. Breaking Down the Task

You need to build 5 main components. Let's look at each one in detail.

### Q1. Reproducible Data Pipeline
**The Goal:** Write a single script (like `make data` or `python build_pipeline.py`) that downloads the raw data, cleans it, and organizes it so it's ready for your models to use.

- **Download:** Fetch the zip files for MIND-large and EB-NeRD-large.
- **Clean and Parse:** The raw data comes in CSV or TSV files. You'll need to use a library like `pandas` or `polars` to load this data, handle any missing values, and structure it uniformly (e.g., standardizing column names across both datasets).
- **Temporal Split:** In real life, you predict the *future* based on the *past*. Therefore, you must split your data by time (e.g., train on days 1-14, validate on day 15, test on day 16). **Never split interaction data randomly**, as this causes "data leakage" (peeking into the future).
- **Feature Store:** Save the processed data in a fast, reusable format (like `.parquet` files). You should store:
  - Article features: title, abstract, body, embeddings.
  - User features: their click history, recency of clicks.

### Q2. Lexical Candidate Generation (BM25)
**The Goal:** Find relevant articles by matching the words in the user's past clicks with the words in new candidate articles. 

- **BM25 Algorithm:** This is a classic search algorithm (an improvement over TF-IDF). It ranks documents based on how often search terms appear in them, while penalizing overly common words (like "the" or "is").
- **How to do it:**
  1. Build an "inverted index" over the article texts (combining title + abstract). You can use libraries like `rank_bm25` in Python.
  2. For a given user, take the titles of the articles they clicked recently and combine them into a "search query".
  3. Run this query through your BM25 index to retrieve the top-$K$ articles.
  4. Evaluate: Calculate `Recall@K`. If the user *actually* clicked an article in the future, was it in your top 50, 100, or 200 retrieved articles?

### Q3. Semantic Candidate Generation (Embeddings)
**The Goal:** Find relevant articles based on their *meaning*, even if they don't share the exact same words (e.g., "puppy" and "dog").

- **Embeddings:** These are dense mathematical vectors (arrays of numbers) that represent the meaning of text. The datasets already provide pre-trained embeddings for the articles (or you can generate them using BERT/XLM-RoBERTa).
- **How to do it:**
  1. Build an Approximate Nearest Neighbor (ANN) index. Libraries like `FAISS` or `ScaNN` allow you to quickly search through millions of vectors to find the closest ones.
  2. For a user, average out (mean-pool) the embeddings of the articles they clicked on. This single vector represents the user's "interest profile".
  3. Query your ANN index with this user profile vector to find the top-$K$ closest article vectors.
  4. Evaluate with `Recall@K` and compare the results with your BM25 approach.

### Q4. Offline Evaluation Harness
**The Goal:** Write code to measure how good your recommendations are, beyond just simple accuracy.

- **Accuracy Metrics:** Implement AUC (Area Under the ROC Curve), MRR (Mean Reciprocal Rank), and nDCG@5/10 (Normalized Discounted Cumulative Gain). Libraries like `scikit-learn` have functions for AUC, but you might need to write or find implementations for MRR and nDCG for ranking.
- **Beyond Accuracy:** 
  - *Diversity:* Are the top-$K$ articles too similar to each other?
  - *Novelty:* Are you recommending articles the user hasn't seen or that aren't globally obvious?
  - *Coverage:* Out of all available articles, what percentage does your system actually recommend to people?
- **Slicing:** Test your metrics on specific subgroups to see where your model struggles. E.g., Warm users (lots of history) vs. Cold-start users (new users with few clicks).
- **Confidence Intervals:** Use "Bootstrapping" (randomly resampling your evaluation data many times) to calculate 95% confidence intervals, proving your metric differences aren't just due to random chance.

### Q5 & Q6. Submission & Design Note
- **Codabench:** You must format your predictions exactly as the competition leaderboards expect and submit them online.
- **Design Note (Max 4 pages):** A short report explaining your architecture. Why did you choose Polars over Pandas? Why FAISS? Did BM25 beat Semantic Search, and why? What would break if the dataset was 10 times larger?

---

## 3. Recommended Tech Stack
To build this efficiently, here are the standard Python libraries you should use:
- **Data processing:** `polars` (highly recommended for speed) or `pandas`.
- **Lexical Search (BM25):** `rank_bm25` or `pyserini`.
- **Semantic Search (ANN):** `faiss-cpu` or `scann`.
- **Machine Learning / Metrics:** `scikit-learn`, `numpy`.
- **Progress Tracking:** `tqdm` (to see how long loops take).

## 4. Next Steps to Start Coding
1. **Set up your environment:** Create a virtual environment (`python -m venv venv`) and install the libraries mentioned above.
2. **Download the Data:** Run the bash scripts to download the MIND-large and EB-NeRD-large datasets.
3. **Explore:** Open a Jupyter Notebook and just try to load one of the CSV/Parquet files to see what the columns look like. 
4. **Start Q1:** Write the script that cleans the data and splits it by time.

Good luck! This is an excellent, real-world machine learning engineering task.
