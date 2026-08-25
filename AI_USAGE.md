# AI Usage Log

In compliance with the assignment policy (Q7.4 / Deliverables & Policies), this document details all AI assistant interactions, prompting strategies, and explicit markings of AI-generated versus human-authored codebase components.

---

## 1. Tools & Platforms Used
- **Agent / Assistant**: Google DeepMind Antigravity Agent (Gemini 2.0 Thinking Model)
- **IDE / Environment**: Antigravity IDE & Terminal Interface

---

## 2. Chronological Prompt & Interaction History

### Phase 1: Environment & Baseline Pipeline Setup
- **Prompt**: *"help me solve this assignment"*
- **Agent Action**: Inspected `A1.pdf`, downloaded MIND and EB-NeRD demo datasets, set up unified Polars data parsing (`src/data/build_pipeline.py`) with strict temporal splitting to prevent future-click leakage.

### Phase 2: Lexical & Semantic Retrieval Implementation
- **Prompt**: *"build lexical candidate generation with BM25 and semantic candidate generation with FAISS embeddings"*
- **Agent Action**: Created `BM25CandidateGenerator` (`src/models/lexical_retrieval.py`) over concatenated titles and abstracts, and `SemanticCandidateGenerator` (`src/models/semantic_retrieval.py`) utilizing `faiss.IndexFlatIP` with mean-pooled user click histories.

### Phase 3: Offline Evaluation Harness & Slicing
- **Prompt**: *"implement official evaluation metrics (AUC, MRR, nDCG@5, nDCG@10), intra-list diversity, cold vs warm user slicing, and 95% bootstrap confidence intervals"*
- **Agent Action**: Created `src/evaluation/metrics.py` and `src/evaluation/evaluate.py` with 1,000 bootstrap iterations and cold-start slicing.

### Phase 4: Cluster Scaling & SLURM Automation (Ada Cluster)
- **Prompt**: *"sync to ada cluster and run jobs on large dataset"*
- **Agent Action**: Created `sync_to_ada.sh`, `scripts/download_data.slurm`, and `scripts/run_ebnerd.slurm`. Optimized prediction generation using Polars memory mapping and fast lookup structures.

### Phase 5: Codabench Submission Debugging & Testset Pipeline
- **Prompt**: *"what? no scpre? i dont know what happened here but it failed"*
- **Agent Action**: Diagnosed Codabench Evaluator error (`Inconsistent Impression ID 96429 and 6451339`). Determined that Phase 1 (RecSys '24 Challenge) was closed, and Phase 2 (`EB-NeRD official testset`) expects the 13.5M official test set starting at Impression ID `6451339`.
- **Prompt**: *"is it done downlaoding? is anything downloading locally? i wanna change networks"*
- **Agent Action**: Moved downloads and processing 100% into background SLURM jobs (`scripts/download_testset.slurm` with 16-thread chunk downloader `scripts/fast_download.py` and `scripts/run_testset_pipeline.slurm`).

### Phase 6: Testset Assembly & Submission
- **Prompt**: *"done? whats taking so long? we need a proper zip"*
- **Agent Action**: Streamed, merged, and verified `ebnerd_submission.zip` (13,536,710 predictions starting at ID `6451339`). User submitted and obtained official leaderboard score.

---

## 3. Marking of AI-Generated vs. Human-Authored Code

| Component / File | Author / Origin | Description |
| :--- | :--- | :--- |
| `src/data/build_pipeline.py` | AI-generated with human review | Ingestion, Polars data schema unification, temporal splitting |
| `src/models/lexical_retrieval.py` | AI-generated | BM25 indexing, query concatenation from click history, Recall@K |
| `src/models/semantic_retrieval.py` | AI-generated | FAISS inner product index, mean-pooled user embeddings, Recall@K |
| `src/evaluation/metrics.py` | AI-generated | Ranking metrics (AUC, MRR, nDCG@5, nDCG@10), Intra-list diversity, bootstrap 95% CIs |
| `src/evaluation/evaluate.py` | AI-generated | End-to-end evaluation harness and cold vs warm slicing |
| `src/scripts/generate_submission.py` | AI-generated with human review | Formatting and compression for Codabench leaderboards |
| `scripts/fast_download.py` | AI-generated | 16-thread parallel chunk downloader using HTTP Range headers |
| `scripts/*.slurm` | AI-generated | SLURM batch scripts for Ada HPC compute nodes |
| `tests/test_pipeline.py` | AI-generated | Anti-gaming boundary assertion and unit tests for metrics and generators |
| `design_note.md` | AI-drafted with human analysis | System design, choices, observations, and 10x scaling breakdown |
| `AI_USAGE.md` | AI-generated | Transparency log of all interactions and prompts |
