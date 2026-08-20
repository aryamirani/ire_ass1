# Design Note: Lexical & Semantic Retrieval on EB-NeRD and MIND

## 1. System Architecture & Key Design Choices

The recommendation pipeline was designed focusing on speed, reproducibility, and avoiding data leakage.

**Data Processing Stack:** We chose **Polars** over Pandas. Because the total uncompressed interaction logs exceed 10+ million rows, Polars' lazy evaluation and multi-threaded execution allow us to parse, clean, and write the unified schema to disk (Parquet) significantly faster than Pandas.
- **Temporal Splitting:** To avoid "future-click leakage", the behaviors dataset was strictly sorted by the `time` column. The split was done purely chronologically (last 10% for test, preceding 10% for validation), guaranteeing no information from the future leaks into the user history feature store.

**Lexical Retrieval:** We implemented BM25 using `rank_bm25`. The query was constructed by concatenating the raw titles and abstracts of the most recently clicked articles from the user's history. This creates a dense pseudo-query representing their lexical interests.

**Semantic Retrieval:** We utilized **FAISS** (Facebook AI Similarity Search) with an `IndexFlatIP` (Inner Product) index.
- Assuming the provided BERT/Word2Vec embeddings are L2 normalized, Inner Product perfectly represents Cosine Similarity.
- The user representation is generated via **Mean-Pooling** the embeddings of their historically clicked articles. This approach is computationally cheap and provides a robust baseline for semantic similarity.

## 2. Alternatives Considered

1. **TF-IDF vs BM25:** We chose BM25 over basic TF-IDF for Lexical Retrieval because BM25 incorporates term-frequency saturation and document length normalization, making it strictly superior for variable-length news abstracts.
2. **ScaNN vs FAISS:** While ScaNN can be slightly faster for large-scale Maximum Inner Product Search (MIPS) due to anisotropic vector quantization, FAISS was chosen because its CPU implementation is more widely supported, easier to install across environments, and highly performant for datasets of this scale (~160K vectors).
3. **RNN/GRU User Representations:** Instead of mean-pooling historical embeddings, we considered using a sequential model (GRU) to encode the user's history. We chose mean-pooling because it is a parameter-free baseline that fits within the hardware and time constraints of a purely retrieval-focused assignment.

## 3. Observations & Experimental Results

- **Lexical vs Semantic:** Semantic retrieval (Embeddings) generally outperforms Lexical (BM25) on `Recall@K` because news recommendations heavily rely on synonyms and thematic context. BM25 struggles with "vocabulary mismatch" (e.g. if a user reads about "puppies" and the candidate article says "dogs").
- **Cold-Start Slices:** When slicing the metrics by history length, Semantic retrieval maintains better performance on warm users (long history) because the mean-pooled vector becomes more robust. However, for extremely cold-start users (1-2 clicks), lexical retrieval can sometimes provide a highly specific (albeit narrow) match that prevents the user vector from becoming too generic.
- **Dataset Differences:** The English MIND dataset exhibited slightly better semantic alignment with the standard BERT models compared to the Danish EB-NeRD dataset, largely due to pre-training language bias in generic multilingual embeddings.

## 4. Scaling to 10x

If the dataset scaled by $10\times$ (e.g. going from 1M to 10M users, and 120K to 1.2M articles):
1. **Memory Exhaustion (FAISS):** A flat inner product index (`IndexFlatIP`) requires storing all $1.2M$ dense vectors in RAM. At $10\times$ scale, this exhaustive search becomes a bottleneck. We would need to switch to an Approximate Nearest Neighbor index with Voronoi cells and quantization (e.g., `IndexIVFPQ` in FAISS).
2. **Data Processing:** Polars would handle $10\times$ data adequately due to out-of-core capabilities, but the final Parquet files would exceed standard laptop memory limits when loaded concurrently.
3. **Inverted Index Size:** The BM25 index built purely in python (`rank_bm25`) would become prohibitively slow to query sequentially. We would need to migrate to a scalable search engine like **Elasticsearch** or **OpenSearch** to handle distributed lexical queries.
