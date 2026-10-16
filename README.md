# Scientific Evidence Retrieval Engine

Search and rank scientific research using **BM25, semantic embeddings, and a fine-tuned PyTorch reranker**. The Streamlit app searches **5,183 SciFact abstracts** and shows three ranked papers with two relevant source sentences per paper.

## How it works

1. **Retrieve:** BM25 and `all-MiniLM-L6-v2` semantic search.
2. **Combine:** Reciprocal Rank Fusion selects 20 candidate papers.
3. **Rerank:** A SciFact-fine-tuned cross-encoder selects the top 3 papers.
4. **Select evidence:** A neural sentence ranker extracts 2 sentences per paper.

Corpus embeddings are stored in a persistent cache for faster subsequent runs.

## Results

Evaluated on **188 evidence-annotated SciFact development claims**:

| Metric | Score |
|---|---:|
| BM25 document Recall@5 | 0.8848 |
| Hybrid retrieval document Recall@5 | 0.8914 |
| Hybrid + fine-tuned reranker document Recall@5 | **0.9239** |
| Complete evidence-set hit rate (top 3 papers, 2 sentences each) | **0.7394** |

See [evaluation details](docs/EVALUATION.md).

## Run locally

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_scifact.py
```

Download `scifact-reranker-v1.0.tar.gz` from the repository's **v1.0 Release** and save it to `~/Downloads`, then run:

```bash
python scripts/install_checkpoint.py ~/Downloads/scifact-reranker-v1.0.tar.gz
python scripts/check_setup.py
python -m streamlit run app_retrieval.py
```

The app provides sample queries and an editable search box. For command-line inference and timings:

```bash
python profile_retrieval.py
```

## Training and tests

- `fine_tune_reranker.py` — PyTorch reranker training.
- `evaluate_*.py` — retrieval and evidence-ranking evaluation.
- `test_*.py` and `tests/` — cache and release tests.

## Data

[SciFact](https://github.com/allenai/scifact) — Wadden et al., *Fact or Fiction: Verifying Scientific Claims* (2020).
