# Scientific Evidence Retrieval Engine

An end-to-end scientific **information retrieval** project: BM25, transformer embeddings, reciprocal-rank fusion, a **fine-tuned PyTorch cross-encoder**, and neural sentence ranking. It searches **5,183 SciFact research abstracts** and displays the original source passages relevant to a query.

> **Scope:** This system discovers *candidate evidence*. It does **not** verify that a claim is true or search the live scientific literature. The corpus is fixed and limited to abstracts.

## What you can try

The Streamlit demo has a text box and three built-in examples, including:

> CRISPR systems preferentially acquire spacers from certain phage genome regions.

The output contains the top three papers, with two source sentences for each paper and stable corpus IDs such as `D26996935:S6`. Papers 2 and 3 can be expanded if desired. The interface has **no PDF upload, LLM, or API-key requirement**.

A [recorded example result](examples/observed_crispr_result.json) lets you inspect an earlier local run without installing anything.

## Retrieval architecture

```text
Query
  ├── BM25 over title + abstract
  └── all-MiniLM-L6-v2 semantic retrieval
          │
          ▼
  Reciprocal Rank Fusion (RRF), top 20
          │
          ▼
  SciFact-fine-tuned cross-encoder reranker, top 3
          │
          ▼
  Pretrained cross-encoder sentence ranking, top 2 per paper
          │
          ▼
  Ranked research abstracts + exact source sentences
```

The training code uses a custom PyTorch training loop. Corpus embeddings are persisted to a validated local NumPy cache, and model instances are cached within the running Streamlit process.

## Run locally

Use Python **3.10 or newer**. On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**1. Install the SciFact corpus from its [official dataset source](https://github.com/allenai/scifact):**

```bash
python scripts/download_scifact.py
```

This creates `data/corpus.jsonl`, along with other official SciFact files. No training labels are used to answer search queries.

**2. Download the trained checkpoint** `scifact-reranker-v1.0.tar.gz` from the **v1.0 GitHub Release assets** (published separately from the source repository). Place the downloaded archive in Downloads, then install it:

```bash
python scripts/install_checkpoint.py ~/Downloads/scifact-reranker-v1.0.tar.gz
```

The installer checks its SHA-256 fingerprint, validates the model files, refuses symlink targets, and never silently overwrites an existing checkpoint.

Expected SHA-256:

```text
fb47bab93e50fad6368e26b2cdd3ba0a046cae3e5b47abe6daa0a85c6f8ce458
```

**3. Check the installation and run the interface:**

```bash
python scripts/check_setup.py
python -m streamlit run app_retrieval.py
```

The first run downloads the public pretrained encoder and sentence-ranker from Hugging Face and builds the document embedding cache. After this one-time setup, restarts reuse the cache.

**Command-line inference/profiling** (with document IDs, selected sentence indices, cache hits, and timings):

```bash
python profile_retrieval.py --claim "Synaptic stimulation triggers BDNF release from dendritic spines."
```

No live demo URL is required to use the repository. A hosted Streamlit link can be added later.

## Experimental results

On the **188 SciFact development claims that contain annotated evidence** (not the entire 300-claim dev set):

| Metric | Result |
|---|---:|
| BM25 mean document Recall@5 | 0.8848 |
| Hybrid BM25 + semantic mean document Recall@5 | 0.8914 |
| Hybrid + fine-tuned reranker mean document Recall@5 | **0.9239** |
| Document hit within top 3 | **0.9309** |
| Complete annotated evidence-set hit within top 3 papers × 2 sentences | **0.7394** |

The reranked top-20 candidate pool is **not expanded** by reranking; performance at top 20 is limited by initial retrieval. These are SciFact annotation-based scores, not measurements of scientific truth. See [evaluation methodology and source files](docs/EVALUATION.md).

One local **warm-cache** profile on an Apple laptop measured **5.90 seconds** for total process initialization and one query, including **0.44 seconds** for the query itself. Timing is hardware/environment dependent and not a service-level guarantee.

## Research code and tests

- `bm25_retriever.py`, `semantic_retriever.py`, `hybrid_retriever.py`: retrieval baselines and fusion
- `cached_hybrid_retriever.py`, `retrieval_embedding_cache.py`: persistent embedding cache
- `prepare_training_data.py`, `fine_tune_reranker.py`: data construction and PyTorch fine-tuning
- `evidence_pipeline.py`, `rerank_evidence.py`: document and sentence ranking
- `evaluate_*.py`, `compare_rerankers.py`: reproducible SciFact evaluations
- `docs/*.csv`: saved evaluation outputs used for the reported summary metrics

Run the offline unit tests:

```bash
python -m unittest -v \
  test_retrieval_embedding_cache \
  test_cached_hybrid_retriever \
  tests.test_release_tools \
  tests.test_setup_contract
```

For a real local smoke test after installing data and weights:

```bash
python profile_retrieval.py
```

## Limitations

SciFact does **not** supply exhaustive annotations for every possibly relevant research paper. In particular, unannotated papers should not automatically be considered irrelevant. Data preparation uses **presumed negatives** for unannotated documents, and evaluation scores should be interpreted with that limitation in mind. The trained reranker is optimized for relevance, not truth verification.

The interface relies on exact SciFact sentence text and zero-based indices; it does not provide external article URLs or current medical advice.

## Dataset and attribution

SciFact: Wadden et al., *Fact or Fiction: Verifying Scientific Claims* (2020). Dataset and source: <https://github.com/allenai/scifact>. The pretrained models are fetched through the Sentence Transformers library; the fine-tuned reranker weights are this project's model artifact.

The source archive does not include model weights or datasets, and the release asset must be uploaded separately to GitHub Releases for the public installation process to work.

## Troubleshooting a fresh installation

**Initial search is slow:** First run downloads pretrained text encoders and
computes embeddings for all 5,183 abstracts. Later runs reuse the validated
`.cache/semantic_index` embedding cache. Search after initialization typically
takes much less time than the first cold run.

**Repeated `ModuleNotFoundError: No module named 'torchvision'` in the terminal:**
In some environments Streamlit's file watcher introspects Hugging Face
Transformers' *optional image-processing* modules. This project is text-only;
`torchvision` is not required. The shipped `.streamlit/config.toml` disables
that watcher. Restart Streamlit after installing or editing dependencies.
Since file watching is disabled, Python source edits require a manual restart.

**Unexpected Transformers/Sentence Transformers versions:** The supported
package ranges are in `requirements.txt`. If you previously upgraded them
separately, the setup checker flags unsupported major versions. With the
repository virtual environment activated, run:

```bash
python -m pip install -r requirements.txt
python -m pip check
python scripts/check_setup.py
```

To inspect the active environment: `python -m pip --version` and
`python -c 'import sys; print(sys.executable)'`. Both should refer to the
same virtual environment, not a global Python installation. Do not install
`torchvision` to silence the watcher warnings in this text-only project.
