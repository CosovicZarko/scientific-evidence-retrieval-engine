# Before publishing

1. Upload this **source-only** repository to GitHub. Ensure `data/`, `checkpoints/`, `.cache/`, and reports are ignored.
2. Create a GitHub Release tagged `v1.0.0` and attach `scifact-reranker-v1.0.tar.gz` (80 MB).
3. On a clean clone with Python 3.10+, install `requirements.txt`, run the two asset installers, and run `scripts/check_setup.py`.
4. Run `python profile_retrieval.py` and then `python -m streamlit run app_retrieval.py`.
5. Verify a sample query returns cited SciFact sentences. Capture a screenshot if desired.
6. Only then describe the public download and local demo as verified.

Optional: host Streamlit later. A live URL is not required for a valid source release.

### Clean-install dependency diagnostic (October 2026)

- Verify `.streamlit/config.toml` disables file watching (avoids optional `torchvision` introspection).
- Use `python -m pip install -r requirements.txt`, **not** unconstrained `pip install -U sentence-transformers`.
- Run `python -m pip check` and `python scripts/check_setup.py` inside the fresh `.venv`.
- Confirm a cold run builds the index, a second run hits the persistent cache, and the app completes an actual retrieval query.
