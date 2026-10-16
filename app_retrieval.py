"""Streamlit interface for the SciFact Scientific Evidence Retrieval Engine.

No LLM, truth labels, or PDF ingestion are used in the published demo.
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock

import streamlit as st

from evidence_pipeline import SENTENCE_MODEL_NAME, run_evidence_pipeline

BASE = Path(__file__).resolve().parent
CORPUS = BASE / "data" / "corpus.jsonl"
CHECKPOINT = BASE / "checkpoints" / "scifact_reranker"

EXAMPLE_CLAIMS = [
    "CRISPR systems preferentially acquire spacers from certain phage genome regions.",
    "Synaptic stimulation triggers BDNF release from dendritic spines.",
    "Rifampicin accumulates in necrotic regions of tuberculosis lesions.",
]


@st.cache_resource(show_spinner=False)
def load_resources():
    """Reuse the models and cache-backed index across Streamlit reruns."""
    # Public deployments start without local data or model files.
    # Reuse the project's validated installers on the first search.
    from scripts.prepare_hosted_assets import prepare_assets
    prepare_assets(BASE)

    from bm25_retriever import load_corpus
    from cached_hybrid_retriever import CachedHybridRetriever
    from semantic_retriever import MODEL_NAME
    from sentence_transformers import CrossEncoder, SentenceTransformer

    documents = load_corpus(str(CORPUS))
    documents_by_id = {int(doc["doc_id"]): doc for doc in documents}
    if len(documents_by_id) != len(documents):
        raise ValueError("Duplicate corpus document IDs")

    encoder = SentenceTransformer(MODEL_NAME)
    hybrid = CachedHybridRetriever(
        documents,
        encoder,
        cache_dir=BASE / ".cache" / "semantic_index",
        encoder_key=MODEL_NAME,
        encoder_revision="default",
        candidate_k=100,
        rrf_constant=60,
    )

    return {
        "documents": documents_by_id,
        "hybrid": hybrid,
        "document_reranker": CrossEncoder(str(CHECKPOINT.resolve())),
        "sentence_reranker": CrossEncoder(SENTENCE_MODEL_NAME),
        "lock": Lock(),
    }


def search_claim(query: str, resources: dict) -> list[dict]:
    """Use the original retrieval implementation; no new ranking logic."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Enter a scientific question or claim")
    with resources["lock"]:
        return run_evidence_pipeline(
            query.strip(),
            resources["documents"],
            resources["hybrid"],
            resources["document_reranker"],
            resources["sentence_reranker"],
            batch_size=16,
        )


def render_results(papers: list[dict]) -> None:
    if not papers:
        st.info("No candidate abstracts found for this query.")
        return
    st.subheader("Retrieved research evidence")
    st.caption(
        "Ranked candidate passages from the SciFact corpus. "
        "These are not verified scientific conclusions."
    )
    for paper in papers:
        rank = int(paper["rank"])
        title = str(paper["title"])
        with st.expander(f"{rank}. {title}", expanded=(rank == 1)):
            st.caption(f"SciFact document ID: {paper['doc_id']}")
            for sentence in paper["sentences"]:
                citation = f"D{paper['doc_id']}:S{sentence['index']}"
                st.markdown(f"**Source {citation}**")
                st.write(sentence["text"])
            st.caption("Original abstract excerpts. Sentence indices are zero-based.")


def main() -> None:
    st.set_page_config(page_title="Scientific Evidence Retrieval Engine", layout="wide")
    st.title("Scientific Evidence Retrieval Engine")
    st.write(
        "Explore scientific research using BM25, semantic retrieval, "
        "a fine-tuned PyTorch reranker, and neural sentence selection."
    )
    st.caption(
        "Research demo searching 5,183 SciFact abstracts. "
        "It does not determine whether a claim is true."
    )

    if "claim_text" not in st.session_state:
        st.session_state["claim_text"] = EXAMPLE_CLAIMS[0]

    def choose_example():
        selected = st.session_state["example_choice"]
        if selected != "Custom query":
            st.session_state["claim_text"] = selected

    st.selectbox(
        "Example queries",
        EXAMPLE_CLAIMS + ["Custom query"],
        index=0,
        key="example_choice",
        on_change=choose_example,
    )

    with st.form("retrieval_form"):
        claim = st.text_area("Scientific claim or research query", key="claim_text", height=100, max_chars=2000)
        submitted = st.form_submit_button("Search research evidence", type="primary")

    if submitted:
        st.session_state.pop("search_result", None)
        if not claim.strip():
            st.warning("Enter a scientific claim or research query.")
        else:
            try:
                with st.spinner("Searching SciFact research abstracts..."):
                    resources = load_resources()
                    papers = search_claim(claim, resources)
                st.session_state["search_result"] = {"query": claim.strip(), "papers": papers}
            except (OSError, ValueError, RuntimeError) as exc:
                st.error(f"Could not complete retrieval: {exc}")
                st.info("Check the corpus, model checkpoint, and setup instructions in README.md.")

    result = st.session_state.get("search_result")
    if result:
        st.divider()
        st.markdown("**Search query**")
        st.write(result["query"])
        st.caption("Hybrid top 20 → fine-tuned reranking top 3 → top 2 original sentences per paper")
        render_results(result["papers"])

    st.divider()
    st.caption(
        "SciFact research-abstract collection | Fixed offline corpus | "
        "Evidence discovery only; no automatic truth verdicts."
    )


if __name__ == "__main__":
    main()
