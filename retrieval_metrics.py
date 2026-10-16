"""Simple retrieval evaluation metrics for the SciFact project."""


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    """Calculate Recall@k for one scientific claim.

    Args:
        retrieved_ids: Ranked document IDs returned by the retriever.
        relevant_ids: IDs of documents annotated as evidence.
        k: Number of highest-ranked results to consider.

    Returns:
        Fraction of distinct relevant document IDs retrieved in the top k.

    Raises:
        ValueError: If k is not a positive integer or relevant_ids is empty.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty")

    top_k_ids = set(retrieved_ids[:k])  # Slice before deduplicating.
    retrieved_relevant = top_k_ids & relevant_ids
    return len(retrieved_relevant) / len(relevant_ids)


if __name__ == "__main__":
    # SciFact claim 47: one annotated evidence document.
    # This is an illustrative ranking, NOT a model-generated result.
    claim_47_ranking = [3512154, 26996935]
    claim_47_evidence = {26996935}

    assert recall_at_k(claim_47_ranking, claim_47_evidence, k=1) == 0.0
    assert recall_at_k(claim_47_ranking, claim_47_evidence, k=2) == 1.0
    print("SciFact claim 47, illustrative ranking:")
    print(f"  Recall@1: {recall_at_k(claim_47_ranking, claim_47_evidence, 1):.4f}")
    print(f"  Recall@2: {recall_at_k(claim_47_ranking, claim_47_evidence, 2):.4f}")

    # Synthetic example: 2 of 3 evidence documents found in the first 4
    # returned positions. Duplicate 101 does not count as another hit.
    synthetic_ranking = [101, 101, 999, 102, 103]
    synthetic_evidence = {101, 102, 103}
    score = recall_at_k(synthetic_ranking, synthetic_evidence, k=4)
    assert abs(score - 2 / 3) < 1e-12
    print(f"Synthetic example, Recall@4: {score:.4f}")

    # The first k positions are counted, not the first k unique IDs.
    assert recall_at_k([999, 101], {101}, k=1) == 0.0
    assert recall_at_k([], {101}, k=5) == 0.0

    for invalid_k in (0, -1, 1.5, True):
        try:
            recall_at_k([101], {101}, invalid_k)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for k={invalid_k!r}")

    try:
        recall_at_k([101], set(), k=1)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty relevant_ids")

    print("All tests passed.")
