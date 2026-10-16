# Evaluation and reproducibility

## Source, population, and metrics

The attached CSV files were produced during development from SciFact's **development** partition. Each contains 188 annotated dev claims. Other unannotated dev claims were excluded from these gold-evidence metrics. The official test set's hidden labels were not used.

`Recall@k` is average annotated-evidence document recall across eligible claims, **not** percent of all relevant scientific studies found. `Document hit within top 3` counts at least one gold-evidence document among the selected top 3; `complete evidence-set hit` requires at least one complete annotated evidence group among the selected sentences.

## Document retrieval baselines (188 annotated development claims)

| Method | Mean Recall@5 | Mean Recall@10 | Mean Recall@20 |
|---|---:|---:|---:|
| BM25 | 0.8848 | 0.9158 | 0.9437 |
| Semantic embeddings | 0.8685 | 0.9117 | 0.9516 |
| Hybrid BM25 + semantic (RRF) | 0.8914 | 0.9468 | 0.9545 |
| Hybrid candidates + fine-tuned cross-encoder reranking | **0.9239** | **0.9545** | 0.9545 |

The fine-tuned model reorders the hybrid candidate list. It cannot recover annotated documents missing from the hybrid top-20 pool.

## Neural sentence selection

Across those same 188 annotated dev claims, the document hit rate at top 3 was **0.9309**; complete annotated evidence-set hit within the selected **three papers × two sentences** was **0.7394**. These are not stance or entailment classification accuracy measures.

## Pretrained versus fine-tuned cross-encoder, same candidate pool

In the saved reranker comparison, pretrained Recall@5 was 0.9207; fine-tuned Recall@5 was 0.9239. Pretrained MRR@5 was 0.8452; fine-tuned MRR@5 was 0.8712. Improvements were moderate, and the pretrained baseline was strong. Avoid presenting this as a dramatic gain.

## Experimental discipline and limitations

- Model training uses SciFact training claims. Development labels were used for offline evaluation; no claim is made of an untouched final deployment test set.
- Unannotated claim-document pairs used as training negatives are **presumed**, not independently verified irrelevant.
- Lexical and semantic similarity can retrieve topical background passages that do not logically support a specific claim.
- The official SciFact data source is <https://github.com/allenai/scifact>.

## Preserved source outputs

- `retrieval_claim_analysis.csv`: BM25, semantic, hybrid rankings and metrics
- `reranking_finetuned_analysis.csv`: hybrid versus fine-tuned reranking
- `reranker_comparison.csv`: pretrained versus fine-tuned cross-encoder
- `evidence_pipeline_analysis.csv`: end-to-end evidence localization

The numbers in this document were recomputed directly as column means over these saved CSVs. Re-running the same evaluation scripts requires the published dataset, model weights, and any intermediate rankings specified by the relevant script.
