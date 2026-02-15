## Objective

To analyze the impact of retrieval augmentation (RAG) on structured JSON scene generation,
with focus on:

- Structural correctness
- Semantic alignment with query
- Hallucination leakage

---

## Experimental Settings

| Setting        | RAG Enabled | Top-K | Similarity Threshold |
|---------------|------------|--------|----------------------|
| No RAG        | ❌         | -      | -                    |
| Top-1         | ✅         | 1      | 0.0                  |
| Top-3         | ✅         | 3      | 0.0                  |
| Thresholded   | ✅         | 3      | 0.2                  |

---

## Metrics

1. JSON Validity Rate
2. Structural Consistency Score
3. Query Alignment Score (word overlap metric)
4. Hallucination Incidence (manual inspection)

---

## Observations

- Top-3 RAG increases semantic drift due to context contamination.
- Threshold filtering reduces cross-query leakage.
- No RAG produces cleaner but less detailed outputs.
- Thresholded Top-1 achieves best balance.

---

## Conclusion

Structured LLM generation benefits from controlled retrieval.
Unfiltered multi-document retrieval increases hallucination risk.
Similarity-based filtering improves grounding discipline.

---