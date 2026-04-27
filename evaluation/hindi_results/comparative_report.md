# Hindi Cross-Lingual QA Comparison

| Model | chrF | BERTScore F1 | Citation Presence | Citation Validity | Refusal Accuracy |
|---|---:|---:|---:|---:|---:|
| groq | 0.233 |  | 86.7% | 80.9% | 33.3% |
| retrieval_only | 0.198 |  | 0.0% |  | 0.0% |

## Manual Review

Fill `fluency_manual` and `adequacy_manual` on a 1-5 scale in `hindi_manual_review.csv`. Mark `causal_overclaim_manual` as 1 when the answer makes an unsupported causal claim and 0 otherwise.

## Edge Cases

Inspect `tag_breakdown.csv` for code-mixed, romanized Hindi, Reddit slang, named-entity, and adversarial subsets.