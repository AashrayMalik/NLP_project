# Hindi Cross-Lingual QA and Extended RAG Evaluation for r/jobs

**NLP Project Report**

## 1. System Overview

The project builds a retrieval-augmented question answering system over a cleaned `r/jobs` Reddit corpus. The corpus contains posts, top comments, topic summaries, flair metadata, and month-level structure stored in a local SQLite database. Retrieval combines three evidence channels:
*   Dense vector retrieval over post, comment, and topic-summary chunks
*   Graph expansion across posts, comments, topics, flairs, and months
*   Deterministic SQL facts for count, date-range, flair, topic, and comment-policy questions

The answer generation layer uses Groq, Gemini, or a retrieval-only debug mode. The answer prompt requires grounding, explicit citation of evidence IDs, refusal when evidence is insufficient, and careful language around community tendencies rather than universal claims.

## 2. RAG Implementation

### 2.1 Chunking and Indexing
Posts are chunked with paragraph-aware splitting. Comments are indexed as atomic chunks with parent-post context. Topic summaries are synthesized from BERTopic labels, representative arguments, and trend metadata. The vector store persists the chunk embeddings, while the graph artifact encodes object-level links between comments, posts, topics, flairs, months, and trend nodes.

### 2.2 Retrieval and Reranking
At query time the retriever classifies the query as factual, trend, opinion, or general. It then:
1.  Retrieves initial dense-vector candidates
2.  Expands the graph around the highest-ranked seeds
3.  Adds SQL facts for obvious structural questions
4.  Reranks the merged candidate pool with source-type, score, and graph-distance boosts

This implementation was extended for the present report with provenance output, source-type counts, retrieval-origin counts, comment-evidence counts, and optional English retrieval-bridge queries for Hindi cross-lingual evaluation.

## 3. Baseline English Evaluation

The baseline English QA evaluation uses the hand-written ground-truth set already present in the repository. It measures ROUGE-L, BERTScore, manual faithfulness, refusal accuracy, and now citation quality.

**Table 1: Baseline English QA metrics.**

| Model | ROUGE-L | BERTScore | Faithfulness | Refusal |
| :--- | :--- | :--- | :--- | :--- |
| retrieval_only | 0.070 | -- | fill manually | 0.000 |

Qualitatively, strong English answers should ground factual responses in SQL facts, summarize opinions from comments or topic summaries, refuse adversarial prompts clearly, and preserve the top-comments-only caveat.

## 4. Hindi Cross-Lingual QA

The Indian-language evaluation in this submission uses **Hindi**. The task is cross-lingual QA: questions are asked in Hindi, retrieval is performed over the English corpus, and models must answer in Hindi with evidence citations.

### 4.1 Dataset Design
The Hindi benchmark contains at least 24 examples spanning:
*   factual questions
*   opinion-summary questions
*   trend-comparison questions
*   adversarial questions
*   code-mixed Hindi-English inputs
*   romanized Hindi inputs
*   named-entity and acronym-heavy cases

Each item stores the Hindi question, Hindi reference answer, English reference answer, answerability label, type, tags, and an English retrieval bridge query when needed. The retrieval bridge is important because the indexed corpus and embedding model are English-oriented.

### 4.2 Metrics
The Hindi evaluator reports chrF, multilingual BERTScore, refusal accuracy, citation presence, citation validity, and manual fluency and adequacy review. It also surfaces causal-language candidates for later inspection.

**Table 2: Hindi cross-lingual QA metrics.**

| Model | chrF | BERTScore | Citation Presence | Citation Validity | Refusal |
| :--- | :--- | :--- | :--- | :--- | :--- |
| retrieval_only | 0.198 | -- | 0.000 | -- | 0.000 |

## 5. Extended RAG Diagnostics

To move beyond assignment-minimum evaluation, the system now includes a second diagnostic track focused on retrieval behavior rather than only final answer similarity.

### 5.1 Structure Preservation
Diagnostic questions test whether corpus structure is visible in retrieval outputs. These checks verify that factual questions surface SQL facts, trend questions surface topic summaries, and comment-policy queries preserve the "top comments, not full threads" caveat.

### 5.2 Comment Usefulness
Comment-heavy diagnostic questions require comment evidence in top-k retrieval. In addition, a sampled comment probe routine selects comments across month and score buckets, turns them into probe queries, and measures whether the original comment returns in the initial vector candidates and final reranked evidence.

### 5.3 Paraphrase Stability
Paraphrase groups compare evidence-overlap scores across semantically equivalent queries. Stable retrieval should not collapse when the wording changes.

**Table 3: Extended retrieval diagnostics.**

| Model | Query Type Acc. | Source Hit | Comment Sat. | Avg Comment Ev. | Paraphrase Jaccard |
| :--- | :--- | :--- | :--- | :--- | :--- |
| retrieval_only | 0.667 | 1.000 | 1.000 | 5.83 | 0.213 |

**Table 4: Sampled comment retrievability probes.**

| Sampled Probes | Initial Hit Rate | Final Hit Rate | Mean Final Rank |
| :--- | :--- | :--- | :--- |
| 6.00 | 0.833 | 0.500 | 7.67 |

## 6. Causal Overclaim Audit

The system's graph is **structural**, not causal. It preserves Reddit structure and temporal grouping, but it does not identify causal effects. For that reason, this report includes a causal-overclaim audit rather than a causal inference module.

The audit flags answers that use language such as "because", "caused", "led to", or equivalent Hindi formulations when the evidence only supports anecdote, correlation, or community perception. This is especially relevant for topics such as AI layoffs, hiring slowdowns, or workplace behavior where Reddit comments can imply explanations without establishing causality.

## 7. Limitations

*   Only the top comments per post were collected, so the corpus does not represent full discussion threads.
*   The Hindi task relies on an English corpus and an English-oriented retriever, so cross-lingual performance may partly depend on the English retrieval bridge query.
*   The dataset is observational Reddit data; it supports tendencies, anecdotes, and descriptive patterns, not causal claims.
*   Manual review is still required for faithfulness, Hindi fluency and adequacy, and unsupported causal claims.
*   This environment did not include a local LaTeX engine, so the report was prepared as a compile-ready `.tex` file rather than compiled in place.
