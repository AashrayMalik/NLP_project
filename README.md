# r/jobs RAG Question Answering

This project builds a retrieval-augmented generation system over the collected r/jobs Reddit corpus. It retrieves relevant posts, top comments, and topic summaries from the local SQLite database, then passes that evidence to Groq or Gemini to produce a grounded answer with citations.

## Setup

Install dependencies:

```bash
uv sync
```

Create a local `.env` file from `.env.example` and add your endpoint keys:

```bash
cp .env.example .env
```

Required keys:

- `GROQ_API_KEY` for the Groq endpoint.
- `GEMINI_API_KEY` for the Google Gemini endpoint.

Optional model overrides:

- `GROQ_MODEL`, default `llama-3.3-70b-versatile`.
- `GEMINI_MODEL`, default `gemini-2.5-flash`.
- `RAG_EMBEDDING_DEVICE`, default `cpu`. Set to `mps` on Apple Silicon only if you have enough memory for the full index build.

Endpoint code lives in `rag/llms.py`. Provider selection is exposed through `query_rag.py --model groq|gemini`, `evaluate_rag.py --models groq gemini`, and the Streamlit QA page dropdown.

## Build The RAG Index

Build the full ChromaDB vector index and graph artifact:

```bash
uv run python build_rag_index.py
```

For quick smoke tests only:

```bash
uv run python build_rag_index.py --limit 50
```

The full build should write `scraping/data/rag_manifest.json` with `"limited": false`. Generated index files are intentionally ignored by git.

## Ask Questions

Retrieval-only debug mode:

```bash
uv run python query_rag.py "How large is the collected r/jobs corpus?" --model retrieval_only
```

Hindi answer generation with an English retrieval bridge:

```bash
uv run python query_rag.py "salary negotiation par users kya advice dete hain?" --model groq --language hindi --retrieval-query "What advice do users give about salary negotiation and raises?"
```

Groq:

```bash
uv run python query_rag.py "What do users think about salary negotiation?" --model groq
```

Gemini:

```bash
uv run python query_rag.py "What do users think about salary negotiation?" --model gemini
```

Run the Streamlit app:

```bash
uv run streamlit run app.py
```

Open the `QA` page and choose Retrieval only, Groq, or Gemini.

## Evaluation

The hand-written ground-truth set is in `evaluation/qa_set.jsonl`. It contains factual, opinion-summary, trend, and adversarial questions.

Run the full comparison:

```bash
uv run python evaluate_rag.py --models groq gemini
```

Outputs are written to `evaluation/results/`:

- `results_all.jsonl`: generated answers, gold answers, retrieved evidence IDs, ROUGE-L, BERTScore, and refusal flags.
- `metrics_summary.csv`: model-level comparison table.
- `faithfulness_review.csv`: mark `faithful_manual` as `1` or `0` for each answer.
- `comparative_report.md`: report table and qualitative analysis notes.

After marking `faithfulness_review.csv`, rerun the same evaluation command. Existing manual labels are loaded and aggregated into `faithfulness_pct_manual`.

Fast local smoke test without endpoint keys:

```bash
uv run python evaluate_rag.py --models retrieval_only --skip-bertscore --output-dir /tmp/rag_eval_smoke
```

## Hindi Cross-Lingual QA

The Hindi benchmark lives in `evaluation/hindi_qa_set.jsonl`. Each item stores the Hindi question and answer, an English reference answer, answerability metadata, tags, and an English retrieval bridge query.

Run the Hindi comparison:

```bash
uv run python evaluate_hindi_qa.py --models groq gemini
```

Fast smoke test:

```bash
uv run python evaluate_hindi_qa.py --models retrieval_only --skip-bertscore --output-dir /tmp/hindi_eval_smoke
```

Outputs are written to `evaluation/hindi_results/`:

- `results_all.jsonl`: per-example Hindi answers, citations, and retrieval diagnostics
- `metrics_summary.csv`: model-level Hindi comparison
- `tag_breakdown.csv`: edge-case breakdown for code-mixed, romanized, named-entity, and adversarial subsets
- `hindi_manual_review.csv`: mark `fluency_manual`, `adequacy_manual`, and `causal_overclaim_manual`

## Extended RAG Diagnostics

The extended retrieval checks live in `evaluation/rag_diagnostic_set.jsonl`.

Run diagnostics:

```bash
uv run python evaluate_rag_diagnostics.py --model retrieval_only
```

Outputs are written to `evaluation/diagnostics/`:

- `diagnostic_results.csv`: per-query retrieval behavior
- `metrics_summary.csv`: source-type hit rates, comment satisfaction, paraphrase stability, and citation quality
- `comment_probe_results.csv`: sampled comment retrievability probes
- `causal_overclaim_review.csv`: manual review sheet for unsupported causal wording


## Corpus Caveat

The scraper collected up to the top five comments by score for each qualifying post. Answers should describe community views as tendencies in the collected r/jobs posts and top comments, not as universal claims about every subreddit user.
