"""Generate LaTeX table fragments from evaluation outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
GENERATED = ROOT / "report" / "generated"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _escape(text: object) -> str:
    value = str(text)
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def _format(value: str) -> str:
    stripped = (value or "").strip()
    if not stripped:
        return "--"
    try:
        number = float(stripped)
    except ValueError:
        return _escape(stripped)
    if 0 <= number <= 1:
        return f"{number:.3f}"
    return f"{number:.2f}"


def _write_table(path: Path, headers: list[str], rows: Iterable[list[str]], caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{_escape(caption)}}}",
        f"\\label{{{_escape(label)}}}",
        "\\begin{tabular}{" + "l" * len(headers) + "}",
        "\\hline",
        " & ".join(_escape(header) for header in headers) + " \\\\",
        "\\hline",
    ]
    wrote_row = False
    for row in rows:
        wrote_row = True
        lines.append(" & ".join(row) + " \\\\")
    if not wrote_row:
        lines.append("\\multicolumn{" + str(len(headers)) + "}{c}{Pending evaluation output} \\\\")
    lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ])
    path.write_text("\n".join(lines))


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)

    english_rows = _read_csv(ROOT / "evaluation" / "results" / "metrics_summary.csv")
    _write_table(
        GENERATED / "english_metrics.tex",
        ["Model", "ROUGE-L", "BERTScore", "Faithfulness", "Refusal"],
        [
            [
                _escape(row.get("model", "")),
                _format(row.get("rouge_l", "")),
                _format(row.get("bertscore_f1", "")),
                _format(row.get("faithfulness_pct_manual", "")),
                _format(row.get("adversarial_refusal_accuracy", "")),
            ]
            for row in english_rows
        ],
        "Baseline English QA metrics.",
        "tab:english-metrics",
    )

    hindi_rows = _read_csv(ROOT / "evaluation" / "hindi_results" / "metrics_summary.csv")
    _write_table(
        GENERATED / "hindi_metrics.tex",
        ["Model", "chrF", "BERTScore", "Citation Presence", "Citation Validity", "Refusal"],
        [
            [
                _escape(row.get("model", "")),
                _format(row.get("chrf", "")),
                _format(row.get("bertscore_f1", "")),
                _format(row.get("citation_presence_rate", "")),
                _format(row.get("mean_citation_validity", "")),
                _format(row.get("adversarial_refusal_accuracy", "")),
            ]
            for row in hindi_rows
        ],
        "Hindi cross-lingual QA metrics.",
        "tab:hindi-metrics",
    )

    diagnostic_rows = _read_csv(ROOT / "evaluation" / "diagnostics" / "metrics_summary.csv")
    _write_table(
        GENERATED / "diagnostic_metrics.tex",
        [
            "Model",
            "Query Type Acc.",
            "Source Hit",
            "Comment Sat.",
            "Avg Comment Ev.",
            "Paraphrase Jaccard",
        ],
        [
            [
                _escape(row.get("model", "")),
                _format(row.get("query_type_accuracy", "")),
                _format(row.get("source_type_hit_rate", "")),
                _format(row.get("comment_required_satisfaction_rate", "")),
                _format(row.get("average_comment_evidence_count", "")),
                _format(row.get("paraphrase_mean_jaccard", "")),
            ]
            for row in diagnostic_rows
        ],
        "Extended retrieval diagnostics.",
        "tab:diagnostic-metrics",
    )

    comment_rows = _read_csv(ROOT / "evaluation" / "diagnostics" / "comment_probe_summary.csv")
    _write_table(
        GENERATED / "comment_probe_metrics.tex",
        ["Sampled Probes", "Initial Hit Rate", "Final Hit Rate", "Mean Final Rank"],
        [
            [
                _format(row.get("n", "")),
                _format(row.get("initial_hit_rate", "")),
                _format(row.get("final_hit_rate", "")),
                _format(row.get("mean_final_rank_when_hit", "")),
            ]
            for row in comment_rows
        ],
        "Sampled comment retrievability probes.",
        "tab:comment-probes",
    )


if __name__ == "__main__":
    main()
