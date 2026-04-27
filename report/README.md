# Report Assets

Generate table fragments for the LaTeX report:

```bash
python report/generate_report_assets.py
```

Primary report file:

- `report/report.tex`

Recommended workflow:

1. Run the baseline English evaluation.
2. Run `python evaluate_hindi_qa.py ...`
3. Run `python evaluate_rag_diagnostics.py ...`
4. Run `python report/generate_report_assets.py`
5. Compile `report/report.tex` with your local LaTeX toolchain.

This environment did not include `pdflatex`, so the repository stores a compile-ready LaTeX source rather than a built PDF.
