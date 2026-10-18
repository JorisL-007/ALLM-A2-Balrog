# LaTeX assignment report

`Ass2Reasoning_report.tex` answers the questions in Exercises 1–3 and reports the larger follow-up study separately. The Exercise 1 matrix covers all three completed local-model naive sweeps; `../experiments/summarize_exercise1_naive.py` regenerates it from their run summaries. The highlighted `PLACEHOLDER` fields are for student identity, submission date, and missing Exercise 1 submission information. The repository has no evidenced cloud-model Exercise 1 run, so the report labels that result as missing.

Compile from this directory with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error Ass2Reasoning_report.tex
pdflatex -interaction=nonstopmode -halt-on-error Ass2Reasoning_report.tex
```

The report uses standard LaTeX packages and an inline bibliography; no BibTeX run is needed. The source data and raw-call samples are linked in the report and documented in `../research/followup_report.md`.

No TeX engine was installed in the authoring workspace, so a PDF was not generated there. The source passed brace, environment, and linked raw-log path checks.
