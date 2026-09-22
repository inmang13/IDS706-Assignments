# AI Usage Log

This course permits AI pair programming when the use is documented and
credited, and when the submitted work reflects my own understanding and
modification rather than raw AI output. This file records where AI tools
were used in this repository, what they produced, and what I did to verify
or change it.

Add a new row each time you use an AI tool. Keep the "What I changed or
verified" column honest and specific — that column is the one that shows
understanding.

---

## Assignment 2 — Data analysis and Rust notebook

| Where | Tool | What it helped with | What I changed or verified |
|---|---|---|---|
| `notebooks/basic_analysis.ipynb` — plotting cells | OpenAI Codex | Matplotlib syntax and styling for the yearly/monthly boxplots and the observed-vs-predicted figure | *(fill in: e.g. adjusted the log scale, chose the colors, confirmed the boxplot groups matched my groupby)* |
| `notebooks/basic_analysis.ipynb` — ML section | OpenAI Codex | Exact syntax for the XGBoost and Random Forest calls | *(fill in: e.g. chose the hyperparameters, set the seed, wrote the NSE function myself, checked the train/test split was chronological)* |
| `notebooks/basic_analysis.ipynb` — Pandas vs. Polars benchmark | Claude Code | Polars syntax, which I had not used before | *(fill in: e.g. verified both libraries perform the same operations and return matching results before comparing timings)* |

## Assignment 3 — Testing and CI

| Where | Tool | What it helped with | What I changed or verified |
|---|---|---|---|
| `src/main.py` | Claude (Opus 5, via Claude Code) | Reviewed the repo and wrote a stub outline of the analysis pipeline — module docstring, function signatures, docstrings, and suggested test cases. No function bodies were generated. | *(fill in: I wrote all of the implementations myself by moving code from the notebook)* |
| `tests/test_main.py` | *(fill in)* | | |
| `.github/workflows/test.yml` | *(fill in — note if this was written by hand)* | | |

