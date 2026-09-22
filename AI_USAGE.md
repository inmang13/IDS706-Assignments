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
| `src/main.py` | Claude (Sonnet 5, via Claude Code) | Wrote function bodies for `clean_flow_data`'s data-quality reporting (NA/dupe/missing-day counts), `summarize_flow`'s return value and calendar columns, `split_train_test`'s auto year-boundary 80/20 split, the `_Log1pRegressor` wrapper and `feature_importance` helper, the four plotting functions, and wired everything into `run_pipeline`. Found and fixed several bugs: `from turtle import pd` typo, `train_test_spl` import typo, `split_date="None"` string bug, missing `return metrics`, `clean_flow_data` crashing on empty input, ignored `random_state` param, and matplotlib defaulting to a broken Tk backend. Explained why RF is transform-invariant to feature scaling but not target scaling, which motivated the log1p target transform. | *(fill in: I asked for the log-transform explanation, decided to add explicit lag features after reviewing the feature-importance output, and reviewed each function change before accepting it — confirm this matches what you actually checked)* |
| `tests/test_main.py` | Claude (Sonnet 5, via Claude Code) | Wrote the full test suite from scratch (14 tests): data loading, cleaning (incl. empty-input edge case), feature engineering (target/lag/leakage checks), train/test split (incl. split-past-end edge case), NSE, model evaluation, all four plot functions (via `tmp_path`), and one end-to-end system test. | *(fill in: I ran the suite myself and read through the assertions to confirm they test what I think they test)* |
| `Dockerfile` | Claude (Sonnet 5, via Claude Code) | Found that `data/` was never copied into the image, which would fail `make docker-test` in CI, and added the missing `COPY data ./data` line. | *(fill in: confirm you understand why the container needs the data directory)* |
| `.github/workflows/test.yml` | *(fill in — note if this was written by hand)* | | |

