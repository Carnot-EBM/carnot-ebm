# The Labeling Problem in Hallucination Detection Benchmarks: Code and Data

This repository contains the code and data for reproducing the paper's release analyses from a fixed human-annotated benchmark CSV.

## Paper and data

Companion repository for the paper "The Labeling Problem in Hallucination Detection Benchmarks: An Empirical Evaluation" (Jorma Valjakka, Juhani Kivimäki, Juha Mylläri, Jukka K. Nurminen; University of Helsinki). Dataset: https://doi.org/10.7910/DVN/PCHISZ (Harvard Dataverse). Tested with Python 3.12.

Preprint: https://arxiv.org/abs/2610.08026

The recommended reproduction path is **Level 1 reproduction**: start from `data/human_judge_release_master.csv`, validate it, and recompute the reported tables, figures, and audit reports. This path requires no API keys, no GPU, and no large model downloads.

**Answer generation.** The answers in `data/human_judge_release_master.csv` (`answer_text`) were generated outside this repository; the generation settings are described in the paper (Appendix on answer generation). The released `answer_text` is the canonical dataset and is not regenerated here.

## What this repository contains

- `data/`: release data. The main file is `data/human_judge_release_master.csv`, with a column dictionary in `data/human_judge_release_data_dictionary.csv`. Croissant metadata is distributed with the dataset on Harvard Dataverse (https://doi.org/10.7910/DVN/PCHISZ).
- `prompts/`: verbatim judge and core-extraction prompt texts, extracted from the notebook's Section B cells.
- `scripts/`: command-line entry points for validation, Level 1 analyses, figure/table generation, expected-value construction, and table-value verification.
- `src/`: shared paths, configuration, I/O helpers, metrics, and reusable release-analysis functions.
- `neurips_notebook.ipynb`: notebook with the same default Level 1 release-analysis path plus optional Level 2 label-regeneration cells.
- `THIRD_PARTY_NOTICES.md`: source-dataset and third-party model licence notices.
- `tables/`, `figures/`: created when you run the scripts (not stored in the repository; the final versions are in the paper).
- `audits/`: generated source CSVs, consistency reports, and comparison outputs. `audits/` also contains raw outputs of the judge and extraction models; the terms of the respective model providers apply to those texts.
- `results/expected/`: paper-extracted expected values used by `scripts/31_verify_paper_tables.py`.

## Provenance

The provenance of the automatic labels and baseline scores is the notebook's Section B (CELL B1 through B10b): each cell shows exactly how a given label or score column in `data/human_judge_release_master.csv` was produced from the fixed generated answers, including judge prompts, models, and thresholding. Section B is disabled by default (`RUN_LABEL_REGENERATION = False`); enabling it calls paid APIs and/or requires a GPU, and reproduces the label-generation process rather than the fixed benchmark itself.

Construction of the underlying question pool and generator answers predates this notebook and is not included as runnable scripts in this repository. The construction of the question pool and the generator answers (sampling, generation settings, answer-length limit) is described in the companion paper (sections "Dataset Generation" and the "Answer Generation Setup" appendix); `data/README.md` documents the file structure and column meanings.

For paper-result reproduction, use the locked release master CSV described below.

## Level 1: Default Reproduction

Besides the master CSV, Level 1 also reads a small number of other tracked input files:

| Path | Needed for |
| :--- | :--- |
| `audits/factorial_prompt_run/<run>/labels.csv` | the criterion x structure factorial ablation |
| `audits/elaboration_core_only/<run>/p1_original_vs_core_b4matched/paired_results.csv` | the elaboration qualitative examples, answer-length reduction, and FP-rate summary |
| `audits/elaboration_core_only/<run>/human_validation_template_validated.csv` | the elaboration FP-rate summary |

These input files are tracked in the repository and are not overwritten by the scripts; generated outputs go to audits/ (other subfolders), tables/ and figures/.

Level 1 starts from:

```text
data/human_judge_release_master.csv
```

This CSV is the fixed, human-annotated release dataset and the read-only source of truth for the paper analyses. It contains the fixed question-answer rows, primary and second-annotator labels, and the automatic-label columns needed for the reported tables and figures.

Level 1 recomputes:

- input and schema validation
- inter-annotator agreement
- judge and baseline metrics from existing label columns
- prompt-ablation tables
- lexical/NLI baseline tables
- METEOR / BARTScore-CNN baseline tables (computed from the released scores and labels; the scores themselves are not recomputed in Level 1)
- the criterion x structure factorial ablation
- the elaboration intervention (qualitative examples, answer-length reduction, and the false-positive-rate summary)
- per-dataset breakdowns
- paper figures
- audit and consistency reports

Level 1 does not regenerate labels, call APIs, run GPU inference, or download large models.

### Recommended workflow

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/00_validate_inputs.py
python scripts/20_run_release_analyses.py
python scripts/31_verify_paper_tables.py
```

Expected output locations:

- validation summary: terminal output from `scripts/00_validate_inputs.py`
- tables: `tables/`
- figures: `figures/`
- analysis source CSVs and audits: `audits/`
- paper-table comparison report: `audits/table_comparison/table_value_diffs.md`

### How verification works

`results/expected/` holds values transcribed by hand from the companion paper's tables — it is not generated from this repository's code. `scripts/30_build_expected_from_paper.py` documents the source table for every value and writes `results/expected/*.csv`; those files are already tracked, so it does not need to be run for ordinary reproduction, but it is the place to look to see exactly where each expected value came from.

`scripts/31_verify_paper_tables.py` compares the CSVs produced by `scripts/20_run_release_analyses.py` against `results/expected/`, cell by cell, using the `TABLE_SPECS` entries defined in that script. Because the comparison is against hand-transcribed manuscript values rather than another run of the same code, a passing check means the reproduced table values match the hand-transcribed paper values within the stated tolerances; it does not by itself validate figures or annotations.

A clean run currently reports 1,647 comparisons across all `TABLE_SPECS` entries and ends with:

```text
Blocking findings: 0
```

The combined Level 1 runner is:

```bash
python scripts/20_run_release_analyses.py
```

Individual Level 1 scripts are also available:

```bash
python scripts/10_recompute_iaa.py
python scripts/11_prompt_ablation_tables.py
python scripts/12_lexical_nli_baselines.py
python scripts/13_bootstrap_mcnemar.py
python scripts/14_kappa_matrix.py
python scripts/15_error_direction_space.py
python scripts/16_per_dataset_gpt5mini.py
python scripts/17_per_dataset_prompt_ablation.py
python scripts/18_factorial_ablation.py
python scripts/19_new_metrics.py
python scripts/21_elaboration_examples.py
python scripts/22_elaboration_length.py
python scripts/23_elaboration_summary.py
```

`scripts/30_build_expected_from_paper.py` and `scripts/31_verify_paper_tables.py` (see above) are not part of the Level 1 analysis itself; they build and check the expected-value reference, respectively.

## Level 2: Optional Label Regeneration

Level 2 is optional and is not the default reviewer workflow. It contains notebook code for rerunning automatic labels and baseline scores on the same fixed question-answer rows.

Depending on which cells are enabled, Level 2 may require:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `HF_TOKEN`
- GPU resources
- large model downloads

Use Level 2 only for extended reproducibility or stress-testing regenerated labels. It may incur cost and can take substantially longer than Level 1. API judge outputs may also differ across time because provider-side models can change.

Cells B10 and B10b are additionally skipped unless `RUN_B10` / `RUN_B10B` is set to `True` in the notebook; see the "How to re-run Section B" note in CELL A1 for details.

The notebook defaults are configured for Level 1:

```python
RUN_RELEASE_ANALYSIS = True
RUN_LABEL_REGENERATION = False
ALLOW_EXPENSIVE_RUNS = False
```

## Using The Notebook

Open `neurips_notebook.ipynb` from the repository root. The default cells reproduce the paper tables and figures from `data/human_judge_release_master.csv`.

The notebook uses repository-relative paths:

```text
data/
tables/
figures/
audits/
```

If running the notebook from another working directory, set `REPRO_ROOT` to the repository root before running the setup/path cell.

## Data Notes

Treat `data/human_judge_release_master.csv` as read-only. Do not overwrite it. Recomputed outputs should go only to:

```text
audits/
tables/
figures/
```

`results/expected/` holds the paper's hand-transcribed reference numbers. Ordinary Level 1 analysis and verification scripts only read `results/expected/`. The optional `scripts/30_build_expected_from_paper.py` writes the expected-value CSVs. Generated tables and figures appear in `tables/` and `figures/` (both git-ignored).

For column meanings, see `data/human_judge_release_data_dictionary.csv` and `data/README.md`.

## Acknowledgements

This work was partly supported by local authorities ("Business Finland") under grant agreement 23004 ELFMo of the ITEA4 programme, which funded three of the authors.

## License

Code is released under the MIT License; see `LICENSE`.

Data are licensed CC BY-SA 4.0 (human annotations, judge labels, metric scores, and documentation); the model-generated answers in the `answer_text` column remain subject to the terms of the generation models; questions and reference answers derive from TriviaQA, HotpotQA, and TruthfulQA and remain subject to their original licences. See `LICENSE-DATA`. Judge labels produced by Llama 3 8B (`label_llama_3_8b_local_p1`, `_p2`, `_p3`) are released under CC BY-SA 4.0 together with the other judge labels; the Meta Llama 3 Community License may additionally apply to such outputs. The MIT licence covers the code only. Data and audit files are covered by `LICENSE-DATA` and `THIRD_PARTY_NOTICES.md`; raw model outputs under `audits/` remain subject to the terms of the respective model providers.
