# Human and Automatic Hallucination-Judgment Release Data

Dataset: The Labeling Problem in Hallucination Detection Benchmarks: human-judge release dataset. Authors: Jorma Valjakka, Juhani Kivimäki, Juha Mylläri, Jukka K. Nurminen (University of Helsinki). DOI: https://doi.org/10.7910/DVN/PCHISZ

This directory contains the release-ready data file used for the human-vs-automatic
hallucination-labeling analyses reported in the paper. The file combines human
annotations, an optional second-annotator label, and automatic labels produced by
multiple LLM judges and lexical/NLI/embedding-based baselines.

---

## Files

| File | Purpose |
| :--- | :--- |
| `data/human_judge_release_master.csv` | Main release file. 900 rows x 87 columns. |
| `data/human_judge_release_data_dictionary.csv` | Column-level description file, one row per master column (87 rows). Auto-generated from the release CSV. |
| `data/README.md` | This file. |

### Integrity

SHA-256 of `human_judge_release_master.csv`:

```text
c754d9d66ccc5c53c55fb8c9156135a2ce2f93ce913a1bff0652d6647d27dc0a
```

Recompute with `sha256sum data/human_judge_release_master.csv` and compare before use;
the file is treated as read-only and must not be regenerated or reformatted in place.

### Persistent identifier

DOI: `10.7910/DVN/PCHISZ`

---

## Row Structure

Each row corresponds to one generated answer for one question by one generator model.
The core unit is: **question x generator model**. The release contains three generator
models (`llama3-8b`, `gemma-2-9b`, `mistral-7b`) over 300 questions each, for 900 rows
total, drawn from three source datasets (`hotpotqa`, `triviaqa`, `truthfulqa`).

---

## Column Groups (10 + 21 + 50 + 6 = 87)

The 87 columns fall into four groups. Column descriptions below follow the accompanying data dictionary, with additional release-specific clarifications.

### 1. Core / metadata columns (10)

| Column | Description |
| :--- | :--- |
| `item_id` | Unique row identifier for a question-answer pair. |
| `qid` | Question identifier within the sampled canonical question pool. |
| `qid_order` | Numeric suffix extracted from qid where available. |
| `dataset` | Source dataset: triviaqa, hotpotqa, or truthfulqa. |
| `model` | Generator model that produced the answer. |
| `question` | Question shown to the generator model. |
| `ref_str` | Reference answer(s), serialized as text. |
| `answer_text` | Generated answer from the model. |
| `annotator1_label` | Primary human label: 1=hallucination, 0=correct, -1=missing/uncertain. |
| `annotator2_label` | Second annotator label where available: 1=hallucination, 0=correct, -1=not annotated. |

**Sentinel value:** `annotator2_label` uses `-1` for the 600 rows outside the 300-row
double-annotated subset (100 questions x 3 generators). Always filter to `{0, 1}`
before computing inter-annotator agreement on this column — reading it unfiltered
yields spuriously low raw agreement, since `-1` is not a valid label value, only a
"not annotated" marker.

### 2. Automatic judge label columns (21)

Naming convention: `label_<judge>_<prompt>`, for 7 judges x 3 prompts (p1, p2, p3).

| Column | Description |
| :--- | :--- |
| `label_gpt_5_mini_p1` | Automatic judge label: 1=hallucination, 0=correct, schema allows -1=missing/refusal; none occurs in this release. |
| `label_gpt_5_mini_p2` | Automatic judge label: 1=hallucination, 0=correct, schema allows -1=missing/refusal; none occurs in this release. |
| `label_gpt_5_mini_p3` | Automatic judge label: 1=hallucination, 0=correct, schema allows -1=missing/refusal; none occurs in this release. |
| `label_gpt_5_nano_p1` | Automatic judge label: 1=hallucination, 0=correct, schema allows -1=missing/refusal; none occurs in this release. |
| `label_gpt_5_nano_p2` | Automatic judge label: 1=hallucination, 0=correct, schema allows -1=missing/refusal; none occurs in this release. |
| `label_gpt_5_nano_p3` | Automatic judge label: 1=hallucination, 0=correct, schema allows -1=missing/refusal; none occurs in this release. |
| `label_gpt_5_4_p1` | Automatic judge label: 1=hallucination, 0=correct, schema allows -1=missing/refusal; none occurs in this release. |
| `label_gpt_5_4_p2` | Automatic judge label: 1=hallucination, 0=correct, schema allows -1=missing/refusal; none occurs in this release. |
| `label_gpt_5_4_p3` | Automatic judge label: 1=hallucination, 0=correct, schema allows -1=missing/refusal; none occurs in this release. |
| `label_claude_opus_4_7_p1` | Automatic judge label: 1=hallucination, 0=correct, schema allows -1=missing/refusal; none occurs in this release. |
| `label_claude_opus_4_7_p2` | Automatic judge label: 1=hallucination, 0=correct, schema allows -1=missing/refusal; none occurs in this release. |
| `label_claude_opus_4_7_p3` | Automatic judge label: 1=hallucination, 0=correct, schema allows -1=missing/refusal; none occurs in this release. |
| `label_gemma_2_9b_local_p1` | Automatic judge label: 1=hallucination, 0=correct, schema allows -1=missing/refusal; none occurs in this release. |
| `label_gemma_2_9b_local_p2` | Automatic judge label: 1=hallucination, 0=correct, schema allows -1=missing/refusal; none occurs in this release. |
| `label_gemma_2_9b_local_p3` | Automatic judge label: 1=hallucination, 0=correct, schema allows -1=missing/refusal; none occurs in this release. |
| `label_qwen2_5_7b_local_p1` | Automatic judge label: 1=hallucination, 0=correct, schema allows -1=missing/refusal; none occurs in this release. |
| `label_qwen2_5_7b_local_p2` | Automatic judge label: 1=hallucination, 0=correct, schema allows -1=missing/refusal; none occurs in this release. |
| `label_qwen2_5_7b_local_p3` | Automatic judge label: 1=hallucination, 0=correct, schema allows -1=missing/refusal; none occurs in this release. |
| `label_llama_3_8b_local_p1` | Automatic judge label: 1=hallucination, 0=correct, schema allows -1=missing/refusal; none occurs in this release. |
| `label_llama_3_8b_local_p2` | Automatic judge label: 1=hallucination, 0=correct, schema allows -1=missing/refusal; none occurs in this release. |
| `label_llama_3_8b_local_p3` | Automatic judge label: 1=hallucination, 0=correct, schema allows -1=missing/refusal; none occurs in this release. |

### 3. Lexical and embedding-based metric families (50)

Five metric families, each carrying the same 10-column pattern: a raw score plus nine
oracle/cross-validation threshold and fold-diagnostic columns.

Families: `rougeL`, `bertscore_f1`, `meteor`, `bartscore_r2h`, `bartscore_h2r`.

The derived columns of the `bertscore_f1` family use the prefix
`bertscore_` (for example `bertscore_label_oracle`), not `bertscore_f1_`.

#### `rougeL` family
| Column | Description |
| :--- | :--- |
| `rougeL` | Longest-common-subsequence overlap between the answer and the reference(s), maximum over '\|'-separated references. |
| `rougeL_label_oracle` | ROUGE-L label at the oracle threshold: 1=hallucination, 0=correct. The oracle threshold is tuned on the full evaluation set and is an optimistic diagnostic bound, not a deployable procedure. |
| `rougeL_label_cv` | ROUGE-L label at the cross-validated threshold: 1=hallucination, 0=correct. Concatenates the out-of-fold predictions of a 3-fold split; the threshold is selected on the two training folds and applied to the held-out fold. |
| `rougeL_tau_oracle` | Oracle threshold used for ROUGE-L, selected per generator to maximise agreement with annotator1_label on the full evaluation set. |
| `rougeL_tau_cv_mean` | Mean of the three per-fold ROUGE-L thresholds, constant within a generator. |
| `rougeL_tau_cv_min` | Smallest of the three per-fold ROUGE-L thresholds, constant within a generator. |
| `rougeL_tau_cv_max` | Largest of the three per-fold ROUGE-L thresholds, constant within a generator. |
| `rougeL_cv_fold` | Cross-validation fold assignment for ROUGE-L (0, 1 or 2). Determines which fold the row was held out from. |
| `rougeL_cv_kappa_mean_fold` | Mean per-fold Cohen's kappa for ROUGE-L against annotator1_label, constant within a generator. |
| `rougeL_cv_kappa_oof` | Out-of-fold Cohen's kappa for ROUGE-L against annotator1_label, constant within a generator. |

#### `bertscore_f1` family
| Column | Description |
| :--- | :--- |
| `bertscore_f1` | BERTScore F1 between the answer and the reference(s), maximum over '\|'-separated references. |
| `bertscore_label_oracle` | BERTScore label at the oracle threshold: 1=hallucination, 0=correct. The oracle threshold is tuned on the full evaluation set and is an optimistic diagnostic bound, not a deployable procedure. |
| `bertscore_label_cv` | BERTScore label at the cross-validated threshold: 1=hallucination, 0=correct. Concatenates the out-of-fold predictions of a 3-fold split; the threshold is selected on the two training folds and applied to the held-out fold. |
| `bertscore_tau_oracle` | Oracle threshold used for BERTScore, selected per generator to maximise agreement with annotator1_label on the full evaluation set. |
| `bertscore_tau_cv_mean` | Mean of the three per-fold BERTScore thresholds, constant within a generator. |
| `bertscore_tau_cv_min` | Smallest of the three per-fold BERTScore thresholds, constant within a generator. |
| `bertscore_tau_cv_max` | Largest of the three per-fold BERTScore thresholds, constant within a generator. |
| `bertscore_cv_fold` | Cross-validation fold assignment for BERTScore (0, 1 or 2). Determines which fold the row was held out from. |
| `bertscore_cv_kappa_mean_fold` | Mean per-fold Cohen's kappa for BERTScore against annotator1_label, constant within a generator. |
| `bertscore_cv_kappa_oof` | Out-of-fold Cohen's kappa for BERTScore against annotator1_label, constant within a generator. |

#### `meteor` family
| Column | Description |
| :--- | :--- |
| `meteor` | METEOR score between the answer and the reference(s), maximum over '\|'-separated references. Computed with NLTK defaults and WordNet synonym matching. |
| `meteor_label_oracle` | METEOR label at the oracle threshold: 1=hallucination, 0=correct. The oracle threshold is tuned on the full evaluation set and is an optimistic diagnostic bound, not a deployable procedure. |
| `meteor_label_cv` | METEOR label at the cross-validated threshold: 1=hallucination, 0=correct. Concatenates the out-of-fold predictions of a 3-fold split; the threshold is selected on the two training folds and applied to the held-out fold. |
| `meteor_tau_oracle` | Oracle threshold used for METEOR, selected per generator to maximise agreement with annotator1_label on the full evaluation set. |
| `meteor_tau_cv_mean` | Mean of the three per-fold METEOR thresholds, constant within a generator. |
| `meteor_tau_cv_min` | Smallest of the three per-fold METEOR thresholds, constant within a generator. |
| `meteor_tau_cv_max` | Largest of the three per-fold METEOR thresholds, constant within a generator. |
| `meteor_cv_fold` | Cross-validation fold assignment for METEOR (0, 1 or 2). Determines which fold the row was held out from. |
| `meteor_cv_kappa_mean_fold` | Mean per-fold Cohen's kappa for METEOR against annotator1_label, constant within a generator. |
| `meteor_cv_kappa_oof` | Out-of-fold Cohen's kappa for METEOR against annotator1_label, constant within a generator. |

#### `bartscore_r2h` family
| Column | Description |
| :--- | :--- |
| `bartscore_r2h` | BARTScore-CNN mean token log-likelihood of the answer given the reference, maximum over '\|'-separated references. Unbounded and negative. |
| `bartscore_r2h_label_oracle` | BARTScore ref->hyp label at the oracle threshold: 1=hallucination, 0=correct. The oracle threshold is tuned on the full evaluation set and is an optimistic diagnostic bound, not a deployable procedure. |
| `bartscore_r2h_label_cv` | BARTScore ref->hyp label at the cross-validated threshold: 1=hallucination, 0=correct. Concatenates the out-of-fold predictions of a 3-fold split; the threshold is selected on the two training folds and applied to the held-out fold. |
| `bartscore_r2h_tau_oracle` | Oracle threshold used for BARTScore ref->hyp, selected per generator to maximise agreement with annotator1_label on the full evaluation set. |
| `bartscore_r2h_tau_cv_mean` | Mean of the three per-fold BARTScore ref->hyp thresholds, constant within a generator. |
| `bartscore_r2h_tau_cv_min` | Smallest of the three per-fold BARTScore ref->hyp thresholds, constant within a generator. |
| `bartscore_r2h_tau_cv_max` | Largest of the three per-fold BARTScore ref->hyp thresholds, constant within a generator. |
| `bartscore_r2h_cv_fold` | Cross-validation fold assignment for BARTScore ref->hyp (0, 1 or 2). Determines which fold the row was held out from. |
| `bartscore_r2h_cv_kappa_mean_fold` | Mean per-fold Cohen's kappa for BARTScore ref->hyp against annotator1_label, constant within a generator. |
| `bartscore_r2h_cv_kappa_oof` | Out-of-fold Cohen's kappa for BARTScore ref->hyp against annotator1_label, constant within a generator. |

#### `bartscore_h2r` family
| Column | Description |
| :--- | :--- |
| `bartscore_h2r` | BARTScore-CNN mean token log-likelihood of the reference given the answer, maximum over '\|'-separated references. Unbounded and negative. |
| `bartscore_h2r_label_oracle` | BARTScore hyp->ref label at the oracle threshold: 1=hallucination, 0=correct. The oracle threshold is tuned on the full evaluation set and is an optimistic diagnostic bound, not a deployable procedure. |
| `bartscore_h2r_label_cv` | BARTScore hyp->ref label at the cross-validated threshold: 1=hallucination, 0=correct. Concatenates the out-of-fold predictions of a 3-fold split; the threshold is selected on the two training folds and applied to the held-out fold. |
| `bartscore_h2r_tau_oracle` | Oracle threshold used for BARTScore hyp->ref, selected per generator to maximise agreement with annotator1_label on the full evaluation set. |
| `bartscore_h2r_tau_cv_mean` | Mean of the three per-fold BARTScore hyp->ref thresholds, constant within a generator. |
| `bartscore_h2r_tau_cv_min` | Smallest of the three per-fold BARTScore hyp->ref thresholds, constant within a generator. |
| `bartscore_h2r_tau_cv_max` | Largest of the three per-fold BARTScore hyp->ref thresholds, constant within a generator. |
| `bartscore_h2r_cv_fold` | Cross-validation fold assignment for BARTScore hyp->ref (0, 1 or 2). Determines which fold the row was held out from. |
| `bartscore_h2r_cv_kappa_mean_fold` | Mean per-fold Cohen's kappa for BARTScore hyp->ref against annotator1_label, constant within a generator. |
| `bartscore_h2r_cv_kappa_oof` | Out-of-fold Cohen's kappa for BARTScore hyp->ref against annotator1_label, constant within a generator. |

### 4. NLI entailment baseline (6)

| Column | Description |
| :--- | :--- |
| `nli_entail_prob` | Entailment probability from the reference-entailment NLI baseline (fine-tuned T5-11B), with the reference as premise and the answer as hypothesis. |
| `nli_label_strict` | NLI label at the strict threshold tau=0.5: 1=hallucination, 0=correct. Operationalises reference faithfulness. |
| `nli_label_best_tau` | NLI label at the oracle threshold: 1=hallucination, 0=correct. |
| `nli_label_best` | Identical to `nli_label_best_tau` on every row. |
| `nli_tau_strict` | Strict NLI threshold, fixed at 0.5 for all rows. |
| `nli_tau_best` | Oracle NLI threshold, selected per generator to maximise agreement with annotator1_label. |

---

## Label Encoding

All human and automatic binary label columns use the same two-value scheme for a
decision, plus a shared `-1` sentinel where a column allows "not applicable / not
annotated":

| Value | Meaning |
| :---: | :--- |
| **1** | hallucination |
| **0** | correct / non-hallucination |
| **-1** | not annotated (only in `annotator2_label`) |

The encoding reserves `-1` for missing or refused outputs; in this
release, `-1` occurs only in `annotator2_label` (600 rows).

---

## Recommended Use

Use `human_judge_release_master.csv` as the authoritative row-aligned source for:

* Reproducing human-vs-judge agreement analyses.
* Computing Cohen's kappa.
* Comparing prompt variants.
* Comparing automatic judge families.
* Inspecting false positives and false negatives.
* Analyzing dataset- or generator-specific behavior.

For exact column meanings, always consult `human_judge_release_data_dictionary.csv`.
Do not modify, regenerate, or reformat `human_judge_release_master.csv`; treat it as
read-only, per the SHA-256 above.

## Acknowledgements

This work was partly supported by local authorities ("Business Finland") under grant agreement 23004 ELFMo of the ITEA4 programme, which funded three of the authors.

## License

The human annotations, judge labels, metric scores, and documentation produced
in this work are released under the Creative Commons Attribution-ShareAlike 4.0
International License (CC BY-SA 4.0): https://creativecommons.org/licenses/by-sa/4.0/

**Source datasets.** The questions and reference answers derive from the
datasets below and remain under their original licences. This release does not
relicense them.

| Dataset | Licence |
| :--- | :--- |
| HotpotQA | CC BY-SA 4.0 |
| TriviaQA | The University of Washington does not own the copyright of the questions and documents in TriviaQA (https://nlp.cs.washington.edu/triviaqa/); questions are redistributed for research use under the terms of their original rights holders. The TriviaQA code repository is licensed under Apache License 2.0. |
| TruthfulQA | Apache License 2.0 |

**Generated answers.** The `answer_text` values were generated by the models
below and remain subject to their terms.

| `model` value | Hugging Face model | Terms |
| :--- | :--- | :--- |
| `gemma-2-9b` | `google/gemma-2-9b-it` | Gemma Terms of Use (https://ai.google.dev/gemma/terms) |
| `llama3-8b` | `meta-llama/Meta-Llama-3-8B-Instruct` | Meta Llama 3 Community License |
| `mistral-7b` | `mistralai/Mistral-7B-Instruct-v0.3` | Apache License 2.0 |

Answers labelled `llama3-8b` were generated with Meta Llama 3. Section 1(b)(v)
of the Meta Llama 3 Community License states:

> v. You will not use the Llama Materials or any output or results of the Llama Materials to improve any other large language model (excluding Meta Llama 3 or derivative works thereof).

Source: https://raw.githubusercontent.com/meta-llama/llama3/a0940f9cf7065d45bb6675660f80d305c041a754/LICENSE
(checked 2026-09-30). This refers to the original Llama 3, not Llama 3.1 or
later. It is quoted for information; it is not an additional condition of the
CC BY-SA 4.0 licence above.

**Judge labels from Llama 3.** The labels `label_llama_3_8b_local_p1`, `label_llama_3_8b_local_p2` and `label_llama_3_8b_local_p3` are outputs of Meta Llama 3 8B used as a judge. They are released under CC BY-SA 4.0 together with the other judge labels; to the extent the Meta Llama 3 Community License applies to such outputs, its terms apply in addition.

**Judge models.** The judge label columns contain numeric labels only (no judge
text output). The judges were: `gemma-2-9b-it`, `Qwen2.5-7B-Instruct` and
`Meta-Llama-3-8B-Instruct` (run locally), and `gpt-5-mini`, `gpt-5-nano`,
`gpt-5.4` and `claude-opus-4-7` (via API).

Source-dataset notices and full licence texts: https://github.com/jova486/LPHB/blob/main/THIRD_PARTY_NOTICES.md
