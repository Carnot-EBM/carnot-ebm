# V681 method map — 2026-09-28

Exp7837 registers methods before V681 science. The immutable V681 authority
copies are in `v681-authority-snapshots/`. The fourteen-row contract checks
planning agreement only. All 640 RAGTruth families are historically exposed
development data. Current role separation reduces within-run leakage but does
not make evaluation64 a fresh generalization sample.

| Primary method | Local adaptation and control | Claim limit |
|---|---|---|
| [Length confounding, arXiv:2508.08285](https://arxiv.org/html/2508.08285v2), evaluation and §6 | Exp7848 fits a length-only control, freezes quartile bins on fit256, and permutes source within answer-length bins and a 25% source-length caliper. Compare intact-family Brier and typed cost against energy heads. | A length effect can explain apparent detection. ROUGE and the model's own judgment cannot replace independent human labels. Source-permuted pairs have unknown labels. |
| [Input evidence alignment, arXiv:2608.15804](https://arxiv.org/html/2608.15804v1), task and method | Exp7838 preserves complete answer sentence offsets and source witness spans. Exp7842 removes a proposed witness and a disjoint length-matched sentence, with unchanged question and answer. | Witness-removal sensitivity is distinct from correctness. No reproduction of the paper's trained masked-token encoder is claimed. |
| [Capacity-constrained delayed feedback, arXiv:2606.11711](https://arxiv.org/html/2606.11711v1), §2–4 | Exp7843 records each prediction before feedback, a finite pending set, admission probabilities, dropped observations and matched feedback budgets. Compare with frozen, complete-static and shuffled-past controls. | The discrete predicate bank is not convex OCO; no paper regret bound transfers. A dropped pending label cannot be used later. |
| [Noisy constraint values, arXiv:2609.06921](https://arxiv.org/html/2609.06921v2), §2–4 | Exp7843 records cumulative signed error budget and worst-window overspend after delayed updates. | Classification labels do not establish unbiased finite-variance feedback assumptions. The paper's bound is not a local guarantee. |
| [Distributional EBM, arXiv:2605.18871](https://arxiv.org/html/2605.18871v1), §3 | Exp7844 compares calibrated disagreement with confidence and hash-selected random abstention at the same retained count, thresholds frozen on policy64. | Three seeds of a small head are not five heterogeneous adapters. Low energy, agreement or fixture success does not establish independent verification benefit. |

## Frozen roles and decision protocol

The 640 families have fixed roles: fit256, tune64, policy64,
online_update96, online_admission64, evaluation64 and retention32. Family is
the independent unit. Views, sentences and seeds do not increase N. Invalid,
duplicate and unknown-label units remain visible. Seeds are 68101, 68102 and
68103. Nine Exp7840 arms are `response_set`, `local_set`, `augmented_set`,
`constrained_set`, `augmented_mlp`, `constrained_mlp`, `local_logistic`,
`source_erased_constrained_set` and `complete_static_constrained_set`.
Each receives sixteen epochs, learning rate 0.01 and at most 4096 parameters;
MLPs have width sixteen. The source-erased and complete-static heads train
separately. No fixture checkpoint initializes a natural fit.

Tune64 selects temperature on the fixed [0.25,4] grid. Costs are accept=5p,
reject=1-p and escalate=0.25, with ties escalating. Evaluation64 predictions
are sealed before labels join. Primary Exp7841 contrasts constrained_set
against augmented_set, constrained_mlp and local_logistic on both typed cost
and Brier. Its six tests use paired family randomization, 10000 paired-family
bootstrap draws and Holm correction. Required gains are cost >=0.02,
positive lower95 cost and Brier gains, coverage >=0.20 and Holm p<=0.05.
Source value also requires positive Brier lower95 versus the separately fit
source-erased arm. A null result is terminal, not a reason to change controls.

Exp7848's preregistered length control uses log1p answer/source byte lengths
and their ratio, ridge regularization 1.0 and 200 deterministic steps on
fit256. It tunes on tune64 and scores evaluation64 against prevalence and
always-escalate references. The intact-family energy interpretation requires
cost gain >=0.02 and positive Brier/cost lower95 against length, plus the
source-erased advantage. This veto cannot rescue a failed primary contrast.

Exp7843's capacity intervention records pending occupancy and admission
probability before feedback. It reports dropped and delayed labels, next-query
and next-block decision changes, final and worst-window false-accept debt,
and cold-restart retention. Feedback/update budgets are matched against frozen,
complete-static and shuffled-past controls. Repeated views and seeds are
paired observations, not new families.

The six acceptance gates have distinct reasons: validity proves source and
label custody; readiness proves runnable qualified interfaces; probability
quality tests Brier calibration; decision benefit tests cost after controls;
retention tests delayed learning after restart; efficiency tests complete
service cost. Unmeasured scientific gates remain null in Exp7837. The two
infrastructure slots are Exp7837 and Exp7839. Exp7845 is the separate ARC
generalization slot; Exp7843 is the separate continuous-learning slot.

## V680 failure to remedy

| Historical failure | V681 remedy and preserved limit |
|---|---|
| Exp7823 prior-failure and gate audit failures | Declare every matching predecessor, including Exp7832's scope; compare literal, JSON, staged and active authorities before readiness. The failed Exp7823 artifact remains unchanged. |
| Exp7824 27% coverage and required full-suite timeout | Exp7838 qualifies a smaller source boundary with explicit unit and real CLI coverage. The Exp7824 failure remains failed. |
| Exp7828 zero-exit import receipt without `resolved_imports` | Exp7839 emits actual worktree module paths; Exp7837 checks that same receipt shape. The old zero-exit receipt remains insufficient. |
| Exp7835 integer/slug confusion, coverage and full-suite failure | Use numeric `experiment_id` and separate exact `task_id`; cold replay rejects a slug in the integer field. Historical obligations remain failed. |

`retire_if_same_verdict` applies only to each listed repeated scope. The
contract, fixtures and literature grant no scientific readiness, publication,
roadmap activation or generator change.
