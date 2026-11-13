# V774 adjudication methods, 2026-11-13

REQ-REPORT-9070-V774 and REQ-VERIFY-9070-V774 define this methods audit.
The existing V774 scan in research-references.md was ingested before this update.
No model, embedding extractor or fitted head runs in this task.

The focused update retrieved four primary arXiv papers. Semantic Scholar returned
HTTP429 on both allowed attempts. Requests were sequential, each with a30-second
timeout and at most two attempts. Release and literature acquisition took6.29
seconds after the initial0.40-second GitHub/raw preflight. The300-second total
budget was not exhausted. Exact bodies, URLs and request receipts are retained
privately under `/tmp/carnot-9070-private/`. All five annotation files matched
the Git blob identities from the pinned GitHub tree. SHA256 pins bind replay.

| Method | Primary source | Frozen use | Failure mode |
| --- | --- | --- | --- |
| Multi-perspective adjudication | [Beyond Majority Vote](https://arxiv.org/html/2609.03953v1), [author release](https://github.com/isi-vista/mdhjudgments/tree/30d9349a2a8d6a7395681cfbd23cc8116af5a2da) | Preserve both medical passes; first-pass and fact-checker comparisons are descriptive | The adjudication panel was selected. Agreement does not certify medical truth. Anonymous pass IDs do not count distinct people. |
| Target taxonomy | [What Hallucinations Have We Measured?](https://arxiv.org/html/2610.38520v1) | Define factual-error agreement and retain uncertainty separately | Archived responses cannot identify intervention-based behavioral categories. |
| Strong linear probes | [Weakly Supervised Distillation](https://arxiv.org/html/2604.06277v2) | Full-embedding and projected ridge logistic baselines must converge | A weak or unconverged baseline could manufacture a spline advantage. Projection can discard useful signal. |
| Weak supervision | [Weakly Supervised Distillation](https://arxiv.org/html/2604.06277v2) | LLM judge candidates stay separate from original human targets | Weak labels are dependent on their producer and cannot supply missing human annotations. |
| Calibration | [Continual Calibration](https://arxiv.org/html/2604.23987v1) | Choose temperature on CAL only; retain typed thresholds and full-service cost boundaries | Calibration can degrade under changing data. This static panel does not test continual adaptation. |

These are protocol choices informed by primary papers, not measured Carnot
benefits. The numerical spline machinery already qualified by Exp8986 remains
available. This task adds no duplicate optimizer or alternative embedding path.

The original instructions in AppendixH distinguish major, minor, no-error and
needs-research answers. We independently map original Q1 major/minor to1 and
no_hallucination to0. Needs_research and unresolved values remain unknown.
The author's aggregate_judgment is not imported as a replacement target.
Composite outcomes can depend on reviewer comments. The mapping is deliberately
stricter than importing that composite. Original targets remain in sealed JSONL.

The five exact exports each contain541 response records and4,377 sections.
They have312 distinct question strings. Neither number proves independent
base-question families. IDs contain source-version, question, generator and
template fragments. The author model and conversion code authenticate response
and section joins but provide no construction lineage across source versions.
Equal normalized questions and guessed ID prefixes cannot close that gap.
The retained exposure ledger also leaves family-level overlap unresolved.
Thus every section stays excluded before outcome interpretation. No folds are
allocated and no label-based cohort sufficiency is measured. This is an
external authority block, not an insufficient-cohort null. MDH-Bench stays closed.

The frozen future protocol uses all eligible families, five SHA256-ordered outer
folds and salt `v774-mdhj-v1`. Round-robin assignment needs60 families and12
per test fold. Every fifth hash-ordered non-test family is CAL. FIT and CAL each
need five families representing each original judgment class. There is no
outcome-stratified reshuffle and no reuse of the failed400-group scope.

Features are mean-pooled final-layer Qwen Q4_K_M embeddings of question and full
response, with explicit target-section offsets. The4096-token maximum permits
no truncation. References, labels, judge IDs, explanations, evidence excerpts
and generator IDs are denied. A Gaussian32-dimensional projection uses seed
774070. Scaling is fitted on FIT only to[-3,3]. Prevalence, full ridge logistic,
projected ridge logistic and additive cubic spline energy share the prescribed
inputs. Ridge0.01 excludes the intercept. Cubic knots are[-3,-2,-1,0,1,2,3].
Convergence requires gradient infinity norm<=1e-6 within2000 iterations.
CAL selects temperature from[0.5,1,2,4], with ascending ties. Accept/reject
thresholds are0.1/0.9. No tokenizer or model is loaded to enact this freeze.

Training uses soft-target log loss. Evaluation preserves two original judgments:
average their squared prediction errors, average within each base family, then
average across families. Squared error against their mean is a different metric.
Three paired improvements compare spline with prevalence, projected linear and
full linear. The paired family bootstrap uses10000 draws and seed774077.
All three lower bounds at one-sided tail0.05/3 must exceed zero for scoped
pilot benefit. CI95 uses separate two-sided percentile endpoints. Intervals
condition on frozen out-of-fold predictions. They do not certify training-set
variability or population-wide medical truth. First-pass/fact-checker ranking
reversals on the identical overlap remain descriptive.

Private negative controls deny label fields and generator metadata. They also
reject mismatched IDs, duplicate sections and changed replay summaries. Real
validation receipts and zero-exclusion statement coverage qualify the producer.
They cannot qualify the source or a later learner. The conductor owns ops,
status and traceability reconciliation, as the task's final instruction states.
