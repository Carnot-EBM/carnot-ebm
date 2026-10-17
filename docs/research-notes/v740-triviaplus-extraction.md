# V740 Trivia+ complete-source extraction — 2026-10-17

Exp8620 authenticates the qualified Exp8615 methods primary, Exp8617 release
primary and original author Git blobs. It uses the sealed label-blind roster of
128 normalized questions and retains all 190 sibling answers, including any
siblings outside the roster's selection split. Source and answer text remain in
private `/var/tmp` storage; repository primitives contain measurements and hashes.

The predictor child receives only `opaque_id`, complete `source_bytes`, and
complete `answer_bytes`. It runs the shipped `extract` and
`source_alignment.prepare`; the 128-window and 16-answer-unit limits remain
unchanged. The parent writes the feature seal before reading human annotations.
No generator or energy head runs. Current language-model calls are zero.
The referenced nine-input109-coefficient Exp8431 head is authenticated historical
provenance only: its required released p2 confidence q is absent from Trivia+.
Noisy labels and a constant0.5 cannot replace q.

All 190 answers were measured. There are 135 usable answers across90 questions
(71.0526% answer coverage). The remaining55 answers exceed128 source windows;
none are truncated or discarded. Source window counts range from1 to2098.
The six registered cells contain99 answers at1..32 windows/<=16 answer units,
36 at33..128/<=16, and55 at>128/<=16. All >16-answer-unit cells are empty.
The first two occupied cells have full extraction coverage; the last has zero.
Usable questions supporting response classes0/1 number51/53; a question can
support both classes through different siblings and is counted once per class.

Fixed lexical scores are descriptive exploratory rankings. The question-cluster
bootstrap resamples whole sibling groups10000 times, seed7398607. All10000 draws
are valid for each primary score. No score is selected as a winner.

| Fixed score | AUROC | Question-cluster95% interval |
| --- | ---: | --- |
| numeric_mismatch maximum | 0.602408 | [0.511763, 0.702183] |
| negation_mismatch maximum | 0.489395 | [0.395000, 0.576707] |
| uncovered_fraction mean | 0.709567 | [0.616873, 0.793600] |
| 1-overlap mean | 0.709456 | [0.616848, 0.793600] |

The last two scores express the same lexical reduction algebraically. Floating
point averaging produces tiny tie differences; these are not separate evidence
for a detector. A lexical conflict does not certify a factual error.

Human labels are joined only after sealing scores. Raw annotation arrays are
retained separately from recomputed strict majority labels. Ties use the release
severity priority supplementary, supports, not-mentioned, contradicts. A response
is unfaithful if any aggregated sentence is unsupported or contradicted, and is
missing if any sentence lacks an aggregated vote. Automatic labels are unused.

There are97 unanimous responses (69 usable,56 usable questions) and93 disputed
responses (66 usable,52 usable questions). Both overall strata meet the interval
support floor. Numeric-mismatch AUROC is0.669052 [0.545322,0.798185] in unanimous
responses and0.523148 [0.406703,0.649887] in disputed responses. Uncovered-fraction
AUROC is0.690966 [0.545449,0.822052] and0.733333 [0.598214,0.851436], respectively.
These descriptive intervals do not establish differences between strata.
Disagreement changes the strength of numeric-ranking support; it does not change
the conclusion that calibration and factual-error claims remain unsupported.

Unsupported versus contradicted cross-tabs are retained, including abstentions.
Only3 disputed unsupported responses are usable; neither unsupported stratum has
adequate comparative support. Contradiction-only strata contain only class1 and
cannot support a two-class AUROC. These intervals remain explicitly absent.

A complete honest census qualifies extraction_audit_ready_score even when
individual rows abstain. Long-context coverage, annotation sensitivity and absent
p2 q prevent the proposed calibration study. Historical corpus overlap remains
unknown, so both generalization-benefit scores stay zero. This is a historical
Trivia+ faithfulness audit and makes no claim about current Qwen performance.

Owned validation includes focused tests, actual predictor and rejection children,
Coverage.py executable statement accounting, Ruff check/format, strict mypy,
spec references, private E2E-018/021, original node unions, cold replay, unchanged
adversarial/row validators and atomic primary publication. Exact commands, exits,
durations, stdout/stderr hashes, original operand hashes and bound reader receipts
are stored under `results/raw/experiment_8620_v740_triviaplus_extraction/`.

Exp8603's historical disqualification remains intact. No deadline behavior,
learner, generator weights, roadmap, conductor or shared validator was changed.
The V733 source-information, V735 cold-cost and V737 ARC nulls remain closed;
V738 native throughput does not satisfy NFR01. Ops and traceability reconciliation
remain conductor-owned.
