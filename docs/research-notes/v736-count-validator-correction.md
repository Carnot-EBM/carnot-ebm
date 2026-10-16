# V736 count validator correction — 2026-10-16

Exp8567 repairs the classification of proved discrete event cardinalities. It
uses CPU arithmetic and sealed upstream artifacts. It loads no model and makes
zero current LLM calls. Generalization and learning-benefit scores remain zero.

Exp8566 is pinned at SHA256
`ef6064315aee6e7f2ad32fff2d4163075ce852d54b82236ae9b342480aa4041b`.
Its old validator is pinned at
`16ad1fa750f505182dd996fdd500663d0c777a55a4cd080f377db9ff18cec43c`.
The preserved source snapshot, exact correction diff, source/test hashes,
commands, exits, clocks and log hashes live below
`results/raw/experiment_8567_v736_count_validator_correction/`.

The shared change is confined to `check_tautology` and its count-proof helper.
A producer may attach `discrete_count_proof` with version
`discrete-counter-proof-v1`, bounded tables and claims bound to unchanged numeric
metric fields. Each claim supplies an operation, unit, exact nonnegative JSON
integer, table identity and table hash. Unique bounded namespace/event identities,
boolean event markers, finite before/after parameters, assignment indices and
work limits must pass the frozen checks. Two independent reductions must agree.
Both equal metrics must bind the same table hash. This yields an annotated
TAUTOLOGY warning for a structural count relation. Every metric stays visible.
No name, verdict, artifact identity, float, string or rounded value grants proof.
Invalid or absent proof leaves the old critical path in place. Other audit
checks and severity thresholds remain unchanged.

The sealed corpus has 17 controls. It includes exact equal counts 2400, 1800 and
47472, unequal counts, malformed claims and copied continuous loss/latency values.
Additional tests cover duplicate and inactive events, missing or malformed arrays,
row identity, bounded work and self-consistently rehashed inconsistent evidence.
Original Exp581 and Exp696 findings must remain byte-for-byte equal. The private
Exp8555 adapter retains all original metric names and values and diagnoses its
seven count relations through the corrected check. It keeps the disqualified
verdict. The original primary bytes and seven historical flags remain intact.
Removing the correction restores the count false positives; corrupt evidence
remains critical under both versions.

Qualification requires all scoped tests, real CLI controls, strict mypy, Ruff
check/format, spec references and 100 percent newly owned statements. Coverage
includes genuine publication and rejection children. Original node IDs precede
disjoint bounded children; their terminal outcomes must form the exact union.
Private E2E-018 and E2E-021 exercise authority and publication consumers. Fresh
replay rejects absent inputs, failed owned validation and rehashed summaries.
Final bytes pass replay, adversarial verification and strict row consistency
before atomic publication. The live downstream gate and document readers then
consume the published primary. The primary and its bound sidecar contain the
terminal readiness, coverage evidence and all findings for this invocation.

Exp8530 and Exp8555 remain disqualified. Exp8529's source null, Exp8553's
independent-label rejection, Exp8558's native parity, Exp8559's inference null,
all historical quarantines and the original scientific methods remain intact.
The completion archive ends at V734; V735 primary/conductor outcomes retain
authority. The active roadmap and conductor are unchanged. Same-roadmap outputs
are future dependencies. A subsequent task must make a fresh producer artifact
through the corrected validator. This correction alone cannot promote learning.
The conductor owns status, changelog and traceability reconciliation.

A development coverage attempt on the earlier replay implementation reached its
300-second bound and failed. Its log is retained as
`development_coverage_timeout.stdout`; it grants no qualification. Replay now
rejects inconsistent envelopes before reconstructing original operands and
shares table work across equal-value groups. The final invocation reruns its
required checks against the corrected source and records their own receipts.
