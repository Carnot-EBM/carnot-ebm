# V748 verification methods — 2026-11-02

Exp8726 freezes methods for Exp8729 and Exp8730. It loads no model, performs
no generation and never calls CUDA. Original Exp8712 and Exp8715 result bytes,
labels, assertions and publication validators remain unchanged.

| Primary method | Local adaptation | Executable check |
|---|---|---|
| [P(Yes), 2610.25474](https://research.tearlabs.ai/pyes-search/) | Fixed support question; normalize observed first-token Yes/No log probabilities. Missing logits block scoring. No search or confidence guessing. | `test_protocol` checks stable normalization and rejects missing/nonfinite logits. |
| [VerifierBench, 2610.28679](https://arxiv.org/html/2610.28679v1) | Include correct alternatives and invalid cases; both verifier schedules receive five calls. | `test_protocol` checks full invocation counts and correct-variant acceptance. |
| [Verification Without Sufficiency, 2608.00585](https://arxiv.org/html/2608.00585v1) | Remove a required premise while retaining the proposed answer and its full query. Unknown remains unknown. | `test_protocol` checks every missing-premise label. |
| [On-chip KAN, 2602.02056v4](https://arxiv.org/html/2602.02056v4) | Keep input work, calls and elapsed time separate; transfer no hardware speed claim. | `test_protocol` checks five actual calls and positive measured elapsed time. |

Requests were sequential. Retained response bytes, UTC times, durations, hashes,
URLs and failures are in
`results/raw/experiment_8726_v748_verification_methods/literature/`.
The P(Yes) arXiv HTML endpoint returned404; its full author methods and linked
`tear-labs/pyes-search` verifier code were retrieved. The VerifierBench author
project says “Code soon”; no code release was substituted. The sufficiency
author repository was rechecked. No author code link was identified in the
KAN primary text. No dependencies were installed and no research sweep ran.

The original120 worlds, six variants and eleven transforms come from qualified
Exp8715 and its original upstream custody. The original90 development/30
evaluation IDs are retained, with all120 labeled exposed-development because
V747 already evaluated them. The two schedules are fixed in this order:

- Baseline: truth, equivalent, invalid, unknown, deleted_records.
- Targeted: truth, invalid, entity_bijection, required_premise, required_role.

Every variant/world/arm executes all five calls, even after detecting a defect.
The1440 units therefore make7200 variant invocations. Per-unit costs retain real
elapsed seconds and the number of query predicates supplied. Predicate input
volume is not internal comparison or instruction count. These controls establish
protocol execution; Exp8729 owns utility analysis. Equal calls imply no equal
wall time and no superiority conclusion is made here.

The pilot has the first four IDs in each of three families:12 worlds, each with
supported, contradicted, equivalent and missing-premise views, totaling48 fixed
typed prompts. Explicit entity/family/fact evidence and proposed answer predicates
are bound to the original world hash. Labels remain outside prompt text. The
binary target is sufficient support, while truth retains its three values.
Exp8730 owns actual Qwen calls and scores: temperature0, at most8 output tokens
per prompt and384 total. Frozen thresholds are0.5 for support and0.9 for
high-confidence errors; metrics include support Brier, false support, equivalence
and high-confidence errors. No score was measured by Exp8726.

The supplied CUDA repair reader and qualified cached reader run without loading
a device library or probing a GPU. An absent causal repair keeps runtime
readiness zero and skips the GPU pilot independently of CPU methods readiness.
All14 complete V748 tasks, visible table checks and independent canonical digest
are authenticated separately from original scientific methods.

Publication requires bounded focused tests,100 percent owned statement coverage,
Ruff, strict mypy, scoped spec tracing, retained child exits/nodes/SQLite/JUnit,
cold replay, unchanged primary validation, adversarial verification and strict
row consistency. Private CLI success, blocked, invalid date/path and consistently
rehashed tampering controls cover the applicable private-output E2E behavior.
The final artifact carries exact receipts and outcomes. Ops/status/traceability
reconciliation belongs to the conductor after exit.
