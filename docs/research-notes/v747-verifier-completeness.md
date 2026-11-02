# V747 constructed verifier completeness audit

This audit tests executable relation verifiers against constructed faults. It
loads no language model. It makes no claim about natural-language accuracy,
learning benefit, generalization, batch speed or physical FPGA execution.

The method adapts paired valid alternatives from
[VerifierBench](https://arxiv.org/abs/2610.28679) and behavioral perturbations from
[ReLoop](https://arxiv.org/abs/2602.15983). This is a local constructed audit,
not a reproduction of either benchmark. Their reported gains are not Carnot gains.

The source authority is the pinned V744 protocol, original integer evaluator and
120 original relational worlds. Exp8712's declared symbolic-control field and
terminal acceptance supply the current branch gate. Missing generator
probabilities from Exp8701 do not prevent this separate symbolic audit. Historical
quarantines and qualified nulls remain authoritative.

Six variants are frozen: correct, reject-all, accept-all, omitted required
predicate, reversed arguments and missing-as-false. Every fourth ID in the
original roster is reserved for evaluation. There are90 implementation worlds,
30 evaluation worlds and720 total variant/world units. Variant repeats do not
increase the120 independent constructed cases. These faults do not estimate
language-model error frequencies.

Question predicates are grounded from the question's family and entity roster.
Candidate answers do not define the query inventory. A separate explicit
membership evaluator computes all probe labels before the tested variants run.
The correct variant uses the unchanged shipped relation evaluator.

The baseline has truth, equivalent, invalid, unknown and deleted-record probes.
The extended suite includes these five and adds record-order reversal, a
bijective entity rename, irrelevant facts, removal of a required premise,
perturbation of a required role and reversal of record arguments. Every probe
retains its records, query, independent expected label, observed decision and
unit cost. The baseline probes are executed once and reused in both decisions.
The extended suite therefore costs11 probes per unit, versus5 for the baseline.
This comparison does not establish superiority at equal probe budgets.

Readiness requires all intended units, correct handling of valid and unknown
controls, explicit failure of reject-all and accept-all, authenticated operands,
complete owned coverage and accepted terminal validation. Readiness does not
require an incremental detection gain. Oracle-dependent positive evidence is
circular_positive; a qualified zero incremental gain is null.

Raw invocation evidence retains the premeasurement freeze, every executed probe,
disjoint child SQLite and JUnit, collected node inventory, commands, exits,
clocks and terminal checks. Cold replay rebuilds labels and measurements from
original authority. Report and primitive rehashing cannot grant new authority.
Publication uses the unchanged atomic primary publisher, adversarial verifier and
strict row consistency validator. Private E2E-018/021 exercise the live readers.

Passing these probes does not prove complete semantics. Single-predicate worlds
cannot expose a conjunction-omission defect. A structural transform that changes
records and queries together can preserve a decision while changing the relation
asked about. Explicit role controls are therefore distinct from covariance
checks. This audit does not change production verification or expand a canary.

Ops and traceability reconciliation is reserved for the conductor, as requested.

The first qualification attempt retained a critical TAUTOLOGY finding: a defect
count of120 coincided with the independent world count of120. The scanner's
existing count contract requires explicit count field names. The report now uses
`*_count` names for exact integer detection counts, with unchanged reductions
and raw rows. No validator, threshold or measured value was changed. The original
critical finding remains in that invocation's private and terminal evidence.
The witness regression also requires an explicit present-but-wrong role, rather
than retaining only the first missing-premise failure for an omitted predicate.

Measured outcome,2026-11-02: all720 units completed. On30 sealed worlds, the baseline detected120/150 constructed faulty units; the extended suite detected130/150. The180 variant/world comparisons cost900 baseline probes versus1980 extended probes. All correct-variant controls passed. Readiness is1; verdict is circular_positive. All six owned tests and two private E2E-018/021 reader tests passed. Coverage is233/233 owned statements. Ruff check/format, strict mypy, spec tracing, fresh cold replay, adversarial verification and strict row validation passed. No current adversarial findings remain. Original failed qualification evidence is retained.
