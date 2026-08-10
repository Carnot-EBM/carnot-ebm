# Constraint Verification Capability Specification

**Capability:** constraint-verification
**Version:** 0.1.0
**Status:** Draft

## Overview

Defines deterministic audits for executable constraint-verification corpora.
These audits decide whether a frozen candidate pool is usable for later
selector experiments before any hidden-state extraction is attempted.

## Requirements

### REQ-CONSTRAINT-6274: Bounded ASP Semantic Energy Compiler

The repository SHALL provide an executable bounded ASP semantic compiler for
Exp6274. The compiler SHALL accept only explicit ground facts, grounded normal
rules, default negation in rule bodies, integrity constraints, and standalone
bounded cardinality choice rules. It SHALL reject unsupported syntax before any
energy terms are built.

The compiler SHALL emit inspectable energy terms for facts, normal rules,
integrity constraints, bounded cardinality rules, and stable-model support. For
every enumerated state it SHALL report per-rule violation receipts that name the
local failed rule or semantic support check.

The Exp6274 harness SHALL evaluate at least 40 trusted formal fixtures across
graph coloring, scheduling, non-monotonic defaults, contradictions, and positive
or negative controls. Every fixture SHALL have a bounded finite state space that
is exactly enumerated. The zero-energy states emitted by the compiler SHALL be
compared by set equality against an independent ASP solver answer-set list.

The terminal artifact SHALL be
`results/experiment_6274_asp_energy_semantic_compiler.json`. It SHALL state the
paper source and claim boundary, supported and unsupported ASP constructs,
source and fixture hashes, independent solver version and receipts, fixture
counts, exact state counts, per-fixture answer sets, zero-energy states,
semantic parity, rule-local violation evidence, all required controls, test
commands, exit codes, reproducibility checksum, `verifier_is_oracle=true`, and
an honest verdict. The artifact SHALL NOT claim a learned verifier or an
oracle-distinct verifier moat.

### SCENARIO-CONSTRAINT-6274-SOLVER-PARITY: Energy Matches ASP Answer Sets

Given trusted bounded ASP fixtures in the supported subset,
When Exp6274 compiles each fixture and enumerates all possible atom states,
Then the zero-energy states exactly equal the independent solver answer sets
for every fixture.

### SCENARIO-CONSTRAINT-6274-FAIL-CLOSED: Unsupported Syntax Is Rejected

Given ASP text with variables, disjunction, optimization, arithmetic terms, or
unsupported aggregates,
When the bounded compiler receives that text,
Then it rejects the program before energy construction and reports the failing
syntax class.

### SCENARIO-CONSTRAINT-6274-LOCAL-RECEIPTS: Violations Name Local Causes

Given an enumerated state that violates a fact, normal rule, integrity
constraint, bounded cardinality rule, or stable-model support check,
When the compiler decomposes energy for that state,
Then the receipt names the violated rule or semantic support check and gives a
non-zero local energy contribution.

### REQ-CONSTRAINT-VERIFY-6175: CCTU Headroom Audit Fail-Closed Gate

The repository SHALL provide an Exp6175 audit over the Exp6174 CCTU K8 pool.
The audit SHALL verify the Exp6174 structured gate, upstream bank/split/
validator/preregistration hashes, raw-before-label receipts, exact validator
version, calibration and held seals, K completeness, no-retry receipt,
preregistered gates and power plan, output paths, exclusions, and protected
files before any headroom metric is trusted.

The audit SHALL revalidate every calibration and held label from immutable raw
completion text using the Exp6173 exact validator. It SHALL report all-sample
and parseable denominators, per-candidate competence against the exact floor,
partial step satisfaction, violation taxonomy, duplicate clusters,
family/constraint-count strata, and parseability. Headline denominators SHALL
retain parse failures, refusals, timeouts, truncations, duplicates, and exact
validator failures.

The audit SHALL tune only the preregistered oracle-blind consensus rule on
calibration rows using normalized action/terminal-outcome clusters. Consensus
selection SHALL NOT use held labels, hidden states, arbitrary row identifiers,
answer positions, sample indexes, or validator labels at selection time.

The audit SHALL compute oracle@8, tuned-consensus accuracy, oracle-minus-
consensus delta, case-clustered intervals, consensus-wrong/oracle-right group
count, and error-diversity metrics. Held processing SHALL emit only aggregate
qualification fields plus a sealed held row-label hash, never held row labels
or per-row held outcomes.

The audit SHALL set `phase_d_headroom_ready_score` to bare `1.0` only when all
preregistered conjuncts pass: Exp6174 gate/preconditions, parseability,
above-floor competence, below-saturation error, oracle-minus-consensus at least
0.10 with lower CI above zero, at least 30 selectable minority groups, and
family support. If any conjunct fails, the artifact SHALL set `status` to
`retired`, `future_rows_allowed_by_this_artifact` to bare `false`, and
`honest_verdict` to a `retired:` terminal prefix naming the failed conjuncts.

The terminal artifact SHALL be
`results/experiment_6175_cctu_headroom_audit.json` and SHALL include the task's
required schema fields, including
`inference_substrate="deterministic_exact_tool_trace_headroom_audit"` and
`verifier_is_oracle=true`.

### REQ-CONSTRAINT-VERIFY-6175-AUTHENTICITY: Upstream Pool Authenticity Gate

Exp6175 SHALL fail closed unless the Exp6174 structured gate passed, current
upstream bank/split/validator/preregistration bytes match the hashes declared by
Exp6174, current raw-trace and label-sidecar bytes match Exp6174 receipts, raw
rows were committed before validation, K=8 rows exist for every frozen case, no
correctness-conditioned retry/replacement occurred, calibration and held seals
remain intact, preregistered gates and power are present, output paths are
writable, exclusions are present, and protected files are byte-stable.

### REQ-CONSTRAINT-VERIFY-6175-PARSEABILITY: All-Sample Parseability Accounting

Exp6175 SHALL compute parseability over every frozen candidate row. Headline
denominators SHALL retain unparseable completions, truncations, refusals,
timeouts, duplicates, and exact-validator failures. Parseability SHALL pass only
when the preregistered minimum is met; unparseable rows SHALL retire the domain
rather than being filtered out of the headline denominator.

### REQ-CONSTRAINT-VERIFY-6175-EXACT-FLOOR: Exact Floor Provenance

Exp6175 SHALL copy the exact random executable-plan floor from the Exp6173
preregistration, report its value and provenance, and affirm that no
finite-choice or answer-position floor is used.

### REQ-CONSTRAINT-VERIFY-6175-COMPETENCE: Candidate Competence Gate

Exp6175 SHALL compute per-candidate all-sample exact-validator accuracy and a
case-clustered interval. Competence SHALL pass only when the clustered lower
bound is strictly above the exact floor.

### REQ-CONSTRAINT-VERIFY-6175-UNSATURATION: Imperfect-Pool Gate

Exp6175 SHALL measure whether the candidate pool is competent but imperfect by
checking that candidate accuracy and tuned consensus are below the
preregistered saturation limits. This gate SHALL NOT rescue a failed
parseability, competence, headroom, minority, or family-support conjunct.

### REQ-CONSTRAINT-VERIFY-6175-CONSENSUS: Oracle-Blind Consensus Freeze

Exp6175 SHALL tune only the preregistered oracle-blind consensus family on
calibration rows using normalized action and terminal-outcome clusters. The
selection rule SHALL NOT use held labels, hidden states, arbitrary identifiers,
answer positions, sample indexes, or exact-validator labels at selection time.

### REQ-CONSTRAINT-VERIFY-6175-ORACLE-K: Oracle@K Headroom Measurement

Exp6175 SHALL compute oracle@8 as the case-level accuracy achieved when any of
the frozen K candidates passes the exact terminal validator. It SHALL compute
oracle-minus-consensus and retire unless the delta is at least 0.10 with a
case-clustered lower confidence bound above zero.

### REQ-CONSTRAINT-VERIFY-6175-ERROR-DIVERSITY: Error Diversity and Shortcut Audit

Exp6175 SHALL report normalized-cluster diversity, dominant failure surfaces,
consensus-wrong/oracle-right counts, duplicate clusters, raw duplicate shares,
and shortcut audits for answer-position, arbitrary-id, sample-index, and
hidden-state channels.

### REQ-CONSTRAINT-VERIFY-6175-CLUSTERED-INFERENCE: Case-Clustered Intervals

Exp6175 SHALL compute intervals over case clusters for per-candidate accuracy,
oracle@8, consensus accuracy, and oracle-minus-consensus. The artifact SHALL
declare the deterministic resampling method or constant-cluster shortcut used
for each interval.

### REQ-CONSTRAINT-VERIFY-6175-HELD-AGGREGATE: Held Labels Stay Aggregated

Exp6175 may inspect held labels internally for aggregate qualification, but the
terminal artifact SHALL emit only held aggregate counts/rates, an aggregate
signature, and a sealed held row-label hash. It SHALL NOT emit held sample keys,
row hashes, label hashes, validator-result objects, or per-row held outcomes.

### REQ-CONSTRAINT-VERIFY-6175-NO-SELECTOR: No Selector or Hidden-State Extraction

Exp6175 SHALL be a prerequisite audit only. It SHALL NOT tune a latent selector,
extract hidden states, expose held row labels downstream, use held rows for
consensus tuning, or allow any downstream rows when readiness is zero.

### REQ-CONSTRAINT-VERIFY-6175-FAIL-CLOSED-RETIREMENT: Strict Readiness Conjunction

Exp6175 SHALL set `phase_d_headroom_ready_score` to bare `1.0` only when every
preregistered authenticity, parseability, competence, unsaturation, headroom,
minority, and family-support conjunct passes. Otherwise it SHALL set `status`
to `retired` or `blocked`, `future_rows_allowed_by_this_artifact` to bare
`false`, and `honest_verdict` to a terminal prefix naming the failed conjuncts.

### SCENARIO-CONSTRAINT-VERIFY-6175-RAW-REVALIDATION: Labels Are Replayed From Raw Text

Given immutable Exp6174 raw trace rows and calibration/held label sidecars,
When Exp6175 runs,
Then every label is recomputed from `raw_completion_text` with the Exp6173 exact
validator, raw-row hashes are checked against sidecars, and any mismatch retires
the domain.

### SCENARIO-CONSTRAINT-VERIFY-6175-NO-HELD-ROWS: Held Labels Stay Sealed

Given Exp6175 may inspect held labels internally for aggregate qualification,
When it writes the artifact,
Then the held section contains aggregate rates, counts, a sealed held row-label
hash, and an aggregate signature only, with no held row labels, sample keys, row
hashes, or per-case held outcomes.

### SCENARIO-CONSTRAINT-VERIFY-6175-RETIRE-PARSE-FAILURE: Unparseable Pools Retire

Given Exp6174 contains a complete K8 raw-before-label pool whose candidates are
all retained but unparseable,
When Exp6175 computes the preregistered gates,
Then parseability, competence, headroom, minority, and family-support conjuncts
fail closed, readiness is zero, future rows are forbidden, and the artifact
reports `retired:`.
