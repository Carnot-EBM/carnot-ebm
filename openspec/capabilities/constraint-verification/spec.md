# Constraint Verification Capability Specification

**Capability:** constraint-verification
**Version:** 0.1.0
**Status:** Draft

## Overview

Defines deterministic audits for executable constraint-verification corpora.
These audits decide whether a frozen candidate pool is usable for later
selector experiments before any hidden-state extraction is attempted.

## Requirements

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
