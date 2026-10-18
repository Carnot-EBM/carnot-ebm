# Verdict Naming Convention Standardization

**Author:** operator (Ian Blenke + Claude session 2026-05-01)
**Status:** Draft, ready for .87 retro / .88 planner pickup
**Origin:** 2026-05-01 exp1122 verdict-classification soft gap

## Problem statement

The conductor's `_verdict_is_untrustworthy()` matcher in
`scripts/research_conductor.py` uses substring token-matching
against the experiment artifact's `honest_verdict` string field.
This works empirically because the project has converged on a small
set of canonical tokens (`partial`, `inverted`, `below`, `failed`,
etc.) that the verdict-token taxonomy in
`scripts/in_process_doc_reconcile.py` recognizes.

But the matcher has accumulated **structural gaps** as new experiments
introduce domain-specific verdict strings that the taxonomy did not
anticipate. Each gap manifests as either a **false positive** (verdict
classified as untrustworthy when it's actually a legitimate finding)
or a **false negative** (verdict classified as trustworthy when it's
actually a failure). Today's session shipped two structural patches
to handle specific gaps:

- **Issue 7 broadening (commit `d80b1ae9`):** `honest_negative` /
  `honest_null` / `honest_neutral` recognized anywhere in verdict,
  not just at suffix.
- **Issue 7 extension (commit `aeae17b5`):** verdicts containing both
  a *progress* token (`improved`/`gained`/`above_baseline`) AND a
  *miss* token (`below`/`under_threshold`/`missed_target`) recognized
  as honest_negative.

These are mitigations, not solutions. The underlying problem is that
**the verdict string is doing two jobs** — (1) describing what
happened in human-readable text, (2) carrying machine-readable
classification — and the substring matcher is heuristic, not
deterministic.

## Empirical evidence (collected 2026-05-01)

Verdicts that **succeeded** in their experiment but had no token-
matcher signal until structural patches:

| Verdict | Should classify as | Token issue |
|---------|--------------------|-------------|
| `auroc_improved_below_995` | honest_negative | `below` triggers PARTIAL erroneously; needed Issue 7 ext |
| `corpus_extended_above_7000` | success | no recognized token; passes only by absence of failure tokens |
| `kl_below_threshold_simulation_only` | success (KL < threshold = good for detailed balance) | `below` would trigger PARTIAL; needed operator-rename in .86 |
| `bundle_complete_latex_not_installed` | success | no recognized token; passes by absence |
| `inversion_fixed_ordering_correct` | success | no recognized token; passes by absence |
| `k5_deployed_and_benchmarked` | success | no recognized token; passes by absence |

Verdicts that **failed** their acceptance gate but have no token-
matcher signal:

| Verdict | Should classify as | Token issue |
|---------|--------------------|-------------|
| `v4_kl_above_threshold` | honest_negative or failed | KL above threshold = detailed balance violated, but `above` is not a miss token |
| `corpus_complete_low_tp` | partial | `low_tp` is domain-specific, not in any token list |

The asymmetry is dangerous: success verdicts pass by *absence* of
failure tokens (mostly safe), but failure verdicts using domain-
specific terms can also pass by absence of failure tokens
(unsafe — masks negative results as success).

The .87 retro layer can correctly classify these because the retro
agent reads the full artifact (including boolean fields like
`kl_v4_below_threshold: false`). But the conductor's *interim*
classification (between iteration and retro) drives `prior_failures`
gating, fail-count incrementing, and stable-deliverable detection —
which means the soft gap can cause a real experiment failure to be
counted as "OK, deliverable already exists" and silently advance the
milestone.

## The proposed convention

Replace pure-string verdict matching with a **dual-signal schema**
where the artifact carries:

1. **`acceptance_gate_met`** *(bool, REQUIRED, NEW)* — the
   authoritative boolean. True iff the experiment met its declared
   acceptance criteria. False otherwise. The conductor's
   trustworthy-check uses this field as the primary signal.
2. **`honest_verdict_class`** *(enum, REQUIRED, NEW)* — explicit
   class for retro categorization. Allowed values:
   - `success` — gate met, all expected outcomes
   - `honest_negative` — ran fully, gate not met, real finding
     (the experiment WAS the test of the gate)
   - `partial` — partially met (some criteria met, some not)
   - `blocked` — couldn't run due to environment (no GPU, no model
     cached, network failure, etc.)
   - `failed` — ran into a code error / exception (NOT an honest
     research negative — an actual bug)
   - `inconclusive` — ran but result is ambiguous (insufficient
     sample size, instrumentation noise, etc.)
3. **`honest_verdict`** *(str, REQUIRED, EXISTING)* — free-text
   describing the finding for humans. Continues to exist for
   readability and for retro narrative; no longer load-bearing
   for classification.

The conductor's matcher becomes:

```python
def _verdict_is_untrustworthy(payload: dict) -> tuple[bool, str | None]:
    # Primary: explicit boolean
    gate_met = payload.get("acceptance_gate_met")
    if isinstance(gate_met, bool):
        if gate_met:
            return False, payload.get("honest_verdict")
        # gate not met — check class to distinguish honest negative
        # (trustworthy: the experiment ran and learned something)
        # from blocked/failed (untrustworthy: the experiment didn't
        # actually test the hypothesis)
        cls = payload.get("honest_verdict_class")
        if cls == "honest_negative":
            return False, payload.get("honest_verdict")
        if cls in ("blocked", "failed", "partial", "inconclusive"):
            return True, payload.get("honest_verdict")
        # class missing — fall back to legacy token matching
    # No bool present — fall back to legacy token matching for
    # backwards compatibility with existing .85/.86/.87 artifacts.
    return _legacy_token_match(payload)
```

Legacy token matching stays in place behind the bool check. New
artifacts are unambiguous; old artifacts continue to work.

## Migration plan

### Phase A: Schema additions (.88 first task)

1. Update `scripts/experiment_template.py` `build_result()` to
   accept `acceptance_gate_met: bool` and `honest_verdict_class:
   str` as required kwargs (default = `None`, but emit a deprecation
   warning when missing so existing scripts surface as needing
   update).
2. Update `python/carnot/utils/schema_validation.py` (or wherever
   the result-schema validator lives) to flag artifacts missing the
   new fields — first as warnings, then as errors after a 2-
   milestone grace period.
3. Add `acceptance_gate_met` and `honest_verdict_class` to the
   conductor's `REQUIRED_RESULT_FIELDS` constant so new experiments
   can't ship without them.

### Phase B: Conductor matcher dual-mode (.88 second task)

1. Update `_verdict_is_untrustworthy()` to check the bool/class
   first, fall back to legacy token matching. Behavior preserved
   for old artifacts; new artifacts get unambiguous treatment.
2. Update `scripts/in_process_doc_reconcile.py` to map
   `honest_verdict_class` directly to the retro symbols (✅ /
   ⚠️ Research Finding / ⚠️ Blocked / ❌ Failed) without
   token-parsing.
3. Update `scripts/failure_ledger_v2.py` to count failures using
   `acceptance_gate_met == False AND honest_verdict_class in
   ("blocked", "failed")` rather than token matching. Honest-
   negative outcomes (`acceptance_gate_met == False AND
   honest_verdict_class == "honest_negative"`) do NOT count
   against the fail cap, since they ARE the legitimate research
   finding.

### Phase C: Backfill (.88 third task, optional)

1. Iterate over `results/experiment_*.json` produced before .88
   and write `acceptance_gate_met` + `honest_verdict_class` based
   on the legacy token-matcher output. This makes the
   retrospective reading of project history use the same schema
   going forward.
2. Update `research-complete.yaml` to include the new fields in
   archived task records.

### Phase D: Documentation (in CLAUDE.md, .88 same milestone)

1. Add a "Verdict naming convention" section to CLAUDE.md
   documenting the three required fields and the allowed values
   for `honest_verdict_class`.
2. Update the experiment-template snippet in CLAUDE.md to show
   the new fields.
3. Document the convention in `_bmad/standards.md` (if it exists)
   or create one.

## Risks

### Risk 1: planner Sonnet doesn't honor the new schema

The planner generates experiment scripts that call
`tmpl.build_result(...)`. If the planner's prompt template doesn't
specify the new fields, the planner will omit them, and Phase A's
deprecation warnings become noise.

**Mitigation:** update the planner prompt in
`scripts/research_conductor.py:_plan_next_milestone()` to require
the new fields, with examples for each `honest_verdict_class` value.

### Risk 2: false sense of confidence in `acceptance_gate_met`

If experiment scripts compute `acceptance_gate_met` incorrectly
(e.g., copy-paste error sets it `True` when it should be `False`),
the conductor will silently accept failed experiments as success.

**Mitigation:** the schema validator can require an explicit
*derivation* of `acceptance_gate_met` from a numeric measurement
field (e.g., `acceptance_gate_met = (kl_value < kl_threshold)`).
Cross-check the bool against the relevant numeric field at
schema-validation time. Falsifies any hand-set bool that disagrees
with the data.

### Risk 3: agents continue to write free-text in `honest_verdict`
that contradicts the bool

The free-text continues to be human-readable. If an agent writes
"v4 sparse+inertia validates detailed balance" but
`acceptance_gate_met == False`, the artifact is internally
inconsistent and downstream readers (humans, retros) will be
misled.

**Mitigation:** make the schema validator reject artifacts whose
`honest_verdict` text contains certain "claim" tokens (`validates`,
`achieves`, `meets`) when `acceptance_gate_met == False`. Soft
heuristic, but catches the most common contradiction patterns.

## Decision required

Two viable timings:

### Option A: Ship in .87 (4 tasks remaining)
Insert as exp1126b (between .87 retro and .87 sealing). Pros: fixes
the gap immediately, gets backfill done while milestone is fresh.
Cons: .87 was already planned as 11 tasks; inserting late risks
distracting from retro.

### Option B: Ship in .88 (planner pickup)
Reserve as the first infrastructure slot for .88. Pros: respects
the .87 plan; gets full milestone budget for proper migration with
backfill and documentation. Cons: leaves the soft gap open for one
more milestone.

**Recommendation:** Option B. The .87 retro will note the gap as a
discovered soft issue (exp1122 motivated it), and .88's planner
will reserve the infrastructure slot as the first task. The gap is
not actively breaking the system — it's manifesting as needing
operator intervention or per-issue patches, which the outer-loop
discipline already handles. Spending a clean milestone on a clean
migration is better than rushing it into .87's tail.

## Rationale anchor

This proposal is consistent with three project disciplines already
in CLAUDE.md:

1. **Phase-validation discipline** — empirical instrumentation IS
   adversarial check at scale. Right now the verdict-token matcher
   is the only "instrument" for trustworthiness; it has known
   blind spots. A typed boolean field is more robust.
2. **No-doomed-rerun discipline** — relies on `prior_failures`
   YAML to skip retried-too-many-times experiments. With
   `acceptance_gate_met`, the failure ledger's count becomes
   unambiguous and the rerun-discipline check becomes more reliable.
3. **Honest discipline at the planner layer** — current verdicts
   are sometimes ambiguous (success or failure?) because the
   string is descriptive but classification depends on context.
   The bool removes the ambiguity at the planner's discretion (the
   experiment author decides what "gate met" means for their
   specific experiment), which is the right place for the
   discretion to live.
