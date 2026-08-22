# Experiment claim-refutation audit

One question per artifact: what would REFUTE the headline claim, and was that
checked? Fabrication is out of scope (adversarial_verify covers it); this audit
targets claims that are true by construction, circular, in-sample, baseline-weak,
or contradicted by their own rows.

This audit never edits an artifact and never blocks anything. It surfaces; the
operator decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity
guard rest on evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CLAIM_OVERSTATED | 1 |

## experiment_6478_identifiable_held_exact_energy_selection.json

**CLAIM_OVERSTATED**

## VERDICT
CLAIM_OVERSTATED

## THE HEADLINE CLAIM
Exact-energy selection improves held exact success over first and shuffled controls, verdict `complete_positive`.

## WHAT WOULD REFUTE IT
Two things. First, energy selector could lose or tie the cheapest serious baseline (unweighted `violation_count`) — that would show weighted energy adds nothing. Second, the selector energy is `sum(weight_i for each violated constraint)` and oracle success is zero violated constraints — same quantity. If a zero-violation candidate exists in each set, min-energy pick wins by construction, so beating `first_candidate` (deterministic wrong-perturbation pick) and `shuffled_energy` could not come out otherwise.

## WAS THAT CHECKED
Yes, and both refutations landed. `violation_count` ties `exact_energy` exactly (paired_gain 0.0, all 24 tie, CI includes zero). first/shuffled are sanity-check arms only. Win over them is true-by-construction, not added value. Verdict token still `complete_positive`, consumed downstream as a positive win.

## EVIDENCE
- `"complete_positive: exact-energy selection improves held exact success over first and shuffled controls; exact backend remains the oracle"`
- `"exact_energy_formula": "sum(weight_i for each violated Exp6477 source constraint)"`
- exact_energy `"exact_success_rate": 1.0`; violation_count `"exact_success_rate": 1.0`
- vs violation_count: `"paired_gain": 0.0`, `"tie_count": 24`, `"win_count": 0`, `"interval_excludes_zero": false`
- vs first/shuffled: `"right_arm": "first_candidate"` ... `"paired_gain": 1.0`; `"right_arm": "shuffled_energy"` ... `"paired_gain": 1.0`

## RECOMMENDATION
NARROW_CLAIM
