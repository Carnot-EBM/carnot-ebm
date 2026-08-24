# QA-layer MISSED INPUTS in adversarial_verify.py — findings, 2026-08-23

The milestone-close QA-layer audit (`ops/qa_layer_authenticity_audit_report.md`,
regenerated 2026-08-23 01:27Z) returned six `SILENT_NON_FIRING` verdicts and one
`REAL_BUG`, all against `scripts/adversarial_verify.py`. This note records what
each finding turned out to be, what changed, and what was measured.

Code: commit `08e6d68816`. Tests:
`tests/python/test_adversarial_verify_qa_layer_missed_inputs_2026_08_23.py`.

## Headline

Five findings are real and are fixed. Two are not real, and the framing that
carried this work into the session — "four of the seven are principle-wrapper
blindness recurring, the same defect as the 2026-07-03 origin bug" — does not
survive contact with the code. The wrapper fix from that incident is in place
and working. Reporting that plainly is the point of this note.

Corpus effect of the whole change, all 5775 `results/experiment_*.json` swept
before and after: **2 artifacts change, 0 flags added anywhere, 0 artifacts
change quarantine status.**

## Per-finding disposition

| # | Unit | Audit verdict | Disposition |
|---|---|---|---|
| 1 | `_declares_terminal_artifact_readiness` | SILENT_NON_FIRING | REFUTED — input is not a real shape |
| 2 | `check_terminal_artifact_readiness` | SILENT_NON_FIRING | CONFIRMED — fixed |
| 3 | `_is_finite_number` | REAL_BUG | CONFIRMED — fixed |
| 4 | `_numeric_pairs` | SILENT_NON_FIRING | REFUTED — already handled upstream |
| 5 | `_is_count_field` | SILENT_NON_FIRING | CONFIRMED — fixed |
| 6 | `_is_timestamp_field` | SILENT_NON_FIRING | CONFIRMED — fixed |
| 7 | `_is_chance_floor_score` | SILENT_NON_FIRING | CONFIRMED — fixed |

Plus one defect the audit did not name: the returned report read
`honest_verdict` from the pre-normalization payload.

## The two refutations, with evidence

### `_numeric_pairs` — the "dominant pattern" is not present on the live path

The audit is correct that `_numeric_pairs({"auroc": {"principle": ..., "value":
0.997}, ...})` returns an empty list. It is wrong that this reaches production.
No caller calls it on a raw payload. `verify_artifact` runs
`_normalize_principle_wrapped_fields(d)` immediately after `_flatten_metrics`
and before the first check, so every check sees bare values. That is the
2026-07-02 exp5161 fix, and it is exactly the "one shared unwrap helper applied
at every field read" architecture this task asked for — already built.

Measured directly: a wrapped tautology pair, the same pair nested inside
`metrics`, and a bare pair all produce the identical `TAUTOLOGY` flag. 49
artifacts in the corpus carry two or more principle-wrapped numeric top-level
fields, and 192 carry at least one wrapped field of any kind; all of them
depend on this normalizer.

Why the auditor got it wrong: it reviews one function chunk with no caller
context, and its own findings say so repeatedly ("caller behavior is
unavailable", "which was not supplied"). That is a structural limit of
function-chunked auditing, not a mistake to correct in the prompt. The
practical consequence is that a `SILENT_NON_FIRING` on a pure helper must be
checked against the live path before it is believed.

The refutation is pinned by a test anyway
(`TestNumericPairsWrappedFieldsLivePath`). A refutation that nothing guards is
one refactor away from becoming true — deleting the normalizer call turns that
test RED.

### `_declares_terminal_artifact_readiness` — the input is not a real shape

The audit's missed input wraps the WHOLE payload:
`{"principle": "...", "value": {"status": "ready"}}`. The convention wraps
individual FIELDS, not artifacts. A wrapped `status` field, which 58 artifacts
do carry, still triggers the check, because the trigger is key presence.

Unwrapping here would also contradict a spec'd requirement.
SCENARIO-INFRA-6262-4 states that a principle-wrapped value is deliberately NOT
gate-eligible: the readiness contract rejects the wrapper rather than reading
through it. Doing what the audit recommends would break that on purpose.

The audit's separate observation that `"status" in payload` never inspects the
status VALUE is true, and is not a silent non-firing: the value is classified
downstream by `classify_artifact_payload`. This helper only decides whether to
run the check at all.

## The five fixes

**`_is_finite_number` (REAL_BUG).** Accepted only `int`/`float`, so
`numpy.float32(0.913)` as an `auroc` answered False and that metric skipped
every numeric check silently. A numpy scalar cannot arrive through
`json.load`, but experiment scripts import this module and check their own
in-memory dicts before writing them — that is the reachable path. Now
`numbers.Real`, with the bool carve-out kept (numpy's `bool_` is not a
`numbers.Real`, so it is rejected too). Separately, `float(10**400)` raised
`OverflowError` out of a boolean predicate; one oversized integer in one
artifact would have taken the whole sweep down. Now returns False.

**`_is_timestamp_field`.** A measured span was classified as a wall-clock
instant and removed from tautology detection. The audit named a real corpus
field: `checkpoint_mtime_delta_ns` in
`results/experiment_5039_self_play_verifier_checkpoint.json`. Interval words
(`delta`, `elapsed`, `duration`, `interval`, `span`, `runtime`) now decide
before the instant markers. `runtime` closes the audit's counterexample, where
`time` inside `runtime` exempted a duration.

**`_is_count_field`.** Did not recognize `folds` — the fold count of a
cross-validation, which is precisely what its own docstring describes. Ten
sibling count nouns were missing with it. They are matched as whole words
through a new shared `_name_tokens` helper, so `folds` matches and
`discounted` does not yield `count`.

**`_is_chance_floor_score`.** Did not recognize `permuted`. A permutation
control is the same object as a shuffled control; half the project's vocabulary
for the concept was absent from a list claiming to hold all of it.

**`check_terminal_artifact_readiness`.** `"capstone" in path_name` matched
inside `noncapstone`, so an artifact whose name says it is not a capstone
inherited the capstone exemption and lost its critical flag. Same class as
`"diffusiongemma_met"` matching inside `meta_tensor`, fixed 2026-07-03. Now a
whole-word match. No basename in the corpus currently has `capstone` as a
substring of a longer word, so this is forward-looking.

**Report `honest_verdict` (not in the audit).** `verify_artifact` returned
`d_raw.get("honest_verdict")`, falling back to `""` when the value was a dict.
155 artifacts wrap `honest_verdict`, so every one of them reported an empty
verdict to any caller reading the report — the exact field the origin incident
was about. Now routed through the existing `_verdict_text` helper, which
already unwraps.

## Measurement

Full sweep of 5775 artifacts through `verify_artifact`, before and after.

| | before | after |
|---|---|---|
| artifacts with ≥1 flag | 2526 | 2526 |
| `TAUTOLOGY:critical` total | 378 | 375 |
| artifacts changing quarantine status | — | 0 |
| flags added | — | 0 |

The two changed artifacts, both inspected by hand, both false positives:

* `experiment_4716_held_out_first_win_readiness.json` — loses one critical
  `TAUTOLOGY` on `held_out_variant_attempts == min_held_out_variant_attempts
  == 100`, a declared budget and the value that met it. Note the old rule
  already exempted one side, but by accident: `min_held` contains the `n_`
  substring. It stays quarantined by a `LEVER_EXERCISE_EVIDENCE_DEGENERATE`
  critical.
* `experiment_5211_gap4_sota_local_candidate_expansion_v477.json` — drops two
  of three duplicate `TAUTOLOGY` flags on `accepted_rows`, `candidate_pool_n`
  and `repair_attempts`, all counts over the same 120-row pool. It stays
  quarantined by `DURATION_TOO_SHORT` and the third `TAUTOLOGY`.

Per-variant simulation, run separately against the same baseline, so the
operator can see which widening costs what:

| variant | artifacts changed |
|---|---|
| interval words are not timestamps | 0 |
| + word-boundary time token | 0 |
| count nouns added | 2 |
| + word-boundary `count` | 3 (adds 1 `IMPLAUSIBLE_PERFECT:info`) |
| `permut` marker added | 0 |
| + `un`/`non` negation guard | 0 |

## Mutation proofs

Every fix was deleted, its named test run, then restored and re-run. The file
was restored from a byte-identical copy each time and the sha256 checked.

| fix deleted | mutated | restored |
|---|---|---|
| numeric-type widening | 3 failed, 3 passed | 6 passed |
| count words | 7 failed, 2 passed | 9 passed |
| interval guard | 2 failed, 1 passed | 3 passed |
| `permut` marker | 1 failed, 3 passed | 4 passed |
| capstone word boundary | 1 failed, 1 passed | 2 passed |
| verdict unwrap | 1 failed, 1 passed | 2 passed |

The whole test file was also run BEFORE any fix landed: 15 failed, 17 passed.
The 17 are the guard-rails (existing behavior that must survive the widening)
and the two refutation pins.

## Deferred, with the reason

Two of the audit's over-exemption COUNTEREXAMPLES are not fixed here:
`unshuffled_x` matching the `shuffled` marker, and `count` matching inside
`discounted_return`. Both REMOVE exemptions rather than add them, which is the
direction that produces new critical flags across history. Both measure zero
corpus change today, so there is no evidence either way, and an `un`/`non`
negation guard can itself over-correct on a name like
`unshuffled_control_auroc`, which genuinely is a chance-floor field. That is a
separate decision for the operator, not a side effect of closing the missed
inputs.

## What this says about the audit

Two of seven flagged findings were wrong in a way a reader could not detect
from the report alone, because the report never shows a caller. The audit's
integrity guard filters hallucinated evidence; it does not filter a correct
observation about a function that is moot in context. The working rule this
suggests: for a helper with no field extraction of its own, confirm the missed
input against the live entry point before spending fix effort. Both refuted
findings looked identical to the confirmed ones in the report.

## Cross-references

* `ops/qa_layer_authenticity_audit_report.md` — the 2026-08-23 audit
* CLAUDE.md "QA-Layer Authenticity Discipline" — the `## MISSED INPUT` contract
* CLAUDE.md "Principle-Annotated Artifact Fields" — the wrapper convention
* `scripts/adversarial_verify.py:_normalize_principle_wrapped_fields` — the
  2026-07-02 fix these findings were reported as a recurrence of
* `openspec/capabilities/research-harnesses/spec.md` REQ-INFRA-6262 — why the
  readiness boundary rejects wrapped values rather than unwrapping them
* `results/experiment_5039_self_play_verifier_checkpoint.json` — the real
  `checkpoint_mtime_delta_ns` the interval fix is named for
