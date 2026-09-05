# Substrate class enum, moat-rigor vocabulary, and two live defects: what shipped

Date: 2026-09-05. Branch: `worktree-agent-a9699853b6e22301b`. Author: worktree agent, on
three operator-approved pieces of work that all touch `scripts/adversarial_verify.py` or
`scripts/capstone_milestone_rot_lint.py`. Status: shipped on the branch; operator decisions
listed in section 8. No file under `results/` was written. No GPU run.

Basis: `docs/research-notes/substrate-vocabulary-census-and-recommendation-2026-09-05.md`
(Part A) and the 2026-09-05 triage rows in `ops/audit-findings-ledger.md` (Parts B and C).

## 0. The answer in six sentences

`inference_substrate` stays prose, and a required closed `inference_substrate_class` with
seven values now ships inside the fabrication gate, where the conductor's completion gate
and the 24-hour backfill run it; a pre-commit hook would be inert on the population that
widens the allowlists. Absent class is a WARN, on the 973 artifacts whose name told the
gate nothing; a present class is held to a closed enum, the blocked predicate, typed
invocation evidence, and the class floor, and zero corpus artifacts carry the field, so no
critical fires on the historical corpus. The moat-rigor family now reads `status`, treats
`moat_survives` and `beats_vote` as claims, matches the SC token on the full path, accepts
`MET_*`, gives negation precedence, and no longer quarantines its own receipt; on the
corpus that changes 14 artifacts, all legacy, none in the backfill window. The capstone
rot lint now catches the sibling-raise shape, exempts recovery helpers by mechanism, and
fails closed on an unreadable module; exp6847 recovers its roadmap and design from git,
which fixes four tests that errored today. Every rule is proven by a mutation that bites
the call site: 36 distinct mutations RED, every one restored byte-identically, after three
survivors across two passes exposed decorative entries that were removed; one further
probe (not a proof) is RED. The CLAUDE.md table replacement is drafted in Appendix A and
not applied.

## 1. Populations and method

Every number is labelled MEASURED or INFERRED. A MEASURED number names its population.

| Population | Definition | Size |
|---|---|---|
| P-glob | `results/experiment_*.json` in this worktree at HEAD `a77fe5d668` | 6057 |
| P-readable | P-glob minus 35 skipped: over 20 MB, unparsable, or a non-dict top level | 6022 |
| P-declared | P-readable with a non-empty `inference_substrate` after the shape fix | 2919 |
| P-dict | P-readable whose `inference_substrate` is a dict with no `value` key | 169 |
| P-class | P-readable carrying `inference_substrate_class` | 0 |
| P-tests | `tests/python/*.py` referencing the gate or a touched lint | see section 5 |
| P-assert | grep hits in `tests/python` asserting an empty flag list through the full verifier | 46 |

Method. Corpus measurements ran the gate's own functions on each parsed artifact
(`_normalize_principle_wrapped_fields(_flatten_metrics(d))`, then the check). The
before/after deltas ran the same script against the committed gate and the edited gate
and diffed per-artifact flag sets (kind:severity). Scripts live in the session
scratchpad, not the repo; each is under 80 lines and reads `results/` only.

## 2. Part A: the substrate class

### 2.1 What shipped

- `SUBSTRATE_CLASSES` (7) and `SUBSTRATE_CLASS_FLOORS`, keyed to the constants the gate
  already applies: `aggregation` 0.0001 s, `no_model_load` 0.0001 s,
  `model_load_no_generation` 2 s, `model_bounded_generation` 10 s,
  `model_full_generation` 60 s, `hardware_board` None, `blocked_no_run` None.
- `check_substrate_class`, wired in `_verify_artifact_impl` after `check_duration_vs_claim`.
- `check_substrate_declaration_shape`: `SUBSTRATE_DECLARATION_MALFORMED` (warn) for the
  dict-without-`value` shape.
- `_inference_substrate_text` reads that shape as missing. It no longer returns
  `"{'executes_models': ...}"` and judges it like a name.
- `tests/python/test_substrate_tuples_pinned_20260905.py`: the four alias tuples, the ARC
  lint's floor table, and the class enum are pinned by length.
- Spec: REQ-SUBSTRATE-CLASS-1 and REQ-SUBSTRATE-FREEZE-1 in
  `openspec/capabilities/research-harnesses/spec.md`.

### 2.2 Where the check lives, and why

`_verify_artifact_impl` is reached per task by `research_conductor.py:_log_experiment_completion`
(through `importlib.reload` of the gate) and at plan time by
`adversarial_verify.py --backfill --apply --since-hours 24`. Both conductor commit paths
use verification-disabled commits (`research_conductor.py:2059` and `:6551`). From the
alias lint's own commit `a76b5f03f8`, 30 commits widened the allowlists and 29 were
`[conductor]` commits. A hook never sees them. (MEASURED in the census note, section 4;
not re-measured here.)

### 2.3 Severity, and the one decision I made that the note did not

The operator set: WARN first; no CRITICAL that fires on the existing corpus; forward-only.
The note's ramp says "missing class: WARN". I gated that WARN to the population where the
declared name matched no allowlist and no name rule. Reason, MEASURED: 46 assertions in
`tests/python` require an empty flag list through the full verifier (P-assert), and the
capstone modules they test store that report in their own artifact. A warn on every
artifact would turn every one of those red and would change what a live capstone gate
reports. A mutation that fires the warn on recognised names (A4b in section 5) is RED
against the new suite; the probe against an existing empty-flag test is recorded in
section 5 as well. The cost, stated as a gap: a recognised name with no class is not
nudged by the gate. Adoption for that population rides on the planner prompt and
CLAUDE.md, both operator surfaces.

### 2.4 Measurements

| Number | Value | Population | Status |
|---|---|---|---|
| artifacts carrying `inference_substrate_class` | 0 | P-readable | MEASURED |
| artifacts that would draw `SUBSTRATE_CLASS_MISSING` | 973 | P-readable | MEASURED |
| artifacts that would draw `SUBSTRATE_DECLARATION_MALFORMED` | 169 | P-readable | MEASURED |
| full-gate flag-set changes from the shape fix | 169 changed; +169 MALFORMED warn, +3 METHODOLOGY_MISSING warn, -39 SUBSTRATE_HAS_NO_DURATION_FLOOR warn, 0 critical added or removed | P-dict | MEASURED |
| the three METHODOLOGY_MISSING additions | exp3116, exp3170, exp5380: their stringified dicts had matched a recogniser by accident (`replay`, `live_llm_inference` as a KEY), which skipped the methodology check | P-dict | MEASURED |
| empty-flag assertions a universal warn would hit | 46 | P-assert | MEASURED (count); that each would fail is INFERRED from their shape, plus the one-file probe in section 5 |
| `NO_LLM_SUBSTRATE_ALIASES` | 76 elements, 75 distinct; duplicate `cached_sota_event_energy_calibration` (in the starred `DETERMINISTIC_VERIFIER_SUBSTRATES` and again as a bare literal) | the tuple at HEAD | MEASURED |
| the other tuples | AGGREGATION 14, LIVE_MODEL 40, DETERMINISTIC 24, ARC `SUBSTRATE_DURATION_FLOORS` 13 | at HEAD | MEASURED |

The 973 is the census's "unknown" population minus the dict-shaped artifacts, which the
gate now reads as missing (the census reported 984 unknown over P-string on a 6056-file
snapshot; the difference is the snapshot and the 35 skipped files).

### 2.5 What the pinned test does to the loop

The pinned-length test runs before every conductor step. A widening lands first and then
stalls the next step on a red suite. The census note listed this as "loud, not silent" and
did not recommend it; the operator chose it. The failure message tells the agent to declare
`inference_substrate_class` instead of adding a name. The duplicate is named by the test
and left in place: removing it changes the 76 pin, which is the operator's call.

## 3. Part B: the moat-rigor vocabulary

### 3.1 What changed

- `_MOAT_RIGOR_CLAIM_KEYS` gains `status`. `_claims_moat` reads `status` too.
- `_MOAT_HEADLINE_MARKERS` gains `moat_survives`, `beats_vote`, `beats_majority_vote`.
- Win markers gain `beats_vote`, `beats_majority_vote`, `moat_survives`. Null markers
  gain `does_not_beat_vote`, `not_beat_vote`, and the plural `not_beats_*` spellings
  (exp3996 writes `local_not_beats_vote`). The singular `beat_vote` was added and then
  removed: its only corpus spelling in any claim key is the negated `does_not_beat_vote`
  (exp5161; MEASURED over P-readable), and no test could isolate it (3.4, item 4).
- `success_moat` is gone from the relevance and win tuples; `success_verifier_moat` stays.
- Markers match on a RIGHT token boundary only (`_moat_marker_present`).
- `_moat_rigor_claims_win` returns False when a null marker is present.
- `_moat_rigor_positive_delta_items` matches the SC-equivalent token (`sc`,
  `self_consistency`, `vote`; both boundaries) anywhere in the normalised path.
- `_flips_gate` accepts `MET` as the leading token with a boundary (`_MET_LEADING_RE`).
- `_moat_rigor_uses_naive_sc` uses `_TUNED_SC_RE` (boundary before `tuned_`) and
  `_NAIVE_SC_RE` (`naive`, `untuned`, `vanilla`).
- Spec: REQ-VERIFY-7040 in `openspec/capabilities/verification/spec.md`.

### 3.2 The seven ledger-named artifacts, before and after (MEASURED, helper chain)

| artifact | before | after |
|---|---|---|
| exp3916 `moat_scissor_MOAT_SURVIVES` | relevant False, no flag | relevant, critical (no `verifier_is_oracle`) |
| exp3827 claim in `status` only | relevant False, no flag | relevant, critical |
| exp3923 `moatMOAT_SURVIVES` concatenated | relevant False, no flag | relevant, critical (see 3.4) |
| exp4245 `beats_vote`, delta 0.44 | relevant False, no flag | relevant, WIN; 2 criticals (headroom, paired significance) |
| exp3645 `verifier_over_sc_lift.delta` | relevant, win False; 1 critical + 1 warn | win True; 3 criticals + 1 warn |
| exp4346 `MET_oracle_distinct_...` status | flips only via the verdict regex | structured branch matches; still critical |
| exp5008 the lint's own receipt | relevant via `success_moat`; critical | not relevant; no flag |

### 3.3 Corpus-wide delta (MEASURED, P-readable, moat family only)

14 artifacts change flag set. Added: 12 `MOAT_CLAIM_RIGOR:critical`, 4
`MOAT_CLAIM_RIGOR:warn`, 10 `CIRCULAR_MOAT_OVERCLAIM:warn`. Removed: 1
`MOAT_CLAIM_RIGOR:critical` (exp5008). The 14: exp3827, exp3916, exp3923, exp3996,
exp4007, exp4018, exp4233, exp4239, exp4245, exp4252, exp4254, exp4319, exp5008, exp5161.
Every one carries a real `beats_vote`, `not_beats_vote` or `moat_survives` claim; I read
each claim string. All date from 2026-06 and 2026-07, so none is inside the 24-hour
backfill window, and none is stamped by this change. This is the legacy class the
operator accepted. The diff counts flag SETS, so exp3645 and exp4346, which gain more
flags of a kind they already had, are not among the 14.

### 3.4 Three things my own instrument got wrong, caught by running it

1. The first marker regex had a LEFT boundary too. It rejected exp3923, whose verdict
   spells the claim `moatMOAT_SURVIVES`. The ledger names that artifact. The right
   boundary is what the known false positives need (`score_delta`, `beats_scissor`,
   `moat_provenance`); the left one cost a real catch. Dropped.
2. Two entries survived mutation: `status` in the claim keys and `moat_survives` in the
   relevance tuple. Both were double-covered by `_claims_moat`. A pattern whose deletion
   leaves the suite green is decorative. I removed the decorative relevance entries
   (`moat_survives`, `beats_vote`, `beats_majority_vote`; they live in the headline tuple
   that `_claims_moat` consults first) and added two tests that isolate the surviving
   mechanisms. Six re-run mutations are RED.
3. The plural `not_beats_vote` (exp3996) was not a null marker in the first draft. Found
   by reading the 14 changed claims, not by the tests.
4. The second mutation pass ran without `-x` and named every failing test, because the
   first pass could not tell a mutation's RED from a test that was already failing. That
   found one more survivor (`beat_vote`, singular, in the relevance tuple) and one wrong
   expectation in my own status-only test (`delta_vs_sc` is in the naive-SC leaf set, so
   the naive warn was correct and the test was not). Both fixed; the survivor removed.

### 3.5 A real input the widened family still does NOT catch (MEASURED)

Over P-readable, 19 artifacts carry a moat or beats marker in a top-level string field
that is NOT a claim key, and in no claim key. By key: `prior_milestone_verdicts_summary`
3, `note_path` 3, `experiment` 2, `title` 2, `methodology_note` 2, and seven singletons.
Two of those are claims in substance:
`experiment_4221_oracle_distinct_arc_verifier_beats_vote.json` states its claim in the
`experiment` name, and `experiment_4272_arc_cross_family_transfer_fresh_tgi_pool.json`
in `title`. Neither is read by the family. Adding `experiment` and `title` as claim keys
would also match names that merely mention a marker (`note_path` is a file path). Left
for the operator; see section 8.

## 4. Part C: the two live defects

### 4.1 exp6847 `load_planned_tasks`: a CODE FIX

Decision: code fix. Evidence, MEASURED today: the live roadmap is at `2026.09.615`; the
module froze `2026.09.598`; `tests/python/test_experiment_6847_v598_independent_capstone.py`
had 4 errors at fixture setup (`ValueError: expected V598 roadmap milestone`), and
`scripts/capstone_milestone_rot_lint.py` on its default glob exited 1 naming
`load_planned_tasks() at line 267`. The module landed on conductor commit `89ed3aef60`
(2026-09-01), which skips hooks, so the lint never saw it.

The fix mirrors the V576 and V580 helpers: `_milestone_inputs` uses the live files while
the roadmap still holds V598, and otherwise walks the last 400 commits touching the
roadmap and returns the first that holds it, reading the design document from the SAME
commit (the loader validates the two against each other). Today that commit is
`c6d6645e52`, 18 back in the roadmap's history, and its design document reads
`**Milestone:** 2026.09.598`. The suite for the module is 22 passed. The new
`_milestone_inputs` has the sibling-raise shape, so the widened lint would refuse it
without the mechanism exemption; the exemption is exercised on it (section 5, C2, C4).

### 4.2 exp5008: a CODE FIX for the false positive; the artifact is an ACCEPTED STATE

Decision: the false positive is fixed in code (`success_moat` dropped as a marker; test
`test_the_lints_own_shipping_receipt_is_not_a_moat_claim`). The artifact is left exactly
as it is. It carries `flagged_adversarial: true` and ONE `corrigendum_pending` entry (the
`verifier_is_oracle=None` critical); the other five `MOAT_CLAIM_RIGOR` strings in the file
sit inside `fixtures_passed`, the lint's own embedded fixture payloads (MEASURED by
walking the JSON, not by grep). `results/**` is evidence; `determination_preservation_lint`
refuses a commit that drops a stamp; and the re-verified copy now draws no
`MOAT_CLAIM_RIGOR` flag, so a future operator corrigendum has what it needs. Clearing the
stamp is that operator's call, not mine.

## 5. Mutations

Every mutation was applied by text replacement to the live file, the named test target
was run, the file was restored from the pre-mutation bytes, and the SHA-256 was compared.
Every run restored byte-identically. The runner ran UNLOCKED:
`test_suite_mutation_check.py --mutation-begin` refuses inside a worktree while the same
lock blocks repo-wide. PYTHONPATH was pinned to this worktree's `python/`, and the runner
printed the imported paths of both `carnot` and `scripts.adversarial_verify` before the
first mutation; both resolve inside this worktree. The first pass used `-x`; the second
pass (B1, B2, B2b, B2c, B2d, B3, and the probe) named every failing test, so each RED
below is attributed to the mutation, not to a test that was failing already. Count: 36
distinct mutations RED (A 12, B 14, C 6, D 2, E 1, F 1); 3 survivors across the two passes
(B2 relevance copy, B2d, and B1 on the first pass) resolved by removing two decorative
entries and adding one isolating test; 1 probe RED.

| id | mutation (call site or rule) | target | result |
|---|---|---|---|
| A1 | `check_substrate_class` call removed from `_verify_artifact_impl` | class tests | RED |
| A2 | `check_substrate_declaration_shape` call removed | class tests | RED |
| A3 | enum check dropped (`not in SUBSTRATE_CLASSES`) | class tests | RED |
| A4 | absent-class warn gate replaced by `False` | class tests | RED |
| A4b | absent-class warn fires on recognised names (`True`) | class tests | RED |
| A5 | `blocked_no_run` pairing rule disabled | class tests | RED |
| A6 | model-class-with-blocked-verdict rule disabled | class tests | RED |
| A7 | live-evidence contradiction disabled | class tests | RED |
| A8 | negative-evidence contradiction disabled | class tests | RED |
| A9 | class floor comparison disabled (`< 0`) | class tests | RED |
| A10 | shape fix reverted to the `"value" in value` form | class tests | RED |
| A11 | MALFORMED condition disabled | class tests | RED |
| B1 | `status` removed from the claim keys | vocabulary tests | first pass GREEN (double-covered), RED after the status-only `beats_sc` test |
| B2 | `moat_survives` removed from the relevance tuple | vocabulary tests | first pass GREEN (decorative); entry removed; replaced by B2, B2b, B2c, B2d below |
| B2 | `moat_survives` removed from the headline tuple | vocabulary tests | RED |
| B2b | `beats_vote` removed from the headline tuple | vocabulary tests | RED |
| B2c | `moat_survives` removed from the win tuple | vocabulary tests | RED |
| B2d | `beat_vote` removed from the relevance tuple | vocabulary tests | GREEN (survived); the entry was decorative and is removed from both tuples |
| B3 | `beats_vote` removed from the win tuple | vocabulary tests | RED |
| B4 | delta rule back to leaf-only | vocabulary tests | RED |
| B5 | `MET` exact token only | vocabulary tests | RED |
| B6 | null precedence removed | vocabulary tests | RED |
| B7 | right boundary removed from markers | vocabulary tests | RED |
| B8 | left boundary removed before `tuned_` | vocabulary tests | RED |
| B9 | `success_moat` marker restored | vocabulary tests | RED |
| B10 | `status` removed from `_claims_moat` keys | vocabulary tests | RED |
| B11 | plural null spellings removed | vocabulary tests | RED |
| B12 | both boundaries removed from the SC token | vocabulary tests | RED |
| C1 | sibling-raise walk removed | capstone-rot tests | RED |
| C2 | exemption call site removed | capstone-rot tests | RED (3 tests, including the three real capstones) |
| C3 | unreadable module back to fail-open | capstone-rot tests | RED |
| C4 | `"git"` literal recogniser removed | capstone-rot tests | RED |
| C5 | replay-helper recogniser removed | capstone-rot tests | RED |
| C6 | sibling rule inverted on the Return guard | capstone-rot tests | RED |
| D1 | exp6847 git fallback disabled | recovery test | RED |
| D2 | exp6847 design read from the live file | recovery test | RED |
| E1 | one name appended to `NO_LLM_SUBSTRATE_ALIASES` | pinned test | RED |
| F1 | census dict population taken from gate text again | census tests | RED |

### 5.1 Probe result (MEASURED)

A4b (the missing-class warn fired on every artifact) applied while running an existing
empty-flag test file, `tests/python/test_adversarial_verify_hardening_4635.py`, plus the
exp6847 capstone build test: RED, failing
`test_scenario_arc_wmte_4635_a1_fixture_has_no_adversarial_flags`, which asserts
`report["flags"] == []` on a copy of exp4628 (a recognised substrate, no class). So the
universal warn breaks at least one existing test, on a real artifact. That the other 45
assertions would also break stays INFERRED from their shape.

### 5.1b Regression (MEASURED)

118 test files that reach the gate or a touched lint: 1259 passed, 131 failed, 1 skipped.
The 20 failing files re-run with HEAD's `adversarial_verify.py` swapped in (my bytes
copied aside and written back, SHA-256 compared): 130 failed, and the set difference
between the two runs is empty in both directions. So 130 pre-exist: 88 are
`KeyError: 'carnot_adversarial_verify_37xx'` raised by `_module_with_current_source`
(untouched by this task) when seven archive-experiment tests load the gate under a custom
module name without registering it in `sys.modules`; 10 are exp6780's own roadmap rot
(`selected roadmap is not milestone 2026.08.590`; its file name is outside the capstone
lint's glob); the rest are ARC-env absence (`Game lp85 not found`) and capstone contracts
that drifted from the live repo. The 131st was my own status-only test's wrong
expectation (3.4, item 4), fixed. Not run: the whole suite, about 62,000 tests.

### 5.2 Named by NO mutation

- The detail strings of every new flag (their content is asserted by substring in some
  tests, but no mutation targets a detail string).
- `_statement_lists` coverage of `orelse` and `finalbody` blocks: the fixtures put the
  sibling raise in a function body only.
- The `diffusiongemma_gate` dict-status branch of `_flips_gate`: asserted by a test, no
  mutation written for the dict branch alone.
- `SUBSTRATE_CLASS_FLOORS[hardware_board] is None` and `[blocked_no_run] is None`:
  asserted, not mutated.
- The census's `dict_shaped_gate_view` beyond the one assertion the F1 mutation bites.
- Producer writes: this change contains none. The gate and the lint only read; the census
  is read-only by contract; exp6847's artifact writer is unchanged.

## 6. Audit of my own instrument

The task asked for a real input each new check is supposed to catch and does not.

- Class check: an artifact with a RECOGNISED substrate name and no class draws nothing
  (2.3). An artifact declaring `hardware_board` draws no floor flag at any duration,
  because the gate has no hardware floor and I did not invent one.
- Moat family: a claim that lives only in `experiment` or `title` (exp4221, exp4272; 3.5).
- Capstone lint: a raise nested inside a LATER sibling block (`if`/`try`) after the guard
  `if`, not a bare sibling `Raise`, is not caught. A function that reads the live roadmap
  and mentions the literal `"git"` for an unrelated reason is exempted.
- exp6847 recovery: a roadmap whose V598 commit is more than 400 roadmap-touching commits
  back fails closed rather than recovering. Today it is 18 back.

Self-caught while building (3.4 and section 4.1): the left boundary, the exemption
narrower than its concept (my first draft looked only inside Call nodes and missed the
`git = ["git", ...]` prefix in the very helper it was written for), and the census test
that encoded the old stringify behaviour.

## 7. What I did NOT do, and why

- Did not edit CLAUDE.md (forbidden). Appendix A is the draft.
- Did not edit `scripts/research_conductor.py` (planner prompt paragraph naming
  `inference_substrate_class`). The conductor authors that file hourly; the collision risk
  is the reason this task exists. Listed for the operator.
- Did not add `SUBSTRATE_DECLARATION_UNUSED` from the census note's Appendix B.
  `SUBSTRATE_HAS_NO_DURATION_FLOOR` (2026-08-21) already covers its no-floor half, and
  `SUBSTRATE_CLASS_MISSING` covers the unrecognised-name half.
- Did not remove the duplicate alias, add any name to any tuple, or backfill-stamp any
  artifact.
- Did not touch `ops/known-issues.md`; the overdue-priority lint runs on it and the
  operator decisions below belong in the report and this note.
- Did not run the whole suite (about 62,000 tests); ran every test file that reaches the
  gate or a touched lint (section 5.1 records the count).

## 8. Operator decisions left

1. Set the cutover date and the WARN-to-CRITICAL step for an absent
   `inference_substrate_class`. Not encoded.
2. De-duplicate `cached_sota_event_energy_calibration` in `NO_LLM_SUBSTRATE_ALIASES` and
   re-pin 76 to 75. The test names it.
3. Replace the CLAUDE.md "Inference-Substrate Declaration Discipline" table with
   Appendix A, or edit it.
4. Decide whether `experiment` and `title` join the moat claim keys (3.5).
5. Decide the exp5008 corrigendum: the stamp is historical; the re-verified copy is clean.
6. Decide whether the planner prompt names `inference_substrate_class`.
7. The ledger rows I moved from ACCEPTED to FIXED, each with a dated note: `_claims_moat`,
   `_flips_gate`, `_moat_rigor_claim_text`, `_moat_rigor_positive_delta_items`,
   `_moat_rigor_claims_relevant`, `_moat_rigor_claims_win`, `_moat_rigor_uses_naive_sc`,
   `capstone_milestone_rot_lint.py`; and the OPEN `_inference_substrate_text` row, moved
   to FIXED. Reopen any of them if the reading is wrong.

## Appendix A. Draft replacement for the CLAUDE.md table (operator edit; NOT applied)

> **The rule.** Every experiment artifact MUST declare two top-level fields.
> `inference_substrate` is a one-line description of what ran, in plain words. It is
> prose. `inference_substrate_class` is one value from the closed list below. The linter
> checks the class. It does not check the description.
>
> | class | duration floor | meaning |
> |---|---|---|
> | `aggregation` | 0.0001 s | reads upstream JSON; no compute of its own |
> | `no_model_load` | 0.0001 s | deterministic, CPU, solver, verifier scoring, web, ARC env-stepping; no model loaded |
> | `model_load_no_generation` | 2 s | model loaded; embeddings, logits, hidden states; no decode loop |
> | `model_bounded_generation` | 10 s | model loaded; a handful of short calls, or a bisect |
> | `model_full_generation` | 60 s | full generation, training, or live inference |
> | `hardware_board` | none from the gate today; the Pre-Launch Preconditions table governs | KV260, GateMate, PolarFire |
> | `blocked_no_run` | none; requires a `blocked_*` verdict | preconditions failed |
>
> A class outside this list is CRITICAL. A class that contradicts the verdict or typed
> invocation evidence is CRITICAL. A `duration_s` below the class floor is CRITICAL. A
> missing class is WARN, on artifacts whose description the linter does not recognise,
> until <cutover date>; then CRITICAL. Artifacts written before <cutover date> keep their
> description; the linter never rewrites them. The allowlists in `adversarial_verify.py`
> are frozen by a pinned-length test; do not add names. Declare the class instead.
>
> The check is `scripts/adversarial_verify.py:check_substrate_class`. It runs inside the
> fabrication gate, per task and at backfill. It is not a pre-commit hook.

## Cross-references

- `docs/research-notes/substrate-vocabulary-census-and-recommendation-2026-09-05.md`
- `ops/audit-findings-ledger.md` (the 2026-09-04 rows, triaged 2026-09-05)
- `openspec/capabilities/research-harnesses/spec.md`: REQ-SUBSTRATE-CLASS-1,
  REQ-SUBSTRATE-FREEZE-1, REQ-HARNESS-5945
- `openspec/capabilities/verification/spec.md`: REQ-VERIFY-7040 (and REQ-VERIFY-5008)
- `openspec/capabilities/research-reporting/spec.md`: SCENARIO-RESEARCH-6847-ROADMAP-RECOVERY
- CLAUDE.md "Inference-Substrate Declaration Discipline", "QA-Layer Authenticity
  Discipline", "Test-Run Record Integrity Discipline"
