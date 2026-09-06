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
the call site: 44 distinct mutations RED against the final code, every one restored
byte-identically, after an adversarial review's per-entry deletion sweep found 8 of the 13
marker entries this task added to be decorative (two removed, the rest held by their own
tests) and found the capstone lint's exemption wrong in both directions (replaced by shape);
one further probe (not a proof) is RED. The CLAUDE.md table replacement is drafted in
Appendix A and not applied. Corrections made during the session are in section 9.

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
| P-assert | `assert <x>["flags"] == []` (or `flags == []`) in `tests/python`, by AST parse (corrected, see 9.1; a first draft said 46 grep hits) | 33 in 31 files; 5 call the verifier in the test body, 9 assert a capstone's stored `adversarial_verify_report` |

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
declared name matched no allowlist and no name rule. Reason, MEASURED: 33 assertions in
`tests/python` require an empty flag list (P-assert, by AST parse; corrected, see 9.1), 9
of them on the `adversarial_verify_report` a capstone module stores in its own artifact,
and a probe that fires the warn on every artifact broke an existing test on a real
artifact (5.1). A warn on every artifact would change what a live capstone gate reports.
A mutation that fires the warn on recognised names (A4b in section 5) is RED against the
new suite. The cost, stated as a gap: a recognised name with no class is not nudged by
the gate. Adoption for that population rides on the planner prompt and CLAUDE.md, both
operator surfaces.

### 2.4 Measurements

| Number | Value | Population | Status |
|---|---|---|---|
| artifacts carrying `inference_substrate_class` | 0 | P-readable | MEASURED |
| artifacts that would draw `SUBSTRATE_CLASS_MISSING` | 973 | P-readable | MEASURED |
| artifacts that would draw `SUBSTRATE_DECLARATION_MALFORMED` | 169 | P-readable | MEASURED |
| full-gate flag-set changes from the shape fix | 169 changed; +169 MALFORMED warn, +3 METHODOLOGY_MISSING warn, -39 SUBSTRATE_HAS_NO_DURATION_FLOOR warn, 0 critical added or removed | P-dict | MEASURED |
| the three METHODOLOGY_MISSING additions | exp3116, exp3170, exp5380: their stringified dicts had matched a recogniser by accident (`replay`, `live_llm_inference` as a KEY), which skipped the methodology check | P-dict | MEASURED |
| empty-flag assertions a universal warn could hit | 33 (5 direct verifier calls, 9 capstone stored reports, 19 other shapes) | P-assert | MEASURED (count, AST; corrected, see 9.1); that each would fail is INFERRED from their shape, plus the one-file probe in 5.1 |
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
- Win markers gain `beats_vote`, `beats_majority_vote`, `moat_survives`. Null markers gain
  `not_beat_vote` and the plural `not_beats_*` spellings (exp3996 writes
  `local_not_beats_vote`); `does_not_beat_vote` (exp5161) is a relevance marker and is
  read as a null through `not_beat_vote`. Three entries were added and then removed after a
  per-entry deletion proved them decorative: the singular `beat_vote` (its only corpus
  spelling is `does_not_beat_vote`), `success_verifier_moat` in the relevance tuple
  (`verifier_moat` always matches inside it), and `does_not_beat_vote` in the null tuple
  (double-covered by `not_beat_vote`). See 3.4 items 4 and 5.
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
5. The adversarial review deleted every marker this task added ONE ENTRY AT A TIME and
   found 8 of 13 decorative (review HIGH 1). The mechanism matters more than the count:
   my B11 mutation deleted the three plural null spellings as a GROUP, and a group
   deletion goes RED if any one member is covered, so it cannot show that the others are
   untested; `beats_majority_vote` was mutated nowhere. Fixed by method: a per-entry sweep
   (one deletion per tuple entry, no groups, failing tests named) now covers all 12 added
   entries, 12 RED. Two entries were removed as redundant by construction
   (`success_verifier_moat` in relevance; `does_not_beat_vote` in null); the six others
   received an isolating test each, including a parametrised case per null spelling.
6. The capstone-lint exemption keyed on the literal `"git"` was wrong in both directions
   (review HIGH 2): it exempted a rotter that mentions the word for an unrelated reason and
   refused a recoverer that reads an archive on disk with no version control at all.
   Replaced by shape: a `return` between the milestone guard and the sibling `raise` is a
   fallback that can still succeed. Both counterexamples are tests. A `Return` requirement
   on the guard `if` was then found decorative by mutation and removed; the MILESTONE
   mention is what protects an ordinary schema guard, and that now has its own fixture in
   both shapes.

### 3.5 Markers outside the claim keys, re-derived (MEASURED; corrected, see 9.6)

Definition, stated. Over P-readable (6024 on the merged tree): an artifact whose claim keys
(`_MOAT_RIGOR_CLAIM_KEYS` plus `headline_outcome` and `headline`) carry NO marker from the
union of the four moat tuples (31 markers, matched by the gate's own `_moat_marker_present`
on `_moat_rigor_norm(value)`), while some other top-level string field carries one.
Count: 26. By key: `p0_1_v2_verdict` 3, `prior_milestone_verdicts_summary` 3, `note_path`
3, `p0_1_v5_verdict` 2, `experiment` 2, `title` 2, and singletons (`key_finding`,
`milestone_319_summary`, `p01_route2_verdict`, `p01_route2_fair_verdict`,
`p0_1_v5_paragraph`, `p0_1_v5_summary`).

A first draft reported 19 under a six-marker set and named exp4221 and exp4272 as claims
in substance. Both were wrong on reading. exp4221's `honest_verdict` is
`complete: oracle_distinct_verifier_ties_vote_with_headroom`, a TIE; only its `experiment`
NAME says `beats_vote`. exp4272's `title` is a planner task prompt and its verdict is
`blocked_gate_check_failed`. Reading the 26 by key: they are quoted verdicts of OTHER
experiments carried by capstones and archive tasks (`p0_1_v2_verdict`,
`prior_milestone_verdicts_summary`), file paths (`note_path`), and names. None is the
artifact's own claim. So the honest statement is the opposite of the first draft's: adding
`experiment` or `title` as claim keys would flag ties and blocked prompts. Recommendation
8.4 is withdrawn. The residual the family still has is the general one in section 6: a
claim that lives only in a key nobody has listed; no such key was found in this population.

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
`_milestone_inputs` has the guard-then-sibling-raise shape; the lint reads its `return
archived, ...` inside the history walk as the fallback that makes it a recoverer, and a
test asserts that reading is exercised on all three helpers (section 5, C4).

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
was run WITHOUT `-x` so every failing test is named, the file was restored from the
pre-mutation bytes, and the SHA-256 was compared. Every run restored byte-identically. The
runner ran UNLOCKED: `test_suite_mutation_check.py --mutation-begin` refuses inside a
worktree while the same lock blocks repo-wide. PYTHONPATH was pinned to this worktree's
`python/`, and the runner printed the imported paths of `carnot` and
`scripts.adversarial_verify` before the first mutation; both resolve inside this worktree.

Method correction after review (9.4): the first pass deleted three null spellings as ONE
mutation, and one added entry was mutated nowhere. A grouped deletion goes RED if any one
member is covered and says nothing about the others. A per-entry sweep (5.0b) replaced it.
The count below is of distinct mutations RED against the FINAL code: 44 (A 12, B 20, C 8,
D 2, E 1, F 1). Survivors found along the way, and what each led to, are in 9.4.

### 5.0 Part A, the class check: 12 of 12 RED

| id | mutation | result |
|---|---|---|
| A1 | `check_substrate_class` call removed from `_verify_artifact_impl` | RED |
| A2 | `check_substrate_declaration_shape` call removed | RED |
| A3 | enum check dropped | RED |
| A4 | absent-class warn gate replaced by `False` | RED |
| A4b | absent-class warn fires on recognised names (`True`) | RED |
| A5 | `blocked_no_run` pairing rule disabled | RED |
| A6 | model-class-with-blocked-verdict rule disabled | RED |
| A7 | live-evidence contradiction disabled | RED |
| A8 | negative-evidence contradiction disabled | RED |
| A9 | class floor comparison disabled (`< 0`) | RED |
| A10 | shape fix reverted to the `"value" in value` form | RED |
| A11 | MALFORMED condition disabled | RED |

### 5.0b Part B, per-entry deletion of every marker this task added: 12 of 12 RED

| tuple | entry deleted | failing test(s) |
|---|---|---|
| `_MOAT_HEADLINE_MARKERS` | `moat_survives` | moat_survives spellings; ledger-named artifacts |
| `_MOAT_HEADLINE_MARKERS` | `beats_vote` | beats_vote contract; ledger-named artifacts |
| `_MOAT_HEADLINE_MARKERS` | `beats_majority_vote` | beats_majority_vote win |
| `_MOAT_RIGOR_CLAIM_KEYS` | `status` | status-only claim reaches the win branch |
| `_MOAT_RIGOR_RELEVANCE_MARKERS` | `does_not_beat_vote` | exp5161 shape |
| `_MOAT_RIGOR_WIN_MARKERS` | `beats_vote` | beats_vote contract; ledger-named artifacts |
| `_MOAT_RIGOR_WIN_MARKERS` | `beats_majority_vote` | beats_majority_vote win |
| `_MOAT_RIGOR_WIN_MARKERS` | `moat_survives` | moat_survives with a positive delta |
| `_MOAT_RIGOR_NULL_MARKERS` | `not_beat_vote` | exp5161 shape; parametrised null case |
| `_MOAT_RIGOR_NULL_MARKERS` | `not_beats_sc` | parametrised null case |
| `_MOAT_RIGOR_NULL_MARKERS` | `not_beats_self_consistency` | parametrised null case |
| `_MOAT_RIGOR_NULL_MARKERS` | `not_beats_vote` | negated claim is null; parametrised null case |

Entries added and then REMOVED because a per-entry deletion left the suite green and no
test could isolate them: `moat_survives`, `beats_vote`, `beats_majority_vote` copies in the
relevance tuple (the headline tuple is consulted first); `beat_vote` (singular; no positive
corpus spelling); `success_verifier_moat` in relevance (`verifier_moat` always matches
inside it); `does_not_beat_vote` in the null tuple (`not_beat_vote` matches inside it).

### 5.0c Part B, rule mutations: 8 of 8 RED

| id | mutation | result |
|---|---|---|
| B4 | delta rule back to leaf-only | RED |
| B5 | `MET` exact token only | RED |
| B6 | null precedence removed | RED |
| B7 | right boundary removed from markers | RED |
| B8 | left boundary removed before `tuned_` | RED |
| B9 | `success_moat` marker restored | RED |
| B10 | `status` removed from `_claims_moat` keys | RED |
| B12 | both boundaries removed from the SC token | RED |

### 5.0d The capstone lint: 8 of 8 RED

| id | mutation | failing test(s) |
|---|---|---|
| C1 | sibling-raise walk removed | sibling-raise; incidental-git rotter; name-only helper |
| C2 | replay exemption call site removed | replay fixture; live repository; three real capstones |
| C3 | unreadable module back to fail-open | unreadable module |
| C4 | fallback-return recognition removed | both git spellings; disk archive; live repository; three real capstones |
| C5 | replay-helper recogniser removed | replay fixture; live repository; three real capstones |
| C6 | sibling guard ignores the MILESTONE mention | schema guard; disk archive; live repository; three real capstones |
| C7 | any `"git"` literal exempts again | incidental-git rotter |
| C8 | inline rule ignores the MILESTONE mention | schema guard; live repository |

### 5.0e exp6847, the pin, the census: 4 of 4 RED

| id | mutation | result |
|---|---|---|
| D1 | exp6847 history fallback disabled | RED |
| D2 | exp6847 design read from the live file | RED |
| E1 | one name appended to `NO_LLM_SUBSTRATE_ALIASES` | RED |
| F1 | census dict population taken from gate text again | RED |

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
between the two runs is empty in both directions. So 130 pre-exist: 89 are
`KeyError: 'carnot_adversarial_verify_37xx'` raised by `_module_with_current_source`
(untouched by this task) when seven test files (six archive-activation tests, exp3690 to
exp3743, and the exp3722 convergence-synthesis test) load the gate under a custom module
name without registering it in `sys.modules` (corrected, see 9.2; a first draft said 88
and "seven archive-experiment tests"); 10 are exp6780's own roadmap rot
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
  (2.3). Larger, and omitted from the first draft (9.7): 3103 of 6024 readable artifacts
  declare NO `inference_substrate` at all and draw no class flag either; the marker scan
  governs them, and SCENARIO-SUBSTRATE-CLASS-2 codifies that silence on purpose. That is
  the population where a class would matter most, and it is not nudged. An artifact
  declaring `hardware_board` draws no floor flag at any duration, because the gate has no
  hardware floor and I did not invent one.
- Moat family: a claim that lives only in `experiment` or `title` (exp4221, exp4272; 3.5).
- Capstone lint: a raise nested inside a LATER sibling block (`if`/`try`) after the guard
  `if`, not a bare sibling `Raise`, is not caught. A rotter that returns from an
  unrelated branch between its guard and its raise reads as a recoverer (the shape rule's
  residual, 9.5).
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
4. WITHDRAWN 2026-09-05 after review: the measurement behind "add `experiment` and
   `title` to the moat claim keys" did not reproduce, and its two exemplars are a tie and a
   blocked prompt (3.5, 9.6). Adding those keys would flag ties. No decision to make.
5. Decide the exp5008 corrigendum: the stamp is historical; the re-verified copy is clean.
6. Decide whether the planner prompt names `inference_substrate_class`.
7. The ledger rows I moved from ACCEPTED to FIXED, each with a dated note: `_claims_moat`,
   `_flips_gate`, `_moat_rigor_claim_text`, `_moat_rigor_positive_delta_items`,
   `_moat_rigor_claims_relevant`, `_moat_rigor_claims_win`, `_moat_rigor_uses_naive_sc`,
   `capstone_milestone_rot_lint.py`; and the OPEN `_inference_substrate_text` row, moved
   to FIXED. Reopen any of them if the reading is wrong.

## 9. Corrections made the same session, before the branch was reviewed

Recorded rather than silently patched, per the Error Lifecycle. Both came from the same
cause the coordinator named: a count carried from a surface scan while the parsed count
disagreed.

1. "46 empty-flag assertions" was a grep count. The pattern also matched `n_warn`
   shapes and validators' own `flags` lists. By AST parse the figure is 33 assertions of
   the form `assert <x>["flags"] == []` in 31 files; 5 sit in a test function that calls
   the verifier itself, 9 assert the `adversarial_verify_report` a capstone stores in its
   artifact, 19 are other shapes. The conclusion (a universal warn breaks existing tests
   and changes what a capstone gate reports) stands on the measured probe in 5.1 and on
   the 9, not on the 46. Corrected in 2.3, 2.4, the docstring of
   `check_substrate_class`, and the SCENARIO-2 test comment.
2. "88 KeyError failures in seven archive-experiment tests" was a hand tally of per-file
   counts that dropped the one failure in `test_experiment_3722_convergence_synthesis_
   operator_next_thesis.py`, which is not an archive test. Direct line counts over the
   saved pytest output: 131 `FAILED` lines; 89 `E KeyError: 'carnot_adversarial_verify_'`
   lines; 89 `FAILED` lines across those seven files; 10 in exp6780. Corrected in 5.1b
   and `ops/status.md`.
3. The script written to re-derive item 2 by splitting the pytest output into failure
   sections found 49 sections for 131 failures, so it was itself wrong (its section
   header regex missed most headers). Its numbers were discarded; the direct line counts
   above were used. Recorded because an instrument built to check a count was wrong in
   the same way the count was.
4. Mutation METHOD (review HIGH 1). My table looked complete while carrying a grouped
   deletion (B11: three null spellings in one mutation) and an entry mutated nowhere
   (`beats_majority_vote`). A per-entry sweep found 8 of the 13 added entries decorative:
   `beats_majority_vote` (two sites), `does_not_beat_vote` (two sites),
   `success_verifier_moat`, `not_beat_vote`, `not_beats_sc`,
   `not_beats_self_consistency`. Two were redundant by construction and were removed; the
   rest got isolating tests; the sweep now reads 12 of 12 RED (5.0b). The earlier "36 RED"
   was a true count of mutations and a false picture of coverage.
5. The capstone-lint exemption (review HIGH 2): keyed on the literal `"git"`, it both
   exempted a rotter that mentions the word and refused a recoverer that reads an archive
   on disk. Replaced by shape (a return between the guard and the raise); the `Return`
   requirement on the guard was then found decorative and removed; the MILESTONE mention
   gained its own fixture (5.0d C6, C8).
6. Section 3.5 (review MEDIUM 3): "19 artifacts" did not reproduce under any of six
   marker readings, and both named exemplars were not claims (a tie; a blocked prompt).
   Re-derived under a stated 31-marker set: 26, all summaries, paths and names.
   Recommendation 8.4 withdrawn.
7. The gap statement (review MEDIUM 4) named the smaller silent population and not the
   larger: 3103 artifacts with no declaration at all. Added to section 6.
8. The wrong "46" reached the docstring of `check_substrate_class` in `fb56cb8656` and the
   SCENARIO-2 test comment, not only this note (review MEDIUM 5). Corrected in
   `34d8298277` (note) and `01fee333ab` (docstring and comment). A number in a docstring is
   read by people who never open a note.
9. The commit message of `fb56cb8656` says "none stamped"; the qualifier is "none stamped
   BY THIS CHANGE", since exp5008 carries a historical stamp (review LOW 6). History is not
   rewritten; recorded here.
10. SCENARIO-SUBSTRATE-CLASS-1's "SHALL NOT stamp" clause had no test (review LOW 7);
    `test_the_absent_class_warn_never_stamps_even_through_the_backfill` runs the backfill
    with `apply=True` on a warn-only artifact and asserts the bytes are unchanged.

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

## 11. Repair 2026-09-06: `hardware_board` leaves the enum (REQ-SUBSTRATE-VENUE-1)

Append-only. Section 2.1 above is left as it shipped.

**A correction to my own earlier reasoning, first.** A working note of mine said declaring
`hardware_board` caused a REGRESSION by removing a floor the marker scan would otherwise
apply. I checked before acting and that is FALSE. `SUBSTRATE_CLASS_FIELD` is read in exactly
one place, `check_substrate_class`; neither `_classify_inference_substrate` nor
`duration_floor_for_artifact` reads it, so the marker scan runs regardless. The repair went
ahead for a different and better reason.

**The real defect.** The enum is closed and a non-member is CRITICAL, so its members must be
mutually exclusive answers to ONE question. Six answer "what compute ran", which is what a
duration floor can follow. `hardware_board` answers "where did it run". An artifact running a
full generation on a KV260 had to choose between `model_full_generation` (true, floor-bearing,
silent about the board) and `hardware_board` (true, and its `None` floor makes the class check
contribute nothing). Both facts hold at once; the schema admitted one.

**The change.** Six-value class enum, unchanged floors. `hardware_board` retired into
`RETIRED_SUBSTRATE_CLASSES` so the flag detail can name where the fact moved rather than
report an anonymous bad value. New optional `execution_venue` from a closed set — `host`,
`kv260`, `gatemate`, `polarfire`, all from CLAUDE.md's Hardware-Task Continuity table, none
invented. The venue reads no duration and never will.

**Timing.** 0 artifacts carried `inference_substrate_class` and 0 carried `execution_venue`.
Changing a shipped closed enum at zero adopters costs nothing; a week later it would not.

**A bug the existing suite caught in my change.** The first draft tested
`raw_class in RETIRED_SUBSTRATE_CLASSES` BEFORE the isinstance check, so a dict-shaped class
raised `TypeError: unhashable type: 'dict'` — on exactly the malformed input the check exists
to survive. Three pre-existing tests went red and named it. Fixed by ordering the guard;
mutation M4 removes the guard and goes red, so the fix is under test rather than merely made.

**Mutations, one per pattern, each restored byte-identically and re-verified GREEN.**

| # | mutation | result |
|---|---|---|
| M1 | retired-class branch removed | 1 failed |
| M2 | venue membership check neutered | 3 failed |
| M3 | `check_execution_venue` UNWIRED from `_verify_artifact_impl` | 1 failed |
| M4 | isinstance guard removed (the unhashable regression) | 1 failed |
| M5 | `hardware_board` put back into the floor table | 2 failed |
| M6 | detail string drops its redirection to `execution_venue` | 1 failed |

M3 is the one that matters most: a check nothing calls is this project's most-named bug class,
so the wiring has its own mutation and its own test through the real entrypoint on a real file.

**What this does NOT do, stated so nobody reads a venue field as a floor.** It does not create
the per-board duration floor. CLAUDE.md's Pre-Launch table gives per-board PRECONDITIONS, not
durations, and the census measured 201 of 207 `hardware_smoke` artifacts unfloored. That gap
is untouched and is still an operator decision.
