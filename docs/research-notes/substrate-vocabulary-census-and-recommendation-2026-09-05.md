# The `inference_substrate` vocabulary: census and recommendation

Date: 2026-09-05. Branch: `worktree-agent-ac986fe334205e8c2`. Author: worktree agent, on
an operator question. Status: recommendation for an operator decision. Nothing in the
fabrication gate was changed. No file under `results/` was written.

Reproduce every corpus number below with:

```
.venv/bin/python scripts/substrate_vocabulary_census.py --json
```

**Adversarial review, 2026-09-05, after commits `2c1a1bd855` and `31023cc2f3`.** An
independent reviewer re-derived every corpus number with its own script. Two HIGH findings
and six count corrections; all are recorded in section 10, items 5 to 12, in append form.
The lead finding: the census carried the defect it was built to measure. It named the 169
dict-shaped declarations and then dropped them from every aggregate but `shapes`, while the
gate stringifies them and judges them (70 floored at 60 s, 66 unfloored, 18 with neither
floor nor duration). A recogniser narrower than its concept, inside an instrument for
finding recognisers narrower than their concept; the same class the QA-Layer discipline
names, and the same class this repository fixed twice already today (the alias lint's
literal-only diff scan; the tree-wide `conftest.py` guard). Fixed in the census
(`gate_view`, `dict_shaped_gate_view`, test, mutation M7). Tables below carry corrected
values marked "(corrected, see 10.N)"; the first-draft values stay in section 10. The
review's own summary line then committed the population-mixing error it had filed against
this note, and found it when challenged (10.6). Three occurrences in one day, in the
instrument, the note, and the review: the defect is in how counts get written, not in any
one writer.

## 0. The answer in six sentences

The field is free text and has been for months. The corpus holds 1036 distinct strings
against the six values CLAUDE.md calls legal. The gate does not check the six either. It
checks three allowlists (129 names), one name-suffix rule, and a compute-marker scan, and it
sorts every artifact into five duration classes, a deliberate no-floor for blocked runs, and
"no floor at all" for the rest. Recommendation: keep the string as prose, add a REQUIRED
closed `inference_substrate_class` with those five classes plus `hardware_board` and
`blocked_no_run`, cross-check the class against typed invocation
evidence, and enforce it inside `adversarial_verify.verify_artifact`, because that is the
only layer that runs on the artifacts the conductor writes. The vocabulary decision is the
operator's; this note gives the measurement and a draft.

## 1. Populations and method

Every number is labelled MEASURED or INFERRED. A MEASURED number names its population.

| Population | Definition | Size |
|---|---|---|
| P-files | `results/experiment_*.json` in this worktree at HEAD `ad160b71bf` | 6056 |
| P-string | P-files whose `inference_substrate` is a non-empty string, after unwrapping `{"value": ...}` | 2939 |
| P-dict | P-files whose `inference_substrate` is a dict with NO `value` key | 169 |
| P-declared | P-string plus P-dict, which is what the gate's `_inference_substrate_text` returns non-empty for (it stringifies the dict) | 3108 |
| P-code | the three alias tuples in `scripts/adversarial_verify.py` at HEAD | 129 values |
| P-producers | files under `scripts/experiments/` and `python/carnot/` containing a single-line `"inference_substrate": "<literal>"` | 1463 files, 899 distinct literals |
| P-git | commits on this branch that touched `scripts/adversarial_verify.py` since 2026-08-23 | 37 |

Method differences from the operator's measurement, stated so they can be improved on again:

- The census unwraps the principle-annotated form and names the dict-without-`value` shape
  separately. The operator's glob and unwrap are the same; the dict shape was not named.
- Classification and floor come from the gate's own functions
  (`_classify_inference_substrate`, `duration_floor_for_artifact`), not from a re-derivation.
  So the tables say what the checking layer does, not what a reader thinks it should do.
- Dates come from git add dates (`git log --diff-filter=A`), not mtime. A test run that
  rewrites an artifact in place changes mtime and leaves the add date alone.
- The flag counts in section 3 come from `verify_artifact` over all of P-files, which took
  about eight minutes. The committed census does not call it, so it runs in seconds.

## 2. The vocabulary (MEASURED)

### 2.1 Shape of the field, P-files

| shape | count | share |
|---|---|---|
| missing or empty | 2940 | 48.5% |
| plain string | 2773 | 45.8% |
| principle-wrapped `{"value": ...}` | 166 | 2.7% |
| dict with no `value` key | 169 | 2.8% |
| unreadable JSON | 8 | 0.1% |

The 169 dict-shaped declarations matter for section 5. Their keys are things like
`executes_models`, `downloads_models`, `live_model_invoked`, `no_live_llm_inference`,
`generation_performed`, `model_load_attempted`, across 123 distinct key-sets (corrected,
see 10.10; a first draft said 101). Producers are already, on their own, declaring EVIDENCE
of what ran instead of a name. The gate turns each into the string
`"{'executes_conductor': ..."`, which matches nothing, and then judges it like any other
declaration (added, see 10.5): 70 are floored at 60 s by the compute-marker scan, 20 get
`deterministic_verifier`, 13 `aggregation`, 66 no floor (13 of those are blocked runs), and
18 have neither a floor nor a `duration_s` (16 of them not blocked).

### 2.2 Distinct values, P-string

| measure | count |
|---|---|
| distinct raw strings | 1036 |
| distinct leading tokens (trailing ` -- note` stripped) | 877 |
| raw strings used by exactly one artifact | 881 (85%) |
| raw strings containing `no_llm` | 257 (the operator's 255, on this snapshot) |
| artifacts containing `no_llm` | 396 (the operator's 394) |

The operator's 255 is confirmed. The two extra are today's additions and the unwrap.

### 2.3 Against CLAUDE.md's six legal values, P-string

| legal value | artifacts whose leading token is exactly this |
|---|---|
| `aggregation_from_upstream_artifacts` | 759 |
| `verifier_ensemble_against_cached_candidates` | 402 |
| `live_llm_inference` | 230 |
| `hardware_smoke` | 207 |
| `offline_arcade_live_agent_runtime_self_discovery_no_llm` | 66 |
| `live_llm_embedding_extraction` | 5 |
| total leading with a legal value | 1669 (56.8% of P-string) |
| total exactly equal to a legal value, no note | 1476 (50.2%) |

So the six cover just over half of the declarations. The other 1270 declarations use 871
distinct other names.

### 2.4 Three code vocabularies, none of them the six

| where | what | size |
|---|---|---|
| CLAUDE.md table | "six legal values" with floors | 6 |
| `scripts/adversarial_verify.py` | `AGGREGATION_SUBSTRATE_ALIASES` 14, `NO_LLM_SUBSTRATE_ALIASES` 75 (24 of them via `DETERMINISTIC_VERIFIER_SUBSTRATES`; corrected, see 10.7), `LIVE_MODEL_SUBSTRATE_ALIASES` 40; plus `_declares_no_llm_by_name`; plus 16 floor "reasons" seen in the corpus | 129 names + 1 rule |
| `python/carnot/agentic/arc_solve_artifact_discipline.py` | `SUBSTRATE_DURATION_FLOORS` (ARC lint) | 13 |

Of the 124 non-canonical names in the gate's tuples, 91 are used by exactly one artifact and
114 by three or fewer. Five are used by ten or more (`deterministic_verifier_plus_replay`
19, `web_and_bibliographic_search_only` 16, `aggregation_from_upstream_artifacts_no_llm`
11, `cached_fixture_replay_no_llm` 11, `live_llm_inference_local_gguf_sota` 10; corrected,
see 10.8: a first draft said nine, which counted four canonical names). Seven of the entries are
prose with spaces and capitals (`"CPU exact chronological decision fixture, no LLM"`;
corrected, see 10.9). And `NO_LLM_SUBSTRATE_ALIASES` holds 76 elements but 75 distinct
values: `cached_sota_event_energy_calibration` appears inside the starred
`DETERMINISTIC_VERIFIER_SUBSTRATES` and again as a bare literal (found by the review, see
10.17). Nobody who added the second copy read the list, and the alias lint resolves the
tuple to a set, so it cannot see a duplicate either. The allowlist is a per-experiment
register, not a vocabulary.

## 3. What the checking layer does with it (MEASURED, P-declared unless stated)

### 3.1 The classifier

Populations named per column (corrected, see 10.6; the first draft paired P-declared
artifact counts with P-string distinct counts in one row, which no single census run
reproduces).

| `_classify_inference_substrate` source | P-declared artifacts | P-string artifacts | P-declared distinct raw | P-string distinct raw |
|---|---|---|---|---|
| matched an allowlist entry | 1700 (54.7%) | 1700 (57.8%) | 239 | 239 |
| matched only the `_no_llm` name-suffix rule | 255 (8.2%) | 255 (8.7%) | 212 | 212 |
| unknown | 1153 (37.1%) | 984 (33.5%) | 739 | 585 |

The census prints the P-string columns as `classifier_source` and the P-declared columns
under `gate_view`. One third of declarations are unknown to the gate. The name-suffix rule,
added after exp6593, carries 13.0% of the recognised ARTIFACTS (255 of 1955) and 63.3% of
the recognised distinct LEADING TOKENS (212 of 335); a first draft said "a quarter", which
is neither (corrected, see 10.11).

### 3.2 The floor actually applied, P-string

| floor reason | artifacts |
|---|---|
| `aggregation` (1e-4 s) | 765 |
| `verifier_scoring` (1 s) | 383 |
| `deterministic_verifier` (1e-4 s) | 277 |
| `no_llm_declared_by_name` (1e-4 s) | 159 |
| `arc_live_agent_no_llm` (0.01 s) | 68 |
| `no_llm_declared` (1e-4 s) | 38 |
| `web_bibliographic_search_only` (1e-4 s) | 17 |
| `local_sota_gguf_small_n` (10 s) | 12 |
| `llm_embedding_extraction` (2 s) | 5 |
| six further reasons | 11 |
| `live_model` (60 s, chosen by the compute-marker scan or a live alias) | 453 |
| NO FLOOR, run blocked (`blocked_*` verdict; the gate returns no floor on purpose) | 251 (8.5%) |
| NO FLOOR, declaration ignored (matched nothing, no marker, not blocked) | 500 (17.0%) |

Grouped by what the floor assumes about model compute:

| effective class | artifacts | distinct declared tokens |
|---|---|---|
| `no_model_load` | 952 | 361 |
| `aggregation` | 765 | 15 |
| `unfloored` (declaration ignored) | 500 | 238 |
| `model_full_generation` | 453 | 182 |
| `blocked_no_run` (verdict `blocked_*`) | 251 | 129 |
| `model_bounded_generation` | 13 | 4 |
| `model_load_no_generation` | 5 | 2 |

This is the enum the gate already uses: five floor classes, plus a deliberate no-floor for
blocked runs. The six legal names are not it. Note that `blocked_no_run` is recognised from
the VERDICT, never from the substrate name: 129 distinct names sit in that class, and the
name almost never says "blocked".

Over P-declared, the gate's own view (added, see 10.5; `gate_view` in the census):
`no_model_load` 972 (460 distinct), `aggregation` 778 (65), `unfloored` 553 (294),
`model_full_generation` 523 (264), `blocked_no_run` 264 (153), `model_bounded_generation`
13 (4), `model_load_no_generation` 5 (3). The 169 dict-shaped declarations add 70 to full
generation, 53 to unfloored, 20 to no-model-load, 13 to blocked and 13 to aggregation.

### 3.3 Two legal values the gate does not floor

- `hardware_smoke`: 201 of 207 artifacts get NO floor. Six get 60 s by marker. The gate has
  no hardware branch; CLAUDE.md's "per-board" floor exists only in prose. The 2026-08-29
  known-issues entry found the same gap in the ARC lint.
- `verifier_ensemble_against_cached_candidates`: floored at 1 s in 383 of 402. All 13 that
  draw no floor are `blocked_*` runs (checked verdict by verdict), which is the gate working
  as designed. The 3 that draw 60 s carry typed evidence that a model ran, so the cross-check
  overrode the declaration; also the gate working as designed. A first draft of this note
  blamed a trailing note. That was wrong: 14 of the 16 strings are exact and 2 carry a
  `(principle: ...)` suffix, and the suffix is not what decided the floor (corrected, see
  10.13; the first correction said all 16 were exact).

### 3.4 Where `DURATION_TOO_SHORT` fires, P-declared, from `verify_artifact`

| classifier source | n | DURATION_TOO_SHORT | any CRITICAL | stamped `flagged_adversarial` |
|---|---|---|---|---|
| allowlisted | 1700 | 67 | 166 | 115 |
| suffix rule | 255 | 3 | 14 | 6 |
| unknown | 1153 | 171 | 256 | 171 |

Of the 171 unknown-source DURATION_TOO_SHORT flags, the declared token's keyword bucket is:
live-model words 59, dict-shaped 55, aggregation/audit 21, deterministic/CPU 20, hardware 8,
web 4, blocked 2, `no_llm` 2.

INFERRED: the 57 flags whose declared name says aggregation, deterministic, hardware, web,
blocked or no-LLM are candidates for the "honest no-LLM run with a vestigial GGUF string"
false positive that motivated every alias ever added. Not verified artifact by artifact.

### 3.5 Fail-open corner

60 artifacts in P-string are not blocked, have no floor, AND have a missing or zero
`duration_s`. They draw no flag. The declaration did nothing and the duration check never
ran. (A first draft said 87; that count included 27 blocked runs, which are allowed to have
no duration.)

Over P-declared the corner is 76 (corrected, see 10.5). My derivation, from the census's
`gate_view.unfloored_without_duration` and re-checked by a separate script: 60 string-shaped
(above) plus 16 dict-shaped, all not blocked. Two definitions exist and both are internally
consistent; a reader may use either, but never one term from each:

| definition | P-string | P-dict | P-declared |
|---|---|---|---|
| A: no floor, no duration, blocked runs EXCLUDED (this section, since `31023cc2f3`) | 60 | 16 | 76 |
| B: no floor, no duration, blocked runs INCLUDED (the first draft's reading) | 87 | 18 | 105 |

The figure 78 that circulated during review was 60 from A plus 18 from B, a splice neither
definition produces; the reviewer found this itself when challenged (10.6). I use A and 76.
This moves a headline: the fail-open population is a quarter larger than the first draft
said, and it includes artifacts whose producer was trying to declare evidence.

### 3.6 Typed evidence coverage

75 of 6056 artifacts (1.2%) carry any of the nine invocation booleans
(`generation_invoked`, `model_loaded`, ...) that `_classify_current_task_inference_claim`
reads. This bounds option C in section 5.

## 4. Growth and authorship (MEASURED, git)

New distinct leading tokens by the month their first artifact was added:

| month | new tokens |
|---|---|
| 2026-05 | 124 |
| 2026-06 | 154 |
| 2026-07 | 296 |
| 2026-08 | 265 |
| 2026-09 (five days) | 81 |

Per day since 2026-08-20: 2, 21, 20, 29, 8, 21, 12, 11, 1, 12, 11, 7, 15, 19, 27, 13.
Median about 12 new names per day.

Today's cohort (git author date 2026-09-05): 18 artifacts. 18 of 18 arrived on
`[conductor]` commits. 14 are recognised only by the `_no_llm` suffix rule, each under a
name that did not exist yesterday. 1 is allowlisted (`live_llm_inference`). 1 is unknown
(`live_llm_arc_belief_shadow`). 2 have no declaration. By a UTC window the cohort is 24
files; still 24 of 24 conductor. The operator's "12 of 12" was an earlier snapshot of the
same day and is consistent.

The alias lint, since it shipped on 2026-08-23:

| measure | value |
|---|---|
| commits touching the gate | 37 |
| commits that added an alias value | 27 |
| of those, `[conductor]` commits (hooks skipped) | 26 |
| of those, `[outer-loop]` commits (hooks run) | 1 |
| `NO_LLM_SUBSTRATE_ALIASES` | 50 -> 75 (+26 added, 1 removed) |
| entries in `ops/substrate_alias_acks.md` | 0 |

The guard written to govern widening has governed 1 of 27 widenings. It is not broken. It is
in the wrong place.

Boundary, stated (added, see 10.12 and 10.16). The query was `git log --since=2026-08-23`,
run at about 17:10Z. Git fills a bare date's missing time of day with the current time, so
the boundary was 2026-08-23 at about 13:10 local. The lint's own commit is `a76b5f03f8`,
authored 2026-08-22T23:52:52-04:00, which is 2026-08-23T03:52Z; "shipped on 2026-08-23" in
this note is true in UTC and a day late in local time. My window therefore started about
thirteen hours after the lint shipped. Measured again from the lint's commit
(`a76b5f03f8..HEAD`, my script, reproducing the reviewer): 41 commits touched the gate, 30
added an alias, 29 of those were `[conductor]` commits, 1 was `[outer-loop]`, and
`NO_LLM_SUBSTRATE_ALIASES` went 48 -> 75. **The headline from the lint's own commit is: the
guard has governed 1 of 30 widenings.** Same direction as the first draft's 1 of 27, larger.

## 5. Decision

### 5.1 The options

**A. Collapse to the six with a mapping table; make the field an enum.**
Rejected. The six are not the classes the gate applies (section 3.2). Two of the six draw
no floor (3.3). The six have no value for web ingestion (80 artifacts in P-declared whose
leading token matches a web, literature, or ingestion keyword, my bucket; the gate's own
`web_bibliographic_search_only` floor covers only 17 of them), a blocked run (251, by
verdict; corrected, see 10.15: a first draft said 39, which counted names that SAY
"blocked" and is not a population), bounded generation (13), or GPU training. A mapping
table for 1036 historical values is
either a rewrite of `results/` (barred) or a read-time list that grows by about 12 rows a day
until cutover, which is the pattern-narrower-than-concept bug by construction. And forcing a
string like `deterministic_arc_live_attempt_fixture_no_llm` down to one of six loses the
description, which is the part a reader uses.

**B. Keep the string as prose; add a REQUIRED closed class field.**
Recommended. Detail in 5.2. Precedent in the same spec: REQ-CONDUCTOR-VERDICT-1 keeps
`honest_verdict` free and adds a closed `verdict_class`, cross-checked structurally, "never
a fifth list". The suffix rule shows producers already put the class inside the name
(`_no_llm`); this makes that declaration explicit and checkable.

**C. Admit free text; retire the enum in CLAUDE.md; floor on declared evidence.**
Right direction, not sufficient today. Typed evidence exists in 1.2% of artifacts (3.6). The
evidence keys are themselves free-form: 169 dict-shaped declarations use 101 key-sets. With
no declared class, the only fallback is the compute-marker scan, and that scan is the source
of the false positives in 3.4. Option B is C with a declared anchor: the class is the claim,
the evidence is what checks the claim. As evidence coverage grows the class becomes
derivable and the check tightens without another schema change.

**D. Leave the data alone; change the prose to match.**
Half right. The CLAUDE.md table must change regardless; it is false today (six values, per-
board floor, one floor per value). But the checking state is not acceptable: 29 of 30
widenings unreviewed from the lint's own commit (26 of 27 at the first draft's boundary;
see 10.16), 37% of declarations unknown, 17% ignored outright (500 in P-string, 553 in
P-declared), 76 with no duration and no flag. Prose alone leaves the gate self-widening.

### 5.2 The recommendation in detail

1. **Add `inference_substrate_class`**, required on every artifact written after a cutover
   date, from this closed enum. The floors are the ones the gate applies now.

   | class | floor | meaning | P-string today (effective) |
   |---|---|---|---|
   | `aggregation` | 1e-4 s | reads upstream JSON; no compute of its own | 765 |
   | `no_model_load` | 1e-4 s | deterministic, CPU, solver, verifier scoring, web, ARC env-stepping; no model loaded | 952 |
   | `model_load_no_generation` | 2 s | model loaded; embeddings, logits, hidden states; no decode loop | 5 |
   | `model_bounded_generation` | 10 s | model loaded; a handful of short calls, or a bisect | 13 |
   | `model_full_generation` | 60 s | full generation, training, or live inference | 453 |
   | `hardware_board` | per board, from the Pre-Launch table | KV260, GateMate, PolarFire | 207 (unfloored today) |
   | `blocked_no_run` | none; must pair with a `blocked_*` verdict | preconditions failed | 251 (by verdict, not by name) |

   The 1 s `verifier_scoring` and 0.01 s ARC floors fold into `no_model_load`. The floor that
   matters is the 60 s boundary; the sub-second ones only catch a missing `duration_s`, which
   a separate presence check does better.

2. **Keep `inference_substrate` as prose.** Rename it in CLAUDE.md as "substrate
   description". Stop calling it an enum. It stays required, because it is the one line a
   reader has.

3. **Freeze the three alias tuples and `SUBSTRATE_DURATION_FLOORS`.** No new names. The
   class field makes additions unnecessary. The suffix rule and the tuples stay as the
   read-time deriver for artifacts written before cutover (Appendix C). A frozen tuple can
   also be de-duplicated once, in the open: it holds one duplicate today (2.4), which is
   evidence that entries are appended without reading the list.

4. **Cross-check the class.** `_classify_current_task_inference_claim` already compares a
   declaration with typed evidence. Extend it: `no_model_load` or `aggregation` with live
   evidence is CONTRADICTORY (critical, existing kind); `model_*` with a `blocked_*` verdict
   is contradictory; `blocked_no_run` without a `blocked_*` verdict is contradictory; a class
   outside the enum is critical, as `verdict_class` already is.

5. **Severity ramp.** Missing class: WARN until the cutover date, then CRITICAL for artifacts
   whose run date or `gate_version` is after cutover. Never rewrite historical artifacts;
   derive their class at read time.

6. **Shape is part of the contract (added after the review, see 10.5).** The class field
   MUST be a bare string; a dict-shaped class is a value outside the enum and is CRITICAL. And
   the gate MUST stop stringifying a dict-shaped `inference_substrate`: `_inference_substrate_text`
   returning `"{'kind': ...}"` is the field-shape bug the QA-Layer discipline names (its
   origin bug 2 handled only the `{"value": ...}` form). A dict without `value` should be
   reported as `SUBSTRATE_DECLARATION_MALFORMED` (warn) and treated as missing, never as a
   string that happens to match nothing. Today that shape sends 70 artifacts to the 60 s
   floor by marker and lets 16 through with no floor and no duration.

**What the recommendation now is, after Finding 2.** Unchanged in substance: option B. It
is sharpened in two ways. The fail-open population is 76 (P-declared, not blocked; my
derivation in 3.5), not 60, and includes producers who were declaring evidence, which is
the behaviour the class field legitimises. And the gate needs the shape rule above, or the
class field inherits the same silent-stringify path.

**Cost.** About 150 lines in `adversarial_verify.py` plus tests and a mutation proof; one
paragraph in the planner prompt (`_plan_next_milestone` does not mention substrate at all
today; the planner relies on CLAUDE.md); a `require_inference_substrate_class` flag in
`experiment_5247_slot_artifact_normalizer_v480.normalize_artifact` for template users; the
CLAUDE.md edit; a cutover date. No corpus rewrite. Risk during WARN: none to quarantine.
Risk at CRITICAL: a producer that ignores the prompt is quarantined, which is the intended
behaviour and is forward-only.

**What it does not fix.** 1463 producer files hardcode 899 literals. No template change
reaches them. That is why the check must sit in the gate.

## 6. Where the check has to live to fire

| layer | runs on conductor artifacts? | why |
|---|---|---|
| pre-commit hooks (`substrate-alias-evidence-lint`, `arc-artifact-lint`) | NO | both conductor commit paths use `--no-verify` (`git_commit_and_push`, and the checkpoint path). 26 of 27 widenings and 18 of 18 of today's artifacts went through them. |
| `_log_experiment_completion` in `research_conductor.py` | YES, per task | it calls `adversarial_verify.verify_artifact` on the deliverable with `importlib.reload`, and stamps `flagged_adversarial` on a critical flag. This is the completion gate. |
| `adversarial_verify.py --backfill --apply --since-hours 24` | YES, at plan time | the backstop sweep in the milestone-close path. |
| `_run_operational_retrospective` audits | YES, at milestone close | non-fatal; where the other three adversarial audits report. The census belongs here. |
| a repo test asserting a pinned tuple length | YES, but by stalling | the conductor runs tests before each step, so a widening lands first and stalls the next step. Loud, not silent; the operator has said stalls are the worse failure. Listed, not recommended. |

So: the class check goes into `_verify_artifact_impl`; the census goes into the
retrospective; the hooks stay for operator and agent commits, with their docstrings stating
plainly that they never see conductor commits.

## 7. Operator decisions needed

1. Adopt the seven-class enum in 5.2, or amend its names and floors.
2. Set the cutover date and the WARN-to-CRITICAL step.
3. Replace the CLAUDE.md table with Appendix A, or edit it.
4. Name `scripts/adversarial_verify.py` in a declared scope so the check can be built. The
   file is sealed by `harness_integrity_lint.py`.
5. Decide whether to freeze the tuples with a pinned-length test, knowing it stalls the
   conductor on the next widening rather than refusing the commit.
6. Decide whether the WARN in Appendix B ships before the enum, as a visibility step.

## 8. What shipped in this session

- `scripts/substrate_vocabulary_census.py` (REQ-SUBSTRATE-CENSUS-1). Read-only. Exit 2 if
  the directory cannot be read. Uses the gate's own classifier and floor function.
- `tests/python/test_substrate_vocabulary_census.py`: 8 tests over a `tmp_path` corpus
  (7 in the first draft; the eighth covers the dict-shaped gate view, see 10.5).
- Mutations, each biting a call site, each restored byte-identically (`sha256` compared):

  | mutation | result |
  |---|---|
  | M1 classifier call replaced by a constant "recognised" answer | RED (2 tests) -> GREEN |
  | M2 `duration_floor_for_artifact` call deleted | RED (1) -> GREEN |
  | M3 principle-wrapped unwrap deleted | RED (4) -> GREEN |
  | M4 trailing-note strip deleted | RED (1) -> GREEN |
  | M5 fail-closed exit on unreadable directory replaced by exit 0 | RED (1) -> GREEN |
  | M6 blocked-run detection deleted at its call site (added with correction 1) | RED (2) -> GREEN |
  | M7 gate-view read replaced by the string view (added with correction 5) | RED (1) -> GREEN |

  The mutation runner ran unlocked: `--mutation-begin` refuses inside a worktree. PYTHONPATH
  was pinned to this worktree's `python/` and the tests import the script from this
  worktree's `scripts/` by path.

- Named by NO mutation: the census has no producer writes, so there is nothing of that kind
  to name. Paths no mutation exercised: `iter_artifacts` on a top-level JSON array (counted as
  unreadable), `render` beyond the two substrings `test_main` asserts, and the
  `SHAPE_OTHER` branch (unit-tested directly, not through `census()`).

## 9. Corrections to the operator's measurement

- 255 distinct `no_llm` strings and 394 artifacts: confirmed, 257 and 396 on this snapshot.
- "12 of 12 today conductor-authored": confirmed and larger; 18 of 18 by author date, 24 of
  24 by a UTC window.
- Not previously measured: the whole-corpus distinct count (1036), the 37% unknown share,
  the 500 ignored declarations (17%), `hardware_smoke` drawing no floor, the 169 dict-shaped
  declarations, and that the alias lint has governed 1 of 27 widenings.

## 10. Corrections made the same day, after commit `2c1a1bd855`

Recorded rather than silently patched, per the Error Lifecycle.

1. "751 draw no floor" was under-read. 251 of them are `blocked_*` runs, where
   `duration_floor_for_artifact` returns `None` by design (its first branch). The
   ignored-declaration population is 500. The census now reports `precondition_blocked`
   and the class `blocked_no_run`; tables in 3.2, 3.5, 5.1, 5.2, 9 and Appendix C updated.
2. The 16 `verifier_ensemble_against_cached_candidates` misses in 3.3 were blamed on a
   trailing note. The strings are exact. 13 are blocked runs; 3 carry typed live evidence.
3. Appendix B's backfill count was 1028 (751 + 277, mixing populations). Re-derived from the
   classifier-by-floor cross-tab it is 965 of P-declared.
4. "17 floor reasons" and "eight further reasons" were miscounts; 16 and six.

### 10b. Corrections from the adversarial review, same day, after commit `31023cc2f3`

The reviewer re-derived every number with its own script. Items 5 and 6 were rated HIGH.

5. **HIGH. The census dropped dict-shaped declarations from every aggregate but `shapes`.**
   `census_artifact` classified only string values and `census()` filtered on the string,
   while the gate's `_inference_substrate_text` stringifies a dict and judges it. So for 169
   artifacts the census did not report the gate's view, which its own SCENARIO-GATE-VIEW
   requires. Consequences, reviewer's numbers, reproduced: 70 floored at 60 s, 20
   `deterministic_verifier`, 13 `aggregation`, 66 unfloored, 18 with neither floor nor
   duration. The fail-open corner of 3.5 is 60 in P-string and 76 in P-declared, not
   blocked (my derivation; the population is stated in 3.5). Fix: the census now
   classifies through the gate (`gate_view`,
   `dict_shaped_gate_view`), one new test, mutation M7. Recommendation sharpened in 5.2 item
   6; substance unchanged. The class of the defect: a recogniser narrower than its concept,
   inside the instrument built to find them.
6. **HIGH. Section 3.1 mixed populations in one row.** 1153 unknown artifacts is P-declared;
   585 unknown distinct values is P-string; the census printed 984 and 544 (P-string,
   artifacts and leading tokens). Fixed with one column per population. Over P-declared the
   unknown distinct raw count is 739.

   The same error then appeared in the review itself. The reviewer's summary line gave the
   fail-open corner as "60 to 78". Its 18 dict-shaped artifacts were counted under the
   older definition that included blocked runs (which gives 87 for P-string, not 60), and
   its 60 under the newer one that excludes them. 60 + 18 mixed the two definitions. When
   challenged, the reviewer found this itself and restated 76. So the population-mixing
   defect occurred three times in one day: in the instrument (item 5), in the note about
   the instrument (this item), and in the review of the note. That is not three
   coincidences. It is evidence the defect is intrinsic to how these counts get written:
   a number gets carried without its population, and the population changes underneath
   it. The only protection that worked was naming the population next to every number,
   which is now the rule for this note and for the census's output keys.
7. `DETERMINISTIC_VERIFIER_SUBSTRATES` holds 24 entries, not 25.
8. Five non-canonical allowlist names are used ten or more times (19, 16, 11, 11, 10), not
   nine. The nine counted four canonical names.
9. Seven allowlist entries are prose with spaces, not six.
10. 123 distinct dict key-sets, not 101. The first count truncated each key list to 90
    characters before de-duplicating, which merged distinct sets.
11. The suffix rule carries 13.0% of recognised artifacts and 63.3% of recognised distinct
    leading tokens. "A quarter" was neither.
12. The P-git figures depended on an unstated boundary. Stated now in section 4; the
    reviewer's midnight-boundary figures (41/30/29/1, 48 -> 75) are recorded there. Conclusion
    unchanged.

13. Correction 2 above said "the 16 strings are exact". A parse says 14 of 16; two carry a
    `(principle: ...)` suffix. The cause stated in correction 2 (blocked runs; typed live
    evidence) is unchanged, because the suffix did not decide the floor in either case.
14. Precision on the constraint. This session wrote no evidence file under `results/`. It
    did create an access symlink at `results/arc_leaderboard_eval_runs` (pointing at the
    operator's untracked directory, last written 2026-09-05 00:03 local, about 13.5 hours
    before this session) so a pre-commit hook could run; no target file was changed. A
    first version of this item called the symlink "gitignored". In this worktree it is
    NOT: `.gitignore:309` reads `results/arc_leaderboard_eval_runs/` with a trailing
    slash, which matches a directory and not a symlink, so `git status` shows `??`. Every
    commit here staged explicit paths, so it was never swept; it is removed right after the
    last commit, and the removal is verified with `git status --porcelain --ignored`.
15. Section 5.1 said "a blocked run (39)" while 5.2 said 251. The 39 was a keyword count
    of substrate names containing a blocked-like word, which is not a population the gate
    uses; 251 is the gate's own predicate over P-string. A correction that updated 5.2 and
    left 5.1 is how a reader gets the wrong half. Both now say 251. The "80 artifacts" for
    web ingestion in the same sentence had no stated population; it is my keyword bucket
    over P-declared, and the gate's own floor covers 17. Stated now.
16. The P-git headline was boundary-dependent and the boundary was a time of day, not a
    date. From the lint's own commit `a76b5f03f8` the figures are 41 / 30 / 29 / 1 and
    `NO_LLM_SUBSTRATE_ALIASES` 48 -> 75, reproduced by my own script. The headline is now
    "1 of 30 widenings", larger than the first draft's 1 of 27, same direction. Section 4
    and 5.1 updated with both figures.

17. `NO_LLM_SUBSTRATE_ALIASES` has 76 elements and 75 distinct values; the duplicate is
    `cached_sota_event_energy_calibration`. Found by the review, reproduced by my script.
    Added to 2.4 and to recommendation item 3 (freeze, then de-duplicate once).
18. The reviewer named the two `verifier_ensemble_against_cached_candidates` artifacts that
    carry a `(principle: ...)` suffix and draw no floor:
    `experiment_3694_selection_gap_proper_rediagnosis.json` and
    `experiment_3801_abstention_http_rest_surface.json`. Both are blocked runs, so the
    proof in 3.3 stands on the verdict, not on the string.

Items 7 to 18 change numbers or wording, not conclusions. Item 5 changes one headline
number and adds one requirement to the recommendation. Item 16 makes a headline larger.

## Appendix A. Draft replacement for the CLAUDE.md table (operator edit)

> **The rule.** Every experiment artifact MUST declare two top-level fields.
> `inference_substrate` is a one-line description of what ran, in plain words. It is prose.
> `inference_substrate_class` is one value from the closed list below. The linter checks the
> class. It does not check the description.
>
> | class | duration floor |
> |---|---|
> | `aggregation` | 0.0001 s |
> | `no_model_load` | 0.0001 s |
> | `model_load_no_generation` | 2 s |
> | `model_bounded_generation` | 10 s |
> | `model_full_generation` | 60 s |
> | `hardware_board` | per board (Pre-Launch Preconditions table) |
> | `blocked_no_run` | none; requires a `blocked_*` verdict |
>
> A class outside this list is CRITICAL. A class that contradicts typed invocation evidence
> is CRITICAL. A missing class is WARN until <cutover date>, then CRITICAL. Artifacts written
> before <cutover date> keep their string; the linter derives a class for them and never
> rewrites them. The allowlists in `adversarial_verify.py` are frozen; do not add names.

## Appendix B. Draft WARN for a declaration that did nothing (gate edit, needs a scope)

```python
def check_substrate_declaration_used(d: dict[str, Any], flags: list[Flag]) -> None:
    """WARN when a declared substrate selected no floor, so the declaration was ignored."""
    raw = _inference_substrate_text(d)
    if not raw:
        return
    cls = _classify_inference_substrate(d)
    if cls["kind"] != SUBSTRATE_KIND_UNKNOWN:
        return
    floor = duration_floor_for_artifact(d)
    chosen = floor["reason"] if floor else "no_floor"
    if chosen not in ("live_model", "no_floor"):
        return  # a structural recogniser floored it; the name was irrelevant but harmless
    flags.append(Flag(
        kind="SUBSTRATE_DECLARATION_UNUSED",
        severity="warn",
        detail=(
            f"inference_substrate={raw!r} matched no reviewed value and no name rule; "
            f"the duration floor came from {chosen}. Declare inference_substrate_class."
        ),
    ))
```

On today's corpus this would warn on 965 of P-declared if backfilled (829 string-shaped, 136
dict-shaped), and on 1 of today's 18. It stamps nothing.

A second flag, added after the review (10.5), for the shape bug itself:

```python
def check_substrate_declaration_shape(d: dict[str, Any], flags: list[Flag]) -> None:
    """WARN when inference_substrate is a dict with no `value`; it is not a declaration."""
    value = d.get("inference_substrate")
    if isinstance(value, dict) and "value" not in value:
        flags.append(Flag(
            kind="SUBSTRATE_DECLARATION_MALFORMED",
            severity="warn",
            detail=(
                f"inference_substrate is a dict with keys {sorted(value)[:6]} and no "
                "'value'; treated as missing. Declare a string, and put evidence in "
                "typed invocation fields."
            ),
        ))
```

This needs `_inference_substrate_text` to return "" for that shape instead of `str(dict)`,
so the marker scan and the missing-declaration path apply, as they do for an absent field.

## Appendix C. Read-time class derivation for artifacts written before cutover

Order matters; first match wins. This is the census's `EFFECTIVE_CLASS_OF_REASON` turned
into a producer-facing rule.

1. `blocked_*` verdict, or `precondition_check_only` -> `blocked_no_run`
2. leading token in the aggregation tuple, or `_is_aggregation_only` -> `aggregation`
3. leading token `hardware_smoke`, or a board name in the token -> `hardware_board`
4. `_declares_no_llm_by_name`, or the no-LLM tuple, or `_is_verifier_scoring_only`, or
   `_is_deterministic_verifier`, or the ARC/web/QA recognisers -> `no_model_load`
5. `_is_llm_embedding_extraction` -> `model_load_no_generation`
6. `_is_local_sota_gguf_small_n` or the bisect recogniser -> `model_bounded_generation`
7. live-model tuple, or compute marker present -> `model_full_generation`
8. otherwise -> `unclassified` (the 500; reported, never quarantined, never rewritten)

## Cross-references

- CLAUDE.md "Inference-Substrate Declaration Discipline" (the prose this measures)
- CLAUDE.md "QA-Layer Authenticity Discipline" (pattern list narrower than its concept)
- CLAUDE.md "Test-Run Record Integrity Discipline" (why `results/` is read-only here)
- `openspec/capabilities/research-harnesses/spec.md` REQ-SUBSTRATE-ALIAS-1 (the inert
  hook), REQ-CONDUCTOR-VERDICT-1 (the closed-class precedent), REQ-SUBSTRATE-CENSUS-1
- `ops/known-issues.md` 2026-08-29 entry (58% of ARC artifacts illegal; asked for this sweep)
- `ops/substrate_alias_acks.md` (zero entries)
- `scripts/research_conductor.py:git_commit_and_push` and the checkpoint path (both
  `--no-verify`), `_log_experiment_completion` (the completion gate)
