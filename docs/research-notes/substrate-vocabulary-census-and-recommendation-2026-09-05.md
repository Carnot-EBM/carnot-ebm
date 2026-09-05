# The `inference_substrate` vocabulary: census and recommendation

Date: 2026-09-05. Branch: `worktree-agent-ac986fe334205e8c2`. Author: worktree agent, on
an operator question. Status: recommendation for an operator decision. Nothing in the
fabrication gate was changed. No file under `results/` was written.

Reproduce every corpus number below with:

```
.venv/bin/python scripts/substrate_vocabulary_census.py --json
```

## 0. The answer in six sentences

The field is free text and has been for months. The corpus holds 1036 distinct strings
against the six values CLAUDE.md calls legal. The gate does not check the six either. It
checks three allowlists (129 names), one name-suffix rule, and a compute-marker scan, and it
sorts every artifact into five duration classes plus "no floor". Recommendation: keep the
string as prose, add a REQUIRED closed `inference_substrate_class` with those five classes
plus `hardware_board` and `blocked_no_run`, cross-check the class against typed invocation
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
`generation_performed`, `model_load_attempted`, across 101 distinct key-sets. Producers are
already, on their own, declaring EVIDENCE of what ran instead of a name. The gate turns each
into the string `"{'executes_conductor': ..."`, which matches nothing.

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
| `scripts/adversarial_verify.py` | `AGGREGATION_SUBSTRATE_ALIASES` 14, `NO_LLM_SUBSTRATE_ALIASES` 75 (25 of them via `DETERMINISTIC_VERIFIER_SUBSTRATES`), `LIVE_MODEL_SUBSTRATE_ALIASES` 40; plus `_declares_no_llm_by_name`; plus 17 floor "reasons" | 129 names + 1 rule |
| `python/carnot/agentic/arc_solve_artifact_discipline.py` | `SUBSTRATE_DURATION_FLOORS` (ARC lint) | 13 |

Of the 124 non-canonical names in the gate's tuples, 91 are used by exactly one artifact and
114 by three or fewer. Nine are used by ten or more. Six of the entries are prose with
spaces and capitals (`"CPU exact chronological decision fixture, no LLM"`). The allowlist is
a per-experiment register, not a vocabulary.

## 3. What the checking layer does with it (MEASURED, P-declared unless stated)

### 3.1 The classifier

| `_classify_inference_substrate` source | artifacts | distinct values |
|---|---|---|
| matched an allowlist entry | 1700 (54.7%) | 239 |
| matched only the `_no_llm` name-suffix rule | 255 (8.2%) | 212 |
| unknown | 1153 (37.1%) | 585 |

One third of declarations are unknown to the gate. The name-suffix rule, added after
exp6593, now carries a quarter of the recognised set.

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
| eight further reasons | 11 |
| `live_model` (60 s, chosen by the compute-marker scan or a live alias) | 453 |
| NO FLOOR (declaration matched nothing and no marker found) | 751 (25.6%) |

Grouped by what the floor assumes about model compute:

| effective class | artifacts | distinct declared tokens |
|---|---|---|
| `no_model_load` | 952 | 361 |
| `aggregation` | 765 | 15 |
| `unfloored` | 751 | 358 |
| `model_full_generation` | 453 | 182 |
| `model_bounded_generation` | 13 | 4 |
| `model_load_no_generation` | 5 | 2 |

This is the enum the gate already uses. It has five classes. The six legal names are not it.

### 3.3 Two legal values the gate does not floor

- `hardware_smoke`: 201 of 207 artifacts get NO floor. Six get 60 s by marker. The gate has
  no hardware branch; CLAUDE.md's "per-board" floor exists only in prose. The 2026-08-29
  known-issues entry found the same gap in the ARC lint.
- `verifier_ensemble_against_cached_candidates`: floored at 1 s, but 13 of 402 fall to no
  floor and 3 to 60 s because of a trailing note the matcher did not accept.

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

87 artifacts in P-string have no floor AND a missing or zero `duration_s`. They draw no flag.
The declaration did nothing and the duration check never ran.

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

## 5. Decision

### 5.1 The options

**A. Collapse to the six with a mapping table; make the field an enum.**
Rejected. The six are not the classes the gate applies (section 3.2). Two of the six draw
no floor (3.3). The six have no value for web ingestion (80 artifacts), a blocked run (39),
bounded generation (13), or GPU training. A mapping table for 1036 historical values is
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
board floor, one floor per value). But the checking state is not acceptable: 26 of 27
widenings unreviewed, 37% of declarations unknown, 26% unfloored, 87 with no duration and no
flag. Prose alone leaves the gate self-widening.

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
   | `blocked_no_run` | none; must pair with a `blocked_*` verdict | preconditions failed | about 39 |

   The 1 s `verifier_scoring` and 0.01 s ARC floors fold into `no_model_load`. The floor that
   matters is the 60 s boundary; the sub-second ones only catch a missing `duration_s`, which
   a separate presence check does better.

2. **Keep `inference_substrate` as prose.** Rename it in CLAUDE.md as "substrate
   description". Stop calling it an enum. It stays required, because it is the one line a
   reader has.

3. **Freeze the three alias tuples and `SUBSTRATE_DURATION_FLOORS`.** No new names. The
   class field makes additions unnecessary. The suffix rule and the tuples stay as the
   read-time deriver for artifacts written before cutover (Appendix C).

4. **Cross-check the class.** `_classify_current_task_inference_claim` already compares a
   declaration with typed evidence. Extend it: `no_model_load` or `aggregation` with live
   evidence is CONTRADICTORY (critical, existing kind); `model_*` with a `blocked_*` verdict
   is contradictory; `blocked_no_run` without a `blocked_*` verdict is contradictory; a class
   outside the enum is critical, as `verdict_class` already is.

5. **Severity ramp.** Missing class: WARN until the cutover date, then CRITICAL for artifacts
   whose run date or `gate_version` is after cutover. Never rewrite historical artifacts;
   derive their class at read time.

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
- `tests/python/test_substrate_vocabulary_census.py`: 7 tests over a `tmp_path` corpus.
- Mutations, each biting a call site, each restored byte-identically (`sha256` compared):

  | mutation | result |
  |---|---|
  | M1 classifier call replaced by a constant "recognised" answer | RED (2 tests) -> GREEN |
  | M2 `duration_floor_for_artifact` call deleted | RED (1) -> GREEN |
  | M3 principle-wrapped unwrap deleted | RED (4) -> GREEN |
  | M4 trailing-note strip deleted | RED (1) -> GREEN |
  | M5 fail-closed exit on unreadable directory replaced by exit 0 | RED (1) -> GREEN |

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
  the 26% unfloored share, `hardware_smoke` drawing no floor, the 169 dict-shaped
  declarations, and that the alias lint has governed 1 of 27 widenings.

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

On today's corpus this would warn on 751 + 277 = 1028 historical artifacts if backfilled,
and on 1 of today's 18. It stamps nothing.

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
8. otherwise -> `unclassified` (the 751; reported, never quarantined, never rewritten)

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
