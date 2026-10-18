# Wire Trace2Skill into the production conductor loop

**Status:** Draft change proposal.
**Origin:** User question 2026-04-26 — "is Trace2Skill working yet?"
  Answer: implemented (1,970 LOC across `trajectory_analyst.py`,
  `consolidator.py`, `skill_directory.py`, `orchestrator.py`), tested
  (3 dedicated test files), and **not used by the production milestone
  loop**. Same pattern as DualGPU (RETRO-DUALGPU) and
  ExclusionManifestEnforcer (RETRO-MANIFEST-FULL-SCOPE, 12 consecutive
  milestones unwired).
**Target milestone:** 2026.04.71 — first milestone after .70 work
  proves out the failure-ledger and roadmap-schema layers.
**Priority:** High. The conductor has run 800+ experiments since
  Trace2Skill landed; every one of those experiments left a trajectory
  on disk with no structured Lesson extracted, no skill directory
  updated, no playbook surfaced to the planner. Each milestone we
  delay this wiring is another ~12 trajectories of unrecovered signal.
**Depends on:**
  - Failure-ledger discipline (shipped) — Lessons subsume the ad-hoc
    `prior_failures:` four-part structure with a richer schema.
  - Roadmap schema validation (proposed) — both should land before we
    add more conductor surface area.
**Related to:** RETRO-MANIFEST-FULL-SCOPE — same root cause
  (deployment theater); same fix shape (one explicit wire-in commit
  to `scripts/research_conductor.py`).

## Summary

The `python/carnot/autoresearch/` package implements the full
Trace2Skill pipeline (arxiv 2603.25158): analysts extract structured
`Lesson` objects from experiment trajectories, a consolidator dedups
and merges, a `SkillDirectory` persists them, and they surface to
the hypothesis generator as prompt context. Three demo scripts
exercise the pipeline:

  - `scripts/demo_autoresearch.py`
  - `scripts/run_autoresearch_llm.py`
  - `scripts/run_code_verification_autoresearch.py`

`scripts/research_conductor.py` does **not** import any of it. The
production loop that runs the .68 / .69 / .70 milestones has zero
Trace2Skill integration.

This proposal wires Trace2Skill into the conductor at three points:

1. **Post-experiment hook**: after each experiment commits, dispatch
   `analyze_error` (for ⚠/❌-verdict experiments) or `analyze_success`
   (for ✅-verdict experiments) on the trajectory bundle. Append the
   produced Lessons to the pending list.
2. **End-of-milestone consolidation**: at archive time, run
   `consolidate_lessons` over the pending list and merge into the
   project's persistent `SkillDirectory` (new artifact at
   `ops/skill_directory.json`).
3. **Pre-planner skill injection**: before launching the next-milestone
   planner agent, render the relevant skill subset via
   `SkillDirectory.to_prompt_context(model_tier="all")` and prepend
   it to the planner prompt. The planner reads `Lessons learned from
   prior experiments:` as required context, so the same retrocause
   diagnoses don't repeat.

## What this proposal IS NOT

- **Not a replacement for milestone retrospectives.** The milestone
  retro is human-readable narrative; the Lesson is structured
  machine-consumable knowledge. They serve different audiences and
  should both exist.
- **Not a replacement for the failure-ledger.** The failure-ledger
  enforces "do not re-propose a failed experiment without addressing
  the root cause" — a binary gate. Trace2Skill enriches WHAT the
  root cause is and surfaces success patterns the ledger doesn't
  track. The ledger references an experiment_id; the Skill references
  a generalizable pattern.
- **Not a substitute for CLAUDE.md rules.** Rules are absolute and
  apply to every experiment. Lessons are probabilistic and
  contextual ("on benchmarks like SVAMP, EstimationVerifier
  outperforms FoVer-style step labeling"). When a Lesson has been
  validated repeatedly across milestones, *that* is when it should
  graduate to a CLAUDE.md rule.
- **Not committed to the existing autoresearch demo orchestrator.**
  The conductor is its own loop. We import and call the same
  Trace2Skill primitives, but the conductor's hook points are
  milestone-level, not autoresearch's hypothesis-level.

## Proposed experiments

### Exp A — Post-experiment Lesson extraction hook

**Deliverable:**
edits to `scripts/research_conductor.py` (add post-commit Lesson
extraction) +
`tests/python/test_research_conductor_trace2skill_hook.py` +
`results/experiment_<N>_trace2skill_hook.json`.

**What it does:**

After the conductor's `In-process docs reconciliation` step (already
existing, see `research_step()` ~line 2600), dispatch a Lesson
extraction:

```python
# Existing: artifact + verdict already in scope
artifact_path = PROJECT_ROOT / task["deliverable"]
artifact = json.loads(artifact_path.read_text())
verdict = artifact.get("honest_verdict", "")
label = map_status_label(verdict)

# New: lesson extraction
from carnot.autoresearch.trajectory_analyst import analyze_error, analyze_success, Lesson
trajectory_bundle = {
    "experiment_id": task["id"],
    "title": task["title"],
    "milestone": task["milestone"],
    "verdict": verdict,
    "status_label": label,
    "artifact_path": str(artifact_path),
    "code_paths": _detect_code_paths_for_task(task),
}
if label == "✅ Complete":
    lesson = analyze_success(trajectory_bundle, llm_runner=_get_lesson_llm())
else:
    lesson = analyze_error(trajectory_bundle, llm_runner=_get_lesson_llm())

if lesson is not None:
    _append_pending_lesson(lesson)
```

**Decentralization rule 1 (local-first):** the lesson-LLM defaults to
the same `cached_sota_pair()` local model the conductor already uses
for in-process docs reconciliation. Closed-weight upstreams remain
optional and behind an explicit flag.

**Acceptance:** every committed experiment in milestone .71 has at
least one Lesson appended to the pending list (or a logged reason
why extraction was skipped). `ops/pending_lessons.jsonl` accumulates.

### Exp B — End-of-milestone consolidation + persistent SkillDirectory

**Deliverable:**
edits to `scripts/research_conductor.py` (add consolidation hook to
`_archive_current_milestone` flow) +
`ops/skill_directory.json` (new persistent artifact, version-controlled) +
`tests/python/test_research_conductor_skill_directory.py` +
`results/experiment_<N>_trace2skill_consolidate.json`.

**What it does:**

When the conductor archives a milestone (transition path), call:

```python
from carnot.autoresearch.consolidator import consolidate_lessons
from carnot.autoresearch.skill_directory import SkillDirectory

pending = _read_pending_lessons()  # ops/pending_lessons.jsonl
consolidated = consolidate_lessons(pending, dedup_threshold=0.85)

skill_dir = SkillDirectory.load_or_create(PROJECT_ROOT / "ops" / "skill_directory.json")
for lesson in consolidated:
    skill_dir.add_lesson(lesson)
skill_dir.save()
_clear_pending_lessons()
```

The consolidator already handles deduplication and confidence-boost
on repeated lessons (see `python/carnot/autoresearch/consolidator.py`).

**Acceptance:** `ops/skill_directory.json` exists, contains
non-empty `lessons` list, and grows monotonically across milestones.
Each lesson references the experiment_id(s) that produced it.

### Exp C — Pre-planner skill injection

**Deliverable:**
edits to `scripts/research_conductor.py:_plan_next_milestone()`
(prepend skill context to planner prompt) +
`tests/python/test_planner_skill_injection.py` +
`results/experiment_<N>_planner_skill_context.json`.

**What it does:**

Before the planner Sonnet call, render the relevant skills:

```python
skill_dir = SkillDirectory.load_or_create(...)
skill_context = skill_dir.to_prompt_context(model_tier="all")
planner_prompt = (
    f"Lessons learned from prior experiments (Trace2Skill):\n"
    f"{skill_context}\n\n"
    f"---\n\n"
    f"{existing_planner_prompt}"
)
```

The planner now sees, e.g., "On SVAMP-style word problems,
EstimationVerifier outperforms FoVer-style step labeling because
SVAMP CoT depth < 2" *as documented Carnot knowledge*, not as
prose buried in a milestone retro.

**Acceptance:** the .72 planner output references at least one
SkillDirectory lesson by name (or `LESSONS_NOT_YET_AVAILABLE` if the
directory is genuinely empty after one milestone of operation).

## Decentralization implications

- **Rule 1 (local-first):** Lesson extraction defaults to the
  conductor's existing `cached_sota_pair()` local model. The
  Trace2Skill pipeline does not require a closed-weight upstream.
- **Rule 6 (per-call data minimization):** if a closed-weight
  upstream is opted into for richer Lesson extraction, the
  trajectory bundle must declare `data_handling_class: "summarize"`
  (not `"pass_through"`) — the artifact's full text is too sensitive
  to ship to a vendor in raw form.
- **Rule 7 (no vendor-specific abstractions in the core):** the
  `_get_lesson_llm()` helper goes in
  `python/carnot/conductor/lesson_runner.py` (conductor-specific
  submodule), not in the core verifier stack.

## Why this is in change-proposals, not just a code change

The Trace2Skill pipeline already has a design doc
(`openspec/capabilities/autoresearch/design.md`). What's missing is
the **wire-in commit** to `scripts/research_conductor.py`. RETRO-
MANIFEST-FULL-SCOPE has been blocked for 12 consecutive milestones
on the same kind of wire-in commit, because the conductor has a
self-protection rule that reverts subagent edits to itself. That
constraint applies equally here.

The proposal is the locus where future-Claude can find the
wire-in plan after the rule is enforced and a human grants
permission for the one explicit conductor edit.

## Risks

- **Lesson noise floods the planner prompt.** If the SkillDirectory
  fills with low-confidence lessons, the planner gets distracted.
  Mitigation: `SkillDirectory.to_prompt_context()` already filters
  by `model_tier` and confidence; the conductor passes a confidence
  threshold (default 0.6).

- **Lesson extraction adds wall-time per experiment.** Each Lesson
  is an LLM call. At ~5 sec per call, 12 experiments per milestone
  is +60 sec/milestone (~1% overhead). Negligible.

- **The lesson-LLM disagrees with reality.** A Lesson is the
  analyst's interpretation, not a fact. Mitigation: the consolidator
  dedupes and boosts confidence on repeated patterns; one-shot
  hallucinations stay at low confidence and are filtered out by the
  threshold above.

- **Skill directory grows unboundedly.** Lessons accumulate forever.
  Mitigation: a periodic prune phase (planned for Exp D, future
  proposal) — lessons not referenced in 5+ milestones are archived
  to `ops/skill_directory_archive.json`.

## Acceptance criteria

1. After milestone .71 completes, `ops/skill_directory.json` exists
   and contains at least 12 lessons (one per experiment, minimum
   bar — most should produce more than one).
2. The .72 planner agent's prompt includes a Lessons block, and
   the .72 milestone's design doc references at least one lesson by
   name.
3. The 13th consecutive milestone of "deployment theater" pattern
   does not occur. Either Trace2Skill is operational from .71 onward,
   or its wire-in is escalated to known-issues with a clear human-
   intervention ask (same path RETRO-MANIFEST-FULL-SCOPE took at .70).
4. No regression in conductor wall-time vs the .69 / .70 baseline
   (Lesson extraction overhead < 2% of milestone wall time).
