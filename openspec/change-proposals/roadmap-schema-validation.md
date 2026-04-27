# Roadmap schema validation: structural contract for planner output

**Status:** Draft change proposal. **REQUESTED FOR MILESTONE 2026.04.77.**
The pattern that triggered the proposal recurred on 2026-04-27 when
the .74 planner emitted a roadmap with `experiments:` instead of
`tasks:` and other field renames; an operator had to manually
translate the YAML before activation or .74 would have gone stillborn
exactly like .69. Two stillborn-milestone-prevented incidents in
under two weeks is enough evidence that the bare-lookup contract
needs Pydantic enforcement.
**Origin:** User question 2026-04-25 in response to milestone 2026.04.69
  going stillborn. The .69 planning agent (Sonnet, 50-turn) wrote
  `research-roadmap-next.yaml` with 12 tasks but omitted the `title:`
  field on Exp 896 (svamp-estimation-verifier). The conductor activated
  the milestone, then crashed with `KeyError: 'title'` on the next
  iteration. It re-crashed identically every 10 minutes — three full
  iterations spinning on the same uncaught lookup — before a human
  added the missing line by hand.
**Target milestone:** 2026.04.71 — first milestone after .70 work
  proves out the failure-ledger mechanical-enforcement layer.
**Priority:** Medium-high. This is the same class of failure as
  RETRO-MANIFEST-FULL-SCOPE and DualGPU-idle: we have a contract
  in our heads (and in CLAUDE.md prose) but no schema enforcing it
  at runtime. The fix is small. The recurrence cost — milestones
  silently going stillborn — is large.
**Depends on:** nothing. Pydantic is already a project dependency
  (used in `python/carnot/verify/`).

## Summary

The conductor's roadmap pipeline has three internal call sites that
bare-index `task["title"]`:

  - `scripts/research_conductor.py:736`  `load_research_tasks()`
  - `scripts/research_conductor.py:896`  `pick_next_task()`
  - `scripts/research_conductor.py:1464` `_archive_current_milestone()`

There are similar bare lookups for `id`, `deliverable`, and `prompt`
in adjacent code. Any task missing any required field crashes with
`KeyError`, the conductor's outer `try/except` swallows the error,
logs `"Unexpected error in research step"`, and sleeps another 10
minutes — repeating the same crash forever until a human notices.

This proposal:

1. Adds a Pydantic model `ResearchTask` and `Roadmap` that encodes
   the required fields, optional fields, and field types.
2. Calls the validator at three points:
   - **Pre-commit (planner output):** the conductor parses
     `research-roadmap-next.yaml` immediately after the planner
     agent writes it; if validation fails, the conductor refuses to
     commit and either (a) re-prompts the planner with the
     validation error, or (b) writes a
     `blocked_planner_output_invalid` artifact and surfaces it for
     human attention.
   - **Pre-activation:** before overwriting `research-roadmap.yaml`
     with `research-roadmap-next.yaml`, validate. Refuse to activate
     a malformed milestone — keep the current one open.
   - **Pre-pickup:** in `pick_next_task()`, validate each task before
     handing it to Sonnet. (Belt-and-suspenders; the prior two checks
     should make this a no-op, but it's the last guard against
     hand-edits or partial writes.)
3. Replaces all bare `task["title"]` / `task["id"]` / etc. lookups
   with model attribute access (`task.title`), which gives static
   type-checking and a single source of truth for the contract.

## What this proposal IS NOT

- Not a constraint on what the planner *can* propose. Required-field
  validation is structural, not semantic. The planner is still free
  to design any experiment; it just has to fill out the form
  completely.
- Not a replacement for the planner-prompt updates that document the
  YAML structure. Documentation tells the planner *what* to write;
  schema tells the conductor *how to verify* what was written. Both
  layers needed.
- Not gold-plating. The Pydantic model only encodes fields that the
  conductor actually consumes. We are not enumerating every YAML key
  someone might invent — we are catching the failure mode where
  fields the conductor *requires* are absent.

## Proposed experiments

### Exp A — `ResearchTask` Pydantic model + pre-activation validator

**Deliverable:**
`python/carnot/conductor/roadmap_schema.py` (new module) +
edits to `scripts/research_conductor.py` (call validator at activation) +
`tests/python/test_roadmap_schema.py` (unit tests, no LLM) +
`results/experiment_<N>_roadmap_schema_primitive.json`.

**What it does:**

1. `class ResearchTask(BaseModel)`:
   - Required: `id: str`, `milestone: str`, `title: str`,
     `deliverable: str`, `prompt: str`.
   - Optional: `depends_on: list[Dependency] | None`,
     `prior_failures: list[PriorFailure] | None`,
     `gated_on: dict | None`, `max_turns: int | None`.
   - `id` field validator: matches `^exp\d+-[a-z0-9-]+$`.
   - `milestone` field validator: matches `^\d{4}\.\d{2}\.\d{2}$`.
   - `deliverable` field validator: starts with `results/` and ends
     with `.json`.
2. `class Roadmap(BaseModel)`:
   - Required: `milestone: str`, `milestone_title: str`,
     `milestone_doc: str`, `tasks: list[ResearchTask]` (must be
     non-empty).
   - Cross-task validator: every task's `milestone` field must equal
     the roadmap's `milestone` field.
3. `def validate_roadmap_file(path: Path) -> ValidationResult`:
   - Loads YAML, instantiates `Roadmap`, returns either `ok` or
     `failed` with line-anchored error messages
     (`pydantic.ValidationError` already produces these).

**Pre-launch wiring (in `research_conductor.py`):**

```python
# Before: blind copy
shutil.copy(NEXT_ROADMAP, ROADMAP_FILE)

# After: validate first
result = validate_roadmap_file(NEXT_ROADMAP)
if not result.ok:
    logger.error(
        "Refusing to activate malformed roadmap %s:\n%s",
        NEXT_ROADMAP, result.errors,
    )
    # Write a blocked-activation artifact and surface for human
    artifact_path = PROJECT_ROOT / "results" / f"experiment_{exp_n}_blocked_roadmap_invalid.json"
    artifact_path.write_text(json.dumps({
        "experiment": "blocked_roadmap_activation",
        "honest_verdict": "blocked_roadmap_invalid",
        "blocking_errors": result.errors,
    }, indent=2))
    return  # current milestone stays active
shutil.copy(NEXT_ROADMAP, ROADMAP_FILE)
```

**Acceptance:** The exact .69 failure mode (Exp 896 missing `title:`)
is caught at activation time. The conductor writes a clear blocked
artifact, keeps the previous milestone open, and surfaces the
problem on the next iteration. No silent infinite-loop spin.

### Exp B — Planner output pre-commit validation

**Deliverable:**
edits to `scripts/research_conductor.py` (the
`_plan_next_milestone` flow) +
`tests/python/test_planner_output_validation.py` +
`results/experiment_<N>_planner_validation.json`.

**What it does:**

After the planner agent finishes (Sonnet-50 call returns), but
*before* the conductor commits the new YAML, run
`validate_roadmap_file(NEXT_ROADMAP)`. If it fails:

1. Log the validation errors verbatim.
2. Re-spawn the planner with a corrective prompt:
   ```
   Your previous planning output had structural errors:
   {errors}

   The roadmap YAML at {path} must conform to the ResearchTask
   schema (see python/carnot/conductor/roadmap_schema.py). Each
   task requires: id, milestone, title, deliverable, prompt.
   Re-emit the YAML with these errors corrected.
   ```
3. Cap the retry at one iteration — if the corrected output is also
   invalid, fall through to the blocked-artifact path and surface
   for human attention.

**Acceptance:** When the planner forgets a field, the conductor
catches it immediately and either auto-corrects or refuses to
activate. The .69 stillborn-milestone scenario is impossible.

### Exp C — Bare-lookup migration

**Deliverable:**
edits to `scripts/research_conductor.py` (replace
`task["title"]` etc. with model access at all three sites) +
`tests/python/test_research_conductor_task_access.py` +
`results/experiment_<N>_task_access_migration.json`.

**What it does:**

After Exps A and B land, all three call sites
(`load_research_tasks`, `pick_next_task`,
`_archive_current_milestone`) currently re-parse the YAML into
`dict` objects and bare-index. Migrate them to use the validated
`Roadmap` model from Exp A. Field access becomes
`task.title` instead of `task["title"]`, and the model guarantees
those fields are populated.

**Acceptance:** Bare `task["..."]` lookups are gone from the
conductor. `mypy --strict` passes on the migrated functions.
A future planner that emits valid-but-novel-shape YAML still
works (extra fields are preserved on the model via
`model_config = ConfigDict(extra="allow")`); a planner that
emits incomplete YAML is caught at parse time, not at access time.

## Decentralization implications

Rule 1 (local-first using open models): unaffected. This is
internal infrastructure with no external dependencies.

Rule 7 (no vendor-specific abstractions in the core): the
Pydantic model lives in `python/carnot/conductor/`, which is
already a conductor-specific submodule. No vendor APIs touched.

## Why this is in change-proposals, not just a code change

Same reason `failed-experiment-rerun-enforcement.md` is — the
discipline (every required field on every task) needs an explicit
locus where future-Claude can find it. The CLAUDE.md "do not lose
content" rule and the recurring "infrastructure shipped but not
wired" pattern argue for a written-down contract that the planner
prompt, the conductor code, and the tests can all reference.

The Pydantic model is the single source of truth. Documentation
generated from it stays in sync automatically.

## Risks

- **Schema drift between planner prompt and Pydantic model.** If
  someone updates the planner prompt to use a new field but doesn't
  update the model (or vice versa), the planner's intended output
  will fail validation. Mitigation: the planner prompt should
  reference `python/carnot/conductor/roadmap_schema.py` directly
  ("emit a YAML matching this schema") rather than listing fields
  in prose.

- **Over-validation breaking legitimate experiments.** A real
  research task might want a field structure we didn't anticipate.
  Mitigation: `extra="allow"` on the Pydantic model — required
  fields are validated; extra fields are passed through.

- **The validator itself becomes a bug surface.** A Pydantic version
  bump that changes validation semantics could spuriously reject
  valid roadmaps. Mitigation: pin Pydantic version in
  `pyproject.toml` and gate updates through the test suite.

## Acceptance criteria

1. The exact .69 failure (`KeyError: 'title'` on Exp 896) is
   prevented at planner-output time by Exp B's re-prompt path.
2. Even if Exp B's re-prompt fails, the malformed YAML never
   activates — Exp A's pre-activation validator refuses and writes
   a blocked-artifact.
3. No bare `task["..."]` lookups remain in the conductor.
4. Adding a new required task field is a one-line change to
   `ResearchTask` plus updating the planner prompt; the conductor
   code does not need to change.
5. The conductor never spins indefinitely on a malformed roadmap.
   Either it auto-corrects, or it writes a blocked artifact and
   exits the planning flow with a clear error.
