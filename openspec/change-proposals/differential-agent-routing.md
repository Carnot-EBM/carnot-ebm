# Differential Agent Routing — Pre-emptive Opus for Complex Tasks

**Status:** Implemented (not yet a single-task scope; documents an existing
field that was loosely-typed and now formalized 2026-04-29).

**Origin:** 2026-04-29 milestone .80 close-out observed 11 Opus
escalations across 13 tasks. The C+E (Sonnet→Opus) pattern was
operating reactively; pre-classifying complex tasks would have saved
~2-3 hours of wedge-handling time and 3-4 wasted Sonnet attempts.

**Priority:** Medium. Operator-attention-reduction infrastructure work
of the same class as `conductor-supervisor.md`,
`roadmap-schema-validation.md`, and `conductor-fastpath-bootstrap-skip.md`.

## Problem

The conductor's default agent is Sonnet (max-turns ≤ 50). When Sonnet
exhausts its turn budget on a task, the C+E pattern auto-escalates to
Opus (max-turns 100). This is correct *reactively* — most tasks succeed
on cheap Sonnet — but two failure modes emerge for complex tasks:

1. **Wall-clock cost.** Each failed Sonnet attempt burns 5–15 minutes
   of wall time before triggering the Opus escalation. Across the .80
   milestone, ~30 minutes of wasted Sonnet wall-clock plus the Opus
   retry latency.

2. **Bootstrap-and-bail wedge.** Sonnet's specific failure mode on
   complex tasks (e.g., exp1028 preflight v30) is to write a defensive
   bootstrap artifact (`status: "running"`, all fields false) at the
   prompt's instruction "CRITICAL: write artifact FIRST", then exhaust
   turns before completing the work. Pytest passes (testing the
   conductor's own infra, not Sonnet's task), the conductor logs OK,
   downstream gates read `false` forever, milestone wedges. This
   happened in .80 and required 3 hours + 5 patches to close.

## Solution

Formalize the existing (but undocumented) `model` field on
`ResearchTask` to allow planners to **pre-emptively route complex tasks
to Opus** at planning time, bypassing the wasted Sonnet attempt.

### Schema (now formalized)

```python
class ResearchTask(BaseModel):
    # ...existing fields...
    model: Literal["sonnet", "opus"] | None = None
    escalate_on_max_turns: bool = True
```

- `model: None` (default) → conductor uses `AGENT_MODEL` (Sonnet) and
  the C+E pattern handles max-turns escalation.
- `model: "sonnet"` → explicit Sonnet routing (no behavior change vs.
  default; documents intent).
- `model: "opus"` → skip Sonnet attempt entirely; run Opus directly
  with 100-turn budget. C+E logic detects `task_model == "opus"` and
  does not re-escalate (already on Opus).
- `escalate_on_max_turns: false` → opt out of C+E for tasks that
  genuinely should not retry (e.g., tasks where max-turns indicates
  the task is fundamentally infeasible).

### Planner heuristics

The planner's prompt at `scripts/research_conductor.py:_plan_next_milestone()`
now documents which task categories should set `model: opus`:

1. **Hardware integration:** FPGA bring-up, ROCm probes, KV260 work,
   dual-GPU, nvidia-smi fallbacks.
2. **Schema / preflight infrastructure:** schema validation, manifest
   retirement, gate-cascade fixes, conductor patches.
3. **Multi-step coordination:** tasks bundling several mechanically-
   distinct actions into one experiment.
4. **Bootstrap-and-bail risk:** any prompt instructing "CRITICAL: write
   artifact FIRST" — these are the bootstrap-only artifact wedge class.

Routine experiments (single-question evaluations, training loops,
deliverable-already-exists fast-paths) keep the Sonnet default —
empirically >95% success.

## Cost analysis

Sonnet vs. Opus pricing (rough): Opus is ~3× per token.

- C+E expected cost per task: $1 + (1-X) · $3 = $4 − 3X (where X is
  Sonnet success rate)
- Always-Opus: $3 per task

C+E is cheaper when Sonnet success > 33%. At today's ~77% across the
full milestone, C+E saves ~50% of compute cost.

But for the *complex-task subset* (Sonnet success ~50%), differential
routing saves vs. naive C+E:

- Naive C+E for complex: $1 (failed Sonnet) + $3 (Opus retry) = $4
- Differential routing: $3 (Opus directly) = $3
- Savings: $1 per complex task + ~10 min wall clock

Across milestone .80 (~3 complex tasks), differential routing would
have saved ~$3 in compute and ~30 min wall-clock. More importantly,
it would have prevented the bootstrap-and-bail wedge entirely.

## Acceptance criteria

1. Schema validator (`scripts/roadmap_schema.py`) declares `model` and
   `escalate_on_max_turns` as typed fields.
2. Test coverage: 7 tests in `test_roadmap_schema.py` covering default,
   sonnet, opus, rejection of unknown values, escalate flag default and
   override, YAML round-trip.
3. Planner prompt at `_plan_next_milestone()` documents the `model:
   opus` heuristics for the four complex-task categories.
4. Conductor's `task_model = task.get("model")` reads the field
   correctly (already the case as of `4e6c4d0d`).
5. Future milestones use the field empirically: any task in the four
   categories above has `model: opus` set in the YAML at planner time.

## prior_failures (mandatory)

```yaml
prior_failures:
  - experiment_id: exp1028-preflight-v30-bootstrap
    verdict: bootstrap_artifact_wedged_milestone_80
    addressed_by: |
      The bootstrap-and-bail wedge in .80 was the second occurrence of
      the same failure mode (first was 2026-04-28 17:41Z). This
      proposal pre-emptively routes preflight-class tasks (which all
      use the "CRITICAL: write artifact FIRST" pattern) to Opus
      directly, eliminating the wedge before it can occur.
    retire_if_same_verdict: false  # cannot retire; would re-introduce wedge
```

## Estimated effort

Already implemented. Drafting + tests + this proposal: ~1 hour.
Ongoing planner discipline: trivial (one YAML field per complex task).

## Why this is a "MANDATORY-NEXT-MILESTONE PRIORITIES" entry

This is the *fourth* operator-attention-reduction infrastructure
proposal in four consecutive milestones (after `conductor-supervisor.md`,
`roadmap-schema-validation.md`, and `conductor-fastpath-bootstrap-skip.md`).
Pattern: planner Sonnet drifts toward research breadth; the operator
catches a wedge after-the-fact; the structural fix gets filed; and
unless the proposal is mandatorily picked up in the next milestone,
the same wedge recurs.

This proposal closes the *fourth* wedge class. Hard-pickup for
milestone .81+.

## Strategic rationale

The Round-9 → today's chain proved Carnot's Phase-3 architecture is
optimal modulo a Sawtooth Limit Cycle. Hardware-deployment-class tasks
(KV260 work, FPGA bring-up, ROCm probes) consistently demand more
reasoning depth than Sonnet provides. Pre-emptive Opus routing for
this task class isn't a cost optimization — it's a **prerequisite** for
shipping the hardware-portable architecture without operator-attention
inflation.

## Out of scope

- **Adaptive routing based on past success rate.** A future
  enhancement could classify tasks automatically using the conductor
  log's escalation history. Out of scope for this proposal — manual
  classification by the planner is sufficient and clearer.
- **Re-routing between models mid-task.** The C+E pattern already
  handles max-turns; this proposal only changes *initial* routing.
- **Replacing the C+E pattern.** Differential routing complements
  C+E; routine tasks still benefit from cheap Sonnet attempts.
