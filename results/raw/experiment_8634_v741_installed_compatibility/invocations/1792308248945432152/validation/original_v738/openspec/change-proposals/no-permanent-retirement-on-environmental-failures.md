# No Permanent Retirement on Environmental Failures

**Status:** Draft change proposal, ready for .82 mandatory pickup.
Companion to today's `failed-experiment-rerun-enforcement.md` (the
"don't run doomed reruns") and CLAUDE.md's Failed-Experiment Rerun
Discipline (the "if you re-run, name the prior failure"). This
proposal closes the *complementary* gap: don't retire experiments
*permanently* when their failure was caused by operational issues
rather than experiment-design issues.

**Origin:** 2026-04-29 evening operator directive — *"don't give up
entirely on experiments due to operational interruptions and issues...
find a way to divide up the experiment into smaller experiments or
find another way for the experiments themselves to make forward
progress until their merits are proven or disproven."*

**Priority:** **HIGH** for .82. Without this, today's three .81
retirements (exp1039, exp1042, exp1044) become permanently lost
research effort despite all three having environmental root causes
that are now fixed.

## Problem statement

The conductor's `MAX_FAILURES_PER_TASK = 3` retirement is binary: a
task that fails 3 times for *any* reason gets permanently retired
(via fail-count exhaustion in `pick_next_task`). The planner has no
mechanism to distinguish:

- **Environmental failures** — pre-test wedges, timeout-bug hangs,
  gate-blocks on retired-upstream tasks, conductor self-protection
  guard reverting an edit. These have NOTHING to do with whether the
  experiment's *idea* works.
- **Merit-based failures** — the experiment ran cleanly, produced
  an artifact with an honest verdict like `no_improvement`,
  `blocked_no_signal`, `still_wrong`, `flat`. The idea was tested
  and disproven.

Without distinguishing these, environmental wedges quietly *kill
research progress*. Today's .81 retirements are the canonical case:

- **exp1039** (conductor-fastpath-gate-coercion) — retired after 3
  pre-test SKIP/FAIL during the test_conductor_gates wedge. The
  actual gate-coercion work was never attempted.
- **exp1042** (dualgpu-rocm-torch-v4) — retired after 3 SKIP/FAIL
  during the same wedge + run_agent timeout bug. The actual ROCm
  install was never attempted.
- **exp1044** (triple-integration-v7) — retired after 3 GATE_BLOCKs
  on exp1039's missing artifact. The actual cascade validation was
  never attempted.

All three are now retried-environmental-fix-in-place but absent any
auto-respawn mechanism, the .82 planner will skip them on the
principle of "they were retired, leave them alone."

## The variance discipline (CRITICAL)

**Per the operator directive (2026-04-29 evening):** *"The definition
of insanity is repeating the same thing and expecting a different
outcome. Each of the subsequent retries must have some variance
applied to attempt to work around whatever might have caused the
previous failed attempts."*

**Pure retry without variance is forbidden.** Each respawn attempt
must apply *deliberate variance* to address a different hypothetical
failure cause. The variance ladder progresses from cheap/small
variances to expensive/large ones, and stops at decomposition rather
than infinite retries.

### Variance ladder (mandatory ordering)

**Attempt 1 (tier escalation):** move up the model + max_turns
ladder. Variance dimensions: `model` (sonnet → opus), `max_turns`
(25/30/50 → 100). Rationale: many environmental failures are
capacity-bound, not logic-bound; a stronger model with more turns
often clears them.

**Attempt 2 (backend rotation):** switch agent backend
(claude → codex / gemini). Different training distributions →
different failure modes. Especially relevant for code-heavy tasks
where Codex's broader code corpus may avoid Claude-specific stall
patterns. Variance dimensions: `agent_type`, `model` (vendor-specific
identifier).

**Attempt 3 (decomposition):** operator-driven split into N smaller
sub-experiments, each with smaller scope. Permanent retirement only
fires if even the decomposed sub-experiments fail with *merit*
verdicts (not environmental). Variance dimension: `scope_decomposition`.

### Dependency-resolution variance (special case)

When the failure mode is GATE_BLOCK on a retired upstream, the
*variance* for attempt 1 may be the dependency resolution itself
(running the upstream respawn first), not a model/turns change. If
the upstream respawn produces the gated field, the downstream
respawn becomes a normal task at its original scope.

If the upstream respawn also fails, the downstream's attempt 2
variance is "remove the gate, run unguarded with explicit pre-check"
— eliminates the gate-block failure mode entirely while preserving
the experiment's actual scope.

### Variance is mandatory in the queue schema

Each queue entry MUST include:

- `variance_strategy`: one of `tier_escalation`, `backend_rotation`,
  `decomposition`, `tier_escalation_with_dependency_resolution`.
- `variance_applied`: explicit dict comparing `from_original` vs.
  `to_respawn_N` parameters with a `rationale` field explaining
  *why* this variance addresses the failure mode.
- `next_attempt_variance_plan`: the planned variance for the *next*
  respawn if this one fails. Pre-committing the next variance keeps
  the planner honest — no panic-retry-with-same-config when an
  attempt fails.

The schema validator (Mechanism A test coverage) refuses to load a
respawn entry without these fields populated.

## Solution: respawn queue + auto-respawn discipline

### Mechanism A: respawn queue (file-backed)

A new `ops/respawn-queue.json` lists tasks that retired due to
environmental causes. Schema:

```json
{
  "schema": "carnot.respawn_queue.v1",
  "queue": [
    {
      "original_id": "exp1039-conductor-fastpath-gate-coercion",
      "title_stub": "Conductor Fastpath Bootstrap Skip + Gate Coercion Fix",
      "retire_milestone": "2026.04.81",
      "retire_verdicts": ["pre_test_wedge", "pre_test_wedge", "max_turns"],
      "diagnosed_root_cause": "test_conductor_gates default mismatch + run_agent timeout dead-code bug",
      "fixes_landed": ["7a13304d", "b2c73a08"],
      "respawn_attempt": 1,
      "max_respawn_attempts": 3,
      "model": "opus",
      "max_turns": 100
    }
  ]
}
```

### Mechanism B: planner consumes the queue

When the planner generates `research-roadmap-next.yaml`, it reads
`ops/respawn-queue.json` and emits respawn tasks with:

1. New ID (`<original_id>-respawn-N` where N is `respawn_attempt`)
2. **Auto-populated `prior_failures` block** referencing each prior
   retirement and the diagnosed root cause + fix
3. `retire_if_same_verdict: true` — if it fails the same
   environmental way, it permanently retires (the discipline does
   NOT allow infinite respawns)
4. `model: opus` and `max_turns: 100` by default (respawned tasks
   skip the wasted Sonnet attempt)

### Mechanism C: conductor auto-populates the queue

When `pick_next_task` retires a task via MAX_FAILURES, the
conductor's retirement logic classifies the failure pattern:

- **Environmental class** (auto-queue for respawn):
  - All retirement verdicts are `SKIP` from "pre-tests failing"
  - All retirement verdicts are `FAIL` with "Reached max turns"
    error message
  - All retirement verdicts are `GATE_BLOCK` (gated on retired-upstream)

- **Merit class** (NO respawn — genuine retirement):
  - Any retirement verdict is `FAIL` from a written artifact with an
    honest_verdict containing "no_improvement", "below_baseline",
    "flat", "still_wrong", "no_delta", or similar empirical-failure
    tokens
  - Any retirement verdict is `partial:tests_still_failing` or
    similar
  - Any retirement verdict has been previously respawned 3 times
    with the same environmental class (Mechanism B's
    `max_respawn_attempts`)

### Mechanism D: decomposition fallback

If a task respawns 3 times environmentally and *still* fails on
the same wedge class, the conductor surfaces it to the planner with
a `decompose_recommended: true` flag. The planner then breaks the
task into smaller sub-tasks (`expNNNN-foo-stage-1`, `expNNNN-foo-stage-2`)
each with smaller scope, with the original verdict tracked. This
implements the operator directive's *"divide up the experiment into
smaller experiments"* path.

## Acceptance criteria

1. **Schema** — `ops/respawn-queue.json` schema validated by a new
   test in `tests/python/test_respawn_queue.py`.
2. **Conductor retirement classification** — `pick_next_task`
   distinguishes environmental from merit retirements; auto-queues
   environmental retirements via a new `_classify_retirement_kind()`
   helper.
3. **Planner respawn consumption** — `_plan_next_milestone()`
   reads the queue and emits respawn tasks with auto-populated
   `prior_failures` blocks.
4. **Decomposition trigger** — after 3 respawns same-class, the
   planner is prompted to decompose. Operational discipline, not
   automated YAML rewriting.
5. **Test coverage** — at least 8 tests covering classification of
   each retirement kind, queue round-trip, and respawn-prior_failures
   generation.

## Operator directive translation

The operator's directive maps to these three concrete mechanisms:

> *"not give up entirely on experiments due to operational
> interruptions and issues"*
- ✅ Mechanism A (queue) + B (planner consumes) + C (conductor
  auto-populates).

> *"find a way to divide up the experiment into smaller experiments"*
- ✅ Mechanism D (after 3 respawn-same-class, decompose).

> *"find another way for the experiments themselves to make forward
> progress until their merits are proven or disproven"*
- ✅ Mechanism C distinguishes environmental from merit failures —
  merit-based retirements (e.g., `no_improvement`) are honest
  research dead-ends and are NOT auto-respawned. Environmental
  retirements stay in the queue until merit is *actually*
  evaluated.

## prior_failures (for this proposal itself)

```yaml
prior_failures: []  # Genuinely new mechanism. The closest precedent
                    # is failed-experiment-rerun-enforcement.md (which
                    # prevents doomed reruns), but this proposal is the
                    # complementary "ensure non-doomed reruns happen"
                    # policy.
```

## Today's three .81 retirements — initial respawn queue

To validate the mechanism, today's three retired tasks should be
the seed entries in `ops/respawn-queue.json` (manually populated;
auto-queueing is .82 work):

```json
{
  "schema": "carnot.respawn_queue.v1",
  "queue": [
    {
      "original_id": "exp1039-conductor-fastpath-gate-coercion",
      "title_stub": "Conductor Fastpath Bootstrap Skip + Gate Coercion Fix",
      "retire_milestone": "2026.04.81",
      "retire_verdicts": ["pre_test_wedge_SKIP", "max_turns_FAIL", "pre_test_wedge_SKIP"],
      "diagnosed_root_cause": "test_conductor_gates default 50 vs function default 100 (mismatch) + run_agent timeout parameter was dead code",
      "fixes_landed": ["7a13304d", "b2c73a08"],
      "respawn_attempt": 1,
      "max_respawn_attempts": 3,
      "model": "opus",
      "max_turns": 100
    },
    {
      "original_id": "exp1042-dualgpu-rocm-torch-v4",
      "title_stub": "DualGPU ROCm/CUDA Torch Install + Live Inference v4",
      "retire_milestone": "2026.04.81",
      "retire_verdicts": ["max_turns_FAIL", "pre_test_wedge_SKIP", "pre_test_wedge_SKIP"],
      "diagnosed_root_cause": "Same wedge as exp1039 (test_conductor_gates + run_agent timeout); model: opus + max_turns: 25 was also too tight (Opus needs 100)",
      "fixes_landed": ["7a13304d", "b2c73a08"],
      "respawn_attempt": 1,
      "max_respawn_attempts": 3,
      "model": "opus",
      "max_turns": 100
    },
    {
      "original_id": "exp1044-triple-integration-v7",
      "title_stub": "Triple Integration E2E v7 — Post Gate Coercion Fix",
      "retire_milestone": "2026.04.81",
      "retire_verdicts": ["GATE_BLOCK", "GATE_BLOCK", "GATE_BLOCK"],
      "diagnosed_root_cause": "Gated on exp1039's gate_coercion_fixed field; exp1039 retired without artifact, so gate failed forever",
      "fixes_landed": ["7a13304d", "b2c73a08", "4e46ede6"],
      "depends_on_respawn": ["exp1039-conductor-fastpath-gate-coercion"],
      "respawn_attempt": 1,
      "max_respawn_attempts": 3,
      "model": "sonnet",
      "max_turns": 50
    }
  ]
}
```

The `.82` planner reads this queue and emits respawn tasks for each.

## Strategic alignment

This proposal directly addresses today's operational pattern:
8 conductor structural-bug fixes shipped today, each invalidating a
prior retirement's environmental verdict. Without the respawn
mechanism, those fixes don't auto-recover the lost work — the
operator would have to manually identify retirements, manually
populate prior_failures, and manually queue them. The mechanism
formalizes this work and makes forward progress automatic.

It's the seventh operator-attention-reduction infrastructure
proposal in the recent series:

1. `conductor-supervisor.md` (.81 mandatory — landed)
2. `roadmap-schema-validation.md` (.81 mandatory — landed)
3. `conductor-fastpath-bootstrap-skip.md` (.81 mandatory — landed)
4. `differential-agent-routing.md` (today — landed)
5. `multi-agent-routing.md` (today — landed)
6. `parallel-multi-agent-conductor.md` (today's draft — .82 mandatory)
7. **`no-permanent-retirement-on-environmental-failures.md`** (this
   proposal — .82 mandatory)

## Out of scope

- **Automated experiment decomposition** (Mechanism D's "the
  conductor splits a multi-step experiment into sub-experiments"
  variant) — operator-driven decomposition is sufficient; automated
  YAML rewriting is risky.
- **Distinguishing environmental sub-classes** (e.g., "pre-test
  wedge" vs. "timeout hang" vs. "GATE_BLOCK") — single
  environmental-vs-merit boolean is enough for the queue
  mechanism. Sub-classification is future refinement.
- **Re-evaluating already-permanently-retired pre-2026-04-29
  experiments** — the queue starts with today's three .81
  retirements; older retirements stay retired unless explicitly
  resurrected by an operator.
