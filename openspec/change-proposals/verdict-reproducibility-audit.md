# Verdict Reproducibility Audit

**Status:** Draft change proposal. **REQUESTED FOR MILESTONE 2026.04.81
(MANDATORY)** — pinned in `ops/known-issues.md → MANDATORY-NEXT-MILESTONE
PRIORITIES`.

**Origin:** 2026-04-29 01:13Z observation. Experiment 1031 Energy-
Selection SSD v3 was re-run by the conductor and produced a *different
honest_verdict* than its prior session run:

- 21:12Z run: `honest_verdict: fr11_loop_closed` (flagship Phase 1 result)
- 01:13Z run: `honest_verdict: carnot_filter_below_baseline` (negative result)

Same code path, same milestone, ~4 hours apart. The verdict is **non-
reproducible**. This is the same class of failure as the 2026-04-28
inverted-AUROC bug — but more dangerous because it's not a code bug,
it's a *discipline gap*: experiments use stochastic samplers (jax.random,
numpy.random, torch.cuda) without seed control, and the verdict can flip
across runs without anyone noticing.

**Target milestone:** **2026.04.81** (mandatory pickup).

**Priority:** **High.** The 12-round Zenil chain produced 10 publishable
contributions; the Kinematic Layer Routing chain added 2 more. Several
of those contributions will be backed by empirical experiments shipped
in upcoming milestones. *If those experiments produce non-reproducible
verdicts, the position paper is vulnerable to reviewer reproducibility
audit.* The credibility risk is now load-bearing.

**Depends on:** nothing. Closes the operational gap before the math
ships.

## Summary

Three scoped experiments:

1. **Audit:** rerun the last $N \geq 5$ milestone-flagship verdicts
   with full environmental control. Quantify how many produce the
   *same* verdict label.
2. **Seed discipline:** add `random_seed` to `experiment_template.py:build_result()`
   schema; require all RNGs in experiment code to derive from it
   deterministically.
3. **Headline-result lock-in:** any deliverable whose `honest_verdict`
   appears in `research-complete.yaml` as a flagship result MUST
   include the seed + a reproducibility checksum (hash of code SHA +
   data hashes + seed). On rerun, the checksum is recomputed and
   verified.

## What this proposal IS NOT

- **Not full bit-exact reproducibility.** Floating-point non-associativity
  + GPU non-determinism mean exact reproduction is impossible. The goal
  is *verdict-stable* reproduction: same `honest_verdict` label on rerun.
- **Not a sweeping refactor of every experiment.** Existing experiments
  retain their results; the discipline applies to new flagship results
  going forward.
- **Not a CI-style continuous reproducibility check.** Reruns are
  manual, audit-driven. Continuous re-verification is a future scope.

## Proposed experiments

### Exp A — Audit existing flagship verdicts

**Deliverable:** `scripts/experiment_<N>_verdict_reproducibility_audit.py` +
`results/experiment_<N>_verdict_audit.json`.

**What it does:**

1. Identify the last 5 milestone-flagship results (those in
   `research-complete.yaml` with status="success" and a non-trivial
   verdict): exp1020 (almc_mpemba_wins_both), exp1029 (partial_z3_only),
   exp1031 (fr11_loop_closed), exp1032 (relay_live), exp1020 ALMC v2.
2. For each: rerun the experiment script *without* changes. Compare
   `honest_verdict` on rerun vs original.
3. Report:
   - Count of stable verdicts (same label both runs).
   - Count of unstable verdicts (different labels).
   - For each unstable: the diff between runs (which metric flipped,
     by how much, on what data).
4. Write `verdict_audit_complete` artifact with reproducibility rate.

**Acceptance:** report quantifies reproducibility rate (e.g., "3 of 5
flagship verdicts are stable; 2 of 5 flip across runs"). The report
is the input to Exp B's seed-discipline rollout.

### Exp B — Seed discipline + canonical RNG initialization

**Deliverable:** edits to `scripts/experiment_template.py` +
`python/carnot/eval/seed_control.py` (new) +
`tests/python/test_seed_discipline.py` +
`results/experiment_<N>_seed_discipline.json`.

**What it does:**

1. Add `random_seed: int | None = None` parameter to
   `ExperimentTemplate.__init__`. Default behaviour generates a seed
   from `time_ns()` and *records it in the artifact*.
2. New module `python/carnot/eval/seed_control.py`:
   - `set_global_seeds(seed: int)` — seeds python's `random`,
     `numpy.random`, `jax.random.PRNGKey`, `torch.manual_seed`,
     `torch.cuda.manual_seed_all`.
   - `derive_seed(seed: int, namespace: str) -> int` — deterministic
     sub-seeds from a parent seed for compositional use.
3. `experiment_template.py:build_result()` accepts `random_seed` and
   includes it in the artifact's top-level fields.
4. Add **forbidden patterns** to `scripts/batching_precommit_check.py`:
   reject calls to `numpy.random.rand`, `jax.random.PRNGKey(seed=42)`
   with hardcoded literals, etc., that bypass the seed-control module.
5. 12+ unit tests covering seed propagation, sub-seed determinism, and
   forbidden-pattern rejection.

**Acceptance:** module imports cleanly, tests pass; new experiments
inheriting from `ExperimentTemplate` automatically get
seed-controlled RNGs and seed-recording in artifacts.

### Exp C — Headline-result lock-in via reproducibility checksum

**Deliverable:** `python/carnot/eval/reproducibility_checksum.py` (new) +
edits to `scripts/experiment_template.py:build_result()` +
`tests/python/test_reproducibility_checksum.py` +
`scripts/audit_headline_reproducibility.py` (new utility) +
`results/experiment_<N>_reproducibility_checksum.json`.

**What it does:**

1. New module `python/carnot/eval/reproducibility_checksum.py`:
   - `compute_checksum(seed: int, code_paths: list[str], data_paths: list[str]) -> str`
     returns a SHA256 of (seed bytes + git-blob-hashes of code + content
     hashes of data files). Stable across machines if inputs are identical.
   - `verify_checksum(artifact_path: Path) -> CheckResult` re-derives
     the checksum from the artifact's recorded inputs and compares.
2. `experiment_template.py:build_result()` accepts
   `code_paths` and `data_paths` lists; adds
   `reproducibility_checksum` field to the artifact.
3. `scripts/audit_headline_reproducibility.py` walks
   `research-complete.yaml` flagship-status entries; for each,
   recomputes the checksum from the artifact's recorded inputs and
   reports any mismatches. Used by the planner Sonnet during retro.
4. **Optional rerun mode** (`--rerun`): for each flagged headline
   result, actually re-execute the experiment with the recorded seed
   and verify the verdict label matches. Costly but the gold standard.
5. 10+ unit tests covering checksum stability, mismatch detection,
   rerun-mode comparison.

**Acceptance:** new flagship results carry checksums; audit script
can be run on the entire `results/` directory to surface
reproducibility-flagged deliverables in <30s.

## Decentralization implications

**Rule 1 (local-first):** unaffected.

**Rule 7 (no vendor abstractions in core):** strengthened. Reproducibility
checksums use only stdlib (`hashlib`, `subprocess` for git) — no third-
party reproducibility framework dependencies.

**New implication:** sovereign reproducibility is part of sovereign
publication. If a verdict is sensitive to environment (CUDA version,
specific GPU, etc.), that environment dependency is recorded in the
checksum, making the *required* environment explicit and forkable.

## Risks

- **Not all experiments are deterministic.** GPU non-determinism (e.g.,
  CUDA atomics in attention) means even with seed control, runs differ.
  Mitigation: the verdict label needs to be stable to *small* numeric
  differences; if a verdict flips because a metric crosses a threshold
  by 0.001 due to GPU noise, that's an *experiment design* bug, not a
  reproducibility bug. Exp A audit will surface these.
- **Audit cost.** Rerunning 5 experiments takes wall time. Mitigation:
  use the cheapest experiments first; deferred rerun for expensive
  ones.
- **Retroactive application.** Existing flagship results don't have
  recorded seeds. Mitigation: only apply the discipline to new
  flagship results; treat existing ones as documented-non-reproducible
  and audit them via Exp A's rerun.

## Acceptance criteria

1. Audit report quantifies reproducibility rate of the last 5 flagship
   verdicts.
2. `experiment_template.py:build_result()` records `random_seed` and
   `reproducibility_checksum` in every artifact.
3. `python/carnot/eval/seed_control.py` provides canonical RNG
   initialization; tests pass.
4. `scripts/audit_headline_reproducibility.py` walks results/, surfaces
   reproducibility-flagged deliverables.
5. The forbidden-pattern check in `scripts/batching_precommit_check.py`
   rejects new bypasses of seed control.
6. CLAUDE.md updated with a new "Reproducibility Discipline" section
   in Operational Principles, citing this proposal.

## Why this is in change-proposals, not just a code change

The discipline around reproducibility cuts across:
- The `experiment_template.py` API contract (every new experiment).
- The audit script that the planner Sonnet uses during retro.
- The forbidden-pattern check in pre-commit.
- The decentralization rule on sovereign reproducibility.

CLAUDE.md is the planner's required reading; the rule belongs there
with explicit rationale that links back to the 2026-04-29 verdict-
change incident as the load-bearing precedent.

## Connection to prior proposals

- **`eval-metrics-canonical-and-self-heal-production-bug-detector.md`:**
  addressed metric-implementation provenance (which formula). This
  proposal addresses run-to-run verdict variance (which seed). Together
  they cover the two main reproducibility failure modes.
- **`zenil-grounded-self-distillation-deployable-stack.md`:** Exp E3
  (red-team RL agent) is itself a heavily stochastic experiment;
  reproducibility checksum is mandatory for its verdicts to be
  citable.
