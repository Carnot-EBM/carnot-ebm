# Research Roadmap v254: Venv Hardening + Phase 1 Ship + EORM Tier 0e + ODAR + Ensemble v11

**Milestone:** 2026.05.254
**Previous milestone:** 2026.05.253
**Date:** 2026-05-20
**Status:** PROPOSED

## Root-Cause Analysis: .253 Zero-Execution Incident

Milestone 2026.05.253 had **zero experiments committed**. All 13 tasks (exp2647–exp2659) exhausted
with "SKIP: Pre-tests failing, self-heal failed: [empty]". Conductor conductor-state.json confirms
the conductor was running (iteration 264), but every pre-test attempt produced empty output and
SIGKILL'd pytest, triggering the SKIP path 3×per task until exhausted.

**Root cause (confirmed 2026-05-20T13:37Z by planning agent):**

The `.venv` in the project root was missing `pip`, `pytest`, `jax`, and `scikit-learn`. The
conductor calls `venv_pytest = ".venv/bin/pytest"` which did not exist, so every `subprocess.run()`
call returned returncode=-1 with empty stdout/stderr — matching the "empty self-heal message"
pattern in the conductor log exactly.

**Fix applied in this planning session (2026-05-20T13:38Z):**

```bash
.venv/bin/python -m ensurepip               # install pip 26.0.1
.venv/bin/python -m pip install pytest pytest-cov  # install test runner
.venv/bin/python -m pip install -e .        # install project + JAX deps
.venv/bin/python -m pip install scikit-learn  # restore ML library
```

Post-fix: `.venv/bin/pytest tests/python/test_pipeline_extract.py tests/python/test_docs.py` → **81 passed in 3.10s**.

**Why this persisted for a full milestone (.253):** The `.venv` was re-created (or the packages
were stripped) between sessions. The conductor's `env_autofix` is gated on JAX being importable;
since JAX wasn't installed, `env_autofix` was also unavailable, preventing the only automated
fix path from running. Defense-in-depth: exp2661 creates `scripts/setup-venv.sh` and a Makefile
target to make re-population idempotent.

---

## Post-.253 Planning Sweep New Papers (2026-05-20)

Post-.253 sweep adds one structural finding (no new arxiv papers alter the milestone priorities):

- **Pre-test environment fragility (structural finding):** The .253 incident confirms that
  `pyproject.toml`'s `[project.optional-dependencies]` does not include `pytest` in a
  dev-extras group. If `.venv` is created with `python -m venv .venv` (no `--system-site-packages`)
  and `pip install -e .` is not run before the conductor starts, the pre-test gate silently fails.
  exp2661 closes this by adding `[project.optional-dependencies.dev]` = `["pytest>=7.0", "pytest-cov"]`
  and a `scripts/setup-venv.sh` that runs the full install sequence.

---

## What Milestone .253 Proved

**Nothing new executed.** All 13 tasks SKIPPED due to pre-test infrastructure failure. Key carry-forwards
from .252 (which were .253's inputs):
- Ensemble v10 adversarially validated (exp2638): AUROC headline for paper-v6
- Phase 1 ship readiness audited (exp2642): phase1_ship_ready bool + operator_action_checklist
- arXiv v5 package built (exp2643): submission_package_ready bool (OPERATOR-ONLY submission)
- TTT statistical significance established (exp2639): fr11_tier3_headline_closed bool
- KV260 NON-TERMINAL (exp2644): SD card absent; Branch B executed

---

## Three Biggest Gaps Between Current State and PRD Vision

### Gap 1: Pre-Test Environment Fragility (MUST FIX FIRST)

**State:** `.venv` missing pip/pytest/JAX caused 100% SKIP rate in .253. Planning agent fixed
it interactively, but the fix is not durable across session boundaries.

**Why it matters:** Every subsequent task depends on the conductor's pre-test gate passing. Without
a reproducible setup, the next `python -m venv .venv` + conductor restart will re-enter the zero-
execution failure mode.

**Fix:** exp2661 — create `scripts/setup-venv.sh`, update `pyproject.toml` dev-extras, add Makefile
target `make setup-venv`. Gate all research tasks on exp2661.pre_tests_passing=true.

### Gap 2: Phase 1 Ship Execution (HIGHEST RESEARCH PRIORITY)

**State:** .252 exp2642 audited all 4 Phase 1 gates and exp2645 produced a SHIP/HOLD
recommendation. The actual ship actions (version tag preparation, README Phase 1 section,
release notes) were NOT executed in .252 or .253.

**PRD requirement:** "Phase 1 ship gate is purely software-operational: PyPI package + Apache-2.0
shipped, HuggingFace mirror, MCP server + CLI documentation, at least one independent reproducer."

**Fix:** exp2662 — read exp2642 artifact, execute Branch A (if ready) or Branch B (close remaining
gates). Operator tag release + announcement are OPERATOR-ONLY per CLAUDE.md.

### Gap 3: Ensemble v11 + ODAR Active Inference (RESEARCH ADVANCE)

**State:** Ensemble v10 adversarially validated at mean AUROC ≥ 0.90 (5 seeds). Three new verifiers
from the .253 sweep (Tier 0e EORM, Tier 0l layer-wise, ODAR routing) have not been prototyped.
Phase 4 active inference remains a parallel track with no operational implementation.

**Fix:** exp2663 (Tier 0e), exp2664 (Tier 0l), exp2667 (Ensemble v11 + adversarial val), exp2668
(ODAR routing). Critical path: exp2663 → exp2667 → exp2671 (arXiv v6).

---

## Architecture: .254 Dependency Graph

```
exp2660 (archive)
    └─► exp2661 (pre-test hardening) ─────────────────────────────────────────┐
              └─► exp2662 (Phase 1 ship)                                       │
              └─► exp2663 (Tier 0e EORM) ──► exp2667 (ensemble v11) ──► exp2669 (ext bench)
              └─► exp2664 (Tier 0l)           └────────────────────────► exp2671 (arXiv v6)
              └─► exp2665 (VegAS)                                              │
              └─► exp2666 (NEXUS / FR-11)                                     │
              └─► exp2668 (ODAR routing)                                       │
              └─► exp2670 (KV260 hardware)                                     │
              └────────────────────────────────────────────────────────────────┘
                                                             └─► exp2672 (capstone)
```

All research tasks (exp2662–exp2670) gate on `exp2661.pre_tests_passing == true`.
Ensemble v11 (exp2667) gates on `exp2663.tier0e_viable`.
External benchmark (exp2669) and arXiv v6 (exp2671) gate on `exp2667.adversarially_verified`.

---

## Phase Breakdown

### Phase 0: Infrastructure Recovery (exp2660–exp2661)

- **exp2660**: Archive .253 → research-complete.yaml + activate .254 by copying research-roadmap-next.yaml to research-roadmap.yaml.
- **exp2661**: Pre-test venv hardening — ensurepip + pytest + JAX + scikit-learn install, setup-venv.sh script, pyproject.toml dev-extras, Makefile target; verifies 81 tests pass; writes `pre_tests_passing: true`.

### Phase 1: Ship Execution (exp2662)

- **exp2662**: Phase 1 Ship Close — reads exp2642 audit artifact; Branch A (phase1_ship_ready=true: prepare release tag + README update + RELEASES.md + operator ship checklist) or Branch B (close remaining autonomously-doable gates from exp2642.operator_action_checklist).

### Phase 2: New Verifiers (exp2663–exp2666)

- **exp2663**: Tier 0e — EORM-style TF-IDF logistic regression trained on FoVer (correct, incorrect) pairs via margin-based ranking loss; writes tier0e_auroc + tier0e_viable.
- **exp2664**: Tier 0l — Layer-wise information drift verifier; TF-IDF-based inter-sentence cosine drift proxy; writes tier0l_auroc + tier0l_viable.
- **exp2665**: VegAS — K=3 repair candidates scored by ensemble; lowest-energy selected; writes vegas_delta_vs_argmax + vegas_viable.
- **exp2666**: FR-11 Tier 2 NEXUS — symbolic constraint memory accumulator; violation-pattern → symbolic rule synthesis; writes constraints_synthesized + memory_updated (continuous_self_learning_task: true).

### Phase 3: Ensemble + Active Inference (exp2667–exp2668)

- **exp2667**: Ensemble v11 — incorporates Tier 0e + Tier 0l; 5-seed adversarial validation; writes ensemble_v11_auroc_mean + adversarially_verified; gates arXiv v6.
- **exp2668**: ODAR routing — KL-gated fast/slow path in VerifyRepairPipeline; measures routing_overhead_reduction_pct; writes odar_viable.

### Phase 4: Downstream Artifacts (exp2669–exp2671)

- **exp2669**: External Benchmark — Carnot ensemble v11 vs EORM leaderboard; HalluScan + PARALLAX OOD evaluation; gated on exp2667.adversarially_verified.
- **exp2670**: KV260 Hardware Continuity — SD card absent Branch B (update prep script + document next step for operator).
- **exp2671**: arXiv Final Package v6 — ensemble v11 numbers + EORM comparison; submission_package_ready for OPERATOR submission; gated on exp2667.adversarially_verified.

### Phase 5: Synthesis (exp2672)

- **exp2672**: Capstone v254 — cross-artifact synthesis; Phase 1 ship status; top-3 gaps for .255; requires_claude: true.

---

## Hardware Requirements

| Resource | Required by | Status |
|---|---|---|
| `.venv/bin/pytest` | exp2661 (fix) + all tasks | Fixed in planning session |
| JAX CPU | exp2663, exp2664, exp2666, exp2667 | `.venv/bin/python -c "import jax"` works post-fix |
| scikit-learn | exp2663, exp2664, exp2667 | `.venv/bin/python -c "import sklearn"` works post-fix |
| FoVer corpus | exp2663, exp2664 | `data/fover_corpus.jsonl` (8829 lines confirmed) |
| KV260 | exp2670 | SD card absent; Branch B only |
| RTX 3090 x2 | none (all GGUF tasks deferred to .255) | Both idle and healthy |

---

## Agent Routing Summary

| Task | Agent | Model | Justification |
|---|---|---|---|
| exp2660 | codex | gpt-5.5 | Admin archive — mechanical YAML update |
| exp2661 | codex | gpt-5.5 | Package install + setup script — formulaic |
| exp2662 | codex | gpt-5.5 | Phase 1 ship close — file reads + README edit |
| exp2663 | codex | gpt-5.5 | Tier 0e sklearn logistic regression — formulaic |
| exp2664 | codex | gpt-5.5 | Tier 0l TF-IDF drift — formulaic |
| exp2665 | codex | gpt-5.5 | VegAS K=3 candidate selection — formulaic |
| exp2666 | codex | gpt-5.5 | NEXUS constraint memory — formulaic |
| exp2667 | codex | gpt-5.5 | Ensemble v11 — single file sklearn refit + 5-seed eval |
| exp2668 | codex | gpt-5.5 | ODAR routing — single method modification |
| exp2669 | codex | gpt-5.5 | External benchmark scoring — formulaic |
| exp2670 | codex | gpt-5.5 | KV260 Branch B — documentation update |
| exp2671 | codex | gpt-5.5 | arXiv package update — mechanical LaTeX splice |
| exp2672 | claude | opus | Capstone synthesis — multi-file cross-artifact reasoning |

12/13 codex (92.3%), 1/13 claude+opus (7.7% — capstone only). Codex-Default discipline maintained.

---

## CLAUDE.md Mandatory Disciplines Checklist

- [x] **Codex-Default**: 12/13 codex; 1/13 claude+opus (capstone only)
- [x] **Prior-Failures**: All 13 tasks have prior_failures blocks (exp2660–exp2672 all reference .253 predecessors)
- [x] **PRECONDITIONS step 0**: All compute-bound tasks have explicit precondition checks
- [x] **Principle-Annotated Artifact Fields**: All REQUIRED ARTIFACT FIELDS carry one-line `principle:` annotation
- [x] **Terminal-Prefix Verdict Discipline**: All honest_verdict specs start with `complete:` / `success:` / `passed:` / `shipped:`
- [x] **Operator-Only External Publication**: exp2671 prepares package; never submits; operator checklist produced
- [x] **Hardware-Task Continuity**: exp2670 KV260 (NON-TERMINAL); GateMate + PolarFire TERMINAL (graduated)
- [x] **Exclusion-Manifest Cross-Check**: 0 scope matches found across all retired experiment IDs
- [x] **FR-11 Self-Learning Mandate**: exp2666 (NEXUS Tier 2, continuous_self_learning_task: true)
- [x] **Failed-Experiment Rerun Discipline**: All .253 re-proposals include prior_failures with all 4 sub-fields
- [x] **Structured gated_on fields**: exp2663–exp2671 gate on exp2661.pre_tests_passing; exp2667 gates on exp2663.tier0e_viable; exp2669+exp2671 gate on exp2667.adversarially_verified
