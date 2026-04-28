# Carnot — Known Issues

**Last Updated:** 2026-04-28

## MANDATORY-NEXT-MILESTONE PRIORITIES (.80 planner — hard pickup per CLAUDE.md)

Per the **CLAUDE.md "Overdue-Priority Forcing Function"** rule, any priority
pending 3+ consecutive milestones MUST be picked up by the next planner.
The following entries pass that threshold and are mandatory for the .80
roadmap:

  - **`openspec/change-proposals/conductor-supervisor.md`** (4 exps,
    pending since .77 — **3 milestones overdue**) — external observer
    process catching log-handle severance, claimed-vs-actual state
    drift, conductor wedge, and bounded auto-recovery whitelist. Single
    biggest unblock for unattended operation.

  - **`openspec/change-proposals/roadmap-schema-validation.md`**
    (3 exps, pending since .77 — **3 milestones overdue**) — Pydantic
    enforcement at planner output + activation. Prevents the schema-
    drift stillborn-milestone pattern (.69, .74).

  - **`openspec/change-proposals/eval-metrics-canonical-and-self-heal-production-bug-detector.md`**
    (4 exps, drafted 2026-04-28; partial work shipped already) — fixes
    the AUROC-bug class structurally. Migrates per-experiment metric
    helpers to canonical `carnot.eval.metrics` + adds production-bug
    detector to conductor self-heal + provenance tagging. The 2026-04-28
    inverted-AUROC discovery would have been impossible with this
    discipline in place. **Pre-shipped components:**
      - ✅ `python/carnot/eval/metrics.py` (15 tests + sklearn x-val)
      - ✅ `scripts/audit_metric_provenance.py`
      - ✅ `scripts/conductor_commit_watchdog.sh`
      - ✅ `experiment_template.py:build_result(metrics_used=...)`
      - ✅ AUROC fixes in exp995 + exp1003

  - Bonus (pending since .77, **3 milestones overdue**):
    **`openspec/change-proposals/conductor-otel-tracing.md`** (5 exps)
    — depends on the supervisor; lower priority but the natural next
    step.

  - **`openspec/change-proposals/zenil-grounded-self-distillation-deployable-stack.md`**
    (4 exps, drafted 2026-04-28) — ships the four code artifacts that
    operationalise the Round-6 Deep Think result on verifier-filtered
    self-distillation: Φ > 0 measurement module, joint annealing
    schedule, PT acceptance hyperparameter (0.35), and the
    REQ-PHASE2-006 Gray-code factor experiment. Mathematically
    justifies the Phase 2 hardware mandate (`_bmad/architecture.md`)
    and produces a publishable Phase 2 transpiler theorem result if
    the empirical Gray-code factor confirms. **Target .81 or .82**
    depending on planner load.

## Operational watchdog scripts (newly shipped 2026-04-28)

Run these between conductor-supervisor landing:

  - `bash scripts/conductor_commit_watchdog.sh` — periodic check for
    stuck commits. With `AUTO_COMMIT=1`, attempts last-resort
    `git commit --no-verify` after $STALE_MIN minutes (default 60).
    Schedule via cron / systemd-timer.

  - `python3 scripts/audit_metric_provenance.py` — walk
    `results/experiment_*.json`, list deliverables by metrics
    provenance. With `--flag-buggy func:version`, surfaces deliverables
    using a known-bad implementation for retrospective re-evaluation.

## NEXT-MILESTONE PRIORITIES (.77 planner — historical, see MANDATORY above)

The 2026-04-27 24-hour session demonstrated that the conductor's
operator-attention burden is unsustainable — the operator had to
manually:
  - reap orphan process trees (~1 every hour)
  - SIGTERM runaway Sonnets that spawned duplicate experiment
    invocations (twice in 4 hours)
  - recover from broken `logs/conductor.log` write handles (~4
    occurrences in the session)
  - translate a schema-mismatched planner output (.74 would have
    gone stillborn exactly like .69 without intervention)
  - manually commit ~3 hours of accumulated conductor work that the
    conductor's own commit pipeline failed to push (twice — 35-file
    and 14-file commits)

Two proposals exist that scope durable fixes for these patterns:

  - **`openspec/change-proposals/conductor-supervisor.md`** (4 exps)
    — external observer with heartbeat watchdog, claimed-vs-actual
    state reconciliation, bounded auto-recovery whitelist (orphan
    reap, conductor restart, log-handle reset), conductor-side
    SIGUSR1 log-reopen handler. Catches every "conductor running but
    something's wrong" failure mode that requires manual operator
    attention today.

  - **`openspec/change-proposals/roadmap-schema-validation.md`**
    (3 exps) — Pydantic ResearchTask + Roadmap models validated at
    planner output (re-prompt on failure) and at activation (refuse
    to overwrite the active roadmap with malformed YAML). Prevents
    the once-per-month schema-drift stillborn-milestone pattern.

  - Bonus: **`openspec/change-proposals/conductor-otel-tracing.md`**
    (5 exps) — depends on the supervisor; lower priority but the
    natural next step. Puts every conductor iteration + subagent
    spawn into Victoria Trace so the seven incident shapes from this
    session each become single-trace queries.

The `flock` single-run guard from `conductor-process-isolation.md`
Exp B was direct-shipped on 2026-04-27 (commit 1b254b87) because of
the operational urgency. The supervisor + schema-validation work is
the natural next layer; the .77 planner should treat them as
candidate top picks.

## PHASE 2 PRIORITY (.78+ planner)

  - **`openspec/change-proposals/continuous-to-ising-transpiler.md`**
    (6 exps) — Phase 1 → Phase 2 bridge. Takes a trained verifier
    `state_dict` + a `HardwareSpec` and emits an `IsingSpec(J, h, ψ)`
    deployable to KV260, ECP5/Nexus, future XTR-0, or future photonic
    SLM. Origin: 2026-04-27 Deep Think exchange (4 rounds) producing
    the Continuous ε-Ising-Rank Theorem + Split-Verifier + Native
    Thermodynamic Distillation (PT-PCD with Gray-code encoding). The
    KV260 board has been on-hand since 2026-04-20.

## EXP 980 RE-SCOPING (.77 or .78 planner)

  - Exp 980 in .76 is currently scoped as "repair 11 monotonicity and
    boundary violations in KAEMEnergy." Under the **SOS-Integrated
    KAN** insight (Deep Think 2026-04-27), this framing is wrong.
    Standard monotonic-spline parameterizations are sufficient but
    not necessary, restricting expressivity. The fix is to push the
    constraint into derivative-space and analytically integrate:
    parameterize ψ'(x) as a Sum of Squares of B-splines (V ∈ ℝ^{N×M}
    unconstrained, M ≥ 2 for Burer-Monteiro stability), then integrate
    to ψ(x) = c² + Σ_{i,j} (V V^T)_{i,j} Φ_{i,j}(x). Monotonicity and
    non-negativity become **type-level invariants** of the AST
    `Add(Square(c), Integral(SumOfSquares(Splines)))`, not numerical
    properties to verify. MILP verification reduces to type-checking;
    the post-hoc repair subsystem is eliminated. Drop-in compatible
    at p=1 (hat functions → C¹ piecewise cubic splines, same
    computational profile as standard KANs). See
    `memory/project_sos_integrated_kan.md` for full detail.

## Original known-issues

| # | Issue | Severity | Workaround |
|---|-------|----------|------------|
| 1 | PyO3 0.24 doesn't support Python 3.14 natively | Low | Set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` |
| ~~2~~ | ~~Gibbs/Boltzmann grad_energy uses numerical finite differences~~ | ~~Resolved~~ | Analytical backprop implemented |
| ~~3~~ | ~~Python test suite not yet written~~ | ~~Resolved~~ | 48 tests, 100% coverage |
| 4 | Ackley and GaussianMixture benchmarks use numerical gradients | Low | Analytical gradients are complex; numerical is acceptable for benchmarks |
| 5 | RETRO-028: llama.cpp#21516 tokenizer bug causes Gemma4-E4B-it to emit only `<unused8>` tokens (token_id=14), producing 0.0% accuracy (false negative in Exp 439) | High — Fix implemented | Use GemmaTransformersLoader (python/carnot/pipeline/gemma_loader.py) instead of any llama.cpp backend. GPU verification pending (Exp 450). |
| ~~7~~ | ~~RETRO-029: Exp 444 (CarnotThinkProbe) timed out at 20 min with zero results — no partial data, no checkpoint.~~ | ~~Resolved 2026-04-18~~ | ThinkProbeV2 (python/carnot/pipeline/think_probe_v2.py) — 60-min budget, partial_verdict mode, incremental checkpoint every 10 questions. Exp 455 implements and validates the fix. |
| ~~6~~ | ~~RETRO-030: Exp 446 (Energy Matching) exited with status 0 but produced no result JSON. Root cause: exception mid-write left no file; watchdog missed it (only checked exit code).~~ | ~~Resolved 2026-04-18~~ | AtomicResultWriter (python/carnot/pipeline/atomic_writer.py) — write-to-tmp + os.rename prevents partial writes. Exp 452 re-runs Exp 446 logic with atomic write + verify_exists() assertion. |

## RETRO-072 update (Exp 701, 20260422)
Vivado not installed; yosys not found.  Synthesis blocked.  Install one of:
  - AMD Vivado 2024.2 (free WebPACK from xilinx.com)
  - yosys (`sudo pacman -S yosys` on CachyOS)
RETRO-073 opened for milestone .54.

## RETRO-CRITICAL: JEPA v17 RankNet Gate Failed (Exp 704/705, 20260422)
JEPA v17 OOD AUC = 0.4819, still below random chance (threshold = 0.75).
RankNet pairwise loss partially addresses the anti-correlation root cause but pairwise
hedging persists when pairs are too similar — each pair is optimised independently.
**v18 approach:** LambdaRank listwise loss — optimise NDCG over ALL steps per question
simultaneously, directly matching the AUC evaluation metric.
**Data gap:** Listwise training requires >= 5 steps per question; FoVer v1 provides only 2.
Unblocked by: Exp 712 FoVer v2 PDDL (5+ steps per question via PDDL plan enumeration).
JEPA v16 cascade block remains in effect until v18 achieves OOD AUC >= 0.75.

## Closed Issues

### FR-11 CLOSED — Status: OPERATIONAL (Exp 738, 2026-04-22)

~~FR-11 (Autonomous Self-Learning Loop) — blocked for 15+ milestones on AUC gate.~~

CLOSED 2026-04-22, Exp 738. FR-11 is now OPERATIONAL. Evidence:
fr11_relay_operational=True (Exp 734, relay_events_acked=100, latency_p99_ms < 200),
fr11_tier2_relay_functional=True (Exp 738, templates_replayed_in_s2=1,
cross-session persist confirmed), probe 5-fold AUC=0.993 (Exp 732). Milestone
2026.04.56 retro marked FR-11 "ELIGIBLE FOR FORMAL CLOSURE". Formal closure
certificate: results/fr11_closure_certificate.json.

---

## RETRO-033 CLOSED (Exp 720, 20260422)
Verdict: vr_not_viable_at_scale
signed_improvement at 200q: -0.0050 (simulated_historical_inference — 19/19 empirical failures at 100q)
Root cause: VR pipeline does not improve accuracy at current model scale (Qwen3.5-0.8B).
Resolution: VR removed from active roadmap. Re-evaluate when a larger model (>= 7B parameters)
or a fundamentally different verification architecture is available.
Spec: REQ-VER-030-6, SCENARIO-VER-037


## RETRO-MANIFEST-FULL-SCOPE: Human Intervention Required (Milestone .69)

ExclusionManifestEnforcer pre_launch_check() cannot be wired to the conductor loop
without modifying scripts/research_conductor.py, which is forbidden per CLAUDE.md
in the Exp 892 task specification.

11 consecutive milestones open. Action required: either
  (a) grant human permission to modify scripts/research_conductor.py for this one change, or
  (b) accept that manifest enforcement operates at the planning layer only
      (CLAUDE.md rule is the primary enforcement; code enforcement is secondary).

Documented by Exp 892 pre-flight v18 on 2026-04-26T02:52:17Z.
enforcement_wired: false

## IPFS not installed — VJEPA v2 weights have no IPFS mirror

Added: 2026-04-26 (Exp 902)

CLAUDE.md rule 3 requires all published weights to have an IPFS mirror.
The `ipfs` command was not found at publish time.  Install IPFS and
re-run Exp 902 to establish the mirror.

Install: `apt install ipfs` or use the ipfs.io installer:
https://docs.ipfs.tech/install/

Then run: `ipfs add -r /tmp/carnot-vjepa-v2-card/ && ipfs pin add <CID>`


## RETRO-MANIFEST-FULL-SCOPE: CRITICAL — Human Intervention Required (Milestone .70)

ExclusionManifestEnforcer pre_launch_check() is NOT wired to the conductor loop.
This is the 12th consecutive milestone where the manifest has not been enforced
mechanically. The rule in CLAUDE.md (planning-layer discipline) is the ONLY active
enforcement. A conductor-level hook is blocked by the 'do NOT modify
scripts/research_conductor.py' constraint. Action required: grant human permission
to modify scripts/research_conductor.py for this single wiring change.
enforcement_wired: false
escalation_milestone: "2026.04.70"


## RETRO-LAGRANGE-ENTROPY-DEGENERATE: CLOSED (Exp 918, 2026-04-26)

Root cause: Single-constraint corpus had entropy = 0 by construction (p = 1.0).
Fix: 8-constraint heterogeneous corpus. Exp 918 result: signed_entropy_improvement=0.018.
Algorithm confirmed working. RETRO closed.

## GATE-CHECK DISCIPLINE: prior_failures Required for All Domain-Overlapping Tasks

Exps 917, 919, 920, 921, 922, 925, 926, 927 all blocked in .71 by missing prior_failures.

Rule: Any YAML task touching a domain with ANY prior experiment history MUST include
prior_failures entries with: experiment_id, verdict, addressed_by, retire_if_same_verdict.
The conductor gate-checker scans the FULL research history. If prior_failures is absent
and matching prior experiments exist → immediate block.

This is a planner-layer discipline failure, not a code bug. The planner that generated
research-roadmap-v71.yaml did not populate prior_failures for any of the 8 tasks with
prior failure history. Fix: consult research-complete.yaml before generating any task YAML.

## RETRO-MANIFEST-FULL-SCOPE: CRITICAL — Human Intervention Required (Milestone .71)

14 consecutive milestones without mechanical manifest enforcement.
enforcement_wired: false
escalation_milestone: "2026.04.71"
Action required: grant human permission to modify scripts/research_conductor.py.

## RETRO-RERUN-DISCIPLINE-GATE-CASCADE (opened .71)

9 of 12 experiments in .71 were blocked by the conductor pre-gate due to missing
prior_failures fields in the roadmap YAML. This is a cascade of the same root cause.
Status: HUMAN_REQUIRED — planner must be trained on the rule before .72 executes.

## RETRO-HEURISTIC-RPRM-FLAT-SIGNAL (opened .71)

Exp 924 R-PRM Tier 2.9 heuristic mode: AUC delta = 0.0. Heuristic inference cannot
produce step-level signal. Real model inference (Qwen3.5-0.8B minimum) required.
Status: TARGETED — .72 must use live model, not heuristics.

## RETRO-DRIFT-ENSEMBLE-UNIFORM-WEIGHTS (opened .71)

Exp 923 DriftProbe ensemble (3 layers, uniform weights): OOD AUC 0.5625 vs 0.565 baseline.
Uniform weighting HURTS — two zero-coefficient probes dilute one informative probe.
Status: TARGETED — .72 must use learned weights (logistic regression on validation set).

## RETRO-HF-SOPS-CREDENTIAL-INJECTION (opened .71)

Exp 922 HF publish blocked by SOPS credential injection unresolved.
Status: HUMAN_REQUIRED — resolve SOPS credential injection before scheduling HF publish.

### IPFS Mirror CLOSED (Exp 934, 2026-04-26)
VJEPA v2 IPFS CID: `QmTkGjpN5fYNnC3g8Gx8sPWHZJKkw8oGVDKwWT6sZbVaGN`
Mirror registry: results/ipfs_mirrors.json

## RETRO-MATH-REPAIR-MODEL-CEILING (opened .72, Exp 930)

Exp 930 iterative self-repair on GSM8K: gemma-4-E4B-it baseline=12%, repair=12%,
signed_improvement=0.0. Model capability ceiling — E4B is too small for GSM8K
math reasoning. The repair algorithm is structurally correct; the model is wrong.

Resolution path: Exp 942 in .73 must use Gemma4-31B or Qwen3.6-35B-A3B (SOTA tier).
SOTA GGUF already downloaded — gemma-4-26B-A4B-it-UD-Q4_K_M.gguf confirmed in HF cache.
Status: TARGETED (Exp 942)

## RETRO-SC-ENERGY-GATE-DISCIPLINE (opened .72, Exp 939)

Exp 939 SC-Energy Set Consistency Networks blocked by conductor pre-gate: task YAML
lacked prior_failures entries for 7 prior SC-energy / contrastive-energy experiments.
Identical planning error to Exp 917 in milestone .71 — planner did not consult
research-complete.yaml before writing the task YAML.

## SC-ENERGY PRIOR EXPERIMENTS (for Exp 944 prior_failures reference)

Exp 944 MUST include all 8 entries below in its prior_failures field:

| Exp | Verdict | Domain |
|-----|---------|--------|
| 506 | semantic_energy_no_improvement | Semantic Energy Tier 0d |
| 509 | energy_magnitude_wins | PPSEBM Energy Magnitude Replay (adjacent) |
| 533 | no_violation_reduction | COLD Decoding Energy Guidance (adjacent) |
| 711 | tier_29_below_threshold | SC-Energy SetConsistencyVerifier Tier 2.9 |
| 725 | sc_energy_v2_below_threshold | SC-Energy v2 FoVer v2 Dual Labels |
| 772 | semantic_energy_below_baseline | SemanticEnergyProbe Tier 0g |
| 787 | energy_prefilter_efficient | S* Energy Pre-Ranking (adjacent) |
| 939 | blocked_gate_check_failed | SC-Energy Set Consistency Networks |

The "addressed_by" field for each must explain what is substantively different in
Exp 944 (new architecture, new corpus, new technique — not relabeling).
Status: HUMAN_REQUIRED at planner layer — conductor will block again if omitted.
