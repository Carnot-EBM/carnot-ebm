# Research Roadmap v453 — Milestone 2026.06.453

**Planned by:** outer-loop Claude Opus 4.8 planner, 2026-06-28.
**Milestone:** 2026.06.453 (CalVer; June, seq 453).
**Sprint context:** ARC-AGI-3 Submission Sprint Forcing Function is ACTIVE through
2026-06-30 (CLAUDE.md). Today is 2026-06-28 — this is one of the FINAL pre-deadline
milestones. Majority-ARC, ≥1 level-up attempt, self-play every milestone, 2 reserved
infra, 1 hardware-continuity, 1 SOTA-ingestion all still apply.

---

## 1. What .452 proved (the wall is now a TRUSTED honest negative)

`.452`'s capstone (exp4912) closed the multi-milestone world-model fork as an
honest negative:
`complete_capstone_v452_escalate_wall_survives_four_representations_plus_env_grounding`.

| Lane | Result |
|---|---|
| **A1 (exp4903)** env-grounded, change-LOCATION-pruned, REAL-ENV-VALUE-grounded interleaved search | `WALL_DEEPER_THAN_VALUE_PREDICTION`. first-win delta **−0.04** (CI95 [−0.04, −0.04]); coverage migration **0**. **B1-TRUSTED** (a1_trustworthy: real-env value, planner-blind, positive control tu93 non-degenerate rank-4 score-3853, numbers match). |
| **A1b (exp4904)** latent-action interface (AdaWorld 2503.18938) — the 4th representation | `VALUE_GAP_REPRESENTATION_INVARIANT_4_CLASSES`. value-accuracy delta **−0.103** (CI95 [−0.231, 0.025]); ran genuinely live. |
| A2 (exp4905) level-up m0r0 | no bank (duplicate depth); `reproducible_total_levels` stays **68**. |
| A3 (exp4906) self-play vc33 | checkpoint refreshed (FR-11 relay). |
| A4 (exp4907) held-out first-win readiness | ~0.05 live, CI lower 0 — but **flagged_adversarial** (3rd milestone running). |
| B2 (exp4909) submission package | READY, **15.146 GB < 16 GB**, operator-only, submits=false. |
| C (exp4910) KV260 | SSH-reachable, graduated terminal. |
| D (exp4911) SOTA ingestion | mapped the .453 frontier + the post-sprint pivot (all HTTP-200). |

**The settled, B1-trusted finding (the closure-grade negative).** A1's per-game data
is the decisive evidence: across all 9 held-out games the env-grounded search took
**24 real-env value reads per game** (reading the TRUE change-VALUE for free, never
predicting it), with a **working** change-LOCATION prior (non-degenerate on cn04/su15/wa30;
positive control tu93 ranked the truly-changing action #4 of many) — and **still
migrated 0 games** NEVER_ENUMERATED→COVERED (8 states expanded, best path 6–8, 0
first-wins). So the wall is NOT change-VALUE prediction (sidestepped) and NOT
change-LOCATION (works, 0.727). **The wall is assembling the multi-step winning
prefix** — the search cannot find a length-6–8 winning trajectory under a bounded
action/state budget even when value is free.

**The state of the deliverable (locked).** The Kaggle submission package is READY
(15.146 GB iGPU-frozen Qwen3.5-9B-MTP stack). `reproducible_total_levels = 68`. FoVer
paper `paper_ready = true` (G1∧G2∧G3∧G4 all PASS per `publication_gate.py`, north-star §1/§2).

---

## 2. The .453 strategic frame — FINAL ARC closure DIAGNOSTIC (not representation #5), pivot teed up

The capstone is explicit: **"Do not queue representation #5."** Every offline
value-PREDICTION route is closed, and env-grounding (reading value for free) does not
help either. Testing a 5th representation for change-VALUE would be pure churn
(north-star §1).

But there IS one question the wall finding leaves open and that the sprint's remaining
days should answer for **closure** rather than churn:

> **WHY can't the search assemble the winning prefix?** Is the missing discriminator a
> state variable the agent COULD observe from the ARC interface but its current
> abstraction drops (a *fixable* representation gap), or is it a HIDDEN variable the
> interface cannot expose (*representation-invariant by construction*)?

`.453` A1 (HEADLINE) answers it with the **causal-state-abstraction wall diagnostic**
(D's flagged priority-1, arXiv:2401.12497 "Building Minimal and Reusable Causal State
Abstractions for RL" — Causal Bisimulation Modeling). This is a DIAGNOSTIC, not a
representation: over A1's exact failed transitions it derives the **minimal set of
state variables** causally necessary to predict changed-cell value + progress-to-goal,
then **classifies** each as:

- **OBSERVABLE** from the ARC frame/env interface → the agent's current abstraction
  drops a variable it could retain → `WALL_IS_OBSERVABLE_VARIABLE_GAP` → a concrete,
  named, testable lever survives (post-sprint can try retaining it).
- **HIDDEN** / unobservable (latent counter, off-screen state, interaction-dependent)
  → no representation over observable inputs can recover it → `WALL_IS_HIDDEN_STATE` →
  the representation-invariance is **mechanistically explained** → closure.

### Why this is genuinely a closure result, not another swing

- It does **not** propose a new value-PREDICTION representation. It produces a
  CLASSIFICATION REPORT (per-variable observable/hidden). A claimed "observable"
  variable must be demonstrably readable from frame/env state, or it is HIDDEN.
- **Positive control is load-bearing:** on a game we DID solve (tu93 / a solved L2),
  the diagnostic MUST find the minimal causal abstraction is observable (we solved it
  over observable inputs). If it does not, the diagnostic is broken, not the wall.
- It is **publishable closure** for the FoVer paper's ARC section: it converts "the
  wall survives 4 representations + env-grounding" into a mechanistic "the
  discriminating variable is observable/hidden, here is the per-game evidence."
- It is **forbidden** from becoming a decision-need target table in disguise (exp4911
  `fails_when`): no change-VALUE-predicting table, no solve claim, no static-ranking lift.

### Deadline lanes + post-sprint handoff (the pivot is teed up, not started)

Because the sprint retires 2026-06-30, `.453` also (a) keeps the two deadline lanes —
A4 (the held-out go/no-go number, this time produced CLEAN by fixing the recurring
flagged_adversarial) and B2 (final package harden + operator submission checklist) —
and (b) has D **minimally scaffold** the post-sprint pivot (distributional energy
verifier, arXiv:2605.18871) so the loop executes it the instant the sprint retires.
D scaffolds, it does not start the pivot — majority-ARC still governs through 6/30.

---

## 3. The three biggest gaps vs the PRD vision (and how .453 attacks them)

1. **Live first-win generalization (north-star ACCURACY).** The agent solves ~1/25
   unseen games first-contact and every value-prediction route is invariant-bounded.
   GAP: we don't yet know if the wall is a fixable observable-variable gap or a
   fundamental hidden-state limit. `.453` A1 diagnoses it (closure), and A4 reports the
   honest readiness number for the 6/30 go/no-go.
2. **Verifier value where no cheap oracle exists (north-star §1/§5, the post-6/30
   future).** The FoVer headline is execution-grounded; the OPEN claim is an
   oracle-distinct verifier on non-saturated domains. GAP: we have a map but no built
   harness. `.453` D scaffolds the distributional-energy-verifier-vs-self-consistency
   offline harness so the pivot is executable on 6/30.
3. **Continuous self-learning (PRD FR-11 / research-program.md Tier 3).** GAP: the
   learned verifier must improve across runs. `.453` A3 (self-play every milestone)
   trains + checkpoints the learned verifier on fresh self-play traces — the standing
   FR-11 relay.

---

## 4. Phases & tasks (11 tasks, conductor execution order)

| # | id | Phase | Track | What |
|---|---|---|---|---|
| 1 | exp4913 | PHASE 0 | transition | archive .452 → activate .453; record close-state; resolve poison pre-test |
| 2 | exp4914 | **A1 (HEADLINE)** | arc-north-star | causal-state-abstraction wall diagnostic (2401.12497) — observable-vs-hidden variable classification; the FINAL ARC closure |
| 3 | exp4915 | A2 | arc-north-star | Level-Up Attempt Guarantee — bank ≥1 new level on a rotated target (sp80/su15/cn04 L2→L3) |
| 4 | exp4916 | A3 | arc-north-star | self-play EVERY milestone (FR-11) — rotate target; train + checkpoint the learned verifier |
| 5 | exp4917 | A4 (DEADLINE) | arc-north-star | CLEAN held-out first-win readiness — FIX the 3-milestone recurring flagged_adversarial; the 6/30 go/no-go |
| 6 | exp4918 | B1 (infra 1) | infra | adversarial audit — A1 diagnostic HONEST (real failed transitions, observable claims verified readable, not a value-table, oracle-distinct) |
| 7 | exp4919 | B2 (infra 2, DEADLINE) | infra | submission-package FINAL harden + operator checklist; frozen iGPU stack <16 GB |
| 8 | exp4920 | B3 (infra 3, OPERATIONAL) | infra | retro detector-wire results/-mtime fallback module + write-time duration_s/inference_substrate/compute_bound stamping helper + audit |
| 9 | exp4921 | C | hardware | KV260 SSH-only continuity (always writes a deliverable) |
| 10 | exp4922 | D | sota-ingestion | minimally SCAFFOLD the post-6/30 distributional-energy-verifier pivot (2605.18871) — offline FoVer→MuSR harness + dry-run |
| 11 | exp4923 | E (CAPSTONE) | capstone | aggregate the scorecard; trust A1 only if B1 clean; state the post-6/30 pivot handoff |

> Note: this milestone carries THREE infra arms (B1 diagnostic-audit, B2 deadline
> package, B3 operational fix). The sprint reserves "2 infra"; B2 is a deadline-ARC
> lane (the submission package) and B3 is the operator-flagged retro action, so the
> milestone runs +1 infra this milestone deliberately (the ~79-milestone recurring
> detector false-zero + the duration_s=None gap are now top-3 retro actions). ARC
> majority is preserved: A1/A2/A3/A4/B2 are all ARC/deadline of the non-reserved slots.

### Dependency graph

```
exp4913 (transition)
   └── exp4914 A1 (causal-abstraction diagnostic; HEADLINE)
          └── exp4918 B1 (audits A1)
   exp4915 A2  ─┐
   exp4916 A3  ─┤  (independent reserved ARC lanes)
   exp4917 A4  ─┘
   exp4919 B2  (deadline; independent)
   exp4920 B3  (operational; independent)
   exp4921 C   (hardware; independent)
   exp4922 D   (scaffolds post-sprint pivot; reads A1/A1b verdicts)
   exp4923 E   (capstone; reads all, skips flagged_adversarial)
```

No `gated_on` task this milestone: the diagnostic is terminal closure (it does not
gate a representation swing — we are not doing representation #5), and the pivot
scaffold (D) runs regardless of A1's observable/hidden verdict (the operator decides
when to execute the pivot post-6/30).

---

## 5. Hardware requirements

| Lane | Hardware | Note |
|---|---|---|
| A1 / A3 / A4 (offline induction + diagnostic) | conductor's **GPU-0 CUDA** llama-server (`CARNOT_ARC_GENERATOR_CUDA_GPU=0`) OR iGPU HIP | Per the 2026-06-27 operator GPU-allocation directive: offline induction is NOT iGPU-pinned; accept GPU-0 CUDA (both 3090s idle), do NOT block on `CUDA_VISIBLE_DEVICES`. |
| B2 (submission stack) | **iGPU** (FROZEN) | The LIVE submission generator stays Qwen3.5-9B-MTP on the iGPU (Kaggle parity). Do NOT move it to the 3090s. The 35B/31B SOTA GGUFs do NOT apply to the frozen Kaggle-parity submission generator. |
| C (KV260) | SSH only | `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; NEVER a host `/dev/mmcblk*` precondition. Graduated terminal; stays in the continuity rotation. |
| A2 (level-up) / B3 / D | CPU offline arcade (+GPU-0/iGPU if induction needed) | reproduction-gated deterministic offline sim; B3/D are aggregation/scaffold (no live inference floor). |

---

## 6. Disciplines honored (planner self-audit)

- **ARC-AGI-3 Submission Sprint Forcing Function** — majority-ARC (A1/A2/A3/A4 + the B2
  deadline lane), ≥1 level-up (A2), self-play every milestone (A3), reserved infra
  (B1/B2/B3), 1 hardware (C), 1 SOTA-ingestion (D).
- **ARC Live-Path Reachability** — A1 uses the live e3 induction interface
  (`arc_executable_world_model.load_engine`) as a DIAGNOSTIC over real failed
  transitions; `arc_orphan_solver_lint` must pass; NOT a parallel solver. Every ARC
  task declares `solve_provenance` (A1=development_proxy diagnostic; A2/A3=live_agent_self_discovery).
- **Circularity / Oracle-Distinctness** — A1 sets `verifier_is_oracle: false` (the
  causal-abstraction classifier is oracle-distinct from the env's level-up check; it is
  a DIAGNOSTIC, not a moat claim) → passes `check_circular_moat_overclaim`.
- **Failed-Experiment Rerun + Exclusion-Manifest** — A1 carries `prior_failures`
  (4 fields) naming exp4903/exp4904 and what is different (a classification diagnostic,
  not a 5th value-prediction representation); every reserved-slot continuation carries
  `operator_override`. Do NOT re-propose energy / TTA-on-code-engine /
  stronger-local-code-inducers / coverage / exploration / selection / perception-from-grid /
  decision-need-targets / action-prefix-latents / representation #5 (all nulled/retired).
- **Pre-Launch Preconditions** — every compute-bound task opens with a PRECONDITIONS
  step (arcade / generator / upstream artifact); a missing resource emits
  `blocked_<resource>` and exits, never fabricates.
- **Inference-Substrate + Principle-Annotated Fields** — every task declares
  `inference_substrate`; every REQUIRED ARTIFACT FIELD carries a `principle:`.
- **Verdict Terminal-Prefix** — every `honest_verdict` starts `complete_` / `success_`.
- **Public Documentation Discipline** — B3 delivers a standalone module + audit + wiring
  proposal; it does NOT modify `scripts/research_conductor.py` (operator wires the conductor).
- **Agent routing** — all experiments `agent_type: codex` / `model: gpt-5.5` (sprint
  default); planner/retro stay on Claude Opus via conductor env (operator choice).

---

## 7. Post-sprint handoff (operator-facing) — the fork the operator may redirect

The sprint retires 2026-06-30. The honest position after .452 is settled: **the live
first-win wall survives energy + goal-quality + FOUR world-model representations +
env-grounded real-env-value search.** The deliverable is **the current ~0.05 first-win
agent (package ready, operator-only to submit) + the publishable FoVer
verifier-ensemble paper (paper_ready=true, north-star §1/§2)**.

`.453` makes ONE judgment call the operator can override: it spends the ARC headline on
the **causal-abstraction closure diagnostic** rather than starting the verifier-moat
pivot, because the sprint forcing function (standing operator directive) mandates
majority-ARC through 6/30, and a closure diagnostic is the non-churn ARC headline. The
exp4911 `planner_instruction` framed this as a conditional ("use the diagnostic only if
the operator wants a final ARC closure check"); the still-active sprint resolves it
toward closure. **If the operator prefers to pivot early** (the wall is already
B1-trusted-closed), the D-scaffolded distributional-energy-verifier harness (exp4922) is
the ready entry point — promote it to the headline in `.454` and drop the remaining ARC
diagnostic work. Either way, D (exp4922) leaves the post-sprint verifier-moat track
(north-star §5, oracle-distinct, non-saturated domains) built and ready, so the handoff
is clean when the ARC sprint closes.
