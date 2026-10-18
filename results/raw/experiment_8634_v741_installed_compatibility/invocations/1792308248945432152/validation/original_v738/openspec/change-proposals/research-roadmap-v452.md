# Research Roadmap v452 — Milestone 2026.06.452

**Planned by:** outer-loop Claude Opus 4.8 planner, 2026-06-28.
**Milestone:** 2026.06.452 (CalVer; June, seq 452).
**Sprint context:** ARC-AGI-3 Submission Sprint Forcing Function is ACTIVE through
2026-06-30 (CLAUDE.md). Today is 2026-06-28 — this is one of the FINAL pre-deadline
milestones. Majority-ARC, ≥1 level-up attempt, self-play every milestone, 2 reserved
infra, 1 hardware-continuity, 1 SOTA-ingestion all still apply.

---

## 1. What .451 proved (and why the loop escalated)

`.451`'s capstone (exp4901) **escalated to the operator** with the verdict
`complete_capstone_v451_representation_invariant_escalate_operator`. The arc that
led there:

| Milestone | Lever | Verdict |
|---|---|---|
| `.444`-ish | Oracle-distinct STRUCTURAL ENERGY (S0→S3) | **CONCLUDED negative** — real offline cross-game discriminator, **NO live ARC value** (selection OR generation). `docs/research-notes/oracle-distinct-structural-energy-program-2026-06-26.md`. Do NOT re-propose energy. |
| `.437` | L2 goal-QUALITY (perception-grounded structural alignment) | **SOLVED** — detector fixed, goal predicate satisfiable. Wall shifted to the dynamics ENGINE. |
| `.449` | Test-time dynamics adaptation on the executable-code engine | `INDUCER_CEILING_HARD` — engine learns change-LOCATION (cell_recall 0.727) but NOT change-VALUE; TTA delta −0.0087. |
| `.450`-`.451` A1 | Decision-need target representation (arXiv:2606.25421) | `VALUE_GAP_REPRESENTATION_INVARIANT` — value-accuracy delta −0.10, migration 0. |
| `.451` A1b | Action-prefix latent representation (arXiv:2606.26217) | `VALUE_GAP_REPRESENTATION_INVARIANT_HARD` — value-accuracy delta 0.0. |

**The settled finding (the asymmetry).** Across THREE world-model representations
(executable-code, decision-need-targets, action-prefix-latents), the induced world
model predicts **change-LOCATION** (where cells change — cell_recall 0.727) but
**NOT change-VALUE** (what they become). This is *representation-invariant*. The
agent knows WHERE a step changes the grid but not WHAT it changes it to, so
`plan_in_model` cannot assemble a winning multi-step prefix → the generic
first-win wall (≈0.04 = 1/25 games; A4 readiness 0.05).

**The state of the deliverable.** The Kaggle submission package is READY (vram
~15.146 GB < 16 GB, frozen Qwen3.5-9B-MTP iGPU stack). `reproducible_total_levels
= 68`. FoVer paper `paper_ready = true` (the publishable result, north-star §1).

---

## 2. The .452 strategic reframe — STOP predicting change-VALUE; READ it from the env

The D-ingestion (exp4900) dutifully proposed representation **#4/#5/#6**
(latent-action AdaWorld, reverse-counterfactual, verification-calibrated
abstraction) — "find a better representation." But three representations already
nulled and the capstone escalated. Per north-star §1 ("a milestone that produces a
new version of an existing artifact without moving the headline is churn"), blindly
testing a 4th representation for change-VALUE is **high churn risk**.

The reframe uses the finding's own asymmetry. ARC-AGI-3 is **interactive** —
act→observe→act. When the agent actually takes an action, the **real env tells it
the true change-VALUE for free**. So the open question the loop never asked is not
"which representation predicts change-VALUE?" (answered: none) but:

> **Do we even need to PREDICT change-VALUE? Or can we exploit the one signal that
> WORKS (change-LOCATION, 0.727) as a search-PRUNER and read change-VALUE from the
> real environment at every step?**

### Why this is genuinely new (not already `plan_in_model`)

`plan_in_model` (arc_executable_world_model.py:1495) **already** executes in the
real env and halts on divergence — but it **plans the full prefix IN THE MODEL
FIRST** (value-prediction-dependent), THEN validates. When the model mispredicts
change-VALUE, the *plan* is already wrong before the env sees it; execution halts
immediately on divergence → zero progress. The broken change-VALUE prediction
poisons the plan upstream of the env.

`.452` A1 **demotes the model from state-predictor to action-prior** and grounds
value in the env at EVERY step (interleaved act-and-observe, not plan-then-validate):

```
  PER STATE s (real, observed):
    1. change-LOCATION model ranks legal actions by predicted-change saliency
       (the 0.727 signal — "this action changes something meaningful here").
       The model NEVER predicts the resulting VALUES.
    2. EXECUTE the top-k ranked actions in the REAL env (reset-replay in the
       offline dev twin) → read the TRUE next state s' (real change-VALUE, no
       prediction).
    3. Score progress toward the (already-solved-quality, .437) goal predicate
       via the learned verifier (oracle-distinct).
    4. Best-first expand the most promising real s'. Repeat.
```

This simultaneously fixes the two documented failures:
- **`.432` "proposal distribution misses the winning prefix"** — the expansion
  prior is now the one signal that provably works (change-LOCATION), not novelty
  (nulled .432-.436).
- **representation-invariant change-VALUE** — sidestepped entirely; value is read
  from the env, never predicted.

The model is the **action-PRUNER**; the env is the **value ORACLE**. This is the
north-star §5 verifier-as-action-pruner role made concrete, and it attacks BOTH
north-star metrics: **accuracy** (does location-pruned env-grounded search assemble
a first-win the plan-then-validate path could not?) and **efficiency**
(actions-to-first-win — the env-grounding cost the change-LOCATION prior must keep
bounded).

### The hedge — D's SOTA #1 as the LAST representation swing (gated)

A1b runs the **latent-action interface** (AdaWorld, arXiv:2503.18938 — D's
flagged priority-1) as the FINAL representation swing, **gated on A1 not unlocking
first-wins**. If the env-grounded reframe (A1) wins, we scale it in `.453` and skip
the representation swing. If A1 doesn't lift first-win AND A1b (a distinct
latent-ACTION interface, not another prefix-delta table) also nulls, the
change-VALUE gap is confirmed **representation-invariant across FOUR classes** and
the deliverable locks to the current ~0.05 agent + the publishable FoVer paper.

### Deadline lock + post-sprint handoff

Because the sprint retires 2026-06-30, `.452` also (a) LOCKS the two deadline lanes
— A4 (genuinely-live CLEAN held-out first-win readiness = the 6/30 go/no-go) and B2
(final submission-package harden + operator checklist) — and (b) has D map the
**post-sprint pivot** back to the verifier-moat / oracle-distinct / FoVer-paper
track (north-star §1/§5) so the operator has a clean handoff when the sprint ends.

---

## 3. The three biggest gaps vs the PRD vision (and how .452 attacks them)

1. **Live first-win generalization (the north-star ACCURACY axis).** The agent
   solves ~1/25 unseen games first-contact. GAP: every offline value-prediction
   route is representation-invariant-bounded. `.452` A1 attacks it with an
   env-grounded search that doesn't need value prediction; A4 measures the result
   as the deadline go/no-go.
2. **Efficiency at scale (the north-star EFFICIENCY axis).** Env-grounding costs
   real actions. GAP: an unbounded-action search doesn't scale on the interactive
   benchmark. `.452` A1 reports actions-to-first-win and forks on `SEARCH_BUDGET_BOUND`.
3. **Continuous self-learning (PRD FR-11 / research-program.md Tier 3).** GAP: the
   learned verifier must improve across runs. `.452` A3 (self-play every milestone)
   trains + checkpoints the learned verifier on fresh self-play traces — the
   standing FR-11 relay.

---

## 4. Phases & tasks (11 tasks, conductor execution order)

| # | id | Phase | Track | What |
|---|---|---|---|---|
| 1 | exp4902 | PHASE 0 | transition | archive .451 → activate .452; record close-state; resolve poison pre-test |
| 2 | exp4903 | **A1 (HEADLINE)** | arc-north-star | change-LOCATION-pruned, REAL-ENV-VALUE-GROUNDED interleaved search; both north-star axes; fork |
| 3 | exp4904 | A1b (gated) | arc-north-star | latent-action interface (AdaWorld 2503.18938) — the LAST representation swing; gated on A1 |
| 4 | exp4905 | A2 | arc-north-star | Level-Up Attempt Guarantee — bank ≥1 new level on a rotated target (m0r0/sp80/su15/cn04 L2→L3) |
| 5 | exp4906 | A3 | arc-north-star | self-play EVERY milestone (FR-11) — rotate target; train + checkpoint the learned verifier |
| 6 | exp4907 | A4 (DEADLINE) | arc-north-star | GENUINELY-LIVE + CLEAN held-out first-win readiness (the 6/30 go/no-go); deliverable-first |
| 7 | exp4908 | B1 (infra 1) | infra | adversarial audit — A1 env-value-grounding HONEST (real env, planner-blind, oracle-distinct, numbers match) + A1b genuinely-live |
| 8 | exp4909 | B2 (infra 2, DEADLINE) | infra | submission-package FINAL harden + operator checklist; frozen iGPU stack <16GB |
| 9 | exp4910 | C | hardware | KV260 SSH-only continuity (always writes a deliverable) |
| 10 | exp4911 | D | sota-ingestion | .453 frontier given A1's verdict + the POST-sprint verifier-moat pivot map |
| 11 | exp4912 | E (CAPSTONE) | capstone | aggregate the scorecard; trust A1 only if B1 clean; post-6/30 handoff |

### Dependency graph

```
exp4902 (transition)
   └── exp4903 A1 (env-grounded search; HEADLINE)
          ├── exp4904 A1b  [gated: A1 value_grounded_first_win_delta_median < 0.1]
          └── exp4908 B1   (audits A1 + A1b)
   exp4905 A2  ─┐
   exp4906 A3  ─┤  (independent reserved ARC lanes)
   exp4907 A4  ─┘
   exp4909 B2  (deadline; independent)
   exp4910 C   (hardware; independent)
   exp4911 D   (reads A1/A1b verdicts)
   exp4912 E   (capstone; reads all, skips flagged_adversarial)
```

A1b is the only gated task — it runs only when A1's bare
`value_grounded_first_win_delta_median < 0.1` (the env-grounded reframe did not
meaningfully lift first-win → take the last representation swing). If A1 lifts
first-win ≥ 0.1, A1b is correctly skipped and `.453` scales A1 instead.

---

## 5. Hardware requirements

| Lane | Hardware | Note |
|---|---|---|
| A1 / A1b / A4 (offline induction + search) | conductor's **GPU-0 CUDA** llama-server (`CARNOT_ARC_GENERATOR_CUDA_GPU=0`) OR iGPU HIP | Per the 2026-06-27 operator GPU-allocation directive: offline induction is NOT iGPU-pinned; accept GPU-0 CUDA (both 3090s idle), do NOT block on `CUDA_VISIBLE_DEVICES`. |
| B2 (submission stack) | **iGPU** (FROZEN) | The LIVE submission generator stays Qwen3.5-9B-MTP on the iGPU (Kaggle parity). Do NOT move it to the 3090s. |
| C (KV260) | SSH only | `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`; NEVER a host `/dev/mmcblk*` precondition. Graduated terminal; stays in the continuity rotation. |
| A2 / A3 | CPU offline arcade + GPU-0/iGPU if induction needed | reproduction-gated; deterministic offline sim. |

---

## 6. Disciplines honored (planner self-audit)

- **ARC-AGI-3 Submission Sprint Forcing Function** — majority-ARC (A1/A1b/A2/A3/A4 =
  all 5 non-reserved slots), ≥1 level-up (A2), self-play every milestone (A3), 2
  reserved infra (B1/B2), 1 hardware (C), 1 SOTA-ingestion (D).
- **ARC Live-Path Reachability** — A1/A1b improve the live e3 path
  (StepwiseExplorer / plan_in_model / load_engine); `arc_orphan_solver_lint` must
  pass; NOT parallel solvers. `solve_provenance` declared on every ARC task.
- **Circularity / Oracle-Distinctness** — A1/A1b/A4 set `verifier_is_oracle: false`
  (the change-LOCATION model + learned verifier are oracle-distinct from the env's
  level-up check); A1 is framed as a SEARCH-STRATEGY result, not a verifier-moat
  claim, so it passes `check_circular_moat_overclaim`.
- **Failed-Experiment Rerun + Exclusion-Manifest** — A1/A1b carry `prior_failures`
  (4 fields each) naming the .449/.450/.451 world-model failures and what is
  different; reserved-slot continuations carry `operator_override`. Do NOT
  re-propose energy / TTA-on-code-engine / stronger-local-code-inducers /
  coverage / exploration / selection / perception-from-grid / decision-need-targets
  / action-prefix-latents (all nulled/retired).
- **Pre-Launch Preconditions** — every compute-bound task opens with a
  PRECONDITIONS step (arcade / generator / upstream artifact); a missing resource
  emits `blocked_<resource>` and exits, never fabricates.
- **Inference-Substrate + Principle-Annotated Fields** — every task declares
  `inference_substrate`; every REQUIRED ARTIFACT FIELD carries a `principle:`.
- **Verdict Terminal-Prefix** — every `honest_verdict` starts `complete_` /
  `success_`.
- **Agent routing** — all experiments `agent_type: codex` / `model: gpt-5.5` (sprint
  default); planner/retro stay on Claude Opus via conductor env (operator choice).

---

## 7. Post-sprint handoff (operator-facing)

The sprint retires 2026-06-30. If `.452` A1 (env-grounded search) and A1b
(latent-action) BOTH fail to lift the live first-win rate, the honest position is:
the live first-win wall survives energy + goal-quality + FOUR world-model
representations + env-grounded search. At that point the deliverable is **the
current ~0.05 first-win agent (submitted) + the publishable FoVer verifier-ensemble
paper (paper_ready=true, north-star §1)**. D (exp4911) maps the post-sprint pivot
back to the verifier-moat / oracle-distinct directions (north-star §5) where
self-consistency is NOT near-ceiling, so the operator has a concrete next track when
the ARC sprint closes. The capstone (exp4912) states this plainly rather than
queuing representation #5.
