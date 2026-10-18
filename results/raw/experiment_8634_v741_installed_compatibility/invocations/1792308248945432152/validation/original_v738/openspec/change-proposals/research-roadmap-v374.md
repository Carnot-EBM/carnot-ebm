# Research Roadmap v374 — PROVE the off-ARC verifier transfer (close the CI), and replace open-loop WM search with closed-loop grounding

**Milestone:** 2026.06.374
**Planned:** 2026-06-11 (Claude Opus 4.8 planning agent, post-.373 close)
**Prior milestone doc:** `openspec/change-proposals/research-roadmap-v373.md`
**North star:** `ops/north-star.md` — solve ARC-AGI-3 accurately AND efficiently; the energy
ensemble is the VERIFICATION layer (router / pruner / scorer), never the generator.

---

## 1. What .373 measured (the honest read, from the artifacts not the prose)

.373's job was to convert THREE open arguments into measurements. It did — and the measurements
are more informative than a clean win would have been. Capstone exp4041
(`complete: capstone_v373_arguments_measured_G1_directional_underpowered_ci_touches_zero_G2_no_generalization_G3_absent_games7_arcmemo_win_flagged_skipped1`):

| Track | Result | Honest status |
|---|---|---|
| **G1 — verifier OFF-ARC** (exp4032, operator TOP PRIORITY) | The SAME GAP-4 execution primitive (induce program from demos → restricted exec → exact-output-match) on MBPP code: Arm B demo-fit beat Arm A vote by **+5.0pp**, same sign as ARC-1 (+16pp); positive control PASSES (oracle headroom present). | **DIRECTIONAL but UNDERPOWERED** — n=40, bootstrap CI95 **[0.0, 12.5] touches zero**. The artifact's own recommendation: scale to full HumanEval (164) / MBPP (~500). New gap logged: **GAP-CODE-EXEC-DEMOFIT** (visible tests under-determine hidden semantics). |
| **G2 — search past r11l** (exp4034 + exp4035) | exp4034 induced vc33 `is_goal(state)` at held-out precision/recall **1.0** (solid). exp4035 ran a REAL non-bespoke search (169 nodes, general coded heuristic, subgoals — NOT r11l hardcoded macros) and **found a plan in the verified WM**. | **NO generalization, NEW wall.** The "plan" was DEGENERATE — `[6,38,28]` repeated 70× — that satisfied `is_goal` *inside* the 99%-accurate WM but **failed real-env confirmation** (`levels_completed=0`). Textbook **planner-exploits-world-model-error** (MuZero arXiv:2306.00840): open-loop search exploited a WM defect off the data-collection distribution. |
| **G3 — decentralization** (exp4036 + exp4037) | exp4036 launched gemma-4-31B-it best-of-N backgrounded. | **INCONCLUSIVE — throughput failure, not a clean answer.** exp4037 collected **0 scored tasks** in budget (31B dense too slow for 30×k=8). coverage=0.0 is a non-measurement; latent-vs-absent is still open. |
| **Accuracy** (exp4038) | 7th game solved (dc22-fdcac232), action 20 vs baseline 59, real-env-confirmed | +1 monotonic, CLEAN |
| **Self-learning** (exp4039) | ArcMemo v6 concept-library: 59→18 actions, **0** induction calls | CLEAN win (modest) |
| **Hardware** (exp4040) | KV260 reachable, overlay LOADED, latency-transcript step BLOCKED; GateMate unreachable; PolarFire reachable | not yet terminal |
| **Flagged** (exp4031) | build-half of G1 flagged DURATION_TOO_SHORT (known false-positive: build+launch backgrounds the real run) | skipped by capstone; collect-half exp4032 clean |

**The three things .373 leaves for .374:**

1. **G1 is one power-bump away from the operator's headline.** The verifier's code transfer is
   directionally real (+5pp, right sign, real headroom) but n=40 cannot exclude zero. This is the
   single highest-leverage experiment in the project: convert "directional" → "measured" by
   scaling N **and** adding a stronger discriminator (the +5pp came from naive demo-fit; the
   literature's symbolic-equivalence partition and leave-one-out test-consistency should lift it).
2. **G2's open-loop search is the wrong architecture; the fix is named and cheap.** The vc33
   degenerate plan is not a vc33 quirk — it is the generic failure of planning over a model
   validated only on its data-collection distribution. The remedy is closed-loop: re-observe the
   REAL env after each action, trust the WM only one step ahead, and detect WM↔real divergence.
3. **G3 never got a clean number.** Two attempts (exp4022 flagged, exp4037 0-tasks) failed to
   *measure*. .374 must use a TRACTABLE substrate (a MoE with ~3B active params) so the
   latent-vs-absent question actually resolves — or the lineage retires with an honest bound.

## 2. The three biggest gaps (current state → north-star vision)

| # | Gap | Why it is load-bearing | .374 attack |
|---|---|---|---|
| **G1** | **The verifier's off-ARC generality is directional, not yet significant** (operator TOP PRIORITY) | The verifier IS the project's entire value-add (north-star §5). A +5pp/CI-touches-0 result is not yet publishable or decision-grade. ARC-AGI-3's new domains AND the publication claim need a *significant* cross-domain transfer. | Scale the SAME GAP-4 primitive to **full HumanEval (164) + an MBPP slice**, ADD stronger arms (symbolic-equivalence partition arXiv:2604.06485, ACES leave-one-out arXiv:2604.03922), 10k-resample task-level bootstrap. Does the CI now **EXCLUDE zero**? (Phase 1) |
| **G2** | **Open-loop search over a verified WM exploits model error and fails in the real env** | The .372 r11l win was open-loop-lucky; vc33 proves open-loop search over a data-collection-validated WM does not transfer. If the verified-WM-as-simulator approach can't be made to plan robustly, the whole search-layer pivot is in question. | Replace open-loop search with **closed-loop per-step replanning** (re-observe the REAL env each step; MuZero arXiv:2306.00840, GC-IDM arXiv:2605.08732, World-in-World arXiv:2510.18135) + a **WM-trust/divergence gate** (arXiv:2508.06096) that detects WM↔real divergence and refuses degenerate WM-exploiting plans. Does closed-loop grounding break vc33's wall where open-loop failed? (Phase 2) |
| **G3** | **The sovereign-generator latent-vs-absent question is unmeasured (two failed attempts)** | Decentralization rule 1 (local-first using open models) requires a local generator that can induce ARC rules. gemma-4-12B can't (0.2581, no lift). We still don't know if a stronger LOCAL base can — because the measurement never completed. | TRACTABLE rerun on **Qwen3.6-35B-A3B (MoE, ~3B active)** with batched generation + RoBoN-style early-stop (arXiv:2512.05542), same 30-task ARC-1 pool as exp4012, positive control mandatory. Does coverage rise above 0.2581 (latent → distillation viable, RL vs Distillation arXiv:2505.14216) or stay flat (absent → leash holds)? `retire_if_same_verdict`. (Phase 3) |

These map 1:1 onto the north-star §0 sequence: (1) offline verifier proof / domain expansion (G1),
(2) the path toward the ARC harness via robust planning (G2), (3) the sovereign substrate (G3).
**G1 remains the operator's literal TOP PRIORITY**; .373 brought it within one power-bump of the
CI-excludes-0 gate.

## 3. Architecture — where each .374 experiment sits

```
                         ARC-AGI-3 NORTH STAR (accurate + efficient)
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        │                                  │                                   │
   G1 VERIFIER GENERALITY            G2 ROBUST PLANNING                  G3 SOVEREIGN BASE
   (operator TOP PRIORITY)           (the search-layer's real test)     (local-first rule 1)
        │                                  │                                   │
  exp4044 BUILD+LAUNCH               exp4046 closed-loop replan +        exp4047 BUILD+LAUNCH
  full HumanEval+MBPP, k=8,          WM-trust/divergence gate over       Qwen3.6-35B-A3B MoE
  arms: vote / ACES-LOO /            the vc33 verified WM (re-observe     best-of-N, batched,
  demo-fit / symbolic-partition      REAL env per step; reject           early-stop, SAME pool
  + oracle positive control          WM-exploiting degenerate plans)     as exp4012 (0.2581)
        │                                  │                                   │
  exp4045 COLLECT+VALIDATE           real-env-confirm; honest            exp4048 COLLECT+VALIDATE
  bootstrap CI95 — EXCLUDES 0?       no-solve is COMPLETE; diagnose      coverage vs 0.2581 +
  which arm wins? GAP-CODE-          sim2real-ceiling vs                 positive control →
  EXEC-DEMOFIT residual logged       solvable-with-grounding            latent | absent | retire
        │                                  │                                   │
        └──────────────── feeds ───────────┴──────────── feeds ───────────────┘
                                          │
        ┌──────────────────┬──────────────┼──────────────┬─────────────────────┐
   ACCURACY            SELF-LEARNING     INFRA          HARDWARE              CAPSTONE
   exp4049 8th game     exp4050 ArcMemo   exp4051        exp4052 KV260         exp4053
   explore-first +1     v7 CROSS-GAME     registry+gaps  drive-to-terminal     aggregate, skip
   (monotonic)          library transfer  hygiene        (latency transcript)  flagged, sha256
```

**Provenance / reuse (no rebuild):**
- G1 reuses the model-free demo-fit primitives (`python/carnot/agentic/arc_gap4_execution_verifier.py`),
  the restricted executor (`python/carnot/verify/sandbox.py`), and the exp4032 runner shape.
- G2 reuses the vc33 verified WM (`results/arc3_vc33_world_model_program.py`) and exp4034's induced
  goal predicate (held-out precision 1.0); the NEW part is the closed-loop harness + divergence gate.
- G3 reuses the exact exp4012 30-task ARC-1 pool + model-free verifier; the NEW part is the MoE
  inducer + batched/early-stop throughput so the run actually completes.

## 4. Phases & dependency graph

- **Phase 0 — hygiene (exp4042 archive/activate, exp4043 SOTA-ingestion).** exp4042 is claude/opus
  + `requires_claude_verified` (the milestone-transition lesson); keeps the hardened green-gate +
  poison-test quarantine. exp4043 is the MANDATORY SOTA-ingestion slot (.374 headline is a
  bleeding-edge track: off-ARC verifier proof + closed-loop WM planning).
- **Phase 1 — G1 (exp4044 BUILD+LAUNCH → exp4045 COLLECT+VALIDATE).** Split-long-codex pair; the
  full-corpus generation is backgrounded so no agent is held past the 80-min cap. exp4045 `gated_on`
  exp4044 `runner_ready == true`.
- **Phase 2 — G2 (exp4046 closed-loop replanning).** Single task; the search runs offline over the
  verified WM but executes/confirms in the REAL env. `prior_failures:` cites exp4035.
- **Phase 3 — G3 (exp4047 BUILD+LAUNCH → exp4048 COLLECT+VALIDATE).** Split-long-codex pair;
  `prior_failures:` cites exp4022 + exp4037; `retire_if_same_verdict: true`. exp4048 `gated_on`
  exp4047 `runner_ready == true`.
- **Phase 4 — proven tracks + self-learning + infra + hardware + capstone (exp4049–exp4053).**
  exp4049 8th game (+1 monotonic). exp4050 ArcMemo v7 cross-game transfer (reads exp4049). exp4051
  registry/gaps hygiene (reserved infra slot; logs the G1 off-ARC outcome + closed-loop outcome).
  exp4052 KV260 drive-to-terminal. exp4053 capstone (UNGATED; skips flagged; cites sha256).

**Dependency edges:** exp4045←exp4044 (runner_ready); exp4048←exp4047 (runner_ready); exp4046 reads
exp4034 goal predicate; exp4050 reads exp4049 solve trace; exp4053 aggregates exp4044–exp4052.

## 5. Hardware requirements

- **Phase 1/3 GGUF inference:** single RTX 3090 sufficient; gemma-4-12B-it-GGUF (G1, fast) and
  Qwen3.6-35B-A3B-GGUF (G3, MoE ~3B active for throughput) via llama.cpp. PRECONDITIONS gate each.
- **Phase 2:** CPU-only (search over a Python WM program + live ARC env over the SDK anonymous key).
- **Phase 4 hardware (exp4052):** KV260 over SSH (`ssh kria`), GateMate (`openFPGALoader --detect`),
  PolarFire (`ssh polarfire`) — SSH/USB preconditions only; NEVER a host SD-card block device
  (KV260 SSH-Not-SD-Card Discipline). Distinct wall-clock timers per board.

## 6. Routing & discipline

- **Codex-Default v2 (gemini BANNED):** every experiment task is `agent_type: codex` /
  `model: gpt-5.5`. Only exp4042 (archive/activate) is `claude` / `opus` / `requires_claude_verified`.
- **Split-long-codex:** the two many-GGUF-call experiments (G1 exp4044/4045, G3 exp4047/4048) are
  each split into a fast BUILD+LAUNCH (backgrounds the real run) + a COLLECT+poll task.
- **Failed-Experiment Rerun Discipline:** exp4046 (`prior_failures: exp4035`), exp4047/exp4048
  (`prior_failures: exp4022 + exp4037`, `retire_if_same_verdict: true`). G1 (exp4044/4045) carries
  `operator_override` — it is the operator's standing TOP PRIORITY forward scale-up of a
  directional positive (exp4032), with a STATED difference (full N + stronger discriminator arms).
- **Incremental-Progress (ARC):** exp4049 targets +1 game, never "solve all."
- **Reserved infra slots (≥2 for a ≥10-task milestone):** exp4042 (archive/activate) + exp4051
  (registry/gaps). **SOTA-ingestion slot:** exp4043. **Self-learning mandate:** exp4050.
- **Adversarial-verify / Reading-Results:** every compute artifact carries `inference_substrate`,
  `model_specs`, `random_seed`, `reproducibility_checksum`; the capstone reads each upstream via
  `summarize_artifact.py`, SKIPS any `flagged_adversarial`, and cites sha256.

## 7. Acceptance — what makes .374 a win

.374 is a win if it answers the three questions, positive OR honest-negative:

1. **G1:** does the off-ARC demo-fit transfer's bootstrap CI95 EXCLUDE zero at full HumanEval(+MBPP)
   scale, and/or does a stronger discriminator lift the delta materially? (Either a measured
   positive — the headline — or an honest "the verifier's code-transfer magnitude is small and
   needs a richer discriminator," with the GAP-CODE-EXEC-DEMOFIT residual characterized.)
2. **G2:** does closed-loop per-step grounding + a WM-trust gate break vc33's wall where open-loop
   failed? (Either a real-env-confirmed solve — the search layer generalizes when grounded — or an
   honest "verified-WM-as-simulator has a sim2real ceiling on vc33," with the divergence quantified.)
3. **G3:** does a tractable MoE base's best-of-N coverage rise above 0.2581 (latent) or stay flat
   (absent)? (A clean number either way; if it again fails to *measure*, the stronger-base lineage
   retires per `retire_if_same_verdict`.)

Plus: monotonic accuracy (8th game), a self-learning cross-game transfer datum, registry/gaps kept
honest, and KV260 moved toward terminal.

## 8. Cross-references
- `ops/north-star.md` §0/§5 (the destination + the verifier-is-the-value-add reframe)
- `ops/verifier_gaps.md` (GAP-4 positive, GAP-3 retired-negative, GAP-5 + GAP-CODE-EXEC-DEMOFIT open)
- `results/experiment_4041_capstone_v373.json` (the .373 scorecard this milestone builds on)
- `research-references.md` "2026-06-11 Post-.373 Planning Sweep" (the verified .374 citations)
- `results/experiment_4030_sota_ingestion_receipt.json` (`flagged_for_v374`: off-ARC vs ACES/DOCE,
  vc33 search with re-rooting fallback, symbolic-partition for code — all addressed here)
