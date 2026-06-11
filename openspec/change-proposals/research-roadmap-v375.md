# Research Roadmap v375 — POWER the off-ARC verifier transfer on an UN-SATURATED corpus (resume-not-restart); bank the WM-planning ceiling; accumulate the sovereign-base measurement

**Milestone:** 2026.06.375
**Planned:** 2026-06-11 (Claude Opus 4.8 planning agent, post-.374 close)
**Prior milestone doc:** `openspec/change-proposals/research-roadmap-v374.md`
**North star:** `ops/north-star.md` — solve ARC-AGI-3 accurately AND efficiently; the energy
ensemble is the VERIFICATION layer (router / pruner / scorer), never the generator.

---

## 1. What .374 measured (the honest read, from the artifacts not the prose)

.374's job was to make three .373 measurements decision-grade. It produced ONE decision-grade
result (G2), surfaced a precise NEW root cause for the operator's top priority (G1), and confirmed
the throughput fix works but needs more windows (G3). Capstone exp4053
(`complete: capstone_v374_not_decision_grade_G1_partial_or_incomplete_G2_closed_loop_ceiling_saturated_sim2real_divergence_G3_retired_non_measurement_games8_arcmemo_v7_no_win_flagged_skipped2`).
**Read the capstone against the operator's known-issues entry (2026-06-11): the capstone's G3
`retired_non_measurement` was a FALSE retirement — the line is underpowered, not dead.**

| Track | Result | Honest status |
|---|---|---|
| **G1 — verifier OFF-ARC** (exp4045, operator TOP PRIORITY) | Full-power run completed **22 of N≥160** tasks; on that subset **every arm AND the oracle = 1.0** (delta 0.0, CI [0,0]). Build (exp4044) flagged DURATION_TOO_SHORT. | **NON-measurement, TWO root causes.** (a) **throughput** — 22 tasks fit the 75-min window; (b) **corpus saturation** — base HumanEval/MBPP have **no oracle headroom**, so the demo-fit verifier cannot show value even at full N. Literature-confirmed: base HumanEval/MBPP saturated since 2023; **EvalPlus reopened the gap** with 80× hidden tests. |
| **G2 — closed-loop planning over vc33 WM** (exp4046) | Closed-loop per-step replan + WM-trust gate did **NOT** break vc33's wall; no real-env solve. **Per-step WM↔real divergence = 0.207.** | **DECISION-GRADE NEGATIVE (banked).** The "99%-accurate" verified WM diverges ~21%/step under the *planning* distribution — a **sim2real ceiling** on the verified-WM-as-simulator approach for vc33 (logged GAP-ARC3-VC33-SIM2REAL-CEILING). vc33 WM-planning is **retired**; the forward path is the proven explore-first + verifier-pruner line. |
| **G3 — sovereign base** (exp4048) | MoE Qwen3.6-35B-A3B (the throughput fix) scored **14 tasks/window** (vs 31B-dense's 0); coverage **0.3571 vs 12B 0.2581**, bootstrap95 **[0.143, 0.643] spans the ceiling**. Build (exp4047) flagged. | **UNDERPOWERED, NOT retired** (the 6-task poll's `partial_..._retire` was a false retirement per operator known-issues). The throughput fix works; the run needs more windows to clear N≥30. |
| **Accuracy** (exp4049) | 8th game solved (sb26-7fbdac44, action 9, real-env-confirmed) | +1 monotonic, CLEAN |
| **Self-learning** (exp4050) | ArcMemo v7 cross-game helped vs cold (18→9) but **lost to within-game v6** (7); 1 abstraction reused | weak cross-game transfer |
| **Hardware** (exp4052) | KV260 **TERMINAL** — overlay loaded + board-latency transcript (median 0.002 ms / batch 0.24 ms) | KV260 done → opportunistic; GateMate + PolarFire non-terminal |
| **Flagged** (exp4044, exp4047) | both split-long-codex BUILD halves flagged DURATION_TOO_SHORT (known false-positive: build+launch backgrounds the real run) | skipped by capstone; collect halves (exp4045/4048) clean |

**The three things .374 leaves for .375:**

1. **G1's root cause is now precise — and the fix is literature-grounded.** The +5pp directional
   signal (.373, n=40) plus the .374 all-arms-1.0 degeneracy together say: base HumanEval/MBPP is
   **saturated** — the demo-fit verifier has no headroom to add value there. The fix is not "more N
   on the same corpus"; it is **the right corpus** (EvalPlus HumanEval+/MBPP+ hidden tests, which
   "reopened the gap") **plus** the operator's resume-not-restart accumulation so the run actually
   reaches N≥160.
2. **G2 is settled — bank it, don't churn.** vc33 WM-planning hits a 21%/step sim2real ceiling;
   the working planning method is explore-first + the GAP-4 verifier as an action-pruner (8 games
   solved). .375 pivots planning effort to that proven line + its EFFICIENCY (north-star §"efficient").
3. **G3 is one or two windows from a clean number — resume it.** The MoE throughput fix works
   (14 tasks/window); resume the stable checkpoint toward N≥30 and report accumulated-N.

## 2. The three biggest gaps (current state → north-star vision)

| # | Gap | Why it is load-bearing | .375 attack |
|---|---|---|---|
| **G1** | **The verifier's off-ARC generality is still unmeasured** — the .374 corpus was saturated AND the run truncated (operator TOP PRIORITY) | The verifier IS the project's entire value-add (north-star §5; both energy theses are bounded-negative). "The verifier generalizes off-ARC" is still ARGUED, not MEASURED — and a saturated corpus can never measure it. | **Fix the corpus AND the throughput.** Score arms against **EvalPlus HUMANEval+/MBPP+ hidden tests** (un-saturated → real oracle headroom), keep demo-fit selecting on VISIBLE tests, add a SAGA-style generated-test arm (arXiv:2507.06920). **Resume-not-restart**: a stable corpus+model+k checkpoint accumulates toward N≥160; report ACCUMULATED-N + oracle headroom; the headline gate = best-arm CI95 lower bound > 0 with headroom present. (Phase 1) |
| **G3** | **The sovereign-generator latent-vs-absent question is underpowered, not answered** (operator-mandated resume) | Decentralization rule 1 (local-first using open models) needs a local generator that can induce ARC rules. The MoE throughput fix works; we are 1-2 windows from a verdict. | **Resume the MoE checkpoint** (14 tasks) toward accumulated N≥30; bootstrap CI of (coverage − 0.2581); positive control mandatory; latent (CI excludes 0.2581 → distillation viable) vs absent (flat → the Invisible Leash holds). NO premature retire. (Phase 2) |
| **G3′ / efficiency** | **The verifier's value is proven on accuracy + cost, but the "efficient agent" north-star axis (verifier as action-pruner in the live harness) is not yet measured** | North-star = accurate AND efficient. G2's WM-planning is banked; the forward pivot is the verifier as an **online action-pruner** (arXiv:2602.01070: online verification beats post-hoc reranking). 8 solved games unblock the M3 verifier-as-action-pruner experiment that was gated-on-a-solve. | Run the proven explore-first solver on solved games WITH vs WITHOUT the GAP-4 verifier as an action-pruner; measure action-count + wall-clock at equal solve-rate. The efficiency datum is the agentic proof. (Phase 3) |

These map onto the north-star §0 sequence: (1) offline verifier proof / **domain expansion** (G1),
(2) the **sovereign substrate** (G3), (3) the **efficient agentic harness** (the verifier-as-pruner
pivot). **G1 remains the operator's literal TOP PRIORITY**; .374 diagnosed exactly why it has not
closed, and the fix (un-saturated corpus + resume-accumulate) is concrete and operator-mandated.

## 3. Architecture — where each .375 experiment sits

```
                         ARC-AGI-3 NORTH STAR (accurate + efficient)
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        │                                  │                                   │
   G1 VERIFIER GENERALITY            G3 SOVEREIGN BASE                  EFFICIENT HARNESS
   (operator TOP PRIORITY)          (local-first rule 1, resume)       (the G2 pivot)
        │                                  │                                   │
  exp4056 BUILD+LAUNCH               exp4058 BUILD+LAUNCH               exp4061 verifier-as-
  RESUME candidate-gen checkpoint    RESUME MoE checkpoint (14→N≥30)    action-pruner on solved
  → N≥160; EVALUATE vs EvalPlus      Qwen3.6-35B-A3B, batched,          games: explore-first
  HumanEval+/MBPP+ hidden tests;     early-stop, SAME exp4012 pool      WITH vs WITHOUT GAP-4
  arms: vote / ACES / demo-fit /     (0.2581); stable checkpoint        pruning; actions + wall-
  symbolic-partition / SAGA-tests    accumulates across windows         clock at equal solve-rate
        │                                  │                                   │
  exp4057 COLLECT+VALIDATE           exp4059 COLLECT+VALIDATE           (efficiency datum =
  ACCUMULATED-N; oracle headroom     ACCUMULATED-N; CI vs 0.2581 →      the agentic proof; online
  present? best-arm CI95 > 0?        latent | absent | still-under      verification > reranking)
  GAP-CODE-EXEC-DEMOFIT residual     NO premature retire
        │                                  │                                   │
        └──────────────── feeds ───────────┴──────────── feeds ───────────────┘
                                          │
        ┌──────────────────┬──────────────┼──────────────┬─────────────────────┐
   ACCURACY            SELF-LEARNING     INFRA          HARDWARE              CAPSTONE
   exp4060 9th game     exp4062 ArcMemo   exp4063        exp4064 GateMate +    exp4065
   explore-first +1     v8 richer cross-  registry+gaps  PolarFire toward      aggregate, skip
   (monotonic)          game library      hygiene        terminal (KV260 done) flagged, sha256
```

**Provenance / reuse (no rebuild):**
- **G1** resumes the .374 candidate-generation checkpoint (`experiment_4045_offarc_transfer_power*.checkpoint.json`)
  and the model-free demo-fit primitives (`python/carnot/agentic/arc_gap4_execution_verifier.py`) +
  the restricted executor (`python/carnot/verify/sandbox.py`). The NEW part is the **EvalPlus
  hidden-test evaluation** (un-saturated) + a stable accumulating checkpoint + the SAGA-test arm.
- **G3** resumes the .374 MoE checkpoint (`experiment_4048_decentralization_moe_base_raw.checkpoint.json`)
  + the exact exp4012 30-task ARC-1 pool + the model-free verifier. The NEW part is the **stable
  corpus+model+k-keyed checkpoint** so windows accumulate (the operator resume-not-restart fix).
- **Efficiency (exp4061)** reuses the explore-first solver + GAP-4 verifier over the SOLVED-game
  traces; the NEW part is the action-pruner ablation harness.

## 4. Phases & dependency graph

- **Phase 0 — hygiene (exp4054 archive/activate, exp4055 SOTA-ingestion).** exp4054 is claude/opus
  + `requires_claude_verified` (the milestone-transition lesson); keeps the hardened green-gate +
  poison-test quarantine; **must record G3 as underpowered-NOT-retired** (do not propagate the
  false retirement). exp4055 is the MANDATORY SOTA-ingestion slot (.375 headline is a bleeding-edge
  track: un-saturated execution-verification + verifier-as-pruner efficiency).
- **Phase 1 — G1 (exp4056 BUILD+LAUNCH → exp4057 COLLECT+VALIDATE).** Split-long-codex pair; the
  generation is backgrounded under `setsid` so no agent is held past the 80-min cap and the run
  survives the iteration boundary. exp4057 `gated_on` exp4056 `runner_ready == true`.
- **Phase 2 — G3 (exp4058 BUILD+LAUNCH → exp4059 COLLECT+VALIDATE).** Split-long-codex pair;
  resume-not-restart; `operator_override` cites the 2026-06-11 resume directive. exp4059 `gated_on`
  exp4058 `runner_ready == true`. NO premature retire on a throughput-truncated window.
- **Phase 3 — proven tracks + efficiency + self-learning + infra + hardware + capstone
  (exp4060–exp4065).** exp4060 9th game (+1 monotonic). exp4061 verifier-as-action-pruner efficiency
  (the G2 pivot; the north-star efficient axis). exp4062 ArcMemo v8 richer cross-game library
  (self-learning mandate; reads exp4060). exp4063 registry/gaps hygiene (reserved infra slot; logs
  the G1 accumulated outcome + the G2 sim2real-ceiling gap + the G3 accumulated coverage). exp4064
  GateMate/PolarFire toward terminal (KV260 is done). exp4065 capstone (UNGATED; skips flagged; sha256).

**Dependency edges:** exp4057←exp4056 (runner_ready); exp4059←exp4058 (runner_ready); exp4062 reads
exp4060 solve trace; exp4065 aggregates exp4056–exp4064.

## 5. Hardware requirements

- **Phase 1/2 GGUF inference:** single RTX 3090 sufficient; gemma-4-12B-it-GGUF (G1, fast inducer,
  the SAME model as exp4032/4045 — only N + corpus-evaluation change) and Qwen3.6-35B-A3B-GGUF (G3,
  MoE ~3B active) via llama.cpp. PRECONDITIONS gate each; EvalPlus (HumanEval+/MBPP+) datasets cached.
- **Phase 3 efficiency (exp4061) + accuracy (exp4060):** CPU + live ARC env over the SDK anonymous
  key (no heavy GGUF generation — these are the tractable offline-ARC tasks that DO complete).
- **Phase 3 hardware (exp4064):** GateMate (`openFPGALoader -c dirtyJtag --detect`), PolarFire
  (`ssh polarfire`) — USB/SSH preconditions only. KV260 is TERMINAL (opportunistic confirm only).
  Distinct wall-clock timers per board (no shared-timestamp tautology).

## 6. Routing & discipline

- **Codex-Default v2 (gemini BANNED):** every experiment task is `agent_type: codex` /
  `model: gpt-5.5`. Only exp4054 (archive/activate) is `claude` / `opus` / `requires_claude_verified`.
- **Resume-not-restart (operator #1 MANDATORY, known-issues 2026-06-11):** G1 (exp4056) and G3
  (exp4058) use STABLE (corpus+model+k-keyed) checkpoints that accumulate across milestones, report
  ACCUMULATED-N, and NEVER fire `retire_if_same_verdict` on a throughput-truncated window. The
  detached generation subprocess is `setsid`-detached to survive reaping.
- **Split-long-codex:** the two many-GGUF-call experiments (G1, G3) are each split into a fast
  BUILD+LAUNCH (backgrounds the real run) + a COLLECT+poll task.
- **Failed-Experiment Rerun Discipline:** vc33 WM-planning is RETIRED (G2 sim2real ceiling) — not
  re-proposed. G1/G3 carry `operator_override` citing the standing TOP PRIORITY + the resume
  directive + the STATED forward difference (G1: EvalPlus un-saturated evaluation; G3: stable
  accumulating checkpoint). exp4061 (verifier-as-pruner) is forward work newly unblocked by the 8
  solves (the M3 experiment was gated-on-a-solve).
- **Incremental-Progress (ARC):** exp4060 targets +1 game (9th), never "solve all".
- **Reserved infra slots (≥2 for a ≥10-task milestone):** exp4054 (archive/activate) + exp4063
  (registry/gaps). **SOTA-ingestion slot:** exp4055. **Self-learning mandate:** exp4062.
- **Adversarial-verify / Reading-Results:** every compute artifact carries `inference_substrate`,
  `model_specs`, `random_seed`, `reproducibility_checksum`; gated/required scalar fields emitted
  BARE; the capstone reads each upstream via `summarize_artifact.py`, SKIPS any `flagged_adversarial`,
  and cites sha256.
- **Positive-control / FALSE_NEGATIVE_RISK:** G1 and G3 both REQUIRE an oracle/best-of-pool positive
  control. A null result (no transfer, no base-size lift) is only reported if oracle headroom is
  present; oracle≈baseline ⇒ uninformative/saturated, escalate — never "the verifier fails".

## 7. Acceptance — what makes .375 a win

.375 is a win if it answers the questions, positive OR honest-negative:

1. **G1 (TOP PRIORITY):** with the demo-fit verifier evaluated against EvalPlus hidden tests on an
   ACCUMULATED N≥160 (or the largest N the windows reach), does the best-arm bootstrap CI95 lower
   bound EXCLUDE zero WITH oracle headroom present? (Either a measured positive — the headline
   off-ARC transfer — or an honest "the magnitude is small / the demo-fit verifier needs a
   semantic-partition or generated-test discriminator," with the GAP-CODE-EXEC-DEMOFIT residual
   characterized and the accumulated-N reported so the NEXT window continues it.)
2. **G3:** with the MoE checkpoint accumulated toward N≥30, does best-of-N coverage rise above
   0.2581 (CI excludes the ceiling → latent → distillation viable) or stay flat (absent → the
   Invisible Leash holds)? A clean number either way; if still under N≥30, report accumulated-N as
   PROGRESS (not a retirement).
3. **Efficiency (the G2 pivot):** does the GAP-4 verifier as an online action-pruner reduce
   actions-to-solve / wall-clock on solved games at equal solve-rate? (The agentic-proof efficiency
   datum, or an honest null.)

Plus: monotonic accuracy (9th game), a richer cross-game self-learning datum, registry/gaps kept
honest (G1/G2/G3 outcomes logged), and GateMate/PolarFire moved toward terminal.

## 8. Cross-references
- `ops/north-star.md` §0/§5 (the destination + the verifier-is-the-value-add reframe)
- `ops/known-issues.md` MANDATORY-NEXT-MILESTONE PRIORITIES (the 2026-06-11 resume-not-restart +
  off-ARC TOP PRIORITY operator directives this milestone executes)
- `ops/verifier_gaps.md` (GAP-4 positive, GAP-CODE-EXEC-DEMOFIT open, GAP-ARC3-VC33-SIM2REAL-CEILING
  new, GAP-DECENTRALIZATION-MOE-BASE-4048 open)
- `results/experiment_4053_capstone_v374.json` (the .374 scorecard this milestone builds on)
- `research-references.md` "2026-06-11 Post-.374 Planning Sweep" (the verified .375 citations:
  EvalPlus saturation fix, SAGA arXiv:2507.06920, DryRUN arXiv:2604.21598, online-verification
  arXiv:2602.01070)
