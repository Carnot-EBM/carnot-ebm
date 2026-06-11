# Research Roadmap v373 — MEASURE the verifier OFF-ARC + GENERALIZE the search layer past one bespoke point

**Milestone:** 2026.06.373
**Planned:** 2026-06-11 (Claude Opus 4.8 planning agent, post-.372 close)
**Prior milestone doc:** `docs/research-notes/deep-think-results-2026-06-10.md` (.372 Deep-Think pivot)
**North star:** `ops/north-star.md` — solve ARC-AGI-3 accurately AND efficiently; the energy
ensemble is the VERIFICATION layer (router / pruner / scorer), never the generator.

---

## 1. What .372 proved (and what it left thin)

The .372 Deep-Think pivot committed to building the SEARCH/HEURISTIC layer the planning wall
demands. The capstone (exp4028) verdict: `pivot_central_bet_advanced: true`. Honest reading of
the eight upstream artifacts:

| Track | Result | Honest status |
|---|---|---|
| **Goal-predicate separation** (exp4020) | `is_goal(state)` induced separately from dynamics; held-out precision **1.0** | SOLID, reusable component |
| **Search over verified WM** (exp4021) | Solved **r11l L4** via heuristic search; real-env-confirmed; `wall_was_search_not_representation: true` | **THIN** — `nodes_expanded: 3`, `action_count: 2`, with **r11l-specific hardcoded macros** (`r11l_safe_composite_path_macro`). One bespoke data point, not a general planner. |
| **Decentralization** (exp4022) | Branch B (distillation feasibility); `representational_gap_likely` | **FLAGGED adversarial** (DURATION_TOO_SHORT 0.746s + METHODOLOGY_MISSING) → skipped by the capstone. UNRESOLVED. |
| **Selection demoted** (exp4023) | agreement-as-selector RETIRED (confidence label only); safety gate KEPT | Closed cleanly |
| **Accuracy** (exp4024) | 6th game solved (cd82) via explore-first | +1 monotonic |
| **Self-learning** (exp4025) | ArcMemo transfer 71→21 actions | win |
| **Efficiency** (exp4026) | verifier vs LLM-judge: accuracy parity, **95.3× cheaper wall-clock**, 236× cheaper tokens | CLEAN north-star §5 win |
| **Hardware** (exp4027) | KV260 reachable (overlay absent), GateMate unreachable, PolarFire reachable | continuity recorded |

**The two things .372 left thin, and the one it never measured:**

1. **The search layer is unproven beyond r11l.** `nodes_expanded=3` with hand-coded r11l macros
   is not a planner — it is "one step of lookahead with a goal predicate solved a soft wall."
   We do not yet know whether heuristic search over a verifier-certified world model is a
   GENERAL planning capability or an r11l-specific trick.
2. **The sovereign generator path is blocked at the 12B ceiling, and the one measurement was
   fabrication-flagged.** exp4012 measured gemma-4-12B best-of-N (k=8) at **0.2581 demo-perfect
   coverage = NO lift** over the 3-attempt baseline (5× slower than codex). exp4022's branch
   decision (representational-gap-likely) was flagged and skipped.
3. **The verifier's domain-generality is ARGUED, never MEASURED.** The GAP-4 execution-
   consistency primitive (induce program from demos → execute in a restricted namespace → accept
   iff exact-output-match) is domain-general *by construction* — but `ops/verifier_gaps.md` is
   100% ARC. It has never run off-ARC. "The verifier generalizes" is an open argument.

## 2. The three biggest gaps (current state → north-star vision)

| # | Gap | Why it is load-bearing | .373 attack |
|---|---|---|---|
| **G1** | **Verifier domain-generality is argued, not measured** (operator TOP PRIORITY 2026-06-11) | The verifier IS the project's entire value-add (north-star §5). It is domain-bound today (math strong, code weak, ARC-only-measured). ARC-AGI-3's new domains AND the publication claim both need a *measured* cross-domain transfer. | Run the SAME GAP-4 execution primitive on **MBPP/HumanEval** (visible tests = demos, hidden tests = gold). Two arms, same candidate pool, bootstrap-CI gate + positive control. (Phase 1) |
| **G2** | **The search/navigator layer is unproven past one bespoke point** | The .372 central bet "advanced" on a 3-node r11l-macro search. Generality is the whole claim — if it only works with hand-coded r11l macros, the pivot is game-specific. | Generalize to a **SECOND game's wall — vc33** (the canonical 99%-accurate-WM-still-fails-to-solve case) with **subgoal/landmark decomposition** (HWM / Subgoal-PHS) + a **non-bespoke coded heuristic** + a **real bounded search** (report nodes_expanded, branching). (Phase 2) |
| **G3** | **Sovereign generator blocked at 12B; the one measurement was flagged** | Decentralization rule 1 (local-first using open models) requires a local generator that can induce ARC rules. gemma-4-12B can't. We don't know if a stronger LOCAL base can. | Clean rerun with a **stronger local base** (gemma-4-31B dense / Qwen3.6-35B MoE). Per "The Invisible Leash": does best-of-N demo-perfect coverage rise above 0.2581 (latent → distillation viable) or stay flat (absent → leash holds, need a bigger base)? (Phase 3) |

These map 1:1 onto the north-star §0 sequence: (1) offline verifier proof / domain expansion, (2)
verifier domain expansion, (3) the path toward the ARC harness. **G1 is the operator's literal
TOP PRIORITY** (`ops/known-issues.md`, 2026-06-11).

## 3. Architecture — where each .373 experiment sits

```
                         ARC-AGI-3 NORTH STAR (accurate + efficient)
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
   GENERATOR                       VERIFIER (Carnot's value-add)     NAVIGATOR (the .372 new layer)
   (commodity/local)              ┌─────┴─────────────┐            ┌──────┴───────────┐
        │                         │                   │            │                  │
   G3: stronger LOCAL base    G1: OFF-ARC transfer  efficiency   G2: goal predicate  hierarchical
   best-of-N induction        MBPP/HumanEval        (proven      (exp4020 reuse)     search over the
   (exp4036 build/4037 run)   (exp4031 build/        .372 win)       │               verified WM
        │                      4032 run)            ───────────  vc33 is_goal         w/ SUBGOALS
   "Invisible Leash"          induce→exec→match     registry      (exp4034)          (exp4035)
   latent vs absent           = domain-general      hygiene            └──────┬───────────┘
                              by construction       (exp4033)         depth test: general planner
                                                                       or r11l-specific trick?

   SELF-LEARNING: ArcMemo v6 concept-LIBRARY (LILO/Stitch compression)  (exp4039)
   ACCURACY: 7th game first-solve, explore-first, monotonic +1          (exp4038)
   HARDWARE: KV260 drive-toward-terminal + per-board continuity         (exp4040)
```

## 4. Phases (13 tasks, exp4029–exp4041)

### Phase 0 — Milestone transition + SOTA ingestion (infrastructure)
- **exp4029** — archive .372 → activate .373; keep the hardened green-gate + poison-test
  quarantine; record the .372 close-state. (claude + opus, requires_claude_verified — the
  archive/activate multi-file task hits the codex wall-clock cap; the exp4019/exp4008 lesson.)
- **exp4030** — **SOTA-ingestion slot** (mandatory per the standing 2026-06-11 discipline):
  read research-studying / research-references filtered to the .373 bleeding-edge tracks
  (off-ARC execution verification + hierarchical search over world models), run a FOCUSED fresh
  sweep (sweep_clusters / sweep_semscholar + low-concurrency WebSearch — NOT /deep-research),
  emit a SOTA→experiment mapping artifact, flag the strongest methods for .374.

### Phase 1 — Verifier domain-generality OFF-ARC (G1, the TOP PRIORITY headline)
- **exp4031** — off-ARC exec-verifier transfer **BUILD+LAUNCH**: write the standalone two-arm +
  positive-control runner (reuse `python/carnot/verify/sandbox.py` restricted exec + the GAP-4
  demo-fit/content-hash logic in `python/carnot/agentic/arc_gap4_execution_verifier.py` — do NOT
  write a code-specific path; the point is the primitive is SHARED), smoke it on 2 tasks, commit,
  then launch the full run backgrounded. (Split-long-codex discipline 2026-06-10.)
- **exp4032** — off-ARC exec-verifier transfer **COLLECT+VALIDATE**: poll the backgrounded raw
  result; validate the bootstrap CI95 and the positive control (oracle ≈ vote ⇒ ceiling-saturated
  ⇒ uninformative, escalate; never report "fails to transfer"); emit the three-outcome verdict.
- **exp4033** — verifier-registry + harness registration (infrastructure; GAP-4 conductor
  follow-up #4): register `gap4_program_induction_stack` as a reusable module in
  `ops/verifier_registry.yaml`, bit-exact offline re-eval (must reproduce ARC-1 28/31 and ARC-2
  19/31), and record the off-ARC outcome (a code-domain registry entry if exp4032 positive, a
  `ops/verifier_gaps.md` entry if negative).

### Phase 2 — Generalize the search/navigator layer (G2, the central-bet hardening)
- **exp4034** — goal-predicate induction for a **SECOND game (vc33)**: reuse
  `arc3_vc33_world_model_program.py` (the 99%-accurate verified WM already on disk); induce
  `is_goal(state)` from observed level-up transitions; verify on held-out level-ups. (Eureka-style
  LLM-coded predicate; no new env sweep.)
- **exp4035** — **hierarchical search over the verified vc33 WM**: add subgoal/landmark
  decomposition (HWM arXiv:2604.03208 / Subgoal-PHS arXiv:2506.07255) + a NON-bespoke coded
  heuristic over the exp4034 goal predicate; run a real bounded best-first/A* search (report
  nodes_expanded, branching). The depth test: does the search layer break vc33's wall (where a
  99%-WM single-step approach failed), or only r11l with hand macros?

### Phase 3 — Sovereign generator + self-learning + accuracy (G3 + mandates)
- **exp4036** — decentralization stronger-base **BUILD+LAUNCH**: write the best-of-N runner for a
  STRONGER local base (gemma-4-31B dense or Qwen3.6-35B-A3B MoE), smoke, commit, launch bg run on
  the SAME 30-task ARC-1 pool as exp4012 (apples-to-apples).
- **exp4037** — decentralization stronger-base **COLLECT+VALIDATE**: does demo-perfect coverage
  rise above exp4012's 0.2581? Diagnose latent (distillation viable) vs absent (leash holds). The
  clean replacement for the flagged exp4022 (prior_failures cites it).
- **exp4038** — **7th ARC-AGI-3 game first-solve** via the proven explore-first method (monotonic
  +1 accuracy; incremental-progress rule).
- **exp4039** — **ArcMemo v6: concept-LIBRARY learning** (self-learning mandate): per LILO/Stitch
  (arXiv:2310.19791), compress recurring induced-program fragments into named, documented
  abstractions; measure whether the library makes .373's new solves cheaper than v5.

### Phase 4 — Hardware + capstone
- **exp4040** — hardware continuity (consolidated): per-board reachability + KV260
  drive-toward-terminal (load the `carnot_ising` overlay that exp4027 found absent, then a
  board-latency transcript step toward the KV260 terminal state per north-star §3). SSH-only
  KV260 precondition.
- **exp4041** — **Capstone .373**: did the verifier MEASURABLY generalize off-ARC (G1), and did
  the search layer generalize past the bespoke r11l point to vc33 (G2)? Aggregate all clean
  artifacts; skip any flagged_adversarial; cite upstream sha256.

## 5. Dependency graph

```
exp4029 (archive/activate) ─── gates milestone activation
exp4030 (SOTA ingestion) ───── informs exp4031/exp4035 design (read first)

exp4031 (offARC BUILD+LAUNCH) ──poll──> exp4032 (offARC COLLECT) ──> exp4033 (registry: code entry or gap)
exp4034 (vc33 goal predicate) ──gated(precision≥0.5)──> exp4035 (hierarchical search over vc33 WM)
exp4036 (decentral BUILD+LAUNCH) ──poll──> exp4037 (decentral COLLECT; prior_failures exp4022)
exp4038 (7th game) ─ independent
exp4039 (ArcMemo v6) ─ consumes exp4035/exp4038 solve content if present (soft)
exp4040 (hardware) ─ independent
exp4041 (capstone) ─ aggregates ALL clean upstreams (ungated)
```

## 6. Hardware requirements

- **Local SOTA GGUF (mandatory):** off-ARC transfer + decentralization need local generation.
  exp4031/4032 use `unsloth/gemma-4-12B-it-GGUF` (throughput). exp4036/4037 use a STRONGER base:
  `unsloth/gemma-4-31B-it-GGUF` (dense) or `unsloth/Qwen3.6-35B-A3B-GGUF` (MoE) — whichever is
  cached / faster; PRECONDITION the cache before first use.
- **2× RTX 3090 (CUDA):** GGUF inference for the best-of-N arms. The model-free verifier is CPU,
  ~$0 (~0.1s/task).
- **FPGA boards (continuity):** KV260 (`ssh kria` + `xmutil`), GateMate
  (`openFPGALoader -c dirtyJtag --detect`), PolarFire (`ssh polarfire`). KV260 is the
  drive-toward-terminal board per north-star §3; GateMate/PolarFire opportunistic.

## 7. Disciplines honored

- **Codex-Default v2:** all experiment tasks `codex` / `gpt-5.5`; gemini BANNED; planner/retro/
  audits stay Opus. Archive/activate = `claude` + `opus` + `requires_claude_verified`.
- **Split-long-codex (2026-06-10):** the two many-GGUF-call experiments (off-ARC transfer,
  decentralization) are each split into a fast BUILD+LAUNCH task + a COLLECT/poll task so no agent
  is held open past the 80-min cap.
- **Incremental-progress (ARC):** exp4038 targets +1 game (monotonic), never "solve all."
- **Reserved infra slots (≥2):** exp4029 (archive/activate) + exp4033 (registry/harness).
- **SOTA-ingestion slot:** exp4030 (standing 2026-06-11 discipline).
- **Hardware-Task Continuity:** exp4040 (one consolidated per-board task; KV260 toward terminal).
- **Self-learning:** exp4039 (ArcMemo v6 concept-library).
- **Adversarial / Sample-size rigor:** every compute-bound task declares `inference_substrate`,
  `model_specs`, `random_seed`, `reproducibility_checksum`; statistical claims require N≥30 + a
  bootstrap CI + a positive control (FALSE_NEGATIVE_RISK guard).
- **Pre-Launch Preconditions:** every compute-bound task opens with a PRECONDITIONS block (GGUF
  cache, llama_cpp, corpus loadable, sandbox importable, board SSH).
- **Verdict terminal-prefix + principle-annotated fields:** enforced on every task.
- **prior_failures:** exp4037 (decentralization) cites the flagged exp4022 with
  `retire_if_same_verdict: true`. operator_override on the routine continuations (archive,
  7th-game, ArcMemo, hardware).
