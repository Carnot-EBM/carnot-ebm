# Research Roadmap — Milestone 2026.05.325 (Depth-Over-Breadth XI)

**Status:** Proposed (pre-staged roadmap, Opus 4.8 planner, 2026-05-31)
**Milestone:** `2026.05.325`
**Predecessor:** `2026.05.324` (Depth-Over-Breadth X)
**North star:** `ops/north-star.md` — headline = FoVer AUROC 0.9131 (G1 met);
sole unmet publication gate = **G2** (independent external reproduction).

---

## 1. What the previous milestone proved (and what it left fragile)

`.324` took P0.1's first clean positive (.323's Sudoku result) and tried to make it
*defensible* and *general* while fixing its broken sibling. Read honestly via
`scripts/summarize_artifact.py`, the verdicts are sharper than the titles:

| Exp | Scope | Verdict | The catch |
|---|---|---|---|
| exp3517 | Route 1 — Sudoku hardening | POSITIVE, `solve_rate=1.0`, AR-greedy `0.025`, PT recovered to `0.525` | **CEILING_SATURATION**: `discrete_sa_single` also = 1.0 on EVERY tier. No hard tier where trivial search fails → energy-specific power not shown. |
| exp3518 | Route 1 — graph-coloring generalization | GENERALIZES, `1.0` vs AR `0.5` | **CEILING_SATURATION**: `vanilla_descent` also = 1.0 all tiers; `pt_swap=0.0`. Only proves greedy-AR's Brooks pathology; any non-greedy method wins. NOT headline-eligible. |
| exp3519 | Route 2 — energy reranker fix | DE-DEGENERATED, distinct selections (`flip_count_process=24`) | **FALSE_NEGATIVE_RISK**: oracle `0.475` ≤ SC `0.5`. Corpus has NO selectable headroom → no method could win. Null is uninformative. |
| exp3520 | Step→final aggregation | **CLEAN, REAL MECHANISM** | `best_agg_auroc=0.9055` vs `unagg=0.7192`; shuffle control collapses to `0.4524`. A genuine secondary positive. |
| exp3521 | FR-11 self-learning rule | **CLEAN, DEPLOYABLE** | `conservative_default_beta` is the robust Phase-5 default; no over-regularization. |

**The diagnosis:** P0.1's Route 1 positive is real but **ceiling-saturated on BOTH CSPs** —
every instance is trivially solvable by vanilla descent, so the claim collapses to "any global
optimizer beats greedy-AR," not "energy inference is uniquely capable." Route 2 is
**de-degenerated but headroom-starved** — the reranker now makes distinct selections, but the
test corpus contains no selectable minority-correct answers (oracle ≤ SC even at n=80, exp3516).
Two genuine positives survive and should be promoted: the step→final aggregation mechanism
(exp3520) and the conservative-default self-learning rule (exp3521).

## 2. The three biggest gaps between current state and the PRD vision

1. **The P0.1 existential claim is not yet discriminating.** "Energy-based global inference solves
   what autoregressive generation cannot" is the load-bearing premise for the Phase-3 foundation
   model. The .324 evidence is ceiling-saturated: it does not separate *energy inference* from
   *any non-greedy optimizer*, and it does not exercise a hard tier where trivial local search
   fails. **Gap: a HARD, headroom-preserving corpus + a STRONG non-AR baseline.**
2. **The product-relevant Route 2 (energy-vs-SC on NL math) has never had a fair test.** Every
   attempt has run on a corpus with no selectable headroom (oracle ≤ SC). **Gap: a difficulty-
   matched corpus where oracle STRICTLY exceeds SC**, so a working reranker can be falsified or
   confirmed.
3. **Two real positives are not yet headline-eligible.** exp3520's aggregation mechanism and
   exp3521's self-learning default are clean but single-corpus / single-shot. **Gap: replicate at
   n≥80 with multi-seed CI (exp3520) and deploy end-to-end (exp3521).**

## 3. Milestone design — 12 tasks, 5 phases, cascade-proof

This milestone **carries the proven .322/.323/.324 architecture** and adds nothing experimental
to the orchestration:

- **Agent routing:** all 12 tasks are PLANNED `agent_type: claude` — required to pass the
  MODEL_AGENT_COHERENCE pre-activation gate audit (`scripts/experiment_1152_gate_audit`), which
  does not allow `agent_type: gemini`. This replicates the proven-good `.324` workflow exactly:
  `.324` was planned all-claude (passed the audit) and the outer-loop rerouted 9 tasks to gemini AT
  ACTIVATION once gemini-cli 0.44.0 was confirmed up. Per Gemini-Default + the 2026-05-31
  known-issues entry, the outer-loop SHOULD reroute the 11 MECHANICAL tasks to gemini at activation;
  exp3531 (Route-2 reranker debugging + paired-significance + honest framing) is the SOLE genuine-
  judgment task and stays claude.
- **NO `model: opus` anywhere** (the opus thinking-400 killed .321's builder and .322's first G2).
- **Cascade-proof:** NO depth task is `gated_on` another depth task. exp3531 depends on exp3530's
  corpus but READS it and blocks honestly (`blocked_no_headroom_corpus`) rather than gating, so a
  depth-task retirement cannot cascade-block it. The synthesis (exp3537) is **UNGATED** (reads &
  skips absent/flagged artifacts); only the capstone (exp3538) gates on the synthesis-ready flag.
- **Per-iteration progress flush + hard wall-clock budget** on every loop (defeats the 1201s
  idle-timeout that hit .321/.322).
- **Anti-tautology seed discipline:** aggregation/ops tasks set `random_seed=20260531` (a fixed
  value, never the experiment number); measurement tasks set a CONTENT-DERIVED seed; NEVER store a
  reference value bit-identical to a measured field (the exp3508 flag); the reranker must make
  distinct-from-SC selections at runtime or block honestly (the exp3519 de-flag).

Depth-Over-Breadth does NOT relax: P0.1 is ceiling-saturated (Route 1) and headroom-starved
(Route 2); G2 is not externally run. No `vN+1` re-measurement — every task answers a question its
predecessor structurally could not.

### Phase A — OPS transition
- **exp3527** — Archive `.324`, write the operational retro, activate `.325`. [gemini]

### Phase B — DEPTH BLOCK (majority of slots; no cross-gating)
- **exp3528** — **Route 1 graph-coloring HEADROOM re-test (the operator priority).** Build a HARD
  k-coloring corpus near the freezing transition where `vanilla_descent_solve_rate < 0.9` but
  `exact_baseline = 1.0` (headroom exists, instances solvable); n≥30 stratified; compare energy
  global inference vs (a) greedy-AR AND (b) a STRONG non-AR baseline (DSATUR / tuned SA-restarts /
  backtracking-with-heuristic); fix PT so `pt_swap_acceptance_rate > 0` or report which optimizer
  did the work. Defeats the exp3518 CEILING_SATURATION. [gemini]
- **exp3529** — **Route 1 Sudoku HEADROOM re-test.** Same defeat-ceiling for Sudoku: harder tiers
  (minimal-clue / near-uncolorable) where `discrete_sa_single < 0.9` so the energy-specific
  capability (PT / restarts / IRED-adaptive) is visible, not just "any optimizer beats AR." Defeats
  the exp3517 CEILING_SATURATION. [gemini]
- **exp3530** — **Route 2 selectable-headroom NL-math corpus build (live GPU).** Construct/filter a
  MATH/AIME-style corpus where the oracle (best achievable by selecting among the k candidates)
  STRICTLY exceeds the SC majority — i.e., minority-correct answers are present and selectable
  (`oracle_exceeds_sc == true`, target headroom ≥ +0.05). The positive control the FALSE_NEGATIVE_
  RISK guidance demands before any energy-vs-SC verdict. [gemini, GPU]
- **exp3531** — **Route 2 energy/step-aggregation-vs-SC on the headroom corpus.** Using exp3520's
  CONFIRMED step→final aggregation as the scorer (+ a pessimistic-BoN variant, arXiv:2604.04648),
  re-run the energy-vs-SC comparison on exp3530's corpus; flip-count + delta + McNemar + bootstrap
  CI. Reads the corpus and blocks honestly if `oracle ≤ SC` (no gate → cascade-proof). The first
  fair test of the product-relevant Route 2. [claude]
- **exp3532** — **Promote the step→final aggregation positive toward headline-eligibility.**
  Replicate exp3520's `min`/aggregation mechanism (AUROC 0.9055, shuffle-confirmed) at n≥80 with
  ≥5 seeds + CI95 + a held-out split, on distinct pipelines (no reference==measured). [gemini]
- **exp3533** — **Deploy the conservative-default self-learning rule (mandatory continuous
  self-learning + P0.2).** Run the FR-11 closed loop to N≥200 on a FRESH corpus with exp3521's
  conservative-default β; confirm end-to-end it prevents collapse where β=0 collapses, and report
  the deployed α_t-grounding margin. [gemini]

### Phase C — G2 (the sole publication gate)
- **exp3534** — Clean-room regression-verify the self-contained FoVer package (drift check after
  `.325` changes); refresh the one-click external-ask. NEVER pushes/triggers; G2 stays
  operator-gated (`g2_met=false`, `external_run_pending=true`). [gemini]

### Phase D — HARDWARE (opportunistic continuity per north-star §3; minimal)
- **exp3535** — KV260 terminal board-level latency transcript (SSH precondition only; drive to
  terminal then freeze). [gemini]
- **exp3536** — PolarFire opportunistic reachability + continuity audit (distinct fields). [gemini]

### Phase E — SYNTHESIS (cascade-proof, seed-fixed)
- **exp3537** — G1–G4 gate-status synthesis v325 (UNGATED; reads & skips absent/flagged). [gemini]
- **exp3538** — Capstone v325 (gated only on `gate_status_v325_ready == true`). [gemini]

## 4. Dependency graph

```
exp3527 (archive/activate)
   │
   ├─ exp3528  Route1 graph-coloring headroom ─┐
   ├─ exp3529  Route1 Sudoku headroom          │  (no cross-gating;
   ├─ exp3530  Route2 headroom corpus build    │   each independent,
   │             │ (read-not-gate)             │   blocks honestly)
   ├─ exp3531  Route2 energy-vs-SC ────────────┤
   ├─ exp3532  aggregation promote             │
   ├─ exp3533  self-learning deploy            │
   ├─ exp3534  G2 regression                   │
   ├─ exp3535  KV260                           │
   └─ exp3536  PolarFire ──────────────────────┘
                       │
              exp3537  G1–G4 synthesis (UNGATED — reads & skips absent/flagged)
                       │  gate_status_v325_ready == true
              exp3538  capstone v325
```

## 5. Hardware requirements

- **CPU only** for exp3528/3529 (Ising/QUBO global-opt ladders), exp3531/3532/3533 (cached
  verifier scoring + closed loop), exp3534 (fresh-venv regression), all synthesis/ops.
- **CUDA (RTX 3090)** for exp3530 only (live SOTA-GGUF generation to build the headroom corpus).
  Gated by a PRECONDITIONS block; blocks honestly (`blocked_cuda_unavailable`) if GPU absent.
- **SSH-attached boards** for exp3535 (KV260) / exp3536 (PolarFire); SSH-reachability preconditions
  only (KV260 SSH-Not-SD-Card Discipline).

## 6. SOTA models

exp3530 (the only live-LLM task) uses the mandated SOTA GGUFs via the embedded-tokenizer GGUF path
(NOT `AutoTokenizer` on a `-GGUF` repo id): default `unsloth/gemma-4-26B-A4B-it-GGUF`, fallback
`unsloth/gemma-4-31B-it-GGUF` / `unsloth/Qwen3.6-35B-A3B-GGUF`.

## 7. Success criteria

- **Route 1 discriminating:** exp3528/3529 produce a HARD tier where the trivial baseline is below
  ceiling AND energy global inference beats a STRONG non-AR baseline — or an honest negative that
  bounds the claim (the energy-specific advantage does not survive a strong baseline).
- **Route 2 fair test:** exp3530 yields a corpus with `oracle > SC`, and exp3531 returns a
  defensible verdict (energy aggregation beats SC, or an honest negative WITH headroom present —
  no longer a FALSE_NEGATIVE_RISK).
- **Two positives promoted:** exp3532 replicates the aggregation AUROC at n≥80 with CI; exp3533
  confirms the conservative-default rule deploys end-to-end.
- **G2 stays bulletproof + operator-gated.**
- **Convergence signal:** the capstone reports `unmet_gates` (not a count) and whether
  Depth-Over-Breadth can relax (P0.1 clean+defensible AND G2 external-in-motion).

## 8. Cross-references

- `ops/north-star.md` §1 (headline), §2 (G1–G4 gate), §3 (hardware focus)
- `ops/known-issues.md` "NEW 2026-05-31: P0.1 GRAPH-COLORING RE-TEST ON A HARD, HEADROOM-PRESERVING
  CORPUS" (the operator priority exp3528 implements) + "KONA GLOBAL-OPT CORRECTNESS-FIRST GATE"
- `research-references.md` "2026-05-31 Post-.324 Planning Sweep" (freezing-transition corpus,
  MoB, pessimistic BoN, ThinkPRM)
- CLAUDE.md "Depth-Over-Breadth Forcing Function", "Adversarial Artifact Verification +
  Sample-Size Rigor" (CEILING_SATURATION, FALSE_NEGATIVE_RISK), "Gemini-Default"
- exp3505/3517/3518 (Route 1), exp3519/3516 (Route 2 corpus), exp3520 (aggregation), exp3521
  (self-learning), exp3510 (G2)
