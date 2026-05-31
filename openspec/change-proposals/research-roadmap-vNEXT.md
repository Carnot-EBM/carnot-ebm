# Research Roadmap — Milestone 2026.05.329 (Depth-Over-Breadth XV)

**CHARACTERIZE the regime, OPEN the code-reranking front, RETIRE Route-2 NL-math,
clean the aggregation, drive G2 — with ZERO critical science on a live-GPU path.**

**Status:** staged (pre-activation)
**Planner:** Claude Opus 4.8, 2026-05-31 (the gemini planner failed this cycle with
`thinking`/400 + a 1201s idle-timeout, so Opus planned directly).
**Predecessor:** 2026.05.328 (Depth-Over-Breadth XIV, CONSOLIDATION).
**Milestone doc format:** v7/v8.

---

## 1. What `.328` proved (and disproved)

`.328` tried to harden the `.327` P0.1 Route-1 positive (energy/Ising global inference
beats a STRONG classical solver + autoregressive greedy on graph coloring near the
chromatic threshold: `solve 0.9625 vs DSATUR 0.70`, hard-tier paired diff `+0.38`,
`p=0.000`) into a defensible general claim. Read via `scripts/summarize_artifact.py`:

| Exp | Goal | Verdict | Honest reading |
|---|---|---|---|
| **exp3562** | Generalize Route-1 to a SECOND discriminating CSP (k-SAT) | **BLOCKED** `cannot_construct_discriminating_second_csp` | Strong classical k-SAT solvers solve the hard tier `==1.0` — no headroom to discriminate; energy actually lost `-0.03`, `p=1.0`. A discriminating regime is hard to even *construct* off the coloring phase transition. |
| **exp3563** | Harden coloring: >=5 seeds + a second generator | **BOUNDED** `ci_includes_zero_on_second_generator` | Hard-tier paired-diff CI excluded 0 on generator 1, **included 0 on generator 2**. The positive is **fragile to the instance distribution**, not a robust headline. |
| **exp3564** | Route-2 NL-math final live-GPU attempt + terminal verdict | **NO ARTIFACT** — 3× `Gemini CLI Stalled after 600s silence` | The live-generation task never ran. The **gemini-600s-stall** is the #1 *operational* impediment to P0.1. Route-2 still has **no terminal verdict** after 5 headroom-starved blocks. |
| **exp3565** | Promote cross-corpus aggregation to a secondary headline | **FLAGGED (TAUTOLOGY)** `does_not_transfer` | Corpus C transfer healthy (`0.904`) but corpus B **degenerate** (`0.5` = floor = shuffle — a build bug). Promotion did not land. |
| **exp3566** | FR-11 multi-corpus deploy + P0.2 verifier diversity | **BOUNDED** `verifier_diversity_no_material_gain_p02_bounded` | FR-11 deploys across a non-degenerate battery, but diverse grounding (`0.479`) did not beat single (`0.490`). P0.2 bounded. |
| **exp3567** | G2 regression-verify | **CLEAN** | Package still reproduces FoVer `AUROC=0.9131`. **G2 is the SOLE unmet publication gate** (G1/G3/G4 met). |
| Hardware | KV260 / PolarFire continuity | KV260 `blocked_ssh_unreachable`; PolarFire reachable | KV260 board down; PolarFire opportunistic-clean. |

**Net state:** `depth_forcing_function_can_relax=true`. `G1=T, G2=F, G3=T, G4=T`.
The P0.1 existential positive is **honest but narrow** — energy global inference beats
strong-classical+AR specifically *near the graph-coloring phase transition on one
generator*, and does **not** generalize to a second CSP or a second generator. This is
exactly what the 2025-2026 neural-CO critique literature predicts (see §3).

---

## 2. The three biggest gaps (current state vs PRD vision)

1. **We have a bounded positive but no MAP of where it holds.** Chasing "does it
   generalize" (exp3562/exp3563) answered: *narrowly, and fragile to distribution*.
   The honest, publishable next move is not another CSP — it is to **characterize the
   hardness regime** in which energy global inference beats strong classical solvers,
   and concede the regime where it does not (the KaMIS lesson, arXiv:2502.03669).
2. **The reranker premise was tested only where it cannot win.** Route-2 attacked
   NL-math five times and found no selectable headroom because NL-math is single-basin
   (the correct answer IS the mode — ARBITER arXiv:2605.26172, MoB arXiv:2511.18630).
   The premise plausibly holds on **CODE** (multi-basin, functionally checkable, greedy
   often wrong) — and CODE is where Carnot's genuine surviving positives live
   (exp1999 `0.66→0.84`, exp2090 CRANE `0.70→0.85`). This front is unopened.
3. **G2 — the finish line — is the SOLE unmet gate, and it is external.** The loop
   cannot close G2 (Operator-Only External Publication), but every milestone it must
   keep the self-contained reproducer drift-free and the one-click external ask current.

**Operational gap (the operator's standing question — "get to the bottom of what
impedes scientific advancement"):** the #1 impediment is the **gemini-600s-stall on
live-GPU-generation tasks**, which retired Route-2 three times. The structural fix this
milestone: **put ZERO critical science on a live-GPU-generation path.** Every `.329`
depth task is CPU-Ising or cached-candidate scoring; the one task that could need fresh
generation (code candidates) uses the **resumable-checkpoint + per-problem-flush**
pattern that exp3448 proved defeats the idle-timeout.

---

## 3. Recent research integrated (2026-05-31 sweep)

Filed to `research-references.md` "2026-05-31 Post-.328 Planning Sweep".

- **Q1 — regime maps / neural-CO-vs-classical critique:** arXiv:2502.03669 (KaMIS beats
  AI on MIS even in-distribution — the adversarial baseline the regime map must concede
  to where it loses), arXiv:2508.02510 (the neural-vs-classical gap is *distribution-
  narrow* — our exp3563 fragility stated as a law), arXiv:2605.14624 (Amortized
  Efficiency Threshold — the honest framing for a bounded win: volume-amortized, not
  blanket solve-rate).
- **Q2 — code reranking:** arXiv:2604.06485 (SEP symbolic equivalence partitioning —
  CODE has selectable headroom majority vote misses), arXiv:2604.15618 (Functional
  Majority Voting — the honest baseline our code reranker must beat).
- **Q3 — selectable-headroom theory:** arXiv:2605.26172 (ARBITER reasoning basins —
  THE theory of when reranking can win: multi-basin yes, single-basin no), arXiv:2509.06870
  (AggLM minority-correct training split = a ready headroom-corpus protocol),
  arXiv:2502.18581 (self-certainty BoN baseline + headroom diagnostic).
- **Q4 — learned proposals for the generator-fragility fix:** arXiv:2509.23043
  (IsingFormer learned PT proposals that *transfer across instances* — the mechanism to
  make the Route-1 positive robust across generators), arXiv:2502.10328 (neural
  transports for PT).

---

## 4. Architecture (where `.329` touches the stack)

```
                          +---------------------------------------------+
   P0.1 Route-1 (CSP)     |  energy/Ising GLOBAL inference  vs classical |
   -- REGIME MAP ---------|  + autoregressive greedy                     |
   -- generator-robust ---|  (CPU; PT + SA + exact; learned proposals)   |
                          +---------------------------------------------+
                                            |  characterized, not chased
                          +---------------------------------------------+
   P0.1 Route-2 (reason)  |  verifier / energy RERANKER vs self-consist. |
   -- OPEN: CODE ---------|  CODE: multi-basin, functionally checkable    |
   -- RETIRE: NL-math ----|  NL-math: single-basin -> terminal negative  |
                          +---------------------------------------------+
                          +---------------------------------------------+
   Verification headline  |  FoVer 4-verifier ensemble -> AUROC 0.9131   |
   -- aggregation clean --|  step->final aggregation transfer (A->{B,C}) |
   -- G2 drive -----------|  self-contained reproducer (drift-free)      |
                          +---------------------------------------------+
                          +---------------------------------------------+
   Self-learning (FR-11)  |  conservative-beta grounding -> no-collapse  |
   -- distribution shift -|  robustness under a SHIFTED corpus           |
                          +---------------------------------------------+
   Hardware (opportunistic)  KV260 (SSH, until terminal) - PolarFire (audit)
```

---

## 5. Phases (12 experiments, exp3572-exp3583)

**Phase A — OPS transition (1):** exp3572 archive `.328` / activate `.329`.

**Phase B — DEPTH (6; the majority; no cross-gating, all CPU or cached):**
- **exp3573 — P0.1 Route-1 REGIME MAP (CPU, #1 priority).** Sweep graph-coloring
  instances across a hardness axis (proximity to the chromatic/freezing threshold) and
  measure the energy-vs-strong-classical advantage *as a function of hardness*. Output:
  the regime where energy wins, ties, and loses — with an amortized-efficiency framing.
- **exp3574 — P0.1 Route-1 generator-robustness (CPU).** Directly address exp3563's
  fragility: does a stronger / learned-proposal PT optimizer (IsingFormer-style global
  moves) make the hard-tier paired-diff CI exclude 0 on the second generator?
- **exp3575 — P0.1 Route-2 CODE reranking headroom (cached-first, stall-proof).** Open
  the front where headroom plausibly exists: build/score a code-candidate corpus where
  greedy is wrong and a correct candidate is present; score the Carnot energy/verifier
  reranker vs Functional Majority Voting + self-certainty BoN.
- **exp3576 — P0.1 Route-2 NL-math TERMINAL RETIREMENT (CPU synthesis, no GPU).** Give
  Route-2 NL-math its terminal verdict on the accumulated evidence (5 headroom-starved
  blocks + theory) — no live generation, so the 600s-stall cannot retire it again.
- **exp3577 — Aggregation secondary-headline CLEAN re-run (cached).** Fix the exp3565
  corpus-B degeneracy (non-degenerate B/C splits) and re-run A->{B,C} multi-seed, clean.
- **exp3578 — FR-11 self-learning ADVANCE (CPU, mandatory).** New question: does the
  conservative-beta deploy survive a *distribution shift* between the grounding corpus
  and the deployment corpus (not just depth-N)?

**Phase C — FINISH LINE (1):** exp3579 G2 regression-verify + external-ask refresh.

**Phase D — HARDWARE (2; opportunistic per north-star §3):** exp3580 KV260 SSH
continuity (until terminal), exp3581 PolarFire opportunistic audit.

**Phase E — SYNTHESIS (2):** exp3582 G1-G4 gate-status synthesis v329 (UNGATED,
cascade-proof, seed-fixed), exp3583 capstone v329 (gated on the synthesis-ready flag).

---

## 6. Dependency graph (cascade-proof)

```
exp3572 (ops) --> [all depth tasks activate]
exp3573  exp3574  exp3575  exp3576  exp3577  exp3578   (NO cross-gating; each
   |        |        |        |        |        |        re-asserts its own
   +--------+--------+--------+---+----+--------+        preconditions, blocks
                                  |                      honestly, never gates)
exp3579 (G2)  exp3580 (KV260)  exp3581 (PolarFire)      (independent)
                                  |
exp3582 (synthesis, UNGATED -- reads & SKIPS absent/flagged artifacts)
                                  |  gate_status_v329_ready == true
exp3583 (capstone -- the ONLY gated task)
```

No depth task is `gated_on` another depth task (the `.321` cascade failure mode). The
synthesis is UNGATED and skips absent/flagged inputs; only the capstone gates, on the
synthesis-ready flag.

---

## 7. The `.329` rules (carried from the working `.322-.328` architecture)

1. **AGENT ROUTING:** all 12 tasks PLANNED `agent_type: claude` + `requires_claude: true`
   to pass the `MODEL_AGENT_COHERENCE` pre-activation gate audit (no gemini). The
   outer-loop REROUTES the mechanical tasks to gemini at activation per Gemini-Default +
   the `.325/.326/.327` precedent. The two genuine-judgment tasks — **exp3575**
   (code-reranking judgment) and **exp3576** (honest terminal framing of a permanent
   retirement) — STAY claude.
2. **NO `model: opus` anywhere** (the opus thinking-400 killed `.321`/`.322` builders).
3. **ZERO critical science on a live-GPU-generation path.** Every depth task is CPU-Ising
   or cached-candidate scoring. exp3575 (the only task that could generate) is cached-first
   and, if it must generate, uses the resumable-checkpoint + per-problem-flush pattern
   (exp3448) that defeats the 600s/1201s idle-timeout.
4. **PER-ITERATION progress flush + hard wall-clock budget** on every loop.
5. **ANTI-TAUTOLOGY:** ops/aggregation/G2/hardware/synthesis/capstone set
   `random_seed=20260602` (NOT the exp number); measurement tasks use a CONTENT-DERIVED
   seed; never store the same measured quantity under two field names; references live
   ONLY in `methodology_note` strings; CSP corpora must NOT be ceiling-saturated for the
   STRONG baseline on the hard tier.

---

## 8. Hardware requirements

- **CPU only** for all six depth tasks + aggregation + FR-11 + G2 + synthesis + capstone.
- **GPU (RTX 3090)** only as a *fallback* for exp3575 if no cached code-candidate corpus
  exists — bounded, resumable, per-problem-flush; cached path preferred.
- **KV260** (SSH, `ssh kria`) and **PolarFire** (SSH, `ssh polarfire`) for the two
  opportunistic continuity audits. No host SD-card checks for KV260 (SSH-Not-SD-Card).

---

## 9. SOTA model policy

exp3575 (code reranking) — if it generates — uses a mandated SOTA GGUF via the GGUF path
(NOT `AutoTokenizer` on a `-GGUF` repo id): default `unsloth/gemma-4-26B-A4B-it-GGUF`,
fallback `unsloth/gemma-4-31B-it-GGUF` / `unsloth/Qwen3.6-35B-A3B-GGUF`. All other tasks
score cached candidates or run CPU-Ising and invoke no LLM.

---

## 10. Done criteria

- exp3573 emits a regime map: energy-vs-strong-classical advantage as a function of
  hardness, with the win/tie/lose bands named and an amortized-efficiency framing.
- exp3574 reports whether a stronger/learned PT optimizer makes the coloring positive
  robust across generators (CI excludes 0 on generator 2) — positive or honest bound.
- exp3575 reports whether the reranker beats Functional-Majority-Voting + self-certainty
  on a headroom code corpus — positive (a new code secondary-headline candidate) or
  honest bound (still no win even where headroom exists).
- exp3576 emits Route-2 NL-math's TERMINAL verdict (permanently retired as a trustworthy
  terminal negative, SC near-optimal on NL-math).
- exp3577 lands the A->{B,C} aggregation transfer CLEAN (no degenerate 0.5, no flag) or an
  honest bound.
- exp3578 reports whether the conservative-beta deploy survives a distribution shift.
- exp3579 keeps G2 regression-clean; G2 stays operator-gated (g2_met=false).
- exp3582/exp3583 emit the G1-G4 gate state + a narrowing-clean capstone; flagged
  artifacts excluded from every headline number.
