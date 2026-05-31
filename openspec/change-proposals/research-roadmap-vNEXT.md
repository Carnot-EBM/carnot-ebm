# Research Roadmap — Milestone 2026.05.327 (Depth-Over-Breadth XIII)

**Status:** staged (pre-activation)
**Planner:** Claude Opus 4.8, 2026-05-31
**Predecessor:** 2026.05.326 (Depth-Over-Breadth XII)
**North star:** `ops/north-star.md` — one headline (FoVer 0.9131), one finish line (G1-G4; G2 sole unmet gate).

---

## 1. What the previous milestone proved (.326)

`.326` had one dominant goal — RESCUE P0.1's strongest datapoint (the .325 graph-coloring positive
that was excluded from the headline by a duplicate-field tautology) — and then it got an honest
answer. The milestone was **interrupted after exp3543**; only exp3539/3540/3542/3543 ran.

| Exp | Verdict (read via `summarize_artifact.py`) | Meaning |
|---|---|---|
| **exp3540** Route-1 graph-coloring CLEAN re-run | `complete: p01_energy_does_not_significantly_beat_strong_baseline_at_n60_advantage_was_small_sample_artifact` | **Rescue succeeded (de-tautologized, 0 CRITICAL flags) but the science is NEGATIVE.** Energy `solve_rate=1.0` vs STRONG DSATUR `0.99`, paired diff `0.01`, **p=0.135 (n.s.)**. Still crushes greedy-AR (0.085). `warn` CEILING_SATURATION: DSATUR at 0.99 -> corpus not hard enough to discriminate. |
| **exp3542** Route-2 energy-vs-strong-SC | `complete: blocked_corpus_has_no_selectable_headroom_oracle_le_sc` | The greedy-wrong GPU corpus builder (exp3541) never got its turn; fell back to a no-headroom corpus. `flip_count=0`. Route 2 headroom-starved 4x (exp3507/3530/3531/3542). |
| **exp3543** cross-corpus aggregation | `complete: step_to_final_aggregation_generalizes_cross_corpus_transfer_auroc_08610_secondary_headline_eligible` | **CLEAN POSITIVE — strongest fresh result.** Frozen aggregation fit on A transfers to a DIFFERENT corpus B at **AUROC 0.861** (floor 0.749, within-corpus 0.906), shuffle collapses to 0.496. Secondary-headline-eligible. |
| exp3544/3545/3546/3547/3548/3549 | **NEVER RAN** (interrupt) | FR-11 non-degenerate self-learning, G2 refresh, KV260, PolarFire, synthesis, capstone — carried forward. |

**The honest P0.1 reading after .326:** energy-based global inference (PT/SA on the Ising encoding)
is *competitive with, but not superior to,* strong classical CSP solvers (DSATUR), and *decisively
beats* autoregressive generation. The Kona premise (energy-descent beats AR) holds against AR but
NOT against the best classical heuristic. This is the exact pattern the neural-CO critique
literature documents (arXiv:2502.03669, 2302.03602, 2112.12251). To turn this into a **terminal,
defensible** datapoint, .327 must re-test on a corpus where DSATUR is forced well below ceiling, so
the comparison actually discriminates.

---

## 2. The three biggest gaps (current state vs PRD vision)

1. **P0.1 lacks a clean TERMINAL verdict.** Route 1 came back negative but on a *ceiling-saturated*
   corpus (DSATUR 0.99), and Route 2 has never had a fair test (no headroom corpus ever built).
   Until P0.1 has a terminal verdict — positive OR honest-negative on a *discriminating* test — the
   Depth-Over-Breadth Forcing Function does not relax.
2. **The one fresh positive (cross-corpus aggregation) is single-pair.** exp3543 showed A->B transfer
   at AUROC 0.861, but a single source/target pair is not a secondary headline. It needs multi-seed
   CI on transfer and a THIRD corpus (A->{B,C}).
3. **G2 — the SOLE unmet publication gate — is still external-pending,** and the self-learning
   thesis (PRD FR-11) has never been deployed on a non-degenerate corpus (exp3533 ran on a
   true_acc~0 corpus).

---

## 3. Milestone design — 11 tasks, exp3550-exp3560

Architecture rules carried verbatim from the working .322-.326 chain (they are why those milestones
landed clean):

- **Agent routing:** all 11 tasks PLANNED `agent_type: claude` + `requires_claude: true` — REQUIRED
  to pass the `MODEL_AGENT_COHERENCE` pre-activation gate audit (no gemini). gemini-cli 0.44.0 is up;
  the outer-loop REROUTES the mechanical tasks to gemini AT ACTIVATION per Gemini-Default + the
  .325/.326 reroute precedent. exp3553 (Route-2 fair test) STAYS claude — the sole genuine-judgment task.
- **No `model: opus` anywhere** (opus thinking-400 killed .321's builder + .322's first G2).
- **Cascade-proof:** no depth task `gated_on` another depth task; exp3553 READS exp3552's corpus and
  blocks honestly rather than gating; the synthesis (exp3559) is UNGATED (reads & skips
  absent/flagged); only the capstone (exp3560) gates on the synthesis-ready flag.
- **Per-iteration progress flush + hard wall-clock budget** on every loop (defeats the 1201s idle-timeout).
- **Anti-tautology:** aggregation/ops/hardware/G2 tasks set `random_seed=20260601` (NOT the exp
  number); measurement tasks set a CONTENT-DERIVED seed; NEVER store the same measured quantity under
  two field names (the exp3528 exclusion cause); references go ONLY in `methodology_note` strings; CSP
  corpora must NOT be ceiling-saturated for the STRONG baseline (DSATUR < 0.9, not just vanilla < 0.9).

### Phase A — OPS transition
- **exp3550** — archive .326, activate .327.

### Phase B — DEPTH (P0.1 terminal + secondary headline; majority of slots)
- **exp3551** *(#1 priority, CPU)* — **P0.1 Route-1 graph-coloring TERMINAL.** Re-run on a corpus near
  the chromatic/freezing threshold where the STRONG baseline (DSATUR) is forced **< 0.9** while the
  exact solver confirms solvability (==1.0). Energy vs DSATUR vs exact vs greedy-AR, bootstrap CI +
  paired McNemar/bootstrap significance. If energy STILL ties/loses to DSATUR on a *discriminating*
  corpus -> **P0.1 Route-1 terminally bounded** (energy competitive-not-superior; only beats AR) and
  graph-coloring retires. If energy significantly wins -> the rescued positive. `retire_if_same_verdict: true`.
- **exp3552** *(live GPU)* — **P0.1 Route-2 greedy-wrong headroom corpus build** (now runnable: CUDA up).
  Keep problems where the GREEDY (temp-0) answer is WRONG but >=1 of k>=16 sampled candidates is CORRECT,
  so oracle STRICTLY > SC by construction. SOTA GGUF. If no headroom corpus can be built ->
  **Route-2 terminally bounded on NL-math.** `retire_if_same_verdict: true`.
- **exp3553** *(cached, the SOLE judgment task — stays claude)* — **P0.1 Route-2 fair test.** The fixed
  non-degenerate reranker + exp3520's confirmed step->final aggregation scorer + MoB + pessimistic-BoN,
  vs a STRONG ranked-voting SC, on exp3552's corpus. Cascade-proof: reads the corpus, blocks honestly
  if absent/no-headroom. `retire_if_same_verdict: true`.
- **exp3554** *(cached)* — **PROMOTE the cross-corpus aggregation secondary headline.** Take exp3543's
  A->B transfer (0.861) to a defensible result: multi-seed transfer CI + a THIRD corpus (A->{B,C}) with
  per-target shuffle controls, so the mechanism is shown corpus-general, not single-pair.
  `retire_if_same_verdict: false` (a promotion of a confirmed positive, not a doomed rerun).

### Phase C — SELF-LEARNING + GATE (carry-forwards from .326 interrupt)
- **exp3555** *(mandatory continuous-self-learning, cached)* — **FR-11 conservative-default deploy on a
  NON-DEGENERATE corpus** (starting true acc in [0.3, 0.6]). DEPLOY arm vs CONTROL (beta=0); measure
  collapse-prevention AND real-quality-maintenance (impossible on exp3533's true_acc~0 corpus).
  `retire_if_same_verdict: true`.
- **exp3556** *(cached)* — **G2 clean-room regression-verify** the self-contained FoVer package (drift
  check after .326/.327 changes) + keep the one-click external-ask current. NEVER pushes / triggers CI /
  marks G2 met (Operator-Only External Publication).

### Phase D — HARDWARE (opportunistic per north-star §3)
- **exp3557** — **KV260 terminal latency transcript** (SSH precondition; currently unreachable -> honest
  `blocked_kv260_ssh_unreachable`). Mandatory-until-terminal continuity.
- **exp3558** — **PolarFire opportunistic reachability + continuity audit** (strictly distinct fields).

### Phase E — SYNTHESIS
- **exp3559** *(UNGATED, cascade-proof)* — **G1-G4 gate-status synthesis v327.** Reads & skips
  absent/flagged; computes g1..g4 + unmet_gates; sets `depth_forcing_function_can_relax`.
- **exp3560** *(gated on exp3559's synthesis-ready flag)* — **Capstone v327.**

---

## 4. Dependency graph

```
exp3550 (archive/activate)
   |
   |- exp3551  Route-1 graph-coloring TERMINAL (CPU)
   |- exp3552  Route-2 greedy-wrong corpus build (GPU)
   |- exp3553  Route-2 fair test (reads exp3552 corpus, blocks honestly)   (no cross-gating)
   |- exp3554  aggregation cross-corpus PROMOTE (cached)
   |- exp3555  FR-11 non-degenerate self-learning (cached)
   |- exp3556  G2 regression-verify (cached)
   |- exp3557  KV260 (SSH)
   |- exp3558  PolarFire (SSH)
   |
   |- exp3559  G1-G4 synthesis v327 (UNGATED — reads & skips absent/flagged)
   \- exp3560  capstone v327 (gated_on exp3559.gate_status_v327_ready == true)
```

## 5. Hardware requirements

- **exp3551, 3553, 3554, 3555, 3556, 3559, 3560:** CPU only (`JAX_PLATFORMS=cpu`).
- **exp3552:** live GPU (2x RTX 3090, CUDA verified up) + SOTA GGUF via the embedded-tokenizer GGUF
  path (`llama_cpp`, NOT `AutoTokenizer` on a -GGUF repo id).
- **exp3557:** KV260 over `ssh kria` (currently unreachable — blocks honestly).
- **exp3558:** PolarFire over `ssh polarfire`.

## 6. Depth-Over-Breadth compliance

Every task advances the headline, closes a G-gate, or tests a load-bearing-unproven link. No `vN+1`
re-measurement of an already-answered question: exp3551 changes the corpus hardness (DSATUR < 0.9, the
discriminating test exp3540's saturated corpus could not be); exp3552 uses a genuinely different
construction on now-available GPU; exp3554 adds multi-seed CI + a THIRD corpus (generalization, not
replication); exp3555 uses a non-degenerate corpus (the question exp3533 could not answer). The
forcing function relaxes only when P0.1 has a clean terminal verdict on a discriminating test AND G2 is
external-in-motion — exp3559 computes that condition.
