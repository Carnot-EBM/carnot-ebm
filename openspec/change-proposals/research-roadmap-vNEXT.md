# Research Roadmap — Milestone 2026.05.326 (Depth-Over-Breadth XII)

**Status:** PROPOSED (pre-staged by outer-loop Opus 4.8, 2026-05-31)
**Milestone:** 2026.05.326
**Predecessor:** 2026.05.325 (Depth-Over-Breadth XI)
**Milestone doc:** this file
**Roadmap YAML:** `research-roadmap-next.yaml`

---

## 1. What the previous milestone (.325) proved

`.325` attacked the two ways P0.1 (does energy-based GLOBAL inference actually
SOLVE / SELECT, not just descend fast?) was still not defensible. Read via
`scripts/summarize_artifact.py` (Reading-Results Discipline), the verdicts are
sharper than the titles:

| Exp | Question | Verdict | Status for .326 |
|-----|----------|---------|-----------------|
| 3528 | Route-1 graph-coloring: energy vs a STRONG non-AR baseline on a HARD, headroom-preserving corpus | **Scientifically POSITIVE** (solve_rate 1.0 vs DSATUR 0.956 vs greedy-AR 0.267; vanilla_descent 0.2 → real headroom; PT swap-accept 0.449) **but CRITICAL-FLAGGED and excluded from the headline** by a pure storage bug: two field pairs alias the same value (`calibration_vanilla_descent_solve_rate==vanilla_descent_solve_rate_hard_tier`=0.1333; `pt_mean_swap_rate==pt_swap_acceptance_rate`=0.4488) → TAUTOLOGY | **RESCUE CLEANLY — #1 priority** |
| 3529 | Route-1 Sudoku: energy-power gradient on a discriminating tier | POSITIVE (solve_rate 1.0 vs single-SA 0.733; exact-CP 1.0). PT (0.289) underperformed single-SA — the power came from SA-restarts | Clean enough; not re-run |
| 3530 | Route-2: build a corpus where oracle > SC | **NEGATIVE — no headroom found** (0/24 kept; SC≈optimal even at MATH L4-5) → Route-2 premise BOUNDED on MATH | One genuinely-different attempt, or accept the bound |
| 3531 | Route-2: fair energy-vs-SC test | Informative-negative but STILL headroom-starved (oracle 0.473 ≤ SC 0.505) | Re-test only IF .326 finds headroom |
| 3532 | Promote the step→final aggregation positive (n≥80, multi-seed CI) | **CLEAN POSITIVE, REPLICATED** (AUROC 0.9234, CI [0.899, 0.948], n=93, shuffle collapses to 0.474) | Generalize CROSS-CORPUS |
| 3533 | Deploy the conservative-default self-learning rule end-to-end | Ran on a **degenerate corpus** (initial true_acc 0.0067) → "quality drops" is an artifact | Re-deploy on a NON-DEGENERATE corpus |
| 3534 | G2 regression-verify the FoVer package | CLEAN (reproduces 0.9131) | Keep current; G2 = sole unmet gate |

**Net:** P0.1's strongest datapoint (energy beats a STRONG non-AR baseline on a
non-saturated CSP) exists but is locked out of the headline by a trivial
duplicate-field tautology. The aggregation positive replicated. Route 2 is
structurally headroom-bounded on MATH. G2 (independent reproduction) remains the
SOLE unmet publication gate (`ops/north-star.md` §2).

**Depth-Over-Breadth does NOT relax:** P0.1 still lacks a clean, flag-free,
headline-eligible verdict (the graph-coloring positive is flag-excluded), and G2
has no external run in motion. `.326` stays majority-depth.

---

## 2. The three biggest gaps (PRD vision vs current state)

1. **P0.1 has no clean, headline-eligible "energy global inference is uniquely
   capable" result.** The graph-coloring result that would BE that datapoint is
   flag-excluded by a storage bug. Closing this is the single highest-leverage
   action toward the foundation-model (Kona-parity) endgame — it is the empirical
   evidence that non-autoregressive energy reasoning can SOLVE where AR cannot.
2. **The one surviving headline (FoVer 0.9131) has no independent reproducer
   (G2).** Everything is in place (self-contained package, SHA256/CID, one-click
   workflow); the gate closes only on a non-operator external run. `.326` keeps
   the package regression-clean and the ask current — submission stays
   operator-only.
3. **The self-learning thesis (PRD FR-11 / continuous self-learning) lacks an
   end-to-end deployment on a corpus where output quality is actually
   measurable.** `.325` deployed the rule but on a degenerate corpus, so
   "quality maintained" was vacuous.

---

## 3. Architecture of the milestone

```
PHASE A — OPS transition
  exp3539  archive .325 / activate .326  (seed 20260601)

PHASE B — DEPTH BLOCK (majority of slots; no cross-gating; cascade-proof)
  exp3540  Route-1 graph-coloring CLEAN RE-RUN (de-tautology + expand n + CI + significance)   [CRITICAL]
  exp3541  Route-2 selectable-headroom corpus build — GENUINELY DIFFERENT construction (GPU)
  exp3542  Route-2 fair energy/aggregation-vs-strong-SC test — READS exp3541's corpus, blocks honestly
  exp3543  Promote aggregation positive CROSS-CORPUS (generalization → secondary headline)
  exp3544  FR-11 self-learning deploy on a NON-DEGENERATE corpus  [mandatory continuous self-learning]

PHASE C — G2 (the sole publication gate)
  exp3545  FoVer G2 regression-verify + external-ask refresh (no push, no CI, operator-gated)

PHASE D — HARDWARE (opportunistic per north-star §3; minimal)
  exp3546  KV260 terminal latency transcript (SSH precondition)
  exp3547  PolarFire opportunistic reachability + continuity audit

PHASE E — SYNTHESIS (UNGATED synthesis, cascade-proof; capstone gates on synthesis-ready)
  exp3548  G1–G4 gate-status synthesis v326 (UNGATED; reads & skips absent/flagged; seed 20260601)
  exp3549  Capstone v326 (gated_on exp3548.gate_status_v326_ready==true; seed 20260601)
```

### Dependency graph (cascade-proof)

```
exp3539 (ops) ─ independent
exp3540 (graph coloring) ─ independent (pure CPU; re-asserts encoding + hardness preconditions)
exp3541 (headroom corpus, GPU) ─ independent
exp3542 (route-2 test) ─ READS exp3541's corpus FILE; NOT gated_on it → blocks honestly if absent/no-headroom
exp3543 (aggregation cross-corpus) ─ independent (cached scoring)
exp3544 (self-learning) ─ independent (cached traces)
exp3545 (G2) ─ independent
exp3546/3547 (hardware) ─ independent
exp3548 (synthesis) ─ UNGATED by design (the .321 cascade lesson); reads & skips absent/flagged
exp3549 (capstone) ─ gated_on exp3548.gate_status_v326_ready==true (the ONLY cross-gate)
```

No depth task is gated on another depth task. The `.321` cascade (synthesis
GATE_BLOCKed because it gated on a fragile crux that retired) is structurally
prevented: the synthesis is UNGATED and skips absent/flagged artifacts.

---

## 4. The .326 rules (carried from the proven .322–.325 architecture)

1. **AGENT ROUTING.** All 11 tasks are PLANNED `agent_type: claude` +
   `requires_claude: true` — REQUIRED to pass the `MODEL_AGENT_COHERENCE`
   pre-activation gate audit (`scripts/experiment_1152_gate_audit`), which does
   NOT allow `agent_type: gemini`. This replicates the proven .324/.325
   workflow: the outer-loop REROUTES the mechanical tasks
   (3539/3540/3541/3543/3544/3545/3546/3547/3548/3549) to gemini AT ACTIVATION
   per Gemini-Default + the .325 reroute precedent, once gemini-cli 0.44.0 is
   confirmed up. **exp3542** (Route-2 fair test: reranker/scorer integration +
   paired-significance + honest framing of a possibly-positive-or-informative-
   negative result) STAYS claude — the sole genuine-judgment task.
2. **NO `model: opus` anywhere** (the opus thinking-400 killed .321's builder and
   .322's first G2). Default Sonnet via the claude routing.
3. **CASCADE-PROOF** (see §3): no depth task gated on another depth task; the
   synthesis is UNGATED; only the capstone gates on the synthesis-ready flag.
4. **PER-ITERATION progress flush + hard wall-clock budget** on every loop
   (defeats the 1201s idle-timeout; exp3530/3517 each ran >1200s).
5. **ANTI-TAUTOLOGY (reinforced by the exp3528 lesson).** Aggregation/ops/
   hardware/G2 tasks set `random_seed = 20260601` (a distinct fixed value, NOT
   the exp number). Measurement tasks set a CONTENT-DERIVED seed. **NEVER store
   the same measured quantity under two field names** — the `.325` graph-coloring
   positive was excluded from the headline solely because
   `calibration_vanilla_descent_solve_rate==vanilla_descent_solve_rate_hard_tier`
   and `pt_mean_swap_rate==pt_swap_acceptance_rate`. References go ONLY in
   `methodology_note` strings, never in a numeric field equal to a measured one.
   The corpora must NOT be ceiling-saturated (vanilla baseline < 0.9).

---

## 5. Hardware requirements

- **exp3541** (Route-2 corpus build): CUDA (RTX 3090) + a SOTA GGUF via the
  llama.cpp path (embedded tokenizer; NEVER `AutoTokenizer` on a `-GGUF` repo id,
  per the 2026-05-29 GGUF tokenizer rule). Default
  `unsloth/gemma-4-26B-A4B-it-GGUF`; fallback 31B / Qwen3.6-35B. Blocks honestly
  if CUDA/model unavailable.
- **exp3540 / 3542 / 3543 / 3544**: pure CPU (`JAX_PLATFORMS=cpu`).
- **exp3546** (KV260): SSH reachability to `kria` only (KV260 SSH-Not-SD-Card
  Discipline — host `/dev/mmcblk*` checks are forbidden). **exp3547** (PolarFire):
  SSH reachability to `polarfire`.

---

## 6. Continuous self-learning (mandatory per research-program.md)

**exp3544** is the milestone's continuous-self-learning experiment (Tier 1/3 of
the self-learning architecture; PRD FR-11; Phase-5; Zenil α_t grounding; Q12 Dark
Room). It re-deploys the conservative-default β rule (selected in exp3521,
deployed-but-on-a-degenerate-corpus in exp3533) end-to-end on a NON-DEGENERATE
corpus (starting true accuracy ~0.3–0.6) so both collapse-prevention AND
quality-maintenance are meaningfully measurable.

---

## 7. Acceptance gates (per task)

Each task carries falsifiable acceptance gates with `principle:` annotations
(see the YAML). The milestone's load-bearing gates:

- **exp3540 (the critical task):** `vanilla_descent_solve_rate < 0.9` (headroom
  preserved) AND `solve_rate > strong_baseline_solve_rate` with a bootstrap CI on
  the difference AND zero CRITICAL adversarial flags (no aliased fields). An
  honest negative (energy ties/loses to the STRONG baseline once n is large and
  fields are clean) is equally valuable and bounds the claim.
- **exp3543:** aggregation AUROC holds (CI lower bound > unaggregated floor) on a
  corpus DIFFERENT from exp3532's, with the shuffle control collapsing.
- **exp3544:** deploy-arm prevents collapse to N≥200 while β=0 collapses, AND
  quality is maintained on a corpus where starting true accuracy is non-trivial.
- **exp3548:** emits `g1..g4` + `unmet_gates` (NOT a count); sets
  `depth_forcing_function_can_relax` only when P0.1 has a clean, flag-free,
  headline-eligible verdict AND G2 external-in-motion.

---

## 8. Cross-references

- `ops/north-star.md` — headline claim (§1), G1–G4 gate (§2), hardware focus (§3)
- `ops/known-issues.md` — "NEW 2026-05-31: P0.1 GRAPH-COLORING RE-TEST", "KONA
  GLOBAL-OPT CORRECTNESS-FIRST GATE", the P0.1 chain
- `research-references.md` — "2026-05-31 Post-.325 Planning Sweep" (MoB no-headroom
  insight, ranked-voting SC, multi-state Ising graph coloring, ML-enhanced MC)
- CLAUDE.md — "Depth-Over-Breadth Forcing Function", "Adversarial Artifact
  Verification + Sample-Size Rigor" (the TAUTOLOGY check exp3528 tripped),
  "Failed-Experiment Rerun Discipline", "Paper-v6 Narrowing Discipline"
- `.325` artifacts: exp3528/3529/3530/3531/3532/3533/3534
