# Research Roadmap — Milestone 2026.05.317

**Depth-Over-Breadth III: Finish the Existential Block Cleanly (P0.1 v3 real
harness, G2 clean-room root-cause, P0.2 / Kona / injection OFF gemini)**

**Planner:** Opus 4.8, 2026-05-30 (highest-leverage call per CLAUDE.md
Depth-Over-Breadth Forcing Function — planner is Claude, not gemini, precisely
so this prose discipline is followed reliably).

---

## 1. What the previous milestone (.316) proved — and what it did NOT

`.316` (Depth-Over-Breadth II) aimed at the right targets but **the existential
block is still not clean**, and a systemic infrastructure failure was the
proximate cause for 3 of 5 depth tasks.

| .316 task | Outcome | Status |
|---|---|---|
| P0.1 v2 energy-vs-SC (exp3426) | Ran 642 s, n=200, k=5, **not** flagged — but **every k-sample condition = 0.0 accuracy** (SC, self-certainty BoN, energy-argmin, energy-vote all 0.0) vs greedy AR 0.75. Verdict "energy matches SC" is a **degenerate 0.0-vs-0.0 tie**. | ❌ broken harness — P0.1 unanswered |
| P0.2 diversity (exp3427) | **No artifact** — gemini-cli crashed 3× | ❌ never ran (2nd milestone) |
| Kona solve-rate (exp3428) | **No artifact** — gemini-cli crashed 3× | ❌ never ran (2nd milestone) |
| Ensemble-vs-injection (exp3429) | **No artifact** — gemini-cli crashed 3× | ❌ never ran (2nd milestone) |
| G2 clean-room (exp3430) | **FAILED** — fresh worktree + fresh venv (py3.14.5 / numpy2.4.6 / jax0.10.1): headline did NOT reproduce (`reproduced_in_ci=false`) | ❌ G2 regressed in clean env |
| KV260 terminal (exp3431) | `blocked_kv260_ssh_unreachable` | ⏸ operator action (board power) |
| GateMate (exp3432) | Flow ran, **not flashed**; verdict-fix applied | ⏸ opportunistic |
| Gate synthesis (exp3434) | g1=T g2=F g3=T g4=T; **depth_relax=False** | ✅ |

**Two root causes, both fixable at the planner layer:**

1. **gemini-cli is down.** `ops/conductor-log.md` shows every `agent_type:
   gemini` task in .315 and .316 failing 3× with `Gemini CLI error: Stalled
   after 600s silence` or `Gemini CLI error: .js:345500:14)`. Every
   `agent_type: claude` task landed. The Gemini-Default rule's positive
   criterion for `requires_claude: true` is met ("gemini has demonstrably
   failed at this specific task category in a prior milestone" — cite
   exp3427/3428/3429). **.317 routes all depth tasks to `claude`.**

2. **P0.1 v2's multi-sample harness silently produced 0.0** for every sampled
   condition. The greedy path worked; the k-sample sampling/extraction/vote
   path did not. P0.1 v3 must make the SC baseline itself non-degenerate
   (SC accuracy ≥ greedy AR on GSM8K, the well-known result) before any energy
   comparison is trusted.

**Depth-Over-Breadth Forcing Function remains ACTIVE** (CLAUDE.md, 2026-05-30):
it relaxes only once P0.1 has a CLEAN verdict AND G2 has a concrete in-flight
reproducer. Neither holds. `.316`'s `depth_relax=False` is correct.

---

## 2. The three biggest gaps between current state and the PRD vision

1. **The Phase-3 premise is unproven (P0.1).** The entire foundation-model
   endgame rests on "energy-descent reasoning on continuous latents beats token
   sampling." After 200+ milestones it has never been cleanly measured. v2's
   harness broke. **Gap-closer: P0.1 v3 with a verified-non-degenerate SC
   baseline and a real energy-weighted vote, matched compute.** The sharpened
   bar (arXiv:2410.12608): a program-verifier already beats SC, so energy must
   clear "beats SC," not merely "beats greedy AR."

2. **The sole publication gate (G2) does not reproduce in a clean environment.**
   The headline reproduces in the operator's working venv but FAILED in a fresh
   worktree (py3.14/numpy2.4/jax0.10). Until a clean clone reproduces, no
   external reproducer can close G2. **Gap-closer: root-cause + fix the
   clean-room failure, then re-run the isolated reproduction.**

3. **Continuous self-learning's keystone (α_t grounding / P0.2) is unmeasured at
   production k.** FR-11 self-improvement is sound only if the verifier ensemble
   has real diversity (small joint null space). exp1224 showed k=3 collapse;
   exp2837 showed 3-of-4 FoVer verifiers contributing zero. The audit has been
   displaced/blocked for two milestones. **Gap-closer: run the λ_min(Σ) /
   effective-k audit on `claude`.**

---

## 3. Milestone architecture

```
 PHASE A — ops transition
   exp3436  archive .316 / activate .317                         [claude, 20t]

 PHASE B — DEPTH BLOCK (the majority; all on claude — gemini-cli is down)
   exp3437  P0.1 v3  energy-vote vs SC, REAL multi-sample harness [opus, 100t] *crux*
   exp3438  G2 clean-room ROOT-CAUSE + fix + re-isolated run      [opus, 80t]  *sole gate*
   exp3439  P0.2  verifier lambda_min(Sigma) / effective-k audit  [claude,50t]
   exp3440  Kona  global-opt solve-rate gate (correctness-first)  [opus, 60t]
   exp3441  Ensemble vs adaptive prompt-injection corpus          [claude,50t]

 PHASE C — HARDWARE (light + opportunistic; north-star §3)
   exp3442  KV260 terminal latency transcript (ssh-gated)         [opus,100t]
   exp3443  GateMate opportunistic detect + continuity            [claude,20t]
   exp3444  PolarFire opportunistic reachability                  [claude,20t]

 PHASE D — ops synthesis + capstone
   exp3445  G1-G4 gate-status synthesis (clean depth verdicts)    [claude,30t]
   exp3446  Capstone v317                                         [claude,20t]
```

11 tasks. **5 substantive depth tasks (B)** = the majority of non-ops,
non-hardware slots — satisfies Depth-Over-Breadth. **NO vN+1 re-measurement of
an already-measured artifact** appears: every depth task either fixes a broken
measurement (P0.1 v3, G2 clean-room) or runs a measurement that has never
produced an artifact (P0.2, Kona, injection — all gemini-crash victims).

### Dependency graph

```
exp3436 (activate) ─┬─> exp3437 P0.1 v3 ────────────────┐
                    ├─> exp3438 G2 clean-room            │
                    ├─> exp3439 P0.2                     ├─> exp3445 gate-synthesis ─> exp3446 capstone
                    ├─> exp3440 Kona                     │   (gated_on exp3437 verdict
                    ├─> exp3441 injection                │    contains 'complete')
                    ├─> exp3442/3443/3444 hardware ──────┘
```

The depth tasks are mutually independent; only the synthesis (exp3445) gates on
exp3437's terminal verdict, and the capstone (exp3446) gates on exp3445.

---

## 4. Why each depth task is depth, not breadth

- **exp3437 (P0.1 v3)** — fixes a *broken measurement* of the single most
  load-bearing premise. The headline number (energy-vote − SC at matched
  compute) directly determines whether the Phase-3 endgame is justified. Either
  outcome is high value: beats SC → first real justification; doesn't → retire
  the energy-superiority framing.
- **exp3438 (G2 clean-room)** — directly advances the SOLE unmet publication
  gate, which actively regressed in .316. Without a clean-env reproduction, G2
  cannot be closed by any external reproducer.
- **exp3439 (P0.2)** — the alpha_t-grounding keystone of continuous
  self-learning (PRD FR-11). Measures whether ensemble diversity survives at
  production k.
- **exp3440 (Kona)** — the cleanest concrete instance of P0.1 (does Ising energy
  *solve* hard Sudoku via global optimization?). Correctness-first gate.
- **exp3441 (injection)** — tests whether the k=15 cross-mechanism ensemble
  beats the single-KAN 0.475 floor (Spera joint-null-space theory).

Each maps to a G-gate or a load-bearing-unproven link, per the Depth-Over-Breadth
three-question test.

---

## 5. Continuous self-learning coverage

**exp3439 (P0.2 diversity audit)** is the continuous-self-learning experiment
this milestone (research-program.md "Continuous Self-Learning"): alpha_t
grounding is the precondition that lets the FR-11 self-correcting loop avoid
self-distillation collapse. If the ensemble's joint null space is too large
(effective-k < 3, lambda_min(Sigma) <= 0.1), the grounding signal is at risk and
the self-improvement thesis needs an adversarial-rotation fix.

---

## 6. Hardware requirements

Per north-star §3 (KV260 to terminal, then freeze; GateMate/PolarFire
opportunistic):

- **exp3437 / exp3440** need CUDA (RTX 3090) for live 35B GGUF inference and the
  Ising optimizer. SOTA model: `unsloth/Qwen3.6-35B-A3B-GGUF` (P0.1),
  `unsloth/gemma-4-26B-A4B-it-GGUF` (Kona hybrid proposal step).
- **exp3438 / exp3439 / exp3441** are CPU verifier-scoring (no GPU strictly
  required for the headline path; GPU used only for model-based verifiers).
- **exp3442** needs `ssh kria` reachability (operator action if board is off).
- **exp3443 / exp3444** need the GateMate DirtyJTAG / PolarFire SSH preconditions.

---

## 7. Process changes vs .316

1. **All experiment tasks route to `claude`** (gemini-cli is down — cite
   exp3427/3428/3429 3× crashes). This is the single highest-leverage process
   fix; it converts 3 perpetually-missing depth artifacts into landing tasks.
2. **P0.1 v3 hardens the SC baseline**: a PRECONDITION asserts the SC accuracy
   is non-degenerate (SC ≥ greedy AR on GSM8K, the textbook result) BEFORE any
   energy comparison is reported — making the 0.0-vs-0.0 failure mode impossible
   to ship as a verdict.
3. **G2 task is now root-cause-and-fix**, not just re-run, because the .316
   clean-room actively failed.

---

## 8. Acceptance / exit criteria

The milestone succeeds if:
- P0.1 v3 emits a CLEAN (non-degenerate, non-flagged) energy-vote-vs-SC number
  with paired significance, OR an honest blocked_* if a precondition fails.
- G2 clean-room either reproduces in-CI from an isolated env, OR identifies and
  fixes the concrete root cause of the .316 failure.
- P0.2 / Kona / injection each produce a real artifact (no longer gemini-crash
  victims).
- The gate synthesis recomputes `depth_forcing_function_can_relax` against the
  clean depth verdicts (expected: still False unless P0.1 lands clean AND G2
  has an in-flight external reproducer).

---

## 9. Cross-references
- `ops/north-star.md` §1 (headline), §2 (G1-G4 gate), §3 (hardware focus)
- `ops/known-issues.md` — P0.1 (exp3312), P0.2 (exp3313), Kona correctness gate
  (NEW 2026-05-30), ensemble-vs-injection (NEW 2026-05-28)
- `research-references.md` "2026-05-30 Post-.316 Planning Sweep" (the new
  energy-vote-vs-SC references: arXiv:2410.12608, 2510.17472)
- CLAUDE.md "Depth-Over-Breadth Forcing Function", "Gemini-Default for
  Experiments" (the `requires_claude` carve-out), "Adversarial Artifact
  Verification" (fabrication gate), "KV260 SSH-Not-SD-Card Discipline"
- exp3426 / exp3430 / exp3434 (the .316 outcomes this milestone supersedes)
