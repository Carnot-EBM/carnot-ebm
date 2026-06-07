# Research Roadmap — Milestone 2026.06.361

**PROVE THE VERIFIER EARNS ITS PLACE — complete the offline verifier proof
(ACCURACY moat + EFFICIENCY head-to-head), then start the agentic-proof scaffold.**

Outer-loop plan (Claude Opus 4.8, 2026-06-07).

---

## 1. What the previous milestone (.360) proved

.360 was HARNESS-FIRST: for each fabricating live-model bet, a separate
BUILD+UNIT-TEST task whose deliverable was a passing fixture test, THEN the
measurement. The harness-first split WORKED where it was applied, and surfaced
the real result the prior gate had hidden.

| Bet | .360 outcome | Trustworthy? |
|---|---|---|
| **Reasoner self-verify HARNESS (exp3894)** | **READY** — fixture AUROC **0.9167**, n_caught 6/6, clean 72s. The harness genuinely catches injected errors. | **YES** |
| **Verifier MOAT scissor (exp3895)** | **Mis-gated INCONCLUSIVE.** It actually *computed* `residual_catch_rate=0.905` (CI95 **[0.849, 0.952]**, n_residual=126), `error_overlap_jaccard=0.159`, `carnot_ensemble_auroc=0.967`. The gate returned INCONCLUSIVE **solely** because `reasoner_self_verify_auroc=0.546` fell **0.004 below** the 0.55 control floor. | **The NUMBERS are MOAT_SURVIVES; the GATE is wrong.** |
| **Facts graph-grounding harness (exp3896)** | **NOT_READY + FLAGGED** — 43.8s (DURATION_TOO_SHORT) and fixture AUROC=1.0 (separable fixture). The downstream facts run (exp3897) + complementarity (exp3898) never landed. | **NO — still fabricating.** |
| **EBT FUNDAMENTAL replication (exp3893)** | **Did not finish** — produced two training checkpoints (`seed4.pt`, `seed4_rerun1.pt`) but no adjudicated JSON; retrain-from-scratch-per-seed blew the wall-clock. | **N/A — superseded (see §2).** |

Standing mandates: GateMate (exp3900) + PolarFire/KV260 (exp3901) continuity
landed clean; FR-11 v25 (exp3899) did **not** land (no artifact); capstone exp3902
`paper_ready=TRUE`, frozen 0.9131 unchanged, **G1–G4 all met** (G2 closed by the
2026-05-31 GitHub-Actions independent reproducer).

**The decisive read:** the moat-scissor mis-gate is the headline of .360. The
`reasoner_self_verify_auroc=0.546` is **not a broken harness** (exp3894 proved the
harness at 0.917 on a clean fixture) — it is the **Self-Correction Illusion**
itself (arXiv:2606.05976): a strong reasoner can barely tell its own correct steps
from its incorrect ones in-distribution. With the harness validated, a residual
catch of **0.905 (lower CI 0.849)** at overlap **0.159** is exactly the
MOAT_SURVIVES signal. The gate conflated "the control harness is broken" with
"the reasoner genuinely can't self-verify" — they require opposite conclusions.

## 2. Strategic frame (north-star §5, 2026-06-06): energy VERIFIES, refinement GENERATES

Energy-as-GENERATOR is closed-negative multi-domain (Sudoku v1–v4 + exp3882 EBT
kill-gate + exp3883 K-curve + external NVIDIA-Ising-QEC corroboration) and
energy-as-SELECTION (P0.1) is bounded-negative. Carnot is the **energy VERIFIER**
in a hybrid (commodity/third-party generator + energy verifier), and **the
verifier is now the project's entire value-add — and its value is UNPROVEN.** The
EBT replication is therefore **superseded**: re-confirming a thesis the strategic
reframe already closes is churn (north-star §1). It is dropped from .361.

**Win condition (operator 2026-06-06):** the verifier earns its place if it is
**equally effective as the LM at lower cost/latency** (efficiency-parity; no
accuracy edge required, Pareto-dominate where possible). This is RSI-scale
load-bearing — verifying a machine-scale "virtual lab" can't afford an LLM-judge
per artifact.

## 3. The three biggest gaps (current state vs PRD vision + north-star)

1. **The ACCURACY axis (moat) is one fixed gate away from a verdict.** .360 did all
   the hard work (tested harness exp3894 + in-distribution corpus exp3884 ensemble
   AUROC 0.967 + the scissor compute exp3895) and then threw the answer away on a
   mis-specified control bound. .361 re-runs the scissor with the **harness-validity
   control decoupled from the in-distribution reasoner AUROC** (the fixture proves
   the harness; the in-distribution reasoner AUROC is a *finding*), and adds a
   **STRONG self-verify adversarial baseline** (few-shot/structured self-check, per
   arXiv:2602.07594 "Learning to Self-Verify") so the moat is shown to hold even when
   the reasoner's self-verification is boosted, not only when it is weak.

2. **The EFFICIENCY axis has NEVER been run.** This is the operator's actual win
   condition and the single most convincing card Carnot holds. The energy verifier
   is a cheap forward-pass scorer (the FoVer headline runs in ~16s on CPU,
   `live_model_invoked=False`); an LLM-as-judge needs a 35B GGUF forward pass per
   item. The 2026 literature puts LLM judges at **50–500× the cost** of lightweight
   classifiers (hundreds of ms vs sub-10ms; ~$0.00003/check) — so "parity at
   10–100× cheaper" is a measurable, defensible headline. The 2026 best practice is
   exactly the **classifier-first cascade** (= the Meta-EBM Cascade Router): cheap
   verifier on every item, escalate only close calls to the LLM judge.

3. **The agentic-proof venue (ARC-AGI-3 harness) has not been started.** North-star
   §5 sequences it SECOND, after the offline proof. .361 completes the offline proof
   AND begins the verifier-first scaffold (env adapter + verifier-as-router skeleton)
   as pure infrastructure (a passing unit test, no science claim), so Phase-4 can run
   the moment the offline proof lands.

## 4. Milestone shape (11 tasks, exp3903–exp3913, 4 phases)

```
Phase 0  exp3903  archive .360 / activate .361 + GREEN-GATE (yaml parses, core pretest green,
                  reasoner-self-verify + carnot.verify import)
Phase 1  exp3904  ACCURACY axis — MOAT-SCISSOR RE-GATED (reuse exp3894 tested harness + exp3884
  VERIFIER          corpus; harness-validity control = fixture AUROC, NOT in-dist reasoner AUROC;
  EARNS ITS         + STRONG self-verify adversarial baseline)            [critical]
  PLACE     exp3905  BUILD+UNIT-TEST the cost-instrumented verification harness (wall-clock +
  (offline           tokens/FLOPs/$ per verification for BOTH energy-verifier and LLM-judge) [critical]
  proof)    exp3906  EFFICIENCY axis — energy-verifier vs LLM-as-judge HEAD-TO-HEAD (accuracy
                     parity within CI + cost/latency ratio at matched accuracy)            [critical]
          exp3907  META-EBM CASCADE ROUTER prototype (classifier-first cascade: cheap energy
                     verifier first, escalate close-calls to LLM-judge; accuracy vs cost)  [high]
Phase 2  exp3908  ARC-AGI-3 harness SCAFFOLD — BUILD+UNIT-TEST env adapter + verifier-as-router
  AGENTIC           skeleton (infra only, no science claim; deliverable = passing test)    [medium]
Phase 3  exp3909  FACTS graph-grounding harness-first retry (disciplined: hard fixture, duration
                     >=60s, retire_if_same_verdict) — PRD Tier C; deprioritized              [medium]
Phase 4  exp3910  FR-11 v25 continuous self-learning (research-program.md MANDATE)          [high]
  MANDATES exp3911  GateMate A1 terminal-state confirmation (Hardware-Task Continuity)       [medium]
  +HW      exp3912  PolarFire + KV260 consolidated continuity (SSH-only)                     [medium]
  +CAP     exp3913  Capstone .361 — the VERIFIER SCORECARD (accuracy moat + efficiency parity
                     + cost ratio); FORCE the operator's "does the verifier earn its place" call [high]
```

### Dependency graph (all soft / disk-read — no hard `gated_on` on the critical path)

```
exp3903 (activate)
  ├── exp3904 (moat re-gate)      disk-reads exp3894 harness + exp3884 corpus
  ├── exp3905 (cost harness build)
  │     └── exp3906 (efficiency)  disk-reads exp3905 (blocked_upstream if not ready)
  │           └── exp3907 (cascade router) disk-reads exp3906
  ├── exp3908 (ARC scaffold)      independent infra
  ├── exp3909 (facts retry)       independent
  ├── exp3910 (FR-11 v25)         loads persisted v24 state
  ├── exp3911 / exp3912 (hardware)
  └── exp3913 (capstone)          aggregates all non-flagged upstream
```

Every downstream task disk-reads its upstream and emits `blocked_upstream_*` if
absent — a skipped upstream costs ONE task, never a cascade (the .358 cascade root
cause). No hard `gated_on`.

## 5. The re-gated moat scissor (exp3904) — the load-bearing fix

The .360 gate required `reasoner_self_verify_auroc ∈ [0.55, 0.97]` as the
*positive control*. That bound is wrong: it disqualifies the very finding the
premise predicts (a reasoner near chance on self-verification). The corrected gate:

- **Harness-validity control** = the exp3894 **fixture** AUROC (> 0.6) — proves the
  judge harness is not broken. Already satisfied (0.917).
- **In-distribution reasoner AUROC** = a reported FINDING, not a disqualifier. A low
  value corroborates the Self-Correction Illusion (arXiv:2606.05976).
- **STRONG-self-verify adversarial arm** (arXiv:2602.07594): re-run the reasoner
  self-verify with a boosted prompt (few-shot + structured per-step self-check). The
  moat must hold against the *stronger* arm too, or be honestly narrowed.
- **MOAT_SURVIVES** iff harness valid AND `carnot_ensemble_auroc` non-degenerate AND,
  **on the STRONG (boosted) self-verify arm** (the conservative case — if the moat holds
  against a competent self-verifier it holds against a weak one), `residual_catch_ci95.low
  > 0.5` AND `error_overlap_jaccard < 0.6` AND `n_residual >= 30`. The WEAK-arm metrics are
  reported as supporting evidence (and corroborate the Self-Correction Illusion), not as a
  separate gate condition.

## 6. The efficiency head-to-head (exp3905/exp3906) — the new first-class metric

Same corpus (exp3884 in-distribution + a FoVer slice), two verifiers:
- **Energy verifier** = the k-verifier ensemble (CPU forward-pass scoring; the FoVer
  headline path, `live_model_invoked=False`).
- **LLM-as-judge** = the tested reasoner self-verify (exp3894) over a 35B GGUF.

Report BOTH: (a) accuracy parity within CI (AUROC of each on the same labels), and
(b) the cost/latency ratio (wall-clock + token/FLOP estimate + $ per verification at
matched accuracy). Target headline framing: **"parity at N× cheaper."** HONEST
reporting — parity ≠ beating; a null is still a result; do not fabricate a moat.
Comparators: ThinkPRM (arXiv:2504.16828, beats LLM-judge +7.2% at matched token
budget), CompassVerifier (arXiv:2508.03686), OPV (arXiv:2512.10756).

## 7. Architecture / hardware

No architecture change. Hardware: KV260 + PolarFire opportunistic (SSH-only),
GateMate terminal-state confirmation per Hardware-Task Continuity. The energy
verifier's cheap forward-pass is the hardware-acceleratable primitive (north-star
§5: Ising/FPGA devices EVALUATE energy, they do not generate by minimization).

## 8. Invariants

`paper_ready=TRUE` (G1–G4 met); FoVer 0.9131 frozen, NEVER substituted (.361 adds
ACCURACY + EFFICIENCY lenses, not a new headline); verifier math-domain-bound until
facts proven non-fabricated; both energy theses (selection + generation)
bounded-negative; EBT replication superseded/dropped; never aggregate
`flagged_adversarial` artifacts; BARE-scalar field emission; all tasks
codex+requires_codex (anti-wipeout); no external publication.

## 9. Routing

All tasks `agent_type: codex` + `requires_codex: true` + `model: gpt-5.5`
(anti-wipeout; gemini crashes GPU workloads and 429-wiped .333/.355; standing
operator gemini↔codex flip authority 2026-06-05). GPU/live-model tasks add
`requires_gpu: true` and Run via `{project_root}/.venv/bin/python` (bare python has
no torch → silent CPU drop).
