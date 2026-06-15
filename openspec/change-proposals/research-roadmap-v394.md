# Research Roadmap v394 — HARDEN the first ARC oracle-distinct win, then EXTEND selection→synthesis, then STAGE the DiffusionGemma scale-up

**Milestone:** 2026.06.394
**Planned:** 2026-06-15 (Claude Opus 4.8, outer-loop planner)
**Predecessor:** 2026.06.393 (`openspec/change-proposals/research-roadmap-v393.md`)
**North star:** solve ARC-AGI-3 accurately AND efficiently (`ops/north-star.md` §0)

---

## 0. THE HEADLINE QUESTION

> **Does the program's FIRST ARC oracle-distinct verifier-beats-vote win (exp4245,
> +44pp) SURVIVE adversarial scrutiny — provenance-blind features (no leak),
> multi-seed replication, AND cross-GAME transfer — and if so, can SYNTHESIS
> (AggLM grid-reconciliation) break the oracle@K=0.827 selection ceiling? Only a
> surviving, hardened win justifies STAGING the gated DiffusionGemma scale-up.**

This is **depth on the single most important positive in the program's history**,
not churn. `.393 answered "does a learned (oracle-distinct) verifier beat majority
vote on ARC off-oracle?" → YES (+0.4423, CI95 [0.308, 0.596] excludes 0,
verifier_is_oracle=false). `.394 does NOT re-run that gate. It applies the
MANDATORY "cross-check surprising results" discipline (CLAUDE.md "Adversarial
Artifact Verification + Sample-Size Rigor") to a result that is, today,
**single-seed (4245), n=52, with a plausible provenance-leak vector**, BEFORE it
headlines a paper or triggers an expensive bet.

---

## 1. WHAT `.393 PROVED (and the three caveats `.394 must close)

| Phase | Result | Read |
|---|---|---|
| A (ARC oracle-distinct) | **exp4245 ARC-MOAT-WON**: set_encoder@1 0.692 vs vote@1 0.25, **delta +0.4423**, CI95 [0.308, 0.596] excl 0, oracle@K 0.827, matched-control@1 0.21, n=52, verifier_is_oracle=false, adversarial-clean | **FIRST ARC oracle-distinct win.** `diffusiongemma_gate_resolvable=true`. |
| A2 build | exp4244: DeepSets set-encoder AUROC **0.963** *underperformed* the .392 augmented-logistic **0.980** (delta −0.016) | **The win came from the GROWN POOL (A1, positives 20→48), NOT the set-encoder architecture.** Attribute to data, not architecture. |
| A4 code replication | exp4246 **BLOCKED** (`blocked_code_second_corpus_missing`) | The .392 code win (+3.1pp) stays **single-corpus, unreplicated**. |
| B reward (FR-11) | exp4247 **blocked** (`cannot_run_in_window`) + **flagged_adversarial CRITICAL**; exp4248 gate-blocked | **7th consecutive failure.** In-window training is infeasible. live_lora_retired=true recorded. |
| C ARC north star | exp4249 → L5 / **19 levels**; exp4250 live-env 0 levels (efficiency-only) | Monotonic +1 held. |
| D hardware | KV260 terminal (SSH-only); PolarFire hash-verified CPU dispatch; GateMate unreachable | KV260 opportunistic per north-star §3. |

**The three caveats on the headline win (the load-bearing reason `.394 exists):**

1. **Surprising magnitude, single seed.** +44pp from a TIE, on n=52, seed 4245
   only. Per CLAUDE.md a surprising result is not headline-eligible until
   replicated (n>1 seeds, same direction).
2. **Provenance-leak risk.** Positive labels = exact match to gold-flagged OR the
   **GAP-4 induced `pred_grid`**. Features include `shape/palette family
   indicators` + `duplicate counts`, which can encode candidate ORIGIN
   (induced-program output vs sampled candidate) rather than correctness. A
   detector that learns provenance, not correctness, would inflate AUROC to ~0.96
   and the selection delta to +44pp while being **circular/leaked**, not
   oracle-distinct.
3. **Within-distribution only.** The held-out split is held-out TASKS within the
   SAME games. The real OOD test — does the verifier transfer to UNSEEN games? —
   is unrun (cf. the ARC-1→ARC-2 collapse, `project_gap3_verifier_program`).

---

## 2. THE PLAN — 4 phases + archive (14 tasks, exp4255–exp4268)

### PHASE A — HARDEN + EXTEND THE ARC ORACLE-DISTINCT HEADLINE

```
exp4256  A1  LEAK-AUDIT + provenance-blind re-test  ──┐
exp4257  A2  multi-seed + independent clean-room repl ─┤── win_survives? ──► gate B1/A4
exp4258  A3  cross-GAME transfer (the real OOD test) ──┘
exp4259  A4  AggLM grid-SYNTHESIS (break oracle@K)     (gated_on A1∧A2)
```

- **A1 (exp4256) — Leak-audit.** Re-run the decisive gate with **provenance-blind
  features** (strip every feature that could encode candidate origin: family
  indicators, duplicate counts) keeping only genuinely oracle-distinct content
  signals; PLUS an explicit **origin probe** (train a classifier to predict
  induced-vs-sampled from the features — if it is highly predictive AND correlates
  with `is_correct`, that is the leak). Emit `win_survives_provenance_blind` (bare
  bool). Methods: data-contamination survey (2502.14425), in-context contamination
  detection (2510.27055).
- **A2 (exp4257) — Multi-seed + independent replication.** Re-run the decisive gate
  across **≥5 seeds** (fold reshuffles) → mean delta + cross-seed CI; AND an
  independent clean-room re-score from the persisted pool artifact via a separate
  code path (mirrors the G2 FoVer reproducer). Emit `oracle_distinct_win_replicates`
  (bare bool := cross-seed CI excludes 0).
- **A3 (exp4258) — Cross-GAME transfer.** Train on a subset of GAMES, test on
  HELD-OUT GAMES. The real OOD test. Honest collapse here SCOPES the claim
  (within-game vs cross-game) and is decision-grade. Method: ARC survey (2603.13372).
- **A4 (exp4259) — Selection → SYNTHESIS** (gated_on A1∧A2). DeepSets-weighted
  per-cell grid reconciliation over the top-ranked candidate family → a synthesized
  grid that can differ from every candidate, validated by **exact ARC grid match**,
  with vote / selector-only / no-synthesis matched controls. Target: the 17% of
  tasks where oracle@K fails (no candidate correct). Thesis: Compute-as-Teacher
  (2509.14234) — synthesis can EXCEED the best rollout; Generative-Aggregation
  (2503.04104). Fabrication guard: exact-grid-match gate + explicit no-synthesis
  baseline (per the .393 SOTA-ingestion warning).

### PHASE B — STAGE THE SCALE-UP + ARC NORTH STAR

- **B1 (exp4260) — DiffusionGemma energy-guided PREFLIGHT** (gated_on A1∧A2; NOT the
  full run). Load DiffusionGemma-GGUF (cached), wire the verifier ensemble as a
  guidance energy on a TINY denoising smoke (few steps, few examples), confirm the
  guidance hook reweights token selection, emit GO/NO-GO + a cost estimate for the
  full `.395 run. Methods: discrete-diffusion guidance (2406.01572), GTL
  (2512.10877), EDLM (2506.13759). **The expensive full run is gated to `.395 on
  the win surviving** — per the operator's twice-burned over-claim lesson, do not
  scale up an unverified win.
- **B2 (exp4261) — Monotonic ARC +1** (total_levels ≥ 20). ARC-AGI-3
  Incremental-Progress Scoping (+1, not all-levels).
- **B3 (exp4262) — ARC live-env accuracy probe** targeting a level completion with
  the hardened verifier routing. NO leaderboard submission (operator-only).

### PHASE C — RESOLVE THE OWED AXES (decision, not another doomed attempt)

- **C1 (exp4263) — verifier-as-reward OUT-OF-BAND re-scope OR retire** (FR-11
  self-learning). 7× in-window failures = in-window training is infeasible. Per
  Failed-Experiment Rerun Discipline, **no 8th in-window attempt**: re-scope to
  prepare the offline reward-weighted corpus + a one-command runner + a validation
  harness, emit `ready_for_out_of_band` (operator/outer-loop runs the training like
  the TRM checkpoint), OR if even prep is infeasible, RETIRE the axis (FoVer
  +0.0185 memory-ablation stands as the self-learning evidence). `retire_if_same_verdict`.
- **C2 (exp4264) — code oracle-distinct replication RETRY** on a FRESH second corpus
  (cached SOTA GGUF best-of-N on MBPP/EvalPlus) OR retire. Fixes exp4246's root
  cause (no distinct corpus found). `retire_if_same_verdict`.

### PHASE D — HYGIENE & CAPSTONE

- **D1 (exp4265) — SOTA-ingestion** (mandatory; `.395 forks: synthesis robustness,
  discrete-diffusion guidance, leak-audit methods).
- **D2 (exp4266) — registry + gaps hygiene** + GAP-4 regression guard + record
  `.394 outcomes + log any missing-verifier gaps.
- **D3 (exp4267) — hardware continuity** (KV260 opportunistic/terminal, PolarFire,
  GateMate).
- **D4 (exp4268) — capstone `.394**.

---

## 3. DEPENDENCY GRAPH

```
exp4255 archive/activate
   │
   ├── exp4256 A1 leak-audit ────────► win_survives_provenance_blind ─┐
   ├── exp4257 A2 multi-seed/repl ───► oracle_distinct_win_replicates ┼─► (A1∧A2) ─► exp4259 A4 synthesis
   ├── exp4258 A3 cross-game transfer (independent)                   └─► (A1∧A2) ─► exp4260 B1 DiffusionGemma preflight
   ├── exp4261 B2 ARC +1            (independent)
   ├── exp4262 B3 ARC live accuracy (independent)
   ├── exp4263 C1 reward out-of-band/retire (independent)
   ├── exp4264 C2 code replication retry/retire (independent)
   ├── exp4265 D1 SOTA-ingestion
   ├── exp4266 D2 registry/gaps hygiene
   ├── exp4267 D3 hardware continuity
   └── exp4268 D4 capstone
```

**Conjunctive gates** (`gated_on`): A4 and B1 each require `A1.win_survives_provenance_blind == true` AND `A2.oracle_distinct_win_replicates == true`. If the win is a leak or a single-seed fluke, both downstream builds skip — the discipline that prevents scaling an unverified win.

---

## 4. HARDWARE REQUIREMENTS

- **CPU (primary):** A1/A2/A3/A4 are CPU set-encoder re-scoring over the cached
  grown pool (no GPU, no GGUF). C2 code-pool generation may use the dual RTX 3090
  + a cached SOTA GGUF (best-of-N), bounded.
- **RTX 3090 + DiffusionGemma GGUF (cached):** B1 preflight loads
  `unsloth/diffusiongemma-26B-A4B-it-GGUF` for a tiny denoising smoke.
- **FPGA (opportunistic):** D3 — KV260 (SSH-only, terminal), PolarFire (CPU
  dispatch), GateMate (USB detect).
- **HARD RULE:** the TRM checkpoint (`results/trm_runs/sudoku_extreme_baseline/`,
  val 0.8227) is DONE; no task may launch TRM training, pkill train.py, or write
  that directory. Qwen is FORBIDDEN as a TRAINED base (Spurious-Rewards confound);
  Qwen GGUF as an off-policy teacher/certifier is fine.

---

## 5. DISCIPLINES APPLIED (every task)

- **Codex-Default v2:** all tasks `agent_type: codex`, `model: gpt-5.5`. No gemini.
  Archive + capstone on codex (mechanical). No `requires_claude_verified`.
- **Circularity / Oracle-Distinctness:** every verifier-value task declares
  `verifier_is_oracle` (BARE bool). The whole headline is verifier_is_oracle=false.
- **Adversarial Artifact Verification + Sample-Size:** A1/A2 directly implement the
  "cross-check surprising results" mandate; N≥30 / multi-seed; methodology fields
  (`random_seed`, `reproducibility_checksum`, `model_specs`).
- **Pre-Launch Preconditions:** every compute-bound task opens with a PRECONDITIONS
  step 0 emitting `blocked_<resource>` on a miss.
- **Inference-Substrate Declaration:** each task declares `inference_substrate`.
- **Verdict Terminal-Prefix:** every `honest_verdict` starts with
  `complete:`/`success:`/`passed:`/`shipped:`.
- **Principle-Annotated Artifact Fields + gated-fields-must-be-bare:** every
  REQUIRED ARTIFACT FIELD + gate carries a `principle:`; gated fields are bare.
- **Failed-Experiment Rerun / Exclusion-Manifest:** C1 (reward) + C2 (code) carry
  `prior_failures` with all four sub-fields incl. `retire_if_same_verdict`. The
  oracle-distinct ARC verifier tasks carry `operator_override` (class 3,
  versioned-lineage continuation; standing 2026-05-29 directive + the 2026-06-14 P0
  oracle-distinct mandate).
- **ARC-AGI-3 Incremental-Progress Scoping:** B2 targets +1 level, not all-levels.
- **SOTA-Ingestion Cycle:** D1 reserved (bleeding-edge headline).
- **Operator-Only External Publication:** B3 takes NO leaderboard submission.
