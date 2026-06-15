# Research Roadmap — Milestone 2026.06.391

**Title:** FINISH WHAT .390 STARTED — the oracle-distinct headline was blocked on a
fixable DATA bug (not a null), and the verifier-as-reward run died on a fixable
HARNESS bug (not a science result). .391 de-risks and COLLECTS both decisive reads.

**Planner:** Claude Opus 4.8 (outer-loop), 2026-06-15.
**Milestone doc:** `openspec/change-proposals/research-roadmap-v391.md`
**Roadmap YAML:** `research-roadmap-next.yaml` (11 tasks, exp4219–exp4229)
**Prior milestone:** `openspec/change-proposals/research-roadmap-v390.md`

---

## 0. One-paragraph thesis

`.390 asked the right two questions and got blocked by infra on BOTH of the
load-bearing ones. The HEADLINE (does a LEARNED, oracle-distinct verifier beat
majority vote on ARC where execution is not the oracle?) never ran its decision
gate: the A2 build (exp4209) returned `blocked_arc_pool_no_candidate_labels`
because it looked for per-candidate labels in the wrong file, so A3 (exp4210)
gate-blocked. Yet the SAME milestone's cheap detector probe (exp4208) **labeled
8,041 ARC candidates and measured an oracle-distinct ARC detection AUROC of 0.9016
(CI95 [0.78, 0.998], `verifier_is_oracle=false`)** — the oracle-distinct signal
demonstrably EXISTS; only the selector's labeled pool was missing, and the working
label path is in-repo (`scripts/exp_verifier_detector_auroc.py:load_arc_rows()`).
The OWED verifier-as-reward A-vs-B (exp4211) failed a 3rd straight time on a NEW,
fixable harness bug — PEFT refused the custom `Gemma4ClippableLinear` module — with
the operating point (Phase-0 0.956, Youden-J 0.414) and N-matched corpora
(A=776/B=776/C=742) intact on disk. **.391 is not a new direction; it is the
disciplined COLLECTION of the two reads `.390 set up.** It reuses the proven label
path to BUILD the oracle-distinct ARC verifier and run the beats-vote gate with a
matched control + ARBITER-style conservative override, and applies the project's own
.360 HARNESS-FIRST lesson to the recurring reward-training failure (fix the LoRA
attach, smoke-test it, THEN run). ARC continues its monotonic +1 (→17) and the live
solver pushes for its first level completion. Honest nulls are complete; the point is
that this time the gate actually RUNS.

---

## 1. What .390 produced (the inputs to this plan)

Read via `scripts/summarize_artifact.py`; capstone exp4218.

**Phase A — the oracle-distinct frontier (HEADLINE): blocked on DATA, not science.**
- **exp4208 (detector AUROC) — CLEAN WIN, 73s, not flagged.** Detection AUROC:
  sudoku 1.0, code 1.0, math 1.0, **arc 0.9016 (CI95 [0.78, 0.998])**. Selection
  headroom: sudoku 0.0007, code 0.18, math 0.0, arc 0.129. The DIVERGENCE is real:
  **detection AUROC 1.0 on sudoku/math where selection headroom ≈0** — the verifier
  earns its place as a detector/abstention gate even with zero selection headroom.
  `verifier_is_oracle`: sudoku/code/math = true (execution-grounded/circular), **arc
  = false (the oracle-distinct one)**. Caveats it honestly records: the ARC base rate
  is 0.0024 (≈19 positives / 8,041) so precision@recall0.9 is 0.004 (ranking AUROC is
  decent, precision is poor on the imbalanced pool); sudoku's `valid_but_wrong_auroc`
  is null (no valid-but-wrong negatives) so its 1.0 is partly the trivial invalid-grid
  split.
- **exp4209 (oracle-distinct ARC verifier BUILD) — BLOCKED.**
  `blocked_arc_pool_no_candidate_labels` (accepted=0 / rejected=0 / total=0). It read
  `results/arc3_trm_verifier_rerank.json` for per-candidate `is_correct` labels — that
  file holds rerank SUMMARIES (`per_task`, `oracle_ceiling`), not training rows. The
  WORKING path (used by exp4208) is `load_arc_rows()` over
  `arc3_gap3_stage2_eval_pool.json.gz` + `arc3_gap4_induced_programs.json` (label =
  candidate grid match vs the GAP-4 induced program's `pred_grid`).
- **exp4210 (THE HEADLINE GATE) — gate-blocked** on `exp4209.selector_trained == False`.
  Never ran. The capstone's "NO-HEADROOM-OR-NO-SIGNAL" is therefore an INFRA artifact,
  not a measured null — the verifier was never built.

**Phase B — verifier-as-reward (OWED, FR-11/self-learning): 3rd infra failure.**
- **exp4211 — `progress: accumulating_..._no_eval_yet`, flagged DURATION_TOO_SHORT
  (14.2s).** Training status `failed` with a precise, fixable error:
  `ValueError: Target module Gemma4ClippableLinear(...) is not supported. Currently,
  only torch.nn.Linear, ... are supported.` PEFT/LoRA matched the custom Carnot Gemma4
  wrapper (whose `.linear` submodule is the real `nn.Linear`) instead of a standard
  module. The operating point is real (base gemma-4-E4B-it, K=5, Phase-0 0.956,
  Youden-J 0.4138), corpora N-matched (A=776/B=776/C=742), checkpoint dir intact. This
  is the 3rd straight infra failure (exp4198 background-process death → exp4199
  gate-block → exp4211 PEFT attach), each a DIFFERENT root cause.

**Phase B2 — certified-ARC-corpus distill-lift: twice-uninformative (DROPPED for .391).**
- **exp4212 — flagged (lift `distill_lift_delta=0.0` exactly), corpus stuck at 16
  (precision 0.9375), CI [0.0, 0.0] → ABSENT.** The Invisible-Leash read came back
  ABSENT (the local base's ARC-induction abstraction does not lift from in-context
  certified exemplars). This is the SECOND uninformative/absent read of this exact
  scope (exp4200 → exp4212). Per the Failed-Experiment Rerun Discipline, re-running it
  as-is is churn; the honest answer ("a stronger base is needed for ARC distillation")
  is a .392+ base-selection decision, not a .391 rerun. **Dropped this milestone.**

**Phase C — ARC north star: +1 level, live still efficiency-only.**
- **exp4213 — SUCCESS:** advanced game `sc25` to L2, +1 new level,
  `total_levels_solved = 16` (from 15), `real_env_confirmed = true`. A genuine
  monotonic win.
- **exp4214 — live env, efficiency-only:** `solver_completes_0_levels`, score 0.0,
  beats the random/greedy floor on efficiency but not accuracy (same shape as exp4202).

**Phase D — reserved slots.**
- **exp4215 (SOTA ingestion) — 8 methods mapped; flagged for .391:
  `arbiter_conservative_override_arc_wrong_majority_v391`.**
- **exp4216 (registry hygiene) — flagged** METHODOLOGY_MISSING (no `random_seed`/
  `reproducibility_checksum`) + short duration. `regression_guard_passed = True`.
- **exp4217 (hardware) — OK:** GateMate unreachable (blocked), PolarFire hash-verified
  CPU dispatch succeeded, KV260 terminal confirmed.
- **exp4218 (capstone):** `oracle_distinct = NO-HEADROOM-OR-NO-SIGNAL` (the infra
  block), `verifier_as_reward = ACCUMULATING`, `arc_levels = 16`, DiffusionGemma gate
  `STILL-PENDING`, 2 flagged-skipped.

**Invariants carried into .391.** FoVer headline 0.9131 frozen, `paper_ready = True`
(G1–G4). Verifier execution wins are circular (`verifier_is_oracle = true`); the
oracle-distinct claim is the open frontier. DiffusionGemma stays GATED (activate only
on an oracle-distinct win, matched control, CI95-excl-0). The TRM Sudoku checkpoint is
DONE (val 0.8227) and the conductor stays stood-down on TRM training — NO task may
launch TRM training, `pkill`/`kill` train.py, or write
`results/trm_runs/sudoku_extreme_baseline/`. Qwen is FORBIDDEN as the TRAINED base
(Spurious-Rewards confound); Qwen GGUF as an off-policy teacher/certifier is fine.

---

## 2. Why these three gaps, and why now

The three biggest gaps between the current state and the PRD vision (`escape
hallucination via verifiable reasoning + autonomous directed self-learning`,
north-star = solve ARC-AGI-3 accurately and efficiently):

1. **The oracle-distinct verifier moat is UNPROVEN — and the only thing standing
   between `.390 and a clean read was a wrong file path.** This is the 2026-06-14 P0
   directive and THE headline. The signal exists (exp4208 oracle-distinct ARC detection
   AUROC 0.90); the selector pool just needs the working label path. New literature
   confirms the mechanism is real and beatable: **AggLM (2509.06870)** trains an
   aggregator that recovers minority-but-correct answers and beats reward-model
   baselines; **AgentAuditor (2602.09341)** beats BOTH vote and LLM-judge;
   **GenSelect-BoN (2602.02143)** beats vote on math+code. The headline gate is no
   longer speculative — it is a known-achievable result we have not yet run on ARC.

2. **Verifier-as-reward / self-learning (FR-11) is owed and keeps dying on infra, not
   science.** The operator's 2026-06-11 pivot. Three infra failures in a row, each a
   different harness bug. The project ALREADY solved this class of problem once — the
   `.360 HARNESS-FIRST discipline (build + unit-test the harness as a separate
   deliverable whose acceptance is a PASSING positive-control test, THEN run the
   measurement). `.391 applies it: a separate LoRA-harness fix+smoke task gates the
   3-arm run.

3. **ARC-AGI-3 accuracy on the live env.** The offline solver advances +1/milestone
   (now 16); the live solver completes 0 levels. The flagged-for-.391 method (ARBITER
   conservative-override) directly targets this — better goal-predicate induction /
   action selection by overriding the default only on high learned margin.

These map onto the SAME Phase A/B/C/D structure as `.390 — which is correct: the
milestone is the disciplined completion of `.390's setup, not a pivot. The difference
is every `.390 infra blocker is now diagnosed with a precise fix.

---

## 3. Architecture (what executes)

```
                 ARC GAP-4 pools (cached, on disk)
   arc3_gap3_stage2_eval_pool.json.gz  +  arc3_gap4_induced_programs.json
                          │
        load_arc_rows()  ─┤  per-candidate (features, is_correct)   ← the WORKING
        (reuse exp4208)   │  label = candidate grid == induced pred_grid   label path
                          ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ PHASE A — ORACLE-DISTINCT HEADLINE (the de-risked retry)      │
   │  A1 exp4220  BUILD labeled pool + train V-STaR/aggregator     │
   │     (accepted+rejected, oracle-distinct features, out-of-fold)│
   │     verifier_is_oracle=false → off_fold_auroc, selector_trained│
   │                          │ gated_on selector_trained==true     │
   │  A2 exp4221  BEATS-VOTE gate on held-out ARC:                  │
   │     learned verifier@1 vs vote@1 vs matched control vs oracle@K│
   │     + ARBITER conservative-override; CI95-excl-0 = THE HEADLINE│
   └──────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │ PHASE B — VERIFIER-AS-REWARD (OWED; FR-11) harness-FIRST      │
   │  B1 exp4222  FIX + SMOKE the LoRA harness (standard HF load,   │
   │     nn.Linear target_modules) → assert attach + 1 finite step  │
   │                          │ gated_on harness_smoke_passed==true │
   │  B2 exp4223  RESUME checkpoint, 3-arm A/B/C/D SYNCHRONOUS      │
   │     A(certified) vs B(random-label) vs C(gold) vs D(cold)      │
   │     verifier_is_oracle=true (honest reward axis)               │
   └──────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │ PHASE C — ARC NORTH STAR                                      │
   │  C1 exp4224  monotonic ARC +1 (total_levels >= 17, offline)   │
   │  C2 exp4225  live-env ACCURACY probe + ARBITER override; NO   │
   │     leaderboard submission                                     │
   └──────────────────────────────────────────────────────────────┘

   PHASE D — reserved   D1 exp4226 SOTA-ingestion (learned-aggregator track)
                        D2 exp4227 registry+gaps hygiene (flag-fixed)
                        D3 exp4228 hardware continuity (GateMate/PolarFire/KV260)
                        D4 exp4229 capstone .391
   exp4219 archive .390 → activate .391
```

The energy/learned verifier is the VERIFICATION layer throughout (north-star §5
hybrid): generator = codex/local-LLM/cached candidates; the LEARNED verifier scores
WITHOUT executing (Phase A, `verifier_is_oracle=false`); the EXECUTION verifier is the
honest reward channel (Phase B, `verifier_is_oracle=true`). No task runs the energy
function as a generator (closed-negative).

---

## 4. Phases & tasks (11 tasks, exp4219–exp4229)

**exp4219 — archive .390 → activate .391** (infra, codex). Archive into
research-complete.yaml; assert YAML parses; green pre-test gate; record the .390
close-state truthfully (oracle-distinct A2/A3 blocked on DATA not science; detector
oracle-distinct ARC AUROC 0.90 exists; reward 3rd infra failure = PEFT attach; ARC 16;
2 flagged-skipped).

**PHASE A — ORACLE-DISTINCT HEADLINE (the de-risked retry).**
- **exp4220 — BUILD the labeled ARC pool + train the oracle-distinct verifier**
  (codex). OWNS label construction: reuse `scripts/exp_verifier_detector_auroc.py:
  load_arc_rows()` over the GAP-4 pools to build per-candidate (features, is_correct);
  stratify to wrong-majority tasks (oracle@K > vote — the ARBITER/AggLM headroom);
  train V-STaR (accepted+rejected) / an AggLM-style aggregator on oracle-distinct
  features (NO demo execution at inference) out-of-fold; persist + report
  `off_fold_auroc`, `selector_trained`, `learned_verifier_path`, `wrong_majority_n`.
  `verifier_is_oracle=false`. Honest build-null (AUROC≈0.5 or too-few-positives) is
  complete and still persists the artifact for A2.
- **exp4221 — BEATS-VOTE gate (gated_on exp4220 selector_trained==true)** (codex). THE
  HEADLINE. Held-out ARC split disjoint from A1 folds: per task compute vote@1,
  verifier@1 (learned score, no execution), oracle@K (positive control), matched
  no-verifier control (budget-equal), AND an ARBITER conservative-override arm (keep
  vote unless learned margin clears a fixed threshold). Gate = `oracle_distinct_beats_vote`
  := `verifier_minus_vote` CI95 (≥2000 bootstrap) excludes 0 AND delta>0 AND headroom
  present. Positive-control-FIRST: if oracle@K≈vote the corpus is ceiling-saturated →
  `no_headroom_uninformative` (not a verifier failure). `adversarial_verify.py` must
  stay clean (no CIRCULAR_MOAT_OVERCLAIM, since `verifier_is_oracle=false`).

**PHASE B — VERIFIER-AS-REWARD (OWED; FR-11/self-learning) harness-FIRST.**
- **exp4222 — FIX + SMOKE the LoRA harness** (codex, GPU). Root-cause the
  `Gemma4ClippableLinear` PEFT rejection: load the NON-Qwen base via standard
  `transformers.AutoModelForCausalLM.from_pretrained("google/gemma-4-E4B-it")` (NOT
  the custom Gemma4 wrapper) so PEFT sees standard `nn.Linear`, with explicit
  `target_modules` (q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj) — OR
  unwrap/patch the wrapper to expose its inner `.linear`. Deliverable = a PASSING smoke
  (a positive control on the HARNESS): assert LoRA attaches, 1 training step runs, loss
  is finite, on a 8–16-example fixture from the intact corpora. `harness_smoke_passed`
  BARE bool.
- **exp4223 — 3-arm A-vs-B run (gated_on exp4222 harness_smoke_passed==true)** (codex,
  GPU, live). RESUME the stable checkpoint
  `code_verifier_reward_lora_rft_a83b52882c198954`; run A/B/C/D SYNCHRONOUSLY in-process
  with per-step progress prints (codex idle-timeout safe). VALIDITY GATES FIRST
  (gold-control Arm C ≥ base; truncation <5%) then A-vs-B: pass@1(A certified) −
  pass@1(B same-generator random-label = Spurious-Rewards control arXiv:2506.10947),
  task-level bootstrap CI95. `verifier_label_carries_signal` BARE. `verifier_is_oracle
  =true` (honest reward axis). Accumulate-floor: 3rd consecutive no-usable window →
  `complete_..._retired_substrate_cannot_power`.

**PHASE C — ARC NORTH STAR.**
- **exp4224 — ARC incremental +1 (codex):** advance the solved-LEVEL count to ≥17 via
  the proven explore→induce→hardened-GAP-4-verify→act loop, offline. Honest no-solve is
  complete.
- **exp4225 — ARC live-env ACCURACY probe (codex):** push for the first level
  completion on a live game, applying the ARBITER conservative-override to goal-predicate
  induction / action selection; report accuracy AND efficiency vs the floor. NO
  leaderboard submission (operator-only); anonymous key; bounded budget.

**PHASE D — reserved slots.**
- **exp4226 — SOTA-ingestion (codex):** ingest the `.391 sweep (AggLM 2509.06870,
  AgentAuditor 2602.09341, GenSelect-BoN 2602.02143, MSV 2603.03417, SR-TTRL,
  CoT-verifier-learnability 2603.03538) mapped onto the headline; flag the strongest for
  .392. Real arXiv IDs only.
- **exp4227 — verifier-registry + gaps hygiene (codex):** bit-exact GAP-4 regression
  replay; record the `.391 oracle-distinct A2/A3 + reward + detector outcomes into
  `ops/verifier_gaps.md`. **FLAG FIX:** declare `inference_substrate`,
  `random_seed`, `reproducibility_checksum`, `model_specs` so it does not re-trip
  DURATION_TOO_SHORT / METHODOLOGY_MISSING like exp4216.
- **exp4228 — hardware continuity (codex):** per-board reachability + next step;
  GateMate/PolarFire drive-to-terminal, KV260 opportunistic; SSH/USB-detect
  preconditions ONLY.
- **exp4229 — capstone .391 (codex):** one honest headline on the oracle-distinct
  frontier (did the learned ARC verifier beat vote, matched control, CI95-excl-0?) +
  the verifier-as-reward A-vs-B + ARC progress; honor `verifier_is_oracle` (no circular
  moat headline); set `diffusiongemma_gate_resolvable` true ONLY on an oracle-distinct
  win with a matched control.

---

## 5. Dependency graph

```
exp4219 (archive/activate) ─ runs first

exp4220 (build oracle-distinct verifier)
   └─gated_on selector_trained==true→ exp4221 (beats-vote HEADLINE gate)

exp4222 (LoRA harness fix+smoke)
   └─gated_on harness_smoke_passed==true→ exp4223 (3-arm A-vs-B run)

exp4224 (ARC +1)          ─ independent
exp4225 (ARC live accuracy)─ independent (reuses exp4214/exp4202 live adapter)

exp4226 (SOTA) · exp4227 (registry) · exp4228 (hardware) ─ independent
exp4229 (capstone) ─ reads exp4220..4228; UNGATED (skips flagged/blocked upstreams)
```

Two gate chains (A1→A2, B1→B2) mean a build failure fast-skips its measure task
(saving the Sonnet call) rather than cascading. Both gates are de-risked: A1 owns its
label construction (the `.390 block cannot recur), B1 is a harness smoke (the `.390
PEFT bug is the explicit fix target).

---

## 6. Hardware & model requirements

- **RTX 3090 (CUDA):** Phase B (exp4222 LoRA smoke, exp4223 3-arm SFT). Standard
  `AutoModelForCausalLM` load of `google/gemma-4-E4B-it` (NON-Qwen trained base; small
  is fine — the A-vs-B contrast, not absolute capability, is the measurement).
- **CPU / cached pools:** Phase A (exp4220 trains a small logistic/MLP selector on CPU
  from cached ARC candidates — no GGUF, no GPU), exp4227 (cached GAP-4 replay).
- **Network (anonymous ARC SDK):** exp4225 live env (`pip install arc-agi`, anonymous
  key, no submission).
- **FPGA boards (SSH/USB-detect):** exp4228 — GateMate (`openFPGALoader -c dirtyJtag
  --detect`), PolarFire (`ssh polarfire`), KV260 (`ssh kria`, terminal).
- **SOTA-model note:** Phase A scores CACHED ARC candidates (no live LLM). Phase B's
  trained base is gemma-4-E4B-it (non-Qwen, per the Spurious-Rewards confound). No
  task requires a 35B headline model this milestone; all GGUF use (if any) is via
  `cached_sota_pair()` + the `.gguf` path (never AutoTokenizer on a GGUF repo).

---

## 7. What this milestone is NOT

- **NOT a new direction.** It collects the two reads `.390 set up and got blocked on
  infra. Re-running them with the fixes is the OPPOSITE of churn — `.390's headline was
  never measured.
- **NOT a certified-corpus distill-lift rerun.** exp4200→exp4212 returned ABSENT twice;
  the honest answer is "a stronger base is the .392 question." Dropped here (Depth-Over-
  Breadth).
- **NOT a circular-moat claim.** Every Phase-A task sets `verifier_is_oracle=false` and
  carries a matched control; Phase-B sets it true (honest reward axis). The capstone may
  not headline a circular result as a moat, and the DiffusionGemma gate flips only on an
  oracle-distinct win with a matched control.
- **NOT a TRM-training or conductor-modifying milestone.** TRM checkpoint frozen;
  `scripts/research_conductor.py` untouched; no external publication / leaderboard
  submission.
- **NOT gated on the recurring conductor false-zero retro detector** (an operator-side
  conductor fix, out of scope for an experiment task that cannot modify the conductor —
  flagged to the operator, not a `.391 task slot).

---

## 8. Acceptance — the decision-grade questions

1. **HEADLINE:** Did a LEARNED (oracle-distinct, `verifier_is_oracle=false`) ARC
   verifier BEAT majority vote off-oracle, with a matched control and CI95-excl-0
   (exp4221)? → the first oracle-distinct moat (closes GAP-3-ties-vote; flips the
   DiffusionGemma gate) OR an honest ties-vote-with-headroom null (frontier persists)
   OR no-headroom-uninformative. **This time the gate RUNS** — exp4220 owns the label
   path that blocked `.390.
2. **OWED reward:** Did the verifier's LABEL carry training signal —
   A(certified) ≫ B(random-label), CI excl 0, with the gold control + truncation guard
   passing (exp4223)? Or a clean A≈B distillation null? **This time the LoRA harness
   attaches** (exp4222 smoke-gates the run).
3. **NORTH STAR:** `total_levels_solved ≥ 17` (exp4224); did the live solver complete
   its first level (exp4225)?
4. **Hygiene:** registry regression-guard green and NOT flag-tripped (exp4227);
   capstone honors `verifier_is_oracle` and reports `diffusiongemma_gate_resolvable`
   honestly (exp4229).

A milestone where exp4220 builds the verifier and exp4221 RUNS the gate — whatever the
sign — is a success by north-star §1, because it converts `.390's infra-block into a
measured result on the project's headline question.
