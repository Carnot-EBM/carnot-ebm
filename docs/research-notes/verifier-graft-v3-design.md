# Verifier-as-Reward Graft v3 — Design (valid de-confound)

**Status: QUEUED (2026-06-14). Supersedes exp4168 v1/v2, both INVALID nulls.**
**Origin:** operator pressure-test 2026-06-14 02:31 UTC — "are we absolutely sure this
is a valid null?" Answer: no. v2 was an underpowered, degenerate null. This note specifies
the valid version.

## Why v2's null was invalid (the three defects)

Artifact: `results/experiment_4168_decisive_verifier_graft_v2_gate082.json`.

1. **No headroom + tiny N (rerank arm).** Evaluated on `val -> _valsmall` = **64 puzzles**.
   vote@1 = 0.797, verifier@1 = **0.8125 = oracle@K**. The verifier hit the oracle ceiling
   (it selected optimally), but oracle@K − vote@1 = **0.0156 = exactly 1 puzzle of 64**
   (`verifier_recovers_outvoted: 1`). CI95 [0, 0.047] "includes 0" is *mechanically
   guaranteed* by a 1/64 effect — zero statistical power. This is the FALSE_NEGATIVE_RISK
   trap (oracle barely exceeds baseline → null is uninformative).
2. **The RFT arm never trained.** `rft_native_training_launched: False`,
   `training_mode: matched_label_deconfound_no_native_training`, `a_exact_accuracy: 1.0`
   (a matched-label construction artifact). The headline claim "verifier-reward training ≈
   vote-reward training" was *never tested* — a proxy was substituted.
3. **Degenerate candidate diversity.** `candidate_source: final_logits_k_sampling` on a
   near-deterministic model → the K samples are near-identical → oracle@K ≈ greedy, so there
   is nothing for any selector (or reward) to recover. When the TRM is wrong it is wrong in
   *all* K samples.

Net: the one real signal (verifier@1 == oracle@K) is mildly *encouraging* for the verifier;
the "null" reflects test design, not the verifier.

## Grounded facts (verified 2026-06-14)

- Full `test` split = **4227 puzzles** (`metadata.json: num_test`), 9×9, vocab 12, seq_len 81.
  `val` is a symlink to `_valsmall` (64) — v2's N bug.
- Data is plain `.npy` (`all__inputs.npy`, `all__labels.npy`, `all__puzzle_indices.npy`, …)
  → a STaR rejection-sampling corpus can be written in the same format and **really**
  SFT-trained by swapping `data.data_dir` (train.py is Hydra; datamodule is `SudokuDataModule`).
- `src/nn/sudoku_evaluator.py:SudokuEvaluator` loads the TRM (`TRMModule.load_from_checkpoint`)
  + has `check_sudoku_validity`. The TRM forward is the recursive carry/halt loop
  (`initial_carry` → `inner_forward` until `all_halted`); final logits → argmax (greedy) or
  temperature+multinomial (sampling).
- Baseline checkpoint: `results/trm_runs/sudoku_extreme_baseline/last.ckpt`, banked val 0.8227
  (`last.ckpt.bestval`). Operator-approved faithful baseline (gate 0.82).

## Stage A — HEADROOM PROBE (cheap; gates everything; the positive control)

Question: with better sampling on the **full 4227-puzzle test set**, is there real
recoverable headroom (oracle@K ≫ vote@1)? If not, the rerank/RFT moat is *genuinely* absent
on Sudoku-at-0.82 and Stage B is not worth building.

Method (`scripts/exp_verifier_headroom_probe.py`, outer-loop-owned, GPU 1):
- Load the 0.82 checkpoint via SudokuEvaluator (`eval_split="test"`).
- For each puzzle, generate K candidates (K ∈ {8, 16}) by temperature sampling the final
  logits across a sweep T ∈ {0.7, 1.0, 1.3}; also harvest intermediate-recursion-step
  predictions as additional diverse candidates (the TRM's halt steps give distinct guesses).
- Per puzzle compute: greedy@1 (argmax), vote@1 (majority candidate), verifier@1 (candidate
  maximizing executable-verifier constraint satisfaction; exact-valid preferred),
  oracle@K (any candidate exactly correct).
- Aggregate over all 4227. Report, per (T, K): vote@1, verifier@1, oracle@K,
  **headroom = oracle@K − vote@1**, verifier_lift = verifier@1 − vote@1, with bootstrap CI95.

**GATE:** proceed to Stage B only if some (T, K) yields **oracle@K − vote@1 ≥ 0.05** (real,
recoverable headroom) on N=4227. If max headroom < 0.05 across the sweep → the honest finding
is "Sudoku at this baseline has no recoverable rerank headroom; the verifier-as-selector moat
is not demonstrable here" → do NOT build Stage B; report and move the question to a
higher-headroom domain (harder corpus, or a non-near-ceiling task).

## Stage B — NATIVE RFT de-confound (only if Stage A passes)

Real STaR / rejection-sampling fine-tuning — NO proxy. `rft_native_training_launched` MUST be
True or the task DEFERS honestly (`blocked_native_rft_not_run`); the matched-label proxy is
BANNED.

1. From the 0.82 checkpoint COPY, generate K candidates per *training* puzzle (the Stage-A
   sampling config that produced headroom).
2. **Arm A (verifier-cert):** keep candidates the executable verifier certifies valid →
   STaR corpus A (`.npy`, sudoku format).
3. **Arm B (vote-cert):** keep the majority-vote candidate → STaR corpus B, **N-matched** to A
   (same number of training examples).
4. Resume-SFT each arm from the SAME 0.82 copy via train.py (`data.data_dir=corpusA|corpusB`,
   matched SFT steps, mode=default), outer-loop-owned (conductor off GPU training), with the
   persistence + reaper-exemption fixes already in place.
5. Eval BOTH arms on the **full 4227 test set**; report held-out exact-accuracy delta A−B with
   bootstrap CI95. `verifier_value_added := delta > 0 with CI95 excluding 0`.

N rigor: STaR corpora in the hundreds–thousands; eval N=4227. Sample-size floor per CLAUDE.md.

## Execution ownership

Outer-loop owns Stages A+B (GPU training), per the 2026-06-13 decision (the conductor botched
v2's native RFT into a proxy AND caused the GPU-contention deaths). Conductor stays for
non-training work. The reaper exemptions (`gpu_monitor`, `experiment_template`, commits
1254821ca/c02f14ea8) + the nano-trm persistence save protect any train.py run.

## What each Stage-B outcome means

- **A > B (CI95 excl 0):** verifier-as-reward beats self-consistency where the verifier
  executes → the moat is real → opens the DiffusionGemma energy-guided gate.
- **A ≈ B with real headroom:** a *valid* null — the verifier's exact-checking does not add
  training signal beyond vote even when headroom exists → do NOT build DiffusionGemma; the
  bottleneck is what the verifier can discriminate, not the substrate.
- **Stage A fails (no headroom):** inconclusive on Sudoku; move to a higher-headroom domain
  before judging the thesis. NOT a refutation.

## STAGE A RESULT (2026-06-14) — the experiment is STRUCTURALLY VOID on this generator

The headroom probe (`scripts/exp_verifier_headroom_probe.py`) was built and run on the test
set. Diagnostic (32-puzzle batch, then full run):

- **The TRM is effectively DETERMINISTIC.** Final-step logits are saturated: max |logit| = 185,
  **top1−top2 gap ≈ 61**. Temperature sampling does almost nothing — at T=8, only ~2 of 81
  cells change; oracle@K = vote@1 = greedy@1 across T ∈ {0.7 … 8.0}.
- **The recursion trajectory has diversity, but it is WORSE-answer diversity.** Earlier
  supervision steps differ from the final (31→17→…→9→0 of 32 puzzles across the 16 steps), but
  those are the model's less-converged, *wrong* intermediate guesses. The converged final IS its
  best answer.
- **Therefore `oracle@K ≈ greedy@1` (≈0.73 on test); headroom ≈ 0.** When the TRM is wrong, it is
  wrong in *every* candidate — no recoverable correct answer exists in the pool.

**Implication — both stages are structurally inapplicable here, NOT a verifier failure:**
- Rerank: nothing to select among (one effective candidate) → verifier can't beat vote.
- Native RFT de-confound: the verifier-cert corpus and the vote-cert corpus would be the SAME
  set (one candidate → verifier and vote never disagree) → A≈B by construction. Building Stage B
  on this baseline would manufacture a guaranteed null.
- This is the rigorous explanation of v2's null: the verifier-vs-self-consistency question is
  **undefined on a deterministic generator**. It says nothing about the verifier.

**What a VALID test of the verifier-as-reward moat actually requires:** a generator that emits a
genuine *distribution* of candidates with **recoverable correct answers when it errs** — i.e. a
regime with real epistemic uncertainty, not a near-solved task with saturated logits. Concretely:
1. A **harder / under-trained corpus** where the model is genuinely uncertain (logit gaps small,
   oracle@K ≫ greedy), or
2. An **LLM-class generator** with natural sampling diversity — which is exactly the hybrid
   open-LLM-generator + energy-verifier architecture, and the DiffusionGemma direction.

**Net:** TRM-on-near-solved-Sudoku is the worst possible testbed for the selection/reward moat
(deterministic + saturated). The moat question is not answerable here at all — it must move to a
diversity-bearing generator. This redirects the program away from "graft on the TRM baseline"
and toward measuring the moat where candidates actually vary.

## Cross-references
- `results/experiment_4168_decisive_verifier_graft_v2_gate082.json` — the invalid v2 null
- CLAUDE.md "FALSE_NEGATIVE_RISK" / "Adversarial Artifact Verification + Sample-Size Rigor"
- `docs/research-notes/diffusiongemma-energy-guided-diffusion-spec.md` — the gated scale-up
- `results/trm_runs/sudoku_extreme_baseline/last.ckpt` — the 0.82 faithful baseline
