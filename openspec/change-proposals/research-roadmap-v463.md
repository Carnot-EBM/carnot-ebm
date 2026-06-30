# Research Roadmap v463 — PHASE D (THIRD execution): finally LAND the oracle-distinct verifier-moat measurement

**Milestone:** 2026.06.463
**Planner:** outer-loop Claude Opus 4.8, 2026-06-30 (UTC)
**Prior milestone:** 2026.06.462 (PHASE D, SECOND execution — EXECUTION-INCOMPLETE)
**Theme:** Three PHASE D milestones, zero real trained-verifier measurements. .461 and .462
both went EXECUTION-INCOMPLETE for *pure infrastructure* reasons, never science. .463 fixes the
two precise .462 root causes, DECOUPLES the headline arm from the fragile dependency that keeps
cascading, routes the twice-failed load-bearing arms OFF codex (per the .462 capstone pointer),
and finally measures whether a TRAINED oracle-distinct verifier captures the proven 28pp MuSR
headroom.

---

## 1. What the previous two milestones proved (and failed to prove)

### The opportunity is REAL (.462 B1 succeeded)
The one .462 arm that executed cleanly was B1 (exp5015, the genuine self-consistency baseline fix):

| Quantity | Value | Meaning |
|---|---|---|
| genuine tuned-SC accuracy (MuSR) | **0.585** | the honest K-way majority-vote baseline (K-sweep: K1=0.585, K3=0.53, K5=0.575 — voting doesn't help; only 5 noisy candidates/q) |
| oracle@K (selectable ceiling) | **0.865** | the best-achievable if a verifier always picked the correct candidate |
| **selectable headroom** | **+0.28** | oracle@K − SC: a *large*, genuine, oracle-distinct headroom |
| genuine_headroom_present | **true** | passes the FALSE_NEGATIVE_RISK guard against the genuine baseline |
| degeneracy_guard_fires | **true** | an always-abstain (>50%) selector is now flagged, not counted as a tie |

So MuSR is a confirmed **headroom-present, oracle-distinct** domain (`verifier_is_oracle=False`).
A verifier that selects toward 0.865 from 0.585 would be the first realized oracle-distinct moat.
The cheap *prompted* energy proxy already nulled here (SC best, energy 0.515–0.535) but
`sc_saturated=False` — the headroom is **unrealized, not absent.** The open question (operator
directive 2026-06-30): does a *TRAINED* verifier capture it?

### The moat is still UNTESTED — both PHASE D milestones failed to EXECUTE
Zero real trained-verifier numbers have landed. The .462 capstone (exp5027) verdict:
`complete_capstone_v462_moat_execution_incomplete_ebrm`, `moat_realized=false`,
`moat_retired_bounded=false`, pointer = **"rerun_unexecuted_arm; route off Codex if the same arm
bails twice."** The precise .462 root causes (all infra, all fixable):

| Arm | .462 verdict | Root cause | .463 fix |
|---|---|---|---|
| **D1** LoRA-EBM | `blocked_trainable_qwen_base` (flagged) | `Qwen/Qwen3.5-1.7B` is a **404 — the planner named a nonexistent HF repo** | use a REAL cached base (`Qwen/Qwen3.5-2B`) via a prioritized resolver that probes a list (a single 404 can never kill it again) |
| **B2** logprob cache | `blocked_generation_or_cache_error`, **0 rows in 379s** | regenerate-from-scratch loop wrote nothing under the cap | RE-SCORE the 200 existing cached candidates (not regenerate), INCREMENTAL per-row append, per-question try/except |
| **D2** uPRM | `blocked_b2_logprob_cache` (flagged) | cascaded from B2's 0 rows | add a self-supervised frozen-candidate PRM fallback (no logprobs) so D2 is not single-point-blocked |
| **D3** EBRM | `blocked_gate_check_failed` | gated_on D1 (which never trained) | gated_on the REAL D1 scorer; refine it with uncertainty heads |

### The .462 lesson, encoded in .463's structure
A **skeleton / blocked / degenerate arm is a FAILED EXECUTION, not a scientific null.** The
`retire_if_same_verdict` bounded-retirement only fires when the PROPERLY-EXECUTED D1 (real
training, `train_loss` non-null) AND D2 both *clean-null* (CI95 incl 0) on a headroom-present
corpus. After three execution failures, .463's first job is to make the arms EXECUTE.

---

## 2. The three structural decisions that de-risk .463

1. **DECOUPLE the headline (D1) from the fragile dependency (B2).** The .462 cascade was
   B2-cache → D2 → (D3 gated_on D1, D1 dead on the 404). D1 does **not** need logprobs: it trains
   on candidate TEXT + gold (both already in the 200 cached MuSR checkpoints that B1 used
   successfully) and evals by re-scoring those candidates. So in .463, **D1 depends only on the
   trainer module (B3) + the existing cached candidates + the B1 genuine-SC baseline — never on
   B2.** Even if B2 fails a third time, the headline still lands.

2. **Ship a permanent fix for the 404 / skeleton-bail class as reusable infra (B3).** A new module
   `python/carnot/moat_trainer.py` exposes (a) `resolve_trainable_base()` — probes a prioritized
   list of REAL cached bases and returns the first present (kills the hallucinated-repo class
   forever); (b) `train_energy_head()` — QLoRA + scalar energy head, checkpoint-per-epoch,
   resumable; (c) a **60-second smoke** that trains a few steps on a real base and checkpoints,
   *proving the pipeline end-to-end before D1's full run*. D1 is `gated_on B3.smoke_passed`, so a
   broken trainer skips D1 cleanly instead of burning a full-train attempt.

3. **Route the twice-failed load-bearing arms OFF codex** (per the .462 capstone pointer "route
   off Codex if the same arm bails twice"). B3 (the trainer) and D1 (the headline) run
   `agent_type: claude` + `model: opus` + `requires_claude_verified: true` — the multi-step
   train→checkpoint→eval→adversarial-verify choreography on a path that has failed twice is
   exactly the `requires_claude` positive criterion. Everything else stays codex (the operator's
   standing quota-conserve default); planner/retro stay Opus.

---

## 3. Architecture (the .463 PHASE D dependency graph)

```
                 [exp5028 PHASE 0: archive .462 -> activate .463]
                                    |
        +---------------------------+-----------------------------+
        |                           |                             |
  [exp5029 B2 (codex)]      [exp5030 B3 (claude/opus)]     (B1 already shipped .462:
   robust logprob cache      trainable-base resolver        genuine SC 0.585, oracle 0.865,
   = RE-SCORE the 200         + cap-surviving LoRA-EBM       degeneracy guard — reused as-is
   cached candidates,         trainer + 60s SMOKE            via moat_benchmark_harness.py)
   incremental append              |
        |                          | smoke_passed
        |                          v
        |                  [exp5031 D1 (claude/opus) HEADLINE]
        |                   TRAIN LoRA-EBM (real base) on FoVer + gold-labeled
        |                   cached MuSR pairs; EVAL min-energy select vs genuine SC.
        |                   DECOUPLED from B2. The decisive measurement.
        |                          |
        | candidate_cache_built    | scorer_trained
        v                          v
  [exp5032 D2 (codex)]      [exp5033 D3 (codex)]
   uPRM first-error          EBRM: refine the REAL D1 scorer + uncertainty
   over logprobs;            (UARM/pessimistic + CoT-Entropy 2502.11250
   self-supervised PRM       + conformal CROP 2605.30085), capped abstention <=50%
   FALLBACK (no logprobs)           |
        |                          |
        +------------+-------------+------------------+
                     |                                |
        [exp5034 D6 cascade (codex)]          [exp5035 D4 (codex)]
         VERDI confidence routing             best D1/D2/D3 arm on a 2nd
         (2605.11334); Pareto =               headroom-present corpus
         parity at fewer judge calls          (GPQA / MMLU-Pro-hard / MATH-500-hard)
                     |                                |
                     +---------------+----------------+
                                     v
                        [exp5036 D5: moat GATE + DiffusionGemma status]
                                     |
   reserved/continuity (parallel): [exp5037 C KV260] [exp5038 E1 SOTA-ingest]
                                   [exp5039 E2 self-play] [exp5040 E3 opportunistic ARC]
                                     |
                        [exp5041 CAPSTONE v463 -> .464 pointer]
```

## 4. Phases

- **PHASE 0 — Transition** (exp5028): archive .462 → activate .463; own-test `--no-cov`; record the
  3rd-attempt close-state (root causes: D1 404 base, B2 0-rows; B1 reusable; levels LOCKED at 69).
- **PHASE B — Infra (2 reserved slots)**:
  - B2 (exp5029): robust logprob candidate cache by RE-SCORING the 200 existing cached candidates,
    incremental per-row append, resumable. Feeds D2 (uPRM) only — NOT the headline.
  - B3 (exp5030): `moat_trainer.py` — real-base resolver + cap-surviving LoRA-EBM trainer + 60s
    smoke. The permanent fix for the 404 / skeleton-bail class. claude/opus.
- **PHASE D — Verifier moat**:
  - D1 (exp5031, HEADLINE, claude/opus): train the LoRA-EBM (arXiv:2605.18871) on a real base,
    eval min-energy select vs genuine SC. Decoupled from B2. The decisive measurement.
  - D2 (exp5032): replicate uPRM (arXiv:2605.10158) over the B2 cache + self-supervised
    frozen-candidate PRM fallback (arXiv:2507.01951 / 2502.14356).
  - D3 (exp5033): EBRM (arXiv:2504.13134) refining the REAL D1 scorer + uncertainty
    (UARM 2606.19818 / pessimistic 2606.09073 / CoT-Entropy 2502.11250) + conformal abstention
    (CROP 2605.30085), capped <=50%.
  - D6 (exp5034): efficiency cascade (arXiv:2510.20369) with VERDI confidence routing
    (arXiv:2605.11334); Pareto win = accuracy parity at materially fewer judge calls.
  - D4 (exp5035): best arm on a 2nd headroom-present oracle-distinct corpus.
  - D5 (exp5036): the falsifiable gate + DiffusionGemma status (operator-gated, not auto-flipped).
- **PHASE C — Hardware continuity** (exp5037): KV260 SSH-only reachability + on-board energy smoke.
- **PHASE E — Reserved / continuity**:
  - E1 (exp5038): fresh SOTA-ingestion for .464 (reliable channel; /deep-research banned).
  - E2 (exp5039): continuous self-learning — ARC self-play, train+checkpoint the learned verifier.
  - E3 (exp5040): opportunistic ARC level-up (honest no-bank; well flat at 69 for 8+ milestones).
- **CAPSTONE** (exp5041): aggregate the moat verdict + the .464 pointer.

## 5. The falsifiable gate (the only non-circular evidence)

On a headroom-present ORACLE-DISTINCT domain (`verifier_is_oracle=False`, `headroom_present=True`
vs the GENUINE tuned-SC), **≥1 of {trained LoRA-EBM (D1, `scorer_trained=true`), uPRM (D2), EBRM
(D3, non-degenerate)} beats the genuine tuned-SC with paired CI95 excluding 0 (McNemar p<0.05),
confirmed on MuSR AND ≥1 second corpus (D4)** — OR the cascade (D6) reaches accuracy parity at
materially fewer judge calls (a north-star §5 efficiency Pareto win).

- **POSITIVE** → the off-ARC verifier moat is REALIZED; record the DiffusionGemma gate conditions
  satisfied on the tested domain (activation stays operator-gated — do NOT auto-flip to MET).
- **BOUNDED RETIREMENT** (`retire_if_same_verdict`) → only if the PROPERLY-EXECUTED D1
  (`scorer_trained=true`) AND uPRM (D2) BOTH clean-null on every headroom-present corpus AND no
  efficiency win. A publishable bounded null converging with the ARC tie.
- **EXECUTION-INCOMPLETE** (a skeleton / blocked / degenerate arm) → NOT a null; .464 re-runs it.

## 6. Anti-traps (carried from the discipline + the .461/.462 lessons)

- GENUINE K-way tuned-SC baseline (B1), never a k=1 strawman (the .461 D3 degeneracy).
- `verifier_is_oracle=False` + `headroom_present=True` on every arm (circularity + FALSE_NEGATIVE_RISK).
- `scorer_trained` gate REQUIRES `train_loss` non-null + `n_pairs>0` + `duration_s>60` (anti-skeleton).
- always-abstain degeneracy guard (abstention >50% flagged, not counted as a tie).
- n>=200, paired bootstrap + McNemar; oracle-distinctness lint; skip any `flagged_adversarial` artifact.
- A real-base RESOLVER (probes a list) — a single hallucinated repo id can never block D1 again.

## 7. Hardware requirements

- **GPU-0 CUDA (RTX 3090, conductor-owned, 2026-06-27 allocation)** for D1/B3 training + B2/D2/D6
  generation/scoring. NOT iGPU-pinned (the iGPU constraint is for the LIVE ARC submission stack
  only; offline PHASE D uses GPU-0).
- **KV260 via SSH** (`ssh kria`) for the hardware-continuity slot — SSH-only, never host SD-card.

## 8. Models

- **D1/B3 trainable base:** `Qwen/Qwen3.5-2B` (cached, real, safetensors) — prioritized resolver
  fallback list `[Qwen3.5-2B, Qwen3.5-0.8B, Qwen3-4B, Qwen2.5-0.5B]`, all cached.
- **B2/D2/D6 generator/judge:** `unsloth/gemma-4-12B-it-GGUF` on the GPU-0 CUDA llama-server.
- **Contrastive training data:** `data/fover_train_v4.json` (step-labeled) + gold-labeled cached
  MuSR candidates (`results/distributional_energy_verifier_musr_checkpoints/`, 200 q × 5 cand).

## 9. SOTA ingested this cycle (consumed from the .462 E1 handoff,
`docs/research-notes/verifier-moat-literature-2026-06-30.md`)

| arXiv | Method | Mapped to | Use in .463 |
|---|---|---|---|
| 2507.01951 / 2502.14356 | self-supervised process rewards (frozen candidates) | D2 | logprob-free fallback (removes the B2 single-point-block) |
| 2502.11250 | CoT-Entropy uncertainty-aware PRM | D2 / D3 | uncertainty-aware abstention head |
| 2605.30085 | CROP-style conformal abstention | D3 | calibrated, capped abstention gate |
| 2605.11334 | VERDI single-call decomposed judge confidence | D6 | cheap confidence router (no extra judge calls) |

A FRESH ingestion (E1, exp5038) runs for .464 beyond the now-23 ingested papers, per the
SOTA-Ingestion Cycle Discipline (the .462 pass is consumed here; /deep-research stays banned).

## 10. Decentralization implications (CLAUDE.md rules 1–7)

Local-first throughout: all bases are open-weight, locally cached; the verifier scorer/checkpoints
mirror per Rule 3 (`models/`, HuggingFace + IPFS). No closed-weight dependency. The energy verifier
is the hardware-acceleratable primitive (KV260 energy-eval continuity). Generator stays commodity/
local (gemma-4-12B-it); the verifier is Carnot's value-add.
