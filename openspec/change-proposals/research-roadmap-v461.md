# Research Roadmap v461 — PHASE D: the off-ARC distributional-energy verifier moat (FIRST execution)

**Milestone:** 2026.06.461
**Planned by:** outer-loop Claude Opus 4.8 planner, 2026-06-29 (UTC)
**Supersedes/continues:** v460 (the final-stretch ARC sprint milestone, completed cleanly)
**Status of the ARC sprint:** RETIRED 2026-06-30 (deadline reached + operator lifted it; CLAUDE.md "ARC-AGI-3 Submission Sprint Forcing Function" STATUS). ARC banked-level work continues OPPORTUNISTICALLY but no longer claims the majority.

---

## 1. What the previous milestone (.460) proved, and why we pivot now

`.459`/`.460` closed the ARC sprint cleanly. The capstone (exp4989) recorded
`complete_capstone_v459_submission_ready_levels_69_heldout_0.04_package_ready_pivot_turnkey_7_1`:

- **The ARC deliverable is LOCKED.** `reproducible_total_levels = 69` (flat for 6+ milestones — the deepen well is dry across ALL depth regimes); the held-out first-win rate is a CLEAN full-25 `0.04` (`flag_resolved=true`); the live submission package is READY (15.146 GB, Qwen3.5-9B-MTP on the iGPU, operator-only). The `.453` B1-trusted `WALL_IS_HIDDEN_STATE` closure STANDS — the live first-win wall is representation-invariant, so chasing it with "representation #5" is not productive.
- **The CUDA submission is staged** (`docs/research-notes/arc-agi3-cuda-submission-runbook-2026-06-30.md`); its score is bounded at ~0.08 by the candidate-GENERATION wall (every ARC selection/verifier lever nulled — `project_arc_l1_first_contact_wall`, `project_arc_generation_not_selection`). No lever moves it before the deadline.
- **The S0 oracle-distinct STRUCTURAL-energy program CONCLUDED 2026-06-26** (`docs/research-notes/oracle-distinct-structural-energy-program-2026-06-26.md`): a real OFFLINE cross-game discriminator that adds NO live ARC value. **Do NOT re-propose S4 / energy-as-ARC-lever stages.**

**The operator directive (2026-06-30, known-issues.md MANDATORY-NEXT-MILESTONE):** *"unlock the conductor running PHASE D experiments immediately."* The planner's MAJORITY now shifts from ARC live-solving to **executing PHASE D — the off-ARC distributional-energy verifier moat.** This is the post-6/30 pivot that `.451`–`.460` held TURNKEY (exp4951/4962/4973/4984/4995 dry-run scaffolds + an 13-paper ingested backlog).

**Why off-ARC and why now.** ARC's generation wall cannot host the verifier-moat test: there is no SELECTABLE headroom (the winning candidate is never in the pool, so no selector — energy or otherwise — can win). The moat must be tested where (a) self-consistency is NOT saturated, (b) there is genuine selectable headroom (oracle@K − tuned-SC ≥ ~0.10), and (c) there is NO cheap executable oracle (so a learned/energy verifier is ORACLE-DISTINCT, not circular). MuSR murder-mysteries is exactly such a domain. This is the SAME thesis as the still-pending DiffusionGemma gate (§4), tested off ARC where the headroom is real.

**What is already DONE (do NOT re-run):** the cheap *prompted* energy proxy on MuSR
(`results/distributional_energy_verifier_musr.json`, n=200, 6531 s live run):

| method | accuracy | vs SC | McNemar p |
|---|---|---|---|
| self_consistency (baseline) | **0.560** | — | — |
| distributional_energy (abstain ensemble) | 0.515 | −0.045 (CI [−0.105,+0.015]) | 0.188 |
| distributional_energy (pure min-energy) | 0.535 | −0.025 (CI [−0.095,+0.045]) | — |
| llm_judge | 0.545 | −0.015 | 0.736 |

`energy_beats_sc = False` but **`sc_saturated = False`** — headroom is present and UNREALIZED by the cheap prompted proxy. **The open question PHASE D answers: does a TRAINED verifier capture the headroom the cheap proxy missed?**

---

## 2. The three biggest gaps between current state and the PRD vision

1. **The verifier moat is EXISTENTIAL but UNPROVEN off-ARC** (`feedback_hybrid_pragmatic_architecture`: "verifier-moat now EXISTENTIAL — Carnot's whole value-add"). Every win to date is *execution-grounded / circular* (the verifier IS the oracle: HumanEval tests, `check_sudoku_validity`). The OPEN, deep claim — an oracle-distinct learned/energy verifier capturing headroom where no cheap oracle exists — has never landed with a matched control. **PHASE D is the first honest attempt off ARC.**
2. **We have a published WIN-CONDITION EXISTENCE PROOF we have not replicated.** uPRM (arXiv:2605.10158) — an *unsupervised* process reward model — beats majority-voting/self-consistency by up to 6.9% on ProcessBench. Replicating it tells us which domain + construction beats SC, de-risking the LoRA-EBM bet. We keep re-deriving "does a verifier beat SC" instead of standing on this giant's shoulders (`feedback_literature_priority_discipline`).
3. **Continuous self-learning (PRD FR-11) has no off-ARC verifier instance.** The ARC self-play loop trains+checkpoints a learned verifier across runs, but the project's verifier-as-a-product needs a continual-improvement path on the reasoning corpora it will actually verify. The LoRA-EBM scorer + the self-play checkpoint are two arms of the same self-learning thesis (research-program §"Continuous Self-Learning" Tier 3).

---

## 3. Architecture — where PHASE D sits

```
                 PHASE D: the oracle-distinct verifier moat (OFF-ARC, MAJORITY)
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  generator (SOTA GGUF: gemma-4-12B-it / Qwen3.5-9B-MTP, GPU-0 CUDA)         │
  │      │  K reasoning+answer candidates per question (temperature diversity)  │
  │      ▼                                                                       │
  │  ┌──────────── headroom-present, ORACLE-DISTINCT corpora ───────────────┐   │
  │  │  MuSR murder_mysteries (sc_saturated=False)  +  a SECOND corpus       │   │
  │  │  (GPQA / MMLU-Pro-hard / MATH-500-hard) — oracle@K − tuned-SC ≥ 0.10  │   │
  │  └──────────────────────────────────────────────────────────────────────┘   │
  │      │                                                                       │
  │      ▼  three ORACLE-DISTINCT verifier constructions (verifier ≠ oracle)     │
  │   D1  TRAINED LoRA-EBM holistic-quality scorer   (arXiv:2605.18871)          │
  │   D2  uPRM unsupervised process RM (next-tok-prob first-error) (2605.10158)  │
  │   D3  EBRM energy RM modeling reward DIST + uncertainty (2504.13134)         │
  │      │                                                                       │
  │      ▼  selection accuracy vs TUNED self-consistency                         │
  │   GATE: beats tuned-SC, McNemar/paired-bootstrap CI95 excludes 0,           │
  │         verifier_is_oracle=False, headroom_present=True                      │
  │      │                                                                       │
  │      ▼                                                                       │
  │   D4  cross-corpus generalization (best verifier on the 2nd corpus)          │
  │   D5  moat-gate aggregation  ->  resolves the DiffusionGemma gate (§4)       │
  └──────────────────────────────────────────────────────────────────────────┘
         supported by:  B1 reusable moat benchmark harness (oracle@K, SC, CI)
                        B2 oracle-distinctness + headroom-control adversarial lint
         continuity:    C KV260 SSH-only  ·  E1 SOTA-ingestion  ·  E2 self-play
                        (continuous self-learning / FR-11)  ·  E3 opportunistic ARC
```

**Generator vs scorer (the oracle-distinctness invariant).** ALL candidates come from ONE generator model; the verifier scores reasoning QUALITY and never reads `answer_index`/`answer_choice`/`model_id`. Gold labels are used for EVAL accuracy ONLY (and, for the supervised D1/D3 constructions, to build the TRAINING contrastive pairs — standard RM training; the verifier is oracle-distinct at INFERENCE). uPRM (D2) is fully unsupervised (no gold even in training).

---

## 4. The DiffusionGemma gate this resolves (the same thread)

`docs/research-notes/diffusiongemma-energy-guided-diffusion-spec.md` GATE is **STILL-PENDING**. It is MET only when ALL three hold:

1. **Headroom real and present** (oracle@K − tuned-SC ≥ ~0.10) on the test domain.
2. **The verifier is NON-TRIVIAL** — NOT identical to the executable oracle (a learned/energy/partial-constraint signal that could transfer to a domain WITHOUT a cheap oracle).
3. **That non-trivial verifier captures the headroom with a MATCHED no-verifier control**, `verifier_value_added=true`, CI95 excluding 0, on an oracle-distinct domain.

PHASE D is precisely conditions 1–3 measured off ARC. A PHASE D positive (any of D1/D2/D3 beats tuned-SC, CI95-excl-0, headroom-present, oracle-distinct) satisfies the gate's stated conditions on the tested domain and MOVES the gate toward MET (D5 records this honestly; ARC's ~13pp headroom remains the canonical un-captured target, and DiffusionGemma activation stays operator-gated). A PHASE D all-null **retires the off-ARC verifier moat as bounded** — a publishable null converging with the ARC tie.

---

## 5. Phases and the falsifiable gate

| Phase | Tasks | What it decides |
|---|---|---|
| **0 — transition** | exp5001 | archive .460 → activate .461; record the close-state; resolve any poison pre-test |
| **B — infra (2 reserved slots)** | exp5002 (B1 harness), exp5008 (B2 lint) | a reusable oracle-distinct moat benchmark harness (oracle@K, tuned-SC, McNemar/bootstrap CI, headroom flag) + an adversarial lint enforcing oracle-distinctness + headroom-control on moat artifacts |
| **D — the verifier moat (MAJORITY)** | exp5003 (D1 LoRA-EBM), exp5004 (D2 uPRM), exp5005 (D3 EBRM), exp5006 (D4 2nd corpus), exp5007 (D5 gate aggregation) | does a TRAINED/principled oracle-distinct verifier beat tuned-SC on a headroom-present, oracle-distinct domain? |
| **C — hardware continuity** | exp5009 (KV260 SSH-only) | KV260 reachability + small on-board energy smoke (write-blocked-artifact pattern) |
| **E — ingestion + self-learning + opportunistic ARC** | exp5010 (E1 SOTA-ingestion), exp5011 (E2 self-play / FR-11 continuous self-learning), exp5012 (E3 opportunistic ARC level-up) | keep SOTA flowing into the moat program; advance the continuous-self-learning verifier; bank an opportunistic ARC level if the well is not dry |
| **capstone** | exp5013 | aggregate the milestone; report the moat verdict + gate status; never aggregate `flagged_adversarial` artifacts |

**THE milestone gate (falsifiable, the only non-circular evidence):** on a headroom-present ORACLE-DISTINCT domain (`verifier_is_oracle=False`, `headroom_present=True`, the verifier is NOT the executable oracle), at least one of {trained LoRA-EBM (D1), uPRM (D2), EBRM (D3)} achieves selection accuracy beating TUNED self-consistency with paired-test CI95 excluding 0 (McNemar). `retire_if_same_verdict: true` — if the trained LoRA-EBM AND the uPRM replication BOTH fail to beat SC with CI95-excl-0 on any headroom-present oracle-distinct corpus, the off-ARC verifier moat retires as bounded (D5 writes the retirement). A POSITIVE is the discriminating energy that also resolves the DiffusionGemma gate (§4).

**Anti-traps baked into the design (CLAUDE.md disciplines):**
- *FALSE_NEGATIVE_RISK:* every D task reports `oracle_at_k` and `headroom_present`; a null is only informative on a corpus where oracle@K − tuned-SC ≥ ~0.10 and flips > 0. A corpus with no selectable headroom yields an uninformative null and is NOT counted as a moat-bounding result.
- *Circularity / oracle-distinctness:* every D artifact declares `verifier_is_oracle=False` and must pass `check_circular_moat_overclaim`. The verifier never sees gold at inference.
- *Headroom-control:* tuned-SC (temperature/K swept) is the baseline, not naive SC — so a "win" is not an artifact of an un-tuned baseline.
- *Sample-size rigor:* n ≥ 200 per corpus for the headline accuracy delta; paired bootstrap + McNemar.
- *Pre-launch preconditions:* GGUF/base-model cached, GPU-0 reachable, corpus cached — checked BEFORE any inference; missing → `blocked_<resource>`.

---

## 6. Dependency graph

```
exp5001 (transition)
   └─> exp5002 (B1 harness) ──────────────┐  (enabling; D tasks reuse but do not hard-block on it)
                                           v
        exp5003 (D1 LoRA-EBM) ─┐
        exp5004 (D2 uPRM) ─────┤
        exp5005 (D3 EBRM) ─────┼─> exp5006 (D4 2nd corpus, best verifier) ─> exp5007 (D5 gate aggregation)
                               │                                                   │
        exp5008 (B2 lint) <────┘  (lints D1-D4 artifacts)                          │
   exp5009 (C KV260)  exp5010 (E1 SOTA)  exp5011 (E2 self-play)  exp5012 (E3 ARC)  │
                                                                                   v
                                                                          exp5013 (capstone v461)
```

D4 reads whichever of D1/D2/D3 produced the strongest oracle-distinct verifier; D5 reads D1–D4. Neither hard-gates (each falls back to the cheap-proxy control if an upstream arm blocks), so an upstream block degrades gracefully rather than cascade-skipping.

---

## 7. Hardware requirements

- **GPU-0 (RTX 3090, CUDA, conductor-owned per 2026-06-27 allocation):** D1 LoRA-EBM training (QLoRA, modest base + cached candidates → < 30 min), D2/D4 live generation with logprobs. Accept the conductor's GPU-0 CUDA device; do NOT iGPU-pin or hard-reject `CUDA_VISIBLE_DEVICES`.
- **iGPU:** the LIVE ARC submission stack only (frozen, not touched this milestone).
- **KV260 (kria, SSH-reachable):** C continuity smoke — `ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`, then a small on-board energy eval; SSH-only, NEVER host SD-card.
- **No new FPGA bitstream work** (GateMate/PolarFire attached but under the north-star KV260-focus relaxation; not allocated this milestone).

## 8. Models (CLAUDE.md SOTA-models rule)

- **Generator (mandated SOTA GGUF):** `unsloth/gemma-4-12B-it-GGUF` (the lightweight SOTA — best throughput for many-call candidate generation; verify cached in PRECONDITIONS) for any NEW generation. D1 REUSES the cached MuSR candidates (Qwen3.5-9B-MTP-GGUF) to avoid regeneration churn.
- **Scorer base (trainable, D1/D3):** a trainable HF base (`Qwen/Qwen3.5-4B` base or `Qwen/Qwen3.5-1.7B` for wall-clock) + a LoRA adapter + energy head. Point at the BASE repo, NOT the `-GGUF` repo (GGUF tokenizer rule). Scaling the scorer to 9B/35B is explicit follow-on.

---

## 9. What this milestone does NOT do (do-not-re-propose ledger)

Representation #5 (ARC first-win wall is B1-trusted CLOSED); energy-as-ARC-lever (S0 program CONCLUDED 2026-06-26 — no live ARC value); the ARC dynamics-engine L2 wall (multi-week post-sprint); macro/horizon-collapse; click-heatmap generator; trust-gate flip; MATM similarity-retrieval (NULLED .454); TTT-on-code-engine; local code inducers; decision-need targets; action-prefix latents; coverage/exploration/selection/perception-from-grid. The **live-LoRA verifier-as-REWARD path** (retired exp4247, "operator reopen required") and **verifier-as-reward in-loop training** (retired exp4263) are DISTINCT from PHASE D: PHASE D trains the verifier/SCORER itself and uses it as an oracle-distinct SELECTOR — it does NOT use the verifier as an RFT reward to train a policy. The three PHASE D constructions are explicitly operator-directed (known-issues.md 2026-06-30), so each carries an `operator_override:` for the substring scope-match.

---

## 10. Agent routing

All experiment tasks: `agent_type: codex` + `model: gpt-5.5` (Codex-Default-v2). Planner + retro stay Claude Opus 4.8 via env (the operator quality choice). The transition + capstone are codex (no `requires_claude_verified`).
