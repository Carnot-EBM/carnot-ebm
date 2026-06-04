# Research Roadmap — Milestone 2026.06.348

**POST-CONVERGENCE — ADVANCE THE HEADLINE (G4 RESTORATION) + HARDEN/REPAIR THE BANKED PRODUCT**

**Planned:** 2026-06-04 (Claude Opus 4.8, outer-loop pre-staged roadmap)
**Supersedes activation of:** 2026.06.347 (POST-CONVERGENCE — retry + harden)
**Design doc for:** `research-roadmap-next.yaml`

---

## 1. What the previous milestone (.347) proved

`.347` landed all tasks. The project remains **converged** and the legitimate
forward surface narrowed further:

- **`paper_ready=TRUE` (G1–G4) holds; FoVer 0.9131 frozen.** Unchanged by
  `.347`; re-asserted at the `.347 capstone.
- **The last open energy-existential question was RETRIED and HANDED OFF.**
  `exp3787` (P1 discrete-search adjudication v3) **blocked on no-free-GPU for
  the SECOND consecutive milestone** and set `handoff_to_operator=true`. Per
  its own repeated-block retirement ramp, the in-loop discrete-search
  adjudication is now **handed to the operator** and **must NOT be re-queued**
  — the harness is already corrected (`n_train=40000`); the operator runs it
  directly when a GPU is free. `.348 does NOT re-queue it.
- **The operator's EDLM seed is PREFLIGHTED and GO.** `exp3793` returned
  `readiness=go`, confirmed the canonical repo (`MinkaiXu/Energy-Diffusion-LLM`,
  arXiv:2410.21357), and emitted the one-command seed. **Seeding remains the
  operator's call** (the P3 Verification Trap — the loop cannot self-seed a
  paradigm).
- **The product headline is PARTIALLY restorable.** `exp3792` (G4 provenance):
  `exp2090` (CRANE constrained decoding, n=50, 0.70→0.85, +15pp) **PASSES G4**;
  `exp1999` (code-verification HumanEval repair, 0.66→0.84, +18pp) **FAILS G4**
  — the on-disk artifact carries **no** `random_seed` / `reproducibility_checksum`
  / `n` / `inference_substrate`. Result: `not_yet_headline_eligible`. **A clean
  re-run of exp1999's pipeline with full provenance is the single
  headline-ADVANCING move available** (north-star §1's own definition of a
  non-noise milestone).
- **Tier-3 self-learning landed (exp3788):** the FR11ExtendedJEPA predictor
  reached **predictive AUROC 0.9715** on held-out FoVer, headline ensemble
  unchanged, memory contribution preserved, state persisted. It is **trained
  but not yet USED** — it is not wired as a fast-path gate.
- **Banked verifier product hardened, with two open follow-ups.**
  `exp3789` added the abstention **CLI + batch** surface (default-OFF, E2E).
  `exp3790` characterized **gaming-resistance** and found the ensemble
  **degrades under the `context_compaction` perturbation** (a real, named
  product weakness, not yet mitigated).
- **The endorsed process upgrade is NOT yet usable.** `exp3791` validated the
  `.346 Anomaly-Escalation classifier against the historical corpus and found
  **false-escalation rate 0.83** with frame-violating recall 1.0 →
  `supports_wiring_in=false`. The rules **need tuning** before the operator
  can wire it in.

## 2. The honest strategic state (read this before judging the plan for churn)

The project is **converged**. The substantive next move — **seed a new
Phase-3 thesis (EDLM is GO) or freeze into a product-maintenance cadence** — is
the **operator's** to make; the loop cannot self-seed a paradigm. `.348 does
NOT self-seed EDLM and does NOT re-queue the handed-off P1 GPU adjudication.

`.348 is therefore a **LEAN, NON-CHURN maintenance milestone** whose every
task maps to one of the four legitimate post-convergence surfaces (north-star
§1):

1. **ADVANCE the headline** — restore the demoted product code-repair number
   to G4-eligibility via a clean, provenance-complete re-run (exp1999 lineage).
2. **HARDEN/REPAIR the banked product** — mitigate the `context_compaction`
   gaming evasion; add a third integration surface (HTTP/REST) per
   Decentralization Rule 4.
3. **REPAIR the endorsed process tool** — tune the over-firing anomaly
   classifier (0.83 → target ≤0.2 false-escalation) and re-validate.
4. **CONTINUE mandated self-learning** — wire the trained Tier-3 predictor
   (exp3788) as a deployable fast-path/early-exit gate and measure the
   compute saving (the Partial-Reward-Model early-rejection pattern,
   arXiv:2508.01969).

Plus record hygiene (external refresh, KV260 opportunistic, capstone).

**OPERATOR FLAG (carried into the capstone + status):** the EDLM preflight is
GO and the P1 mechanism question is handed off. Milestones that continue
*without* an operator seed will increasingly approach the churn boundary; the
`.348 surface is real but finite. The next milestone after this one should
either be the operator's EDLM seed, or an explicit freeze.

## 3. The three gaps between current state and the PRD vision

1. **Deployable, robust product** — the verifier is banked and abstention-wired,
   but (a) a real gaming evasion (`context_compaction`) is uncovered and
   unmitigated, and (b) only the Python/MCP/CLI surfaces exist; HTTP/REST is
   missing (Decentralization Rule 4). `.348 closes both.
2. **Headline integrity** — the product code-repair headline is demoted because
   `exp1999` lacks provenance. `.348 restores it (or honestly confirms it still
   cannot be headlined).
3. **Self-learning that pays off at inference** — Tier-3 is trained but unused.
   `.348 turns it into a fast-path gate (the PRD's "learning at inference speed"
   principle) and measures the saving.

## 4. Phases

**Phase A — Milestone hygiene (record).**
`exp3797` archive `.347 / activate `.348; `exp3805` external research refresh;
`exp3806` KV260 opportunistic continuity audit; `exp3807` capstone.

**Phase B — Headline advancement (north-star §1).**
`exp3798` G4 RESTORATION: clean, provenance-complete re-run of the exp1999
code-verification-HumanEval repair pipeline on a mandated SOTA GGUF (GPU
preferred, reduced-n CPU-GGUF fallback, clean blocked fallback). `exp3799`
product-headline provenance RE-confirmation (gated on exp3798 producing a
clean artifact).

**Phase C — Banked verifier product hardening / repair.**
`exp3800` gaming-resistance MITIGATION v2 for the `context_compaction` evasion
(re-measure the degradation curve after mitigation). `exp3801` abstention
HTTP/REST integration surface (the third surface, Decentralization Rule 4).

**Phase D — Process repair + mandated self-learning.**
`exp3802` Anomaly-Escalation classifier v2 — TUNE down the 0.83 false-escalation
rate while preserving frame-violating recall, re-validate. `exp3803` FR-11 v20 —
wire the trained Tier-3 predictor as a fast-path/early-exit cascade gate and
measure compute saving + no accuracy regression (the mandated self-learning
experiment).

## 5. Dependency graph (mostly no hard `gated_on`; disk-presence fallback only)

```
exp3797 (archive/activate) ─ gates nothing; runs first
exp3798 (G4 re-run) ──────► exp3799 (provenance re-confirm)   [HARD gated_on:
                                                               exp3798 produced
                                                               a clean artifact]
exp3800, exp3801, exp3802, exp3803  ─ independent; disk-presence fallback only
exp3805, exp3806  ─ independent record tasks
exp3807 (capstone) ─ reads all upstream; honest not-landed/blocked handling
```

Only `exp3799` uses a hard `gated_on` (it is pure provenance aggregation over
exp3798's output and is meaningless if exp3798 blocked). Every other task reads
upstream with a graceful disk-presence fallback — never crash on a missing
field, never label un-run/blocked work as a research negative (the `.344
capstone-confusion guard).

## 6. Canonical task list (conductor execution order)

| # | id | track | agent | deliverable |
|---|----|-------|-------|-------------|
| 1 | exp3797 archive `.347 / activate `.348 | ops | codex | results/experiment_3797_archive_v347_activate_v348.json |
| 2 | exp3798 G4 product-headline restoration (exp1999 clean re-run) | product | claude+opus | results/experiment_3798_g4_product_headline_restoration.json |
| 3 | exp3799 product-headline provenance re-confirmation (gated) | product | codex | results/experiment_3799_product_headline_provenance_reconfirmation.json |
| 4 | exp3800 gaming-resistance mitigation v2 (context_compaction) | product | codex | results/experiment_3800_gaming_resistance_mitigation_v2.json |
| 5 | exp3801 abstention HTTP/REST surface (3rd surface) | product | codex | results/experiment_3801_abstention_http_rest_surface.json |
| 6 | exp3802 anomaly-escalation classifier v2 tuning | infra | codex | results/experiment_3802_anomaly_escalation_classifier_v2_tuning.json |
| 7 | exp3803 FR-11 v20 Tier-3-as-fast-path cascade gate | self-learning | codex | results/experiment_3803_fr11_v20_tier3_fast_path_gate.json |
| 8 | exp3805 external research refresh `.348 | ops | codex | results/experiment_3805_external_research_refresh.json |
| 9 | exp3806 KV260 opportunistic continuity audit | hardware | codex | results/experiment_3806_kv260_opportunistic_continuity_audit.json |
| 10 | exp3807 capstone `.348 | ops | codex | results/experiment_3807_capstone_v348.json |

## 7. Routing rationale

- **`gemini` is banned this milestone** — it crashes real GPU workloads and has
  wiped whole milestones via 429 (`[[incident_333_gemini_quota_crash_wipeout]]`).
- **Cheap-default is `codex` + `gpt-5.5`** (`requires_codex: true`) for all
  mechanical / aggregation / verifier-scoring / wiring tasks.
- **The ONE GPU + open-ended task is `exp3798`** (G4 re-run): GPU/bootstrap risk
  + open-ended pipeline restoration → `claude` + `opus` + `max_turns: 100`
  (hardware-integration routing; C+E escalation is claude-only). It is
  GPU-*preferred* with a reduced-n CPU-GGUF fallback and a clean blocked exit, so
  it never half-runs and never fabricates.
- Everything else is `codex`.

## 8. Hardware requirements

- **`exp3798`** prefers a free CUDA GPU (≥10 GB) for live SOTA-GGUF HumanEval
  generation+repair; falls back to reduced-n CPU GGUF (llama.cpp) for provenance;
  blocks cleanly if neither path is usable. **Note:** the rig has been
  GPU-contended for two consecutive milestones (exp3777/exp3787 both blocked) —
  this task is written to degrade gracefully, not to gamble a long run.
- **`exp3806`** KV260 via SSH only (`ssh kria`), opportunistic confirm-terminal.
- All other tasks are CPU-only (verifier-scoring against cached candidates,
  aggregation, wiring).

## 9. Invariants re-asserted at capstone (exp3807)

- `paper_ready` stays **TRUE** (G1–G4; confirm via `scripts/publication_gate.py --json`).
- FoVer **0.9131** stays frozen; `.348 uses but never moves it.
- Both energy routes stay **bounded** (`.348 runs no energy-foundation
  experiment; the P1 mechanism question is handed to the operator, not reopened).
- The next-Phase-3-thesis decision (EDLM seed vs freeze) remains an
  **operator-seeding surface**; the loop does not self-commit.

## 10. Anti-recurrence guards carried forward

- **Interpreter discipline** (`[[incident_ebt_training_venv_python_cuda]]`):
  every `Run` command pins `.venv/bin/python` — bare `python` lacks torch/yaml/sklearn.
- **Anti-poison-test** (the `.344 root cause): any shipped test MUST assert
  against the script's real behavior; any new `research-complete.yaml` value
  containing a colon MUST be quoted (`safe_load` must succeed after the write).
- **Inference-substrate hygiene:** every artifact declares `inference_substrate`;
  aggregation/verifier-scoring tasks carry NO GGUF/CUDA markers; every artifact
  passes `scripts/adversarial_verify.py` with no critical flag.
- **FALSE_NEGATIVE_RISK / positive-control discipline:** any null/"robust"/
  "no-regression" claim requires a positive control (the exp1999-G4 re-run must
  reach a non-trivial baseline; the gaming-mitigation must use genuinely
  adversarial perturbations).
- **Public Documentation Discipline:** the loop edits CODE + `research-references.md`
  + `results/*` only; it emits PROPOSALS for operator-curated docs
  (README, landing page, technical report, CLI/MCP docs) and never edits them.
