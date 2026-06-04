# Research Roadmap — Milestone 2026.06.347

**POST-CONVERGENCE — RETRY THE LAST OPEN QUESTION + HARDEN THE BANKED PRODUCT**

**Planned:** 2026-06-04 (Claude Opus 4.8, outer-loop pre-staged roadmap)
**Supersedes activation of:** 2026.06.346 (POST-BOUNDED CONVERGENCE)
**Design doc for:** `research-roadmap-next.yaml`

---

## 1. What the previous milestone (.346) proved

`.346` landed **10 of 11** tasks. The project is now **converged** on its one
surviving positive and has bounded both energy-foundation routes:

- **G1–G4 all met; `paper_ready=TRUE`.** The headline — Carnot's verifier
  ensemble reaches **AUROC 0.9131** on the FoVer step-error corpus (n=1,000,
  5 seeds, dual-condition, CI95 [0.9027, 0.9235]) with the FR-11 self-learning
  component contributing **+0.0185** — is frozen and externally reproduced
  (GitHub Actions run 26725185125, 2026-05-31; re-reproduced locally in `.345`).
- **The verifier product was BANKED.** `.346 exp3779` wired the `.345`
  certified abstention operating point (threshold 0.733, coverage 0.998 at
  selective-risk ≤ 0.05, split-conformal δ=0.05) into the verify API as a
  deployable, default-OFF opt-in mode, E2E-confirmed and reachable through the
  MCP `score_candidates` surface.
- **FR-11 self-learning advanced to Tier-2.** `exp3778` consolidated per-domain
  threshold deltas (`Tier2ThresholdMemory`), AUROC under consolidation 0.9097
  (within the frozen CI), memory contribution preserved, state persisted.
- **The P3 Anomaly-Escalation classifier was prototyped** (`exp3780`,
  recommend-only, conductor unmodified) and a change-proposal written.
- **The operator's EDLM next-thesis seed was scaffolded** (`exp3781`,
  feasibility brief + minimal kill-gate design; the loop does NOT commit).
- **G4 technical-report correction was prepped** (`exp3782`, operator proposal;
  curated doc unedited).

**The one un-landed task — and the single genuinely-open question.**
`exp3777` (P1 discrete-search adjudication v3) **blocked on no-free-GPU**
(`p1_adjudication: blocked_missing_upstream_artifact`). This is the LAST open
energy-existential question: is the energy-as-generator bound an **artifact**
of the Langevin+learned-decoder decode (a NARROW reopen would be warranted) or
**FUNDAMENTAL** (a symmetric energy cannot enforce AR's causal factorization —
the "Causal Inductive Bias Gap")? The harness root-cause was already diagnosed
and fixed in `.346` (`n_train` 20000→40000, the regime where the AR positive
control reaches 0.84). The only thing missing was a free GPU.

## 2. The honest strategic state (read this before judging the plan for churn)

The project has **answered its core questions**:

- **Energy-as-selector** (P0.1) — bounded. Does not beat AR/SC where SC is
  strong (math/CSP) *and* where SC is weak.
- **Energy-as-generator** (Thesis-A) — bounded at scale (EBT 0.000 vs AR 0.84
  at matched compute). P1 only sharpens the *mechanism* behind this bound; it
  does not reopen the strategic conclusion.
- **The verifier moat** (P2) — CLOSED. The defensible claim is the banked
  **math step-error product**, NOT a general independence-moat.
- **Self-seeding a new paradigm** (P3) — the loop CANNOT do it (the Verification
  Trap). A genuinely-different foundation-model thesis needs a **human seed**.
  EDLM (energy as a residual-corrector over discrete diffusion, arXiv:2410.21357)
  is the operator's top candidate; `.346` handed up a feasibility brief.

**What this means for planning.** With both energy routes bounded and the moat
closed, the loop's *legitimate, non-churn* forward surface narrows to exactly
three things:

1. **The one open P1 mechanism question** (retry the GPU adjudication).
2. **Hardening the banked verifier product** (the converged positive) toward
   real deployability and toward *restoring* the demoted product headline by
   confirming provenance — north-star §1's own definition of a headline-
   *advancing* milestone, the opposite of noise.
3. **Mandated continuous self-learning** (research-program.md) — Tier-3.

`.347` does exactly these, plus the operator-seed scaffold (EDLM no-train
preflight) and record/continuity hygiene. **It re-grinds nothing bounded and
runs no scissor-plot/moat sweep.** This is deliberately a LEAN milestone.

**A flag to the operator (north-star §1).** Continued milestones *without* an
operator seed will increasingly approach the churn boundary. The substantive
decision — seed EDLM (or another Phase-3 thesis) or freeze the loop into a
product-maintenance cadence — is **the operator's to make**. `.347` lowers the
EDLM seed cost to a single command (the preflight) and otherwise spends only on
the three legitimate surfaces above.

## 3. The three gaps between current state and the PRD vision

| Gap | PRD anchor | `.347` response |
|---|---|---|
| The last energy-existential mechanism is unadjudicated (GPU-blocked) | "autonomous directed self-learning where the energy function is ground truth" | `exp3787` retry P1 v3 with a GPU-free precondition + clean blocked-fallback |
| The banked verifier is a NUMBER + an opt-in mode, not yet a hardened, gaming-aware, CLI-surfaced product with a restored headline | Phase-1 "ship a useful operational product" | `exp3789` abstention CLI/batch surface; `exp3790` gaming-resistance characterization; `exp3792` product-headline provenance confirmation (G4) |
| Self-learning has Tier-1 + Tier-2; Tier-3 (predictive) is unbuilt on the live verifier | research-program.md "Continuous Self-Learning" Tiers 1–4 | `exp3788` FR-11 v19 — Tier-3 predictive verification on the FoVer corpus |

## 4. Phases

**Phase A — Records & continuity (ops).** `exp3786` archive/activate;
`exp3794` external research refresh; `exp3795` KV260 opportunistic;
`exp3796` capstone.

**Phase B — The last open question (phase3 depth, GPU).** `exp3787` P1
discrete-search v3 RETRY — the depth anchor. Routes claude+opus+max_turns:100
(GPU from-scratch training + bootstrap risk + open-ended artifact-vs-fundamental
adjudication). Precondition checks a *free* GPU and exits cleanly to
`blocked_no_free_gpu` if the rig is busy — never queues, never fabricates.

**Phase C — Harden the banked product (product, CPU).**
`exp3789` wire the abstention mode into a CLI + batch-scoring surface (forward
from `.346`'s API opt-in); `exp3790` programmatic gaming-resistance
characterization of the shipped math step-error verifier (CPU, cached
perturbations — NOT a moat/independence sweep, NOT GPU candidate generation);
`exp3792` confirm the demoted product-headline numbers (exp1999 +18pp / exp2090
+15pp) trace to primary artifacts with seed+checksum (G4) so the operator can
restore them.

**Phase D — Self-learning, process, operator-seed scaffold.**
`exp3788` FR-11 v19 Tier-3 predictive verification (distinct tier from v18's
Tier-2 memory); `exp3791` validate the `.346` anomaly-escalation classifier
against the historical retro corpus; `exp3793` EDLM no-train preflight
readiness (GO/NO-GO; lowers the operator's seed cost without committing the
loop or opening a Phase-3 track).

## 5. Dependency graph (no hard `gated_on`; disk-presence fallback only)

```
exp3786 archive/activate (ops)
   │
   ├─ exp3787 P1 v3 retry (GPU)           ── reads .346 v2 harness + part-b regime
   ├─ exp3788 FR-11 v19 Tier-3 (CPU)      ── reads 4-verifier FoVer scores
   ├─ exp3789 abstention CLI/batch (CPU)  ── reads exp3779 abstention mode + exp3771 cert
   ├─ exp3790 gaming-resistance (CPU)     ── reads cached FoVer + exp2837 verifiers
   ├─ exp3791 anomaly validation (CPU)    ── reads exp3780 classifier + retro corpus
   ├─ exp3792 product-headline G4 (CPU)   ── reads exp1999 / exp2090 / north-star §1
   ├─ exp3793 EDLM no-train preflight     ── reads exp3781 brief + exp3763 menu
   ├─ exp3794 external refresh (ops)
   └─ exp3795 KV260 opportunistic (ops)
            │
         exp3796 capstone .347 ── aggregates all; re-asserts invariants
```

Each downstream task reads upstream artifacts with a graceful disk-presence
fallback (the `.340` proven-safe pattern). No task crashes on a missing field;
no un-run task is labeled a research negative (the `.344` capstone-confusion
guard).

## 6. Canonical task list (conductor execution order)

| # | id | track | agent | GPU | substrate |
|---|---|---|---|---|---|
| 1 | exp3786 archive .346 / activate .347 | ops | codex | no | aggregation |
| 2 | exp3787 P1 discrete-search v3 RETRY | phase3 | claude/opus | **yes** | live_llm_inference |
| 3 | exp3788 FR-11 v19 Tier-3 predictive | self-learning | codex | no | verifier-scoring |
| 4 | exp3789 abstention CLI + batch surface | product | codex | no | verifier-scoring |
| 5 | exp3790 verifier gaming-resistance | product | codex | no | verifier-scoring |
| 6 | exp3791 anomaly-escalation validation | infra | codex | no | aggregation |
| 7 | exp3792 product-headline provenance (G4) | product | codex | no | aggregation |
| 8 | exp3793 EDLM no-train preflight readiness | phase3 | claude | no | aggregation |
| 9 | exp3794 external research refresh .347 | ops | codex | no | aggregation |
| 10 | exp3795 KV260 opportunistic continuity | hardware | codex | no | hardware_smoke |
| 11 | exp3796 capstone .347 | ops | codex | no | aggregation |

## 7. Routing rationale

- **codex is the cheap default** (gemini crashes real GPU workloads and has
  wiped whole milestones via 429 — `incident_333`; codex is the standing
  cheap-default for mechanical/aggregation/verifier-scoring work).
- **The ONE GPU task** (`exp3787`) routes **claude+opus+max_turns:100** —
  hardware-integration + bootstrap + open-ended mechanistic adjudication is
  exactly the class that routes to opus; C+E escalation is claude-only.
- **The ONE judgment task** (`exp3793` EDLM preflight) routes **claude** —
  cross-paper synthesis + honest GO/NO-GO judgment about a genuinely-different
  EBM route.
- **Every `Run` command pins `.venv/bin/python`** (`incident_ebt_training_venv_
  python_cuda`): bare `python` is `/usr/bin/python3` with no torch/yaml/sklearn,
  needed even for CPU tasks (adversarial_verify, summarize_artifact, scoring).

## 8. Hardware requirements

- **`exp3787` only** needs a CUDA GPU with ≥10 GB free. It checks `nvidia-smi`
  free memory and exits to `blocked_no_free_gpu` if the rig is busy — it never
  queues behind another job (the `.346` v1/v2/v3 pattern where the operator
  runs the corrected harness directly).
- **`exp3795`** is an SSH board smoke to the KV260 (SSH-only precondition per
  KV260 SSH-Not-SD-Card discipline; never a host block-device check).
- All other tasks are **CPU-only** (verifier-scoring against cached FoVer
  triples, aggregation, docs).

## 9. Invariants re-asserted at capstone

- `paper_ready` stays **TRUE** (G1–G4, confirmed via `publication_gate.py`).
- FoVer **0.9131** stays frozen; `.347` *uses* but never *moves* it.
- **Both energy routes stay bounded** — `exp3787` sharpens the P1 mechanism
  only; it never reopens the strategic conclusion or seeds a paradigm.
- The **next-Phase-3-thesis decision remains an operator-seeding surface**; the
  loop scaffolds (EDLM preflight) but does NOT commit.
- The agent **never** edits operator-curated docs, triggers CI, publishes, or
  pushes.

## 10. Anti-recurrence guards carried forward

- **Anti-poison-test** (the `.344` root cause): any shipped test MUST assert
  against the script's REAL behavior; any new `research-complete.yaml` value
  containing a colon MUST be quoted (verified via `yaml.safe_load` after write).
- **Inference-substrate hygiene:** every task declares `inference_substrate`
  and runs `adversarial_verify.py`; no vestigial GGUF/CUDA markers on
  aggregation/verifier-scoring tasks.
- **FALSE_NEGATIVE_RISK discipline:** `exp3787`'s null "FUNDAMENTAL" claim is
  only valid if the AR positive control reaches `ar_best ≥ 0.3`; otherwise the
  honest verdict is INCONCLUSIVE (the v1/v2 lesson).
- **Failed-Experiment Rerun Discipline:** `exp3787` carries a `prior_failures`
  block naming v1/v2 and the v3 GPU-block, the diagnosed cause, what is
  different, and `retire_if_same_verdict: true`.
</content>
