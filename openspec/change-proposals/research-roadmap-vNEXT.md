# Research Roadmap — Milestone 2026.06.344

**CONVERGENCE & PRODUCT-BANKING: both Phase-3 energy routes are now empirically
bounded — reconcile the DEFINITIVE Thesis-A close, mechanize the publication
gates, harden the banked verifier toward Phase-1 ship, and surface the
next-thesis decision to the operator.**

Planner: Claude Opus 4.8 — 2026-06-03. Outer-loop pre-staged roadmap per the
Pre-Staged Roadmap Convention.

---

## 1. What the previous milestone(s) proved — the Phase-3 existential question is settled

`.343` was pre-staged to run the operator-seeded **Thesis A** (energy as the
*generator*, EBT, arXiv:2507.02092) kill-gate to completion after two infra
blocks. **It was overtaken by events:** the operator ran the genuine kill-gate
**directly** on the recovered dual-3090 rig (the conductor's `kill_zombies`
reaps unowned GPU processes, so the in-loop EBT tasks could not run cleanly).
Those direct runs are **definitive** and supersede the in-loop `exp3745–3750`
chain:

- **Part-(a) PASS** (`results/thesis_a_direct_definitive_run.json`,
  adversarial-clean): the tiny 38M byte-EBT trained **stably** for 800 GSM8K
  steps (no NaN/divergence, grad-norm bounded) and learned a **generalizing,
  discriminative** held-out energy landscape — pos/neg margin **0.723 vs
  untrained 0.084 (~8.6×)**. Energy-as-generator is **viable at the stability
  gate**.
- **Part-(b) BOUNDED at scale** (`results/thesis_a_part_b_scaled_seed1.json`):
  with a **learned emb→token decoder** + **3-digit addition at 16k steps** (AR
  reaches 82% = clean headroom), the EBT scores **0.000** under *both* argmin
  and descent+learned-decoder decode, vs **AR 0.820 (greedy) / 0.840 (matched
  self-consistency)**. The energy-descended embeddings are **discriminative but
  NOT token-generative**; neither a learned decoder nor scale fixes it.

**Net:** both tested energy mechanisms are now bounded —
**energy-as-SELECTOR** (P0.1, settled) **and energy-as-GENERATOR** (Thesis A,
now definitive). This does **not** disprove the whole EBM-foundation-model
space; it closes the two routes the project actually tested. Per the standing
finding, the autonomous loop will **not** self-initiate a new paradigm — a
genuinely-different foundation-model thesis needs a **human seed**.

**Field consistency (`.344 literature scan):** the negative is consistent with
the field. **EBT-Policy (arXiv:2510.27545)** shows energy-as-generator wins —
but in **low-dimensional continuous control**, not discrete text; nobody has
shown EBT-generation beating AR on text at matched compute. **VerifyBench
(2507.09884)** and **multi-domain RM (2510.00492)** confirm *discriminative*
verifiers are domain-bound field-wide, and that the documented fix is
*generative* verification — not more discriminative members.

**What survives as the defensible, banked positive:** the **verifier product**
— a process-reward-style step-error discriminator at **AUROC 0.9131** on the
FoVer corpus (n=1,000, 5 seeds, dual-condition, CI95 [0.9027, 0.9235]), with an
isolated FR-11 self-learning contribution of **+0.0185** (`exp2837`). G1–G4 are
met; `paper_ready = true` (G2 independently reproduced on GitHub Actions
2026-05-31). The headline FoVer 0.9131 is **frozen**.

### Why `.344 is not churn

North-star §1: *a milestone that produces a new version of an existing artifact
without moving the headline is churn.* `.344 deliberately avoids re-versioning.
Every task either (a) **reconciles the definitive record** of a now-closed
question, (b) **mechanizes a publication gate** (north-star §2 sanctions G2/G3
mechanization as forward work), (c) **banks the verifier toward the Phase-1
software-ship gate** (PRD Phase-1, explicitly *not* gated on paper/hardware), or
(d) **hands the next-thesis decision to the operator**. No energy-selection /
energy-generation re-grind. No cross-corpus-matrix vN+1. No generalization
re-test.

---

## 2. The three biggest gaps between current state and the PRD vision

1. **The one positive is not yet *shipped* as a product.** PRD Phase-1 ships on
   a *software* gate (PyPI + HF/IPFS mirror + MCP/CLI docs + ≥1 external
   reproducer), not on paper or hardware. G2 closed the reproducer; the
   remaining gap is a verified **end-to-end software-ship path**: package
   install, CLI, MCP real-protocol exchange, and a committed local reproducer
   of the headline number. `.344 closes this gap (Phases B/C).

2. **The publication gates are honor-discipline, not mechanical.** North-star §2
   explicitly flags G2 (mechanize the reproducer) and G3 (ship the narrowing
   lint) as *within reach*. Today the Paper-v6 Narrowing Discipline (11
   retractions, now +1 for "energy-as-generator works at scale") is enforced by
   prose only. `.344 ships `paper_v6_narrowing_lint.py` and a committed local
   reproducer script (Phase B).

3. **The next research direction is undefined.** Both energy routes are bounded
   and the loop will not self-seed. The PRD's "autonomous directed
   self-learning" endgame needs the operator to choose the next Phase-3
   paradigm. `.344 produces a **ranked decision menu** of the
   genuinely-different, *untested* routes — led by **EDLM (energy as a residual
   corrector over discrete diffusion, arXiv:2410.21357)**, which reaches AR
   perplexity at matched compute in exactly the discrete-text regime the tiny
   EBT failed — a decision surface, not a commitment (Phase E).

---

## 3. Architecture (unchanged; this milestone touches the product surface, not the core)

```
                 ┌──────────────────────────────────────────────┐
                 │  BANKED POSITIVE — the verifier product       │
                 │  4-verifier ensemble → AUROC 0.9131 (FoVer)   │
                 │  fr11_session_memory (+0.0185), curry_howard, │
                 │  arithmetic_gap, logical_consistency          │
                 └───────────────┬──────────────────────────────┘
        ┌────────────────────────┼─────────────────────────────┐
        ▼                        ▼                              ▼
 Phase B: mechanize gates  Phase C: Phase-1 ship      Phase D: deployable
 - G2 local reproducer     - pip / CLI / MCP E2E       abstention gate
 - G3 narrowing lint       - mirror + publish checklist (AURC + certified
                                                        risk-coverage point)
        ▲                        ▲                              ▲
        └──────────── Phase A: reconcile the DEFINITIVE Thesis-A close ───────┘
                      (both energy routes bounded; record it honestly)

 Phase E: continuous self-learning (FR-11 v17 Tier-1, pivoted back to the
          LIVE verifier from the now-dead EBT-stabilizer lineage)
        + KV260 opportunistic terminal confirm
        + NEXT-THESIS decision menu for the operator
        + capstone
```

**CLOSED (do not re-open without a human-seeded, genuinely-different mechanism):**
energy-as-selector (P0.1), energy-as-generator (Thesis A, this milestone's
record), cross-domain generalization re-tests, cross-corpus-matrix versioning.

---

## 4. Phase descriptions

### Phase A — Reconcile the definitive Thesis-A close (record honesty)
- **exp3754** archive `.343 / activate `.344.
- **exp3755** Thesis-A DEFINITIVE reconciliation: ingest the operator's direct
  runs into `research-complete.yaml`; corrigendum that the in-loop
  `exp3745–3750` chain is **superseded/OBE** by the direct definitive runs;
  mark Thesis A part-(b) **bounded-at-scale (discriminative-not-generative)**,
  citing **EBT-Policy (arXiv:2510.27545)** as the field-consistent boundary
  (EBT-generation wins only low-dim continuous); update
  `docs/research-notes/phase3-alternative-thesis-menu.md` to mark Thesis A
  bounded. Records the bound; does **not** add energy-as-generator to the
  exclusion manifest (it's a finding, not a doomed-rerun id).

### Phase B — Mechanize the publication gates (north-star §2, forward work)
- **exp3756** **G2 mechanical reproducer**: commit a standalone, dependency-light
  script that reproduces FoVer **0.9131 within CI95** in the current `.venv`
  (the GitHub Actions reproducer closed G2 externally; this hardens the local
  ship-gate path and emits `auroc_in_ci95` as a bare-bool gate).
- **exp3757** **G3 narrowing lint**: ship `scripts/paper_v6_narrowing_lint.py`
  scanning the paper/report/`results/paper_v6_*` for the 11 retracted phrasings
  **+ a 12th** ("energy-as-generator works/scales") and wire a pre-commit hook.

### Phase C — Bank the verifier toward the Phase-1 software-ship gate
- **exp3758** **Package + CLI + MCP E2E smoke**: build/install the package, run
  the verify-repair pipeline end-to-end on a tiny example, and hit the MCP
  `score_candidates` tool via a **real protocol exchange**. Phase-1 ship gate:
  ≥1 external integration surface verified E2E. (SOTA GGUF in MODEL_SPECS with a
  small-model CPU fallback for the wiring smoke.)
- **exp3759** **Distribution-mirror + operator publication checklist** (Rule 3):
  verify the PyPI publish workflow + HF mirror + IPFS CID plan are documented
  and ready; emit an **operator-only** publish checklist. Does **not** publish.

### Phase D — The sanctioned new framing: a deployable abstention operating point
- **exp3760** **Certified abstention operating-point selection** on the proven
  0.9131 step-error discriminator: build the risk-coverage (AURC) curve and
  **select a deployable threshold** at a fixed risk target via conformal /
  PAC-Bayes certification (arXiv:2502.06884, 2509.12527; AURC frame
  OpenReview JJPAy8mvrQ). Forward from `.340's *characterization* to a
  *deployable, certified operating point*. Gated on exp3756 `auroc_in_ci95`.

### Phase E — Self-learning (mandated) + hardware + next-thesis + capstone
- **exp3761** **FR-11 v17 Tier-1 online per-verifier precision tracker** on the
  FoVer corpus — pivots FR-11 from the now-dead EBT-stabilizer lineage back to
  the **live verifier product**: track per-verifier precision/recall online,
  upweight high-precision verifiers, and confirm the learned weighting preserves
  the +0.0185 memory contribution. CPU counter updates (Tier-1 hardware path).
- **exp3762** **KV260 opportunistic terminal confirm** (SSH-only; GateMate /
  PolarFire remain opportunistic future-work per north-star §3).
- **exp3763** **Next-Phase-3-thesis decision menu** for operator seeding: a
  ranked survey of the genuinely-different, *untested* foundation-model routes
  now that both energy routes are bounded — EDLM residual-corrector-over-
  diffusion (2410.21357), latent-token diffusion reasoning (2602.03769),
  ParaRNN nonlinear-recurrent substrate (2510.21450), energy-verifier-as-
  test-time-reweighter (T3RL 2603.02203) — a decision surface for the operator,
  not a commitment by the loop.
- **exp3764** **Capstone .344** — aggregate the milestone; re-assert invariants
  (`paper_ready = true`, FoVer 0.9131 frozen, both energy routes bounded).

---

## 5. Dependency graph

```
exp3754 (archive/activate)
   └─ exp3755 (Thesis-A definitive reconcile)
exp3756 (G2 reproducer) ──auroc_in_ci95==true──▶ exp3760 (certified abstention point)
exp3757 (G3 narrowing lint)         [independent]
exp3758 (pkg/CLI/MCP E2E)           [independent]
exp3759 (mirror + publish checklist)[independent]
exp3761 (FR-11 v17 tracker)         [reads frozen FoVer scores]
exp3762 (KV260 opportunistic)       [independent, SSH]
exp3763 (next-thesis menu)          [reads research-references + thesis menu]
exp3764 (capstone)  ◀── aggregates exp3754–exp3763
```

Only **one** structured gate (exp3760 on exp3756) — everything else reads
upstream with a graceful disk-presence fallback (the `.340-proven safe pattern).

---

## 6. Hardware requirements

- **None GPU-critical.** All tasks are CPU verifier-scoring, aggregation, or
  SSH hardware-smoke. exp3758's E2E wiring smoke uses a small CPU model by
  default (SOTA GGUF optional via `cached_sota_pair()`).
- **KV260** reachable via `ssh kria` (SSH-only; never a host block-device
  check). GateMate / PolarFire opportunistic — not blocking.
- The dual-3090 rig is **idle by design** this milestone (no GPU training): the
  EBT route is closed; re-opening it requires an operator-seeded reason to
  expect scale/decoder changes the bound. (Monitor: Extropic Z1 / PIMI FPGA
  p-bit per the `.344 hardware references.)

---

## 7. Invariants (must hold at capstone)

- `paper_ready = true` (G1–G4 met); the milestone never regresses it.
- FoVer headline **0.9131 frozen**; no task re-derives or re-versions it
  (exp3756 *reproduces* it as a gate check, it does not move it).
- Energy-as-**selector** stays settled-bounded (P0.1); energy-as-**generator**
  is now recorded bounded-at-scale (Thesis A) — neither is re-ground.
- The agent never edits `ops/north-star.md`, `docs/index.html`, README, or any
  operator-curated doc; never triggers a CI run; never publishes externally;
  never pushes.
