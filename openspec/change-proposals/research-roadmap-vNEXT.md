# Research Roadmap — Milestone 2026.06.349

**POST-CONVERGENCE — LEAN MAINTENANCE + OPERATOR-FORK STAGING**

**Planned:** 2026-06-04 (outer-loop, Claude Opus 4.8)
**Prior milestone:** 2026.06.348 (POST-CONVERGENCE headline-advancement + product harden/repair)
**Milestone doc convention:** rolling "next" design doc (overwrites the prior `-vNEXT`; the prior is archived in `research-complete.yaml` via the activation task).

---

## 0. One-paragraph summary

The project is **converged**. Both energy mechanisms are bounded honest-negatives
(energy-as-selector P0.1; energy-as-generator Thesis-A/EBT, triple-confirmed
bounded at scale). The verifier product is **banked and hardened**: FoVer AUROC
0.9131 frozen, `paper_ready = TRUE` (G1∧G2∧G3∧G4), the context_compaction gaming
evasion is closed (exp3800), a Tier-3 fast-path gate works (exp3803, 56% skip at
no regression), and the Anomaly-Escalation classifier was tuned to a usable point
(exp3802: false-escalation 0.0, frame-violating recall 1.0). The energy-foundation
dream is closing; the durable asset is the math step-error verifier; the
irreplaceable engine is the **human as frame-generator** (the documented
Verification Trap — the loop cannot self-seed a paradigm).

`.349 does **no** new existential research and re-grinds **nothing** bounded. It
spends only on legitimate post-convergence surfaces (north-star §1): **wire** the
now-usable Anomaly-Escalation advisory hook (the cheap Deep-Think-P3 upgrade,
recommend-only, conductor unmodified), **repair** the blocked HTTP/REST abstention
surface (Decentralization Rule 4, third surface), **record the product-headline
status honestly** (both candidate product positives now fail provenance), **continue**
the mandated Tier-3 self-learning, **confirm** the publication-gate invariants, and
**stage** the EDLM seed as a one-command operator option. It carries the operator
fork forward prominently: **the substantive next move — SEED EDLM (preflight GO) or
FREEZE the loop — is the operator's to make.**

---

## 1. What the previous milestone (.348) proved

| Result | Artifact | Status for .349 |
|---|---|---|
| G4 product-headline re-run reproduced **delta = 0.0pp** (the +18pp code-repair lift did NOT survive a clean provenance run; CPU-GGUF reduced-n, baseline 0.13 → repair 0.13) | exp3798 | Product code-repair headline **stays demoted** |
| Provenance re-confirmation: code-repair re-run is now G4-provenance-complete but delta=0 → `not_yet_eligible` | exp3799 | Headline rests on FoVer methods claim only |
| context_compaction gaming evasion **CLOSED**, clean AUROC preserved (n=240) | exp3800 | Verifier product hardened; no open evasion remains (panel of 5 perturbations: only context_compaction degraded, now closed) |
| HTTP/REST abstention surface **BLOCKED** (`blocked_http_abstention_e2e_failed`) | exp3801 | **Repair target for .349** |
| Anomaly-Escalation classifier tuned: false-escalation 0.83 → **0.0**, recall **1.0**, `supports_wiring_in = true` | exp3802 | **Now wirable as a recommend-only advisory hook (.349)** |
| Tier-3 fast-path gate: **skip 0.56**, effective AUROC 0.9227 in frozen CI, no regression, operating point persisted | exp3803 | Self-learning continues from here |
| Capstone: paper_ready TRUE; frozen 0.9131 unchanged; both energy routes bounded; operator fork carried forward | exp3807 | Invariants to re-assert |

**Reading-discipline catch (2026-06-04, this planning pass):** exp2090 (CRANE +15pp),
which exp3799 reported as G4-passing, **flags CRITICAL on a live adversarial re-check**
(substrate=None, duration 0.009s — not a real 50-problem HumanEval run). So **both**
candidate product positives (code-repair +18pp and CRANE +15pp) now fail provenance.
The **FoVer methods headline (0.9131) is the sole defensible headline** — exactly
north-star §1. `.349` records this honestly rather than consolidating on a flagged
artifact.

---

## 2. The three biggest gaps (PRD vision vs current state) and how .349 touches them

1. **Self-improving at inference speed (PRD FR-11 / Continuous Self-Learning Tier 3).**
   v20 (.348) wired the Tier-3 predictor as a fast-path gate on ONE split. The gap:
   does the 0.56 skip-rate / no-regression operating point **generalize** across
   seeds/splits, or is it a single-split artifact? → exp3813 (FR-11 v21,
   cross-split robustness; the mandated self-learning experiment).

2. **Multiple integration surfaces in parallel (Decentralization Rule 4).** The
   abstention mode is on the verify API + MCP + CLI but the **HTTP/REST surface is
   broken** (exp3801). A non-Python network integrator currently has no working path.
   → exp3810 (repair) + exp3811 (cross-surface parity, so no surface drifts ahead).

3. **Human-as-frame-generator / the operator fork (Deep-Think P3 Verification Trap).**
   The loop cannot self-seed the next paradigm; the EDLM preflight is GO but seeding is
   the operator's call. The gap is **decision-readiness**: the seed should be one
   operator command away with a sound kill-gate design. → exp3815 (EDLM operator-seed
   staging package; documentation only, seeds nothing).

---

## 3. Architecture (what .349 touches — nothing in the frozen core)

```
                        FROZEN / UNTOUCHED
   ┌───────────────────────────────────────────────────────────┐
   │  FoVer 4-verifier ensemble  →  AUROC 0.9131 (frozen CI)    │
   │  publication_gate: G1∧G2∧G3∧G4 = paper_ready TRUE          │
   │  energy-as-selector  (P0.1)            → BOUNDED            │
   │  energy-as-generator (Thesis-A/EBT)    → BOUNDED           │
   └───────────────────────────────────────────────────────────┘
                                │ read-only (use, never move)
        ┌───────────────────────┼───────────────────────────────┐
        ▼                       ▼                                ▼
  PRODUCT HARDEN          ENDORSED-TOOL WIRE              SELF-LEARNING
  exp3810 HTTP/REST v2    exp3809 Anomaly-Escalation      exp3813 FR-11 v21
  exp3811 surface parity  advisory hook (recommend-only,  (Tier-3 fast-path
  (Decentralization R4)   conductor UNMODIFIED, proposal)  cross-split robustness)
        │                       │                                │
        └───────────┬───────────┴───────────────┬────────────────┘
                    ▼                            ▼
            HEADLINE + INVARIANTS         OPERATOR-FORK STAGING
        exp3812 product-headline status   exp3815 EDLM seed package
        exp3814 publication-gate regress   (one-command, kill-gate
        (paper_ready stays TRUE)            design; SEEDS NOTHING)
                    │
                    ▼
        RECORD / CONTINUITY  →  exp3816 research refresh, exp3817 KV260, exp3818 capstone
```

**No task touches:** the frozen ensemble, the energy routes, `scripts/research_conductor.py`,
or any operator-curated doc (README, landing page, technical-report prose, north-star,
roadmap.md). Doc changes are emitted as **proposals**, never edits.

---

## 4. Phases and tasks (11 tasks)

**Phase 0 — Transition (1 task)**
- `exp3808` archive .348 → activate .349. Records the converged state honestly
  (product headline demoted; gaming closed; anomaly classifier now wirable; HTTP/REST
  blocked → repair target; both energy routes bounded; paper_ready TRUE). Carries the
  operator fork forward.

**Phase 1 — Product harden + endorsed-tool wiring (3 tasks)**
- `exp3809` Wire the Anomaly-Escalation classifier as a **recommend-only advisory
  module** + an offline replay over historical milestone negatives (it must escalate
  only the genuine frame-violations, auto-reconcile clean bounded negatives) + a
  conductor-integration **proposal** for the operator. The conductor is NOT modified;
  the classifier never relaxes verification. (Deep-Think-P3 endorsed upgrade, now usable.)
- `exp3810` HTTP/REST abstention surface **v2** — diagnose and fix the exp3801 E2E
  failure (`blocked_http_abstention_e2e_failed`) in `python/carnot/pipeline/abstention_http_rest.py`.
  Default abstention OFF; third surface per Decentralization Rule 4. (prior_failures: exp3801)
- `exp3811` Cross-surface abstention **parity smoke** (gated on exp3810 landing): the
  same cached FoVer candidate produces the same confident/abstain verdict + coverage/risk
  metadata across verify-API, CLI, and HTTP. No surface drifts ahead.

**Phase 2 — Headline status + self-learning + invariants (3 tasks)**
- `exp3812` Product-headline **status consolidation** (aggregation): apply the
  Reading-Results discipline (live adversarial re-check) to show that BOTH candidate
  product positives fail provenance (exp3798 delta=0; exp2090 CRANE flags CRITICAL);
  record that the FoVer methods headline is the sole defensible one; emit a doc-update
  **proposal** to retire the demoted code-repair prose. Edits no operator-curated doc.
- `exp3813` FR-11 continuous self-learning **v21** (the mandated self-learning task):
  validate that the v20 Tier-3 fast-path operating point (skip 0.56 / no regression)
  **generalizes** across a second seed/split — or report honestly that it does not.
  Reuses the persisted predictor; runs no live model.
- `exp3814` Publication-gate **regression confirmation**: `publication_gate.py --json`
  still reports G1∧G2∧G3∧G4 = `paper_ready TRUE`; frozen 0.9131 unchanged; G3 narrowing
  stays clean. The standing convergence invariant check.

**Phase 3 — Operator-fork staging + record/continuity (4 tasks)**
- `exp3815` **EDLM operator-seed staging package** (documentation only; SEEDS NOTHING):
  package the exp3793 one-command seed + a tiny-scale kill-gate design doc (matched-COMPUTE,
  internal 3090, hard cuda-block, honest-negative-if-it-diverges — mirroring Thesis-A
  .341) so the operator can seed `.350` in one decision. Explicitly operator-gated.
- `exp3816` External research refresh for .349 (confirm the `.349 additions` section
  parses; append-only).
- `exp3817` KV260 opportunistic continuity audit (SSH-reachable, overlay loadable;
  terminal state holds — opportunistic per north-star §3).
- `exp3818` Capstone .349. Re-assert invariants; carry the operator fork forward.

---

## 5. Dependency graph

```
exp3808 (archive/activate)
   ├─ exp3809 (anomaly advisory hook)        ─┐
   ├─ exp3810 (HTTP/REST v2) ── exp3811 (parity, gated_on exp3810.http_rest_surface_added==true)
   ├─ exp3812 (product-headline status)       │
   ├─ exp3813 (FR-11 v21 self-learning)       │  all read upstream with graceful
   ├─ exp3814 (publication-gate regression)   │  disk-presence fallbacks
   ├─ exp3815 (EDLM seed staging)             │
   ├─ exp3816 (research refresh)              │
   └─ exp3817 (KV260 audit)                  ─┘
        └─ exp3818 (capstone — aggregates exp3808..exp3817)
```

One hard gate: `exp3811` on `exp3810` producing `http_rest_surface_added == true`
(a parity smoke is meaningless if the surface didn't land). Every other task reads
upstream with a graceful fallback and never labels un-run/blocked work as a negative.

---

## 6. Hardware requirements

**None GPU-bound.** All tasks are verifier-scoring against cached candidates,
aggregation, wiring + E2E smoke, documentation, or an SSH hardware smoke (KV260).
No live LLM generation is queued — the .348 GPU-contention pattern (exp3798 fell back
to CPU-GGUF; HumanEval re-runs hand off to the operator) makes a fresh GPU-preferred
open-ended task low-value this milestone. KV260 is opportunistic (terminal since .340).

**Routing:** all tasks `agent_type: codex` + `model: gpt-5.5` + `requires_codex: true`.
gemini crashes real GPU workloads and has wiped milestones via 429 (incident_333); no
task here needs Claude/Opus (no open-ended GPU/bootstrap work). Every Run command pins
`.venv/bin/python` (bare `python` is `/usr/bin/python3` with no torch/yaml/sklearn —
incident_ebt_training_venv_python_cuda).

---

## 7. Invariants (re-asserted at capstone)

- `paper_ready` stays **TRUE** (G1∧G2∧G3∧G4 via `publication_gate.py`).
- FoVer **0.9131 frozen** — used, never moved.
- Both energy routes stay **bounded** (.349 runs no energy-foundation experiment).
- `scripts/research_conductor.py` **unmodified**; no operator-curated doc edited
  (proposals only); nothing pushed; no CI triggered; no external publication.
- The EDLM-seed-vs-freeze decision remains an **operator-seeding surface** — the loop
  does not self-commit a paradigm (Verification Trap).

---

## 8. The operator flag (carried forward — now actionable in one command)

The project is converged and the EDLM preflight is GO. The substantive next move is
the operator's: **SEED EDLM** (`exp3815` stages the one-command package + kill-gate
design) **or FREEZE** the loop into product-maintenance. Continued lean-maintenance
milestones increasingly approach the churn boundary (north-star §1). `.349` is a
defensible lean milestone that finishes concrete .348 loose ends and makes the seed
one decision away — but the milestone after `.349` should be the **EDLM seed or an
explicit freeze**, not a fourth deferral.
