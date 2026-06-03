# Research Roadmap — Milestone 2026.06.345

**PRODUCT-BANKING RECOVERY: re-execute the `.344 convergence agenda that a
whole-milestone YAML SKIP cascade erased. Both Phase-3 energy routes stay
bounded; bank the one surviving positive (the FoVer 0.9131 verifier) toward the
Phase-1 software ship gate, mechanize the publication gates locally, ship a
*certified* abstention operating point, and continue self-learning on the live
verifier.**

Planner: Claude Opus 4.8 — 2026-06-03. Outer-loop pre-staged roadmap per the
Pre-Staged Roadmap Convention.

---

## 1. What the previous milestone(s) proved — and why `.344 must be re-run

### The settled science (carried forward from `.340–`.344)

Both tested energy mechanisms are **bounded**:

- **energy-as-SELECTOR** — P0.1 settled: energy/verifier reranking of AR/SC
  outputs does not beat self-consistency where SC is strong (math/CSP) and adds
  no value where SC is weak. `[[project_energy_selection_thesis_bounded]]`.
- **energy-as-GENERATOR** — Thesis A (EBT, arXiv:2507.02092) closed by the
  operator's **direct** dual-3090 runs: part-(a) **PASS** (tiny 38M byte-EBT
  trains stably 800 steps; held-out discriminative margin 0.723 vs untrained
  0.084), part-(b) **BOUNDED at scale** (with a learned emb→token decoder +
  3-digit + 16k steps, EBT scores **0.000** vs AR **0.84** at matched compute —
  discriminative-not-generative; neither decoder nor scale fixes it). Field-
  consistent: EBT-Policy (arXiv:2510.27545) shows energy-generation wins only in
  low-dim continuous control, never discrete text at matched compute.
  `[[project_thesis_a_ebt_seeded]]`.

This does **not** disprove the EBM-foundation-model space; it closes the two
routes the project tested. Per the standing finding, **the autonomous loop will
not self-initiate a new paradigm** — a genuinely-different thesis needs a human
seed. `exp3763` already handed the operator a ranked next-thesis menu (top
route: **EDLM**, energy as a residual-corrector over discrete diffusion,
arXiv:2410.21357); the operator has not yet seeded one, so `.345 does **not**
open a new Phase-3 track.

The defensible, banked positive — **the verifier product**: a process-reward-
style step-error discriminator at **AUROC 0.9131** on the FoVer corpus
(n=1,000, 5 seeds, dual-condition, CI95 **[0.9027, 0.9235]**), with an isolated
FR-11 self-learning contribution of **+0.0185** (CI95 [0.0125, 0.0245]),
sourced from `exp2837`. **G1–G4 are met; `paper_ready = true`** (G2
independently reproduced on GitHub Actions 2026-05-31; confirmed live via
`scripts/publication_gate.py`). The headline FoVer 0.9131 is **frozen**.

### The reason `.345 exists: `.344 produced ZERO experiments

`.344 was a correct, operator-pre-staged convergence/product-banking milestone.
**It did not land.** A malformed entry in `research-complete.yaml` (an unquoted
embedded colon in an `exp3742` result value → `yaml.ScannerError` →
`test_public_docs_*` failed) poisoned the conductor's pre-test gate and
triggered a **whole-milestone SKIP cascade** (`[[incident_agent_shipped_test_cascade]]`
pattern). The YAML was fixed post-hoc (commit `1726bfbb2`), but by then only the
last three tasks left partial artifacts:

| `.344 task | Scope | Landed? |
|---|---|---|
| exp3754 | archive `.343 / activate `.344 | skipped |
| exp3755 | Thesis-A definitive reconcile | skipped |
| exp3756 | G2 local FoVer reproducer | skipped |
| exp3757 | G3 narrowing-lint extend+wire | skipped |
| exp3758 | package/CLI/MCP E2E smoke | skipped |
| exp3759 | distribution-mirror + publish checklist | skipped |
| exp3760 | certified abstention operating point | skipped |
| exp3761 | FR-11 v17 self-learning (live verifier) | skipped |
| exp3762 | KV260 opportunistic continuity audit | partial artifact |
| exp3763 | next-Phase-3-thesis decision menu | partial artifact |
| exp3764 | capstone `.344 | partial — honestly reports `both_energy_routes_bounded=false` because its upstream (exp3755…) was missing |

`operational_retro_2026_06_344.json`: *"produced no experiment commits … 0
completed experiments."*

**Therefore `.345 re-executes the `.344 agenda.** These tasks are **unstarted
work skipped by infrastructure, not doomed reruns** — there is no prior negative
verdict to address, and the YAML cascade is fixed. Re-running them is exactly
the legit-continuation case the Exclusion-Manifest auto-override covers; each
carries an `operator_override` citing the `.344 cascade.

### Why this is convergence, not churn (north-star §1)

Every `.345 task either **closes a publication gate** (G2-local, G3-lint),
**advances the Phase-1 software-ship gate** (package/CLI/MCP E2E, mirror
checklist), or **turns the banked positive into a deployable product** (certified
abstention operating point, PRM-positioning). None re-versions an existing
artifact for its own sake; none re-grinds a bounded energy route.

---

## 2. The three biggest gaps between current state and the PRD vision

1. **Phase-1 software ship is not E2E-verified.** PRD Phase 1 = "ship a useful,
   operational software product" on a *software* gate (PyPI + HF/IPFS mirror +
   MCP/CLI docs + ≥1 external reproducer). G2's CI reproducer satisfies the
   external-reproducer condition, but the E2E path an integrator actually uses
   (`import carnot` → VerifyRepairPipeline → MCP `score_candidates` over a real
   protocol exchange → CLI) has never been verified end-to-end this cycle, and
   the operator publish checklist was never emitted. **Gap → exp3769, exp3770.**

2. **The publication gates are not *mechanically* enforced locally.** G2 was
   closed by a remote CI run; there is no committed, dependency-light *local*
   reproducer an integrator can run, and the G3 narrowing lint
   (`scripts/paper_v6_narrowing_lint.py`) exists but is **not pre-commit-wired**
   and lacks the 12th retraction (energy-as-generator). **Gap → exp3767, exp3768.**

3. **The one banked positive has no *deployable, certified* operating point.**
   `.340 characterized the 0.9131 discriminator's risk–coverage envelope, but no
   milestone selected and *certified* a threshold ("at coverage C, selective risk
   ≤ R with probability ≥ 1−δ"). That certificate is the difference between "an
   AUROC number" and "a shippable selective-prediction product." **Gap → exp3771**,
   sharpened by the new `.345 conformal references (arXiv:2603.24704 / 2605.30085).

The mandatory continuous-self-learning thread (research-program.md) is carried by
**exp3772** (FR-11 v17, Tier-1 online verifier-precision tracker on the live
verifier — pivoted off the now-dead EBT-stabilizer lineage).

---

## 3. Architecture — what this milestone touches

```
                         FoVer step-error corpus (n=1000, frozen)
                                      |
        +-----------------------------+-----------------------------+
        |                             |                             |
  4-verifier ensemble          G2 LOCAL reproducer           FR-11 v17 Tier-1
  (fr11_session_memory,   ->    (exp3767: re-derive 0.9131 -> online per-verifier
   tier0r_curry_howard,        within CI95, emit                precision tracker
   tier0s_arithmetic_gap,      auroc_in_ci95 BARE bool)         (exp3772, CPU
   tier0u_logical_consistency)        |                          counter updates)
        |                             |  gate (==true)
        |                             v
        |                   CERTIFIED ABSTENTION (exp3771)
        |                   risk-coverage -> select threshold
        |                   -> conformal / PAC-Bayes certificate
        v                             |
  PRM-positioning (exp3773)           v
  vs GenPRM/ThinkPRM/uPRM    DEPLOYABLE selective-prediction product
        |
        v
  PHASE-1 SHIP SURFACES (exp3769)              PUBLICATION GATES
  import carnot -> pipeline -> MCP -> CLI       G3 narrowing lint wired (exp3768)
        |                                       Thesis-A reconcile (exp3766)
        v
  distribution mirror + OPERATOR publish checklist (exp3770)
```

**No GPU training anywhere.** Every task is CPU verifier-scoring, aggregation,
E2E-software smoke (small CPU model), or an SSH hardware audit. There is no GPU
risk and no gemini-crash exposure (see Routing).

---

## 4. Phases & tasks

**Phase A — Recover the record (ops).**
- `exp3765` archive `.344 honestly (the SKIP cascade) + activate `.345.
- `exp3766` Thesis-A **definitive reconcile** (re-do skipped `.344 exp3755):
  ingest the operator's direct runs, corrigendum the superseded in-loop chain,
  mark the thesis menu, do NOT add energy-as-generator to the exclusion manifest.

**Phase B — Mechanize the publication gates.**
- `exp3767` G2 **local** mechanical reproducer (re-derive 0.9131 within CI95;
  emit `auroc_in_ci95` BARE bool). Hardens the local ship-gate path; the gate for
  exp3771.
- `exp3768` G3 narrowing lint **extend + wire** (add the 12th retraction; add the
  missing `.pre-commit-config.yaml` stanza; extend scan to `results/paper_v6_*`).

**Phase C — Bank the verifier toward Phase-1 ship + the certified product.**
- `exp3769` package + CLI + MCP **E2E smoke** (real protocol exchange).
- `exp3770` distribution-mirror readiness audit + **operator-only** publish
  checklist (publishes nothing).
- `exp3771` **certified abstention operating point** (gated on
  `exp3767.auroc_in_ci95 == true`) — the one judgment-heavy statistical task.

**Phase D — Self-learning, positioning, hardware, capstone.**
- `exp3772` FR-11 v17 Tier-1 online verifier-precision tracker (mandatory
  self-learning; live verifier; preserves the +0.0185 memory contribution).
- `exp3773` **verifier-product honest positioning** vs PRM SOTA
  (GenPRM/ThinkPRM/uPRM/ProcessBench) — genuinely-new analysis; claim
  cost/objectivity + certified abstention, NOT raw F1; no generalization re-test.
- `exp3774` KV260 opportunistic continuity audit (terminal-state confirm; SSH).
- `exp3775` capstone `.345.

### Dependency graph

```
exp3765 (archive/activate)
   |-> exp3766 (thesis-A reconcile)
   |-> exp3767 (G2 local reproducer) --auroc_in_ci95==true--> exp3771 (certified abstention)
   |-> exp3768 (G3 lint)
   |-> exp3769 (pkg/CLI/MCP E2E)
   |-> exp3770 (mirror + publish checklist)
   |-> exp3772 (FR-11 v17 self-learning)
   |-> exp3773 (PRM positioning)
   |-> exp3774 (KV260 audit)
        |-> exp3775 (capstone, aggregates 3765-3774)
```

Only `exp3771` has a structured `gated_on` (on `exp3767.auroc_in_ci95`, a BARE
bool per `[[feedback_gated_fields_must_be_bare]]`). Every other task reads
upstream with a graceful disk-presence fallback (the `.340 proven-safe pattern):
never crash on a missing field, never label un-run work as a negative.

---

## 5. Routing & discipline (carried from `.344, unchanged)

- **CODEX is the cheap default** (`requires_codex: true`). gemini **crashes real
  GPU workloads and has wiped milestones via quota-429s**
  (`[[incident_333_gemini_quota_crash_wipeout]]`); codex is the standing routing
  reality since 2026-06-02. No task trains on GPU, so there is no GPU risk.
- **Two CLAUDE tasks** (judgment-heavy, open-ended): `exp3771` (conformal /
  PAC-Bayes abstention certification — statistical judgment under ambiguity) and
  `exp3773` (PRM-positioning synthesis — cross-paper judgment). Both meet the
  `requires_claude` positive criterion.
- **Interpreter discipline** (`[[incident_ebt_training_venv_python_cuda]]`):
  EVERY Run command pins `.venv/bin/python` — bare `python` is `/usr/bin/python3`
  with no torch/yaml/sklearn; even CPU tasks need project deps for
  `scripts/adversarial_verify.py` + `scripts/summarize_artifact.py`.
- **Anti-poison-test** (the `.344 cause): any shipped test MUST assert against the
  script's real behavior. A test that fails against its own artifact poisons the
  pre-test gate and SKIPs the whole milestone.
- **Adversarial-verify** every artifact; declare `inference_substrate`; no
  vestigial GGUF/CUDA markers on aggregation/scoring artifacts.
- **No operator-curated edits, no CI trigger, no publish, no push.**

---

## 6. Hardware requirements

None beyond the dev box. `exp3774` needs the KV260 reachable via `ssh kria`
(SSH-only — never a host block-device check, per the KV260 SSH-Not-SD-Card
discipline); if unreachable it emits `blocked_kv260_ssh_unreachable` and exits.
All other tasks are CPU-only.

---

## 7. Invariants re-asserted at capstone

- `paper_ready` stays **TRUE** (G1 AND G2 AND G3 AND G4).
- FoVer **0.9131** stays **frozen** — `.345 *reproduces* it, never moves it.
- **Both energy routes stay bounded** — no re-grind of selection or generation.
- The next-Phase-3-thesis decision remains an **operator-seeding surface**; the
  loop does not self-commit.

## 8. SOTA-model note

`exp3769` (package/CLI/MCP E2E) is the only LLM-invoking task; it runs a small
CPU model (Qwen3.5-0.8B smoke tier) for the WIRING smoke and MAY include a SOTA
GGUF (`unsloth/Qwen3.6-35B-A3B-GGUF`) via `cached_sota_pair()` when cached — but
the smoke must pass on the small model so it is never GPU/cache-gated. All other
tasks are verifier-scoring against the cached FoVer corpus or pure aggregation
(no LLM), which is the *strength* of the headline's reproducibility.
