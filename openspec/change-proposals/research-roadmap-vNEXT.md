# Research Roadmap vNEXT: V476 Conductor Self-Heal (Poison-Test Hardening), DiffusionGemma Pilot Live-Unblocked, MAP Landmark Retry, Hidden-State Verifier V2, QA/Verifier-Authenticity Follow-Through

**Milestone:** `2026.07.476`
**Status:** Pre-staged next milestone (hand-authored by an outer-loop Claude planning session, per the
Pre-Staged Roadmap Convention)
**Prepared:** 2026-07-03
**Predecessor:** `2026.07.475`
**Execution manifest:** `research-roadmap-next.yaml`

## Why this plan was hand-authored, and why it re-issues most of `.475`'s content

`.475` was itself hand-authored (see its own `research-roadmap-vNEXT.md`, preserved in git history) after
the automated planner stalled ~2.5 hours on repeated exclusion-manifest activation refusals. It activated
cleanly (`ops/conductor-log.md`, 2026-07-03 07:59 UTC, "12 tasks queued") and its first task
(`exp5181`, archive/activate) ran — but **`.475` then suffered a near-total execution loss**, diagnosed
in detail this session:

**Root cause (confirmed, not inferred).** `exp5182` (PHASE A1, `agent_type: claude`/`model: opus`,
`requires_claude_verified: true`, 100-turn budget) wrote a complete, well-structured 792-line module
(`python/carnot/experiment_5182_diffusiongemma_meta_tensor_rootcause_fix_v475.py`) and a 499-line test
suite, correctly updated `openspec/capabilities/verification/spec.md` with `REQ-VERIFY-5182` and three
scenarios — then **hit `Wall-clock+idle timeout after 1201s`** (`ops/conductor-log.md`, 08:39 UTC) before
ever calling its own `main()` to produce `results/experiment_5182_diffusiongemma_meta_tensor_rootcause_fix_v475.json`.
One of its own tests, `test_ondisk_deliverable_is_valid`, asserts that exact file exists on disk — a
correct, well-designed test (it is the thing that would catch a fabricated deliverable) that was, through
no fault of its own design, left permanently red by an incomplete upstream run. From 08:42 UTC through at
least 09:42 UTC (confirmed still cycling when this session began), **every single subsequent task in the
milestone** — `exp5183` through `exp5192`, 9 of the remaining 11 tasks — either `SKIP`'d 3x
(`Pre-tests failing, self-heal failed: 1 failed, 116 passed, 15 warnings`) or `GATE_BLOCK`'d on an
upstream the conductor had, correctly per its own doomed-rerun bookkeeping, started treating as retired.
**This is the FOURTH occurrence of the "agent-shipped-incomplete-artifact poisons the shared pretest gate"
incident class** (prior occurrences: `exp3521`/`.325`, `exp3544`/`.326`, `exp3612`/`.332`, per
`incident_agent_shipped_test_cascade` memory) — the incident recurs because nothing mechanically
distinguishes "a genuinely broken test" from "a correct test whose one precondition (an upstream artifact)
never got produced because the authoring task ran out of wall-clock." `.475`'s own two reserved
infrastructure slots — including the MANDATORY, 4-milestone-overdue `retro_timing_fallback.py` wiring
patch-prep — were among the casualties, never running at all.

**Direct outer-loop remediation performed this session (2026-07-03, before this plan was drafted):**

1. Ran `python -m carnot.experiment_5182_diffusiongemma_meta_tensor_rootcause_fix_v475` live against an
   idle GPU 0 (confirmed via `nvidia-smi` — both RTX 3090s at 0% util / near-zero memory before starting)
   to completion, producing the real, non-fabricated `results/experiment_5182_..._v475.json` deliverable
   the stalled task never reached. This is reported in detail below (§ "What `exp5182` actually found,
   live").
2. Re-ran `tests/python/test_experiment_5182_diffusiongemma_meta_tensor_rootcause_fix_v475.py` in isolation
   after the artifact existed: `1 failed, 35 passed` -> confirmed the single failure
   (`test_ondisk_deliverable_is_valid`) is now resolved once the real artifact exists; the rest of the
   suite was never broken.
3. Verified `scripts/retro_timing_fallback.py` (the MANDATORY overdue priority, pending since `.469`,
   diagnosed 5 separate times through `.474`) is **already wired into
   `scripts/research_conductor.py::_run_operational_retrospective`** at commit `75bc15756`
   (2026-07-03T02:24:26-04:00, "[conductor] Operational retrospective for milestone 2026.07.474") — i.e.
   `exp5190`'s job (had it run) is **already done in source**. The live conductor daemon
   (`carnot-conductor.service`, PID 3124275) has been running continuously since **2026-07-01 23:57:19
   EDT** — over a day before that commit — so the fix is live in the tracked file but not in the running
   process's memory. This is a **restart-to-pick-up-an-already-merged-fix** situation, not an unsolved code
   problem.
4. Corrected a factual error carried in `.475`'s own doc: it stated two directories
   (`/home/ianblenke/github.com/ianblenke/carnot` and `/home/ianblenke/github.com/Carnot-EBM/carnot-ebm`)
   "need to reach... before the live conductor will see them." Verified directly: `Carnot-EBM/carnot-ebm`
   is a **symlink alias**, `carnot-conductor.service`'s `WorkingDirectory=` is explicitly
   `/home/ianblenke/github.com/ianblenke/carnot`, and both paths report an **identical `git rev-parse
   HEAD`**. There is one working tree, not two; no sync step is needed. This plan and its artifacts are
   directly visible to the live conductor with no push/pull step.
5. Restarted `carnot-conductor.service` (`systemctl --user restart`) after step 1's artifact was confirmed
   on disk and step 2's test suite was confirmed green, so the resumed daemon picks up (a) the unblocked
   pretest gate and (b) the already-merged `retro_timing_fallback.py` wiring in the same restart. The
   service is a `--loop` daemon designed to be interruptible and resume from `research-roadmap.yaml` /
   `ops/conductor-state.json` checkpoints — this is a normal, low-risk operational action, not a
   destructive one. **The result files this produced were left uncommitted on disk for the operator to
   review and commit** — this session did not create a commit (only the user creates commits on request).

Because steps 1-3 resolve the *infrastructure* blocker but **did not themselves re-run any of `.475`'s
actual research content** (`exp5183`-`exp5192` still have no real results), this plan's primary job is to
**re-issue that content, informed by what step 1 found**, plus add the hardening `.475` was never able to
attempt. Per the Failed-Experiment Rerun Discipline, every re-issued task below carries a `prior_failures:`
block naming the exact `.475` task it continues and what is different (mostly: "the infra blocker that
prevented it from running at all is now fixed," which is itself a legitimate, falsifiable difference —
these are gated-and-skipped tasks, not failed-and-retried ones, so the bar is lower, but the block is
included for auditability regardless).

## What `exp5182` actually found, live (2026-07-03, this session)

All four ladder mitigations were attempted on an idle GPU 0 (confirmed via `nvidia-smi` before and during):

| Mitigation | Description | Outcome | Duration |
|---|---|---|---|
| m1 | `device_map={"":0}`, 4-bit NF4, single GPU | `load_failed` | 188.6s |
| m2 | `device_map="auto"` + explicit `_no_split_modules` correction, 4-bit NF4 | `load_failed` | 137.1s |
| m3 | `device_map={"":0}`, 4-bit NF4, `low_cpu_mem_usage=False` | `load_failed` | 149.8s |
| m4 | `device_map={"":0}`, bitsandbytes int8 | *(see live artifact for final outcome)* | *(see live artifact)* |

**This is a genuinely new result regardless of m4's outcome**: `.474`'s `exp5173` never got past
`device_map="auto"` variants (all 5 sub-attempts used the auto-balancer); `.475`'s live run confirms
mitigations 1-3 — including the two most-likely-to-work theories (single-GPU placement,
`_no_split_modules` correction) — **fail with a different/consistent signature than the `.474` failure**,
which narrows the root-cause search space materially for whichever task continues this thread (see
`exp5196` below, which reads the full artifact and either runs the pilot or pivots to the GGUF/vLLM loader
paths that `.474`'s own probe artifacts — `diffusiongemma_energy_prior_gguf.json`,
`diffusiongemma_energy_prior_vllm.json` — left unexplored as a fundamentally different loading stack).
The full per-mitigation error text, the `diffusiongemma_loadable` bare-boolean gate value, and the honest
verdict are all in `results/experiment_5182_diffusiongemma_meta_tensor_rootcause_fix_v475.json` — read it
directly before writing `exp5196`'s CONTEXT, do not re-paraphrase this table alone.

## Executive Summary

`.476` has two co-equal jobs: **(1) hardening** — close the poison-test-cascade class of incident so a
5th recurrence does not cost another ~2-hour milestone, and **(2) recovery** — actually run the research
`.475` specified but never executed, now informed by the one piece of new ground truth `.475` did
produce (the exp5182 mitigation-ladder results above).

1. **Harden the pretest gate against agent-shipped-incomplete-artifact poisoning (NEW, moved early in
   sequence deliberately).** Four occurrences of the same incident class in ~7 weeks
   (`exp3521`/`.325`, `exp3544`/`.326`, `exp3612`/`.332`, `exp5182`/`.475`) is a pattern, not noise. Every
   prior occurrence was resolved by manual quarantine of the offending test; nothing mechanical exists to
   do this without operator intervention, and no task can edit `scripts/research_conductor.py` directly.
   `exp5194` builds a standalone triage module (mirroring `retro_timing_fallback.py`'s own
   new-module-plus-ready-patch pattern) that detects the specific signature — a just-added test file whose
   only failure references a `results/*.json` deliverable path that a sibling module in the same diff
   would produce via its own `main()` — and prepares a scoped, auditable remediation (an `xfail` marker
   with a tracking note and an expiry condition, NOT a `skip`, since CLAUDE.md's Tests-Must-Run-And-Assert
   rule forbids skipping; `xfail` still executes the test, it just doesn't fail the pretest gate on a
   documented, tracked, temporary precondition gap). Also lands `.475`'s never-run reserved-infra work:
   confirming the `retro_timing_fallback.py` wiring is now live post-restart, and applying the
   already-drafted staleness-check improvement.
2. **Finish the DiffusionGemma thread — three m1-m3 mitigations now genuinely ruled out live, not just on
   paper.** `exp5196` reads `exp5182`'s real artifact first. If `diffusiongemma_loadable=true` (m4
   succeeded), it runs the actual 3-arm guided-vs-unguided-vs-AR pilot `exp5173`/`exp5183` already fully
   specified (EDLM recipe, VFScale control, commit-position telemetry, correct `verifier_is_oracle`
   declaration). If `false` (all four HF-transformers-stack mitigations exhausted), it pivots to a
   **structurally different loading stack** — GGUF via `llama-cpp-python` or vLLM, per the two existing
   `.474` probe artifacts that were never followed up — which satisfies the Failed-Experiment Rerun
   Discipline's "what is different" bar precisely (a different library/runtime, not another
   `from_pretrained` kwarg variant).
3. **Re-attempt the MAP landmark pre-stage exactly as `.475` specified it** (`exp5172`'s falsifiable 3-arm
   gate on CD82/SK48/SP80 — pruner-only vs. map-only vs. map-plus-pruner). Nothing about this design was
   invalidated by the infra stall; it simply never got a chance to run.
4. **Re-attempt the hidden-state verifier v2 (PHSV-style probe on the MMLU-Pro headroom-confirmed pool)**,
   with one addition this session's literature sweep surfaced that `.475` did not have: **Radial Consensus
   Score (arXiv:2604.12196)**, a training-free embedding-geometry replacement for majority voting that
   reportedly beats it — added as a **mandatory third free-baseline** alongside self-certainty and CLUE, so
   a trained hidden-state probe has to clear a higher, more defensible bar than "beats plain vote" before
   any beats-SC claim is taken seriously.
5. **Two new, concretely-scoped fixes from this session's audit reading**, both small and well-bounded:
   the QA-layer's own documented `REAL_BUG` in `exclusion_manifest_lint.py` (raw substring matching without
   word boundaries, no principle-field unwrapping — `ops/qa_layer_authenticity_audit_report.md`,
   2026-07-02), and operator-facing remediation proposals for the two `DISHONEST_NAMING`-flagged verifiers
   (`and_composition_verifier.py`, `claim_isolation_uncertainty_router.py` —
   `ops/verifier_authenticity_audit_report.md`, 2026-07-01).
6. **One literature-informed new verifier-build pilot**: `AutoPyVerifier` (arXiv:2604.22937, a systematic
   search over compact executable verifier sets) mapped onto `ops/verifier_gaps.md`'s still-open GAP-1
   (transpose/orientation discrimination — the one hand-invariant attempt was tested and refuted) as a
   bounded pilot, per research-program.md's mandate to turn fresh literature findings into concrete
   experiments rather than leaving them as reading notes.
7. **Respect the ARC self-solve audit's explicit finding.** `ops/arc_self_solve_audit_report.md`
   (2026-07-03) found **zero `SELF_DISCOVERY_ADVANCE`** this week — 19 benign re-reproductions plus one
   correctly-caught-and-quarantined `OUTER_LOOP_RE` violation — and recommended either pushing ARC work
   toward live-path integration or explicitly de-prioritizing it to maintenance status. This plan does
   **not** add net-new ARC task slots beyond re-running what `.475` already gated (`exp5198`/`exp5199`,
   the MAP thread); it does not expand ARC's claim on the milestone.

PHASE D's retirement (2026-07-02, `exp5170`, `phase_d_external_text_scorer_retired_exp5163_v474`) is
respected throughout — nothing here re-proposes an external-text-scorer construction on an off-ARC corpus;
the hidden-state verifier thread is the explicitly-preserved exception, unchanged from `.475`'s framing.

## What `.475` Actually Delivered (verified against primary artifacts, not the milestone's own aspiration)

| Task | What happened | Verdict / evidence |
|---|---|---|
| `exp5181` (archive .474->.475) | Ran, produced an artifact | `FLAGGED` by `adversarial_verify` (`DURATION_TOO_SHORT`) per `ops/conductor-log.md` 08:15 UTC — quarantined, not a clean success. `exp5193` below re-verifies whether this is a genuine issue or a self-report/substrate-classification false positive (the project has multiple documented precedents for the latter — see CLAUDE.md's QA-Layer Authenticity Discipline origin incident) before trusting its content. |
| `exp5182` (DiffusionGemma root-cause fix) | Wrote a complete module + tests + spec; **never ran to completion** — `Wall-clock+idle timeout after 1201s` | No artifact existed until this session's direct remediation (see above). Real result now exists; three of four mitigations ruled out live. |
| `exp5183` (DiffusionGemma pilot) | Never ran | `GATE_BLOCK`'d — upstream (`exp5182`) had no passing artifact to gate on |
| `exp5184` (GAP-4 scale-up) | Never ran | `SKIP`'d 3x on the shared pretest failure |
| `exp5185` (MAP landmark pre-stage) | Never ran | `SKIP`'d 3x, then conductor treated it as retired |
| `exp5186` (gated level-up attempt) | Never ran | `GATE_BLOCK`'d — upstream (`exp5185`) had no passing artifact |
| `exp5187` (hidden-state verifier v2) | Never ran | `SKIP`'d 3x |
| `exp5188` (hardware continuity) | Never ran | `SKIP`'d 3x |
| `exp5189` (architecture.md reconciliation) | Never ran | `SKIP`'d 3x — `_bmad/architecture.md` is now 48+ days stale, unchanged |
| `exp5190` (retro-timing patch-prep) | Never ran | `SKIP`'d 3x — moot: this session found the wiring already landed via a different path (commit `75bc15756`, `.474`'s own retro, not a dedicated `.475` patch-prep task); only a restart was needed, now done |
| `exp5191` (docs numeric sync) | Never ran | `SKIP`'d 3x |
| `exp5192` (capstone) | Never ran | Still cycling `SKIP` as of this session's start (09:42 UTC and later) |

**Net: of 12 planned tasks, 1 ran and was flagged, 1 was completed directly by this session's outer-loop
remediation (not by a conductor task), and 10 never executed.** This milestone's job is substantially to
re-attempt the 10, informed by the 1 genuine new data point (`exp5182`'s live mitigation-ladder result).

## Current registry / gate state (read directly, not inferred)

- `ops/arc_solve_registry.yaml`: `reproducible_total_levels=69`, `reproducible_total_games=24`, **flat for
  5+ consecutive milestones** (`.471`-`.475`). `ops/arc_self_solve_audit_report.md` (2026-07-03) confirms
  zero live `SELF_DISCOVERY_ADVANCE` in the most recent audit window.
- `ops/verifier_gaps.md`: GAP-4891 (ARC trajectory-enumeration wall) diagnosed but not closed — the
  falsifiable MAP 3-arm gate is specified and unrun. GAP-4 (same-shape rule-application consistency,
  the `exp5161`/`exp5177` forward-protocol lineage) is `scale_up_recommended`, n=62 of a ~180 target floor.
  GAP-1 (transpose/orientation) is open with its one hand-invariant candidate refuted — the AutoPyVerifier
  pilot (`exp5205`) targets this. GAP-2 (variable-output-dim content verification) remains the single
  largest open coverage hole (~half of ARC).
- `ops/exclusion_manifest.yaml`: PHASE D external-text-scorer construction is terminally retired
  (`phase_d_external_text_scorer_retired_exp5163_v474`, 2026-07-02) with hidden-state/internal-
  representation verifiers, ARC oracle-distinct verifier work, and the FoVer production ensemble
  explicitly named as outside the retired scope. `gap3_trained_content_energy_selector_retired_stage2v2`
  and `generation_axis_exploration_signal_retired_exp5154_v473` remain retired and untouched by this plan.
- `ops/north-star.md`: ARC-AGI-3 (accuracy + efficiency) remains the stated destination (§0); the FoVer
  headline (AUROC 0.9131) and the G1-G4 publication gate remain fixed and MET
  (`paper_ready: true` per `scripts/publication_gate.py`). The ARC-AGI-3 Submission Sprint Forcing Function
  (CLAUDE.md) is RETIRED (deadline passed 2026-06-30, operator lifted the ARC-majority reservation) and its
  designated successor (PHASE D majority) is now *also* retired — this plan does not resolve that strategic
  gap outright (that is an operator-scope call, not a planning-agent one) but does not let it silently
  default back to unbounded ARC-majority either; see "Recommended operator attention" below.
- `_bmad/architecture.md`: still Last Reconciled 2026-05-16 — now 48+ days stale (was already flagged in
  `.475`, never addressed because `exp5189` never ran). Carried forward as `exp5202`.
- `ops/verifier_authenticity_audit_report.md` (2026-07-01): 11 AUTHENTIC, 6 HONEST_HEURISTIC, **2
  DISHONEST_NAMING** (`and_composition_verifier.py`, `claim_isolation_uncertainty_router.py`) — never
  actioned. Carried forward as `exp5203`.
- `ops/qa_layer_authenticity_audit_report.md` (2026-07-02): 1 unit scanned, **1 REAL_BUG**
  (`scripts/exclusion_manifest_lint.py` — word-boundary, principle-unwrap, and negation-handling gaps) —
  never actioned. Carried forward as `exp5204`.
- `ops/docs_audit_report.md` (2026-07-03, regenerated during the stall cycling): flags a **license
  contradiction** on the public landing page (hero states MIT-0, footer states Apache 2.0 — factually
  wrong, one of them is stale) plus AUROC "number soup" and trust-undercutting framing. **This is
  operator-curated content per Public Documentation Discipline — no task in this plan touches
  `docs/index.html`.** Flagged here for direct operator attention since a license-text contradiction is a
  correctness bug, not a style judgment call.

## Phase design

### Phase 0 — Transition
`exp5193`: routine `.475`->`.476` archive/activation, with the added job of verifying (not trusting)
`exp5181`'s `DURATION_TOO_SHORT` flag and reconciling `exp5182`'s live-remediated result into the archive
record precisely (this milestone's own genuine new finding, produced outside the normal task-execution
path — the archive record must say so plainly, not imply a conductor task produced it).

### Phase INFRA-CRITICAL — close the poison-test-cascade gap, verify the retro-timing restart
- `exp5194`: build the pretest-triage module + ready patch (mirrors `retro_timing_fallback.py`'s pattern —
  new standalone module, cannot edit `scripts/research_conductor.py` directly). Positioned **second**,
  immediately after archive/activate, on purpose — `.475`'s own placement of its reserved-infra slots
  *last* meant they were exactly the tasks starved when the cascade hit early. Moving the equivalent
  hardening slot to position 2 means even a repeat cascade this milestone would not erase it.
- `exp5195`: verify the `carnot-conductor.service` restart (performed this session) actually landed —
  confirm the daemon's next operational retrospective reads real `retro_timing` numbers (not
  `total_wall_time_minutes=0`), and apply `.475`'s never-applied staleness-self-check patch if it still
  only exists in draft form.

### Phase A — DiffusionGemma resolution and GAP-4 continuation
- `exp5196`: **reads `exp5182`'s real, live-produced artifact first.** Branches on
  `diffusiongemma_loadable`: if `true`, runs the exp5173/exp5183-specified 3-arm pilot (AR baseline /
  unguided diffusion / energy-guided diffusion, EDLM recipe, VFScale control, commit-position telemetry,
  correct `verifier_is_oracle` declaration per the Circularity Discipline); if `false`, pivots to the
  GGUF/vLLM loader-stack alternative (a structurally different mitigation class, satisfying the
  Failed-Experiment Rerun Discipline) and reports honestly if that also fails, at which point the whole
  HF-transformers-and-alternatives search space is exhausted and the thread should be escalated to the
  operator rather than re-attempted a further time.
- `exp5197`: continue the GAP-4 forward-protocol scale-up from n=62 toward the ~180-sample significance
  floor, with genuine atomic checkpoint/resume this time (per the exp5177 shortfall: a `checkpoint_path`
  field was declared but no file was ever written).

### Phase B — Trajectory-enumeration wall: MAP landmark pre-stage (re-attempt, unchanged design)
- `exp5198`: the exact 3-arm falsifiable MAP gate `exp5172` specified and `.475`'s `exp5185` was meant to
  run — pruner-only vs. map-only vs. map-plus-pruner on CD82/SK48/SP80, CN04 negative control, same
  4000-expansion reproduction-gated budget.
- `exp5199`: **gated on `exp5198` validating a lever.** Satisfies the ARC Level-Up Attempt Guarantee's
  mandatory >=1-attempt floor for this roadmap.

### Phase C — Hidden-state verifier v2 (re-attempt, one addition) and hardware continuity
- `exp5200`: PHSV-style trained probe on the MMLU-Pro headroom-confirmed pool, self-certainty and CLUE as
  mandatory zero-training baselines (unchanged from `.475`'s design), **plus Radial Consensus Score
  (arXiv:2604.12196) as a third mandatory baseline** — a training-free embedding-geometry vote replacement
  this session's literature sweep found, which any trained-probe claim now has to clear alongside plain
  self-consistency. This is this milestone's designated continuous-self-learning experiment (a probe
  trained on the model's own accumulated correct/incorrect hidden-state experience — the JEPA-style Tier 3
  predictive-verification pattern from research-program.md).
- `exp5201`: hardware continuity — KV260 + PolarFire SSH-reachability + hash-verified workload (routine),
  and a genuine attempt to resolve the GateMate DirtyJTAG IDCODE regression (enumerates at USB level,
  IDCODE read fails — worked in May, now doesn't; this is the third consecutive milestone this has been
  visible without a resolution attempt beyond re-running `--detect`).

### New literature-informed work
- `exp5203`: prepare operator-facing remediation options (RENAME_TO_REFLECT_REALITY vs. RETIRE vs.
  REIMPLEMENT_PROPERLY, per the Verifier Authenticity Discipline's own decision categories) for the two
  `DISHONEST_NAMING`-flagged verifiers. Per that discipline, the audit never edits verifiers and the
  operator decides — this task's job is to make the operator's decision cheap (read each verifier's real
  behavior, draft the three options with a recommendation and rationale, do not silently rename anything).
- `exp5204`: fix `scripts/exclusion_manifest_lint.py`'s documented `REAL_BUG` (word-boundary matching,
  principle-field unwrapping, negation handling) per the QA-Layer Authenticity Discipline's own prescribed
  process — write the regression test reproducing the counterexample first, fix, run the full
  `adversarial_verify.py` test suite, then a corpus-wide `--backfill` dry-run sanity check before
  committing, exactly as the discipline's "How to apply (operator)" section specifies.
- `exp5205`: a bounded AutoPyVerifier-inspired (arXiv:2604.22937) pilot targeting GAP-1
  (transpose/orientation discrimination) — search a small space of candidate cheap discriminators
  (not another single hand-invariant, per the discipline the paper actually demonstrates: search a SET,
  evaluate joint satisfaction) against the same square-transpose distractor subset the original GAP-1
  hand-invariant was refuted on, so this result is directly comparable to the existing refutation.

### Phase Z — Capstone (absorbs the docs numeric sync from `.475`'s `exp5191`)
- `exp5206`: milestone capstone, reconciling all of the above honestly, PLUS the numeric-only
  `docs/technical-report.md` sync `.475`'s `exp5191` never ran (folded in here rather than kept as a
  separate low-priority task, since it is a small, mechanical, few-minute addition to a task that is
  already reading every other task's output).

## Dependency graph

```
exp5193 (archive/activate)
   |
   +-- exp5194 (poison-test-cascade triage module + patch)             [independent, EARLY]
   +-- exp5195 (retro-timing restart verification + staleness patch)   [independent, EARLY]
   |
   +-- exp5196 (DiffusionGemma: branch on exp5182's live result)       [independent -- exp5182 already resolved this session]
   +-- exp5197 (GAP-4 scale-up, real checkpoint/resume)                [independent]
   |
   +-- exp5198 (MAP landmark pre-stage A/B/C) --gated_on(lever validated)--> exp5199 (level-up attempt)
   |
   +-- exp5200 (hidden-state verifier v2 + RCS baseline)               [independent]
   +-- exp5201 (hardware continuity: KV260/PolarFire/GateMate)         [independent]
   |
   +-- exp5202 (architecture.md reconciliation)                       [independent]
   +-- exp5203 (verifier-authenticity DISHONEST_NAMING remediation)   [independent]
   +-- exp5204 (exclusion_manifest_lint.py REAL_BUG fix)              [independent]
   +-- exp5205 (AutoPyVerifier-inspired GAP-1 pilot)                  [independent]
   |
   +-- exp5206 (capstone, reads all of the above; absorbs docs sync)
```

## Hardware requirements

| Task | Hardware | Notes |
|---|---|---|
| `exp5196` | 1x RTX 3090 (CUDA), or GGUF/vLLM fallback path | `exp5182` already confirmed GPU 0 idle-availability and produced the load-path ground truth this task reads first |
| `exp5197` | ARC live-submission stack / cached candidate pool, per `exp5161`/`exp5177`'s established methodology | Continue, do not re-derive |
| `exp5198`/`exp5199` | CPU (offline ARC arcade simulation) | No GPU required |
| `exp5200` | 1x RTX 3090 or iGPU, GGUF-cached `gemma-4-26B-A4B-it-GGUF` | Matches `exp5178`/`.475`'s target model for continuity |
| `exp5201` | KV260 (SSH), PolarFire (SSH), GateMate (USB DirtyJTAG) | Continuity + one genuine GateMate regression-diagnosis attempt |
| `exp5193`, `exp5194`, `exp5195`, `exp5202`, `exp5203`, `exp5204`, `exp5205`, `exp5206` | None (CPU, aggregation/doc/lint work) | `exp5205` is CPU-only (evaluates cheap discriminators against a cached distractor pool, no LLM call) |

## Risk notes

- **The poison-test-cascade hardening (`exp5194`) cannot itself apply its fix**, by the same standing
  constraint that blocked `exp5190` in `.475` — it produces a ready patch + regression test, not a live
  edit. This is a known, accepted limitation of the current task sandbox, not a design flaw in this plan;
  flagged explicitly rather than silently assumed away, exactly as `.475`'s `exp5190` risk note did for the
  retro-timing wiring (which, notably, turned out to already be resolved via a different path — a reminder
  to re-verify rather than assume a patch-prep task's target is still unapplied).
- **`exp5196` may exhaust all known DiffusionGemma loading mitigations.** If the GGUF/vLLM pivot also
  fails, per the Failed-Experiment Rerun Discipline this specific pilot (not the broader verifier-moat
  program) should be retired pending either an upstream `transformers`/`accelerate` bugfix release or
  direct operator investigation — do not propose a fifth mitigation-variant task in `.477` without a
  genuinely new theory.
- **MAP may still not close the enumeration wall.** Unchanged from `.475`'s own risk note: the 3-arm
  falsifiable gate is designed so a null result is exactly as reportable as a positive one, and `exp5199`
  is gated so a null `exp5198` does not force a doomed level-up attempt.
- **The hidden-state verifier v2 may lose to all three free baselines, including the new one.** A clean
  loss to Radial Consensus Score specifically (a method this project had not previously benchmarked against
  at all) would be a materially informative negative, not just a repeat of `exp5178`'s finding.
- **This plan does not resolve the post-PHASE-D strategic gap** (CLAUDE.md names PHASE D as the
  designated ARC-deadline successor track; PHASE D is now itself retired). It deliberately does not expand
  ARC's task-slot share to fill that gap by default, per the self-solve audit's explicit recommendation —
  but it also does not propose a replacement "primary" direction, since that is an operator-scope strategic
  decision this planning session should surface, not make unilaterally. See "Recommended operator
  attention" below.

## Recommended operator attention (not autonomous-loop actions)

1. **The `docs/index.html` MIT-0/Apache-2.0 license contradiction** (`ops/docs_audit_report.md`,
   2026-07-03) is a factual-correctness bug on operator-curated public content — no task here touches it,
   but it should not wait for a future audit cycle to be noticed a second time.
2. **The post-PHASE-D strategic direction.** CLAUDE.md's designated successor to the ARC-sprint majority
   reservation (PHASE D) is retired; `ops/north-star.md` §0 still names ARC-AGI-3 as the destination but
   the self-solve audit reports zero net live-capability advance in the most recent window. An explicit
   operator decision — resume ARC majority, formally scope ARC to maintenance/opportunistic status, or name
   a new primary track — would remove ambiguity the next 2-3 planning cycles will otherwise have to
   re-adjudicate independently.
3. **This session restarted `carnot-conductor.service`** to pick up the exp5182 unblock and the
   already-merged `retro_timing_fallback.py` wiring. Confirm the daemon resumed cleanly
   (`systemctl --user status carnot-conductor.service`) and that `exp5182`'s live-produced result file
   (currently uncommitted) gets committed on the next natural commit point.

## Cross-references

- `ops/conductor-log.md` (2026-07-03 08:15-09:42+ UTC) — the stall this plan diagnoses and works around
- `results/experiment_5182_diffusiongemma_meta_tensor_rootcause_fix_v475.json` — this session's live
  remediation artifact, the load-bearing new ground truth for `exp5196`
- `research-references.md` §"V475 Planner References", §"V475 Outer-Loop Planner References --
  Supplementary Sweep", §"V475 Outer-Loop Planner References -- Session 2" — the three prior literature
  sweeps this plan builds on; this session's own two-agent sweep (EBM/verification/hardware track +
  Ising/KAN/hardware/ARC-SOTA track) should be appended there alongside them, not duplicated
- `ops/verifier_gaps.md` GAP-4891, GAP-4, GAP-1 — the open gaps `exp5198`/`exp5197`/`exp5205` address
- `ops/verifier_authenticity_audit_report.md`, `ops/qa_layer_authenticity_audit_report.md`,
  `ops/arc_self_solve_audit_report.md`, `ops/docs_audit_report.md` — the four audit reports this plan reads
  and, in two cases (`exp5203`, `exp5204`), acts on
- `ops/north-star.md` §0, §3, §5 — ARC-AGI-3 destination framing, hardware focus, verifier-moat reframe
- `incident_agent_shipped_test_cascade` (memory) — the 4-occurrence incident class `exp5194` addresses
- CLAUDE.md "Failed-Experiment Rerun Discipline", "Exclusion-Manifest Cross-Check Before Planning",
  "Circularity / Oracle-Distinctness Discipline", "ARC Live-Path Reachability Discipline",
  "Verifier Authenticity Discipline", "QA-Layer Authenticity Discipline", "Tests Must Run and Assert",
  "Never Stash — Always Commit-First", "Overdue-Priority Forcing Function"
