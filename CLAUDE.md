# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Claude Code Guidelines
If you notice the user's request is based on a misconception, say so.
Never claim 'all tests pass' when output shows failures.
Keep text between tool calls to <=25 words.
Spawn an adversarial sub-agent to review non-trivial changes before reporting completion.

## Documentation and Communication Standards

- **No emojis in public documentation.** README, landing page, technical report, and usage guide must be emoji-free. Professional presentation is critical for community credibility.
- **Verbose layman explanations in code.** All docstrings and comments should explain WHY, not just WHAT. Write for engineers who are not EBM specialists.
- **Never remove existing content** from ops/spec docs when updating. Add new sections, move completed items to "Completed" — do not delete historical records.
- **All headline results must have live GPU provenance.** Simulated and unverified results are preserved in the repo but labeled explicitly and excluded from headline claims.

## Security Requirements

- **All embedded secrets must use SOPS encryption** at rest. Never commit plaintext API keys, tokens, or credentials.
- **Code execution sandbox:** Use `CARNOT_USE_SANDBOX=1` for gvisor-sandboxed execution of untrusted code. Default is in-process exec for development speed.
- **trust_remote_code is gated:** HuggingFace model loading requires `CARNOT_TRUST_REMOTE_CODE=1` to enable remote code execution. Default is False (safe).
- **Production autoresearch:** Use Docker with the gVisor (runsc) runtime for sandbox isolation when running autonomous experiments in production. Firecracker was initially considered but cannot pass GPUs through, and the pipeline needs CUDA/ROCm; gVisor intercepts syscalls in userspace and plays nicely with nvidia-container-toolkit.

## Project Vision (Three Phases)

1. **Phase 1 (current):** Verify and repair LLM outputs using constraint-based energy models. Ship a useful product.
2. **Phase 2 (medium-term):** Hardware acceleration via Extropic TSU, FPGA Ising machines, and potentially photonic computing.
3. **Phase 3 (long-term):** Evolve into an open-source foundation model based on hardware-acceleratable EBM/EBT. Functional parity with Kona — continuous latent space, non-autoregressive reasoning, self-correcting. Apache 2.0, hardware-portable.

The verify-repair pipeline is Phase 1, not the endgame. Every architectural decision should ask: "does this move us toward the foundation model?"

## Decentralization-Respecting Design Constraints (MANDATORY)

Every architectural decision must also ask: **"does this preserve users'
ability to run Carnot without depending on a closed-source vendor we do
not control?"** If the honest answer is no, the decision is wrong.

**The threat model**: closed-source frontier models can be deprecated,
re-priced, withdrawn, or geofenced at any time. APIs change. Vendors
fail or capture markets. Cloud providers raise prices once switching
costs are high enough. Distribution channels (model hubs, package
registries, container registries) become gatekeepers. Carnot's value
proposition — second-pair-of-eyes verification grounded in objective
energy — must survive any of these failures.

**The non-negotiable rules**:

1. **Local-first using open models, always.** Every Carnot capability
   must work end-to-end with locally-hosted open-weight models
   (Qwen / Gemma / Llama / equivalents). The `cached_sota_pair()`
   helper in `scripts/experiment_template.py` is the canonical
   pattern; experiments that depend on a closed-weight model with no
   local fallback are not acceptable.

2. **Closed frontier-model integration is optional, never required.**
   Carnot may be more *useful* when paired with Claude / GPT / Gemini
   for capabilities those models uniquely provide (broad world
   knowledge, long-context recall). It must never be *broken* without
   them. If a feature only works with a closed-weight upstream, it
   ships behind an opt-in flag with a clearly-labelled
   "decentralization-degraded" tier in the docs.

3. **Distribution mirroring for any published artifact.** Trained
   weights, model cards, datasets, and Python packages must be
   published through at least two independent channels (e.g.
   HuggingFace + IPFS, or HF + a Carnot-controlled gitea/git mirror).
   The conductor's multi-URL git remote (gitea + github) is the
   precedent — apply the same pattern to model weights and any other
   downstream-consumed artifact. Single-point distribution failure
   is unacceptable.

4. **Multiple integration surfaces in parallel.** Carnot exposes its
   capabilities via Python API, CLI, MCP server, and HTTP REST. None
   of these is allowed to become the *only* well-trodden path. If
   one integration surface drifts ahead in features and the others
   atrophy, the project becomes implicitly locked into that surface.
   Treat this as a review gate on any change touching public API
   surfaces.

5. **Hardware portability as a political requirement, not just an
   engineering one.** Already encoded in REQ-KONA-006 and the
   `SamplerBackend` protocol. The political dimension: nation-states,
   institutions, and individuals subject to compute-resource
   sanctions or supply-chain constraints must still be able to run
   Carnot. The KV260 / ECP5 / Nexus open-FPGA tracks and the future
   Extropic XTR-0 path are *sovereignty infrastructure*, not just
   accelerators.

6. **Per-call data minimization on closed-weight LLM integrations.**
   Any closed-weight call must declare a `data_handling_class`
   (`minimize` / `summarize` / `redact` / `pass_through`) with
   `minimize` as the default. Customer prompts, internal reasoning
   traces, and verification artifacts must not flow through
   closed-weight providers without an explicit, logged decision.

7. **No vendor-specific abstractions in the core.** The core
   verifier stack (`python/carnot/verify/`, `python/carnot/pipeline/`)
   must not import from vendor-specific SDKs. Vendor adapters live
   in clearly-named submodules (`closed_weight/`, `proprietary/`)
   that the core depends on through abstract protocols
   (`SamplerBackend`, `LLMComponent`, etc.) — never directly.

**How to apply this rule:**

- When drafting a new change proposal, add a one-line "Decentralization
  implications" subsection answering whether the proposal preserves
  rules 1–7.
- When reviewing code, refuse changes that add closed-weight
  dependencies to the core or that remove a local fallback path.
- When publishing weights or papers, ensure mirroring (rule 3) is in
  place *before* the announcement, not after.
- When the conductor or planner generates work, the proposals it
  drafts must respect these rules. The planner prompt at
  `scripts/research_conductor.py:_plan_next_milestone()` should be
  updated to include this section as required reading.

**Why this is in the Project Vision, not a Risks section.** Phase 3's
endgame is an open-source foundation model. Every decision before then
either compounds toward sovereignty or compounds toward enclosure.
There is no neutral middle position that will accidentally land on
sovereignty at Phase 3; it has to be a continuous, conscious choice
from Phase 1 onward. Naming it explicitly here makes the choice
auditable.

## Failed-Experiment Rerun Discipline (MANDATORY)

If an experiment fails to complete, times out, blocks on a gate, or
produces a `partial` / `not_viable` / `still_*` honest verdict, **the
same experiment must not be re-proposed in a subsequent milestone
without an explicit plan to address the suspected underlying cause.**

This rule directly answers the .60–.65 retros' "slow-5 carryover"
finding: Exps 786/527/491/627/603 ran unguarded for six consecutive
milestones, burning ~224 min/milestone of wall time without
progress, because the planner kept proposing them and the conductor
kept running them. That pattern must not recur.

**Definitions.**

A task is "failed" for the purposes of this rule when its
honest_verdict maps to ⚠️ Blocked, ⚠️ Research Finding (any token in
`partial / inverted / insufficient / no_improvement / still_wrong /
no_delta / below / regression / negative / flat / plateau /
collapsed`), or ❌ Failed in the in-process reconciler's mapping
table (`scripts/in_process_doc_reconcile.py:_PARTIAL_TOKENS`,
`_BLOCKED_TOKENS`, `_FAILED_TOKENS`).

A "rerun" is any new task whose substantive scope (the experiment
script's behaviour, the deliverable shape, the underlying technique)
matches a previously-failed task. Trivial relabeling does not
qualify as a different experiment.

"Addressing the suspected underlying cause" means the new task's
specification explicitly:

1. **Names the prior failure** — by experiment ID and verdict.
2. **Names the diagnosed root cause** — what specifically failed.
   "We don't know" is a valid honest answer, in which case the rerun
   is rejected because root-cause-unknown does not improve the
   prior odds.
3. **Names what is different** — the technique, the corpus, the
   parameters, the gate condition, or the upstream prerequisite that
   has changed since the prior failure. If nothing has changed, the
   rerun is rejected.
4. **States a falsifiable acceptance gate** — if the new attempt
   produces the same verdict as before, the experiment is *retired*
   from future milestones (added to a permanent exclusion list)
   rather than re-proposed yet again.

**Planner responsibility.** When the planner generates a new
milestone roadmap (`research-roadmap-next.yaml`), it must consult
the project's failure record (`research-complete.yaml`,
`results/operational_retro_*.json`, `ops/changelog.md`) before
proposing any task. For any task whose scope matches a prior failed
attempt, the YAML must include an optional `prior_failures:` field
with structure:

```yaml
prior_failures:
  - experiment_id: exp850-sota-code-repair-v5
    verdict: model_not_cached
    addressed_by: "Exp 849 shipped GGUFCacheResolver; Exp 855 LIVE-ENV
                   permanent fix; this attempt explicitly downloads the
                   model from HuggingFace before invoking the cache."
    retire_if_same_verdict: true
```

**Conductor responsibility.** Before launching an experiment, the
conductor consults the failure record. If the task's scope matches a
prior failure and the YAML has no `prior_failures:` entry that
satisfies the four definitions above, the conductor refuses to
launch — writes a `blocked_doomed_rerun_no_root_cause` artifact and
moves on.

**Retirement.** When a `retire_if_same_verdict: true` task fails
again with the same verdict, the experiment ID is added to the
permanent exclusion manifest (`ops/exclusion_manifest.yaml` —
mechanism already exists per `_ensure_exclusion_manifest_loaded` in
`scripts/research_conductor.py`). Future planners cannot propose it
without explicit human override.

**Why this is in CLAUDE.md, not just a code change.** The rule
governs what the *planner* does, and the planner reads CLAUDE.md as
required reading. Mechanical enforcement at the conductor layer is
the safety net; the planner respecting the rule at design time is
the primary discipline.

**Pending mechanical enforcement.** A failure ledger module +
conductor pre-launch check are scoped at
`openspec/change-proposals/failed-experiment-rerun-enforcement.md`
(separate proposal). Until that ships, this rule is enforced by
honest discipline at the planner layer alone.

## Operational Principles

- **Meta-reflection:** After milestones, evaluate HOW work was executed, not just WHAT was produced. Feed operational improvements back into the process.
- **Continuous improvement:** Domain (verification accuracy), process (experiment speed), and strategy (research direction) all improve together as a unified self-learning system.
- **The energy function is ground truth.** It cannot be gamed. This is the invariant across all three phases.
- **No-doomed-rerun discipline:** see "Failed-Experiment Rerun Discipline" above.

## Phase Prototype + Empirical Validation + Adversarial Check Discipline (MANDATORY)

**Origin:** 2026-04-30 Phase-3 architecture blind-spot audit caught
5 FATAL findings that three rigorous theoretical Deep Think rounds
missed. Lesson: *unless we have adversarial checks at each phase
boundary, we are building a house of cards that cannot function in
the end.*

Every Carnot phase (1a/1b/1c/1d... 2a/2b/2c... 3a/3b/3c...) must
satisfy three requirements before ANY scaling decision is committed:

1. **Software prototype** — concrete code artifact in the repo, not
   just an architecture document. The prototype must be runnable
   end-to-end at small scale (e.g., 6,500-pair FoVer corpus for
   Phase-3 substrate).

2. **Empirical validation criteria** — a documented list of
   measurable pass/fail tests with explicit thresholds. Examples:
   `inf_t α_t > 0.1` over 100 MLD steps; decoder
   `joint-constraint pass rate > 85%`; verifier joint null-space
   `dim < 5%` of input space; FPGA sampler `KL(P_fpga || P_gibbs) < ε`.

3. **Adversarial check** — a hostile-reviewer round explicitly
   commissioned to find ways the prototype could pass acceptance
   gates without actually working. Required BEFORE scaling, not
   after. Examples of the attack patterns to demand:
   - Could the encoder be learning a degenerate identity?
   - Could the decoder be ignoring the bottleneck and using language-
     model prior alone?
   - Could the EBM be converging to a single low-energy point?
   - Could the verifier suite share a pathological joint null space?
   - Could the hardware sampler be sampling from a different
     distribution than the model intends?

**Empirical instrumentation IS adversarial check at scale.** A
prototype that emits the right diagnostics surfaces architecture-
level flaws automatically. A prototype that doesn't will let
flaws ship. Therefore every phase prototype MUST include the
diagnostic instrumentation for EVERY theoretical concern the phase
rests on (α_t tracking, joint null-space estimation, KL divergence,
decoded-text diversity, etc.).

**Cross-phase verification.** Every phase artifact must produce
empirical pass/fail data visible to downstream phases. A Phase-3
prototype that depends on Phase-1c's k=15 AND-composition must
VERIFY at integration time that Phase-1c's empirical claims hold
on the deployed verifier suite — not trust them.

**Planner instructions:**

- When proposing tasks for any phase, include the prototype +
  empirical-criteria + adversarial-check trio. A task that proposes
  scaling without one of these three is rejected.
- The .85+ planner has 5 candidate tasks already filed under this
  discipline (see `ops/known-issues.md`):
  1. Phase 1a Adversarial Verifier Robustness Audit
  2. Phase 1c Verifier Joint Null-Space Measurement
  3. Phase 2a Sampler Correctness Audit
  4. Phase 3a Pre-Prototype Adversarial Round
  5. Diagnostic instrumentation library
- Architecture-level Deep Think rounds remain valuable but cannot
  substitute for empirical instrumentation. Treat any architecture
  decision as provisional until the phase prototype confirms
  empirically.

**Cross-references:**
- Full framework: `docs/research-notes/phase-prototype-and-validation-framework.md`
- Audit precedent: `docs/research-notes/phase3-architecture-blindspot-audit-results.md`

## Overdue-Priority Forcing Function (MANDATORY)

If a `ops/known-issues.md` "MANDATORY-NEXT-MILESTONE PRIORITIES" entry has been
pending for 3+ consecutive milestones without pickup, the next planner Sonnet
**MUST** include at least one of those entries as an experiment in its
roadmap, taking precedence over fresh research-breadth exploration.

The 2026-04-27 → 2026-04-28 sessions demonstrated the recurring failure mode:
the planner Sonnet has a strong attention bias toward research breadth and
will repeatedly skip operator-attention-reduction infrastructure work
(`conductor-supervisor.md`, `roadmap-schema-validation.md`,
`eval-metrics-canonical-and-self-heal-production-bug-detector.md` etc.)
even when those are explicitly marked as `NEXT-MILESTONE PRIORITIES` in
`ops/known-issues.md`. Three milestones in a row (.77, .78, .79) skipped
the supervisor proposal despite it being the load-bearing fix for repeated
log-handle-severance + commit-truncation incidents.

**Mechanic:** the conductor's `_plan_next_milestone()` planner-prompt MUST
include the section labelled `MANDATORY-NEXT-MILESTONE PRIORITIES` from
`ops/known-issues.md` *prefixed* with the count of milestones each priority
has been pending. Any priority with `pending_count >= 3` is a hard pickup
requirement; the planner cannot skip it without producing an explicit
written rationale in `research-roadmap-next.yaml` (which the activation
guard then checks for plausibility before activating the milestone).

**Reserved infrastructure slots:** every milestone with ≥10 tasks reserves
at least 2 slots for infrastructure-class work (supervisor, schema
validation, metric canonicalisation, audit scripts, etc.). The reservation
is enforced at planner-output time by the same activation guard.

**Why this is in CLAUDE.md, not just in known-issues.md:** the planner
reads CLAUDE.md as required context; a rule that lives only in
known-issues.md is advisory and routinely ignored. Mandatory rules need
this file's authority.

## Development Workflow (MANDATORY)

This project uses **spec-anchored development** (BMAD + OpenSpec). Every code change follows:

1. **Spec First** — Update `openspec/capabilities/*/spec.md` with new REQ-* and SCENARIO-*. Create/update story in `epics/stories/`.
2. **Write Tests** — Tests reference REQ-* and SCENARIO-* in comments.
3. **Implement** — Code to satisfy spec requirements.
4. **Verify** — Run unit tests, type checks, builds per commands below.
5. **E2E Verify (MANDATORY)** — Run end-to-end tests per `ops/e2e-test-plan.md`. All changes derived from user instruction MUST be verified E2E before reporting done. See E2E Testing below.
6. **Reconcile Specs** — Update Implementation Status in spec.md. Update story status. Update `_bmad/traceability.md` impl status column. If implementation diverged from spec, update spec to match reality with rationale.
7. **Update Ops** — Update `ops/status.md` (what's working/next) and `ops/changelog.md` (what you did).
8. **Update `_bmad`** — Update any part of `_bmad` that is relevant to the changes you made. Never leave specs and code disagreeing silently.

### Architecture Freshness Check

If `_bmad/architecture.md` "Last Reconciled" date is >30 days old, flag to user before starting new capability work.

### Documentation Update Rules (MANDATORY)

When updating `ops/status.md`, `_bmad/traceability.md`, or any ops/spec document:

1. **NEVER remove existing content without explicit user approval.** Completed work, historical results, and infrastructure descriptions must be preserved.
2. **ADD new sections** for new work. Do not replace existing sections with summaries that lose detail.
3. **Move items to "Completed" sections** rather than deleting them. If something was "What's Next" and is now done, move it to "What's Working" — don't delete it.
4. **Preserve historical results** (autoresearch runs, benchmark numbers, experiment data). These are the project's research record.
5. **Items in "Known Constraints" or "What's Next"** stay until explicitly resolved. If a constraint is fixed, mark it with ~~strikethrough~~ and add the fix date — don't delete the line.
6. **When rewriting a document**, first read the ENTIRE existing content and ensure every substantive item appears in the new version. If in doubt, keep it.

## E2E Testing (MANDATORY)

**Every change derived from user instruction must be verified end-to-end.** This means:

- **EBM models**: Full training + sampling pipeline producing statistically correct distributions
- **Cross-language**: Rust and Python implementations producing equivalent results for same inputs
- **Serialization**: Model saved in one language loads correctly in the other

E2E tests must exercise the full stack, not just unit tests. The test plan lives at `ops/e2e-test-plan.md` and results are documented at `ops/test-results.md`.

### Tests Must Run and Assert (MANDATORY)

Every test must have at least one assertion. Skipping tests (`pytest.mark.skip`, `pytest.mark.skipif`, `@unittest.skip`, or equivalent) is never allowed — skipped tests are invisible failures that accumulate silently and erode confidence in the suite. If a test depends on Docker/GPU/network, mock the dependency and test the logic. If a test genuinely cannot run in any environment, do not write it.

## Build / Test / Deploy

```bash
# Build (Rust)
cargo build --workspace --exclude carnot-python
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build -p carnot-python

# Test (Rust unit)
cargo test --workspace --exclude carnot-python

# Test (Python unit with 100% coverage)
pytest tests/python --cov=python/carnot --cov-report=term-missing --cov-fail-under=100

# Test (spec coverage — every test must trace to REQ-*/SCENARIO-*)
python scripts/check_spec_coverage.py

# Lint/Type-check (Rust)
cargo fmt --all -- --check
cargo clippy --workspace --exclude carnot-python -- -D warnings

# Lint/Type-check (Python)
ruff check python/ tests/
ruff format --check python/ tests/
mypy python/carnot

# Pre-commit (all of the above)
pre-commit run --all-files

# Test (Rust with coverage via tarpaulin)
cargo tarpaulin --workspace --exclude carnot-python --out Html --fail-under 100
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Core compute (Rust) | Rust stable, ndarray, rayon |
| Core compute (Python) | Python 3.11+, JAX, Flax, Optax |
| Python-Rust bridge | PyO3 0.24+, maturin |
| Serialization | safetensors (both languages) |
| Rust testing | cargo test, cargo-tarpaulin |
| Python testing | pytest, pytest-cov |
| Rust linting | rustfmt, clippy |
| Python linting | ruff, mypy (strict) |
| Pre-commit | .pre-commit-config.yaml |

## Hardware Acceleration Portfolio

Carnot's hardware-acceleration paths, ordered by current investment
priority. Updated 2026-04-30 after FPGA re-scope + user clarification
that GPU + NPU paths continue.

### Active acceleration paths (continue investing)

1. **2x NVIDIA RTX 3090 (CUDA) — PRIMARY for training**:
   Discrete dual-GPU rig. Use onnxruntime-gpu (CUDA EP), PyTorch
   CUDA build. 48 GB discrete VRAM. Headline performance + Phase-3
   prototype training target. When forced to choose ONE backend,
   pick CUDA: more VRAM, mature tooling, every paper/tool ships
   CUDA first.

2. **AMD Strix Point gfx1150 APU (ROCm) — SECONDARY for portability**:
   Integrated GPU on the dev laptop. PyTorch 2.11.0+rocm7.2 with
   native gfx1150 support, 67 GB unified memory (shared with CPU).
   Requires `sg render -c '...'` for GPU group access and
   `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` for flash attention.
   Use `.cuda()` on model and inputs.

3. **NPU (consumer edge devices) — SOVEREIGNTY ANCHOR**:
   2026-era consumer hardware ships with NPUs: Intel AI Boost, AMD
   Ryzen AI / XDNA, Apple Silicon Neural Engine, Qualcomm Hexagon
   (Snapdragon X), etc. ONNX Runtime supports them via execution
   providers (DirectML on Windows, OpenVINO on Intel, CoreML on
   Apple, Qualcomm QNN, Ryzen AI EP). **Strategic value:** Carnot's
   verifier (Phase 1) and inference-time deep EBM (Phase 3
   deployment) can run on consumer hardware without a $700 discrete
   GPU. This is the load-bearing technical foundation for the
   sovereignty / decentralization claim.

4. **WebGPU gateway (`carnot-webgpu-gateway`)** — for Carnot's OWN
   energy computations only (Ising batch eval, SAT constraints,
   repair). Distributes WGSL compute shaders to browser GPUs over
   WebSocket. NOT a path for running transformers or training.

### Future production hardware (research-class, monitor for availability)

5. **Extropic Z1** (ASIC for thermodynamic computing): planned
   production hardware target. Awaiting public Z1 specs + SDK
   availability.

6. **Photonic** (research-grade chips, long-horizon): no near-term
   action.

### Re-scoped (proof-of-concept tier only)

7. **KV260 FPGA** (POC tier, not production): demonstrates "energy
   evaluable in dedicated hardware" on simple quadratic-Ising
   constraints (exp1041 / exp1068 / exp1081 with sampler-correctness
   caveats). The deep-EBM-on-FPGA aspiration is FUTURE WORK, not
   load-bearing — see `docs/research-notes/phase3-architecture-blindspot-audit-results.md`
   for the 5 FATAL findings that drove the re-scope.

### How to apply

- **New training experiments:** default to CUDA path (`.cuda()`,
  onnxruntime-gpu). RTX 3090 rig.
- **Verifier deployment / edge inference:** include NPU EP support
  via onnxruntime (DirectML, OpenVINO, CoreML, QNN, Ryzen AI as
  appropriate). Sovereignty claims anchor here.
- **Position paper sovereignty story:** anchor to NPU-class hardware
  ("runs on the laptop you already own"), not GPU-class.
- onnxruntime-rocm and onnxruntime-gpu are mutually exclusive in a
  single venv (same `import onnxruntime` name) — pick CUDA on the
  dev machine; NPU EPs are usually separate runtime distributions.
- Don't propose new FPGA-bitstream-redesign tasks (re-scoped); do
  propose Extropic Z1 vendor / hardware-access tasks if Z1 is
  approaching availability.

## SOTA Local Models (mandatory for new experiments)

New experiments that need an LLM must include at least one of these three
state-of-the-art GGUF-quantized local models in their `MODEL_SPECS`:

1. `unsloth/Qwen3.6-35B-A3B-GGUF` — Qwen 3.6 35B MoE, ~3B active, flagship MoE
2. `unsloth/gemma-4-31B-it-GGUF` — Gemma 4 31B dense, instruction-tuned, flagship dense
3. `unsloth/gemma-4-26B-A4B-it-GGUF` — Gemma 4 26B MoE, ~4B active, middle MoE

Use the llama.cpp loader path (already wired — Exp 450 closed the Gemma 4
tokenizer bugs). Keep Qwen3.5-0.8B / Gemma4-E4B only for cheap CPU smoke-tests
or reproduction runs; they are not acceptable as headline-result models.

## Model Tiers

| Tier | Name | Crate | Python Module |
|------|------|-------|---------------|
| Large | Boltzmann | `carnot-boltzmann` | `carnot.models.boltzmann` |
| Medium | Gibbs | `carnot-gibbs` | `carnot.models.gibbs` |
| Efficient | KAN | `carnot-kan` | `carnot.models.kan` |
| Small | Ising | `carnot-ising` | `carnot.models.ising` |

## Session Metrics (MANDATORY)

Track execution time and token consumption every turn:

1. **Turn start**: Run `date -u +"%Y-%m-%dT%H:%M:%SZ"` at start of each response
2. **Turn end**: Run `date -u +"%Y-%m-%dT%H:%M:%SZ"` right before responding to user
3. **Log both** in `ops/metrics.md` turn log table
4. **Subagent metrics**: Record tokens and duration from agent result metadata
5. **On context compaction or session end**: Run `python3 scripts/session-metrics.py` to extract authoritative token counts and costs from the session JSONL, then update `ops/metrics.md` Session Summary

## User Input Tracking (MANDATORY)

Every user instruction must be captured and traceable to outcomes:

1. **Log user instructions**: At the start of each turn, record a 1-line summary of the user's request in `ops/metrics.md` turn log (Description column)
2. **Cycle time**: The turn log's Start/End columns capture wall-clock time between user input and agent completion — this IS the cycle time. Review it to identify slow turns.
3. **Instruction → outcome mapping**: Each entry in `ops/changelog.md` should be traceable to the user instruction that triggered it. If a change was agent-initiated (refactoring, cleanup), note that explicitly.
4. **Session handoff**: Before session ends, update `ops/status.md` with what's working and what's next. This is the handoff document for the next session — human or AI.

## Build Environment

- Rust: stable toolchain
- Python: 3.11+ (3.14 requires `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`)
- JAX: CPU by default, CUDA 12 via `pip install carnot[cuda]`
- JAX on ROCm: `JAX_PLATFORMS=cpu` to force CPU when ROCm plugin is loaded (thrml crashes on ROCm, see extropic-ai/thrml#41)
- Research experiments: always prefix with `JAX_PLATFORMS=cpu` for reproducibility

## Key Paths

| What | Where |
|------|-------|
| BMAD strategic docs | `_bmad/` |
| Capability specs | `openspec/capabilities/*/spec.md` |
| Capability designs | `openspec/capabilities/*/design.md` |
| Change proposals | `openspec/change-proposals/` |
| Epics & stories | `epics/` |
| Operational status | `ops/status.md` |
| Work log | `ops/changelog.md` |
| Known issues | `ops/known-issues.md` |
| E2E test plan | `ops/e2e-test-plan.md` |
| Test results | `ops/test-results.md` |
| Session metrics | `ops/metrics.md` |
| Spec coverage script | `scripts/check_spec_coverage.py` |
| Research roadmap | `research-roadmap.yaml` |
| Research history | `research-complete.yaml` |
| Research program | `research-program.md` |
| Research studying | `research-studying.md` |
| Research references | `research-references.md` |
| Hardware wishlist | `research-hardware-wishlist.md` |
| Research conductor | `scripts/research_conductor.py` |
| Rust crates | `crates/carnot-*/` |
| Python package | `python/carnot/` |

## Experiment Template (New Experiments)

When writing a new experiment script, use `scripts/experiment_template.py` to eliminate
cold-start boilerplate.  This cuts 15-20 min of repetitive setup per experiment.

```python
from scripts.experiment_template import ExperimentTemplate, BatchedInferenceRunner

# 1. Instantiate and setup (creates dirs, loads checkpoint if present)
tmpl = ExperimentTemplate(307, "My experiment title",
                           "results/experiment_307_results.json",
                           requires_gpu=True)
tmpl.setup()

# 2. (If GPU needed) Pre-warm + health-check — Exp 294 pattern.
#    ALWAYS call this before timed inference to avoid lazy-load GPU stalls.
MODEL_SPECS = [{"name": "Qwen3.5-0.8B", "hf_id": "Qwen/Qwen3.5-0.8B", "gpu": 0}]
gpu_status = tmpl.setup_gpu(MODEL_SPECS)
if not gpu_status["all_healthy"]:
    artifact = tmpl.build_result({}, status="blocked",
                                  stall_details=gpu_status["models"])
    # write artifact and exit

# 3. Batch inference (8-16 questions per batch for throughput; timeout=batch_size*60s)
bir = BatchedInferenceRunner(my_inference_fn, batch_size=8)
results = bir.run_batch(questions)   # InferenceResult list in original order
print(bir.batch_log)                 # [{batch_id, batch_size, batch_time_s}, ...]

# 4. Save checkpoint periodically
tmpl.checkpoint_save({"done": [r.response for r in results[:50]]}, step=50)

# 5. Build standardised artifact (auto-populates experiment, run_date, schema, duration_s)
artifact = tmpl.build_result({"responses": [...], "batch_log": bir.batch_log},
                              status="success")
```

Key contract:
- `setup_gpu()` must be called before any timed inference when `requires_gpu=True`.
- `BatchedInferenceRunner.batch_timeout_s = batch_size * 60` (per-batch, not per-question).
- `build_result()` always emits all `REQUIRED_RESULT_FIELDS`; add extras via `**kwargs`.
- Template setup overhead: < 0.5 s (validated by Exp 306 benchmark).

## When to Read Deeper

- **Before starting a new capability**: First review all documents in `_bmad/` and determine if the new capability is already implemented or if there are any relevant change proposals, or if the new capability implies an evolution of the architecture. Read the relevant `openspec/capabilities/*/spec.md` and `design.md`
- **Before deploying or debugging server issues**: Read `ops/known-issues.md`
- **Before architectural decisions or adding new components**: Read `_bmad/architecture.md`
- **To understand project scope or requirements**: Read `_bmad/prd.md`
- **To check what's built vs. spec'd**: Read `_bmad/traceability.md` (has implementation status per FR)
- **Before reporting work as done**: Read `ops/e2e-test-plan.md` and execute relevant E2E tests
