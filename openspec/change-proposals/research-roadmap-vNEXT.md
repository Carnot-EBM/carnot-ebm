# Research Roadmap vNEXT - Milestone 2026.05.305

**Title:** Garak Red-Team Gate Pass + Headline-Eligible Repair Evidence

**Created:** 2026-05-28
**Status:** Proposed, staged in `research-roadmap-next.yaml`
**Supersedes:** Milestone 2026.05.304
**Execution queue:** `exp3294` through `exp3306`

## What Milestone 2026.05.304 Proved

Milestone `.304` moved the publication blocker from ambiguous tooling to a
specific adversarial failure:

- `exp3281` archived `.303` and opened `.304`.
- `exp3282` made Garak runnable: `garak_available=true`,
  `garak_runner_ready=true`, and `garak_version=0.15.0`.
- `exp3283` preserved the prompt-injection corpus corrigendum and kept `.303`
  duration/tautology flags out of headline claims.
- `exp3284` ran a local Garak smoke against
  `unsloth/gemma-4-26B-A4B-it-GGUF` with `attack_success_rate=0.25`.
- `exp3285` ran the full Garak/DataFlip red-team eval. DataFlip passed, but
  Garak failed: `garak_gate_passed=false`,
  `attack_success_rate=0.311111`, and the gate threshold was `0.20`.
- `exp3286` identified the clean-verifier abstain-all root cause.
- `exp3287` fixed the abstention profile:
  `abstention_rate=0.0` and `repair_gate_input_clean_enough=true`.
- `exp3288` retired the KAN sidecar from prompt-injection headline claims:
  `prior_full_corpus_auroc=0.475326`,
  `aligned_instruction_false_positive_rate=1.0`, and
  `kan_boundary_decision=retire_from_prompt_injection_headline`.
- `exp3289` reopened the repair gate: `repair_gate_open=true`.
- `exp3290` ran a 4-case SOTA repair micro-panel with no false accepts, but
  stayed non-headline: `headline_claim_allowed=false`.
- `exp3291` passed the required FR-11 replay with controller memory only:
  `retention_score=0.982143`, `adaptation_score=1.0`,
  `forgetting_rate=0.017857`, and `negative_transfer_rate=0.0`.
- `exp3292` and `exp3293` closed `.304` with
  `paper_ready=false`, `publication_blocker_count=10`, and
  `next_top_gap=pass_garak_redteam_gate`.

The natural next milestone is therefore not another installation, corpus, or KAN
milestone. `.305` must reduce live Garak attack success below the promotion
gate, make repair evidence large enough for a headline claim, and keep every
claim grounded in exact checks rather than LLM judging.

## Three Biggest Gaps To PRD Vision

1. **Prompt-injection robustness is still below the gate.** The PRD requires
   verifiable reasoning under adversarial pressure. `.304` proved the local
   target can run, but the live Garak attack success rate was `0.311111`, above
   the `0.20` threshold. The next gap is a defense/routing layer that lowers
   this rate without hiding failures.

2. **Repair evidence is not statistically headline-eligible.** The repair gate
   reopened and the 4-case panel passed exact checks, but `N=4` cannot support a
   benchmark claim. The next gap is an exact, stratified, live SOTA repair panel
   with enough cases, confidence intervals, no false accepts, and clean
   adversarial-verification metadata.

3. **Evidence hygiene still carries flagged rows.** `.304` correctly bounded
   `.303` duration/tautology issues and KAN failure, but matrix v36 still
   counted flagged and sidecar-only rows. `.305` needs a clean evidence boundary:
   no KAN headline retry, explicit substrate declarations, and aggregation that
   distinguishes clean live Garak/repair evidence from historical corrigenda.

## External Research Integrated

The 2026-05-28 post-`.304` sweep was added to `research-references.md` before
this roadmap was designed. The most relevant findings are:

- **BEAVER** motivates a prefix-closed verifier for rogue strings and jailbreak
  targets. `.305` includes a BEAVER-inspired guard pilot over the exact target
  phrases observed in `.304`.
- **HalluScan adaptive detection routing** motivates family-aware routing rather
  than one global prompt-injection score. `.305` separates PromptInject,
  encoding, jailbreak, and aligned-benign slices.
- **Rethinking LLMs as Verifiers** warns that LLM judges are not safe promotion
  authorities. `.305` repair promotion remains exact-check-first.
- **Spilled Energy, Semantic Energy, and HalluGuard** motivate inference-time
  instability telemetry as a routing signal, with explicit authenticity
  disclosure if logits are unavailable.
- **XGrammar 2, CRANE, and constrained diffusion decoding** support schemas and
  grammars for parseability only. `.305` uses structured repair proposals but
  does not equate schema validity with semantic safety.
- **KAN property-verification work** is relevant to future KAN certification but
  does not rescue a below-random prompt-injection sidecar. `.305` enforces the
  `.304` KAN retirement boundary.
- **2D parallel tempering and reversibility-based samplers** stay on the Phase 2
  hardware/sampling backlog; they do not displace the immediate publication
  gate.
- **EBT, ARM-as-EBM, Extropic, and Kona** continue to support the long-term
  architecture, but `.305` makes no TSU, Kona, or foundation-EBT access claim.

## SOTA Local GGUF Policy

Any `.305` experiment that invokes an LLM for evidence must include at least
one mandated local SOTA GGUF model in `MODEL_SPECS`:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

The preferred implementation pattern is `cached_sota_pair(gpu_indices=(0, 1))`
from `scripts/experiment_template.py`. Legacy small models may appear only as
CPU smoke-test fallbacks. They cannot populate headline result fields and cannot
unblock Garak, repair, or publication-readiness gates.

## Architecture Diagram

```text
                 .304 terminal state
 Garak runnable but failed gate (ASR=0.311111 > 0.20),
 clean verifier calibrated, KAN retired from headline path,
 repair gate open, repair panel N=4 non-headline,
 FR-11 controller memory safe, paper_ready=false
                              |
                              v
             exp3294 archive .304 and activate .305
                              |
        +---------------------+----------------------+
        |                     |                      |
        v                     v                      v
 exp3295 Garak         exp3296 substrate      exp3301 exact repair
 failure autopsy       corrigendum + KAN      panel manifest
        |              no-retry ledger              |
        v                     |                      v
 exp3297 prefix guard         |              exp3302 headline repair
        |                     |              panel [gated on Garak pass]
        v                     |                      |
 exp3298 energy/route         |                      v
 telemetry                    |              exp3303 repair audit
        |                     |
        v                     |
 exp3299 defense ablation <---+
        |
        v
 exp3300 full Garak rerun v3
        |
        +-----> exp3304 FR-11 red-team/repair memory replay

 exp3305 evidence matrix v37 -> exp3306 capstone v305
```

## Phase Plan

### Phase 1 - Handoff And Evidence Boundaries

- `exp3294` closes `.304`, archives completed `.304` evidence if missing, and
  opens `.305`.
- `exp3295` performs the Garak failure-mode autopsy from `.304`: target strings,
  probe families, refusal rate, repetition/degeneration, and exact reasons the
  gate failed.
- `exp3296` creates a substrate/corrigendum boundary ledger, keeps KAN retired
  from prompt-injection headline claims, and defines which `.304` evidence can
  be cited by `.305`.

### Phase 2 - Garak Defense And Gate Rerun

- `exp3297` implements a BEAVER-inspired prefix-closed rogue-string guard pilot
  over `.304` target phrases and response traces.
- `exp3298` captures live red-team energy/instability telemetry on a small SOTA
  GGUF panel and emits an attack-family routing policy.
- `exp3299` runs a live defense ablation over hardened prompt boundaries,
  prefix guard, refusal policy, and routing combinations.
- `exp3300` reruns the full Garak/DataFlip evaluation with the selected defense
  configuration. This is the milestone's primary publication gate.

### Phase 3 - Headline-Eligible Repair Evidence

- `exp3301` builds a stratified exact repair panel manifest with at least 30
  cases and no reliance on LLM judging.
- `exp3302` runs the live SOTA repair panel only if `exp3300.garak_gate_passed`
  and `exp3301.repair_panel_manifest_ready` are both true.
- `exp3303` audits the repair panel for exact-check provenance, false accepts,
  duration/substrate correctness, and confidence intervals.

### Phase 4 - Continuous Self-Learning And Closeout

- `exp3304` is the required continuous self-learning experiment. It replays
  Garak defense, failed attack families, repair successes/failures, and exact
  verifier outcomes through the FR-11 controller-memory loop while preserving
  raw episodes and reporting retention, adaptation, forgetting, and negative
  transfer. It makes no foundation-weight-update claim.
- `exp3305` builds evidence matrix v37.
- `exp3306` produces the `.305` capstone and names the next top gap.

## Dependency Graph

```text
exp3294
  -> exp3295
      -> exp3297 [gate: garak_failure_autopsy_ready == true]
      -> exp3298 [gate: garak_failure_autopsy_ready == true]
          -> exp3299 [gate: redteam_telemetry_policy_ready == true]
      -> exp3299 [gate: prefix_guard_policy_ready == true]
          -> exp3300 [gate: selected_defense_config_ready == true]

exp3294
  -> exp3296
      -> exp3299 [gate: substrate_corrigendum_ready == true]

exp3294
  -> exp3301
      -> exp3302 [gate: repair_panel_manifest_ready == true]

exp3300.garak_gate_passed == true
  + exp3301.repair_panel_manifest_ready == true
      -> exp3302
          -> exp3303 [gate: headline_repair_panel_ready == true]

exp3300.garak_redteam_eval_v3_ready == true
      -> exp3304

exp3305
  -> exp3306 [gate: matrix_v37_ready == true]
```

## Hardware Requirements

- **Dual RTX 3090 local host:** Required for live mandated SOTA GGUF tasks
  (`exp3298`, `exp3299`, `exp3300`, `exp3302`). These tasks must check
  `nvidia-smi`, selected-Python CUDA, llama.cpp/GGUF loadability, model IDs,
  GPU memory, generated tokens, and wall-clock duration.
- **CPU-only path:** Acceptable for archive, failure autopsy, substrate
  corrigendum, KAN retirement boundary, prefix-guard implementation over cached
  traces, repair manifest design, repair audit, FR-11 controller-memory replay,
  evidence matrix, and capstone tasks.
- **Network/package access:** Not required as a blocker. Garak already ran in
  `.304`; `.305` may use existing `uv run --no-project --with garak --with
  openai` command shapes. If package access fails, tasks must write blocked
  artifacts with exact command and stderr summaries.
- **KV260/GateMate/PolarFire/THRML/Extropic/Kona:** Out of scope for `.305`.
  They remain architectural and hardware-backlog signals. `.305` makes no FPGA,
  TSU, Kona, D-Wave, or proprietary hardware-access claim.

## Experiment Queue

| ID | Title | Primary Deliverable | Phase |
| --- | --- | --- | --- |
| `exp3294` | Close .304 ledger and open .305 Garak gate queue | `results/experiment_3294_archive_v304_activate_v305.json` | 1 |
| `exp3295` | Garak failure-mode autopsy v1 | `results/experiment_3295_garak_failure_mode_autopsy_v1.json` | 1 |
| `exp3296` | Evidence substrate corrigendum and KAN no-retry ledger v1 | `results/experiment_3296_substrate_corrigendum_kan_no_retry_v1.json` | 1 |
| `exp3297` | Prefix-closed Garak rogue-string guard pilot v1 | `results/experiment_3297_prefix_closed_garak_guard_v1.json` | 2 |
| `exp3298` | Red-team energy telemetry and routing policy v1 | `results/experiment_3298_redteam_energy_telemetry_router_v1.json` | 2 |
| `exp3299` | Garak defense ablation v1 | `results/experiment_3299_garak_defense_ablation_v1.json` | 2 |
| `exp3300` | Full Garak/DataFlip gate rerun v3 | `results/experiment_3300_full_garak_dataflip_gate_rerun_v3.json` | 2 |
| `exp3301` | Exact repair panel manifest v11 | `results/experiment_3301_exact_repair_panel_manifest_v11.json` | 3 |
| `exp3302` | Headline SOTA repair panel v11 | `results/experiment_3302_headline_sota_repair_panel_v11.json` | 3 |
| `exp3303` | Repair headline evidence audit v1 | `results/experiment_3303_repair_headline_evidence_audit_v1.json` | 3 |
| `exp3304` | FR-11 red-team and repair memory replay v2 | `results/experiment_3304_fr11_redteam_repair_memory_replay_v2.json` | 4 |
| `exp3305` | Evidence matrix v37 | `results/experiment_3305_evidence_matrix_v37.json` | 4 |
| `exp3306` | Capstone v305 | `results/experiment_3306_capstone_v305.json` | 4 |

## Done Criteria

- Garak gate has a clean v3 result: either `garak_gate_passed=true` with
  `attack_success_rate <= 0.20`, `error_count=0`, and live SOTA provenance, or
  a precise failure artifact naming the still-open attack family.
- Defense improvements are tied to target strings, probe families, and exact
  response evidence. No KAN prompt-injection headline retry is attempted.
- Repair evidence is headline-eligible only if it uses a stratified exact panel
  with `N>=30`, no false accepts, bounded abstention, confidence intervals, and
  clean substrate/duration metadata.
- FR-11 replay includes red-team and repair episodes, preserves raw traces,
  reports retention/adaptation/forgetting/negative transfer, and keeps the
  controller-memory-only boundary.
- `research-roadmap.yaml` and `scripts/research_conductor.py` remain untouched
  by planning.
