# Research Roadmap vNEXT: Milestone 2026.05.269

**Title:** SOTA Runtime Gate + Multi-Corpus Evidence + LoopUS Self-Learning

**Planned:** 2026-05-22

**Previous milestone:** 2026.05.268

**Execution queue:** `research-roadmap-next.yaml`

## What 2026.05.268 Proved

Milestone `.268` did not prove the multi-corpus hypothesis. It proved a more operationally useful fact:
the conductor can now produce honest blocked artifacts rather than fabricated headline numbers when
SOTA runtime preconditions are missing.

Observed `.268` outcome:

- `exp2827` archived `.267` and activated `.268`.
- `exp2828` FoVer leakage, `exp2829` MBPP, `exp2830` HumanEval, and `exp2831` TruthfulQA did not
  produce live AUROC measurements. They blocked on runtime/model preconditions.
- `exp2832` correctly produced an empty matrix instead of inferring missing per-verifier rows.
- `exp2833` correctly refused to make the multi-corpus paper table cite-ready.
- `exp2834` capstone identified the load-bearing root cause: system `python3` had no `torch`, the
  `.venv/bin/python` interpreter had CUDA-capable torch, and the mandated SOTA GGUFs were not cached.

The current cite-safe result remains the FoVer-only headline carried forward from prior milestones:
production FoVer AUROC 0.9857 on the validated N=1000 / 5-seed setting. The FR-11 memory-leakage
delta and non-FoVer generalization claims remain unconfirmed.

## Three Biggest Gaps

### Gap 1: SOTA Runtime Is Not a First-Class Gate

The PRD vision requires reliable local verification with contemporary open models, but `.268` spent
multiple agent attempts on tasks that were impossible under the selected interpreter/model cache.
The next milestone must first establish:

- `.venv/bin/python` is the canonical runtime for CUDA torch.
- At least one mandated SOTA GGUF is cached and loadable:
  - `unsloth/Qwen3.6-35B-A3B-GGUF`
  - `unsloth/gemma-4-31B-it-GGUF`
  - `unsloth/gemma-4-26B-A4B-it-GGUF`
- `scripts/experiment_template.py::cached_sota_pair()` resolves to usable local files.
- Downstream expensive experiments are skipped by structured `gated_on` fields if the preflight fails.

### Gap 2: Cross-Corpus Evidence Is Still Missing

Carnot's PRD vision is not a FoVer-only system. The current strongest result is excellent but narrow.
The next milestone must measure architecture-only versus production self-learning conditions on:

- FoVer: isolate whether persistent FR-11 state is responsible for the headline.
- MBPP: code generation correctness.
- HumanEval: full code benchmark transfer.
- TruthfulQA: factuality transfer with an honest local scorer, not a closed-weight judge.
- HaluEval / FEVER pilot: next-tier factuality readiness, explicitly non-headline unless clean.

### Gap 3: FR-11 Repair and Continuous Self-Learning Are Not Yet Quantified

The PRD centers autonomous directed self-learning, but repair deltas have repeatedly collapsed to zero
or been blocked. The next milestone must include a smaller, externally measurable self-learning loop:
candidate generation, energy scoring, targeted feedback, bounded recurrence, and early exit. It should
not require model-weight edits, and it must report whether each loop lowers constraint energy and improves
final correctness.

## New Research Integrated

The 2026-05-22 sweep added or promoted the following actionable findings in `research-references.md`:

- **Distributional EBMs for Structured LLM Reasoning** (arXiv:2605.18871): use decomposed deterministic
  penalties plus uncertainty analysis as the framing for the cross-corpus verifier matrix.
- **BEAVER v2** (arXiv:2512.05439v2): deterministic frontier bounds for prefix-closed constraints; this
  milestone includes a bounded-prefix feasibility probe but does not claim BEAVER soundness.
- **LoopUS** (arXiv:2605.11011): latent recurrence and adaptive early exit motivate an external-loop
  FR-11 self-learning pilot.
- **Causal Energy Minimization** (arXiv:2605.07588): theory support for energy-descent transformer
  interpretations; citation-level only for this milestone.
- **Extropic TSU / THRML**: confirms the hardware path for EBM sampling, but no attached TSU hardware
  exists, and all local FPGA boards are terminal. Hardware work stays simulator/interface-only.

## Architecture Snapshot

```text
                         +-------------------------------+
                         |  Mandated local SOTA GGUFs     |
                         |  Qwen3.6 / Gemma4 dense/MoE    |
                         +---------------+---------------+
                                         |
                                         v
                         +-------------------------------+
                         | exp2836 SOTA runtime preflight |
                         | .venv CUDA torch + GGUF cache  |
                         +---------------+---------------+
                                         |
                         gated_on: sota_runtime_ready == true
                                         |
       +---------------------------------+---------------------------------+
       |                                 |                                 |
       v                                 v                                 v
+--------------+                 +--------------+                 +--------------+
| FoVer dual   |                 | Code corpora |                 | Factuality   |
| conditions   |                 | MBPP/HumanEval                 | TruthfulQA   |
+------+-------+                 +------+-------+                 +------+-------+
       |                                |                                |
       +----------------+---------------+---------------+----------------+
                        |                               |
                        v                               v
              +------------------+             +------------------+
              | Cross-corpus     |             | BEAVER/EPR       |
              | verifier matrix  |             | bounded probes   |
              +--------+---------+             +---------+--------+
                       |                                 |
                       +----------------+----------------+
                                        |
                                        v
                         +-------------------------------+
                         | LoopUS-style FR-11 external   |
                         | recurrence + early exit       |
                         +---------------+---------------+
                                         |
                                         v
                         +-------------------------------+
                         | Paper table + capstone        |
                         | honest claim boundary         |
                         +-------------------------------+
```

## Phase Structure

### Phase A: Archive and Runtime Gate

- `exp2835` archives `.268` and activates `.269`.
- `exp2836` establishes the SOTA runtime contract and cache manifest.

This phase is intentionally first. All expensive live-model tasks are structurally gated on `exp2836`.

### Phase B: Multi-Corpus Dual-Condition Measurements

- `exp2837` FoVer memory-leakage isolation v3.
- `exp2838` MBPP dual-condition v3.
- `exp2839` HumanEval full dual-condition v3.
- `exp2840` TruthfulQA honest dual-condition v4.
- `exp2841` HaluEval / FEVER 50-example pilot.

The first four tasks are headline candidates only if they pass sample-size, runtime, and methodology
gates. The HaluEval/FEVER pilot is deliberately scoped as readiness evidence.

### Phase C: New Verifier and Self-Learning Probes

- `exp2842` builds the verifier x corpus x condition matrix from real upstream rows.
- `exp2843` implements a BEAVER/EPR bounded-prefix feasibility probe without overstating soundness.
- `exp2844` runs a LoopUS-style external recurrence pilot for FR-11 continuous self-learning.

### Phase D: Synthesis and Claim Boundary

- `exp2845` prepares the paper-v6 Section 5 table and self-learning disclosure.
- `exp2846` capstone reconciles artifacts, gaps, and the next action list.

## Dependency Graph

```text
exp2835
  -> exp2836
       -> exp2837
       -> exp2838
       -> exp2839
       -> exp2840
       -> exp2841
       -> exp2843
       -> exp2844

exp2837 + exp2838 + exp2839 + exp2840
  -> exp2842
       -> exp2845

all artifacts, including blocked states
  -> exp2846
```

Structured gates:

- `exp2837`-`exp2841`, `exp2843`, and `exp2844` gate on `exp2836.sota_runtime_ready == true`.
- `exp2842` gates on non-null AUROC fields from `exp2837`-`exp2840`.
- `exp2845` gates on `exp2842.cross_corpus_matrix_built == true`.
- `exp2846` is intentionally ungated so it can write an honest capstone even if upstream tasks are
  blocked by environment or cache constraints.

## Hardware Requirements

Required:

- Dual RTX 3090 CUDA host, accessed through `.venv/bin/python`.
- Enough local storage for at least one mandated SOTA GGUF. Prefer Qwen3.6; accept Gemma 4 dense or
  Gemma 4 MoE if Qwen is unavailable.
- Existing Carnot datasets and FR-11 state files.

Not required:

- KV260, GateMate, PolarFire. All three local FPGA boards are terminal and do not impose mandatory
  continuity tasks in `.269`.
- Extropic TSU hardware. THRML/TSU work remains a future simulator-track item, not a blocker for the
  SOTA runtime and corpus evidence milestone.

## Agent Routing

- `codex/gpt-5.5`: formulaic preflight, dataset/evaluation wiring, verifier probes.
- `gemini/gemini-3.1-pro-preview`: long-context matrix/literature synthesis when it does not need
  Claude-specific escalation.
- `claude/opus`: capstone-level synthesis only.

## Acceptance Criteria

1. `exp2836` writes a preflight artifact before any expensive work and records the selected Python
   interpreter, CUDA torch status, SOTA GGUF cache state, and `cached_sota_pair()` result.
2. Expensive live-model tasks are preemptively skipped if `sota_runtime_ready` is false.
3. Every LLM experiment lists at least one mandated SOTA GGUF in `MODEL_SPECS`.
4. Legacy small models appear only as CPU smoke-test fallbacks and never as headline models.
5. FoVer, MBPP, HumanEval, and TruthfulQA tasks either produce dual-condition AUROC rows or honest
   `blocked_*` verdicts with precondition evidence.
6. `prior_failures` are present on every scope-matched retry and include `retire_if_same_verdict: true`.
7. The cross-corpus matrix refuses to infer missing rows.
8. The BEAVER/EPR probe clearly labels whether it is exact BEAVER, an EPR probe, or a bounded proxy.
9. At least one continuous self-learning task runs or blocks honestly; `exp2844` is the primary FR-11
   continuous self-learning experiment.
10. The paper table remains `arxiv_ready=false` unless non-FoVer rows are clean and sample-size gates pass.
11. `ops/status.md`, `ops/changelog.md`, and `ops/metrics.md` are updated by the planning session.
12. No task modifies `scripts/research_conductor.py`; no task pushes.

## CLAUDE.md Compliance Notes

- SOTA GGUF mandate is enforced through `exp2836` and repeated `MODEL_SPECS` instructions.
- Failed-experiment rerun discipline is explicit for `.268` blocked corpus tasks and `.267` fabricated
  TruthfulQA.
- Verifier authenticity discipline is enforced by requiring blocked artifacts rather than inferred metrics.
- Operator-only publication discipline applies to `exp2845`: table preparation only, no submission.
- Hardware continuity has no mandatory local-board tasks because all known boards are terminal.
