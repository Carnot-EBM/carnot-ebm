# Milestone .114 Activation Manifest

Predecessor milestone: `2026.04.113`
Target milestone: `2026.04.114`
Run date: `20260507`
.113 criteria: `12` of `12` met

## Allowed .114 Tracks

| track | guardrail |
|---|---|
| Adversarial Balanced Telemetry | Use live local SOTA telemetry only with balanced labels and superficial baselines. |
| BEAVER-lite Calibration | Expand only the sound bounded-prefix calibration lane; any violation blocks claims. |
| HalluGuard-style Risk-Bound Fit | Fit risk-bound language only for implemented assumptions and label missing assumptions. |
| FR-11 Query-Time Self-Learning | Promote only opt-in query-time utility with zero soundness mistakes. |
| CCTU-style Executable Constraints | Use deterministic local validators; no closed model dependency or broad benchmark claim. |
| V_1 Pairwise Verification | Compare pairwise self-verification against Carnot energy on bounded candidate sets. |
| THRML Preflight/Parity | Run install/import preflight and simulator parity only; no TSU hardware claim. |
| Partial-Trace Localization | Audit injected-failure localization without claiming decoded quality or Kona internals. |

## Forbidden Reopen Tracks

| track | source | rule |
|---|---|---|
| Telemetry Headline Claims | Exp 1473 adversarial validity audit | Do not make a headline telemetry claim unless a future adversarial audit beats superficial baselines. |
| Repair-Executor Reruns | Exp 1464 and .113 carry-forward guardrails | Do not rerun repair-executor or validation-error-context work without a new root cause and falsifiable gate. |
| GRPO/VPRM | Exp 1456 retirement | Do not reopen GRPO/VPRM variants unless an operator reopens the line with changed evidence. |
| WOPR Puzzle Cartridges | Exp 1457 retirement | Do not add puzzle cartridges or gallery work unless an operator reopens the thesis link. |
| HardNet++/DSP | Exp 1458 retirement | Do not reopen HardNet++/DSP or FSNet-as-DSP work without non-replay evidence. |
| Broad VNN-COMP Runners | Exp 1465 and Exp 1470 BEAVER-lite narrowing | Do not build broad VNNLIB/VNN-COMP runners before bounded BEAVER calibration earns expansion. |
| KV260 Board Claims | Exp 1476 source-level RTL regression | Do not claim board, bitfile, or latency evidence without live board execution evidence. |
| THRML/TSU Hardware Claims | Exp 1477 THRML unavailable simulator probe | Do not claim THRML, TSU, XTR-0, Z1, or Extropic hardware execution from simulator preflight. |

## Hardware Claim Boundaries

- dual_rtx_3090_runtime: allowed_evidence=local_sota_gguf_runtime, live_logprob_telemetry; hardware_claim_allowed=True; Runtime evidence only for local open GGUF inference; no accelerator-substrate claim.
- kv260: allowed_evidence=rtl_source, rtl_simulation; hardware_claim_allowed=False; Source-level RTL lint/simulation only; no board, bitfile, or latency claim.
- thrml_tsu: allowed_evidence=install_import_preflight, simulator_parity; hardware_claim_allowed=False; THRML software preflight and simulator parity only; no TSU or Extropic hardware execution claim.

## Carry-Forward Guardrails

- telemetry_headline_block_preserved: True
- self_learning_followup_allowed: True
- live telemetry remains non-headline until adversarial baselines are beaten.
- FR-11 follow-up must prove query-time utility without soundness mistakes.
- hardware evidence remains bounded to dual RTX 3090 runtime, KV260 RTL source/sim, and THRML simulator preflight.

## No-Change Confirmation

- research-roadmap.yaml: unchanged_by_exp1479_activation_workflow
- scripts/research_conductor.py: unchanged_by_exp1479_activation_workflow
