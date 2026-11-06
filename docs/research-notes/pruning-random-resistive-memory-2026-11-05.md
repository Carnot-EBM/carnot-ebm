# Pruning random resistive memory for analog AI: reading note (2026-11-05)

Status: reading note. No experiment was run. Nothing here is a Carnot result.

Source: "Pruning random resistive memory for optimizing analog AI", Nature Communications 17:1190 (2026). Received 2024-03-24, accepted 2025-12-12. URL: https://www.nature.com/articles/s41467-025-67960-6. Read from the PDF text: abstract, introduction and results. The Discussion, the Methods, the supplement, the seeds and the error bars were not read. Code release and replications were not checked.

Access note: WebFetch fails on this URL (HTTP 303 to a cookie redirect). `curl -sL` with a cookie jar and a browser user agent returns the PDF.

## What the method is

- The network weights stay fixed and random. There is no weight tuning.
- Training prunes and reinstates edges. Learned importance scores pick which edges. This builds on the lottery-ticket idea (Ramanujan et al.).
- Each weight is a pair of resistive-memory cells: w = G+ minus G-. A pruned edge is a physical reset of the pair. A reinstated edge is a set.
- The random weights come free from the stochastic electroforming of a 40 nm, 256K resistive-memory chip.

## Reported results

| Item | Result |
|---|---|
| Fashion-MNIST accuracy | +17.3% vs hardware-in-the-loop weight tuning |
| Spoken Digit accuracy | 98.1% vs 90.8% (hardware-in-the-loop). The abstract says +19.9%. |
| DRIVE segmentation | +9.8% PR-AUC. Simulated. |
| ResNet-50, ImageNet-100 | 87.6% Top-1. Simulated. |
| Programming operations | -99.94% and -99.93% vs hardware-in-the-loop |
| Energy vs GPU, abstract | -78.3% (Fashion-MNIST), -67.9% (Spoken Digit), -99.7% (DRIVE). ResNet-50: -99.3%. |
| Energy per image, Fashion-MNIST | about 3.67 uJ hybrid vs 5.76 uJ GPU. About 1.25 uJ when scaled to 5 nm. |
| Energy per sample, Spoken Digit | about 2.02 uJ hybrid (0.68 uJ at 5 nm) vs 2.15 uJ GPU |
| Sparsity effect | 0.5 sparsity cuts forward-pass energy 30.6% (image) and 25.9% (audio) vs weight tuning |

Llama 3 LoRA fine-tuning appears only in the supplement. It was not read.

## Limits and open questions

- The accuracy gains are against a hardware-in-the-loop baseline that device noise limits. Against a software weight-tuning baseline the gap on Spoken Digit is about 0.2 points. The headline gain measures a noisy baseline.
- The "+19.9%" in the abstract and the 98.1% vs 90.8% pair do not match as percentage points (7.3 points). The cause is unresolved: it may be a relative gain or a different baseline. Check the paper before citing either number.
- The 78.3% and 67.9% energy cuts rely on scaling the 40 nm chip to 5 nm to match the GPU node. Unscaled, Fashion-MNIST is about 36% lower, and Spoken Digit is about 6% lower.
- The GPU baseline is one chosen GPU.
- The 99.7% and 99.3% energy claims come from simulation.
- Seeds, error bars and training-time cost of the pruning were not checked.

## What it means for Carnot

- "Energy" here means electrical power in joules. It is not an energy-based-model energy function. The paper makes no claim about verification, calibration or verifier value.
- It does not touch `GAP-ORACLE-DISTINCT`, the calibrated-decision floor or any step in the JEPA consensus plan (`jepa-two-model-consensus-plan-2026-11-05.md`).
- Hardware track, weak tie: analog compute that tolerates device noise and random substrates fits the hardware-portability rule. It is context only.
- Paper-v6 narrowing #7 already retracts analog and photonic substrates as a production target for the Boolean-coupled DAE-DEBM. This paper covers sparse neural networks, not Boolean constraints. It does not reopen that retraction.
- One design analogy: train by selecting structure from a fixed random pool, not by tuning weights. It resembles selection from a fixed pool in the planner diversity audit. No test follows.

## Rule checks

- Circularity / Oracle-Distinctness: no verifier-value claim. The rule does not apply.
- ARC stays suspended. No ARC task follows.
- No weight update to the mandated generator.

## Verdict

Low value for the energy-based verifier goals. Useful as background for the hardware track only. No experiment follows.
