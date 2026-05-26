# Experiment 3105 cLUT Sampler Mapping Notes

**Date:** 2026-05-26
**Status:** Architecture context only
**Spec refs:** `REQ-SAMPLE-3105`, `SCENARIO-SAMPLE-3105`

## Scope

Experiment 3105 tests a CPU-only compressed lookup-table random-variate
primitive for Bernoulli draws with probabilities `sigmoid(local_field)`. The
microbench is intended to validate software behavior before any hardware work:
distribution error is measured against exact sigmoid probabilities, and timing
is scoped to a CPU scalar exact-logistic baseline.

## Candidate GateMate/FPGA Mapping

- Store the logistic cLUT as fixed-point thresholds in LUTRAM or BRAM.
- Address the table with a clipped, quantized local field.
- Draw a fixed-width random word from the existing sampler RNG stream.
- Emit spin `1` when `random_word < threshold`, else emit spin `0`.
- For the default CPU prototype shape, a 1025-entry Q0.15 table occupies 2050
  bytes before any packing or symmetry compression.
- The datapath is a table read plus unsigned comparator. The local-field
  accumulation remains separate from this random-variate primitive.

## Claim Boundary

No FPGA, GateMate, KV260, Vivado, Yosys, nextpnr, openFPGALoader, board IO, or
hardware timing command is part of this experiment. These notes are not a
synthesis report and do not justify a hardware speedup claim.

Hardware execution would require a separate artifact with authenticated command
transcripts, device metadata, bitstream or simulation provenance, and sample
quality checks from the hardware path itself.
