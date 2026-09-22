# V657 Placement and Board Continuity

## Scope

Experiment 7513 measures a finite integer approximation of the frozen Exp7505
small head. It uses training and calibration feature rows only. It does not
open held-out labels. It also preserves the three-board evidence ledger.

The current run uses host CPU arithmetic. It loads no language model. Cached
Qwen evidence is historical. The run issues no SSH, detect, flash, synthesis,
toolchain, driver, or board command.

## Numeric gate

The float32 head is the reference. Signed int16 and signed int8 variants use
quantized basis values, quantized coefficients, and explicit integer
accumulators. Each row records:

- absolute probability error;
- action parity for all nine frozen cost policies;
- overflow count and accumulator bounds;
- coefficient, bias, knot lookup, basis, and normalization storage.

The frozen calibration gate requires zero overflow, at most 0.01 absolute
probability error, and zero action disagreements. The CPU benchmark uses
warmup work and at least 30 independently timed batches. Allocation,
quantization, kernel, dequantization, and policy decision costs stay separate.

These values are host emulation. They are not FPGA performance.

## Whole-service boundary

The operation inventory covers prediction, a sparse Brier update, the numeric
guard, serialization, and durable acknowledgement. Transfer costs remain
unknown. Exp7514 must measure the complete service denominator. Experiment
7513 therefore reports no whole-service speedup.

The conditional binary energy has two states and normalizes exactly. It needs
no TSU sampling task. Current TSU access remains unverified.

## Board ledger

The board score is independent of numeric readiness. KV260 keeps only its
historical FPGA-fabric scope. PolarFire keeps only its historical board-CPU
dispatch scope. GateMate keeps its physical-state blocker unless the approved
local receipt parser finds a newer operator-authored physical change.

Current reachability is `unknown_not_probed` for all three boards. Historical
graduation does not imply current network, USB, or JTAG reachability.

## Interpretation

`numeric_placement_ready_score` reports host numeric equivalence. It does not
graduate hardware. `board_continuity_complete_score` reports whether all three
historical dispositions remain present. A blocked numeric input does not erase
board history, and a completed board ledger does not prove device performance.
