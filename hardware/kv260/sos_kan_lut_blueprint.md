# SOS-KAN LUT FPGA Blueprint - KANELE Q8 Datapath

**Experiment:** Exp 1162
**Spec refs:** REQ-KAN-1162, SCENARIO-KAN-1162
**Status:** Specification only. No Vivado synthesis was run.
**Target board:** AMD Kria KV260, XCK26, estimated clock 300 MHz

## Source Model

| Field | Value |
|-------|-------|
| Model source | `_carnot_sos_kan_direct.SOSKANEnergyV3` |
| Inputs | 3 |
| Spline basis functions per input | 8 |
| Knot count | 8 |
| Rank | 4 |
| Hidden dimension | 16 |
| Exp 1148 compressed bytes | 2704 |
| Exp 1148 compressed AUROC | 0.9718 |
| Exp 1148 centroids | 32 |

## Q8 LUT Table Specification

For each input dimension `x_j` and each hat basis function `b_i`, sample
`b_i(x)` at 256 uniformly spaced points over `[-1, 1]`.
Quantize with `q8 = round(clamp(b_i(x), 0, 1) * 255)`, stored as one unsigned
byte per table entry.

| Field | Value |
|-------|-------|
| Table shape | `(3, 8, 256)` = `(n_inputs, k_splines, n_lut_points)` |
| Total storage | 6144 bytes |
| Table SHA-256 | `53dcb9f436a02a0130d079c6ee24cf82bf607480e673f033c1cc7a4256b1c8f4` |
| First basis first 16 Q8 entries | `[255, 248, 241, 234, 227, 220, 213, 206, 199, 192, 185, 178, 171, 164, 157, 150]` |
| Last basis last 16 Q8 entries | `[150, 157, 164, 171, 178, 185, 192, 199, 206, 213, 220, 227, 234, 241, 248, 255]` |

## Lookup And Interpolation Datapath

For each input `x_j`:

```text
idx   = floor((x_j + 1) / 2 * 255)
frac  = ((x_j + 1) / 2 * 255) - idx
y0    = LUT[j][i][idx]
y1    = LUT[j][i][idx + 1]
b_i   = y0 + frac * (y1 - y0)   # linear interpolation, dequantized from Q8
acc_j = sum_i b_i
```

The LUT access is estimated at 1 cycle, linear interpolation at 3 cycles, and
serial accumulation at `8` cycles per input dimension.

## Hardware Complexity Metrics

| Metric | Value |
|--------|-------|
| RM per inference | 24 |
| BOP per inference | 192 |
| NABS per inference | 75 |
| Total cycles | 36 |
| Estimated FPGA latency | 0.120000 microseconds |
| CPU baseline latency | 289.0 ms |
| Estimated speedup | 2408333.33x |

## Implementation Notes

This is a LUT blueprint, not RTL.  The next hardware step is to map the table
ROMs, interpolation arithmetic, and per-dimension accumulators into Verilog or
HLS and then synthesize on a Vivado-capable host.
