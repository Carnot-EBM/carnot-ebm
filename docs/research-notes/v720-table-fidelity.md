# Frozen spline table fidelity, 2026-10-09

This constructed numerical study uses the original Exp8334 checkpoint and V717 rule.
Current model loads and generation calls are zero. H1/H2 remain unmeasured.
Global slope, intercept, temperature, interpolation and sigmoid remain float64.
The panel has4096 random vectors,72 knot controls and6 action boundary controls.
One warmup and five paired repetitions do not add independent scientific sources.

Verdict: complete_circular_positive_spline_table_fidelity. Measurement readiness: 1.
Candidate score: 1. Direct probability parity: 1.1102230246251565e-16.

| Configuration | Max probability error | Flips | Boundary flips | Saturation | Bytes | Direct ns/vector | Table ns/vector |
|---|---:|---:|---:|---:|---:|---:|---:|
| 65-float64-nearest | 0.0027465688056 | 0 | 0 | 0 | 2080 | 107913 | 6443 |
| 65-float64-linear | 0.000130238275 | 0 | 0 | 0 | 2080 | 108885 | 10720 |
| 65-int16-nearest | 0.00272604609569 | 3 | 3 | 0 | 520 | 108424 | 6562 |
| 65-int16-linear | 0.00014686768869 | 3 | 3 | 0 | 520 | 108424 | 11632 |
| 257-float64-nearest | 0.000713884571021 | 0 | 0 | 0 | 8224 | 108333 | 6522 |
| 257-float64-linear | 7.57874782809e-06 | 0 | 0 | 0 | 8224 | 108614 | 10730 |
| 257-int16-nearest | 0.000708511925426 | 3 | 3 | 0 | 2056 | 110637 | 6713 |
| 257-int16-linear | 9.8990262426e-05 | 3 | 3 | 0 | 2056 | 112091 | 12113 |
| 1025-float64-nearest | 0.000150113062398 | 0 | 0 | 0 | 32800 | 115877 | 6973 |
| 1025-float64-linear | 4.74030232178e-07 | 0 | 0 | 0 | 32800 | 108824 | 10811 |
| 1025-int16-nearest | 0.000171581090617 | 3 | 3 | 0 | 8200 | 108724 | 6603 |
| 1025-int16-linear | 9.28842654305e-05 | 3 | 3 | 0 | 8200 | 117791 | 12614 |

All per-vector errors and paired timings remain in the byte-bound JSONL evidence.
All six storage/grid trajectories retain64 updates, support entry lists and six timing samples per update.
Full and scoped refresh require identical encoded bytes after every update and fresh-process restart.
A deliberately missed entry must fail. Update, refresh and byte serialization timings are separate.
Candidate gates are maximum probability error<=.001 and no flips outside a .002 action margin.
Near-threshold flips are retained. Candidate choice uses bytes, then error, then frozen order.
This is circular_positive engineering evidence and provides no natural accuracy or utility claim.
The board operation manifest creates no RTL and makes no device speed, resource or energy claim.
[Protocol](v720-table-protocol.md) binds the methods, source versions and fixed adaptation.
