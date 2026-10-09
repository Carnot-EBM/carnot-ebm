# V719 hardware cost boundary — 2026-10-09

Exp8343 measures constructed spline arithmetic on the host CPU. It uses 128
fixed four-feature vectors, alternating binary targets, the frozen V717 cubic
basis and step rule. SciPy basis values, centered loss differences and a
separate NumPy update reference qualify the arithmetic before timing. One
warmup precedes five paired repetitions; arm order alternates. Repetitions
remain clocks of the same workload and add no independent source observations.

Basis evaluation, gradient computation, coefficient touches and wall cost have
separate receipts. Wall cost retains loop and clock overhead. This branch does
not measure cache invalidation, serialization, durable writes, recovery or
dispatch. Those costs require an independently authenticated compatible current
producer at its exact declared deliverable. Exp8333's failed execution receipt
cannot qualify a durable branch; absent Exp8336/8337 primaries are missing
observations. These findings provide no H1/H2 null result.

| Boundary | Evidence and scope |
|---|---|
| Arithmetic CPU cost | Independent constructed workload; circular scientific control |
| Durable and natural CPU cost | Separate authenticated producer obligations |
| KV260 | Quadratic Ising overlay only, k_max<=5; actual execution uses ssh kria |
| Unsupported KV260 work | Spline basis, coefficient updates and database persistence |
| PolarFire | Exp8259 authenticated dispatch and output parity; board-local Linux CPU |

The arithmetic branch cannot supply complete-service speedup, a compatible
service fraction or an Amdahl upper bound. With no compatible authenticated
spline kernel, accelerator benefit is unproved. Any future bound must retain
CPU and host-transfer costs in its denominator. NFR-01 tenfold and Tier1 100x
claims require complete empirical service costs. No model loads, generator
weight changes or external publication occur. The terminal JSON and its
byte-bound validator sidecar carry the actual readiness scores and findings.

Next condition: authenticate a compatible deployed spline kernel and compare
identical inputs and outputs through a bounded SSH transcript, including host
transfer, CPU work and complete service timing. CPU arithmetic readiness remains
separate from this board-execution obligation. The conductor owns ops and
traceability reconciliation.
