# V665 native total-cost measurement

Experiment 7613 measured arithmetic-elimination bounds. Those bounds were about
1.006 to 1.098 times by stratum. They were not measured consumer speedups.

Experiment 7627 tests the complete durable consumer boundary. It compares the
registered in-process Python caller, the existing Rust JSON-lines caller, and
the direct PyO3 caller. Each arm receives the same event workload. Each arm must
persist feedback and agree after reload.

The design freezes four strata: cold or warm execution at batch size one or
eight. Each stratum has 30 independent paired blocks. Seed 7627 generates the
workloads and randomized arm order. Cold totals include process or import and
initialization cost. Warm totals include calls, updates, persistence,
acknowledgment, and reload verification. The report does not pool fast-only or
memory-only measurements.

The primary comparator is in-process Python. The old Rust path is a secondary
comparison and cannot replace Python after timing. The reducer uses 2,000 paired
bootstrap resamples in each stratum. It also computes an equal-stratum geometric
mean. Native benefit requires an estimate and lower bound of at least 1.10. Each
stratum lower bound must be at least 0.95. All decisions and durable state must
match, with no extra native errors. The separate 10-times NFR requires a measured
total-cost ratio of at least 10. Arithmetic bounds cannot satisfy that target.

Forty paired telemetry controls measure instrumentation overhead. They do not
select a comparator or change the primary gate.

The task reuses Experiment 7626 E2E-003 and E2E-004 evidence. It authenticates
the exact private extension and existing JSON-lines binary before timing. It
does not rebuild a missing native producer.

Hardware scope does not change. KV260 remains graduated FPGA-fabric evidence
with `k_max<=5`. PolarFire remains graduated Linux CPU dispatch, not fabric
sampling. GateMate remains blocked on a new operator physical-chain receipt
after the `0xffffffff` observation. The local RTX 3090 pair is historical lease
evidence only. Extropic TSU and AMD XDNA remain unavailable prospective devices.
This experiment runs no model and issues no hardware operation.

No result from this task changes a board graduation, production default, model
weight, generator weight, prior null, or prior verdict.
