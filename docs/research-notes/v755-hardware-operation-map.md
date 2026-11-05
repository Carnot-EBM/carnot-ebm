# V755 hardware operation map — 20261105

This is a source and design audit. It runs no device probe, flash, model load,
training call or benchmark. Static arithmetic counts do not measure elapsed
learning time. All three board boundaries remain visible when a producer is
absent. The terminal artifact records exact source hashes and retained inputs.

The source categorical head has nine inputs, four SiLU hidden units and one
energy output. Three candidate evaluations require 3 × (9 × 4 + 4) = 120 scalar
multiplies before activation arithmetic. The head has 45 float32 parameters,
or 180 binary parameter bytes. These bytes exclude JSON, templates, database
journals, acknowledgements and recovery state. Dense backpropagation costs and
content-dependent symbolic joins require counters. Three-label normalization
can enumerate all candidates exactly. It does not require MCMC. The audit
makes zero sampler calls and claims no sampler-eligible measured time.

| Board | Preserved boundary | Next evidence |
|---|---|---|
| KV260 | Supported quadratic Ising fabric at k_max<=5. Future access is SSH-only. This does not qualify the categorical head. | Define a compatible kernel and transfer layout. Check precision and state parity. Measure complete host, transfer, kernel and durability costs. |
| PolarFire | Authenticated Linux CPU dispatch. Fabric remains unqualified for the categorical learner. | Run categorical CPU parity and whole-operation timing after a compatible dispatch exists. Qualify any fabric path separately. |
| GateMate | Unchanged 0xffffffff JTAG blocker. There is no dated physical change receipt. No probe retry is authorized by this audit. | Obtain a dated cable, port or power change receipt. Then establish IDCODE0x20000001 and an ordered n16 smoke before learner work. |

Dense Gibbs backpropagation has no established spline-local FPGA implementation.
CPU operation counters are the immediate next measurement. GPU batch training
is a feasible future path. Neither path has a measured speedup in this audit.

Exp8820 learning and Exp8822 complete-cost evidence are admitted independently.
Absent learning does not suppress board design rows. Absent timing does not
become a zero cost. Exp8808 component_cost_ready_score=0 remains a rejected
profiler-fraction authority. It supplies no Amdahl fraction or speedup curve.
If an authenticated producer supplies whole_cost_ready_score=1 and valid complete
operation rows, each measured total/100 is a future100x latency requirement.
It is not a speedup prediction. Startup, transfer and durable-write unknowns
remain explicit. hardware100x requires measured compatible work and complete
costs. No measurement here justifies a purchase request.

The entrypoint supports --date 20261105, --private-output PATH and --replay PATH.
Private outputs exercise publication and replay without claiming owned validation.
Normal execution retains unit/child coverage, lint/type/spec receipts, private
E2E-018/021, repository-wide test debt and terminal scanner reports under raw/.
Owned validation failure disqualifies and sets mapping readiness to zero.
A valid mapping can be ready while external learning, timing or physical work
remains blocked. No constructed audit establishes natural-text or hidden-game
population generalization. Ops and traceability reconciliation is conductor-owned.
