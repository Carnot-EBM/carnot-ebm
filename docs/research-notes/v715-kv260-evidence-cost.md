# V715 KV260 evidence and learning cost boundary

Exp8287 resolves fit, tune, trained-head, reserved-view and admitted-counter
operands independently. Missing current evidence produces complete blocked
rows. Historical board custody remains readable. Execution readiness grants
no scientific benefit; both generalization scores stay zero.

The exact preserved board transcript is
`results/experiment_3709_kv260_drive_to_terminal_latency_transcript.json`, SHA256
`b813db8619cf29c3350fe9e806f68e20f0c8a5c381b6c11cb8dc096f524f6e5e`.
The continuity primary is
`results/experiment_3721_hardware_kv260_terminal_confirm_and_continuity.json`, SHA256
`acdfd841c75649279515fab6b91115d07587094b8703cc72a04febae123236d2`.
Access remains SSH only: `ssh kria`. This task schedules no probe, new RTL,
synthesis or flash. It loads no neural weights and makes zero current LLM calls.

The supported fabric workload is quadratic Ising with k_max<=5. To reopen
device timing, provide a useful workload within that size, bounded coupling
encoding, authenticated SSH fabric dispatch, CPU numerical parity, measured
transfers and complete request clocks showing its useful cost share.
Gaussian distance and exponential bases, additive tanh heads, token generation,
keyed lookup, sparse causal counter updates and durable database commits each
need their own implemented numerical operation map. A Gaussian energy head
is not an implemented quadratic Ising workload.

A complete current request includes tokenization, three view requests, head
scoring, durable feedback, host/transfer work and shutdown. Cost rows bind
producer invocations. Imported canaries, actual fit/tune/reserved calls, cold
loads and audit-only cached scoring have separate accounting. Each cold start
is charged once. Sequential latency and observed parallel makespan remain
separate. Overlapping requests are never summed into elapsed parallel time.
Eligible measured spans determine f in 1/(1-f). No compatible spans gives f=0
only when clocks exist. Missing current clocks produce an unavailable bound.
Exp8273 used historical Exp8242 spans because current captures did not exist;
its f=0 bound remains historical. NFR-01 Rust/Python 10x stays unmet.

Q8.8 and Q16.16 checks use eligible actual head coefficients and admitted
counter states. When operands are absent, separate synthetic threshold and
overflow fixtures test numerical mechanics. Head coefficient storage is
quantized while nonlinear bases stay float64 CPU. Counter arithmetic tests
quantized counts, weights and probabilities. The rows retain overflow,
probability error, raw action flips and exact CPU fallback. They establish
neither natural learning benefit nor FPGA measurement.

[FPGA-ASIC co-design](https://arxiv.org/abs/2602.15985) motivates measuring
orchestration and memory work with complete service costs. Extropic's
[Z1T design](https://extropic.ai/writing/z1t) motivates a separate sparse and
digital partition. These sources motivate the boundary; their performance
claims are not local board measurements. NPU acceleration and TSU hardware
access remain unqualified. Vendor estimates remain estimates.

Validation commands are frozen before measurement. Private E2E-015/019,
consumer tests, Ruff, strict mypy, spec references and full changed-code
statement coverage qualify the owned implementation. A single bounded full
Python suite run records repository health separately. Its failures cannot
establish a global pass. Fresh-process replay rejects negative and rehashed
primitive or summary mutations. Unchanged terminal auditors check candidate
bytes before atomic publication. The conductor owns ops and traceability updates.
