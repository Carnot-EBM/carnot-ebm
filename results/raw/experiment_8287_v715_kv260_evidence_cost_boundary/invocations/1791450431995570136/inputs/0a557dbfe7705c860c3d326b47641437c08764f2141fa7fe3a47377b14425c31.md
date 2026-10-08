# V714 KV260 evidence and learning cost boundary

Exp8273 resolves current acquisition, head and admitted-state operands separately
from execution readiness. Missing V714 capture, head, reserved seal or counters
produce blocked rows; they do not remove qualified historical KV260 evidence.
Both generalization and learning benefit scores remain zero.

The preserved board transcript is
`results/experiment_3709_kv260_drive_to_terminal_latency_transcript.json`, SHA-256
`b813db8619cf29c3350fe9e806f68e20f0c8a5c381b6c11cb8dc096f524f6e5e`.
The continuity primary is
`results/experiment_3721_hardware_kv260_terminal_confirm_and_continuity.json`,
SHA-256 `acdfd841c75649279515fab6b91115d07587094b8703cc72a04febae123236d2`.
Access remains SSH only: `ssh kria`. This task schedules no probe, RTL,
synthesis or flash. It runs without neural weights and makes zero LLM calls.

The useful workload required to reopen device timing is a quadratic Ising
operation with k_max<=5 and bounded coupling encoding, an authenticated SSH
fabric transcript, CPU numerical parity, measured dispatch and transfer costs,
and complete request clocks proving its useful share. Gaussian distances and
exponentials, additive tanh bases, token generation, count lookup, sparse causal
counter updates and durable database commits have no implemented numerical map
to that historical fabric. A Gaussian energy head is not a quadratic workload.

A complete current request includes tokenization, all three view requests, head
scoring, durable feedback, host and transfer work, and shutdown. Cost rows retain
their producer invocation. Imported canary rows, new fit/tune/reserved requests,
cold loads and audit-only cached scoring are distinct. Sequential latency and
observed parallel makespan remain separate; overlapping requests are not summed
as elapsed time, and each cold load is counted once. Phase wall shares use unions
of actual clock intervals. Only measured compatible kernels enter f in 1/(1-f).
Absent current clocks yield unavailable, not a measured zero. Exp8258's f=0
bound used historical Exp8242 costs because current captures did not exist;
Exp8273 retains that scope. NFR-01 Rust/Python 10x remains unmet.

Q8.8 and Q16.16 checks use eligible actual coefficients and admitted counters.
When those operands are absent, separate synthetic threshold/overflow fixtures
exercise numerical mechanics. Coefficient storage is quantized; nonlinear head
bases remain float64 CPU. Counter arithmetic includes quantized counts, mixture
weights and probabilities. Overflow, probability error and raw action flips
remain visible, with exact CPU fallback protecting decisions. These checks are
neither natural learning benefit nor FPGA measurements.

FPGA/ASIC co-design literature motivates measuring the complete operation and
cost boundary. Vendor projections remain estimates. NPU execution and Extropic
TSU hardware access require authenticated evidence; none is supplied here.
Private cold replay, negative/rehashed tamper cases, E2E-015/019 and unchanged
terminal auditors qualify publication. Bounded full-suite health diagnostics are
recorded separately and cannot establish a global pass.
