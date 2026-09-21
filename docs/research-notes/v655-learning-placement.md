# V655 retained-state learning placement

## Scope

This experiment measures the full retained-state importance anchor on the host CPU.
It does not run an FPGA, GPU, or TSU. It does not authorize a purchase, install,
board flash, vendor request, or prefix-cache change.

## Evidence rows

| Branch | Evidence | Current result |
|---|---|---|
| Numeric placement | Measured | The terminal artifact contains 72 rows. They cover four coefficient sizes, three seeds, three arithmetic arms, and a durable no-update control for each arm. |
| Service envelope | Conditional | The capture artifacts do not split tokenization from prefill. The online rows omit verifier and durable-write time. No measured end-to-end speedup is reported. |
| KV260 | Historical | The authenticated date is 20260915. The scope is FPGA-fabric sampling only. Future access uses `ssh kria`, with `k_max<=5`. |
| PolarFire | Historical | The authenticated date is 20260915. The scope is hash-matched Linux CPU dispatch, not FPGA sampling. |
| GateMate | Blocked | No operator-authored cable, port, power, board, JTAG, or DirtyJTAG change exists after 20260823. The experiment issues no detect or flash command. |

The exact per-size and per-seed costs are in
`results/experiment_7487_v655_learning_placement.json`.

## Placement decision

Continue host CPU measurement and keep accelerator placement conditional. Defer a
100x service claim until one compatible denominator includes tokenization, prefill,
verifier work, feedback, all updates, serialization, durable writes, and `fsync`.
The Amdahl bound is `S=1/((1-f)+f/r)`. A 100x path needs an unaccelerated fraction
below one percent. Sparse arithmetic targets are not observed service performance.
