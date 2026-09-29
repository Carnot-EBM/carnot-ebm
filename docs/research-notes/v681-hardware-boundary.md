# V681 hardware boundary — 2026-09-29

Exp7847 reads prior board bytes. It does not run a board. Exp7834 remains
disqualified because its required coverage check failed. Its repository health
run also timed out. The Exp7846 science producer is absent at its exact path.
No old speed figure fills that missing service measurement.

| Board | Last qualified evidence | Current limit | Reopen only after |
| --- | --- | --- | --- |
| KV260 | 2026-09-13; hash-matched fabric receipt and original transcript; quadratic Ising k_max<=5 | ARM host work is separate from fabric. A new fabric build depends on a commercial toolchain. | A dated workload with k<=5, authenticated SSH fabric execution, transfer bytes, setup time, and whole-service timing. |
| PolarFire | 2026-09-13; hash-matched Linux CPU dispatch transcript | The dispatch is CPU work, not FPGA execution. | A distinct FPGA workload with device-side execution and timing transcript. |
| GateMate | 2026-09-15; physical continuity receipt | JTAG IDCODE remains 0xffffffff. | A dated, authenticated cable, port, power, board, or DirtyJTAG change, followed by a valid GM1Ax IDCODE. |

The [pipelined p-computer paper](https://arxiv.org/abs/2607.21077) treats
coupling bandwidth and local updates as material to dense Ising sampling.
The [spline-local learning paper](https://arxiv.org/abs/2602.02056) shows
sparse coefficient updates in its FPGA KAN setting. Each is a useful design
boundary. Neither establishes that the present deterministic decision head is
an Ising sampler or that its measured host traffic fits either device.

[Extropic Z1T](https://extropic.ai/writing/z1t) is vendor evidence. Its
efficiency estimates are not local measurements. There is no authenticated
NPU or TSU access in this record. No purchase, synthesis, flash, or new board
operation follows from board continuity alone.
