# Exp 5266 Thermodynamic Sampler Boundary

Status: future requirements only. No speedup claim is allowed.

Sources reviewed:
- Scaling Up Thermodynamic AI Models, arXiv:2607.00170: https://arxiv.org/abs/2607.00170
- Extropic TSU 101: https://extropic.ai/writing/tsu-101-an-entirely-new-type-of-computing-hardware
- Extropic X0/XTR-0: https://extropic.ai/writing/inside-x0-and-xtr-0
- Extropic hardware overview: https://extropic.ai/hardware

Boundary:
- The V481 thermodynamic-model reference treats Gibbs-sampled Ising systems as inference substrates and links inference cost, accuracy, and autocorrelation. For Carnot, that becomes a future requirement to record sampler-cost and autocorrelation receipts before any thermodynamic speedup claim.
- Extropic public pages describe TSUs and XTR-0 as programmable EBM/PGM sampling hardware, but this repo still has no local SDK or device, no authenticated TSU transcript, and no local XTR-0 execution.
- KV260 and PolarFire reachability checks are board-continuity receipts only. GateMate remains a physical/JTAG blocker unless the operator changes the physical setup.

Future requirement:
- A valid acceleration claim must include workload hash, executable or bitstream hash, output hash, wall-clock timing, sample-quality/autocorrelation diagnostics, CPU/GPU baseline parity, and device identity from the same reproducible run.

No speedup claim: Exp 5266 records reachability and boundary requirements only.
