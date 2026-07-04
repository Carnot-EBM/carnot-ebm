# Exp 5243 KAN/p-bit speedup-boundary note

Status: boundary plan only. No speedup claim is allowed from this note.

Inputs reviewed:
- Exp 5242: `results/experiment_5242_kan_certificate_abstraction_scale_v479.json` shows a bounded deterministic KAEM/PWA/MILP certificate boundary, not analog execution or hardware readiness.
- V479 analog KAN reference: arXiv:2606.27892 motivates circuit-level error modeling and pruning for future analog KAN mapping.
- V479 p-bit references motivate partitioned sampler telemetry, boundary exchange accounting, and hash/correctness parity before benchmark claims.
- Extropic TSU remains watch-only until authenticated local TSU/XTR-0 hardware evidence exists.

Minimum valid workload before a speedup experiment:
- same canonical KAN/p-bit workload on CPU baseline and candidate hardware.
- same seeds, same partitions, same boundary exchange schedule, and same KAN-to-p-bit energy coupling.
- Board-local graph, partition, packet-stream, executable, and output hashes.
- Correctness parity against CPU reference energy, final state checksum, and accepted exchange count.
- Analog/KAN error-budget accounting when analog KAN approximation is part of the path.
- Data movement, host dispatch, device setup, sampler steps, validation, and end-to-end wall clock measured in one transcript.

Until those conditions exist, KV260, PolarFire, GateMate, and TSU evidence support only reachability/hash continuity.
