# Tenstorrent Wormhole Block-Gibbs Preflight

Spec refs: REQ-SAMPLE-064, SCENARIO-SAMPLE-092.

## Scope

This is a preflight/prototype plan for Tenstorrent Wormhole n150d as a possible
open-toolchain sovereignty platform for Carnot block-Gibbs sampling. It is not a
hardware execution result. The current artifact verdict is:
`complete: wormhole_preflight_blocked_no_access_no_hardware_claim`.

## Availability Verdict

- `wormhole_access_available`: `False`
- `tt_metalium_available`: `False`
- `wormhole_preflight_ready`: `False`
- `blocked_reason`: `TT-Metalium was not detected; Wormhole hardware or cloud access was not detected.`
- `hardware_transcript_path`: `/home/ianblenke/github.com/ianblenke/carnot/logs/experiment_1584_tenstorrent_wormhole_preflight_transcript.txt`

## Acquisition Or Cloud Next Steps

1. Order or allocate a Tenstorrent Wormhole n150d host with suitable PCIe,
   power, cooling, and Linux support.
2. If local n150d access is unavailable, request Tenstorrent Cloud access or a
   Koyeb Tenstorrent Wormhole instance for a chip-family smoke; record the
   instance type and do not relabel it as n150d hardware.
3. Build TT-Metalium from the official open-source tt-metal instructions in an
   isolated user environment and record the release or commit SHA.
4. Capture a transcript for device enumeration, TT-Metalium import/version, and
   a non-destructive smoke such as `tt-smi`.
5. Run the benchmark protocol below only after the smoke transcript exists.

## Benchmark Protocol

- Workload: THRML-compatible even/odd block-Gibbs Ising sampling at n=16 exact, n=128 sampled, and n=256 stress sizes with candidate warm starts.
- Baseline: Vendored THRML 0.1.3 CPU/JAX block-Gibbs transition operator on the same seeds and beta schedule.

## Acceptance Gates

- KL to THRML: n=16 exact-state KL <= 1e-3 and n>=128 energy-histogram KL <= 0.05 with matched burn-in/sweeps.
- samples/sec: Report raw samples/sec and pass prototype only when Wormhole is at least 2x the local THRML CPU baseline at n=128.
- samples/W: Report board-power-normalized samples/W from tt-smi or platform telemetry and pass only when it is at least 2x the CPU baseline samples/J.
- open-toolchain reproducibility: Record tt-metal commit or release, Carnot commit, build flags, kernel source, seed schedule, and transcript paths from a clean checkout.

## Claim Boundary

No Wormhole execution, throughput, samples/W, or kernel result may be claimed
unless `hardware_transcript_path` points to a successful TT-Metalium/Wormhole
smoke transcript. Public TT-Metal reachability only proves that source is
reachable; it is not hardware access.

## Reference Links

- Tenstorrent Wormhole product page: https://future.tenstorrent.com/hardware/wormhole
- TT-Metalium source: https://github.com/tenstorrent/tt-metal
- Tenstorrent Cloud: https://tenstorrent.com/en/hardware/cloud
- Koyeb Tenstorrent n300s docs: https://www.koyeb.com/docs/hardware/tenstorrent-n300
