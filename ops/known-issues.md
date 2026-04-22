# Carnot — Known Issues

**Last Updated:** 2026-04-18

| # | Issue | Severity | Workaround |
|---|-------|----------|------------|
| 1 | PyO3 0.24 doesn't support Python 3.14 natively | Low | Set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` |
| ~~2~~ | ~~Gibbs/Boltzmann grad_energy uses numerical finite differences~~ | ~~Resolved~~ | Analytical backprop implemented |
| ~~3~~ | ~~Python test suite not yet written~~ | ~~Resolved~~ | 48 tests, 100% coverage |
| 4 | Ackley and GaussianMixture benchmarks use numerical gradients | Low | Analytical gradients are complex; numerical is acceptable for benchmarks |
| 5 | RETRO-028: llama.cpp#21516 tokenizer bug causes Gemma4-E4B-it to emit only `<unused8>` tokens (token_id=14), producing 0.0% accuracy (false negative in Exp 439) | High — Fix implemented | Use GemmaTransformersLoader (python/carnot/pipeline/gemma_loader.py) instead of any llama.cpp backend. GPU verification pending (Exp 450). |
| ~~7~~ | ~~RETRO-029: Exp 444 (CarnotThinkProbe) timed out at 20 min with zero results — no partial data, no checkpoint.~~ | ~~Resolved 2026-04-18~~ | ThinkProbeV2 (python/carnot/pipeline/think_probe_v2.py) — 60-min budget, partial_verdict mode, incremental checkpoint every 10 questions. Exp 455 implements and validates the fix. |
| ~~6~~ | ~~RETRO-030: Exp 446 (Energy Matching) exited with status 0 but produced no result JSON. Root cause: exception mid-write left no file; watchdog missed it (only checked exit code).~~ | ~~Resolved 2026-04-18~~ | AtomicResultWriter (python/carnot/pipeline/atomic_writer.py) — write-to-tmp + os.rename prevents partial writes. Exp 452 re-runs Exp 446 logic with atomic write + verify_exists() assertion. |

## RETRO-072 update (Exp 701, 20260422)
Vivado not installed; yosys not found.  Synthesis blocked.  Install one of:
  - AMD Vivado 2024.2 (free WebPACK from xilinx.com)
  - yosys (`sudo pacman -S yosys` on CachyOS)
RETRO-073 opened for milestone .54.
