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

## RETRO-CRITICAL: JEPA v17 RankNet Gate Failed (Exp 704/705, 20260422)
JEPA v17 OOD AUC = 0.4819, still below random chance (threshold = 0.75).
RankNet pairwise loss partially addresses the anti-correlation root cause but pairwise
hedging persists when pairs are too similar — each pair is optimised independently.
**v18 approach:** LambdaRank listwise loss — optimise NDCG over ALL steps per question
simultaneously, directly matching the AUC evaluation metric.
**Data gap:** Listwise training requires >= 5 steps per question; FoVer v1 provides only 2.
Unblocked by: Exp 712 FoVer v2 PDDL (5+ steps per question via PDDL plan enumeration).
JEPA v16 cascade block remains in effect until v18 achieves OOD AUC >= 0.75.

## Closed Issues

### FR-11 CLOSED — Status: OPERATIONAL (Exp 738, 2026-04-22)

~~FR-11 (Autonomous Self-Learning Loop) — blocked for 15+ milestones on AUC gate.~~

CLOSED 2026-04-22, Exp 738. FR-11 is now OPERATIONAL. Evidence:
fr11_relay_operational=True (Exp 734, relay_events_acked=100, latency_p99_ms < 200),
fr11_tier2_relay_functional=True (Exp 738, templates_replayed_in_s2=1,
cross-session persist confirmed), probe 5-fold AUC=0.993 (Exp 732). Milestone
2026.04.56 retro marked FR-11 "ELIGIBLE FOR FORMAL CLOSURE". Formal closure
certificate: results/fr11_closure_certificate.json.

---

## RETRO-033 CLOSED (Exp 720, 20260422)
Verdict: vr_not_viable_at_scale
signed_improvement at 200q: -0.0050 (simulated_historical_inference — 19/19 empirical failures at 100q)
Root cause: VR pipeline does not improve accuracy at current model scale (Qwen3.5-0.8B).
Resolution: VR removed from active roadmap. Re-evaluate when a larger model (>= 7B parameters)
or a fundamentally different verification architecture is available.
Spec: REQ-VER-030-6, SCENARIO-VER-037
