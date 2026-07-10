# Research Roadmap vNEXT - 2026.07.503

Milestone: `2026.07.503`
Previous milestone: `2026.07.502`
Conductor queue: `research-roadmap-next.yaml`
Task range: `exp5550` - `exp5563`

## Executive Summary

Milestone `.502` closed the immediate planning debt from the `.501` capstone: it made the SOTA panel duration correction honest, proved the deterministic GBNF grammar-table preflight, produced an exact no-LLM FSM fixture, repaired the CSL residue metric, found a useful sparse-repair signal, and kept the hardware and ARC lanes honest. It also exposed three hard blockers that now dominate the research risk:

1. The SOTA hard/soft panel still cannot make a quality claim because live SOTA rows are incomplete even after duration and grammar preflight fixes.
2. Continuous self-learning has evidence but no claim: the five-arm CSL ablation was adversarially flagged as tautological, and the cross-model SOTA transfer gate correctly blocked on zero cross-family delta.
3. The embodied verification lanes remain one step short of PRD-grade proof: sparse repair is promising but too narrow, hardware has receipts but no matched timing, and ARC clean live level-up produced no registry delta.

Milestone `.503` therefore focuses on proof-producing infrastructure instead of new broad claims. The central move is to build deterministic constrained-generation and exact-constraint receipts first, then run small gated SOTA, CSL, repair, hardware, and ARC experiments only when their upstream receipts are clean.

## Inputs Read

The planner read the required project sources in order:

- `research-program.md`
- `_bmad/prd.md`
- `_bmad/architecture.md`
- `ops/status.md`
- `ops/changelog.md`
- `research-complete.yaml`
- `research-roadmap.yaml`
- `openspec/change-proposals/`
- `ops/conductor-log.md`
- `research-references.md`
- `research-hardware-wishlist.md`

The planner also read `CLAUDE.md`, `CODEX.md`, the current exclusion manifest, the `.502` result artifacts, and prior v7/v8/vNEXT roadmap formats before designing this milestone.

## Literature Refresh

The following findings were appended to `research-references.md` before experiment design:

- `ASP energised` (`arXiv:2607.08136`) suggests treating Answer Set Programming as a practical bridge between exact symbolic constraints and EBM-style energy surfaces. `.503` uses this as the basis for the ASP/FSM nonmonotonic fixture.
- `P-GCD` (`arXiv:2606.01926`) frames constrained decoding around tractable proposal distributions and tensorized finite automata. `.503` turns this into an automaton row-completion receipt before another SOTA panel.
- `NOVA` (`arXiv:2606.27243`) provides a verification-aware harness pattern with failure memory. `.503` maps this into forbidden-direction CSL memory, not a generic text reward.
- `Memory for Autonomous LLM Agents` (`arXiv:2603.07670`) gives a write-manage-read taxonomy for agent memory. `.503` uses that taxonomy for the causal CSL memory task.
- `schoolmarm` on GitHub is a 2026 Rust GBNF constrained-decoding implementation worth auditing, but not vendoring, unless the local `llama_cpp_gbnf` path fails.
- Extropic TSU and Logical Intelligence Kona updates remain watch-only context. They are not used as local evidence or speedup baselines.

## What 2026.07.502 Proved

| Area | `.502` Result | Planner Consequence |
| --- | --- | --- |
| SOTA duration/substrate | `exp5538` made the duration correction honest but showed no schema validity. | Duration is no longer the only blocker; row completion is now the blocker. |
| Grammar preflight | `exp5539` proved deterministic GBNF fixtures accept valid rows and reject invalid rows. | Build an automaton row-completion receipt before calling the LLM again. |
| Hard/soft panel | `exp5540` improved exact validation on emitted rows but still had missing rows and no claim. | Gate panel v4 on a grammar-forced row smoke. |
| Exact fixtures | `exp5541` shipped a deterministic FSM exact fixture. | Extend it to ASP/nonmonotonic constraints and then stress sparse repair. |
| CSL metrics | `exp5542` fixed residue metric independence, but `exp5543` was flagged for TAUTOLOGY. | Repair the five-arm ablation before any CSL headline. |
| Cross-model CSL | `exp5544` correctly blocked SOTA transfer with zero delta. | Only retry cross-model transfer after causal memory shows action impact. |
| Sparse repair | `exp5545` found descriptor-guided sparse repair signal without matched timing. | Scale to ASP/FSM rows and keep speedup claims disabled. |
| Hardware | `exp5546` produced clean receipts but no matched timing. | Build matched-timing hygiene and receipt sidecars, no speedup claim. |
| ARC | `exp5547` found a clean target; `exp5548` produced an honest null. | Rotate target and improve live-path mechanism, not offline solve. |
| Capstone | `exp5549` accepted one flagged CSL artifact and one gated skip. | `.503` must keep claim boundaries explicit and capstone-adversarial. |

## Three Biggest Gaps to PRD Vision

1. **Verified reasoning over live SOTA models is still structurally fragile.** The PRD asks for verifiable reasoning and trustworthy generation. `.502` showed that exact validators work on emitted rows, but the local SOTA models still do not reliably emit the complete candidate table needed for a fair panel.

2. **Continuous self-learning is not yet causal.** The PRD and `research-program.md` require online, self-improving behavior. `.502` found useful retrieval-warmed signals, but the best baseline collapsed into a tautology and cross-model transfer showed no positive delta.

3. **Embodied proof is missing across ARC and hardware.** The architecture requires live hidden-game discovery and hardware-backed substrate evidence. `.502` kept the ARC and hardware lanes clean, but no ARC registry delta and no matched timing mean both lanes are still evidence-limited.

## Architecture Direction

```text
                 recent literature + .502 artifacts
                               |
                               v
        +---------------- deterministic receipts ----------------+
        |                                                        |
        v                                                        v
  automaton/GBNF row completion                         ASP + exact FSM fixture
        |                                                        |
        v                                                        v
 grammar-forced SOTA row smoke                     descriptor sparse repair scale
        |                                                        |
        v                                                        |
 gated SOTA hard/soft panel v4                                  |
        |                                                        |
        +------------------ claim ledger ------------------------+
                               |
                               v
              causal write-manage-read CSL memory
                               |
                               v
              gated cross-model SOTA CSL transfer
                               |
                               v
        ARC live-path target rotation + level-up attempt
                               |
                               v
             capstone reconciliation and ops docs

 Hardware/timing receipt hygiene runs in parallel as evidence plumbing.
```

The milestone intentionally separates deterministic support checks from live LLM calls. The LLM tasks are small, gated, and use the mandated local GGUF models:

- `unsloth/Qwen3.6-35B-A3B-GGUF`
- `unsloth/gemma-4-31B-it-GGUF`
- `unsloth/gemma-4-26B-A4B-it-GGUF`

Legacy small models may appear only as CPU smoke tests, never as headline-result models.

## Phase Plan

### Phase 0 - Transition and Source Freshness

`exp5550` records the `.502` evidence ledger, claim boundaries, and protected-file checks for the new conductor queue.

`exp5551` does a narrow execution-time source delta and appends only genuinely new findings to `research-references.md`. It must not reopen retired PHASE D external text-scorer scopes.

### Phase 1 - Grammar-Forced SOTA Rows

`exp5552` builds an automaton/schema row-completion receipt from the `.502` GBNF preflight and the P-GCD finite-automata idea. This is a no-LLM gate.

`exp5553` runs a small grammar-forced local SOTA row smoke only if the automaton receipt is clean. It checks whether the mandated GGUF models can produce complete schema-valid rows under GBNF constraints.

`exp5554` runs the SOTA hard/soft panel v4 only if `exp5553` proves complete grammar-forced rows. It may produce a small quality claim only if schema validity, exact validation, candidate completeness, and adversarial checks all pass.

### Phase 2 - ASP/FSM Sparse Repair

`exp5555` extends the exact FSM fixture with ASP/nonmonotonic constraint cases inspired by the 2026 ASP+EBM paper.

`exp5556` scales descriptor-guided sparse repair over the ASP/FSM fixture only if the exact fixture is ready. It preserves the no-speedup boundary unless matched timing exists.

### Phase 3 - Causal Continuous Self-Learning

`exp5557` repairs the `.502` five-arm CSL tautology and refuses any CSL claim unless the best-constant and per-query-random baselines separate.

`exp5558` implements a causal write-manage-read memory check with forbidden-direction memory. This is the milestone's required continuous self-learning experiment.

`exp5559` retries cross-model SOTA CSL transfer only if the causal CSL memory task produces an allowed CSL claim.

### Phase 4 - Hardware Receipts, ARC, and Capstone

`exp5560` cleans hardware and timing receipt hygiene without claiming speedup. It keeps the conductor untouched.

`exp5561` rotates the ARC target using registry precheck and FSM action abstraction, avoiding duplicate solves and recent no-bank targets.

`exp5562` runs a gated live ARC level-up attempt using only `live_agent_self_discovery` provenance.

`exp5563` performs the capstone reconciliation, updates ops docs, records flags/skips/nulls, and confirms `research-roadmap.yaml` and `scripts/research_conductor.py` were not modified.

## Dependency Graph

```text
exp5550
exp5551

exp5552
  -> exp5553
       -> exp5554

exp5555
  -> exp5556

exp5557
  -> exp5558
       -> exp5559

exp5560

exp5561
  -> exp5562

exp5550, exp5551, exp5554, exp5556, exp5558, exp5559, exp5560, exp5562
  -> exp5563
```

All structured gates are also encoded in `research-roadmap-next.yaml` so the conductor can skip downstream work without spending a synthesis call.

## Hardware Requirements

The milestone can complete without new hardware. Hardware-sensitive tasks must follow the current continuity rules:

- Dual RTX 3090 is the preferred local SOTA path for GGUF inference receipts.
- KV260 work must use SSH, `xmutil`, and UIO-safe paths only. Do not use SD-card, `/dev/mmcblk`, or host-block-device preconditions.
- GateMate work remains IDCODE/dirtyJTAG limited unless a clean authenticated path appears.
- PolarFire work remains SSH/workload-receipt limited unless matched timing is available.
- Extropic TSU hardware is watch-only; no local TSU claim is allowed.
- No hardware speedup claim is allowed unless matched hardware-vs-baseline timing receipts are captured in the same experiment.

## Claim Boundaries

- No SOTA hard/soft claim unless grammar-forced live rows are complete, schema-valid, exactly validated, and adversarial-clean.
- No CSL claim unless the five-arm tautology is resolved and causal write-manage-read memory beats shuffled/no-memory controls.
- No cross-model CSL transfer claim unless cross-family transfer shows positive delta over shuffled memory and no negative-transfer spike.
- No sparse-repair speedup claim without matched timing.
- No ARC solve credit without `solve_provenance: live_agent_self_discovery`.
- No hardware speedup claim in `.503` unless matched timing unexpectedly becomes available and passes adversarial review.

## Expected Exit Criteria

The milestone is successful if it produces at least one of the following:

- A complete grammar-forced SOTA row path that allows a bounded hard/soft panel claim.
- A non-tautological CSL memory result with causal support and explicit no-weight-mutation evidence.
- A stronger exact ASP/FSM sparse-repair fixture with descriptor-guided improvement over random controls.
- A clean ARC live-path registry delta.

It is also acceptable, and useful, if the milestone produces honest nulls that narrow these claims while keeping receipts clean and docs reconciled.
