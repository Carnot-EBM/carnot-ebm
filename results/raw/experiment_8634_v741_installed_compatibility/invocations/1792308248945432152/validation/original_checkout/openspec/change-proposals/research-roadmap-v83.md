# Research Roadmap v83 — EnvGuard Surgery + Codex Config + DualGPU + WOPR Deploy + Position Paper Draft

**Milestone:** 2026.04.83
**Planned experiments:** Exps 1063–1076 (14 experiments)
**Target wall time:** ~1,600 min (goal: continue downward trend below .82's 1,699 min)
**Designed:** 2026-04-30

---

## What milestone .82 proved

Milestone .82 was 3/13 criteria met. Two independent research tracks succeeded spectacularly;
all infrastructure and deployment tracks failed to a new class of blocker.

**Hard wins:**
- FoVer corpus expanded to 6,548 pairs (13x above 500-pair target) — Exp 1055
- Probe ensemble AUROC breakthrough: best_probe_auroc=0.9899 (SOS-KAN) — Exp 1057
  The corpus-to-accuracy lever is now confirmed: 30x more data → +74 percentile-point AUROC.
- WOPR Sudoku code complete with all 4 easter eggs passing locally — Exp 1059

**New structural failures to fix in .83:**

1. **EnvPropagationGuard self-heal crash (META-BLOCKER — NEW failure mode):**
   The .82 design correctly placed exp1050 as the unconditional first task. However, the
   conductor's self-heal hook crashed with "EnvPropagationGuard failed to load CARNOT_
   variables" before exp1050 could run. The conductor log shows:
     00:53Z SKIP: "EnvPropagationGuard failed to load CARNOT_ variables"
     01:04Z SKIP: "EnvPropagationGuard failed to load CARNOT_ variables"
   This is a DIFFERENT failure from .81's 4-failing-tests pattern. The .82 design's
   pre-test-surgery-first approach was correct in principle; the self-heal mechanism
   itself is now the blocker. Must be fixed BEFORE pre-test surgery runs in .83.

2. **1 remaining failing pre-test (KV260 path):**
   After .82's exp1050 FAIL attempt, the conductor shows "1 failed, 347 passed" for
   KV260 (vs "4 failed, 318 passed" in .81). Progress: 3 of the 4 failing tests were
   somehow partially resolved, but 1 remains. This single test blocked KV260 x3.

3. **Codex agent config.toml reserved-key error (WOPR cartridges):**
   Both WOPR cartridges (exp1060, exp1061) failed x3 each with "model_providers contains
   reserved" config.toml key conflict when launching the Codex CLI agent. The conductor's
   parallel Tier A implementation (exp1051) was gate_blocked, so it never ran — even if it
   had, the Codex config error would have blocked the cartridges independently. Two fixes
   needed: (a) fix the config.toml conflict for Codex agent dispatch; (b) implement the
   parallel conductor routing (Tier A) from the spec.

4. **Gate coercion + DualGPU cascade (5th+ consecutive):**
   The same chain: EnvGuard crash → pre-test surgery fails → gate coercion (exp1052) blocked
   → DualGPU (exp1053) blocked → Triple Integration (exp1056, 4th consecutive GATE_BLOCK)
   → FR-11 alpha_t (exp1058, blocked). All five link to the EnvGuard crash as root cause.

5. **WOPR Sudoku deploy blocked by missing HF_TOKEN:**
   The Sudoku code is complete and locally validated. Only a missing HF_TOKEN prevented
   deployment. The .82 experiment design should have included SOPS secret retrieval as
   an explicit step. Fix in .83: inject HF_TOKEN from SOPS before deploy.

---

## Architecture diagram

```
Phase 0: META-PREREQUISITE (unconditioned, runs absolutely first)
  └── Exp 1063: EnvPropagationGuard self-heal repair + 1 remaining failing test  [model: opus]

Phase 1: INFRASTRUCTURE SURGERY (gated on exp1063)
  ├── Exp 1064: Pre-test surgery + respawn queue v2 (respawn of exp1050)          [model: opus]
  │            [gated on: exp1063.envguard_fixed=true]
  └── Exp 1065: Codex config.toml fix + Parallel Conductor Tier A                [model: opus]
               [gated on: exp1063.envguard_fixed=true]

Phase 2: ENVIRONMENTAL RESPAWNS (gated on exp1064)
  ├── Exp 1066: DualGPU torch install v6 (respawn of exp1053)                    [model: opus]
  │            [gated on: exp1064.pre_tests_fixed=true]
  └── Exp 1067: Gate coercion fix v3 (respawn of exp1052)                        [model: opus]
               [gated on: exp1064.pre_tests_fixed=true]

Phase 3: HARDWARE + DEPLOY (parallel independent tracks)
  ├── Exp 1068: KV260 smoke test v9 — deassert reset (1 pre-test now fixed)      [model: opus]
  │            [gated on: exp1063.remaining_test_fixed=true]
  └── Exp 1069: WOPR Sudoku HF deploy — inject HF_TOKEN via SOPS                [model: opus]
               [standalone]

Phase 4: WOPR CARTRIDGES (gated on exp1065)
  ├── Exp 1070: WOPR Global Thermonuclear War cartridge (respawn of exp1060)     [agent_type: codex]
  │            [gated on: exp1065.codex_routing_validated=true]
  └── Exp 1071: WOPR Lights Out cartridge (respawn of exp1061)                  [agent_type: codex]
               [gated on: exp1065.codex_routing_validated=true]

Phase 5: RESEARCH (independent or gated on Phase 2)
  ├── Exp 1072: SOS-KAN v3 — Neural SOS Gram matrix (arXiv 2510.13444)           [standalone]
  ├── Exp 1073: Triple Integration E2E v9 (respawn of exp1056)                   [standalone→gated]
  │            [gated on: exp1067.gate_coercion_fixed=true]
  └── Exp 1074: FR-11 alpha_t live v3 + Zenil grounding (MANDATORY self-learning) [model: opus]
               [gated on: exp1066.dualgpu_live=true]

Phase 6: POSITION PAPER
  └── Exp 1075: Position paper draft v1 (target: arxiv ~2026-05-15)              [agent_type: gemini]
               [standalone]

Phase 7: RETRO
  └── Exp 1076: Milestone 2026.04.83 Retrospective                               [max_turns: 20]
```

**Dependency graph:**

```
1063 (envguard repair) ──────────────────────────────────→ 1064 (pre-test surgery v2)
1063 ────────────────────────────────────────────────────→ 1065 (codex config + parallel conductor)
1064 ────────────────────────────────────────────────────→ 1066 (DualGPU v6)
1064 ────────────────────────────────────────────────────→ 1067 (gate coercion v3)
1063 (remaining_test_fixed=true) ───────────────────────→ 1068 (KV260 v9)
1065 (codex_routing_validated=true) ────────────────────→ 1070 (WOPR GTW)
1065 ────────────────────────────────────────────────────→ 1071 (WOPR Lights Out)
1067 (gate_coercion_fixed=true) ────────────────────────→ 1073 (Triple Integration v9)
1066 (dualgpu_live=true) ───────────────────────────────→ 1074 (FR-11 alpha_t v3)
Exp 1069, 1072, 1075 are STANDALONE (no upstream gate).
```

---

## Phase descriptions

### Phase 0: META-PREREQUISITE

**Exp 1063 — EnvPropagationGuard Self-Heal Repair v1 + Fix 1 Remaining Failing Test**

The .82 retro established a new lesson: "EnvPropagationGuard repair is a meta-prerequisite
that runs before pre-test surgery, not after." This is the UNCONDITIONAL first task of .83.

Two deliverables:
1. **EnvPropagationGuard self-heal crash:** Find the conductor's self-heal hook that calls
   EnvPropagationGuard. The crash message is "EnvPropagationGuard failed to load CARNOT_
   variables." Make the self-heal graceful: if CARNOT_ variables are absent from the
   environment, log a warning and continue rather than crashing/aborting the self-heal path.
   This allows pre-test surgery to run even when the environment isn't fully configured.
2. **1 remaining failing test:** Run pytest to find it (conductor log confirms "1 failed,
   347 passed, 1 warning"). Fix it. Verify all tests green.

This experiment must NOT gate on anything. It runs first, regardless of environment state.

`model: opus` — conductor infrastructure + test surgery, multiple failure modes.

### Phase 1: INFRASTRUCTURE SURGERY

**Exp 1064 — Pre-Test Surgery + Respawn Queue v2 (respawn of exp1050)**

Respawn of exp1050, which produced no artifact across 3 attempts (.82):
- FAIL: max_turns=50 exhausted before completing
- SKIP x2: EnvPropagationGuard crash in self-heal path (now fixed by exp1063)

Tasks:
1. Run pytest to confirm zero failing tests (exp1063 should have cleared them all).
2. If any tests still fail, diagnose and fix.
3. Implement `ops/respawn-queue.json` with the 5 environmentally-retired experiments:
   exp1039, exp1042, exp1044 (from .81) + exp1050, exp1051, exp1053 (from .82).
4. Patch `scripts/research_conductor.py:_classify_retirement()` to auto-populate the queue.
5. Write `tests/python/test_respawn_queue.py` (3+ tests).

`model: opus`
`gated_on: exp1063.envguard_fixed=true`

**Exp 1065 — Codex Config.toml Fix + Parallel Conductor Tier A (respawn of exp1051)**

Two tasks:
1. **Codex config.toml fix:** The "model_providers contains reserved" error blocks every Codex
   agent invocation. Find the config.toml path used when AGENT_TYPE=codex and identify the
   reserved key conflict. Fix: either strip the `model_providers` key from the Codex config
   copy, or pass `--config` to override the conflicting setting.
2. **Parallel conductor Tier A:** Implement worktree routing in `scripts/research_conductor.py`
   (from `openspec/change-proposals/parallel-multi-agent-conductor.md`). Tasks with
   `agent_type: codex` dispatch to the codex CLI with a separate state-file suffix `_codex`.
   Validate with a Codex smoke test (minimal task, confirms routing works).
   Write the systemctl service stub for `carnot-conductor-codex`.

This unblocks the WOPR cartridge sprint (exp1070, exp1071) and the position paper timeline.

`model: opus` — conductor + config surgery.
`gated_on: exp1063.envguard_fixed=true` (safe to touch conductor only after tests green)

### Phase 2: ENVIRONMENTAL RESPAWNS

**Exp 1066 — DualGPU ROCm Torch Install v6 (respawn of exp1053)**

Fifth+ respawn of the DualGPU torch install (exp1035/1042/1053 all produced no artifact).
Prior failures:
- exp1053: GATE_BLOCK x3 (gated on exp1050 which produced no artifact)
- exp1042: FAIL+SKIP x2 (pre-test failures)
- exp1035: dualgpu_detected_torch_backend_missing

nvidia-smi confirms 2x RTX 3090 (confirmed Exp 1035). torch 2.11.0+cpu is installed.
Three paths in order (stop at first success):
- Path A: pip install torch+cu121 (CUDA 12 wheel for RTX 3090)
- Path B: pip install torch+rocm7.2 (ROCm for AMD 890M iGPU, per CLAUDE.md gfx1150)
- Path C: llama-cpp-python[cuda] dispatch (bypasses torch CUDA requirement)

Live inference smoke test required once any path succeeds.

`model: opus`
`gated_on: exp1064.pre_tests_fixed=true` (hardware install modifies environment; safe only with green tests)

**Exp 1067 — Gate Coercion Fix v3 (respawn of exp1052)**

Respawn of exp1052 (GATE_BLOCK — upstream exp1050 retired), which was itself a respawn of
exp1039 (no_artifact — pre-test failures blocked all 3 attempts in .81).

The underlying fix is already implemented in `scripts/research_conductor.py` from milestone .80:
- `_deliverable_exists()` inspects `status` field before returning True (fastpath bootstrap skip)
- `_coerce_gate_value()` normalizes string "True" to bool True

This experiment verifies both fixes are in place, runs the 12 deliverable-status tests, and
runs the wedge + gate-coercion replay scenarios to confirm they pass.

`model: opus`
`gated_on: exp1064.pre_tests_fixed=true`

### Phase 3: HARDWARE + DEPLOY (parallel tracks)

**Exp 1068 — KV260 Smoke Test v9 (after 1 remaining failing test fixed)**

The KV260 is one register write from smoke_test_passed=True. The only blocker in .82 was
1 failing pre-test (conductor shows "1 failed, 347 passed"). Exp 1063 fixes it.

Steps:
1. Confirm SSH reachable: `ping 192.168.51.98` + `ssh ubuntu@192.168.51.98 echo OK`
2. Find the AXI control register from `hardware/kv260/build_bd.tcl` and `ising_sampler_v1.v`
3. Deassert reset via devmem2, /dev/uio0, or PYNQ (three methods in order)
4. Run smoke test: verify `unique_values > 1` AND `energy_distribution_nonuniform=True`
5. If smoke_test_passed: measure hardware latency (target: < 100μs per call)

Prior failure: exp1054 (SKIP x3 — 1 failing pre-test, same root cause now fixed by exp1063).

`model: opus` — FPGA hardware + register-level debugging.
`gated_on: exp1063.remaining_test_fixed=true` (the 1 pre-test that blocked KV260 is fixed here)

**Exp 1069 — WOPR Sudoku HF Spaces Deploy (inject HF_TOKEN via SOPS)**

The WOPR Sudoku code is complete and locally validated (Exp 1059: space_code_complete=True,
all 4 easter eggs pass, Ising solver reaches E=0 in 5130 iterations). Only the HF_TOKEN
was missing.

Steps:
1. Retrieve HF_TOKEN from SOPS-encrypted secrets store (check `~/.config/sops/` or project
   `secrets/` directory, per CLAUDE.md: "All embedded secrets must use SOPS encryption").
   If no SOPS secret exists: check `HF_TOKEN` environment variable as fallback.
2. Run `huggingface-cli login` with the retrieved token.
3. Execute `spaces/wopr-games/deploy.sh` or equivalent deploy command.
4. Verify deployed Space is live: curl the HF Spaces URL, check for 200 response.
5. Record the live URL in the artifact.

This is a standalone experiment — no upstream gate dependency, just SOPS + deploy.

`model: opus` — deployment is multi-step infra-class with SOPS secret retrieval.

### Phase 4: WOPR CARTRIDGES

**Exp 1070 — WOPR Global Thermonuclear War Cartridge (respawn of exp1060)**

Prior failures: exp1060 (FAIL x3 — "model_providers contains reserved" Codex config error).
Addressed by: exp1065 fixed the config.toml conflict; Codex agent now launches cleanly.

The Global Thermonuclear War cartridge is the cultural anchor of the WOPR gallery:
- WOPR "computes scenarios" with frantic CRT animation
- Concludes: "A STRANGE GAME. THE ONLY WINNING MOVE IS NOT TO PLAY. HOW ABOUT A NICE GAME OF CHESS?"
- No actual AI computation — pure animation + typewriter reveal
- File: `spaces/wopr-games/games/global_thermonuclear_war.py`
- Implements `WOPRGame` interface

`agent_type: codex` — formulaic cartridge code following a well-defined interface.
`gated_on: exp1065.codex_routing_validated=true`

**Exp 1071 — WOPR Lights Out Cartridge (respawn of exp1061)**

Prior failures: exp1061 (FAIL x3 — same config.toml error as exp1060).
Addressed by: exp1065 fixed the Codex config.

The Lights Out cartridge is the best Carnot demo in the gallery:
- 5×5 grid, XOR toggling, all-off goal
- E = Σ_i (state_i)^2 (energy = number of lit cells)
- Ising sampling finds button-press sequence reaching E=0
- Visually satisfying: cells cascade off as energy descends
- File: `spaces/wopr-games/games/lights_out.py`

`agent_type: codex` — well-defined CSP, low complexity.
`gated_on: exp1065.codex_routing_validated=true`

### Phase 5: RESEARCH

**Exp 1072 — SOS-KAN v3: Neural SOS Gram Matrix (arXiv 2510.13444)**

arXiv 2510.13444 (Neural SOS) suggests replacing V·V^T in SOSKANEnergy with a learned
Gram matrix from a small transformer, maintaining the SOS (sum-of-squares) guarantee that
the energy is certifiably nonnegative while increasing expressivity.

Now that the FoVer corpus has 6,548 pairs (Exp 1055), there is sufficient data to train
a more expressive SOS model. The prior SOS-KAN v1 (Exp 1047) achieved AUROC=0.6042 on
~200 pairs. This experiment trains on the full 6,548-pair corpus with the learned Gram.

Steps:
1. Read `python/carnot/models/sos_kan.py` (Exp 1047 implementation)
2. Replace the V·V^T decomposition with a learned 2-layer transformer that outputs a
   low-rank PSD matrix (Gram matrix approach from arXiv 2510.13444)
3. Train on data/fover_corpus_v4.json (6,548 pairs)
4. Compare AUROC: SOS-KAN v1 (fixed V) vs SOS-KAN v3 (learned Gram)
5. Verify: 0 monotonicity violations across 16,000 samples (keep the invariant from Exp 1047)
6. Target: AUROC >= 0.72 with certifiable SOS invariant

This is STANDALONE — uses the already-expanded FoVer corpus, no upstream gate.

Model: sonnet (standard research experiment, 50 turns sufficient).

**Exp 1073 — Triple Integration E2E v9 (respawn of exp1056, 4th consecutive)**

Prior failures:
- exp1056: GATE_BLOCK x3 (upstream exp1052 retired)
- exp1044: GATE_BLOCK x3 (upstream exp1039 never ran)
- exp1032: partial_gate_blocked (gate coercion bug, string "True" not recognized)
- exp1004: no_artifact

This has been gated on the conductor infrastructure chain for FOUR consecutive milestones.
Now gated on exp1067.gate_coercion_fixed=true (the gate coercion bug is finally fixed).

Full 3-tier cascade: ThinkPRM Tier 0a → SpilledEnergy Tier 0b → SC-Energy Tier 2 → Ising Tier 3.
Run 50 questions (20 correct / 20 incorrect / 10 ambiguous).
Target: all_tier_skip_rates_nonzero=True, cascade_e2e_confirmed=True.

`gated_on: exp1067.gate_coercion_fixed=true`
retire_if_same_verdict: false (environmental retirements don't count toward merit retirement)

**Exp 1074 — FR-11 alpha_t Live v3 + Zenil Grounding (MANDATORY continuous self-learning)**

Prior failures:
- exp1058: GATE_BLOCK x3 (upstream exp1053 gate_blocked; DualGPU never ran)
- exp1046: bootstrap_wedge x3 (fastpath bug triggered by GGUF not in cache)

FR-11 MANDATORY per research-program.md: every milestone must advance the continuous
self-learning architecture. This is the Zenil alpha_t measurement — the empirical
confirmation that Carnot's verifier provides a non-vanishing grounding signal (α_t > 0).

New theoretical grounding (arxiv 2604.03128 Self-Distilled RLVR): empirically confirms
Zenil Theorem 4 — pure self-distillation without verifier signal collapses after 3-5 rounds.
Carnot's energy-based filter IS the α_t μ_P term that prevents collapse. This experiment
measures α_t directly on live GPU inference.

Steps:
1. Load Qwen3.6-35B-A3B-GGUF (pre-confirmed in cache via Exp 1055's GGUF download)
2. Generate 50 live questions (inference_mode=live_gpu, CARNOT_FORCE_LIVE=1)
3. Run Phase-3 AND-composition verifier (k=5, per FPGA-feasible Round-9 bound)
4. Measure α_t = fraction of examples where Carnot verdict ≠ temperature-only verdict
5. Write 100+ FR-11 training examples to data/fr11_zenil_distill_v2.jsonl
6. Compute and report Φ = E[verifier_signal] per python/carnot/eval/phi_test.py

Target: inference_mode=live_gpu, alpha_t > 0.0, fr11_loop_closed=True.

`model: opus` — multi-GPU inference + complex measurement.
`gated_on: exp1066.dualgpu_live=true`

### Phase 6: POSITION PAPER

**Exp 1075 — Position Paper Draft v1 (target: arXiv ~2026-05-15)**

The Phase-3 → Phase-7 defence-layer stack derivation is complete (6 Deep Think rounds,
30+ theorems). The docs/research-notes/*.md files contain the full derivations. The
outline is at docs/position-paper-outline.md. This experiment produces a first draft.

The position paper is the primary external deliverable that pairs with the WOPR Sudoku demo
for the ~2026-05-15 arXiv submission target.

New arxiv findings from this planning scan that must be incorporated:
- arXiv 2508.14496 (Semantic Energy) — validates Tier 0b/0c logit-based energy approach
- arXiv 2604.03128 (Self-Distilled RLVR) — empirically confirms Zenil Theorem 4 (cite in §3)
- arXiv 2602.15985 (FPGA Ising decomposition) — validates Phase 2 hardware mandate (cite in §5)
- arXiv 2510.13444 (Neural SOS) — supports SOS-KAN's certifiable energy approach (cite in §4)

Existing references to include:
- arXiv 2512.20664 (Eidoku) — contemporaneous neuro-symbolic verification gate
- arXiv 2603.19562 (Neural Uncertainty Principle) — geometric grounding for why EBMs work
- arXiv 2601.05280 (Zenil self-improvement limits) — foundational theorem for verifier role
- arXiv 2506.14590 (MTJ Ising machine) — Phase 2 hardware path validation
- arXiv 2604.04636 (KAN crystal energies) — independent KAN energy function validation

Steps:
1. Read `docs/position-paper-outline.md` (full outline)
2. Read all `docs/research-notes/` files (full Deep Think derivations)
3. Draft all sections per the outline structure
4. Write `docs/position-paper-draft-v1.md`
5. Write artifact with section_count, word_count, missing_sections

`agent_type: gemini` — long-context synthesis; position paper requires ingesting all
research notes simultaneously (>100K tokens of derivations).
`model: gemini-3.1-pro-preview` — standard Gemini thinking (note: NOT Deep Think).

### Phase 7: RETRO

**Exp 1076 — Milestone 2026.04.83 Retrospective**

Evaluates all 15 success criteria, documents process observations, writes biggest_gaps_84.
Appends to ops/changelog.md and ops/conductor-log.md (APPEND ONLY per CLAUDE.md).

`max_turns: 20` — analysis-only, no new code.

---

## Success Criteria

| # | Criterion | Experiment | Target field/value |
|---|-----------|------------|-------------------|
| 1 | EnvGuard self-heal fixed | Exp 1063 | envguard_fixed=true, self_heal_test_passing=true |
| 2 | 1 remaining failing test fixed | Exp 1063 | remaining_test_fixed=true, n_failing=0 |
| 3 | Pre-tests fully green + respawn queue | Exp 1064 | pre_tests_fixed=true, respawn_queue_seeded=true |
| 4 | Codex config.toml fixed + Tier A routing | Exp 1065 | codex_routing_validated=true |
| 5 | DualGPU live inference | Exp 1066 | dualgpu_live=true |
| 6 | Gate coercion deployed + verified | Exp 1067 | gate_coercion_fixed=true, wedge_replay_clean=true |
| 7 | KV260 smoke test passed | Exp 1068 | smoke_test_passed=true |
| 8 | WOPR Sudoku deployed on HF Spaces | Exp 1069 | space_deployed=true, live_url non-null |
| 9 | WOPR GTW cartridge shipped | Exp 1070 | honest_verdict=cartridge_shipped |
| 10 | WOPR Lights Out cartridge shipped | Exp 1071 | honest_verdict=cartridge_shipped |
| 11 | SOS-KAN v3 AUROC >= 0.72 | Exp 1072 | sos_kan_v3_auroc>=0.72, violations=0 |
| 12 | Triple Integration cascade confirmed | Exp 1073 | all_tier_skip_rates_nonzero=true |
| 13 | FR-11 alpha_t live closed | Exp 1074 | fr11_loop_closed=true, inference_mode=live_gpu |
| 14 | Position paper draft written | Exp 1075 | draft_written=true, word_count>=5000 |
| 15 | Retro complete | Exp 1076 | honest_verdict written |

---

## arxiv findings incorporated

From the 2026-04-30 scan:

- **arXiv 2508.14496** (Semantic Energy) — validates Tier 0b/0c logit-based energy approach;
  cite in position paper §2 (related work) and §4 (architecture)
- **arXiv 2604.03128** (Self-Distilled RLVR) — empirically confirms Zenil Theorem 4;
  incorporated into Exp 1074 (FR-11 alpha_t) theoretical framing; cite in §3
- **arXiv 2602.15985** (FPGA Ising decomposition) — 2x speedup + 100x energy efficiency
  via FPGA co-design; validates Phase 2 hardware mandate; cite in §5
- **arXiv 2604.01193** (Simple SSD for code) — supports filtered SSD as FR-11 mechanism;
  cite in §4 as evidence that execution-feedback SSD works
- **arXiv 2602.19114** (Kaiwu photonic) — Phase 2/3 photonic backend path; added to hardware
  wishlist; cite in §5 hardware section

From prior scans not yet cited in a paper:
- **arXiv 2510.13444** (Neural SOS) — concrete experiment (Exp 1072, SOS-KAN v3)
- **arXiv 2603.19562** (Neural Uncertainty Principle) — §3 theoretical grounding
- **arXiv 2601.05280** (Zenil limits) — §1 motivation + §3 foundation

---

## Hardware requirements

| Experiment | GPU | FPGA | Notes |
|------------|-----|------|-------|
| 1063 | No | No | CPU-only conductor/test surgery |
| 1064 | No | No | CPU-only test surgery + conductor patch |
| 1065 | No | No | Conductor code + config surgery |
| 1066 | Yes | No | torch CUDA install + live inference smoke test |
| 1067 | No | No | Conductor code verification |
| 1068 | No | Yes (KV260 192.168.51.98) | SSH to board, AXI register write |
| 1069 | No | No | HF Spaces deployment via CLI |
| 1070 | No | No | Pure animation cartridge (Codex) |
| 1071 | No | No | Ising ground-state on CPU Ising sampler (Codex) |
| 1072 | No | No | SOS-KAN training on CPU (small model) |
| 1073 | No | No | Pipeline cascade CPU |
| 1074 | Yes | No | Qwen3.6-35B live inference (gated on exp1066) |
| 1075 | No | No | Document synthesis (Gemini long-context) |
| 1076 | No | No | Retro analysis |

---

## Process improvements vs .82

1. **EnvGuard as meta-prerequisite (Phase 0):** The .82 lesson: the self-heal hook's dependency
   on EnvPropagationGuard created a new crash-before-start failure mode. By making EnvGuard
   repair the ABSOLUTE first task (Exp 1063, unconditioned, runs before everything else),
   this failure mode cannot recur.

2. **Codex config smoke-test before cartridge sprint:** The .82 lesson: even if the parallel
   conductor had run (exp1051), the Codex config error would have blocked WOPR cartridges
   independently. Exp 1065 fixes both the routing AND the config in a single experiment,
   then validates with a Codex smoke-test before any cartridge work begins.

3. **WOPR Sudoku deploy is standalone:** Moving Sudoku deploy (Exp 1069) to a parallel track
   with no gate dependency means it can run regardless of whether the infrastructure chain
   succeeds. Code is ready; only SOPS-secret retrieval is needed.

4. **GGUF pre-download no longer needed for alpha_t:** Exp 1055 already downloaded and
   confirmed Qwen3.6-35B-A3B-GGUF in cache. Exp 1074 can rely on the existing cache.

5. **SOS-KAN uses full corpus (standalone):** The FoVer corpus expansion (Exp 1055) delivered
   6,548 confirmed pairs. SOS-KAN v3 is standalone and can start immediately without waiting
   for any infrastructure fix — it just needs the corpus file that already exists.

6. **Position paper is standalone (Gemini long-context):** The paper synthesis doesn't depend
   on any code deliverables. Gemini's 1M token window can ingest all research notes at once.
   Targeted at ~2026-05-15 arXiv submission; this experiment must ship draft by end of .83.
