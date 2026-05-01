# Research Roadmap v82 — Pre-Test Surgery + Respawn + WOPR Gallery + Parallel Conductor + KV260 Smoke

**Milestone:** 2026.04.82
**Planned experiments:** Exps 1050–1062 (13 experiments)
**Target wall time:** ~1,500 min (goal: continue downward trend below .81's 1,699 min)
**Designed:** 2026-04-29

---

## What milestone .81 proved

Milestone .81 was a mixed hardware/infrastructure cycle: 6/13 criteria met.

**Hard wins:**
- KV260 bitstream loaded for the first time across 7 milestones (Exp 1041, state=operating)
- SOS-KAN type-level invariants confirmed: 0 violations across 16,000 samples (Exp 1047)
- Eval metrics canonical: `_check_auroc_anomaly()` wired into conductor self-heal (Exp 1048)
- Verdict stability/seed discipline: stability_rate=1.0, random_seed added to build_result() (Exp 1040)

**Structural failures to fix in .82:**
1. **Pre-test deadlock (CRITICAL):** 4 failing tests from Exp 1028's .80 suite expansion blocked
   Exp 1039 on all 3 attempts (SKIP/FAIL/SKIP). This cascaded to block Exp 1044 (Triple
   Integration GATE_BLOCK x3) — fourth consecutive milestone Triple Integration has failed.
   The pattern: the infrastructure fix (gate coercion) cannot run because tests are broken;
   tests stay broken because the infrastructure fix cannot run. Must be the VERY FIRST task.

2. **Environmental retirement deadlock:** Exp 1039 (gate coercion fix), Exp 1042 (DualGPU),
   Exp 1044 (Triple Integration) all retired due to environmental failures, NOT merit failures.
   The `no-permanent-retirement-on-environmental-failures.md` change proposal requires a respawn
   queue mechanism. These three must be queued for respawn in .82.

3. **FoVer MetaQA still stub (2nd consecutive milestone):** n_metamorphic_validated=0 because
   the GGUF (gemma-4-26B-A4B) is not cached locally. MetaQA branch exits early silently.
   Must pre-download the GGUF as a Phase 0 preflight step BEFORE the MetaQA generator runs.

4. **DualGPU 16th consecutive idle:** torch 2.11.0+cpu cannot do GPU inference. Live GPU
   experiments (Zenil alpha_t FR-11, benchmarks) all blocked. Pre-test fix + explicit torch
   CUDA install needed.

5. **KV260 one register write from smoke test:** bitstream_loaded=True, state=operating.
   The sampler control register reset bit has not been deasserted. One SSH command away
   from smoke_test_passed=True.

6. **WOPR gallery not started:** Known-issues.md mandates 3 specific WOPR tasks for .82:
   Sudoku v1 HF Space, Global Thermonuclear War cartridge, Lights Out cartridge. The
   position paper targets arxiv ~2026-05-15; the Sudoku Space demo provides the clickable
   artifact to pair with it.

7. **Parallel multi-agent conductor not shipped:** `worktree: codex` field exists in schema
   (commit aa3c2707) but conductor routing not implemented. Without it, Codex cartridge tasks
   run on the main Claude conductor — slower and burns Claude quota.

---

## Architecture diagram

```
Phase 0: MANDATORY INFRASTRUCTURE (standalone, run first — no gates)
  ├── Exp 1050: Pre-test surgery + respawn queue v1              [model: opus]
  └── Exp 1051: Parallel multi-agent conductor Tier A            [model: opus]
                                                                  [gated on: exp1050]

Phase 1: ENVIRONMENTAL RESPAWNS (from respawn queue)
  ├── Exp 1052: Fastpath + gate coercion fix v2 (respawn 1039)   [model: opus]
  │            [gated on: exp1050.pre_tests_fixed=true]
  └── Exp 1053: DualGPU ROCm torch install v5 (respawn 1042)    [model: opus]
               [gated on: exp1050.pre_tests_fixed=true]

Phase 2: HARDWARE + CORPUS (standalone)
  ├── Exp 1054: KV260 smoke test v8 — deassert reset             [model: opus]
  └── Exp 1055: FoVer corpus expansion v4 — GGUF pre-download    [GPU required]

Phase 3: CASCADE + PROBES + FR-11 (self-learning mandatory)
  ├── Exp 1056: Triple Integration E2E v8 (respawn 1044)
  │            [gated on: exp1052.gate_coercion_fixed=true]
  ├── Exp 1057: Probe ensemble v6 (ThinkPRM + GS-KAN + NK-KAEM)  [GPU required]
  │            [gated on: exp1055.n_total_pairs>=200]
  └── Exp 1058: Zenil alpha_t FR-11 live v2 (FR-11 MANDATORY)   [model: opus, GPU required]
               [gated on: exp1053.dualgpu_live=true]

Phase 4: WOPR GALLERY (3 mandatory per known-issues.md)
  ├── Exp 1059: WOPR Spaces Sudoku v1 — HF deploy               [model: opus]
  ├── Exp 1060: WOPR Global Thermonuclear War cartridge           [agent_type: codex]
  └── Exp 1061: WOPR Lights Out cartridge                        [agent_type: codex]

Phase 5: RETRO
  └── Exp 1062: Milestone 2026.04.82 Retrospective
```

**Dependency graph:**

```
1050 (pre-test surgery) ──────────────────────────────────→ 1051 (parallel conductor)
1050 ────────────────────────────────────────────────────→ 1052 (gate coercion fix)
1050 ────────────────────────────────────────────────────→ 1053 (DualGPU install)
1052 (gate coercion fixed) ────────────────────────────→ 1056 (triple integration)
1055 (FoVer 200+ pairs) ───────────────────────────────→ 1057 (probe ensemble)
1053 (dualgpu_live=true) ──────────────────────────────→ 1058 (Zenil alpha_t FR-11)
All WOPR tasks standalone (no upstream gate).
```

---

## Phase descriptions

### Phase 0: MANDATORY INFRASTRUCTURE

**Exp 1050 — Pre-test surgery + respawn queue v1**

CLAUDE.md "Mandatory-first" rule: the 4 failing pre-tests from .80's Exp 1028 suite expansion
are the root blocker for EVERYTHING else in .81 and .82. They must be fixed FIRST.

The conductor log shows:
- Exp 1039 attempt 1: SKIP — "1 failed, 302 passed" (16:23Z)
- Exp 1039 attempt 2: FAIL — max_turns=30 (16:38Z)
- Exp 1039 attempt 3: SKIP — "4 failed, 318 passed" (17:10Z)

The failing test count GREW from 1 to 4 between attempts 1 and 3, suggesting experiment runs
themselves broke additional tests. This must be diagnosed carefully.

This experiment also implements the `no-permanent-retirement-on-environmental-failures.md`
respawn queue mechanism (MANDATORY per known-issues.md .82 pickup):
- Create `ops/respawn-queue.json` with the 3 environmentally-retired experiments
- Patch `scripts/research_conductor.py:pick_next_task()` to classify retirement as
  environmental vs. merit and auto-populate the queue on environmental retirements
- Seed the queue with exp1039/1042/1044 using their already-documented prior_failures

`model: opus` — multi-step test surgery + conductor code modification.

**Exp 1051 — Parallel multi-agent conductor Tier A**

The `worktree: codex` YAML field already exists in the schema (commit aa3c2707). What remains:
1. In `scripts/research_conductor.py`, route tasks with `worktree: codex` to a codex subprocess
   (AGENT_TYPE=codex with a separate state-file suffix `_codex`).
2. Validate routing: run a dry-run with `.82`'s YAML and confirm the 2 codex tasks (1060, 1061)
   parse and route correctly.
3. Write `systemctl --user` service file stub for `carnot-conductor-codex`.

This is the prerequisite for the WOPR sprint delivering on schedule (position paper ~2026-05-15).

`model: opus` — conductor infrastructure modification.
`gated_on: exp1050.pre_tests_fixed=true` — cannot risk breaking conductor while tests are red.

### Phase 1: ENVIRONMENTAL RESPAWNS

**Exp 1052 — Conductor fastpath bootstrap skip + gate coercion fix v2**

Respawn of exp1039 (environmental retirement: pre-test failures blocked all 3 attempts).
The change proposal `conductor-fastpath-bootstrap-skip.md` is already implemented in the
conductor (`_deliverable_exists` status-aware fast-path, 12 tests in
`test_conductor_deliverable_status.py`). This experiment closes it out:
1. Verify 12 tests pass (pre-tests now fixed by exp1050).
2. Implement gate coercion normalization in `_evaluate_gate()`.
3. Run wedge replay: create bootstrap stub, confirm it is NOT treated as completed.
4. Run gate coercion replay: confirm "True" string evaluates to bool True.

`model: opus` — conductor code modification + replay testing.
`gated_on: exp1050.pre_tests_fixed=true`

**Exp 1053 — DualGPU ROCm torch install v5**

Respawn of exp1042 (environmental retirement: pre-test failures + max_turns too tight).
The root cause: torch 2.11.0+cpu is installed, but nvidia-smi confirms 2x RTX 3090.
Three paths in order (stop at first success):
- Path A: pip install torch+cu121 (CUDA 12 wheel for RTX 3090)
- Path B: pip install torch+rocm6.0 (ROCm for AMD 890M iGPU)
- Path C: llama-cpp-python[cuda] dispatch (bypasses torch CUDA requirement)
Target: `dualgpu_live=true`, at least 5 live GPU inference calls succeed.

`model: opus` — hardware install + diagnostic.
`gated_on: exp1050.pre_tests_fixed=true`

### Phase 2: HARDWARE + CORPUS

**Exp 1054 — KV260 smoke test v8 — deassert reset**

From Exp 1041 notes: "Read hardware/kv260/build_bd.tcl to find the sampler's control/start
register and write a 1 to deassert reset." The bitstream is already loaded (bitstream_loaded=True,
slot_handle=0, state=operating). The smoke test returns all-zeros because the Ising sampler
core is still held in reset.

This experiment:
1. SSHs to 192.168.51.98, reads the AXI register map from `hardware/kv260/build_bd.tcl`.
2. Writes the reset-deassert command via `/dev/uio0` or `devmem2` to the sampler control register.
3. Reads back 100 spin samples; verifies energy_distribution_nonuniform=True.
4. If smoke_test_passed: measures hardware latency (target: <100μs per Ising call).
5. Fallback: if reset mechanism unclear, write a detailed v3 guide with the exact register address.

`model: opus` — FPGA hardware integration, multiple failure modes.

**Exp 1055 — FoVer corpus expansion v4 — GGUF pre-download + MetaQA generator**

Root cause of MetaQA stub across .80 and .81: `gemma-4-26B-A4B-it-GGUF` is not in the local
HuggingFace cache. The generator silently exits rather than raising a clear model-not-cached error.

This experiment:
1. Downloads `unsloth/gemma-4-26B-A4B-it-GGUF` BEFORE any MetaQA generation attempt.
   Use `huggingface_hub.snapshot_download()` with progress bar; timeout=3600s.
   If download fails: write `blocked_model_not_downloadable` and exit (no silent stub).
2. ONLY AFTER download: run MetaQA generator with 200 candidate steps.
3. Combine with existing 216 Z3-confirmed pairs from exp1043.
4. Target: n_total_pairs >= 500.

GPU required for MetaQA generation.

### Phase 3: CASCADE + PROBES + FR-11

**Exp 1056 — Triple Integration E2E v8**

Respawn of exp1044, which was GATE_BLOCK x3 because exp1039 never produced an artifact.
Now gated on exp1052's gate_coercion_fixed=true (not exp1039's, since 1039 is the same
infrastructure work requeued through the respawn chain as exp1052).

The full 3-tier verification cascade (ThinkPRM Tier 0a → SpilledEnergy Tier 0b →
SC-Energy Tier 2 → Ising Tier 3) has NEVER been validated E2E. This closes that gap.

Runs 50 questions (20 correct / 20 incorrect / 10 ambiguous), measures skip rates per tier,
confirms all_tier_skip_rates_nonzero.

`gated_on: exp1052.gate_coercion_fixed=true`

**Exp 1057 — Probe ensemble v6**

Carry-forward from exp1045 (best_probe_auroc=0.5694, all below 0.72 — root cause: only
216 pairs). Now gated on exp1055 producing n_total_pairs>=200.

Trains ThinkPRM (with real gemma-4-31B-it-GGUF inference, not CI stub) + GS-KAN +
NK-KAEM on expanded FoVer corpus. Implements NK step reduction K=5 for stability
(vs K=10 in exp1045 which diverged). Target: best_probe_auroc >= 0.72.

GPU required.
`gated_on: exp1055.n_total_pairs>=200`

**Exp 1058 — Zenil alpha_t FR-11 live v2 (MANDATORY continuous self-learning)**

Respawn of exp1046 (FR-11 MANDATORY). Three failures in .81 were
`artifact_not_updated_past_bootstrap` — the bootstrap wedge triggered because
Qwen3.6-35B-A3B GGUF is not in cache AND the fastpath bug (now fixed by exp1052)
prevented retries.

This is the MANDATORY continuous self-learning experiment for the milestone per FR-11:
- Implements Φ measurement module (python/carnot/eval/phi_test.py)
- Deploys Phase-3 AND-composition verifier k=5 per Deep Think Round-9 FPGA-feasible recipe
- Generates 50 live questions via Qwen3.6-35B-A3B-GGUF (MUST be inference_mode=live_gpu)
- Measures alpha_t (fraction of examples where Carnot's AND-composed verdict ≠ temperature verdict)
- Writes 100+ FR-11 training examples to data/fr11_zenil_distill_v1.jsonl

`gated_on: exp1053.dualgpu_live=true` (live GPU is mandatory per experiment spec)
`model: opus` — GPU inference + multi-step measurement + complex prior_failures history

**NEW arxiv finding informing this experiment:** arXiv 2603.19562 (Neural Uncertainty Principle)
provides a geometric interpretation of why α_t > 0 is necessary: hallucinations concentrate
near constraint-violation boundaries, and Carnot's verifier specifically probes this boundary.
α_t measures how often Carnot's probe changes the selection — a direct empirical test of the
theoretical prediction.

### Phase 4: WOPR GALLERY (3 MANDATORY per known-issues.md)

**Exp 1059 — WOPR HuggingFace Spaces Sudoku v1**

From `openspec/change-proposals/huggingface-spaces-sudoku-demo.md`:
- Base WOPR shell: CRT terminal aesthetic, typewriter streaming, green phosphor energy bar
- Sudoku solver using Ising energy descent with visual animation of energy decreasing
- Easter eggs: `LIST GAMES`, `GLOBAL THERMONUCLEAR WAR`, `HOW ABOUT A NICE GAME OF CHESS`,
  `GREETINGS PROFESSOR FALKEN`
- Deployed at HuggingFace Spaces as a publicly-accessible URL
- This is the clickable demo paired with the ~2026-05-15 position paper preprint

`model: opus` — Spaces deployment is multi-step infra-class work (known bootstrap-and-bail risk)
`agent_type: claude` — synthesis/deployment work, not formulaic cartridge code

**Exp 1060 — WOPR Global Thermonuclear War cartridge**

From known-issues.md: "The cultural anchor — WOPR 'computes scenarios' with frantic CRT animation,
then concludes: 'A STRANGE GAME. THE ONLY WINNING MOVE IS NOT TO PLAY. HOW ABOUT A NICE GAME
OF CHESS?' Pure marketing win. Must ship in week 1."

Cartridge structure: `spaces/wopr-games/games/global_thermonuclear_war.py`
Implements `WOPRGame` interface. Animation: rapid scenario computation display → pause →
typewriter-reveal of the conclusion. No actual AI computation needed.

`agent_type: codex` — formulaic cartridge code following a well-defined interface
`model: sonnet` (Codex equivalent for this task pattern)

**Exp 1061 — WOPR Lights Out cartridge**

From known-issues.md: "The single best Carnot demo in the gallery: 5×5 grid, XOR toggling,
all-off goal. Mathematically a pure Ising-model ground-state search — Carnot's energy formulation
literally IS the natural-language solver. Visually satisfying: cells cascade off as energy descends."

Implements the constraint naturally: E = Σ_i (state_i)^2 (energy = number of lit cells).
Ising sampling finds the sequence of button presses that reaches ground state E=0.

`agent_type: codex` — formulaic cartridge (well-defined constraint encoding)
`model: sonnet` (Codex)

### Phase 5: RETRO

**Exp 1062 — Milestone 2026.04.82 Retrospective**

Evaluates all 13 success criteria, documents process observations, writes biggest_gaps_83.
Appends to ops/changelog.md and ops/conductor-log.md (APPEND ONLY per CLAUDE.md).

---

## Success Criteria

| # | Criterion | Experiment | Target field/value |
|---|-----------|------------|-------------------|
| 1 | Pre-tests fully green | Exp 1050 | pre_tests_fixed=true, n_failing=0 |
| 2 | Respawn queue implemented | Exp 1050 | respawn_queue_seeded=true, n_queued>=3 |
| 3 | Parallel conductor routes codex | Exp 1051 | codex_routing_validated=true |
| 4 | Gate coercion fix deployed | Exp 1052 | gate_coercion_fixed=true, wedge_replay_clean=true |
| 5 | DualGPU live inference | Exp 1053 | dualgpu_live=true |
| 6 | KV260 smoke test passed | Exp 1054 | smoke_test_passed=true |
| 7 | FoVer >= 500 pairs | Exp 1055 | n_total_pairs>=500 |
| 8 | Triple Integration all tiers | Exp 1056 | all_tier_skip_rates_nonzero=true |
| 9 | Best probe AUROC >= 0.72 | Exp 1057 | best_probe_auroc>=0.72 |
| 10 | FR-11 alpha_t closed live | Exp 1058 | fr11_loop_closed=true, inference_mode=live_gpu |
| 11 | WOPR Sudoku Space deployed | Exp 1059 | space_deployed=true |
| 12 | WOPR cultural cartridges shipped | Exps 1060+1061 | both honest_verdict=cartridge_shipped |
| 13 | Retro complete | Exp 1062 | honest_verdict written |

---

## Hardware requirements

| Experiment | GPU | FPGA | Notes |
|------------|-----|------|-------|
| 1050 | No | No | CPU-only test surgery |
| 1051 | No | No | Conductor code |
| 1052 | No | No | Conductor code |
| 1053 | Yes | No | torch CUDA install + live inference smoke test |
| 1054 | No | Yes (KV260 192.168.51.98) | SSH to board |
| 1055 | Yes | No | MetaQA with gemma-4-26B-A4B |
| 1056 | No | No | Pipeline cascade CPU |
| 1057 | Yes | No | gemma-4-31B-it-GGUF probe scoring |
| 1058 | Yes | No | Qwen3.6-35B-A3B live inference (gated on exp1053) |
| 1059 | No | No | Spaces Python deployment |
| 1060 | No | No | Pure animation cartridge |
| 1061 | No | No | Ising ground-state (CPU Ising sampler) |
| 1062 | No | No | Retro analysis |

---

## arxiv findings incorporated

From the 2026-04-29 scan:

- **arXiv 2512.20664** (Eidoku neuro-symbolic gate) — validates Carnot's verifier cascade
  architecture; cite in position paper Section 2 (related work)
- **arXiv 2510.13444** (Neural SOS) — future direction for SOS-KAN v3 (milestone .83+)
- **arXiv 2603.19562** (Neural Uncertainty Principle) — provides geometric grounding for why
  α_t > 0 is necessary; cited in Exp 1058 (Zenil alpha_t) theoretical framing
- **arXiv 2506.14590** (MTJ Ising machine) — Phase 2 hardware path; added to hardware wishlist
- **arXiv 2604.04636** (KAN crystal energies) — independent validation of KAN for energy functions;
  cite in position paper Section 4 (architecture)

---

## Process improvements vs .81

1. **Pre-test surgery FIRST:** exp1050 is unconditioned and runs before everything else.
   The .81 deadlock pattern (tests broken → fix blocked → tests stay broken) cannot recur
   if the pre-test fix is the unconditional first task.

2. **Respawn queue prevents silent retirement:** exp1050 seeds the queue with the 3 environmentally-
   retired experiments, and the conductor patch auto-classifies future retirements. Environmental
   retirements no longer silently disappear from the roadmap.

3. **WOPR gallery committed:** Known-issues.md mandated these for .82. They are in Phases 4+5
   with clear deliverables (deployed Space + JSON artifacts). Cannot be bumped again.

4. **GGUF pre-download before MetaQA:** exp1055 explicitly downloads the model before any
   MetaQA generation attempt. Silent stub-out with no error is now blocked.

5. **DualGPU gate on Zenil alpha_t:** exp1058 is gated on exp1053.dualgpu_live=true.
   The Zenil experiment explicitly refuses synthetic_fallback — if GPU is not live, it blocks
   cleanly rather than silently bootstrapping.
