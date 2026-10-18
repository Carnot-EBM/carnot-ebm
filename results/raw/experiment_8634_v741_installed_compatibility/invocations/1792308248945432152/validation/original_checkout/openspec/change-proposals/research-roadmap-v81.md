# Research Roadmap v81 — Infrastructure Close-Out + Hardware Unblocks + Probe Ensemble

**Milestone:** 2026.04.81
**Planned experiments:** Exps 1039–1049 (11 experiments)
**Target wall time:** ~1,700 min (goal: below 2026.04.80's record low of 1,938 min)
**Designed:** 2026-04-29

---

## What milestone .80 proved

Milestone .80 was the infrastructure hardening cycle. Key outcomes:

**Hard wins:**
- Schema linter (Exp 1026): caught 2 prior_failures violations in .79 YAML; .80 = 0 violations
- Conductor supervisor (Exp 1027): deployed with 6 passing tests; heartbeat/orphan/log-handle monitors live
- Preflight v30 (Exp 1028): pre_test_fixed=True + manifests 786/641/906 finally retired (after 20 consecutive milestones)
- FR-11 SSD loop closed (Exp 1031): 100 training examples written end-to-end
- PPSEBM relay live (Exp 1032): relay_live=True, AUROC=0.6875

**Structural failures to fix in .81:**
1. **Gate coercion bug:** conductor gate evaluator read `pre_test_fixed: "True"` (string) not `True` (bool) — blocked Triple Integration 3 times before pre-test failures took over. Root cause: YAML/JSON coercion in gate check logic.
2. **Fastpath bootstrap skip:** `_deliverable_exists()` treated bootstrap stubs (`status: "running"`) as completed artifacts. Already implemented in `conductor-fastpath-bootstrap-skip.md`; needs .81 close-out.
3. **FoVer MetaQA generator not implemented:** n_metamorphic_candidates=0 in Exp 1029 (only 2.267s wall time — no real LLM work ran). FoVer at 85 pairs blocks ThinkPRM, GS-KAN, NK-KAEM.
4. **torch CPU-only build:** Exp 1035 confirmed nvidia-smi sees 2x RTX 3090 but torch 2.11.0+cpu cannot do live inference. DualGPU live blocked.
5. **KV260 bitstream format mismatch:** Exp 1037 guide identified exact fix — need `bootgen -arch zynqmp -process_bitstream bin -image carnot.bif` to convert `.bit` to `.bit.bin` format.
6. **Non-reproducible verdicts:** Exp 1031 produced `fr11_loop_closed` at 21:12Z and `carnot_filter_below_baseline` at 01:13Z from the same code. Verdict-reproducibility-audit is mandatory for .81.

---

## Architecture diagram

```
Phase 0: MANDATORY INFRASTRUCTURE (standalone, run first)
  ├── Exp 1039: Fastpath bootstrap skip + gate coercion fix  [model: opus]
  └── Exp 1040: Verdict reproducibility audit + seed discipline

Phase 1: HARDWARE UNBLOCKS (standalone, can run in parallel with Phase 0)
  ├── Exp 1041: KV260 bitstream format fix v7               [model: opus]
  └── Exp 1042: DualGPU ROCm/CUDA torch install v4         [model: opus]

Phase 2: CORPUS + CASCADE
  ├── Exp 1043: FoVer corpus expansion v3 (MetaQA generator fix)
  ├── Exp 1044: Triple Integration E2E v7         [gated on exp1039]
  └── Exp 1045: Probe ensemble v5: ThinkPRM+GS-KAN+NK-KAEM [gated on exp1043]

Phase 3: SELF-LEARNING + THEORY (FR-11 mandatory)
  ├── Exp 1046: Zenil alpha_t grounding + Phase-3 AND-composition k=5  [FR-11 mandatory]
  ├── Exp 1047: SOS-Integrated KAN (Exp 980 re-scope: type-level invariants)
  └── Exp 1048: Eval-metrics canonical + conductor self-heal wiring

Phase 4: RETRO
  └── Exp 1049: Milestone 2026.04.81 Retrospective
```

**Dependency graph:**

```
1039 (gate fix) ──────────────────────────────────────────→ 1044 (triple integration)
1043 (fover 500+) ────────────────────────────────────────→ 1045 (probes)
1039 ────────────────────────────────────────────────────→ 1049 (retro)
All others standalone (no upstream gate)
```

---

## Phase descriptions

### Phase 0: MANDATORY INFRASTRUCTURE

**Exp 1039 — Conductor Fastpath Bootstrap Skip + Gate Coercion Fix**

The `.80` wedge had two structural root causes:
1. `_deliverable_exists()` treated `status: "running"` stubs as completed. Fix already implemented in `scripts/research_conductor.py` (status-aware fast-path, 12 tests in `test_conductor_deliverable_status.py`). This experiment closes it out: merge, replay `.80` wedge (rm exp1028 artifact, restart conductor, confirm re-run), then retire exp1030's GATE_BLOCK history.
2. Gate evaluator coerced YAML `"True"` string to Python `True` inconsistently. Root cause: `_evaluate_gate()` used naive equality without normalization. Fix: normalize `True`/`"True"`/`"true"`/`1` before comparing.

Both fixes are operator-attention-reduction infrastructure. `model: opus` because it involves multi-step conductor code modification and replay.

**Exp 1040 — Verdict Reproducibility Audit + Seed Discipline**

Exp 1031 produced two different honest_verdicts from the same code path. This experiment:
1. Reruns last 5 flagship experiments (1031/1032/1029/1027/1026) to measure stability_rate.
2. Adds `random_seed` field to `experiment_template.py:build_result()` schema.
3. Adds reproducibility checksum (SHA of code + data + seed) to flagship results.
4. Target: stability_rate >= 0.80.

Critical for position paper credibility — the Zenil/Kinematic chain's empirical follow-ups must be verdict-stable.

### Phase 1: HARDWARE UNBLOCKS

**Exp 1041 — KV260 Bitstream Format Fix v7**

Exp 1037 guide identified the exact fix: `bootgen -arch zynqmp -process_bitstream bin -image carnot.bif` converts `.bit` to `.bit.bin` format required by Kria's `fpgautil`. SSH is confirmed working at 192.168.51.98. This experiment:
1. Runs bootgen on the host (or the board if bootgen not available locally).
2. Transfers the `.bit.bin` firmware to the board.
3. Re-runs `xmutil loadapp carnot_ising_v4`.
4. If successful: runs smoke test, verifies non-uniform energy distribution.

`model: opus` — hardware integration with multiple possible failure modes.

**Exp 1042 — DualGPU ROCm/CUDA torch Install v4**

Two viable paths:
- **Path A (ROCm):** `pip install torch==2.11.0+rocm7.2 --index-url https://download.pytorch.org/whl/rocm7.2` — installs torch with HIP support for the AMD Radeon 890M + ROCm environment.
- **Path B (CUDA):** The 2x RTX 3090 need CUDA-enabled torch. `pip install torch==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121`.
- **Path C (llama.cpp pivot):** If neither PyTorch path works, pivot to llama.cpp-only dual-GPU dispatch (LLAMA_SPLIT_MODE_ROW for 2-GPU tensor parallelism). Does not require torch CUDA.

`model: opus` — hardware integration, environment configuration.

### Phase 2: CORPUS + CASCADE

**Exp 1043 — FoVer Corpus Expansion v3 (MetaQA Generator Fix)**

Exp 1029 ran for 2.267 seconds and produced 0 MetaQA candidates because the generator was not actually implemented (it returned an empty list). This experiment fixes that:
1. Implements the MetaQA query generator: for each candidate CoT step, generate 5 paraphrase/negation/substitution questions using `unsloth/gemma-4-26B-A4B-it-GGUF`.
2. Also expands Z3 labeling to 1000+ MATH dataset problems (v2 only used 500).
3. Target: n_total_pairs >= 500, n_violation_pairs >= 50.

Standalone — does not gate on any prior experiment.

**Exp 1044 — Triple Integration E2E v7**

Same as exp1030 but with gate coercion fixed. Gates on `exp1039.gate_coercion_fixed=true`. The underlying pipeline cascade (ThinkPRM → SpilledEnergy → SC-Energy → Ising) is implemented and tested. Only the gate evaluation logic was broken.

**Exp 1045 — Probe Ensemble Training v5 (ThinkPRM + GS-KAN + NK-KAEM)**

All three probes failed in .80 due to insufficient training data (ThinkPRM AUROC=0.5 CI stub, GS-KAN below baseline 0.65 vs 0.6875, NK-KAEM diverged). Common root cause: CI stub models + insufficient real features. This experiment:
1. Gates on `exp1043.n_total_pairs >= 200`.
2. ThinkPRM: trains on expanded corpus using real `gemma-4-31B-it-GGUF` probe scoring (not CI stub).
3. GS-KAN: trains with proper FoVer corpus (G=4, INT8 quantized). Target AUROC >= 0.72.
4. NK-KAEM: warm-starts from Adam at G=4 for 20 epochs before switching to Newton-Kaczmarz. Per-layer LR decay. Target convergence_speedup >= 2.0.
5. Reports best_probe_auroc across the three.

### Phase 3: SELF-LEARNING + THEORY

**Exp 1046 — Zenil α_t Grounding + Phase-3 AND-composition k=5 (FR-11 MANDATORY)**

The FR-11 mandatory self-learning experiment for milestone .81. Based on:
- Zenil (2026): `inf_t α_t > 0` is the necessary condition for self-distillation convergence to truth.
- Carnot's energy function IS the α_t μ_P term (the external grounding signal).
- Phase-3 architecture (Deep Think Rounds 8-9): 6 base verifiers + AND-composition (k=5) is the deployable recipe for k_max=5 on KV260 hardware.

This experiment implements a subset of `zenil-grounded-self-distillation-deployable-stack.md`:
1. **Φ measurement module** (`python/carnot/eval/phi_test.py`): measures the Carnot energy function's contribution as α_t using the Zenil bound. For a batch of N self-distillation examples, α_t = fraction where Carnot's verdict changed the model's output vs temperature-only selection.
2. **AND-composition k=5 verifier**: composes 5 available verifiers (Z3 + 2×fuzz + 2×LLM/Mutant) into a combined verifier. Bypass rate suppression: ε→ε²cos(θ_F).
3. **FR-11 training examples**: generates 100+ examples where α_t > 0 (Carnot actually changed selection vs temperature baseline), writes to `data/fr11_zenil_distill_v1.jsonl`.
4. Uses `unsloth/Qwen3.6-35B-A3B-GGUF` for candidate generation.

Acceptance: `alpha_t_measured=True` (Φ module deployed), `fr11_examples_written >= 100`.

**Exp 1047 — SOS-Integrated KAN (Exp 980 Re-scope)**

Per `ops/known-issues.md` EXP-980-RE-SCOPING: standard MILP repair of KAEMEnergy monotonicity violations is the wrong framing. The correct approach is Sum-of-Squares re-parameterization:

`ψ(x) = c² + Σ_{i,j} (V·V^T)_{i,j} Φ_{i,j}(x)` where `ψ'(x) = SOS(B-splines)`.

Monotonicity and non-negativity become type-level invariants of the AST — no post-hoc MILP verification needed. This is a direct AUROC improvement path:
1. Implements `python/carnot/models/sos_kan.py`: `SOSKANEnergy` layer with SOS-on-derivative basis.
2. Trains on FoVer corpus (57-pair baseline; expanded if Exp 1043 ran).
3. Compares AUROC vs KAEMEnergy baseline.
4. FPGA resource estimate: SOS-on-derivative reduces DSP48 usage vs floating-point spline (shared SOS basis computation).

No GPU required. `JAX_PLATFORMS=cpu`.

**Exp 1048 — Eval-Metrics Canonical + Conductor Self-Heal Wiring**

Pre-shipped in .80 (metrics.py, audit_metric_provenance.py, conductor_commit_watchdog.sh). What remains for .81:
1. Wire `carnot.eval.metrics` AUROC into the conductor's self-heal loop: flag any new experiment result where computed AUROC deviates by >0.1 from raw (checks for inverted-AUROC bug class).
2. Add `metrics_used` field to `build_result()` call sites in experiment_template.py.
3. Run `python3 scripts/audit_metric_provenance.py` on last 10 experiment results; report any using legacy buggy implementations.
4. Validate `model: opus` field is parsed correctly from .81 roadmap YAML.

### Phase 4: RETRO

**Exp 1049 — Milestone 2026.04.81 Retrospective**

Evaluates all 13 success criteria, updates ops/ docs, writes `results/operational_retro_2026_04_81.json`.

---

## Hardware requirements

| Experiment | GPU | Rationale |
|-----------|-----|-----------|
| 1039-1041 | No | Infrastructure/hardware scripts |
| 1042 | Requires RTX 3090 or ROCm | DualGPU install validation |
| 1043 | Yes (ROCm/CUDA) | gemma-4-26B MetaQA generation |
| 1044 | No | Pipeline dry-run with synthetic |
| 1045 | Yes | ThinkPRM real inference scoring |
| 1046 | Yes | Qwen3.6-35B candidate generation |
| 1047 | No | JAX CPU SOS-KAN training |
| 1048 | No | Metrics wiring |
| 1049 | No | Retro analysis |

GPU experiments use: `sg render -c 'JAX_PLATFORMS=cpu TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python scripts/...'`

---

## SOTA model requirements (CLAUDE.md mandatory)

All GPU experiments include at least one of:
- `unsloth/Qwen3.6-35B-A3B-GGUF` (Exps 1042, 1046)
- `unsloth/gemma-4-31B-it-GGUF` (Exps 1042, 1045)
- `unsloth/gemma-4-26B-A4B-it-GGUF` (Exps 1043, 1045)

Small models (Qwen3.5-0.8B, Gemma4-E4B) acceptable only for CPU smoke-tests.

---

## Success criteria

| # | Criterion | Experiment | Field | Target |
|---|-----------|-----------|-------|--------|
| 1 | Gate coercion fixed | 1039 | gate_coercion_fixed | true |
| 2 | Fastpath bootstrap wedge replay clean | 1039 | wedge_replay_clean | true |
| 3 | Verdict stability >= 80% | 1040 | stability_rate | >= 0.80 |
| 4 | Seed discipline deployed | 1040 | seed_discipline_deployed | true |
| 5 | KV260 bitstream loaded | 1041 | bitstream_loaded | true |
| 6 | DualGPU live inference active | 1042 | dualgpu_live | true |
| 7 | FoVer >= 500 pairs | 1043 | n_total_pairs | >= 500 |
| 8 | Triple Integration all tiers fire | 1044 | all_tier_skip_rates_nonzero | true |
| 9 | Best probe AUROC >= 0.72 | 1045 | best_probe_auroc | >= 0.72 |
| 10 | α_t measurement deployed | 1046 | alpha_t_measured | true |
| 11 | SOS-KAN AUROC no regression | 1047 | auroc_no_regression | true |
| 12 | Eval metrics wired | 1048 | eval_metrics_wired | true |
| 13 | Retro complete | 1049 | honest_verdict | milestone_*_of_13* |

---

## Decentralization implications

- Hardware experiments (KV260, DualGPU) use locally-owned hardware — no closed vendor dependency.
- SOTA models are all open-weight GGUFs downloadable from HuggingFace — satisfies rule 1.
- SOS-KAN is purely local JAX computation.
- Zenil α_t grounding experiment uses local models only (Qwen3.6-35B-A3B-GGUF).
- No new closed-weight integrations introduced.
