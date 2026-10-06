# Carnot — E2E Test Plan

**Last Updated:** 2026-10-06

## E2E Test Strategy

Energy Based Models are mathematical constructs — E2E verification means running the full training + sampling pipeline and verifying statistical properties of the outputs.

### E2E-001: Ising Model Training + Sampling (Rust)

**Objective:** Verify that training an Ising model with CD-1 and sampling with Langevin dynamics produces samples from the correct distribution.

**Steps:**
1. Create Ising model with known coupling matrix (e.g., 2D lattice with J=1)
2. Generate synthetic data from known Boltzmann distribution
3. Train model with CD-1 for N steps
4. Sample from trained model with Langevin dynamics
5. Verify sample statistics match training data statistics (mean, covariance)

**Pass criteria:** Sample mean within 0.2 of training data mean; sample covariance Frobenius norm error < 0.5.

### E2E-002: Ising Model Training + Sampling (Python/JAX)

Same as E2E-001 but using the Python/JAX implementation. Cross-validate that Rust and Python produce statistically equivalent results.

### E2E-003: PyO3 Binding Round-Trip

**Objective:** Verify that a model created in Rust, exposed via PyO3, and called from Python produces correct results.

**Steps:**
1. Create Ising model in Rust via PyO3
2. Compute energy for test inputs from Python
3. Compare with pure-Python JAX computation
4. Verify zero-copy array transfer for contiguous arrays

**Operational Note (2026-09-16, Exp7339):** E2E-003 also crossed the actual
task-owned CPython 3.12 PyO3 extension for immutable schedule constraints. The
round trip preserved ordered term energies and exact errors with zero parity
mismatches; source-only and build-only checks did not satisfy the gate.

### E2E-004: Serialization Cross-Language

**Objective:** Verify that a model saved from Rust can be loaded in Python and vice versa.

**Steps:**
1. Save model parameters from Rust via safetensors
2. Load in Python via safetensors
3. Verify identical energy computation

**Operational Note (2026-09-26, Exp7710):** The typed record service also
checks E2E-003 and E2E-004 through a task-owned PyO3 extension. Its focused
tests compare 128 Python and Rust decisions, then reopen pending native state
after both normal destruction and a hard child exit. The test fixture builds
the current extension for standalone runs when no producer path is supplied.

### E2E-005: Packaged Code Verification Generate-Verify-Repair

**Objective:** Verify that the packaged end-user code-verification surfaces can
take an LLM-style generated Python candidate, detect a prompt-implied bug with
PBT, and confirm the repaired candidate cleanly.

**Steps:**
1. Build a generated candidate function body from a HumanEval-style prompt
2. Run the official weak harness to confirm the buggy candidate can still pass
3. Verify the candidate through the packaged code-verification path with
   additive Hypothesis-backed PBT
4. Use the packaged repair feedback to produce a repaired candidate
5. Re-run packaged verification and the official harness on the repaired code

**Pass criteria:** The initial candidate passes the weak harness but fails the
packaged verifier, and the repaired candidate passes both packaged verification
and the official harness.

### E2E-006: EBRM Trace Scorer CPU/KV260 Verification

**Objective:** Verify that extracted logical traces are scored by the CPU EBRM
scorer and the KV260 q=3 Potts backend with matching energy results and
auditable per-case provenance.

**Spec refs:** `REQ-VERIFY-1656`, `SCENARIO-VERIFY-1656`,
`REQ-VERIFY-1657`, `SCENARIO-VERIFY-1657`, `REQ-VERIFY-1658`,
`SCENARIO-VERIFY-1658`.

**Source artifacts:** `results/experiment_1656_ebrm_trace_scorer.json`,
`results/experiment_1657_kv260_ebrm_binding.json`,
`results/experiment_1658_hw_eval.json`.

**Steps:**
1. Confirm the Exp 1656 CPU scorer artifact is complete, uses continuous
   energy, and reports `score_accuracy >= 0.8`.
2. Confirm the Exp 1657 KV260 binding artifact is complete, uses q=3 Potts
   states, and records whether hardware execution or software fallback was
   used.
3. Run or inspect Exp 1658 on bounded local SOTA output rows and compare CPU
   and KV260 energies over the same trace batch.
4. Verify every case score includes CPU energy, KV260 energy, absolute score
   delta, backend provenance, and Potts state metadata.

**Pass criteria:** Exp 1656, Exp 1657, and Exp 1658 artifacts are complete;
CPU/KV260 `max_score_delta <= 1e-6`; CPU and KV260 scoring accuracy match;
`scoring_delta_within_tolerance=true`; and no hardware execution claim is made
unless authenticated hardware evidence is present.

### E2E-007: SMGI Certified Update Verification

**Objective:** Verify that SMGI policy and memory updates become reusable only
when CerCE certificate evidence, replay retention, SessionMemory hash changes,
and model-weight immutability gates all pass.

**Spec refs:** `REQ-LEARN-1659`, `SCENARIO-LEARN-1659`,
`SCENARIO-LEARN-1660`.

**Source artifacts:** `results/experiment_1659_smgi_certified_updates.json`.

**Steps:**
1. Confirm the Exp 1659 artifact is complete and
   `continuous_self_learning_task=true`.
2. Verify the CerCE ledger gates report `accepted_violation_count=0`,
   `false_accept_delta <= 0`, `soundness_mistakes=0`, and
   `nonforgetting_certificate_rate=1.0`.
3. Inspect every certified update for matching certificate ID, present and
   changed SessionMemory hashes, full replay retention, zero replay failures,
   provenance, and `no_model_weight_mutation=true`.
4. Verify unsafe candidates remain in `rejected_updates` and never contribute
   to `certified_update_success=true`.

**Pass criteria:** `smgi_certified_update_ready=true`,
`certified_update_success=true`, at least one certified update is present,
all certified updates pass replay and hash gates, and no update mutates model
weights.

### E2E-008: CLaRa-V Continuous Latent Space Evaluation

**Objective:** Verify that CLaRa-V continuous latent variables map to ContinuousEBM instances and can be correctly evaluated for constraints and energy.

**Spec refs:** `REQ-KONA-040`, `SCENARIO-KONA-041`.

**Source artifacts:** `results/experiment_2000_e2e_pipeline.json`.

**Steps:**
1. Instantiate a ContinuousLatentState with constraints.
2. Instantiate a ContinuousEBM.
3. Evaluate the continuous latent state energies with the EBM.
4. Ensure ContinuousLatentState can project using the DouglasRachfordPiNetLayer.

**Pass criteria:** ContinuousLatentState initialization succeeds, energy is evaluated correctly using ContinuousEBM, PiNet projection runs without error, and output coordinates represent the applied constraints.
### E2E-009: Cross-call ARC induction memory (CPU)

Spec refs: REQ-ARC-WMTE-7040, REQ-ARC-WMTE-7041, REQ-ARC-WMTE-7042.

Run `tests/python/test_arc_induction_state_persistence.py` with the worktree
PYTHONPATH, `--no-cov`, and a private `--basetemp`. Its scripted HTTP transport
drives the real scored policy, local generator, prompt builder, and engine writer.
Both induction branches must deliver prior work on their second attempt. The
default arm must keep identical payloads. The offline twin must dispatch `e3`
to the same scored policy and eval runner, writing its receipt to a temporary path.

Also run the offline twin with `--mechanism e3 --game r11l --max-actions 12`
and an output path in `/tmp`. Set `CARNOT_ARC_DISABLE_INDUCTION=1` for this
CPU environment smoke. This confirms environment plumbing, not memory efficacy.
Any action-efficiency claim requires a separately authorized local-model A/B.

### E2E-010: Local grammar tool transport (CPU)

Spec refs: REQ-ARC-WMTE-7043, REQ-ARC-WMTE-7044, REQ-ARC-WMTE-7045.

Run `tests/python/test_arc_tool_grammar_transport.py` with worktree PYTHONPATH,
`--no-cov`, `-n 0` and a private `--basetemp`. Drive scored primary, bounded
refinement, repair and offline CLI construction through actual HTTP construction,
tool dispatch, feedback and engine writing using scripted completions. Default
requests must be preserved; invalid replies and early failures must be diagnosed.

For real serving confirmation, use the CPU probe sources and exact commands in
`docs/research-notes/local-serving-validation-transcript-2026-09-05.json`. Preserve
negative results. Require raw server output, parse/dispatch observations and a
no-grammar control before claiming constrained transport. Successful synthesis is
a separate outcome: the 2026-09-05 full-loop trial parsed two calls but omitted
required code and wrote no engine; the nested-copy trial truncated. This passes
the transport/error-feedback scope and supplies no ARC efficacy evidence.

Also retain E2E-009's real offline environment smoke. It is LLM-off plumbing,
not a grammar or ARC improvement measurement. Any efficacy claim requires a
separate local-model trial; no default enablement follows from these checks.

### E2E-011: ARC decision telemetry parity (CPU)

Spec ref: REQ-ARC-WMTE-7465.

Run `tests/python/test_arc_decision_telemetry.py` with the worktree
`PYTHONPATH`, `--no-cov`, and `-n0`. The scripted fake environment must drive
the real `E3AgentPolicy` once with telemetry off and once with it on. Actions,
existing provenance, fake model calls, environment calls, and random state
must match. The enabled run must write bounded JSONL to `tmp_path`.

This check uses no real game, model, network, or GPU.

### E2E-012: E6 exclusive timing parity and reduction gates (CPU)
Spec refs: REQ-ARC-WMTE-7491 and REQ-ARC-WMTE-7492.
Run `tests/python/test_experiment_7491_e6_timed_live_profile.py`,
`tests/python/test_experiment_7492_e6_timed_cost_profile.py`, and
`tests/python/test_arc_decision_telemetry.py` with the worktree `PYTHONPATH`,
`--no-cov`, `-n 0`, and a private `--basetemp`. The enabled and disabled
paths must preserve actions, calls, provenance, environment work, and random
state. Nested exclusive spans must reconcile. Token use must join by request
ID. Failed gates must suppress every numeric share.
This check uses no real game, model, network, or GPU. The live GPU run is a
separate Stage 2 operation.

### E2E-013: B2 induction-attempt outcome telemetry (CPU)
Spec ref: REQ-ARC-WMTE-7530.
Run `tests/python/test_arc_decision_telemetry.py`,
`tests/python/test_experiment_7491_e6_timed_live_profile.py`,
`tests/python/test_experiment_7531_b2_induction_gate_measurement.py`, and
`tests/python/test_semif_arc_readout_eval.py` with the worktree `PYTHONPATH`,
`--no-cov`, `-n 0`, and a private `--basetemp`. The off and on paths must
preserve actions, calls, provenance, environment work, and random state. Fired
attempt IDs must join token, timing, verifier, and bounded-progress fields.
Episode-end censoring and the 32-action closure must retain every attempt. The
reducer must suppress numeric gate claims below either sample floor and must
report a no-headroom oracle result honestly.
This CPU check uses scripted inputs. Experiment 7531 is the separate live GPU
check. It completed 56 episodes and retained the 88 budget-excluded schedule
rows as unstarted rather than dropping them.

### E2E-014: Exp7770 Qwen fixture transport and terminal replay (CPU)

Spec ref: REQ-REPORT-7770. Run
`tests/python/test_experiment_7770_v676_qwen_runner_qualification.py` with
`-n 0 --no-cov -o addopts=`. The test drives the real CLI, a local HTTP peer,
strict schema rejection, timeout and malformed replies, then cold replay.
Run the CLI with `--cold-replay results/experiment_7770_v676_qwen_runner_qualification.json`
and check the result with `scripts/adversarial_verify.py --json` and
`scripts/verdict_row_consistency_lint.py --strict`. Require 48 calls from 24
families and zero model loads. Runner readiness requires a passing full Python
suite receipt; a collection failure keeps readiness zero.

### E2E-015: Exp7852 public source fixture (CPU)

Spec ref: REQ-REPORT-7852. Run
`pytest -n 0 -o addopts= --no-cov -q tests/python/test_source_boundary_7852.py`.
The test invokes the real script-path CLI with a private public-only fixture,
cold-replays its feature shard, and confirms that a row containing an evaluator
label fails the same CLI gate. Both routes must work without an ambient
`PYTHONPATH` that includes the repository root.

### E2E-016: Exp7868 intervention fixture and cold replay (CPU)

Spec ref: REQ-REPORT-7868-V683. Run
`python scripts/experiments/experiment_7868_v683_intervention_protocol.py --date 20260929 --fixture-e2e /tmp/carnot-7868-fixture.json`, then use
`--cold-replay /tmp/carnot-7868-fixture.json` through the same CLI. Require 24
independent families, zero model calls, and a passing replay. The focused
`tests/python/test_experiment_7868_v683_intervention_protocol.py` exercises
checkpoint reuse, code-bound identity drift, fixture hash drift, and the
terminal validator paths; combined unit and CLI coverage must reach 100% for
the two Exp7868 modules and its CLI.

### E2E-017: Exp7874 supervisor delta CLI (CPU)

Spec ref: REQ-REPORT-7874. Run
`pytest -n 0 -o addopts= --no-cov -q tests/python/test_arc_supervisor_delta_7874.py`.
The direct script-path CLI must reduce a private empty ledger, reject missing
required arguments, and cold-replay a zero-firing candidate. A candidate that
claims a firing without receipts must fail replay. No game or model is run.

### E2E-018: V685 authority lifecycle CLI (CPU)

Spec ref: REQ-REPORT-7891-V685. Run
`pytest -n 0 -o addopts= --no-cov -q tests/python/test_experiment_7891_v685_authority_lifecycle.py`,
then run `python scripts/experiments/experiment_7891_v685_authority_lifecycle.py --date 20260929`.
The private CLI test must accept matching activated authority, reject a changed
candidate during cold replay, and execute the manifest's negative replay with
exit 1. The current CLI must pass all required validation receipts, all twelve
authority mutations, adversarial verification, strict row consistency, and a
cold replay of the published artifact. It makes no model or science claim.

### E2E-019: Exp7942 direct sentence annotation replay (CPU)

Spec ref: REQ-REPORT-7942, SCENARIO-REPORT-7942-DIRECT-CLI and
SCENARIO-REPORT-7942-COVERAGE-WORKSPACE. Run
`pytest -n 0 -o addopts= --no-cov -q tests/python/test_experiment_7942_v689_sentence_labels.py`.
Require private fixture publication, external blocking, date rejection, valid
replay and aggregate-tamper rejection. The replay regression invokes the actual
script from outside the checkout with `PYTHONPATH` removed; valid replay reports
`replay_passed`, and tampered aggregates exit one with `reduction_drift`.
The frozen fixture command must also save and combine real coverage in private
scratch without the pytest child artifact guard redirecting its data writes.

### E2E-020: Exp8206 measured hard exit and exact resume (CPU)

Spec ref: REQ-VERIFY-8206, SCENARIO-VERIFY-8206-CHILD and
SCENARIO-REPORT-8206-CLI. Run
`pytest -n 0 -o addopts= --no-cov -q tests/python/test_hard_exit_learning_qualification_8206.py`.
Private children invoke the unchanged legacy trainer directly under a frozen
Coverage.py `patch=_exit` configuration before import. Require uninterrupted
exit0, genuine exits73 at slots90/170, exact resumed complete states, pending
predictions saved before release, and measured save/flush/exit statements.
The same tests exercise private CLI publication, external blocking, date
rejection, cold replay and rehashed tamper rejection. Combine only invocation
unit/child shards and require100 percent newly owned statements.

### E2E-021: Exp8210 frozen audit replay CLI (CPU)

Spec refs: REQ-REPORT-8210, SCENARIO-REPORT-8210-CLI and
SCENARIO-REPORT-8210-MANIFEST. Run
`pytest -n 0 -o addopts= --no-cov -q tests/python/test_restricted_decision_audit_8210.py`.
The standalone manifest must freeze the audit's own CLI and owned paths, then
restore upstream bindings. Execute its frozen replay command from outside the
checkout without `PYTHONPATH`: valid private evidence passes and rehashed
aggregate tampering fails. Natural cached evidence retains all 128 slots;
failed owned validation disqualifies. Combine unit and real CLI coverage and
require 100 percent statements in the two audit modules and direct CLI.
