# Carnot — E2E Test Plan

**Last Updated:** 2026-05-10

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

### E2E-004: Serialization Cross-Language

**Objective:** Verify that a model saved from Rust can be loaded in Python and vice versa.

**Steps:**
1. Save model parameters from Rust via safetensors
2. Load in Python via safetensors
3. Verify identical energy computation

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
