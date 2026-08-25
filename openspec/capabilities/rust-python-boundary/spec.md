# Rust/Python Boundary Capability Specification

**Capability:** rust-python-boundary
**Status:** Draft

## Requirements

### REQ-RUSTPY-6194: Mode-Jump PyO3 Boundary Contract

Carnot MUST expose the Exp6194 fixed mode-jump sampler through PyO3 without
silent fallback. The boundary SHALL preserve Rust sampler semantics exactly:
typed construction, energy/proposal queries, one-step, multi-step, state
snapshot, state restore, serialization, deserialization, deterministic seed
replay, and explicit invalid-input errors.

Sub-requirements:
- REQ-RUSTPY-6194-TYPED-CONSTRUCTION: PyO3 SHALL expose typed configuration,
  state, and core classes for the mode-jump sampler.
- REQ-RUSTPY-6194-STEP-TRACE: A one-step PyO3 call SHALL return the same trace
  fields as the Rust kernel: proposal uniform, proposed label, acceptance
  uniform, current/proposed energies, proposal log probabilities, log
  acceptance, acceptance probability, accept decision, counters, RNG state, and
  after-state snapshot.
- REQ-RUSTPY-6194-MULTI-STEP: A multi-step PyO3 call SHALL return deterministic
  frequencies, acceptance counts, final state, and sample count for a bounded
  positive step budget.
- REQ-RUSTPY-6194-SERIALIZATION: Snapshot/restore and serialization/
  deserialization SHALL round-trip exactly and reject corrupt labels, counters,
  schemas, and strings with explicit `ValueError` exceptions.
- REQ-RUSTPY-6194-NO-FALLBACK: The binding SHALL not silently route to a
  Python implementation when Rust construction, stepping, or serialization
  fails.
- REQ-RUSTPY-6194-NO-HARDWARE: The boundary SHALL not create or imply any FPGA,
  TSU, CUDA, THRML scaling, latency, power, energy, or speedup claim.

### SCENARIO-RUSTPY-6194-BOUNDARY-PARITY: Python Calls Rust And Replays The Fixture

**Given** the frozen Exp6194 Python transition fixture and Rust/PyO3 classes
**When** Python constructs the Rust configuration and state, runs exact steps,
serializes/restores state, and runs a long chain
**Then** the returned traces and diagnostics match the Python fixture and
invalid inputs raise explicit errors instead of falling back.

## Implementation Status (REQ-RUSTPY-6194)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-RUSTPY-6194 | Planned (`crates/carnot-python/src/mode_jump.rs`, `crates/carnot-python/src/lib.rs`, `python/carnot/_rust_compat.py`) | Planned (`tests/python/test_experiment_6194_mode_jump_rust_pyo3_parity.py`) |

### REQ-RUSTPY-6550: Safety-Net Router ABI Parity Contract

Carnot MUST expose the compact Safety-Net request and routing decision contract
through PyO3. The ABI SHALL use only safe structural fields: schema version,
request ID, candidate hashes, split name, seed, frozen feature values, router
contract hash, exception-table hashes, and explicit forced fallback flags. It
SHALL NOT load models, parse natural language, inspect hidden states, use labels,
or claim release authority.

Sub-requirements:
- REQ-RUSTPY-6550-SCHEMA: The request and decision SHALL carry an explicit
  `carnot.safety_net.router_abi.v1` schema. Missing or stale schema values SHALL
  fail closed to native fallback with an explicit schema error.
- REQ-RUSTPY-6550-NUMERIC: Integer-like structural feature values SHALL
  normalize identically across Python and Rust. NaN, infinity, non-integer
  floats, and out-of-range integers SHALL fail closed to native fallback.
- REQ-RUSTPY-6550-SERIALIZATION: Python and Rust decisions SHALL serialize to
  identical canonical JSON bytes for the same input bytes.
- REQ-RUSTPY-6550-ERRORS: Malformed, null, unsupported, extra-key, forbidden,
  Unicode-candidate, and version-skew requests SHALL return the same error type
  and fallback reason in Python and Rust.
- REQ-RUSTPY-6550-PARITY: Supported requests SHALL match exactly for route,
  abstain, uncertainty bucket, exception hit, fallback reason, chosen order,
  request hash, and schema version.
- REQ-RUSTPY-6550-FALLBACK: Unsupported requests SHALL route to
  `native_exact_fallback` and keep fallback reachability true.
- REQ-RUSTPY-6550-ROLLBACK: The Python production adapter rollback path SHALL
  remain exact and independent of the Rust ABI.
- REQ-RUSTPY-6550-NO-AUTHORITY: The ABI SHALL not change accepted exact
  downstream results. Native exact verification remains release authority.

### SCENARIO-RUSTPY-6550-BOUNDARY-PARITY: Python And Rust Replay Identical Safety-Net Bytes

**Given** a frozen set of supported, boundary, exception, malformed, null,
extreme numeric, version-skew, unknown-feature, field-order, Unicode, and NaN
Safety-Net request bytes
**When** Python and PyO3 route each request through the compact ABI
**Then** supported rows are decision-equal and byte-equal, unsupported rows fail
closed to fallback with equal error semantics, exact downstream results remain
unchanged, and Python-only rollback disables routing exactly.

## Implementation Status (REQ-RUSTPY-6550)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-RUSTPY-6550 | Implemented (`python/carnot/pipeline/safety_net_abi.py`, `crates/carnot-python/src/safety_net.rs`, `python/carnot/experiment_6550_rust_pyo3_safety_net_parity.py`) | Implemented (`tests/python/test_safety_net_rust_pyo3_parity.py`, `tests/python/test_experiment_6550_rust_pyo3_safety_net_parity.py`) |

### REQ-RUSTPY-6564: Safety-Net Batch Router ABI Throughput Contract

Carnot MUST expose the Safety-Net compact router through a versioned PyO3
batch boundary. The batch boundary SHALL accept an ordered sequence of the same
raw request bytes used by the scalar ABI. It SHALL return one decision per
input in the same order. It SHALL preserve the V567 scalar ABI, schema version,
canonical decision bytes, fallback reasons, error classes, and exact downstream
result for every request.

Sub-requirements:
- REQ-RUSTPY-6564-BATCH-SCHEMA: The batch ABI SHALL use
  `carnot.safety_net.router_batch_abi.v1` as its batch contract and SHALL route
  each item through the unchanged scalar `carnot.safety_net.router_abi.v1`
  request contract.
- REQ-RUSTPY-6564-BATCH-PARITY: Python scalar, PyO3 scalar, and PyO3 batch
  decisions SHALL be byte-identical for supported, abstain, fallback,
  exception, malformed, and unsupported requests.
- REQ-RUSTPY-6564-BATCH-ERRORS: Malformed items SHALL fail closed per item
  without dropping later items or changing order.
- REQ-RUSTPY-6564-NO-SCOPE-CREEP: The batch boundary SHALL NOT move Z3, LLM
  inference, exact verification, natural-language extraction, or policy
  authority into Rust.

### SCENARIO-RUSTPY-6564-BATCH-ORDERED-PARITY: Batch Routing Preserves Scalar Bytes

**Given** the Exp6563 frozen production Safety-Net workload request bytes plus
malformed and unsupported ABI requests
**When** Python scalar, PyO3 scalar, and PyO3 batch routing process the same
ordered bytes
**Then** every decision, error type, fallback reason, request hash, downstream
result, and output byte string matches exactly, and the batch result order
matches the request order.

## Implementation Status (REQ-RUSTPY-6564)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-RUSTPY-6564 | Planned (`crates/carnot-python/src/safety_net.rs`, `python/carnot/experiment_6564_rust_pyo3_safety_net_nfr01.py`) | Planned (`tests/python/test_safety_net_rust_pyo3_parity.py`, `tests/python/test_experiment_6564_rust_pyo3_safety_net_nfr01.py`) |

### REQ-RUSTPY-6612: Spectral k-Block Ising PyO3 Parity Contract

Carnot MUST expose the reusable bounded block heat-bath kernel from
`carnot-samplers` through PyO3. Python and Rust SHALL accept the same coupling
matrix, fields, temperature, block membership, initial state, seed, burn-in,
and retained count. Both paths SHALL use the documented 64-bit LCG random
stream and return identical retained spins for matched descriptors.

The boundary SHALL reject non-square, asymmetric, non-finite, empty-block,
duplicate-spin, missing-spin, invalid-state, nonpositive-temperature, zero
retention, and count-overflow inputs. It SHALL not silently use a Python
fallback when the Rust module is absent or rejects an input. The boundary is a
CPU software interface. It SHALL make no attached-hardware performance claim.

### SCENARIO-RUSTPY-6612-MATCHED-CHAIN-PARITY

**Given** a frozen frustrated Ising fixture and one validated spin partition
**When** Python and PyO3 run the same seeded descriptor
**Then** retained samples, final state, RNG state, transitions, and spins
updated match exactly
**And** malformed descriptors raise explicit errors without fallback.

## Implementation Status (REQ-RUSTPY-6612)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-RUSTPY-6612 | Implemented (`crates/carnot-samplers/src/spectral_k_block.rs`, `crates/carnot-python/src/spectral_k_block.rs`) | Implemented (`crates/carnot-samplers/tests/spectral_k_block.rs`, `tests/python/samplers/test_spectral_k_block.py`) |
