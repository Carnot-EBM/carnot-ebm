# Python Bindings Capability Specification

### REQ-PYBIND-7710: Existing PyO3 service exposes durable typed record calls

The existing `RustPortableRecalibrationService` SHALL add opt-in record predict,
feedback and state summary calls. It SHALL call Rust record logic, validate the
Exp7700-derived typed feature schema and explicit binary-energy parameters,
and persist pending and processed event identities for restart. E2E-003 SHALL
cross the loaded extension and E2E-004 SHALL reload the exact native state.
No default consumer or natural-language parser change follows.

### REQ-PYBIND-7723: Recheck the unchanged typed record ABI and durability

The V672 qualification SHALL load the private extension from its measured path,
record its binary hash and ABI, and exercise the existing fixed typed-record
methods. Normal restart and hard child exit before and after feedback SHALL
recover the expected pending and processed identities. The API scope SHALL
exclude the later source alignment model and remain opt-in.

#### SCENARIO-PYBIND-7723-RESTART

A real PyO3 service persists a prediction and exactly one feedback release
through both clean reopen and abrupt child termination.

#### SCENARIO-PYBIND-7710-ROUNDTRIP

A private task-owned extension accepts a typed record payload, returns its
normalized probability and action, and preserves feedback state after reload.

**Capability:** python-bindings
**Status:** Implemented

## Requirements

### REQ-PYBIND-7339: Compiled Schedule Constraints SHALL Use An In-Process Batch Boundary

`carnot._rust` SHALL expose an immutable compiled schedule-constraint object.
Construction SHALL copy ordered constraint terms into Rust-owned memory. The
object SHALL expose one batch method. The method SHALL accept ordinary Python
request objects and return ordinary Python result objects in the same order.
It SHALL not use JSON or a subprocess inside the call.

The binding SHALL reuse `carnot_constraints::evaluate_schedule_batch`. It
SHALL preserve Python evaluator errors for malformed schemas, versions, integer
types, slot domains, schedules, assignments, activities, and overflows. Boolean
values SHALL not count as integers. Values outside signed 64-bit range SHALL
fail with the same request error as the Python evaluator. Each result SHALL
include validity, feasibility, oracle-certificate status, total energy, ordered
term rows, and the first error.

The compiled object SHALL own its constraints. Mutating the constructor input
after construction SHALL not change later evaluations. Each call SHALL return
detached Python containers. Mutating one result SHALL not change Rust state or
a later result. A batch conversion error SHALL fail explicitly. The binding
SHALL not fall back to Python.

The experiment SHALL build `carnot-python` for `sys.executable` in an isolated
Cargo target. It SHALL load a task-owned copy by exact path. It SHALL not install
or overwrite the shared worktree extension. The artifact SHALL bind the loaded
file, extension suffix, SOABI, Python version, build output, binary hash, and
the schedule and binding source hashes.

#### SCENARIO-PYBIND-7339-ROUNDTRIP: Schedule Energy Crosses PyO3 Exactly

Given immutable ordered constraints and valid or malformed schedule requests,
When Python calls the imported native batch method,
Then its output matches the Python evaluator field for field,
And ordered term energies survive the input-to-energy-to-output round trip.

**Spec traces:** REQ-PYBIND-7339

#### SCENARIO-PYBIND-7339-MUTATION: Caller Mutation Cannot Change Native State

Given a compiled evaluator and caller-owned input and output containers,
When the caller changes those containers after a completed call,
Then later native results remain unchanged,
And no Python fallback supplies the result.

**Spec traces:** REQ-PYBIND-7339

#### SCENARIO-PYBIND-7339-E2E003: The Actual Extension Completes The Binding Round Trip

Given the interpreter-specific extension loaded from the task-owned path,
When Python submits schedules and receives energies and decisions,
Then the results match the pure-Python control,
And a source-only or build-only check cannot satisfy E2E-003.

**Spec traces:** REQ-PYBIND-7339

## Implementation Status (REQ-PYBIND-7339)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-PYBIND-7339 and SCENARIO-PYBIND-7339-* | Implemented in `crates/carnot-python/src/schedule.rs` and `python/carnot/experiment_7339_v644_native_binding.py`. | `tests/python/test_experiment_7339_v644_native_binding.py`, the focused binding unit test, and the real imported-extension replay verify the boundary. |

### REQ-PYBIND-7340: Native Timing SHALL Include Python Conversion And Result Ownership

Exp7340 SHALL benchmark the existing `RustCompiledScheduleEvaluator` through
its ordinary Python call boundary. Native elapsed time SHALL start before
per-call Python request detachment and end only after detached ordinary Python
result dictionaries are available. Constraint compilation and cold extension
import SHALL be measured separately and charged to amortization exactly once.
The timed call SHALL not substitute a Rust-only kernel timer.

The Python control, compiled native evaluator, and persistent JSON service
SHALL receive semantically equivalent requests in each randomized paired
block. Each block SHALL retain exact output hashes and require zero parity
mismatches. E2E-003 SHALL replay the benchmarked loaded extension rather than
accept a source, build, or stale binary identity.

#### SCENARIO-PYBIND-7340-BOUNDARY: Timing Returns Detached Python Results

Given a compiled evaluator and an equivalent Python control batch,
When a timed native repetition converts requests, evaluates them, and converts results,
Then elapsed time covers the complete ordinary-Python boundary,
And the resulting output hash matches the control and persistent-service hashes.

**Spec traces:** REQ-PYBIND-7340

#### SCENARIO-PYBIND-7340-E2E003: The Measured Binary Is The Replayed Binary

Given the Exp7339-declared extension hash and current source identities,
When Exp7340 performs its round trip and terminal reduction,
Then the loaded module hash and result parity are rechecked,
And a stale, missing, or differently built extension blocks score consumption.

**Spec traces:** REQ-PYBIND-7340

## Implementation Status (REQ-PYBIND-7340)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-PYBIND-7340 and SCENARIO-PYBIND-7340-* | Implemented in the Exp7340 complete-boundary measurement; no binding code change is required. | `tests/python/test_experiment_7340_v644_native_cost.py` and measured E2E-003 evidence cover the boundary. |

### REQ-PYBIND-8027: Opt-in calibrated numerical update

The prototype SHALL port only the frozen nine-input scale/cubic basis, calibrated
110-coefficient sparse update and JSON lazy-decay state. Reject nonfinite inputs,
malformed geometry, invalid labels and shape mismatches before mutation. Preserve
exact eligibility masks and typed action ties. Do not change production defaults.

#### SCENARIO-PYBIND-8027-PARITY

Tests SHALL first fail without the implementation. The actual task-owned loaded
PyO3 binary SHALL replay every natural update and at least 256 separate boundary
and random fixtures. Float64 probability and effective coefficient errors SHALL
be at most 1e-10. Disclose non-bitwise arithmetic. Serialized decay state SHALL
survive a fresh-process restart. E2E-003 and E2E-004 require binary hashes.

REQ-PYBIND-8027 implementation: `crates/carnot-core/src/numerical_update_8027.rs`
owns float64 geometry and sparse arithmetic. The matching carnot-python module
exposes explicit construction, feature design, update, effective coefficients
and serialized lazy state. Traced Rust and Python tests exercise the actual
extension. LLVM source-statement receipts retain generated PyO3 attribute
expansions separately; Python and CLI statements are measured with coverage.py.
Final numerical readiness is determined by the Exp8027 artifact gates.

#### SCENARIO-PYBIND-8027-SHUTDOWN

Repeated construction and publication SHALL preserve already loaded shared
library mappings. Copy a replacement binary to a new inode and atomically
replace its destination; never truncate an imported extension in place.
A fresh Python process SHALL load the same destination twice, execute the
numerical entrypoint, and exit normally. Failed copies SHALL preserve the
previous binary bytes and clean up temporary files.

Implemented by `copy_extension` in the Exp8027 producer for build and evidence
copies. The fresh-process repeated-load regression fails with SIGSEGV before
the repair and exits zero afterward. The original numerical Rust implementation
and all 13 original Python test/helper definitions remain intact. New regressions
also retain the previous open inode and verify cleanup after a partial copy.
