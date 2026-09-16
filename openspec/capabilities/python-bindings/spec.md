# Python Bindings Capability Specification

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
