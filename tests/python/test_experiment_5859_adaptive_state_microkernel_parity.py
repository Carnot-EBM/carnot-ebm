"""Exp5859 adaptive-state microkernel parity tests.

Spec refs: REQ-LEARN-5859, SCENARIO-LEARN-5859-PRECONDITIONS,
SCENARIO-LEARN-5859-OPERATION-PARITY,
SCENARIO-LEARN-5859-STATE-HASH-ROUNDTRIP,
SCENARIO-LEARN-5859-FAIL-CLOSED.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from carnot import adaptive_state as mod


REPO = Path(__file__).resolve().parents[2]
SELF_LEARNING_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"
PYTEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5859_adaptive_state_microkernel_parity.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/adaptive_state.py "
    "-m pytest tests/python/test_experiment_5859_adaptive_state_microkernel_parity.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/adaptive_state.py --fail-under=100"
)
RUST_COMMAND = "cargo test -p carnot-core adaptive_state --lib"
BINDING_COMMAND = (
    "PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build -p carnot-python && "
    "cp target/debug/libcarnot_python.so "
    "python/carnot/_rust$(.venv/bin/python -c "
    "\"import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))\")"
)
RUSTFMT_COMMAND = (
    "rustfmt --check crates/carnot-core/src/adaptive_state.rs "
    "crates/carnot-python/src/adaptive_state.rs"
)
RUFF_COMMAND = (
    ".venv/bin/ruff check python/carnot/adaptive_state.py "
    "tests/python/test_experiment_5859_adaptive_state_microkernel_parity.py"
)
CLIPPY_CORE_COMMAND = "cargo clippy -p carnot-core --lib -- -D warnings"
CLIPPY_BINDING_COMMAND = (
    "cargo clippy -p carnot-python --lib -- -D warnings -A unused-imports "
    "-A deprecated -A clippy::type-complexity -A clippy::needless-range-loop "
    "-A clippy::too-many-arguments"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5859_adaptive_state_microkernel_parity.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5859_adaptive_state_microkernel_parity.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    '.venv/bin/python -c "from pathlib import Path; '
    "assert Path('scripts/research_conductor.py').exists()\""
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
TEST_COMMANDS = [
    PYTEST_COMMAND,
    COVERAGE_COMMAND,
    RUST_COMMAND,
    BINDING_COMMAND,
    RUSTFMT_COMMAND,
    RUFF_COMMAND,
    CLIPPY_CORE_COMMAND,
    CLIPPY_BINDING_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
    FULL_TEST_COMMAND,
]
TEST_EXIT_CODES = {command: 0 for command in TEST_COMMANDS}


def _run_op(kernel: object, operation: dict[str, object]) -> dict[str, object]:
    name = operation["op"]
    if name == "apply_event":
        return kernel.apply_event(operation["event"])  # type: ignore[attr-defined]
    if name == "acquire_core":
        return kernel.acquire_core(operation["event_id"])  # type: ignore[attr-defined]
    if name == "quarantine":
        return kernel.quarantine(operation["event_id"], operation["reason_code"])  # type: ignore[attr-defined]
    if name == "promote":
        return kernel.promote(operation["event_id"])  # type: ignore[attr-defined]
    if name == "select_replay":
        return kernel.select_replay(operation["limit"])  # type: ignore[attr-defined]
    if name == "roll_back":
        return kernel.roll_back(operation["version_id"])  # type: ignore[attr-defined]
    raise AssertionError(f"unknown test operation: {name}")


def _assert_kernel_equal(py_kernel: mod.AdaptiveStateKernel, rust_kernel: object) -> None:
    assert rust_kernel.canonical_state_json() == py_kernel.canonical_state_json()
    assert rust_kernel.canonical_state_hash() == py_kernel.canonical_state_hash()
    assert bytes(rust_kernel.serialize()) == py_kernel.serialize()


def test_req_learn_5859_spec_declares_bounded_abi() -> None:
    """REQ-LEARN-5859: OpenSpec preregisters the bounded parity contract."""

    section = SELF_LEARNING_SPEC.read_text(encoding="utf-8")
    section = section[section.index("## REQ-LEARN-5859") :]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-LEARN-5859",
        "SCENARIO-LEARN-5859-PRECONDITIONS",
        "SCENARIO-LEARN-5859-OPERATION-PARITY",
        "SCENARIO-LEARN-5859-STATE-HASH-ROUNDTRIP",
        "SCENARIO-LEARN-5859-FAIL-CLOSED",
        "python/carnot/adaptive_state.py",
        "crates/carnot-core/src/adaptive_state.rs",
        "RustAdaptiveStateKernel",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`adaptive_state_microkernel_ready_score`",
    ):
        assert marker in section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_learn_5859_python_reference_roundtrip_and_capacity() -> None:
    """SCENARIO-LEARN-5859-STATE-HASH-ROUNDTRIP: Python owns readable semantics."""

    kernel = mod.AdaptiveStateKernel(capacity=2, history_capacity=16)
    rollback_version = 0
    for operation in mod.deterministic_fixture_trace():
        result = _run_op(kernel, operation)
        assert result["accepted"] is True
        if operation["op"] == "promote" and operation["event_id"] == "evt-0004":
            rollback_version = int(result["version_id"])

    assert kernel.select_replay(4)["selected_replay"] == ["evt-0005", "evt-0004"]
    assert kernel.canonical_state()["evicted"] == [
        {"event_id": "evt-0001", "version_id": 12}
    ]
    checkpoint = kernel.serialize()
    restored = mod.AdaptiveStateKernel.restore(checkpoint)
    assert restored.canonical_state_json() == kernel.canonical_state_json()
    assert restored.canonical_state_hash() == kernel.canonical_state_hash()
    assert restored.serialize() == checkpoint

    assert kernel.roll_back(rollback_version)["accepted"] is True
    assert kernel.select_replay(4)["selected_replay"] == ["evt-0004", "evt-0001"]


def test_scenario_learn_5859_invalid_inputs_fail_closed() -> None:
    """SCENARIO-LEARN-5859-FAIL-CLOSED: malformed and unbounded inputs do not mutate."""

    kernel = mod.AdaptiveStateKernel(capacity=2, history_capacity=8)
    first_event = mod.make_event("evt-0001", 0, "addition", 10)
    assert kernel.apply_event(first_event)["accepted"] is True
    before = kernel.canonical_state_hash()

    invalid_ops = [
        ("duplicate", lambda: kernel.apply_event(first_event), "DUPLICATE_EVENT"),
        (
            "out_of_order",
            lambda: kernel.apply_event(mod.make_event("evt-0000", 0, "addition", 1)),
            "OUT_OF_ORDER_EVENT",
        ),
        (
            "bad_hash",
            lambda: kernel.apply_event(
                {
                    **mod.make_event("evt-0002", 2, "addition", 1),
                    "payload_hash": "sha256:nope",
                }
            ),
            "INVALID_HASH",
        ),
        (
            "overflow",
            lambda: kernel.apply_event(mod.make_event("evt-0003", 3, "addition", mod.U16_MAX + 1)),
            "FIXED_WIDTH_OVERFLOW",
        ),
        (
            "bad_change",
            lambda: kernel.apply_event(mod.make_event("evt-0004", 4, "fabricated", 1)),
            "UNQUALIFIED_OPERATION",
        ),
        ("missing_core", lambda: kernel.promote("evt-missing"), "UNKNOWN_EVENT"),
        ("rollback_missing", lambda: kernel.roll_back(99), "ROLLBACK_VERSION_MISSING"),
        ("replay_limit", lambda: kernel.select_replay(mod.MAX_REPLAY_LIMIT + 1), "REPLAY_LIMIT_EXCEEDED"),
    ]

    for _name, call, code in invalid_ops:
        result = call()
        assert result["accepted"] is False
        assert result["code"] == code
        assert kernel.canonical_state_hash() == before

    with pytest.raises(ValueError, match="capacity"):
        mod.AdaptiveStateKernel(capacity=0)
    with pytest.raises(ValueError, match="checkpoint"):
        mod.AdaptiveStateKernel.restore(b"not-json")


def test_req_learn_5859_helper_edges_cover_fail_closed_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-5859: helper edges preserve deterministic fail-closed semantics."""

    assert mod.sha256_bytes(b"adaptive").startswith("sha256:")
    kernel = mod.AdaptiveStateKernel(capacity=2, history_capacity=8)
    assert kernel.apply_event("bad")["code"] == "MALFORMED_EVENT"
    assert kernel.apply_event({})["code"] == "MALFORMED_EVENT"
    assert kernel.apply_event(mod.make_event("", 0, "addition", 1))["code"] == "INVALID_EVENT_ID"
    assert (
        kernel.apply_event(mod.make_event("evt-overflow", mod.U32_MAX + 1, "addition", 1))[
            "code"
        ]
        == "FIXED_WIDTH_OVERFLOW"
    )
    with pytest.raises(ValueError, match="history_capacity"):
        mod.AdaptiveStateKernel(capacity=1, history_capacity=1)

    checkpoint = json.loads(kernel.serialize().decode("utf-8"))
    for field, value, message in (
        ("schema", "bad", "schema mismatch"),
        ("abi_version", mod.ABI_VERSION + 1, "ABI version mismatch"),
    ):
        mutated = dict(checkpoint)
        mutated[field] = value
        with pytest.raises(ValueError, match=message):
            mod.AdaptiveStateKernel.restore(mod.canonical_json(mutated).encode("utf-8"))
    mutated = dict(checkpoint)
    mutated["history"] = []
    with pytest.raises(ValueError, match="incomplete"):
        mod.AdaptiveStateKernel.restore(mod.canonical_json(mutated).encode("utf-8"))
    mutated = json.loads(kernel.serialize().decode("utf-8"))
    mutated["active"]["version_id"] = 7
    with pytest.raises(ValueError, match="active version missing"):
        mod.AdaptiveStateKernel.restore(mod.canonical_json(mutated).encode("utf-8"))
    mutated = json.loads(kernel.serialize().decode("utf-8"))
    mutated["history"][0]["capacity"] = 1
    with pytest.raises(ValueError, match="active state differs"):
        mod.AdaptiveStateKernel.restore(mod.canonical_json(mutated).encode("utf-8"))

    kernel = mod.AdaptiveStateKernel(capacity=2, history_capacity=8)
    recurrence = mod.make_event("evt-recur", 0, "recurrence", 1)
    addition = mod.make_event("evt-add", 1, "addition", 2)
    supersession = mod.make_event("evt-super", 2, "supersession", 3)
    assert kernel.apply_event(recurrence)["accepted"] is True
    assert kernel.acquire_core("\n")["code"] == "INVALID_EVENT_ID"
    assert kernel.acquire_core("missing")["code"] == "UNKNOWN_EVENT"
    assert kernel.acquire_core("evt-recur")["code"] == "UNQUALIFIED_OPERATION"
    assert kernel.apply_event(addition)["accepted"] is True
    assert kernel.acquire_core("evt-add")["accepted"] is True
    assert kernel.acquire_core("evt-add")["code"] == "DUPLICATE_CORE"
    assert kernel.apply_event(supersession)["accepted"] is True
    assert kernel.quarantine("\n", "superseded")["code"] == "INVALID_EVENT_ID"
    assert kernel.quarantine("evt-super", "")["code"] == "INVALID_REASON"
    assert kernel.quarantine("missing", "superseded")["code"] == "UNKNOWN_EVENT"
    assert kernel.quarantine("evt-super", "superseded")["accepted"] is True
    assert kernel.quarantine("evt-super", "superseded")["code"] == "DUPLICATE_QUARANTINE"
    assert kernel.promote("\n")["code"] == "INVALID_EVENT_ID"
    assert kernel.promote("evt-super")["code"] == "QUARANTINED_EVENT"
    assert kernel.promote("evt-add")["accepted"] is True
    assert kernel.promote("evt-add")["code"] == "DUPLICATE_PROMOTION"
    assert kernel.select_replay(-1)["code"] == "INVALID_REPLAY_LIMIT"
    assert kernel.roll_back(-1)["code"] == "ROLLBACK_PAST_ROOT"
    assert mod._dispatch(kernel, {"op": "roll_back", "version_id": 0})["accepted"] is True
    with pytest.raises(ValueError, match="unsupported operation"):
        mod._dispatch(kernel, {"op": "bogus"})

    def _raise_os_error(*_args: object, **_kwargs: object) -> None:
        raise OSError("missing command")

    monkeypatch.setattr(subprocess, "run", _raise_os_error)
    assert mod._command_version(["missing-tool"])["available"] is False

    class MismatchingRustKernel:
        def __init__(self, capacity: int = 8, history_capacity: int = 32) -> None:
            self.inner = mod.AdaptiveStateKernel(capacity, history_capacity)

        @classmethod
        def restore(cls, _checkpoint: bytes) -> "MismatchingRustKernel":
            return cls()

        def apply_event(self, event: dict[str, object]) -> dict[str, object]:
            result = self.inner.apply_event(event)
            result["code"] = "MISMATCH"
            return result

        def acquire_core(self, event_id: str) -> dict[str, object]:
            return self.inner.acquire_core(event_id)

        def quarantine(self, event_id: str, reason_code: str) -> dict[str, object]:
            return self.inner.quarantine(event_id, reason_code)

        def promote(self, event_id: str) -> dict[str, object]:
            return self.inner.promote(event_id)

        def select_replay(self, limit: int) -> dict[str, object]:
            return self.inner.select_replay(limit)

        def roll_back(self, version_id: int) -> dict[str, object]:
            return self.inner.roll_back(version_id)

        def serialize(self) -> bytes:
            return b"mismatch"

        def canonical_state_json(self) -> str:
            return "mismatch"

        def canonical_state_hash(self) -> str:
            return "sha256:" + "0" * 64

    monkeypatch.setattr(mod, "_load_rust_binding", lambda: MismatchingRustKernel)
    receipt = mod._run_parity_receipts()
    assert receipt["cross_language_operation_parity"]["parity_failures"]


def test_scenario_learn_5859_python_rust_binding_operation_parity() -> None:
    """SCENARIO-LEARN-5859-OPERATION-PARITY: binding mirrors Python decisions."""

    carnot_rust = pytest.importorskip("carnot._rust")
    py_kernel = mod.AdaptiveStateKernel(capacity=2, history_capacity=16)
    rust_kernel = carnot_rust.RustAdaptiveStateKernel(capacity=2, history_capacity=16)

    for operation in mod.deterministic_fixture_trace():
        py_result = _run_op(py_kernel, operation)
        rust_result = _run_op(rust_kernel, operation)
        assert rust_result == py_result
        _assert_kernel_equal(py_kernel, rust_kernel)

    for trace in mod.randomized_operation_traces(seed=5859, trace_count=5, events_per_trace=8):
        py_kernel = mod.AdaptiveStateKernel(capacity=3, history_capacity=32)
        rust_kernel = carnot_rust.RustAdaptiveStateKernel(capacity=3, history_capacity=32)
        for operation in trace:
            py_result = _run_op(py_kernel, operation)
            rust_result = _run_op(rust_kernel, operation)
            assert rust_result == py_result
            _assert_kernel_equal(py_kernel, rust_kernel)


def test_scenario_learn_5859_serialization_restart_and_rollback_parity() -> None:
    """SCENARIO-LEARN-5859-STATE-HASH-ROUNDTRIP: Python and Rust restore identically."""

    carnot_rust = pytest.importorskip("carnot._rust")
    py_kernel = mod.AdaptiveStateKernel(capacity=3, history_capacity=32)
    rust_kernel = carnot_rust.RustAdaptiveStateKernel(capacity=3, history_capacity=32)
    rollback_version = 0

    for operation in mod.deterministic_fixture_trace():
        py_result = _run_op(py_kernel, operation)
        rust_result = _run_op(rust_kernel, operation)
        if operation["op"] == "promote" and operation["event_id"] == "evt-0004":
            rollback_version = int(py_result["version_id"])
        assert rust_result == py_result

    assert bytes(rust_kernel.serialize()) == py_kernel.serialize()
    py_restored = mod.AdaptiveStateKernel.restore(py_kernel.serialize())
    rust_restored = carnot_rust.RustAdaptiveStateKernel.restore(rust_kernel.serialize())
    _assert_kernel_equal(py_restored, rust_restored)

    assert rust_restored.roll_back(rollback_version) == py_restored.roll_back(rollback_version)
    _assert_kernel_equal(py_restored, rust_restored)
    restart_event = mod.make_event("evt-0099", 99, "addition", 65535)
    for operation in (
        {"op": "apply_event", "event": restart_event},
        {"op": "acquire_core", "event_id": "evt-0099"},
        {"op": "promote", "event_id": "evt-0099"},
    ):
        assert _run_op(rust_restored, operation) == _run_op(py_restored, operation)
    _assert_kernel_equal(py_restored, rust_restored)


def test_req_learn_5859_terminal_artifact_is_hash_bound(tmp_path: Path) -> None:
    """REQ-LEARN-5859: result JSON records conformance and provenance."""

    carnot_rust = pytest.importorskip("carnot._rust")
    assert carnot_rust.RustAdaptiveStateKernel is not None
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    artifact = mod.run(
        result_path=result_path,
        duration_s=2.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )
    loaded = mod.read_json(result_path)

    assert loaded == artifact
    assert mod.validate_artifact(artifact) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "ready"
    assert artifact["honest_verdict"].startswith("parity:")
    assert artifact["adaptive_state_microkernel_ready_score"] == 1.0
    assert isinstance(artifact["adaptive_state_microkernel_ready_score"], float)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["preconditions_checked"]["preconditions_ready"] is True
    assert artifact["qualified_operation_mapping"]["operations"] == list(mod.ABI_OPERATIONS)
    assert artifact["cross_language_operation_parity"]["accept_reject_parity"] is True
    assert artifact["canonical_state_and_hash_parity"]["hash_parity"] is True
    assert artifact["serialization_restart_and_rollback_parity"]["round_trip_parity"] is True
    assert artifact["invalid_input_and_capacity_controls"]["fail_closed"] is True
    assert artifact["per_operation_latency_receipts"]["claim"] == "descriptive_only_no_speedup_claim"
    assert artifact["test_commands"] == TEST_COMMANDS
    assert artifact["test_exit_codes"] == TEST_EXIT_CODES
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)


def test_req_learn_5859_blocked_artifact_when_test_exit_fails(tmp_path: Path) -> None:
    """REQ-LEARN-5859: failed conformance exit codes cannot produce readiness."""

    bad_exit_codes = dict(TEST_EXIT_CODES)
    bad_exit_codes[PYTEST_COMMAND] = 1
    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=2.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes=bad_exit_codes,
        write=False,
    )
    assert artifact["status"] == "blocked"
    assert artifact["adaptive_state_microkernel_ready_score"] == 0.0
    assert artifact["honest_verdict"].startswith("blocked:")
