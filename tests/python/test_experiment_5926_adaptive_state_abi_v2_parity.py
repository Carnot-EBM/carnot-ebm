"""Exp5926 adaptive-state ABI v2 parity tests.

Spec refs: REQ-LEARN-5926, REQ-STORE-5926,
SCENARIO-LEARN-5926-PRECONDITIONS,
SCENARIO-LEARN-5926-ORDERING,
SCENARIO-LEARN-5926-FAIL-CLOSED,
SCENARIO-LEARN-5926-PARITY,
SCENARIO-STORE-5926.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sysconfig
from typing import Any

import pytest

from carnot import adaptive_state_abi_v2 as mod
import carnot._rust as rust


REPO = Path(__file__).resolve().parents[2]
SELF_LEARNING_SPEC = REPO / "openspec/capabilities/self-learning/spec.md"
STORE_SPEC = REPO / "openspec/capabilities/constraint-store/spec.md"
PYTEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5926_adaptive_state_abi_v2_parity.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/adaptive_state_abi_v2.py "
    "-m pytest tests/python/test_experiment_5926_adaptive_state_abi_v2_parity.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/adaptive_state_abi_v2.py --fail-under=100"
)
RUST_COMMAND = "cargo test -p carnot-core adaptive_state_abi_v2 --lib"
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
    ".venv/bin/ruff check python/carnot/adaptive_state_abi_v2.py "
    "tests/python/test_experiment_5926_adaptive_state_abi_v2_parity.py"
)
CLIPPY_CORE_COMMAND = "cargo clippy -p carnot-core --lib -- -D warnings"
CLIPPY_BINDING_COMMAND = (
    "cargo clippy -p carnot-python --lib -- -D warnings -A unused-imports "
    "-A deprecated -A clippy::type-complexity -A clippy::needless-range-loop "
    "-A clippy::too-many-arguments"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5926_adaptive_state_abi_v2_parity.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5926_adaptive_state_abi_v2_parity.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py "
    "ops/changelog.md ops/status.md _bmad/traceability.md"
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


def _apply(
    kernel: Any,
    operation: dict[str, Any],
    snapshots: dict[str, str],
    proposals: dict[str, str],
) -> dict[str, Any]:
    before = kernel.canonical_state_hash()
    name = operation["op"]
    if name == "snapshot":
        result = kernel.snapshot(
            operation["event_id"],
            operation["event_index"],
            operation["row_prefix_checksum"],
            before,
        )
        snapshots[operation["alias"]] = result["snapshot_id"]
        return result
    if name == "lookup":
        return kernel.lookup(
            operation["event_id"],
            snapshots[operation["snapshot"]],
            operation["key"],
            before,
        )
    if name == "propose":
        result = kernel.propose(
            operation["event_id"],
            snapshots[operation["snapshot"]],
            operation["proposal_kind"],
            operation["key"],
            operation["payload_hash"],
            before,
        )
        proposals[operation["alias"]] = result["proposal_id"]
        return result
    if name == "commit":
        return kernel.commit(operation["event_id"], proposals[operation["proposal"]], before)
    if name == "validate":
        return kernel.validate(
            operation["event_id"],
            proposals[operation["proposal"]],
            operation["validator_receipt_hash"],
            operation["validator_status"],
            before,
        )
    if name == "supersede":
        return kernel.supersede(operation["event_id"], proposals[operation["proposal"]], before)
    if name == "promote":
        return kernel.promote(operation["event_id"], proposals[operation["proposal"]], before)
    if name == "quarantine":
        return kernel.quarantine(
            operation["event_id"],
            proposals[operation["proposal"]],
            operation["reason_code"],
            before,
        )
    if name == "reject":
        return kernel.reject(operation["event_id"], proposals[operation["proposal"]], before)
    raise AssertionError(f"unknown ABI v2 test operation: {name}")


def _run_plan(kernel: Any, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snapshots: dict[str, str] = {}
    proposals: dict[str, str] = {}
    receipts = []
    for operation in plan:
        result = _apply(kernel, operation, snapshots, proposals)
        receipts.append(result)
        assert result["accepted"] is True, result
    return receipts


def _assert_equal(py_kernel: mod.AdaptiveStateAbiV2Kernel, rust_kernel: Any) -> None:
    assert rust_kernel.canonical_state_json() == py_kernel.canonical_state_json()
    assert rust_kernel.canonical_state_hash() == py_kernel.canonical_state_hash()
    assert bytes(rust_kernel.serialize()) == py_kernel.serialize()


def test_req_learn_store_5926_specs_declare_abi_v2_contract() -> None:
    """REQ-LEARN-5926/REQ-STORE-5926: specs anchor ABI v2 before code."""

    learn = SELF_LEARNING_SPEC.read_text(encoding="utf-8")
    store = STORE_SPEC.read_text(encoding="utf-8")
    learn_section = learn[learn.index("## REQ-LEARN-5926") : learn.index("## REQ-LEARN-5859")]
    store_section = store[store.index("### REQ-STORE-5926") :]
    normalized = " ".join(learn_section.split())

    for marker in (
        "REQ-LEARN-5926",
        "SCENARIO-LEARN-5926-PRECONDITIONS",
        "SCENARIO-LEARN-5926-ORDERING",
        "SCENARIO-LEARN-5926-FAIL-CLOSED",
        "SCENARIO-LEARN-5926-PARITY",
        "python/carnot/adaptive_state_abi_v2.py",
        "RustAdaptiveStateAbiV2Kernel",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`adaptive_state_abi_v2_ready_score`",
    ):
        assert marker in learn_section
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert f"`{field}`" in learn_section
        assert " ".join(principle.split()) in normalized
    for marker in (
        "REQ-STORE-5926",
        "SCENARIO-STORE-5926",
        "Python/Rust/PyO3 conformance ledger",
        "released cores reject use-after-release and double release",
    ):
        assert marker in store_section


def test_scenario_learn_5926_python_reference_ordering_rollback_and_recover() -> None:
    """SCENARIO-LEARN-5926-ORDERING: Python executes the transaction ABI."""

    kernel = mod.AdaptiveStateAbiV2Kernel(active_capacity=2, quarantine_capacity=3)
    receipts = _run_plan(kernel, mod.exp5924_derived_conformance_trace())
    checkpoint_hash = receipts[5]["resulting_state_hash"]
    pre_rollback_hash = kernel.canonical_state_hash()
    rollback = kernel.rollback("exp5924-rollback", checkpoint_hash, pre_rollback_hash)

    assert rollback["accepted"] is True
    assert rollback["resulting_state_hash"] == checkpoint_hash
    assert {entry["operation"] for entry in receipts} == {
        "snapshot",
        "lookup",
        "propose",
        "commit",
        "validate",
        "promote",
        "quarantine",
        "supersede",
        "reject",
    }
    assert kernel.canonical_state()["active"]
    assert len(kernel.canonical_state()["quarantine"]) <= 3
    recovered = mod.AdaptiveStateAbiV2Kernel.recover(kernel.serialize())
    assert recovered.canonical_state_json() == kernel.canonical_state_json()
    assert recovered.canonical_state_hash() == kernel.canonical_state_hash()


def test_scenario_learn_5926_python_rust_pyo3_byte_state_status_error_parity() -> None:
    """SCENARIO-LEARN-5926-PARITY: PyO3 mirrors Python receipts and bytes."""

    py_kernel = mod.AdaptiveStateAbiV2Kernel(active_capacity=2, quarantine_capacity=3)
    rust_kernel = rust.RustAdaptiveStateAbiV2Kernel(active_capacity=2, quarantine_capacity=3)
    py_snapshots: dict[str, str] = {}
    rust_snapshots: dict[str, str] = {}
    py_proposals: dict[str, str] = {}
    rust_proposals: dict[str, str] = {}

    for operation in mod.exp5924_derived_conformance_trace():
        py_result = _apply(py_kernel, operation, py_snapshots, py_proposals)
        rust_result = _apply(rust_kernel, operation, rust_snapshots, rust_proposals)
        assert rust_result == py_result
        _assert_equal(py_kernel, rust_kernel)

    py_restored = mod.AdaptiveStateAbiV2Kernel.recover(py_kernel.serialize())
    rust_restored = rust.RustAdaptiveStateAbiV2Kernel.recover(rust_kernel.serialize())
    _assert_equal(py_restored, rust_restored)
    py_release = py_restored.release()
    rust_release = rust_restored.release()
    assert py_release["accepted"] is True
    assert rust_release == py_release


def test_scenario_learn_5926_invalid_order_stale_replay_tamper_and_lifetime_reject() -> None:
    """SCENARIO-LEARN-5926-FAIL-CLOSED: unsafe operations fail without mutation."""

    kernel = mod.AdaptiveStateAbiV2Kernel(active_capacity=2, quarantine_capacity=3)
    row = mod.exp5924_event_receipts()[0]
    before = kernel.canonical_state_hash()
    snapshot = kernel.snapshot(
        row["event_id"], row["event_index"], row["row_prefix_checksum"], before
    )
    proposal = kernel.propose(
        row["event_id"],
        snapshot["snapshot_id"],
        "exact_outcome_fact",
        "fact::stale",
        row["payload_hash"],
        kernel.canonical_state_hash(),
    )
    with pytest.raises(ValueError, match="checkpoint is not valid ABI v2 JSON"):
        mod.AdaptiveStateAbiV2Kernel.recover(b"{")

    bad_prior = kernel.commit(row["event_id"], proposal["proposal_id"], "sha256:" + "0" * 64)
    assert bad_prior["code"] == "PRIOR_STATE_MISMATCH"
    assert kernel.canonical_state_hash() == proposal["resulting_state_hash"]

    commit = kernel.commit(row["event_id"], proposal["proposal_id"], kernel.canonical_state_hash())
    assert commit["accepted"] is True
    replay = kernel.commit(row["event_id"], proposal["proposal_id"], kernel.canonical_state_hash())
    assert replay["code"] == "REPLAYED_COMMIT"
    assert kernel.canonical_state_hash() == commit["resulting_state_hash"]

    other = mod.exp5924_event_receipts()[1]
    other_snapshot = kernel.snapshot(
        other["event_id"],
        other["event_index"],
        other["row_prefix_checksum"],
        kernel.canonical_state_hash(),
    )
    other_proposal = kernel.propose(
        other["event_id"],
        other_snapshot["snapshot_id"],
        "exact_outcome_fact",
        "fact::fresh",
        other["payload_hash"],
        kernel.canonical_state_hash(),
    )
    assert (
        kernel.validate(
            other["event_id"],
            other_proposal["proposal_id"],
            other["validator_receipt_hash"],
            "valid",
            kernel.canonical_state_hash(),
        )["code"]
        == "INVALID_ORDER"
    )

    assert (
        kernel.validate(
            row["event_id"],
            proposal["proposal_id"],
            row["validator_receipt_hash"],
            "valid",
            kernel.canonical_state_hash(),
        )["accepted"]
        is True
    )
    assert (
        kernel.promote(row["event_id"], proposal["proposal_id"], kernel.canonical_state_hash())[
            "accepted"
        ]
        is True
    )
    stale = kernel.propose(
        row["event_id"],
        snapshot["snapshot_id"],
        "exact_outcome_fact",
        "fact::stale",
        row["payload_hash"],
        kernel.canonical_state_hash(),
    )
    assert stale["code"] == "STALE_SNAPSHOT"
    partial = kernel.partial_state_transition_probe(kernel.canonical_state_hash())
    assert partial["code"] == "PARTIAL_STATE_TRANSITION_REJECTED"
    assert partial["accepted"] is False

    checkpoint = json.loads(kernel.serialize().decode("utf-8"))
    for field, value, message in (
        ("schema", "wrong", "checkpoint schema mismatch"),
        ("abi_version", mod.ABI_VERSION + 1, "checkpoint ABI version mismatch"),
    ):
        mutated = deepcopy(checkpoint)
        mutated[field] = value
        with pytest.raises(ValueError, match=message):
            mod.AdaptiveStateAbiV2Kernel.recover(mod.canonical_json(mutated).encode("utf-8"))

    assert kernel.release()["accepted"] is True
    assert kernel.release()["code"] == "DOUBLE_RELEASE"
    assert (
        kernel.lookup(row["event_id"], snapshot["snapshot_id"], "fact::stale", before)["code"]
        == "USE_AFTER_RELEASE"
    )


def test_req_learn_5926_validation_edges_and_released_methods_fail_closed(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5926: ABI v2 validation edges return typed rejections."""

    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object required"):
        mod.read_json(scalar)
    with pytest.raises(ValueError, match="active_capacity"):
        mod.AdaptiveStateAbiV2Kernel(active_capacity=0)
    with pytest.raises(ValueError, match="quarantine_capacity"):
        mod.AdaptiveStateAbiV2Kernel(quarantine_capacity=0)

    row0, row1, row2, row3 = mod.exp5924_event_receipts(4)
    kernel = mod.AdaptiveStateAbiV2Kernel(active_capacity=2, quarantine_capacity=3)
    root = kernel.canonical_state_hash()
    assert (
        kernel.snapshot(row0["event_id"], 0, row0["row_prefix_checksum"], "bad")["code"]
        == "PRIOR_STATE_MISMATCH"
    )
    assert kernel.snapshot("", 0, row0["row_prefix_checksum"], root)["code"] == "INVALID_EVENT_ID"
    assert (
        kernel.snapshot(row0["event_id"], -1, row0["row_prefix_checksum"], root)["code"]
        == "FIXED_WIDTH_OVERFLOW"
    )
    assert kernel.snapshot(row0["event_id"], 0, "bad", root)["code"] == "INVALID_PREFIX_HASH"
    snapshot = kernel.snapshot(
        row0["event_id"],
        row0["event_index"],
        row0["row_prefix_checksum"],
        root,
    )
    assert (
        kernel.lookup(row0["event_id"], "missing", "fact::x", kernel.canonical_state_hash())["code"]
        == "STALE_SNAPSHOT"
    )
    assert (
        kernel.lookup(
            row0["event_id"],
            snapshot["snapshot_id"],
            "",
            kernel.canonical_state_hash(),
        )["code"]
        == "INVALID_KEY"
    )
    assert (
        kernel.propose(
            row0["event_id"],
            snapshot["snapshot_id"],
            "",
            "fact::x",
            row0["payload_hash"],
            kernel.canonical_state_hash(),
        )["code"]
        == "INVALID_PROPOSAL_KIND"
    )
    assert (
        kernel.propose(
            row0["event_id"],
            snapshot["snapshot_id"],
            "exact_outcome_fact",
            "",
            row0["payload_hash"],
            kernel.canonical_state_hash(),
        )["code"]
        == "INVALID_KEY"
    )
    assert (
        kernel.propose(
            row0["event_id"],
            snapshot["snapshot_id"],
            "exact_outcome_fact",
            "fact::x",
            "bad",
            kernel.canonical_state_hash(),
        )["code"]
        == "INVALID_PAYLOAD_HASH"
    )
    proposal = kernel.propose(
        row0["event_id"],
        snapshot["snapshot_id"],
        "exact_outcome_fact",
        "fact::x",
        row0["payload_hash"],
        kernel.canonical_state_hash(),
    )
    assert (
        kernel.propose(
            row0["event_id"],
            snapshot["snapshot_id"],
            "exact_outcome_fact",
            "fact::x",
            row0["payload_hash"],
            kernel.canonical_state_hash(),
        )["code"]
        == "REPLAYED_PROPOSAL"
    )
    assert (
        kernel.commit(row1["event_id"], "missing", kernel.canonical_state_hash())["code"]
        == "INVALID_ORDER"
    )
    assert (
        kernel.commit(row0["event_id"], proposal["proposal_id"], kernel.canonical_state_hash())[
            "accepted"
        ]
        is True
    )
    assert (
        kernel.validate(
            row0["event_id"],
            proposal["proposal_id"],
            "bad",
            "valid",
            kernel.canonical_state_hash(),
        )["code"]
        == "INVALID_VALIDATOR_RECEIPT"
    )
    assert (
        kernel.validate(
            row0["event_id"],
            proposal["proposal_id"],
            row0["validator_receipt_hash"],
            "bogus",
            kernel.canonical_state_hash(),
        )["code"]
        == "INVALID_VALIDATOR_STATUS"
    )
    assert (
        kernel.validate(
            row0["event_id"],
            proposal["proposal_id"],
            row0["validator_receipt_hash"],
            "valid",
            kernel.canonical_state_hash(),
        )["accepted"]
        is True
    )
    assert (
        kernel.supersede(row0["event_id"], proposal["proposal_id"], "bad")["code"]
        == "PRIOR_STATE_MISMATCH"
    )
    assert (
        kernel.supersede(row0["event_id"], proposal["proposal_id"], kernel.canonical_state_hash())[
            "code"
        ]
        == "NO_ACTIVE_TARGET"
    )
    assert (
        kernel.promote(row0["event_id"], "missing", kernel.canonical_state_hash())["code"]
        == "INVALID_ORDER"
    )
    assert (
        kernel.promote(row0["event_id"], proposal["proposal_id"], "bad")["code"]
        == "PRIOR_STATE_MISMATCH"
    )
    assert (
        kernel.promote(row0["event_id"], proposal["proposal_id"], kernel.canonical_state_hash())[
            "accepted"
        ]
        is True
    )
    assert (
        kernel.promote(row0["event_id"], proposal["proposal_id"], kernel.canonical_state_hash())[
            "code"
        ]
        == "INVALID_ORDER"
    )
    assert (
        kernel.lookup(
            row0["event_id"],
            snapshot["snapshot_id"],
            "fact::x",
            kernel.canonical_state_hash(),
        )["code"]
        == "SAME_EVENT_READ_AFTER_WRITE"
    )
    assert (
        kernel.rollback("rollback", "sha256:" + "0" * 64, kernel.canonical_state_hash())["code"]
        == "ROLLBACK_TARGET_MISSING"
    )
    assert (
        kernel.rollback("rollback", kernel.canonical_state_hash(), "bad")["code"]
        == "PRIOR_STATE_MISMATCH"
    )

    q_snapshot = kernel.snapshot(
        row1["event_id"],
        row1["event_index"],
        row1["row_prefix_checksum"],
        kernel.canonical_state_hash(),
    )
    q_proposal = kernel.propose(
        row1["event_id"],
        q_snapshot["snapshot_id"],
        "poison_burst",
        "fact::q",
        row1["payload_hash"],
        kernel.canonical_state_hash(),
    )
    assert (
        kernel.commit(row1["event_id"], q_proposal["proposal_id"], kernel.canonical_state_hash())[
            "accepted"
        ]
        is True
    )
    assert (
        kernel.validate(
            row1["event_id"],
            q_proposal["proposal_id"],
            row1["validator_receipt_hash"],
            "quarantine",
            kernel.canonical_state_hash(),
        )["accepted"]
        is True
    )
    assert (
        kernel.supersede(
            row1["event_id"], q_proposal["proposal_id"], kernel.canonical_state_hash()
        )["code"]
        == "INVALID_ORDER"
    )
    assert (
        kernel.reject(row1["event_id"], q_proposal["proposal_id"], kernel.canonical_state_hash())[
            "code"
        ]
        == "INVALID_ORDER"
    )
    assert (
        kernel.quarantine(
            row1["event_id"], q_proposal["proposal_id"], "", kernel.canonical_state_hash()
        )["code"]
        == "INVALID_REASON"
    )
    assert (
        kernel.quarantine(row1["event_id"], q_proposal["proposal_id"], "poison", "bad")["code"]
        == "PRIOR_STATE_MISMATCH"
    )
    assert (
        kernel.quarantine(
            row1["event_id"],
            q_proposal["proposal_id"],
            "poison",
            kernel.canonical_state_hash(),
        )["accepted"]
        is True
    )

    r_snapshot = kernel.snapshot(
        row2["event_id"],
        row2["event_index"],
        row2["row_prefix_checksum"],
        kernel.canonical_state_hash(),
    )
    r_proposal = kernel.propose(
        row2["event_id"],
        r_snapshot["snapshot_id"],
        "model_candidate",
        "fact::r",
        row2["payload_hash"],
        kernel.canonical_state_hash(),
    )
    assert (
        kernel.commit(row2["event_id"], r_proposal["proposal_id"], kernel.canonical_state_hash())[
            "accepted"
        ]
        is True
    )
    assert (
        kernel.validate(
            row2["event_id"],
            r_proposal["proposal_id"],
            row2["validator_receipt_hash"],
            "reject",
            kernel.canonical_state_hash(),
        )["accepted"]
        is True
    )
    assert (
        kernel.reject(row2["event_id"], r_proposal["proposal_id"], "bad")["code"]
        == "PRIOR_STATE_MISMATCH"
    )
    assert (
        kernel.reject(row2["event_id"], r_proposal["proposal_id"], kernel.canonical_state_hash())[
            "accepted"
        ]
        is True
    )
    s_snapshot = kernel.snapshot(
        row3["event_id"],
        row3["event_index"],
        row3["row_prefix_checksum"],
        kernel.canonical_state_hash(),
    )
    s_proposal = kernel.propose(
        row3["event_id"],
        s_snapshot["snapshot_id"],
        "exact_outcome_fact",
        "fact::x",
        row3["payload_hash"],
        kernel.canonical_state_hash(),
    )
    assert (
        kernel.commit(row3["event_id"], s_proposal["proposal_id"], kernel.canonical_state_hash())[
            "accepted"
        ]
        is True
    )
    assert (
        kernel.validate(
            row3["event_id"],
            s_proposal["proposal_id"],
            row3["validator_receipt_hash"],
            "valid",
            kernel.canonical_state_hash(),
        )["accepted"]
        is True
    )
    assert (
        kernel.promote(row3["event_id"], s_proposal["proposal_id"], kernel.canonical_state_hash())[
            "code"
        ]
        == "SUPERSEDE_REQUIRED"
    )
    assert kernel.partial_state_transition_probe("bad")["code"] == "PRIOR_STATE_MISMATCH"

    released = mod.AdaptiveStateAbiV2Kernel()
    released_hash = released.canonical_state_hash()
    released_snapshot = released.snapshot(
        row0["event_id"], 0, row0["row_prefix_checksum"], released_hash
    )
    released.release()
    assert (
        released.snapshot(row0["event_id"], 0, row0["row_prefix_checksum"], released_hash)["code"]
        == "USE_AFTER_RELEASE"
    )
    assert (
        released.propose(
            row0["event_id"],
            released_snapshot["snapshot_id"],
            "exact_outcome_fact",
            "fact::x",
            row0["payload_hash"],
            released_hash,
        )["code"]
        == "USE_AFTER_RELEASE"
    )
    assert (
        released.commit(row0["event_id"], "missing", released_hash)["code"] == "USE_AFTER_RELEASE"
    )
    assert (
        released.validate(
            row0["event_id"], "missing", row0["validator_receipt_hash"], "valid", released_hash
        )["code"]
        == "USE_AFTER_RELEASE"
    )
    assert (
        released.supersede(row0["event_id"], "missing", released_hash)["code"]
        == "USE_AFTER_RELEASE"
    )
    assert (
        released.promote(row0["event_id"], "missing", released_hash)["code"] == "USE_AFTER_RELEASE"
    )
    assert (
        released.quarantine(row0["event_id"], "missing", "why", released_hash)["code"]
        == "USE_AFTER_RELEASE"
    )
    assert (
        released.reject(row0["event_id"], "missing", released_hash)["code"] == "USE_AFTER_RELEASE"
    )
    assert (
        released.rollback("rollback", released_hash, released_hash)["code"] == "USE_AFTER_RELEASE"
    )
    assert released.partial_state_transition_probe(released_hash)["code"] == "USE_AFTER_RELEASE"


def test_req_learn_5926_checkpoint_artifact_and_helper_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5926: corrupt artifacts and helper failures stay blocked."""

    kernel = mod.AdaptiveStateAbiV2Kernel()
    checkpoint = json.loads(kernel.serialize().decode("utf-8"))
    missing_payload = deepcopy(checkpoint)
    missing_payload.pop("state")
    with pytest.raises(ValueError, match="checkpoint payload is incomplete"):
        mod.AdaptiveStateAbiV2Kernel.recover(mod.canonical_json(missing_payload).encode("utf-8"))
    bad_hash = deepcopy(checkpoint)
    bad_hash["state_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="checkpoint state hash mismatch"):
        mod.AdaptiveStateAbiV2Kernel.recover(mod.canonical_json(bad_hash).encode("utf-8"))
    missing_history = deepcopy(checkpoint)
    missing_history["history"] = []
    with pytest.raises(ValueError, match="active state missing"):
        mod.AdaptiveStateAbiV2Kernel.recover(mod.canonical_json(missing_history).encode("utf-8"))
    mismatched_history = deepcopy(checkpoint)
    mismatched_history["history"][0]["state"]["version"] = 9
    with pytest.raises(ValueError, match="active state differs"):
        mod.AdaptiveStateAbiV2Kernel.recover(mod.canonical_json(mismatched_history).encode("utf-8"))

    with pytest.raises(ValueError, match="unsupported ABI v2 operation"):
        mod.run_plan(mod.AdaptiveStateAbiV2Kernel(), [{"op": "bogus"}])

    artifact = mod.run(
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        duration_s=0.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=False,
    )
    for field, message in (
        ("status", "missing required artifact fields"),
        ("field_provenance", "field_provenance"),
    ):
        broken = deepcopy(artifact)
        broken.pop(field)
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(broken)
    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)
    bad_oracle = deepcopy(artifact)
    bad_oracle["verifier_is_oracle"] = False
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        mod.validate_artifact(bad_oracle)
    bad_provenance = deepcopy(artifact)
    bad_provenance["field_provenance"]["status"]["principle"] = "wrong"
    with pytest.raises(ValueError, match="field_provenance:status"):
        mod.validate_artifact(bad_provenance)
    bad_provenance_type = deepcopy(artifact)
    bad_provenance_type["field_provenance"] = []
    with pytest.raises(ValueError, match="field_provenance"):
        mod.validate_artifact(bad_provenance_type)
    bad_score = deepcopy(artifact)
    bad_score["adaptive_state_abi_v2_ready_score"] = 0.0
    with pytest.raises(ValueError, match="ready_score"):
        mod.validate_artifact(bad_score)
    bad_status = deepcopy(artifact)
    bad_status["status"] = "blocked"
    with pytest.raises(ValueError, match="status"):
        mod.validate_artifact(bad_status)
    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "blocked: wrong"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)
    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = mod.sha256_text("wrong")
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        mod.validate_artifact(bad_checksum)

    failed_codes = dict(TEST_EXIT_CODES)
    failed_codes[PYTEST_COMMAND] = 1
    blocked = mod.run(
        result_path=tmp_path / "blocked.json",
        duration_s=0.0,
        test_commands=TEST_COMMANDS,
        test_exit_codes=failed_codes,
        write=False,
    )
    assert blocked["status"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked:")
    assert mod.blocked_reasons(
        {
            "gate_replay_receipt": {},
            "preconditions_checked": {},
            "byte_state_status_and_error_parity": {"parity_failures": [1]},
            "invalid_order_stale_replay_and_tamper_rejection": {},
            "crash_prefix_recovery_and_rollback": {},
            "serialization_and_fresh_process_receipts": {},
            "task_owned_test_boundary_and_global_failure_delta": {},
        }
    ) == [
        "exp5924_gate",
        "preconditions",
        "parity",
        "tamper",
        "rollback",
        "fresh_process",
        "global_delta",
    ]

    monkeypatch.setattr(mod, "load_rust_binding", lambda: None)
    assert mod.parity_receipts()["parity_failures"] == [{"case": "pyo3_binding_missing"}]

    class MismatchingRustKernel:
        def __init__(self, active_capacity: int = 2, quarantine_capacity: int = 3) -> None:
            self.inner = mod.AdaptiveStateAbiV2Kernel(active_capacity, quarantine_capacity)

        @classmethod
        def recover(cls, checkpoint: bytes) -> "MismatchingRustKernel":
            recovered = cls()
            recovered.inner = mod.AdaptiveStateAbiV2Kernel.recover(checkpoint)
            return recovered

        def snapshot(self, *args: Any) -> dict[str, Any]:
            result = self.inner.snapshot(*args)
            result["code"] = "MISMATCH"
            return result

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

        def canonical_state_json(self) -> str:
            return "mismatch"

        def canonical_state_hash(self) -> str:
            return "sha256:" + "0" * 64

        def serialize(self) -> bytes:
            return b"mismatch"

    monkeypatch.setattr(mod, "load_rust_binding", lambda: MismatchingRustKernel)
    failures = {failure["case"] for failure in mod.parity_receipts()["parity_failures"]}
    assert {
        "operation_receipts",
        "canonical_state_json",
        "state_hash",
        "serialized_bytes",
    } <= failures

    def _raise_os_error(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("missing command")

    monkeypatch.setattr(subprocess, "run", _raise_os_error)
    assert mod.command_version(["missing-tool"])["available"] is False
    assert mod.historical_artifacts_unchanged_receipt()["unchanged"] is True
    assert mod.protected_files_unchanged_receipt()["unchanged"] is True


def test_scenario_learn_5926_serialization_recovers_in_fresh_process() -> None:
    """SCENARIO-LEARN-5926-PARITY: checkpoint bytes recover outside this process."""

    kernel = mod.AdaptiveStateAbiV2Kernel(active_capacity=2, quarantine_capacity=3)
    _run_plan(kernel, mod.exp5924_derived_conformance_trace())
    checkpoint = kernel.serialize()
    script = (
        "from carnot import adaptive_state_abi_v2 as mod; "
        "import sys; "
        "kernel = mod.AdaptiveStateAbiV2Kernel.recover(bytes.fromhex(sys.stdin.read())); "
        "print(kernel.canonical_state_hash()); "
        "print(kernel.serialize().hex())"
    )
    completed = subprocess.run(
        [str(REPO / ".venv/bin/python"), "-c", script],
        cwd=REPO,
        check=True,
        capture_output=True,
        input=checkpoint.hex(),
        text=True,
    )
    recovered_hash, recovered_hex = completed.stdout.strip().splitlines()

    assert recovered_hash == kernel.canonical_state_hash()
    assert bytes.fromhex(recovered_hex) == checkpoint


def test_req_learn_5926_terminal_artifact_is_hash_bound(tmp_path: Path) -> None:
    """REQ-LEARN-5926: result JSON records complete ABI v2 parity."""

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    assert (REPO / f"python/carnot/_rust{ext_suffix}").exists()
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH.name
    artifact = mod.run(
        result_path=result_path,
        duration_s=1.25,
        test_commands=TEST_COMMANDS,
        test_exit_codes=TEST_EXIT_CODES,
        write=True,
    )

    loaded = mod.read_json(result_path)
    assert loaded == artifact
    assert mod.validate_artifact(artifact) is True
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["adaptive_state_abi_v2_ready_score"] == pytest.approx(1.0)
    assert isinstance(artifact["adaptive_state_abi_v2_ready_score"], float)
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is True
    assert artifact["exp5859_preserved_and_scope_delta"]["exp5859_rewritten"] is False
    assert artifact["byte_state_status_and_error_parity"]["parity_failures"] == []
    assert artifact["invalid_order_stale_replay_and_tamper_rejection"]["all_rejected"] is True
    assert artifact["crash_prefix_recovery_and_rollback"]["rollback_exact"] is True
    assert artifact["serialization_and_fresh_process_receipts"]["fresh_process_recovered"] is True
    assert artifact["protected_files_unchanged"]["unchanged"] is True
    assert artifact["historical_artifacts_unchanged"]["unchanged"] is True
    assert artifact["test_commands"] == TEST_COMMANDS
    assert artifact["test_exit_codes"] == TEST_EXIT_CODES
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    for field, principle in mod.REQUIRED_FIELD_PRINCIPLES.items():
        assert artifact["field_provenance"][field]["principle"] == principle
