"""Tests for native lossless packed-belief parity and cost.

Spec refs: REQ-CL-7230, SCENARIO-CL-7230-*, REQ-RUSTPY-7230,
and SCENARIO-RUSTPY-7230-*.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from carnot import experiment_7226_v636_belief_compiler as exp7226
from carnot import experiment_7230_v636_native_belief as exp7230


@pytest.fixture(scope="session")
def native_extension() -> Path:
    """REQ-RUSTPY-7230: compile the changed crate for the test interpreter."""

    extension, _ = exp7230.build_native_extension(exp7230.REPO_ROOT, show_progress=False)
    return extension


@pytest.fixture()
def native(native_extension: Path):
    """SCENARIO-RUSTPY-7230-BINDING: load the exact task-owned binary."""

    return exp7230.load_native_extension(native_extension)


def test_req_cl_7230_compiled_batch_parity_and_unknowns(native) -> None:
    """REQ-CL-7230: native majority, tie, unknown, and energy match Python."""

    controller = native.RustPackedBeliefController()
    state = exp7230.reference_semantic_state(exp7226.PackedBeliefController())
    controller.load_state(exp7230.canonical_json(state))
    families = np.ascontiguousarray([0, 0, 4], dtype=np.uint8)
    values = np.ascontiguousarray([16, 8, 1], dtype=np.int64)
    labels = np.ascontiguousarray([1, 0, 1], dtype=np.int8)
    result = dict(controller.query_batch(families, values, labels))
    reference = exp7230.reference_query(state, families, values, labels)
    assert result["decisions"] == reference["decisions"]
    assert result["disagreements"] == reference["disagreements"]
    assert result["energy_status"] == reference["energy_status"]
    assert result["energies"][:2] == reference["energies"][:2]
    assert result["energies"][2] is None
    assert result["kernel_ns"] >= 0


def test_scenario_cl_7230_parity_update_reset_rollback_and_state(native) -> None:
    """SCENARIO-CL-7230-PARITY: updates, reset, and rollback keep exact bytes."""

    python = exp7226.PackedBeliefController.from_survivors({"lower_bound": {0}})
    state = exp7230.reference_semantic_state(python)
    controller = native.RustPackedBeliefController()
    controller.load_state(exp7230.canonical_json(state))
    before = controller.serialize_state()
    families = np.ascontiguousarray([0, 0], dtype=np.uint8)
    values = np.ascontiguousarray([16, 8], dtype=np.int64)
    labels = np.ascontiguousarray([0, 1], dtype=np.int8)
    roles = np.ascontiguousarray([1, 0], dtype=np.uint8)
    receipt = dict(controller.update_batch(families, values, labels, roles))
    releases = [
        exp7230.release("reset", 0, 16, 0, 1),
        exp7230.release("validation", 0, 8, 1, 0),
    ]
    python.commit_batch(
        releases,
        current_cycle=0,
        expected_parent_hash=python.state_hash(),
    )
    assert receipt["reset_count"] == 1
    assert controller.serialize_state() == exp7230.canonical_json(
        exp7230.reference_semantic_state(python)
    )
    rollback = dict(controller.rollback())
    assert rollback["byte_identical"] is True
    assert controller.serialize_state() == before
    with pytest.raises(ValueError, match="no rollback state"):
        controller.rollback()


def test_req_rustpy_7230_boundary_rejects_bad_arrays_and_state(native) -> None:
    """REQ-RUSTPY-7230: malformed batches and checkpoints fail at the boundary."""

    controller = native.RustPackedBeliefController()
    one_family = np.ascontiguousarray([0], dtype=np.uint8)
    one_value = np.ascontiguousarray([1], dtype=np.int64)
    one_label = np.ascontiguousarray([1], dtype=np.int8)
    with pytest.raises(ValueError, match="query batch lengths"):
        controller.query_batch(
            one_family,
            np.ascontiguousarray([], dtype=np.int64),
            one_label,
        )
    with pytest.raises(ValueError, match="update batch lengths"):
        controller.update_batch(
            one_family,
            one_value,
            one_label,
            np.ascontiguousarray([], dtype=np.uint8),
        )
    with pytest.raises(ValueError, match="invalid update"):
        controller.update_batch(
            np.ascontiguousarray([9], dtype=np.uint8),
            one_value,
            one_label,
            np.ascontiguousarray([1], dtype=np.uint8),
        )
    for state in (
        "{}",
        exp7230.canonical_json(
            {"epochs": [0] * 4, "survivor_masks": [0] * 4, "version": 0, "vote_counts": []}
        ),
        exp7230.canonical_json(
            {
                "epochs": [0] * 4,
                "survivor_masks": [1 << 40] * 4,
                "version": 0,
                "vote_counts": [[0] * 33 for _ in range(4)],
            }
        ),
    ):
        with pytest.raises(ValueError):
            controller.load_state(state)


def test_scenario_cl_7230_exhaustive_and_fixed_sequence_parity(native) -> None:
    """SCENARIO-CL-7230-PARITY: finite states and fixed update sequences match."""

    exhaustive = exp7230.run_exhaustive_parity(native, max_subset_size=1)
    assert exhaustive["passed"] is True
    assert exhaustive["mismatch_count"] == 0
    assert exhaustive["case_count"] == 4 * 34 * 33
    rows = exp7230.run_sequence_parity(native, seeds=(7_230_001, 7_230_002), steps=32)
    assert len(rows) == 2
    assert all(row["passed"] for row in rows)
    assert all(row["native_state_b64"] == row["reference_state_b64"] for row in rows)


def test_scenario_cl_7230_restore_uses_a_fresh_process(native_extension: Path, native) -> None:
    """SCENARIO-CL-7230-RESTORE: checkpoint bytes survive a new interpreter."""

    controller = native.RustPackedBeliefController()
    checkpoint = controller.serialize_state()
    receipt = exp7230.run_cross_process_restore(native_extension, checkpoint)
    assert receipt["passed"] is True
    assert receipt["module_file"] == str(native_extension.resolve())
    assert receipt["serialized_state"] == checkpoint
    assert receipt["python_fallback_used"] is False


def test_scenario_rustpy_7230_cost_rows_and_gates(native) -> None:
    """SCENARIO-RUSTPY-7230-COST: paired rows retain full costs and gate honestly."""

    rows = exp7230.run_cost_benchmark(
        native,
        batch_sizes=(1, 3),
        repetitions=2,
        seed=7_230_100,
        timeout_s=30.0,
    )
    assert len(rows) == 8
    assert {(row["batch_size"], row["repetition"], row["arm"]) for row in rows} == {
        (batch, repetition, arm)
        for batch in (1, 3)
        for repetition in range(2)
        for arm in ("native_pyo3", "python_reference")
    }
    assert all(row["end_to_end_ns"] > 0 for row in rows)
    assert all(row["checkpoint_bytes"] > 0 for row in rows)
    summary = exp7230.summarize_cost(rows, required_batches=(1, 3), repetitions=2)
    assert set(summary) == {"cells", "native_cost_value_score", "nfr_01_10x_met"}
    assert len(summary["cells"]) == 2

    fast_rows = exp7230.synthetic_cost_rows((1, 3), 2, python_ns=1_000, native_ns=10)
    fast = exp7230.summarize_cost(fast_rows, required_batches=(1, 3), repetitions=2)
    assert fast["native_cost_value_score"] == 1
    assert fast["nfr_01_10x_met"] is True
    slow_rows = exp7230.synthetic_cost_rows((1, 3), 2, python_ns=10, native_ns=1_000)
    assert (
        exp7230.summarize_cost(slow_rows, required_batches=(1, 3), repetitions=2)[
            "native_cost_value_score"
        ]
        == 0
    )


def test_req_rustpy_7230_preconditions_authenticate_gate_and_quarantine(tmp_path: Path) -> None:
    """REQ-RUSTPY-7230: gates unwrap only principled values after quarantine checks."""

    paths = exp7230.ExperimentPaths.under(tmp_path)
    checks, hashes = exp7230.collect_preconditions(exp7230.REPO_ROOT, paths)
    assert all(row["passed"] for row in checks)
    assert hashes["results/experiment_7226_v636_belief_compiler.json"].startswith("sha256:")
    assert exp7230.unwrap_principled_value({"principle": "why", "value": 1}) == 1
    wrapped_extra = {"principle": "why", "value": 1, "extra": True}
    assert exp7230.unwrap_principled_value(wrapped_extra) == wrapped_extra

    upstream = json.loads(exp7230.UPSTREAM_PATH.read_text(encoding="utf-8"))
    upstream["flagged_adversarial"] = True
    blocked = tmp_path / "blocked-upstream.json"
    blocked.write_text(json.dumps(upstream), encoding="utf-8")
    failed, _ = exp7230.collect_preconditions(
        exp7230.REPO_ROOT,
        paths,
        upstream_path=blocked,
    )
    assert any(row["check"] == "exp7226_not_quarantined" and not row["passed"] for row in failed)
    assert any(
        row["check"] == "exp7226_ready_gate"
        and row["observed_value"] == "not_consumed_due_to_quarantine"
        for row in failed
    )


def test_req_rustpy_7230_artifact_validator_and_blocked_contract(tmp_path: Path) -> None:
    """REQ-RUSTPY-7230: terminal and external-block schemas fail closed."""

    artifact = exp7230.complete_artifact_fixture_for_test()
    assert exp7230.validate_artifact(artifact) == []
    for mutation, expected in (
        ({"MODEL_SPECS": [{}]}, "model_declaration"),
        ({"native_belief_ready_score": 0}, "ready_score"),
        ({"nfr_01_10x_met": True}, "nfr_gate"),
        ({"scientific_value_inherited": True}, "scientific_value"),
    ):
        changed = deepcopy(artifact)
        changed.update(mutation)
        changed["reproducibility_checksum"] = exp7230.artifact_checksum(changed)
        assert expected in exp7230.validate_artifact(changed)

    failed = exp7230.check("missing", "external", "field", 1, None, False)
    blocked = exp7230.blocked_artifact_for_test(failed)
    assert exp7230.validate_artifact(blocked) == []
    assert blocked["rows"] == blocked["parity_rows"] == blocked["cost_rows"] == []
    output = tmp_path / "artifact.json"
    receipt = exp7230.atomic_write(output, blocked)
    assert receipt["atomic_replace"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == blocked


def test_req_rustpy_7230_build_artifact_uses_measured_components(
    monkeypatch: pytest.MonkeyPatch,
    native_extension: Path,
    native,
) -> None:
    """REQ-RUSTPY-7230: artifact assembly derives readiness and cost scores."""

    checks = [exp7230.check("ok", "fixture", "value", 1, 1, True)]
    monkeypatch.setattr(exp7230, "collect_preconditions", lambda *_args, **_kwargs: (checks, {}))
    monkeypatch.setattr(
        exp7230,
        "build_native_extension",
        lambda *_args, **_kwargs: (native_extension, {"command": ["cargo"], "exit_code": 0}),
    )
    monkeypatch.setattr(exp7230, "load_native_extension", lambda _path: native)
    monkeypatch.setattr(
        exp7230,
        "run_exhaustive_parity",
        lambda *_args, **_kwargs: {"passed": True, "mismatch_count": 0, "case_count": 1},
    )
    sequence = exp7230.sequence_fixture_row(7_230_001)
    monkeypatch.setattr(exp7230, "run_sequence_parity", lambda *_args, **_kwargs: [sequence] * 20)
    monkeypatch.setattr(
        exp7230,
        "run_cross_process_restore",
        lambda *_args, **_kwargs: {
            "passed": True,
            "module_file": str(native_extension.resolve()),
            "serialized_state": exp7230.initial_semantic_state_json(),
            "python_fallback_used": False,
        },
    )
    cost_rows = exp7230.synthetic_cost_rows(
        exp7230.BATCH_SIZES,
        exp7230.REPETITIONS,
        python_ns=1_000,
        native_ns=10,
    )
    monkeypatch.setattr(exp7230, "run_cost_benchmark", lambda *_args, **_kwargs: cost_rows)
    artifact = exp7230.build_artifact(exp7230.REPO_ROOT, exp7230.ExperimentPaths.defaults())
    assert artifact["native_belief_ready_score"] == 1
    assert artifact["native_cost_value_score"] == 1
    assert artifact["nfr_01_10x_met"] is True
    assert exp7230.validate_artifact(artifact) == []


def test_req_rustpy_7230_run_and_cli_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-RUSTPY-7230: command validation and atomic publication are bounded."""

    artifact = exp7230.complete_artifact_fixture_for_test()
    monkeypatch.setattr(exp7230, "build_artifact", lambda *_args, **_kwargs: artifact)
    output = tmp_path / "result.json"
    assert exp7230.run_experiment(exp7230.REPO_ROOT, output, exp7230.RUN_DATE) == artifact
    assert output.is_file()
    assert exp7230.main(["--date", "bad", "--output", str(output)]) == 2
    assert exp7230.main(["--validate", str(output)]) == 0
    output.write_text("{}", encoding="utf-8")
    assert exp7230.main(["--validate", str(output)]) == 2

    script = exp7230.REPO_ROOT / "scripts/experiments/experiment_7230_v636_native_belief.py"
    monkeypatch.setattr(
        sys, "argv", [str(script), "--date", exp7230.RUN_DATE, "--output", str(output)]
    )
    runpy.run_path(str(script), run_name="experiment_7230_wrapper_test")
    assert importlib.util.find_spec("carnot.experiment_7230_v636_native_belief") is not None
