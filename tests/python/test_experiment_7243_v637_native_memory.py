"""Tests for native archive-memory parity and complete boundary cost.

Spec refs: REQ-CL-7243, SCENARIO-CL-7243-*, REQ-RUSTPY-7243, and
SCENARIO-RUSTPY-7243-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from carnot import experiment_7230_v636_native_belief as exp7230
from carnot import experiment_7243_v637_native_memory as exp7243


@pytest.fixture(scope="session")
def native_extension() -> Path:
    """REQ-RUSTPY-7243: build the real extension for this interpreter."""

    extension, _ = exp7243.build_native_extension(exp7243.REPO_ROOT, show_progress=False)
    return extension


@pytest.fixture(scope="session")
def native(native_extension: Path):
    """SCENARIO-RUSTPY-7243-BINDING: import the exact compiled binary."""

    return exp7230.load_native_extension(native_extension)


def _release(index: int, label: str = "accept") -> dict[str, object]:
    """Build one released witness for a bounded controller probe."""

    return {
        "event_id": f"release-{index}",
        "family_id": "lower_bound",
        "numeric_value": index % 33,
        "observed_label": label,
        "role": "support",
        "request_index": index,
        "release_index": index,
    }


def test_req_cl_7243_preconditions_authenticate_fixture_and_quarantine(tmp_path: Path) -> None:
    """REQ-CL-7243: accept Exp7240 and reject the quarantined Exp7230 claim."""

    paths = exp7243.ExperimentPaths.under(tmp_path)
    checks, hashes, sources = exp7243.collect_preconditions(exp7243.REPO_ROOT, paths)
    assert exp7243.gate_summary(checks)["passed"] is True
    assert hashes[str(exp7243.UPSTREAM_RELATIVE)].startswith("sha256:")
    assert sources["exp7240"]["recurrence_fixture_ready_score"] == 1
    history = sources["exp7230"]
    assert history["flagged_adversarial"] is True
    assert history["native_cost_value_score"] == 1
    assert exp7243.unwrap_principled_value({"principle": "why", "value": 1}) == 1
    ordinary = {"principle": "why", "value": 1, "source": "kept"}
    assert exp7243.unwrap_principled_value(ordinary) is ordinary

    quarantined = deepcopy(sources["exp7240"])
    quarantined["flagged_adversarial"] = True
    with patch.object(
        exp7243, "read_object", side_effect=[quarantined, sources["exp7217"], history]
    ):
        failed, _, _ = exp7243.collect_preconditions(exp7243.REPO_ROOT, paths)
    assert any(row["check"] == "exp7240_not_quarantined" and not row["passed"] for row in failed)
    gate = next(row for row in failed if row["check"] == "exp7240_ready_gate")
    assert gate["observed_value"] == "not_consumed_due_to_quarantine"

    exclusion_text = (exp7243.REPO_ROOT / exp7243.EXCLUSION_RELATIVE).read_text(encoding="utf-8")
    safe_load = exp7243.yaml.safe_load

    def reject_exclusion_yaml(text: str):
        if text == exclusion_text:
            raise exp7243.yaml.YAMLError("malformed exclusion manifest")
        return safe_load(text)

    with (
        patch.object(exp7243.yaml, "safe_load", side_effect=reject_exclusion_yaml),
        patch.object(
            exp7243.exp7240,
            "reproducibility_checksum",
            side_effect=ValueError("malformed upstream checksum"),
        ),
    ):
        defensive, _, _ = exp7243.collect_preconditions(exp7243.REPO_ROOT, paths)
    checksum = next(
        row for row in defensive if row["check"] == "exp7240_checksum_and_stream_hashes"
    )
    assert checksum["passed"] is False


def test_scenario_cl_7243_native_active_and_archive_parity(native) -> None:
    """SCENARIO-CL-7243-PARITY: the shared archive controller stays exact."""

    python = exp7243.PythonArchiveController(archive_cap=2)
    native_arm = exp7243.NativeArchiveController(native, archive_cap=2)
    event = {"event_id": "probe", "family_id": "lower_bound", "numeric_value": 9}
    for index in range(12):
        assert native_arm.predict(event) == python.predict(event)
        assert native_arm.energy("accept", event) == python.energy("accept", event)
        tie_ranks = {"probe": 0}
        assert native_arm.select_request([event], tie_ranks) == python.select_request(
            [event], tie_ranks
        )
        release = _release(index, "accept" if index % 3 else "reject")
        python.commit_batch(
            [release], current_cycle=index, expected_parent_hash=python.state_hash()
        )
        native_arm.commit_batch(
            [release], current_cycle=index, expected_parent_hash=native_arm.state_hash()
        )
        assert native_arm.state_bytes() == python.state_bytes()
    assert native_arm.native_call_count > 0
    assert native_arm.native_module_file == str(Path(native.__file__).resolve())
    restored = exp7243.NativeArchiveController.from_state_with_binding(
        native, native_arm.state_dict()
    )
    assert restored.state_hash() == native_arm.state_hash()


def test_scenario_cl_7243_stream_parity_and_fresh_continuation(
    native_extension: Path, native
) -> None:
    """SCENARIO-CL-7243-RESTORE: real stream rows and delayed updates survive restart."""

    rows, checkpoints = exp7243.run_stream_parity(native, stream_limit=2, event_limit=96)
    assert len(rows) == 4
    assert len(checkpoints) == 2
    assert all(row["mismatch_count"] == 0 for row in rows)
    assert all(row["predictions"] and row["energies"] for row in rows)
    assert all(row["queries"] and row["state_hashes"] for row in rows)
    assert all(row["archive_state_hashes"] for row in rows)
    assert {row["implementation"] for row in rows} == {"python", "native_pyo3"}

    receipt = exp7243.run_fresh_process_continuation(native_extension, checkpoints[-1])
    assert receipt["passed"] is True
    assert receipt["mismatch_count"] == 0
    assert receipt["module_file"] == str(native_extension.resolve())
    assert receipt["delayed_updates_continued"] > 0


def test_scenario_rustpy_7243_cost_rows_and_batch_one_gates(native) -> None:
    """SCENARIO-RUSTPY-7243-PAIRED-COST: charge each full boundary component."""

    rows = exp7243.run_cost_benchmark(
        native,
        capacities=(1, 2),
        batch_sizes=(1, 3),
        blocks=2,
        seed=7_243_100,
    )
    assert len(rows) == 16
    required = {
        "python_dispatch_ns",
        "binding_conversion_ns",
        "lookup_ns",
        "query_ns",
        "validation_archive_update_ns",
        "serialization_ns",
        "restore_ns",
        "total_event_ns",
    }
    assert all(required <= set(row["component_ns"]) for row in rows)
    assert all(row["total_block_ns"] > 0 and row["total_event_ns"] > 0 for row in rows)
    summary = exp7243.summarize_cost(rows, capacities=(1, 2), batch_sizes=(1, 3), blocks=2)
    assert len(summary["cells"]) == 4
    assert summary["batch_one_lower_ci95"] == min(
        cell["paired_lower_ci95"] for cell in summary["cells"] if cell["batch_size"] == 1
    )

    fast = exp7243.synthetic_cost_rows(
        capacities=(1, 2), batch_sizes=(1, 3), blocks=2, python_ns=2_000, native_ns=10
    )
    fast_summary = exp7243.summarize_cost(fast, capacities=(1, 2), batch_sizes=(1, 3), blocks=2)
    assert fast_summary["native_archive_cost_value_score"] == 1
    assert fast_summary["nfr_01_10x_met"] is True
    assert fast_summary["research_program_100x_met"] is True
    slow = exp7243.synthetic_cost_rows(
        capacities=(1,), batch_sizes=(1,), blocks=2, python_ns=10, native_ns=20
    )
    assert (
        exp7243.summarize_cost(slow, capacities=(1,), batch_sizes=(1,), blocks=2)[
            "native_archive_cost_value_score"
        ]
        == 0
    )


def test_req_cl_7243_validator_enforces_oracle_classification_and_rows() -> None:
    """REQ-CL-7243: exact parity is circular-positive and a cost miss is null."""

    artifact = exp7243.complete_artifact_fixture_for_test(cost_pass=True)
    assert exp7243.validate_artifact(artifact) == []
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["native_archive_cost_value_score"] == 1

    null = exp7243.complete_artifact_fixture_for_test(cost_pass=False)
    assert exp7243.validate_artifact(null) == []
    assert null["verdict_class"] == "null"
    assert null["native_archive_ready_score"] == 1

    for mutation, expected in (
        (("verdict_class", "positive"), "verdict_class"),
        (("verifier_is_oracle", False), "verifier_is_oracle"),
        (("native_archive_ready_score", 0), "ready_score"),
    ):
        changed = deepcopy(artifact)
        changed[mutation[0]] = mutation[1]
        changed["reproducibility_checksum"] = exp7243.artifact_checksum(changed)
        assert expected in exp7243.validate_artifact(changed)

    changed = deepcopy(artifact)
    changed["cost_rows"].pop()
    changed["rows"] = [*changed["parity_rows"], *changed["cost_rows"]]
    changed["reproducibility_checksum"] = exp7243.artifact_checksum(changed)
    assert "cost_rows" in exp7243.validate_artifact(changed)

    failed = exp7243.check("missing", "upstream", "field", 1, None, False)
    blocked = exp7243.blocked_artifact_for_test(failed)
    assert exp7243.validate_artifact(blocked) == []
    assert blocked["status"] == "blocked"
    assert blocked["rows"] == []


def test_req_cl_7243_thin_entrypoint_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CL-7243: the runnable script contains no experiment implementation."""

    script = exp7243.REPO_ROOT / "scripts/experiments/experiment_7243_v637_native_memory.py"
    calls: list[list[str] | None] = []
    monkeypatch.setattr(exp7243, "main", lambda argv=None: calls.append(argv) or 0)
    monkeypatch.setattr(sys, "argv", [str(script), "--date", exp7243.RUN_DATE])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(script), run_name="__main__")
    assert exit_info.value.code == 0
    assert calls == [None]
    assert len(script.read_text(encoding="utf-8").splitlines()) <= 12


def test_req_cl_7243_cli_validation_and_atomic_write(tmp_path: Path) -> None:
    """REQ-CL-7243: validation is read-only and terminal writes are atomic."""

    artifact = exp7243.complete_artifact_fixture_for_test(cost_pass=False)
    output = tmp_path / "artifact.json"
    receipt = exp7243.atomic_write(output, artifact)
    assert receipt["atomic_replace"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert exp7243.main(["--validate", str(output)]) == 0
    artifact["run_date"] = "bad"
    output.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp7243.main(["--validate", str(output)]) == 2
    assert exp7243.main(["--date", "bad", "--output", str(output)]) == 2


def test_req_cl_7243_defensive_helpers_and_native_boundaries(
    tmp_path: Path, native, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7243: malformed sources, state, batches, and rosters fail closed."""

    missing = tmp_path / "missing.json"
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    assert exp7243.read_object(missing) == {}
    assert exp7243.read_object(scalar) == {}
    assert exp7243._task_contract(tmp_path) is None
    (tmp_path / exp7243.ROADMAP_RELATIVE).write_text("bad: [", encoding="utf-8")
    assert exp7243._task_contract(tmp_path) is None
    (tmp_path / exp7243.ROADMAP_RELATIVE).write_text("tasks: []", encoding="utf-8")
    assert exp7243._task_contract(tmp_path) is None
    assert exp7243._receipt_hashes_match(tmp_path, {}) is False
    assert (
        exp7243._receipt_hashes_match(tmp_path, {"stream_receipts": dict.fromkeys("abcd", "bad")})
        is False
    )
    receipts = {name: {"path": "missing", "sha256": "sha256:bad"} for name in ("a", "b", "c", "d")}
    assert exp7243._receipt_hashes_match(tmp_path, {"stream_receipts": receipts}) is False

    active = exp7243.NativePackedActive(
        native,
        exp7243.exp7226.PackedBeliefController().state_dict(),
        {"binding_conversion_ns": 0, "native_kernel_ns": 0, "native_call_count": 0},
    )
    invalid = {"event_id": "bad", "family_id": "bad", "numeric_value": 0}
    assert active.predict(invalid) == ("abstain", 0.0)
    assert active.energy("bad", invalid)["status"] == "unknown_label"
    assert active.energy("accept", invalid)["status"] == "unknown_input"
    empty_state = exp7243.exp7226.PackedBeliefController.from_survivors(
        {"lower_bound": set()}
    ).state_dict()
    empty = exp7243.NativePackedActive(
        native,
        empty_state,
        {"binding_conversion_ns": 0, "native_kernel_ns": 0, "native_call_count": 0},
    )
    event = {"event_id": "empty", "family_id": "lower_bound", "numeric_value": 0}
    assert empty.energy("accept", event)["status"] == "empty"
    masks = {family: 1 for family in exp7243.exp7226.FAMILIES}
    assert exp7243.NativePackedActive.from_masks(
        native,
        masks,
        {"binding_conversion_ns": 0, "native_kernel_ns": 0, "native_call_count": 0},
    ).state_dict()
    active._native = SimpleNamespace(serialize_state=lambda: "{}")
    with pytest.raises(RuntimeError, match="native_shadow_state_mismatch"):
        active.state_dict()

    controller = exp7243.NativeArchiveController(native, archive_cap=1)
    release = _release(40)
    with pytest.raises(exp7243.exp7226.CommitRejected, match="stale_parent"):
        controller._active().commit_batch(
            [release], current_cycle=40, expected_parent_hash="sha256:bad"
        )
    first = controller._active()
    first.commit_batch([release], current_cycle=40, expected_parent_hash=first.state_hash())
    with pytest.raises(exp7243.exp7226.CommitRejected, match="duplicate_release"):
        first.commit_batch([release], current_cycle=40, expected_parent_hash=first.state_hash())
    durable = tmp_path / "native-controller.json"
    controller.save(durable)
    controller.commit_batch(
        [_release(41)],
        current_cycle=41,
        expected_parent_hash=controller.state_hash(),
        state_path=durable,
    )
    assert controller._load_durable_state(durable).state_hash() == controller.state_hash()
    active_durable = tmp_path / "native-active.json"
    writable_active = controller._active()
    writable_active.commit_batch(
        [_release(42)],
        current_cycle=42,
        expected_parent_hash=writable_active.state_hash(),
        state_path=active_durable,
    )
    assert json.loads(active_durable.read_text(encoding="utf-8"))["version"] > 0
    masks = {family: 1 for family in exp7243.exp7226.FAMILIES}
    assert controller._active_from_masks(masks).state_dict()

    with pytest.raises(ValueError, match="roster"):
        exp7243.run_cost_benchmark(native, capacities=(), batch_sizes=(1,), blocks=1)
    with pytest.raises(ValueError, match="paired ratios"):
        exp7243._bootstrap_lower([], exp7243.RANDOM_SEED)
    with pytest.raises(ValueError, match="incomplete paired"):
        exp7243.summarize_cost([], capacities=(1,), batch_sizes=(1,), blocks=1)

    native_identity = exp7243._native_identity(
        exp7243.REPO_ROOT,
        Path(native.__file__),
        native,
        {},
        build_duration_s=1.0,
        import_duration_s=0.1,
    )
    assert native_identity["compiled_execution"] is True
    fast = exp7243.synthetic_cost_rows(
        capacities=(1,), batch_sizes=(1,), blocks=1, python_ns=100, native_ns=10
    )
    assert exp7243._amortization(fast, native_identity)["break_even_events"] is not None
    slow = exp7243.synthetic_cost_rows(
        capacities=(1,), batch_sizes=(1,), blocks=1, python_ns=10, native_ns=100
    )
    assert exp7243._amortization(slow, native_identity)["break_even_events"] is None


def test_req_cl_7243_native_build_and_atomic_cleanup_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-RUSTPY-7243: build/install failures remain explicit and clean."""

    target = tmp_path / exp7243.TARGET_RELATIVE / "release"
    target.mkdir(parents=True)
    library = target / "libcarnot_python.so"
    library.write_bytes(b"compiled-extension")
    environment = {
        "PYO3_PYTHON": sys.executable,
        "CARGO_TARGET_DIR": str(tmp_path / exp7243.TARGET_RELATIVE),
    }
    monkeypatch.setattr(
        exp7243.exp7217,
        "interpreter_build_environment",
        lambda *_: environment,
    )
    monkeypatch.setattr(
        exp7243.exp7217,
        "_stream_process",
        lambda *_, **__: {"command": ["cargo", "build"]},
    )
    destination, receipt = exp7243.build_native_extension(tmp_path)
    assert destination.read_bytes() == library.read_bytes()
    assert receipt["loaded_copy"] == str(destination.resolve())

    library.unlink()
    with pytest.raises(RuntimeError, match="native build output missing"):
        exp7243.build_native_extension(tmp_path, show_progress=False)
    library.write_bytes(b"compiled-extension")
    with (
        patch.object(exp7243.sysconfig, "get_config_var", return_value=None),
        pytest.raises(RuntimeError, match="extension suffix unavailable"),
    ):
        exp7243.build_native_extension(tmp_path, show_progress=False)

    with (
        patch.object(exp7243.os, "replace", side_effect=OSError("replace failed")),
        pytest.raises(OSError, match="replace failed"),
    ):
        exp7243.build_native_extension(tmp_path, show_progress=False)
    load_dir = tmp_path / exp7243.LOAD_RELATIVE
    assert not list(load_dir.glob("*.tmp"))

    artifact_path = tmp_path / "failed-atomic.json"
    with (
        patch.object(exp7243.os, "replace", side_effect=OSError("publish failed")),
        pytest.raises(OSError, match="publish failed"),
    ):
        exp7243.atomic_write(
            artifact_path,
            exp7243.complete_artifact_fixture_for_test(cost_pass=False),
        )
    assert not list(tmp_path.glob("*.tmp"))


def test_scenario_rustpy_7243_cost_heartbeat_is_truthful(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-RUSTPY-7243-PAIRED-COST: long blocks emit measured progress."""

    measured = {
        "total_block_ns": 2,
        "total_event_ns": 2,
        "component_ns": {},
        "native_kernel_ns": 0,
        "native_call_count": 0,
        "serialized_bytes": 1,
        "final_state_hash": "sha256:fixture",
    }
    with (
        patch.object(exp7243, "_seed_cost_state", return_value={}),
        patch.object(exp7243, "_measure_cost_arm", return_value=measured),
        patch.object(exp7243.time, "monotonic", side_effect=[0.0, 61.0, 62.0, 63.0]),
    ):
        rows = exp7243.run_cost_benchmark(
            SimpleNamespace(), capacities=(1,), batch_sizes=(1,), blocks=1
        )
    assert len(rows) == 2
    assert "completed_blocks=1/1" in capsys.readouterr().out


def test_req_cl_7243_validator_defensive_branches(tmp_path: Path, native_extension: Path) -> None:
    """REQ-CL-7243: validator attacks cannot retain a passing terminal claim."""

    assert exp7243.validate_artifact({}) == ["missing_fields"]
    artifact = exp7243.complete_artifact_fixture_for_test(cost_pass=True)
    artifact["source_artifact_hashes"] = {
        str(native_extension): exp7243.sha256_file(native_extension)
    }
    artifact["native_binary_receipt"]["module_file"] = str(native_extension)
    artifact["native_binary_receipt"]["binary_sha256"] = exp7243.sha256_file(native_extension)
    artifact["reproducibility_checksum"] = exp7243.artifact_checksum(artifact)
    assert exp7243.validate_artifact(artifact, check_files=True) == []
    artifact["source_artifact_hashes"][str(native_extension)] = "sha256:bad"
    artifact["native_binary_receipt"]["binary_sha256"] = "sha256:bad"
    artifact["reproducibility_checksum"] = exp7243.artifact_checksum(artifact)
    errors = exp7243.validate_artifact(artifact, check_files=True)
    assert "source_artifact_hashes" in errors
    assert "native_binary_hash" in errors

    malformed = exp7243.complete_artifact_fixture_for_test(cost_pass=True)
    malformed["cost_rows"] = [object()]
    malformed["rows"] = malformed["parity_rows"]
    malformed["reproducibility_checksum"] = "uncomputable"
    errors = exp7243.validate_artifact(malformed)
    assert "reproducibility_checksum" in errors
    assert "cost_summary" in errors

    failed = exp7243.blocked_artifact_for_test(
        exp7243.check("missing", "source", "field", 1, None, False)
    )
    failed.update(
        {
            "verdict_class": "positive",
            "inference_substrate": "bad",
            "inference_substrate_class": "bad",
            "rows": [{}],
            "native_archive_ready_score": 1,
            "gate_check_summary": {},
            "honest_verdict": "bad",
        }
    )
    failed["reproducibility_checksum"] = exp7243.artifact_checksum(failed)
    blocked_errors = exp7243.validate_artifact(failed)
    assert {
        "blocked_class",
        "blocked_substrate",
        "blocked_substrate_class",
        "blocked_rows",
        "blocked_ready",
        "blocked_gate",
        "blocked_verdict",
    } <= set(blocked_errors)


def test_req_cl_7243_orchestration_complete_blocked_and_errors(
    tmp_path: Path,
    native_extension: Path,
    native,
) -> None:
    """REQ-CL-7243: orchestration writes complete or blocked artifacts only."""

    paths = exp7243.ExperimentPaths.under(tmp_path)
    passed = exp7243.check("fixture", "fixture", "field", 1, 1, True)
    failed = exp7243.check("fixture", "fixture", "field", 1, 0, False)
    sources = {
        "exp7240": {"recurrence_fixture_ready_score": 1},
        "exp7217": {"native_abi_ready_score": 1},
        "exp7230": {
            "flagged_adversarial": True,
            "verifier_is_oracle": True,
            "verdict_class": "positive",
            "native_cost_value_score": 1,
        },
    }
    checkpoint = exp7243._continuation_fixture(
        exp7243.PythonArchiveController().state_dict(), exp7243.STREAM_SEEDS[-1]
    )
    parity = exp7243._fixture_parity_rows()[:-1]
    fresh = exp7243._fixture_parity_rows()[-1]
    costs = exp7243.synthetic_cost_rows(
        capacities=exp7243.ARCHIVE_CAPACITIES,
        batch_sizes=exp7243.BATCH_SIZES,
        blocks=exp7243.PAIRED_BLOCKS,
        python_ns=10,
        native_ns=20,
    )
    build_receipt = {
        "command": ["cargo", "build"],
        "PYO3_PYTHON": sys.executable,
        "CARGO_TARGET_DIR": str(tmp_path),
    }
    hashes = {str(exp7243.HISTORY_RELATIVE): exp7243.EXPECTED_EXP7230_SHA256}
    with (
        patch.object(exp7243, "collect_preconditions", return_value=([passed], hashes, sources)),
        patch.object(
            exp7243,
            "build_native_extension",
            return_value=(native_extension, build_receipt),
        ),
        patch.object(exp7243.exp7230, "load_native_extension", return_value=native),
        patch.object(exp7243, "run_stream_parity", return_value=(parity, [checkpoint])),
        patch.object(exp7243, "run_fresh_process_continuation", return_value=fresh),
        patch.object(exp7243, "run_cost_benchmark", return_value=costs),
    ):
        artifact = exp7243.build_artifact(exp7243.REPO_ROOT, paths)
    assert artifact["status"] == "complete"
    assert artifact["verdict_class"] == "null"
    assert exp7243.validate_artifact(artifact) == []

    with patch.object(exp7243, "collect_preconditions", return_value=([failed], {}, sources)):
        blocked = exp7243.build_artifact(exp7243.REPO_ROOT, paths)
    assert blocked["status"] == "blocked"
    assert exp7243.validate_artifact(blocked) == []

    broken_fresh = deepcopy(fresh)
    broken_fresh["passed"] = False
    with (
        patch.object(exp7243, "collect_preconditions", return_value=([passed], hashes, sources)),
        patch.object(
            exp7243,
            "build_native_extension",
            return_value=(native_extension, build_receipt),
        ),
        patch.object(exp7243.exp7230, "load_native_extension", return_value=native),
        patch.object(exp7243, "run_stream_parity", return_value=(parity, [checkpoint])),
        patch.object(exp7243, "run_fresh_process_continuation", return_value=broken_fresh),
    ):
        with pytest.raises(RuntimeError, match="parity"):
            exp7243.build_artifact(exp7243.REPO_ROOT, paths)

    with pytest.raises(ValueError, match="run date"):
        exp7243.run_experiment(exp7243.REPO_ROOT, paths.artifact, "bad")
    fixture = exp7243.complete_artifact_fixture_for_test(cost_pass=False)
    with (
        patch.object(exp7243, "build_artifact", return_value=fixture),
        patch.object(exp7243, "validate_artifact", return_value=[]),
    ):
        assert exp7243.run_experiment(tmp_path, paths.artifact, exp7243.RUN_DATE) == fixture
    with (
        patch.object(exp7243, "build_artifact", return_value=fixture),
        patch.object(exp7243, "validate_artifact", return_value=["bad"]),
    ):
        with pytest.raises(ValueError, match="invalid Exp7243"):
            exp7243.run_experiment(tmp_path, paths.artifact, exp7243.RUN_DATE)


def test_req_cl_7243_cli_success_errors_and_module_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7243: CLI paths preserve exit status and read-only validation."""

    output = tmp_path / "result.json"
    fixture = exp7243.complete_artifact_fixture_for_test(cost_pass=False)
    output.write_text(json.dumps(fixture), encoding="utf-8")
    with patch.object(exp7243, "run_experiment", return_value=fixture) as run:
        assert exp7243.main(["--date", exp7243.RUN_DATE, "--output", "relative.json"]) == 0
        assert run.call_args.args[1] == exp7243.REPO_ROOT / "relative.json"
    with patch.object(exp7243, "run_experiment", side_effect=RuntimeError("owned")):
        assert exp7243.main(["--date", exp7243.RUN_DATE, "--output", str(output)]) == 2
    assert exp7243.main(["--validate", str(tmp_path / "missing.json")]) == 2

    monkeypatch.setattr(sys, "argv", ["experiment_7243", "--validate", str(output)])
    with pytest.warns(RuntimeWarning), pytest.raises(SystemExit) as exit_info:
        runpy.run_module("carnot.experiment_7243_v637_native_memory", run_name="__main__")
    assert exit_info.value.code == 0
