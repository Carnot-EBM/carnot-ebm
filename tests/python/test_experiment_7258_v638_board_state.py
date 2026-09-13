"""Tests for the V638 host-only board-state receipt.

Spec: REQ-ISING-7258 and SCENARIO-ISING-7258-PREFLIGHT through
SCENARIO-ISING-7258-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7258_v638_board_state as experiment


ROOT = Path(__file__).resolve().parents[2]


def _valid_operator_receipt(path: Path) -> Path:
    """Create one explicit operator change for the positive parser branch."""

    path.write_text(
        json.dumps(
            {
                "exists": True,
                "receipt_date": "20260913",
                "source": (
                    "operator directive 2026-09-13T12:00:00Z: GateMate DirtyJTAG cable changed"
                ),
                "operator_authored": True,
                "board": "Cologne Chip GateMate A1-EVB-2M",
                "usb_dirtyjtag": "1209:c0ca DirtyJTAG",
                "dirtyjtag": "new DirtyJTAG cable path",
                "changes": [
                    {
                        "field": "dirtyjtag",
                        "description": "operator changed the DirtyJTAG cable path",
                    }
                ],
                "action": "detect",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_req_ising_7258_authenticates_current_board_chains(tmp_path: Path) -> None:
    """REQ-ISING-7258: current immutable board transcripts must hash-match."""

    checks, hashes, upstreams = experiment.collect_preconditions(
        ROOT, experiment.ExperimentPaths.under(tmp_path)
    )
    assert all(row["passed"] for row in checks)
    assert upstreams["receipt"]["experiment_id"] == 7244
    assert upstreams["kv260_row"]["terminal_criterion"] == experiment.KV260_TERMINAL_CRITERION
    assert upstreams["polarfire_row"]["processor_class"] == "cpu"
    assert hashes[experiment.UPSTREAM_PATH.as_posix()].startswith("sha256:")
    assert hashes[experiment.POLARFIRE_RAW_PATH.as_posix()].startswith("sha256:")


def test_scenario_ising_7258_gatemate_absence_is_explicit(tmp_path: Path) -> None:
    """SCENARIO-ISING-7258-GATEMATE: missing state keeps all commands at zero."""

    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    receipt = experiment.search_gatemate_operator_receipts(
        ROOT,
        tmp_path / "search.json",
        candidate_paths=[missing, malformed],
    )
    assert receipt["exists"] is False
    assert receipt["source_path"] is None
    assert receipt["author_evidence"] is None
    assert receipt["date_evidence"] is None
    assert receipt["changed_conditions"] == {}
    assert receipt["evidence_hash"] is None
    assert receipt["observed_missing_receipt"].startswith("no operator-authored")
    assert receipt["newly_enabled_next_action"] is None
    assert receipt["hardware_operations_issued"] == []
    raw = json.loads((tmp_path / "search.json").read_text(encoding="utf-8"))
    assert len(raw["candidate_rows"]) == 2
    assert all(row["valid"] is False for row in raw["candidate_rows"])


def test_scenario_ising_7258_gatemate_change_only_enables_future_action(
    tmp_path: Path,
) -> None:
    """SCENARIO-ISING-7258-GATEMATE: changed state never runs the action here."""

    candidate = _valid_operator_receipt(tmp_path / "operator.json")
    receipt = experiment.search_gatemate_operator_receipts(
        ROOT,
        tmp_path / "search.json",
        candidate_paths=[candidate],
    )
    assert receipt["exists"] is True
    assert receipt["source_path"] == str(candidate)
    assert receipt["author_evidence"] is True
    assert receipt["date_evidence"] == "20260913"
    assert receipt["changed_conditions"] == {
        "dirtyjtag": "operator changed the DirtyJTAG cable path"
    }
    assert receipt["evidence_hash"] == experiment.sha256_file(candidate)
    assert receipt["newly_enabled_next_action"] == (
        "one bounded GateMate DirtyJTAG detect in a future hardware task"
    )
    assert receipt["hardware_operations_issued"] == []


def test_scenario_ising_7258_negative_fixtures_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-ISING-7258-FIXTURES: both tiny negative controls reject."""

    path = tmp_path / "negative-fixtures.json"
    receipt = experiment.write_negative_fixture_receipt(path)
    assert receipt["fixture_count"] == 2
    assert {row["fixture"] for row in receipt["rows"]} == {
        "missing_receipt",
        "malformed_receipt",
    }
    assert all(row["accepted"] is False for row in receipt["rows"])
    assert all(row["failed_closed"] is True for row in receipt["rows"])
    assert receipt["hardware_operations_issued"] == []
    assert receipt["external_commands_issued"] == []
    assert experiment.sha256_file(path).startswith("sha256:")


def test_req_ising_7258_historical_models_stay_in_sidecar(tmp_path: Path) -> None:
    """REQ-ISING-7258: source model history is separate from current counters."""

    path = tmp_path / "history.json"
    receipt = experiment.write_historical_model_receipt(ROOT, path)
    assert receipt["current_invocation_model_count"] == 0
    assert receipt["current_invocation_model_specs"] == []
    assert receipt["source_receipts"]
    assert all(row["source_sha256"].startswith("sha256:") for row in receipt["source_receipts"])
    assert experiment.sha256_file(path).startswith("sha256:")


def test_scenario_ising_7258_boards_preserve_exact_boundaries(tmp_path: Path) -> None:
    """SCENARIO-ISING-7258-BOARDS: fabric, CPU, and physical state stay separate."""

    paths = experiment.ExperimentPaths.under(tmp_path)
    _, _, upstreams = experiment.collect_preconditions(ROOT, paths)
    operator = experiment.search_gatemate_operator_receipts(
        ROOT, paths.operator_search, candidate_paths=[tmp_path / "missing.json"]
    )
    rows = experiment.build_board_rows(ROOT, upstreams, operator)
    by_board = {row["board"]: row for row in rows}
    assert set(by_board) == {"KV260", "GateMate", "PolarFire"}
    assert by_board["KV260"]["terminal_criterion"] == experiment.KV260_TERMINAL_CRITERION
    assert by_board["KV260"]["processor_class"] == "fpga_fabric"
    assert by_board["GateMate"]["disposition"] == "blocked_changed_physical_state"
    assert "operator-authored" in by_board["GateMate"]["observed_missing_receipt"]
    assert by_board["PolarFire"]["terminal_criterion"] == experiment.POLARFIRE_TERMINAL_CRITERION
    assert by_board["PolarFire"]["processor_class"] == "cpu"
    assert by_board["PolarFire"]["programmable_logic_sampling_observed"] is False
    assert all(row["hardware_operations_issued"] == [] for row in rows)
    assert all(row["censored"] is False for row in rows)
    reduced = experiment.reduce_board_rows(rows)
    assert reduced["board_disposition_complete_score"] == 1
    assert reduced["hardware_command_count"] == 0


def test_scenario_ising_7258_artifact_is_complete_host_aggregation(tmp_path: Path) -> None:
    """SCENARIO-ISING-7258-ARTIFACT: three dispositions complete the receipt."""

    paths = experiment.ExperimentPaths.under(tmp_path)
    artifact = experiment.build_artifact(ROOT, paths)
    assert paths.checkpoint.is_file()
    assert not paths.artifact.exists()
    assert paths.operator_search.is_file()
    assert paths.historical_models.is_file()
    assert paths.negative_fixtures.is_file()
    assert experiment.validate_artifact(artifact) == []
    assert artifact["status"] == "complete"
    assert artifact["board_disposition_complete_score"] == 1
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["model_invocation_count"] == 0
    assert artifact["model_load_count"] == 0
    assert artifact["generation_count"] == 0
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["hardware_operations_issued"] == []
    assert artifact["gate_check_summary"]["board_blocks"][0]["verdict"] == (
        "blocked_changed_physical_state"
    )
    assert artifact["honest_verdict"].startswith("complete:")


def test_scenario_ising_7258_positive_operator_receipt_validates(tmp_path: Path) -> None:
    """SCENARIO-ISING-7258-GATEMATE: valid change produces a future-action row."""

    candidate = _valid_operator_receipt(tmp_path / "operator.json")
    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        candidate_paths=[candidate],
    )
    gate = next(row for row in artifact["board_rows"] if row["board"] == "GateMate")
    assert gate["disposition"] == "changed_physical_state_future_action_enabled"
    assert gate["metric"] is True
    assert artifact["gate_check_summary"]["board_blocks"] == []
    assert experiment.validate_artifact(artifact) == []


def test_scenario_ising_7258_preflight_block_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ISING-7258-PREFLIGHT: a failed contract emits no board rows."""

    monkeypatch.setattr(experiment, "_task_contract", lambda _root: None)
    artifact = experiment.build_artifact(ROOT, experiment.ExperimentPaths.under(tmp_path))
    assert artifact["status"] == "blocked_external_precondition"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate"] == "blocked_no_run"
    assert artifact["board_rows"] == []
    assert artifact["hardware_operations_issued"] == []
    assert artifact["gate_check_summary"]["failed_check"] == "roadmap_task_contract"
    assert experiment.validate_artifact(artifact) == []


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ({"board_disposition_complete_score": 0}, "board_reducer"),
        ({"hardware_operations_issued": ["jtag"]}, "hardware_operations"),
        ({"MODEL_SPECS": ["model"]}, "model_declaration"),
        ({"model_load_count": 1}, "model_declaration"),
        ({"inference_substrate": "cpu_exact_solver_or_simulator"}, "substrate"),
        ({"reproducibility_checksum": "sha256:bad"}, "reproducibility_checksum"),
    ),
)
def test_req_ising_7258_reducer_rejects_mutations(
    tmp_path: Path, mutation: dict[str, Any], error: str
) -> None:
    """REQ-ISING-7258: the independent reducer rejects unsafe mutations."""

    artifact = experiment.build_artifact(ROOT, experiment.ExperimentPaths.under(tmp_path))
    changed = deepcopy(artifact)
    changed.update(mutation)
    if "reproducibility_checksum" not in mutation:
        changed["reproducibility_checksum"] = experiment.artifact_checksum(changed)
    assert error in experiment.validate_artifact(changed)


def test_req_ising_7258_validation_log_receipts_are_hashed(tmp_path: Path) -> None:
    """REQ-ISING-7258: command receipts retain exit status and exact log hash."""

    log_dir = tmp_path / "validation"
    log_dir.mkdir()
    passed = log_dir / "focused.log"
    passed.write_text("$ pytest focused\n1 passed\n[exit_code] 0\n", encoding="utf-8")
    malformed = log_dir / "malformed.log"
    malformed.write_text("no command marker\n", encoding="utf-8")
    rows = experiment.read_validation_receipts(log_dir)
    assert rows[0] == {
        "command": "pytest focused",
        "exit_code": 0,
        "log_path": str(passed),
        "log_hash": experiment.sha256_file(passed),
    }
    assert rows[1]["command"] == "unknown"
    assert rows[1]["exit_code"] is None


def test_req_ising_7258_atomic_run_and_cli_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ISING-7258: the writer and read-only CLI use the independent reducer."""

    paths = experiment.ExperimentPaths.under(tmp_path)
    artifact = experiment.run_experiment(ROOT, paths)
    assert json.loads(paths.artifact.read_text(encoding="utf-8")) == artifact
    assert experiment.main(["--validate", str(paths.artifact)]) == 0
    assert "validation_passed" in capsys.readouterr().out
    assert experiment.main(["--date", "20260912", "--root", str(ROOT)]) == 2
    with pytest.raises(ValueError, match="invalid Exp7258 artifact"):
        experiment.atomic_write(tmp_path / "invalid.json", {})


def test_req_ising_7258_cli_writes_blocked_result_under_private_root(tmp_path: Path) -> None:
    """REQ-ISING-7258: relative private output remains fail-closed."""

    assert experiment.main(["--root", str(tmp_path), "--output", "private/result.json"]) == 0
    saved = json.loads((tmp_path / "private/result.json").read_text(encoding="utf-8"))
    assert saved["status"] == "blocked_external_precondition"
    assert saved["board_rows"] == []


def test_req_ising_7258_task_contract_and_row_reducer_fail_closed(tmp_path: Path) -> None:
    """REQ-ISING-7258: malformed contracts and row sets cannot pass by default."""

    assert experiment._task_contract(tmp_path) is None
    (tmp_path / experiment.ROADMAP_PATH).write_text("[]\n", encoding="utf-8")
    assert experiment._task_contract(tmp_path) is None
    (tmp_path / experiment.ROADMAP_PATH).write_text("tasks: []\n", encoding="utf-8")
    assert experiment._task_contract(tmp_path) is None
    assert experiment.reduce_board_rows([])["board_disposition_complete_score"] == 0
    assert experiment.validate_artifact({})[0].startswith("missing_fields:")


def test_req_ising_7258_defensive_reducer_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ISING-7258: malformed metadata and invalid builds remain explicit."""

    assert experiment._changed_conditions(
        {"raw_receipt": {"changes": [None], "power": "power cycled"}}
    ) == {"power": "power cycled"}

    paths = experiment.ExperimentPaths.under(tmp_path)
    paths.validation_dir.mkdir(parents=True)
    log = paths.validation_dir / "focused.log"
    log.write_text("$ pytest focused\n1 passed\n[exit_code] 0\n", encoding="utf-8")
    baseline_dir = paths.validation_dir.parent / "baseline_failures"
    baseline_dir.mkdir()
    baseline = baseline_dir / "full.log"
    baseline.write_text("$ pytest all\nfailures\n[exit_code] 1\n", encoding="utf-8")
    artifact = experiment.build_artifact(ROOT, paths)
    assert any(row["command"] == "pytest focused" for row in artifact["validation_receipts"])
    assert artifact["baseline_validation_failures"][0]["exit_code"] == 1
    assert str(log) in artifact["source_artifact_hashes"]
    assert str(baseline) in artifact["source_artifact_hashes"]

    invalid_time = deepcopy(artifact)
    invalid_time["started_at_utc"] = "not-a-time"
    invalid_time["reproducibility_checksum"] = experiment.artifact_checksum(invalid_time)
    assert "started_at_utc" in experiment.validate_artifact(invalid_time)

    invalid_json = deepcopy(artifact)
    invalid_json["validation_receipts"] = {object()}
    assert "reproducibility_checksum" in experiment.validate_artifact(invalid_json)

    monkeypatch.setattr(experiment, "build_artifact", lambda _root, _paths: {})
    monkeypatch.setattr(experiment, "validate_artifact", lambda _artifact, **_kwargs: ["bad"])
    with pytest.raises(ValueError, match="invalid Exp7258 artifact"):
        experiment.run_experiment(ROOT, paths)


def test_req_ising_7258_missing_cli_validation_fails_closed(tmp_path: Path) -> None:
    """REQ-ISING-7258: a missing validation target cannot report success."""

    assert experiment.main(["--validate", str(tmp_path / "missing.json")]) == 2


def test_req_ising_7258_thin_entrypoint() -> None:
    """REQ-ISING-7258: the experiment script delegates to the tested module."""

    source = (ROOT / experiment.ENTRYPOINT_PATH).read_text(encoding="utf-8")
    assert "experiment_7258_v638_board_state import main" in source
    assert source.count("main()") == 1
