"""Tests for the V640 host-only board-state continuation.

Spec: REQ-ISING-7286 and SCENARIO-ISING-7286-PREFLIGHT through
SCENARIO-ISING-7286-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7286_v640_board_state as experiment


ROOT = Path(__file__).resolve().parents[2]


def _valid_operator_receipt(path: Path) -> Path:
    """Create a real-shaped later operator change for the positive parser branch."""

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


def test_req_ising_7286_authenticates_current_receipt_chain(tmp_path: Path) -> None:
    """REQ-ISING-7286: Exp7272 and its original transcripts authenticate."""

    checks, hashes, upstreams = experiment.collect_preconditions(
        ROOT, experiment.ExperimentPaths.under(tmp_path)
    )
    assert all(row["passed"] for row in checks)
    assert upstreams["receipt"]["experiment_id"] == 7272
    assert upstreams["kv260_row"]["processor_class"] == "fpga_fabric"
    assert upstreams["polarfire_row"]["processor_class"] == "cpu"
    assert upstreams["reference_observation"]["all_match"] is True
    assert hashes[experiment.UPSTREAM_PATH.as_posix()].startswith("sha256:")


def test_scenario_ising_7286_gatemate_fixtures_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-ISING-7286-FIXTURES: bad receipts authorize no command."""

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    receipt = experiment.search_gatemate_operator_receipts(
        ROOT,
        tmp_path / "search.json",
        candidate_paths=[tmp_path / "missing.json", malformed],
    )
    assert receipt["exists"] is False
    assert receipt["source_path"] is None
    assert receipt["changed_conditions"] == {}
    assert receipt["observed_missing_receipt"] == experiment.MISSING_RECEIPT
    assert receipt["hardware_operations_issued"] == []
    assert all(
        row["valid"] is False
        for row in json.loads((tmp_path / "search.json").read_text())["candidate_rows"]
    )


def test_scenario_ising_7286_gatemate_change_only_enables_future_probe(
    tmp_path: Path,
) -> None:
    """SCENARIO-ISING-7286-GATEMATE: a later change only enables future work."""

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
    assert receipt["evidence_hash"] == experiment.sha256_file(candidate)
    assert receipt["changed_conditions"] == {
        "dirtyjtag": "operator changed the DirtyJTAG cable path"
    }
    assert receipt["newly_enabled_next_action"] == experiment.GATEMATE_FUTURE_ACTION
    assert receipt["hardware_operations_issued"] == []


def test_scenario_ising_7286_boards_preserve_distinct_scopes(tmp_path: Path) -> None:
    """SCENARIO-ISING-7286-BOARDS: fabric, CPU, and physical state stay separate."""

    paths = experiment.ExperimentPaths.under(tmp_path)
    _, _, upstreams = experiment.collect_preconditions(ROOT, paths)
    operator = experiment.search_gatemate_operator_receipts(
        ROOT, paths.operator_search, candidate_paths=[tmp_path / "missing.json"]
    )
    rows = experiment.build_board_rows(ROOT, upstreams, operator)
    by_board = {row["board"]: row for row in rows}
    assert set(by_board) == {"KV260", "GateMate", "PolarFire"}
    assert by_board["KV260"]["processor_class"] == "fpga_fabric"
    assert by_board["GateMate"]["disposition"] == "blocked_changed_physical_state"
    assert by_board["GateMate"]["exact_next_condition"] == experiment.GATEMATE_OPERATOR_ACTION
    assert by_board["PolarFire"]["processor_class"] == "cpu"
    assert by_board["PolarFire"]["programmable_logic_sampling_observed"] is False
    assert all(row["hardware_operations_issued"] == [] for row in rows)
    reduced = experiment.reduce_board_rows(rows)
    assert reduced["board_count"] == 3
    assert reduced["board_disposition_complete_score"] == 1
    assert reduced["hardware_command_count"] == 0
    assert reduced["row_hashes_match"] is True


def test_scenario_ising_7286_artifact_is_complete_host_aggregation(
    tmp_path: Path,
) -> None:
    """SCENARIO-ISING-7286-ARTIFACT: visibility completes despite row block."""

    paths = experiment.ExperimentPaths.under(tmp_path)
    artifact = experiment.build_artifact(ROOT, paths, candidate_paths=[tmp_path / "missing.json"])
    assert paths.checkpoint.is_file()
    assert not paths.artifact.exists()
    assert paths.historical_models.is_file()
    assert paths.negative_fixtures.is_file()
    assert experiment.validate_artifact(artifact, root=ROOT) == []
    assert artifact["status"] == "complete"
    assert artifact["board_disposition_complete_score"] == 1
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["hardware_operations_issued"] == []
    assert artifact["gate_check_summary"]["board_blocks"][0]["verdict"] == (
        "blocked_changed_physical_state"
    )
    assert artifact["honest_verdict"].startswith("complete:")


def test_scenario_ising_7286_valid_change_artifact_still_issues_nothing(
    tmp_path: Path,
) -> None:
    """SCENARIO-ISING-7286-GATEMATE: a valid change keeps this task read-only."""

    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        candidate_paths=[_valid_operator_receipt(tmp_path / "operator.json")],
    )
    gate = next(row for row in artifact["board_rows"] if row["board"] == "GateMate")
    assert gate["disposition"] == "changed_physical_state_future_action_enabled"
    assert gate["operator_author_evidence"] is True
    assert gate["metric"] is True
    assert artifact["gate_check_summary"]["board_blocks"] == []
    assert experiment.validate_artifact(artifact, root=ROOT) == []


def test_scenario_ising_7286_preflight_block_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ISING-7286-PREFLIGHT: failed intake has no success-shaped rows."""

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
        ({"invocation_counts": {"model_loads_attempted": 1}}, "invocation_counts"),
        ({"inference_substrate": "cpu_exact_solver_or_simulator"}, "substrate"),
        ({"reproducibility_checksum": "sha256:bad"}, "reproducibility_checksum"),
    ),
)
def test_req_ising_7286_validator_rejects_mutations(
    tmp_path: Path, mutation: dict[str, Any], error: str
) -> None:
    """REQ-ISING-7286: the inherited validator rejects unsafe mutations."""

    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        candidate_paths=[tmp_path / "missing.json"],
    )
    changed = deepcopy(artifact)
    changed.update(mutation)
    if "reproducibility_checksum" not in mutation:
        changed["reproducibility_checksum"] = experiment.artifact_checksum(changed)
    assert error in experiment.validate_artifact(changed)


def test_req_ising_7286_atomic_run_cli_and_thin_entrypoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ISING-7286: publication and the CLI use the tested validator."""

    paths = experiment.ExperimentPaths.under(tmp_path)
    artifact = experiment.run_experiment(ROOT, paths)
    assert json.loads(paths.artifact.read_text(encoding="utf-8")) == artifact
    assert experiment.main(["--validate", str(paths.artifact)]) == 0
    assert "validation_passed" in capsys.readouterr().out
    assert experiment.main(["--date", "20260912", "--root", str(ROOT)]) == 2
    with pytest.raises(ValueError, match="invalid Exp7286 artifact"):
        experiment.atomic_write(tmp_path / "invalid.json", {})
    source = (ROOT / experiment.ENTRYPOINT_PATH).read_text(encoding="utf-8")
    assert "experiment_7286_v640_board_state import main" in source
    assert source.count("main()") == 1


def test_req_ising_7286_blocked_private_root_and_defensive_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ISING-7286: missing inputs and invalid builds remain terminal failures."""

    assert experiment.main(["--root", str(tmp_path), "--output", "private/result.json"]) == 0
    saved = json.loads((tmp_path / "private/result.json").read_text(encoding="utf-8"))
    assert saved["status"] == "blocked_external_precondition"
    assert saved["board_rows"] == []
    assert experiment._task_contract(tmp_path) is None
    (tmp_path / experiment.ROADMAP_PATH).write_text("[]\n", encoding="utf-8")
    assert experiment._task_contract(tmp_path) is None
    (tmp_path / experiment.ROADMAP_PATH).write_text("tasks: []\n", encoding="utf-8")
    assert experiment._task_contract(tmp_path) is None
    assert experiment.validate_artifact({})[0].startswith("missing_fields:")
    assert experiment.main(["--validate", str(tmp_path / "missing.json")]) == 2

    paths = experiment.ExperimentPaths.under(tmp_path / "run")
    paths.validation_dir.mkdir(parents=True)
    passed_log = paths.validation_dir / "passed.log"
    passed_log.write_text("$ pytest focused\n1 passed\n[exit_code] 0\n", encoding="utf-8")
    baseline_dir = paths.validation_dir.parent / "baseline_failures"
    baseline_dir.mkdir()
    failed_log = baseline_dir / "failed.log"
    failed_log.write_text("$ pytest red\n1 failed\n[exit_code] 1\n", encoding="utf-8")
    artifact = experiment.build_artifact(ROOT, paths, candidate_paths=[tmp_path / "missing.json"])
    assert artifact["validation_receipts"][0]["log_hash"] == experiment.sha256_file(passed_log)
    assert artifact["baseline_validation_failures"][0]["log_hash"] == (
        experiment.sha256_file(failed_log)
    )

    invalid_identity = deepcopy(artifact)
    invalid_identity["schema"] = "wrong"
    invalid_identity["reproducibility_checksum"] = experiment.artifact_checksum(invalid_identity)
    assert "identity" in experiment.validate_artifact(invalid_identity)
    invalid_principles = deepcopy(artifact)
    invalid_principles["field_principles"] = {}
    invalid_principles["reproducibility_checksum"] = experiment.artifact_checksum(
        invalid_principles
    )
    assert "field_principles" in experiment.validate_artifact(invalid_principles)
    invalid_json = deepcopy(artifact)
    invalid_json["validation_receipts"] = {object()}
    assert "reproducibility_checksum" in experiment.validate_artifact(invalid_json)

    monkeypatch.setattr(experiment, "build_artifact", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(experiment, "validate_artifact", lambda *_args, **_kwargs: ["bad"])
    with pytest.raises(ValueError, match="invalid Exp7286 artifact"):
        experiment.run_experiment(ROOT, paths)
