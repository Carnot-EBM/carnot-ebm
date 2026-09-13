"""Tests for the V639 host-only board-state continuation.

Spec: REQ-ISING-7272 and SCENARIO-ISING-7272-PREFLIGHT through
SCENARIO-ISING-7272-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7272_v639_board_state as experiment


ROOT = Path(__file__).resolve().parents[2]


def _valid_operator_receipt(path: Path) -> Path:
    """Create one explicit later operator change for the parser branch."""

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


def test_req_ising_7272_authenticates_current_receipt_chain(tmp_path: Path) -> None:
    """REQ-ISING-7272: Exp7258 and every referenced transcript must authenticate."""

    checks, hashes, upstreams = experiment.collect_preconditions(
        ROOT, experiment.ExperimentPaths.under(tmp_path)
    )
    assert all(row["passed"] for row in checks)
    assert upstreams["receipt"]["experiment_id"] == 7258
    assert upstreams["kv260_row"]["terminal_criterion"] == (experiment.KV260_TERMINAL_CRITERION)
    assert upstreams["polarfire_row"]["processor_class"] == "cpu"
    assert upstreams["reference_observation"]["all_match"] is True
    assert hashes[experiment.UPSTREAM_PATH.as_posix()].startswith("sha256:")


def test_scenario_ising_7272_reference_observation_fails_closed(tmp_path: Path) -> None:
    """SCENARIO-ISING-7272-PREFLIGHT: missing evidence cannot hash-match."""

    observation = experiment.reference_observation(
        tmp_path,
        {
            "board": "KV260",
            "latest_receipt_path": "missing.json",
            "latest_receipt_hash": "sha256:expected",
            "referenced_evidence": [None, {"path": None, "sha256": None}],
        },
        None,
    )
    assert observation["all_match"] is False
    assert observation["rows"][0]["observed_sha256"] is None


def test_scenario_ising_7272_gatemate_absence_is_explicit(tmp_path: Path) -> None:
    """SCENARIO-ISING-7272-GATEMATE: missing state keeps all commands at zero."""

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
    assert receipt["observed_missing_receipt"] == experiment.MISSING_RECEIPT
    assert receipt["newly_enabled_next_action"] is None
    assert receipt["hardware_operations_issued"] == []
    raw = json.loads((tmp_path / "search.json").read_text(encoding="utf-8"))
    assert len(raw["candidate_rows"]) == 2
    assert all(row["valid"] is False for row in raw["candidate_rows"])


def test_scenario_ising_7272_gatemate_change_only_enables_future_action(
    tmp_path: Path,
) -> None:
    """SCENARIO-ISING-7272-GATEMATE: a later change only enables future work."""

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
    assert receipt["newly_enabled_next_action"] == experiment.GATEMATE_FUTURE_ACTION
    assert receipt["hardware_operations_issued"] == []


def test_scenario_ising_7272_fixture_and_history_sidecars(tmp_path: Path) -> None:
    """SCENARIO-ISING-7272-FIXTURES: controls fail closed and history stays separate."""

    fixture_path = tmp_path / "negative.json"
    negative = experiment.write_negative_fixture_receipt(fixture_path)
    assert negative["fixture_count"] == 2
    assert negative["all_failed_closed"] is True
    assert all(row["accepted"] is False for row in negative["rows"])
    assert negative["hardware_operations_issued"] == []

    history_path = tmp_path / "history.json"
    history = experiment.write_historical_model_receipt(ROOT, history_path)
    assert history["current_invocation_model_specs"] == []
    assert history["current_invocation_model_count"] == 0
    assert history["source_receipts"][0]["source_path"] == (experiment.UPSTREAM_PATH.as_posix())
    assert history["source_receipts"][0]["source_sha256"].startswith("sha256:")


def test_scenario_ising_7272_boards_preserve_exact_boundaries(tmp_path: Path) -> None:
    """SCENARIO-ISING-7272-BOARDS: fabric, CPU, and physical state stay separate."""

    paths = experiment.ExperimentPaths.under(tmp_path)
    _, _, upstreams = experiment.collect_preconditions(ROOT, paths)
    operator = experiment.search_gatemate_operator_receipts(
        ROOT, paths.operator_search, candidate_paths=[tmp_path / "missing.json"]
    )
    rows = experiment.build_board_rows(ROOT, upstreams, operator)
    by_board = {row["board"]: row for row in rows}
    assert set(by_board) == {"KV260", "GateMate", "PolarFire"}
    assert by_board["KV260"]["terminal_criterion"] == (experiment.KV260_TERMINAL_CRITERION)
    assert by_board["KV260"]["processor_class"] == "fpga_fabric"
    assert by_board["GateMate"]["disposition"] == "blocked_changed_physical_state"
    assert by_board["GateMate"]["operator_author_evidence"] is None
    assert by_board["PolarFire"]["terminal_criterion"] == (experiment.POLARFIRE_TERMINAL_CRITERION)
    assert by_board["PolarFire"]["processor_class"] == "cpu"
    assert by_board["PolarFire"]["programmable_logic_sampling_observed"] is False
    assert all(row["latest_receipt_path"] for row in rows)
    assert all(row["hardware_operations_issued"] == [] for row in rows)
    reduced = experiment.reduce_board_rows(rows)
    assert reduced["board_disposition_complete_score"] == 1
    assert reduced["hardware_command_count"] == 0


def test_scenario_ising_7272_artifact_is_complete_host_aggregation(tmp_path: Path) -> None:
    """SCENARIO-ISING-7272-ARTIFACT: three dispositions complete the receipt."""

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
    assert artifact["invocation_counts"] == {
        "model_loads_attempted": 0,
        "model_loads_completed": 0,
        "generation_calls_attempted": 0,
        "generation_calls_completed": 0,
        "usable_answers": 0,
    }
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["hardware_operations_issued"] == []
    assert artifact["gate_check_summary"]["board_blocks"][0]["verdict"] == (
        "blocked_changed_physical_state"
    )
    assert artifact["honest_verdict"].startswith("complete:")


def test_scenario_ising_7272_positive_operator_receipt_validates(tmp_path: Path) -> None:
    """SCENARIO-ISING-7272-GATEMATE: a valid change produces a future-action row."""

    candidate = _valid_operator_receipt(tmp_path / "operator.json")
    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        candidate_paths=[candidate],
    )
    gate = next(row for row in artifact["board_rows"] if row["board"] == "GateMate")
    assert gate["disposition"] == "changed_physical_state_future_action_enabled"
    assert gate["operator_author_evidence"] is True
    assert gate["operator_date_evidence"] == "20260913"
    assert gate["metric"] is True
    assert artifact["gate_check_summary"]["board_blocks"] == []
    assert experiment.validate_artifact(artifact) == []


def test_scenario_ising_7272_preflight_block_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ISING-7272-PREFLIGHT: failed intake emits no board rows."""

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
        (
            {"invocation_counts": {"model_loads_attempted": 1}},
            "invocation_counts",
        ),
        ({"inference_substrate": "cpu_exact_solver_or_simulator"}, "substrate"),
        ({"reproducibility_checksum": "sha256:bad"}, "reproducibility_checksum"),
    ),
)
def test_req_ising_7272_reducer_rejects_mutations(
    tmp_path: Path, mutation: dict[str, Any], error: str
) -> None:
    """REQ-ISING-7272: the independent validator rejects unsafe mutations."""

    artifact = experiment.build_artifact(ROOT, experiment.ExperimentPaths.under(tmp_path))
    changed = deepcopy(artifact)
    changed.update(mutation)
    if "reproducibility_checksum" not in mutation:
        changed["reproducibility_checksum"] = experiment.artifact_checksum(changed)
    assert error in experiment.validate_artifact(changed)


def test_req_ising_7272_validation_log_receipts_are_hashed(tmp_path: Path) -> None:
    """REQ-ISING-7272: command receipts retain exit status and exact log hash."""

    log_dir = tmp_path / "validation"
    log_dir.mkdir()
    passed = log_dir / "focused.log"
    passed.write_text("$ pytest focused\n1 passed\n[exit_code] 0\n", encoding="utf-8")
    malformed = log_dir / "malformed.log"
    malformed.write_text("no command marker\n", encoding="utf-8")
    rows = experiment.read_validation_receipts(log_dir)
    assert rows[0]["command"] == "pytest focused"
    assert rows[0]["exit_code"] == 0
    assert rows[0]["log_hash"] == experiment.sha256_file(passed)
    assert rows[1]["command"] == "unknown"
    assert rows[1]["exit_code"] is None


def test_req_ising_7272_atomic_run_and_cli_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ISING-7272: writer and read-only CLI use the independent validator."""

    paths = experiment.ExperimentPaths.under(tmp_path)
    artifact = experiment.run_experiment(ROOT, paths)
    assert json.loads(paths.artifact.read_text(encoding="utf-8")) == artifact
    assert experiment.main(["--validate", str(paths.artifact)]) == 0
    assert "validation_passed" in capsys.readouterr().out
    assert experiment.main(["--date", "20260912", "--root", str(ROOT)]) == 2
    with pytest.raises(ValueError, match="invalid Exp7272 artifact"):
        experiment.atomic_write(tmp_path / "invalid.json", {})


def test_req_ising_7272_cli_writes_blocked_result_under_private_root(
    tmp_path: Path,
) -> None:
    """REQ-ISING-7272: a missing private repository remains terminal blocked."""

    assert experiment.main(["--root", str(tmp_path), "--output", "private/result.json"]) == 0
    saved = json.loads((tmp_path / "private/result.json").read_text(encoding="utf-8"))
    assert saved["status"] == "blocked_external_precondition"
    assert saved["board_rows"] == []


def test_req_ising_7272_contract_and_validator_fail_closed(tmp_path: Path) -> None:
    """REQ-ISING-7272: malformed contracts and incomplete artifacts cannot pass."""

    assert experiment._task_contract(tmp_path) is None
    (tmp_path / experiment.ROADMAP_PATH).write_text("[]\n", encoding="utf-8")
    assert experiment._task_contract(tmp_path) is None
    (tmp_path / experiment.ROADMAP_PATH).write_text("tasks: []\n", encoding="utf-8")
    assert experiment._task_contract(tmp_path) is None
    assert experiment.validate_artifact({})[0].startswith("missing_fields:")


def test_req_ising_7272_defensive_validation_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ISING-7272: invalid metadata and failed builds remain explicit."""

    paths = experiment.ExperimentPaths.under(tmp_path)
    paths.validation_dir.mkdir(parents=True)
    log = paths.validation_dir / "focused.log"
    log.write_text("$ pytest focused\n1 passed\n[exit_code] 0\n", encoding="utf-8")
    baseline_dir = paths.validation_dir.parent / "baseline_failures"
    baseline_dir.mkdir()
    baseline = baseline_dir / "unrelated.log"
    baseline.write_text("$ pytest unrelated\nfailed\n[exit_code] 1\n", encoding="utf-8")
    artifact = experiment.build_artifact(ROOT, paths)
    assert any(row["command"] == "pytest focused" for row in artifact["validation_receipts"])
    assert artifact["baseline_validation_failures"][0]["exit_code"] == 1
    assert str(log) in artifact["source_artifact_hashes"]
    assert str(baseline) in artifact["source_artifact_hashes"]

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

    monkeypatch.setattr(experiment, "build_artifact", lambda _root, _paths: {})
    monkeypatch.setattr(experiment, "validate_artifact", lambda _artifact, **_kwargs: ["bad"])
    with pytest.raises(ValueError, match="invalid Exp7272 artifact"):
        experiment.run_experiment(ROOT, paths)


def test_req_ising_7272_missing_cli_validation_fails_closed(tmp_path: Path) -> None:
    """REQ-ISING-7272: a missing validation target cannot report success."""

    assert experiment.main(["--validate", str(tmp_path / "missing.json")]) == 2


def test_req_ising_7272_thin_entrypoint() -> None:
    """REQ-ISING-7272: the experiment script delegates to the tested module."""

    source = (ROOT / experiment.ENTRYPOINT_PATH).read_text(encoding="utf-8")
    assert "experiment_7272_v639_board_state import main" in source
    assert source.count("main()") == 1
