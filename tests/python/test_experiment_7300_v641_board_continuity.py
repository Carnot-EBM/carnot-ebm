"""Tests for the V641 read-only board-continuity receipt.

Spec: REQ-ISING-7300 and SCENARIO-ISING-7300-PREFLIGHT through
SCENARIO-ISING-7300-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7300_v641_board_continuity as experiment


ROOT = Path(__file__).resolve().parents[2]


def _valid_operator_receipt(path: Path) -> Path:
    """Create an operator receipt that proves a later physical change."""

    path.write_text(
        json.dumps(
            {
                "exists": True,
                "receipt_date": "20260914",
                "source": ("operator directive 2026-09-14T12:00:00Z: GateMate JTAG cable changed"),
                "operator_authored": True,
                "board": "Cologne Chip GateMate A1-EVB-2M",
                "usb_dirtyjtag": "1209:c0ca DirtyJTAG",
                "dirtyjtag": "new operator-confirmed JTAG cable path",
                "changes": [
                    {
                        "field": "dirtyjtag",
                        "description": "operator changed the JTAG cable path",
                    }
                ],
                "action": "detect",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_req_ising_7300_authenticates_v640_and_original_evidence(tmp_path: Path) -> None:
    """REQ-ISING-7300: the source receipt and original evidence authenticate."""

    checks, hashes, upstreams = experiment.collect_preconditions(
        ROOT, experiment.ExperimentPaths.under(tmp_path)
    )
    assert all(row["passed"] for row in checks)
    assert upstreams["receipt"]["experiment_id"] == 7286
    assert upstreams["reference_observation"]["all_match"] is True
    assert hashes[experiment.UPSTREAM_PATH.as_posix()].startswith("sha256:")


def test_scenario_ising_7300_boards_keep_venue_and_capability_scopes(
    tmp_path: Path,
) -> None:
    """SCENARIO-ISING-7300-BOARDS: board execution scopes stay distinct."""

    paths = experiment.ExperimentPaths.under(tmp_path)
    _, _, upstreams = experiment.collect_preconditions(ROOT, paths)
    changed_state = experiment.search_gatemate_operator_receipts(
        ROOT, paths.changed_state_search, candidate_paths=[tmp_path / "missing.json"]
    )
    rows = experiment.build_board_rows(ROOT, upstreams, changed_state)
    by_board = {row["board"]: row for row in rows}
    assert set(by_board) == {"KV260", "PolarFire", "GateMate"}
    assert by_board["KV260"]["observed_venue"] == "kv260_fpga_fabric"
    assert by_board["KV260"]["fabric_execution_completed"] is True
    assert by_board["KV260"]["host_emulation"] is False
    assert by_board["PolarFire"]["observed_venue"] == "polarfire_linux_cpu"
    assert by_board["PolarFire"]["fabric_execution_completed"] is False
    assert by_board["PolarFire"]["programmable_logic_sampling_observed"] is False
    assert by_board["GateMate"]["disposition"] == "blocked_changed_physical_state"
    assert by_board["GateMate"]["failed_value"] == experiment.MISSING_RECEIPT
    assert all(row["preserved_capability"] for row in rows)
    assert all(row["prerequisite_check"] for row in rows)
    assert all(row["hardware_operations_issued"] == [] for row in rows)


def test_scenario_ising_7300_artifact_completes_with_row_scope_block(
    tmp_path: Path,
) -> None:
    """SCENARIO-ISING-7300-ARTIFACT: three dispositions complete continuity."""

    paths = experiment.ExperimentPaths.under(tmp_path)
    artifact = experiment.build_artifact(ROOT, paths, candidate_paths=[tmp_path / "missing.json"])
    assert paths.checkpoint.is_file()
    assert paths.raw_rows.is_file()
    assert not paths.artifact.exists()
    assert experiment.validate_artifact(artifact, root=ROOT) == []
    assert artifact["status"] == "complete"
    assert artifact["board_continuity_complete_score"] == 1
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert artifact["hardware_operations_issued"] == []
    assert artifact["changed_state_receipt"]["exists"] is False
    block = artifact["gate_check_summary"]["board_blocks"][0]
    assert block == {
        "verdict": "blocked_changed_physical_state",
        "upstream": "changed_state_receipt",
        "field": "operator-authored USB/JTAG/cabling/board/power change after Exp6559",
        "expected_value": experiment.GATEMATE_OPERATOR_ACTION,
        "observed_value": experiment.MISSING_RECEIPT,
    }
    assert artifact["honest_verdict"].startswith("complete:")


def test_scenario_ising_7300_changed_state_only_names_later_experiment(
    tmp_path: Path,
) -> None:
    """SCENARIO-ISING-7300-GATEMATE: a change only enables later work."""

    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        candidate_paths=[_valid_operator_receipt(tmp_path / "operator.json")],
    )
    gate = next(row for row in artifact["board_rows"] if row["board"] == "GateMate")
    assert gate["disposition"] == "changed_physical_state_future_action_enabled"
    assert gate["exact_next_condition"] == experiment.GATEMATE_FUTURE_ACTION
    assert gate["failed_value"] is None
    assert artifact["changed_state_receipt"]["date_evidence"] == "20260914"
    assert artifact["gate_check_summary"]["board_blocks"] == []
    assert artifact["hardware_operations_issued"] == []
    assert experiment.validate_artifact(artifact, root=ROOT) == []


def test_scenario_ising_7300_preflight_failure_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ISING-7300-PREFLIGHT: bad intake emits one blocked record."""

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
        ({"board_continuity_complete_score": 0}, "board_continuity_score"),
        ({"hardware_operations_issued": ["jtag"]}, "hardware_operations"),
        ({"MODEL_SPECS": ["model"]}, "model_declaration"),
        ({"invocation_counts": {"model_loads_attempted": 1}}, "invocation_counts"),
        ({"inference_substrate": "cpu_exact_solver_or_simulator"}, "substrate"),
        ({"reproducibility_checksum": "sha256:bad"}, "reproducibility_checksum"),
    ),
)
def test_req_ising_7300_validator_rejects_unsafe_mutations(
    tmp_path: Path, mutation: dict[str, Any], error: str
) -> None:
    """REQ-ISING-7300: the validator rejects unsafe receipt mutations."""

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


def test_req_ising_7300_atomic_run_cli_and_defensive_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ISING-7300: the CLI publishes only validated terminal bytes."""

    paths = experiment.ExperimentPaths.under(tmp_path)
    artifact = experiment.run_experiment(ROOT, paths)
    assert json.loads(paths.artifact.read_text(encoding="utf-8")) == artifact
    assert experiment.main(["--validate", str(paths.artifact)]) == 0
    assert "validation_passed" in capsys.readouterr().out
    assert experiment.main(["--date", "20260913", "--root", str(ROOT)]) == 2
    with pytest.raises(ValueError, match="invalid Exp7300 artifact"):
        experiment.atomic_write(tmp_path / "invalid.json", {})
    source = (ROOT / experiment.ENTRYPOINT_PATH).read_text(encoding="utf-8")
    assert "experiment_7300_v641_board_continuity import main" in source
    assert source.count("main()") == 1

    assert (
        experiment.main(
            [
                "--root",
                str(tmp_path / "empty"),
                "--output",
                "private/result.json",
            ]
        )
        == 0
    )
    blocked = json.loads((tmp_path / "empty" / "private/result.json").read_text(encoding="utf-8"))
    assert blocked["status"] == "blocked_external_precondition"
    assert experiment._task_contract(tmp_path) is None
    assert experiment.validate_artifact({})[0].startswith("missing_fields:")
    assert experiment.main(["--validate", str(tmp_path / "missing.json")]) == 2

    monkeypatch.setattr(experiment, "build_artifact", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(experiment, "validate_artifact", lambda *_args, **_kwargs: ["bad"])
    with pytest.raises(ValueError, match="invalid Exp7300 artifact"):
        experiment.run_experiment(ROOT, paths)


def test_req_ising_7300_defensive_validation_and_receipt_logs(tmp_path: Path) -> None:
    """REQ-ISING-7300: malformed metadata fails and command timing is retained."""

    roadmap = tmp_path / experiment.ROADMAP_PATH
    roadmap.write_text("[]\n", encoding="utf-8")
    assert experiment._task_contract(tmp_path) is None
    roadmap.write_text("tasks: []\n", encoding="utf-8")
    assert experiment._task_contract(tmp_path) is None

    paths = experiment.ExperimentPaths.under(tmp_path / "receipts")
    paths.validation_dir.mkdir(parents=True)
    log = paths.validation_dir / "focused.log"
    log.write_text(
        "$ pytest focused\n12 passed\n[elapsed_s] 3.5\n[exit_code] 0\n",
        encoding="utf-8",
    )
    artifact = experiment.build_artifact(ROOT, paths, candidate_paths=[tmp_path / "missing.json"])
    assert artifact["validation_receipts"][0]["elapsed_s"] == 3.5
    assert log.as_posix() in artifact["source_artifact_hashes"]

    for mutation, expected in (
        ({"schema": "wrong"}, "identity"),
        ({"field_principles": {}}, "field_principles"),
    ):
        changed = deepcopy(artifact)
        changed.update(mutation)
        changed["reproducibility_checksum"] = experiment.artifact_checksum(changed)
        assert expected in experiment.validate_artifact(changed)

    bad_row = deepcopy(artifact)
    del bad_row["board_rows"][0]["observed_venue"]
    bad_row["reproducibility_checksum"] = experiment.artifact_checksum(bad_row)
    assert "board_row_scope" in experiment.validate_artifact(bad_row)

    bad_json = deepcopy(artifact)
    bad_json["validation_receipts"] = {object()}
    assert "reproducibility_checksum" in experiment.validate_artifact(bad_json)

    bad_counts = deepcopy(artifact)
    bad_counts["invocation_counts"] = None
    bad_counts["reproducibility_checksum"] = experiment.artifact_checksum(bad_counts)
    assert "invocation_counts" in experiment.validate_artifact(bad_counts)
