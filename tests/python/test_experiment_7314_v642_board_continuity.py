"""Tests for the V642 read-only board-continuity receipt.

Spec: REQ-ISING-7314 and SCENARIO-ISING-7314-PREFLIGHT through
SCENARIO-ISING-7314-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7314_v642_board_continuity as experiment


ROOT = Path(__file__).resolve().parents[2]


def _valid_operator_receipt(path: Path) -> Path:
    """Create a private receipt that proves the parser's eligibility boundary."""

    path.write_text(
        json.dumps(
            {
                "exists": True,
                "receipt_date": "20260915",
                "source": (
                    "operator directive 2026-09-15T12:00:00Z: GateMate DirtyJTAG cable changed"
                ),
                "operator_authored": True,
                "board": "Cologne Chip GateMate A1-EVB-2M",
                "usb_dirtyjtag": "1209:c0ca DirtyJTAG",
                "dirtyjtag": "new operator-confirmed cable path",
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


def test_req_ising_7314_authenticates_exp7300_and_original_evidence(
    tmp_path: Path,
) -> None:
    """REQ-ISING-7314: authenticate the producer and original terminal bytes."""

    checks, hashes, upstreams = experiment.collect_preconditions(
        ROOT, experiment.ExperimentPaths.under(tmp_path)
    )
    assert all(row["passed"] for row in checks)
    assert upstreams["receipt"]["experiment_id"] == 7300
    assert upstreams["reference_observation"]["all_match"] is True
    assert hashes[experiment.UPSTREAM_PATH.as_posix()].startswith("sha256:")
    assert hashes["results/experiment_3709_kv260_drive_to_terminal_latency_transcript.json"]
    assert hashes["results/raw/experiment_7231/polarfire_dispatch.json"]


def test_scenario_ising_7314_artifact_preserves_board_scopes_and_gate(
    tmp_path: Path,
) -> None:
    """SCENARIO-ISING-7314-BOARDS: retain three distinct board authorities."""

    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        candidate_paths=[tmp_path / "missing.json"],
    )
    by_board = {row["board"]: row for row in artifact["board_rows"]}
    assert set(by_board) == {"KV260", "GateMate", "PolarFire"}
    assert by_board["KV260"]["processor_class"] == "fpga_fabric"
    assert by_board["KV260"]["fabric_execution_completed"] is True
    assert by_board["KV260"]["exact_next_condition"] == (
        "none; any future access uses ssh kria only"
    )
    assert by_board["PolarFire"]["processor_class"] == "cpu"
    assert by_board["PolarFire"]["programmable_logic_sampling_observed"] is False
    assert by_board["PolarFire"]["host_emulation"] is False
    assert by_board["GateMate"]["disposition"] == "blocked_changed_physical_state"
    assert by_board["GateMate"]["abstention"] is True
    assert artifact["board_continuity_complete_score"] == 1
    assert artifact["hardware_operations_issued"] == []
    assert experiment.validate_artifact(artifact, root=ROOT) == []

    block = artifact["gate_check_summary"]["board_blocks"][0]
    assert block["check"] == "gatemate_changed_physical_state_receipt"
    assert block["upstream"] == "physical_state_receipt"
    assert block["field"] == "receipt_date/operator_authored/provenance/changed_field"
    assert block["expected_value"] == experiment.PHYSICAL_RECEIPT_CONTRACT
    assert block["observed_value"]["accepted_receipt_count"] == 0
    assert block["observed_value"]["absence"] == experiment.MISSING_RECEIPT


def test_scenario_ising_7314_gatemate_receipt_only_enables_future_work(
    tmp_path: Path,
) -> None:
    """SCENARIO-ISING-7314-GATEMATE: a receipt cannot authorize this task."""

    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        candidate_paths=[_valid_operator_receipt(tmp_path / "operator.json")],
    )
    gate = next(row for row in artifact["board_rows"] if row["board"] == "GateMate")
    assert gate["disposition"] == "changed_physical_state_future_action_enabled"
    assert gate["exact_next_condition"] == experiment.GATEMATE_FUTURE_ACTION
    assert artifact["physical_state_receipt"]["date_evidence"] == "20260915"
    assert artifact["gate_check_summary"]["board_blocks"] == []
    assert artifact["hardware_operations_issued"] == []
    assert artifact["external_actions"] == []


def test_scenario_ising_7314_deployment_context_is_not_a_result(tmp_path: Path) -> None:
    """SCENARIO-ISING-7314-DEPLOYMENT: retain relevance and access limits."""

    artifact = experiment.build_artifact(
        ROOT,
        experiment.ExperimentPaths.under(tmp_path),
        candidate_paths=[tmp_path / "missing.json"],
    )
    relevance = artifact["deployment_relevance"]
    assert relevance["extropic"]["local_device_access"] is False
    assert relevance["extropic"]["procurement_or_vendor_contact"] is False
    assert relevance["kan"]["deployment_status"] == "deferred"
    assert relevance["kan"]["local_performance_claim"] is False
    assert relevance["availability_is_scientific_result"] is False
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["invocation_counts"].values()) == {0}
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"


def test_scenario_ising_7314_preflight_failure_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ISING-7314-PREFLIGHT: failed intake emits no board rows."""

    monkeypatch.setattr(experiment, "_task_contract", lambda _root: None)
    artifact = experiment.build_artifact(ROOT, experiment.ExperimentPaths.under(tmp_path))
    assert artifact["status"] == "blocked_external_precondition"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["board_rows"] == []
    assert artifact["hardware_operations_issued"] == []
    assert artifact["gate_check_summary"]["failed_check"] == "roadmap_task_contract"
    assert experiment.validate_artifact(artifact) == []


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        ({"board_continuity_complete_score": 0}, "board_continuity_score"),
        ({"hardware_operations_issued": ["ssh"]}, "hardware_operations"),
        ({"MODEL_SPECS": ["model"]}, "model_declaration"),
        ({"invocation_counts": {"model_loads_attempted": 1}}, "invocation_counts"),
        ({"inference_substrate": "hardware_smoke"}, "substrate"),
        ({"deployment_relevance": {}}, "deployment_relevance"),
        ({"reproducibility_checksum": "sha256:bad"}, "reproducibility_checksum"),
    ),
)
def test_req_ising_7314_validator_rejects_unsafe_mutations(
    tmp_path: Path, mutation: dict[str, Any], error: str
) -> None:
    """REQ-ISING-7314: reject mutations that broaden or fabricate the receipt."""

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


def test_scenario_ising_7314_artifact_is_atomic_and_cli_is_thin(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ISING-7314-ARTIFACT: publish only validated terminal bytes."""

    paths = experiment.ExperimentPaths.under(tmp_path)
    artifact = experiment.run_experiment(ROOT, paths)
    assert json.loads(paths.artifact.read_text(encoding="utf-8")) == artifact
    assert experiment.main(["--validate", str(paths.artifact)]) == 0
    assert "validation_passed" in capsys.readouterr().out
    assert experiment.main(["--date", "20260914", "--root", str(ROOT)]) == 2
    assert experiment.main(["--validate", str(tmp_path / "missing.json")]) == 2
    empty_root = tmp_path / "empty"
    assert experiment.main(["--root", str(empty_root)]) == 0
    blocked = json.loads((empty_root / experiment.RESULT_PATH).read_text(encoding="utf-8"))
    assert blocked["status"] == "blocked_external_precondition"
    with pytest.raises(ValueError, match="invalid Exp7314 artifact"):
        experiment.atomic_write(tmp_path / "invalid.json", {})

    source = (ROOT / experiment.ENTRYPOINT_PATH).read_text(encoding="utf-8")
    assert "experiment_7314_v642_board_continuity import main" in source
    assert source.count("main()") == 1

    monkeypatch.setattr(experiment, "build_artifact", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(experiment, "validate_artifact", lambda *_args, **_kwargs: ["bad"])
    with pytest.raises(ValueError, match="invalid Exp7314 artifact"):
        experiment.run_experiment(ROOT, paths)


def test_req_ising_7314_defensive_inputs_and_validation_logs(tmp_path: Path) -> None:
    """REQ-ISING-7314: malformed inputs fail closed and logs retain timing."""

    roadmap = tmp_path / experiment.ROADMAP_PATH
    roadmap.write_text("[]\n", encoding="utf-8")
    assert experiment._task_contract(tmp_path) is None
    roadmap.write_text("tasks: []\n", encoding="utf-8")
    assert experiment._task_contract(tmp_path) is None
    assert (
        experiment._historical_health_observation(
            tmp_path, {"baseline_validation_failures": [None]}
        )["all_match"]
        is False
    )

    paths = experiment.ExperimentPaths.under(tmp_path / "receipts")
    paths.validation_dir.mkdir(parents=True)
    log = paths.validation_dir / "focused.log"
    log.write_text(
        "$ pytest focused\n14 passed\n[elapsed_s] 3.5\n[exit_code] 0\n",
        encoding="utf-8",
    )
    artifact = experiment.build_artifact(ROOT, paths, candidate_paths=[tmp_path / "missing.json"])
    assert artifact["validation_receipts"][0]["elapsed_s"] == 3.5
    assert log.as_posix() in artifact["source_artifact_hashes"]

    for mutation, expected in (
        ({"schema": "wrong"}, "identity"),
        ({"field_principles": {}}, "field_principles"),
        ({"physical_state_receipt": {}}, "physical_state_receipt"),
    ):
        changed = deepcopy(artifact)
        changed.update(mutation)
        changed["reproducibility_checksum"] = experiment.artifact_checksum(changed)
        assert expected in experiment.validate_artifact(changed)

    bad_row = deepcopy(artifact)
    bad_row["board_rows"][0]["processor_class"] = "cpu"
    bad_row["reproducibility_checksum"] = experiment.artifact_checksum(bad_row)
    assert "board_row_scope" in experiment.validate_artifact(bad_row)

    bad_json = deepcopy(artifact)
    bad_json["validation_receipts"] = {object()}
    assert "reproducibility_checksum" in experiment.validate_artifact(bad_json)

    bad_counts = deepcopy(artifact)
    bad_counts["invocation_counts"] = None
    bad_counts["reproducibility_checksum"] = experiment.artifact_checksum(bad_counts)
    assert "invocation_counts" in experiment.validate_artifact(bad_counts)
