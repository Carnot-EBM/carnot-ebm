"""Tests for the read-only V637 board disposition.

Spec: REQ-ISING-7244 and SCENARIO-ISING-7244-PREFLIGHT through
SCENARIO-ISING-7244-ARTIFACT.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7244_v637_board_disposition as experiment


ROOT = Path(__file__).resolve().parents[2]


def test_req_ising_7244_preconditions_authenticate_public_inputs(tmp_path: Path) -> None:
    """REQ-ISING-7244: authenticate requirements, sources, hashes, and outputs."""

    checks, hashes, upstreams = experiment.collect_preconditions(
        ROOT, experiment.ExperimentPaths.under(tmp_path)
    )
    assert all(row["passed"] for row in checks)
    assert upstreams["continuity"]["experiment_id"] == 7231
    assert upstreams["memory"]["experiment_id"] == 7243
    assert hashes[experiment.CONTINUITY_PATH.as_posix()].startswith("sha256:")
    assert hashes[experiment.POLARFIRE_RAW_PATH.as_posix()].startswith("sha256:")
    assert hashes[experiment.KV260_PATH.as_posix()].startswith("sha256:")


def test_req_ising_7244_unwraps_only_exact_principle_records() -> None:
    """REQ-ISING-7244: arbitrary dictionaries remain evidence dictionaries."""

    wrapped = {"principle": "explain", "value": 7}
    assert experiment.unwrap_principled_value(wrapped) == 7
    assert experiment.unwrap_principled_value({**wrapped, "extra": True}) == {
        **wrapped,
        "extra": True,
    }


def test_scenario_ising_7244_gatemate_records_explicit_absence(tmp_path: Path) -> None:
    """SCENARIO-ISING-7244-GATEMATE: absence authorizes no board command."""

    receipt = experiment.search_gatemate_operator_receipts(
        ROOT,
        tmp_path / "search.json",
        candidate_paths=[ROOT / "results/experiment_7146_v627_gatemate_changed_state.json"],
    )
    assert receipt["exists"] is False
    assert receipt["newer_than_exp6559"] is False
    assert receipt["receipt_timestamp"] is None
    assert receipt["changed_conditions"] == {"cable": None, "port": None, "power": None}
    assert receipt["evidence_hash"] is None
    assert receipt["authorized_next_task"] is None
    raw = json.loads((tmp_path / "search.json").read_text(encoding="utf-8"))
    assert raw["explicit_absence"] is True
    assert raw["hardware_operations_issued"] == []


def test_scenario_ising_7244_gatemate_new_receipt_only_names_later_task(
    tmp_path: Path,
) -> None:
    """SCENARIO-ISING-7244-GATEMATE: a new state only scopes later work."""

    candidate = tmp_path / "operator_gatemate_receipt.json"
    candidate.write_text(
        json.dumps(
            {
                "run_date": "20260912",
                "physical_state_receipt": {
                    "exists": True,
                    "authored_by": "operator",
                    "receipt_timestamp": "2026-09-12T15:00:00+00:00",
                    "changed_conditions": {
                        "cable": "USB-C cable reseated",
                        "port": "moved to host port 2",
                        "power": "board power cycled",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    receipt = experiment.search_gatemate_operator_receipts(
        ROOT, tmp_path / "positive-search.json", candidate_paths=[candidate]
    )
    assert receipt["exists"] is True
    assert receipt["newer_than_exp6559"] is True
    assert receipt["changed_conditions"]["port"] == "moved to host port 2"
    assert receipt["evidence_hash"] == experiment.sha256_file(candidate)
    assert receipt["authorized_next_task"] == "one bounded GateMate detect in a later task"
    assert receipt["hardware_operations_issued"] == []


def test_scenario_ising_7244_boards_preserve_independent_criteria(tmp_path: Path) -> None:
    """SCENARIO-ISING-7244-BOARDS: authenticate each board independently."""

    paths = experiment.ExperimentPaths.under(tmp_path)
    _, _, upstreams = experiment.collect_preconditions(ROOT, paths)
    operator = experiment.search_gatemate_operator_receipts(
        ROOT,
        paths.gatemate_search,
        candidate_paths=[ROOT / "results/experiment_7146_v627_gatemate_changed_state.json"],
    )
    rows = experiment.build_board_rows(ROOT, upstreams, operator)
    by_board = {row["board"]: row for row in rows}
    assert set(by_board) == {"KV260", "GateMate", "PolarFire"}
    assert by_board["KV260"]["terminal_criterion"] == experiment.KV260_TERMINAL_CRITERION
    assert by_board["KV260"]["disposition"] == "graduated_preserved"
    assert by_board["KV260"]["exact_next_prerequisite"] == "none; future access uses ssh kria only"
    assert by_board["GateMate"]["disposition"] == "blocked_inherited_no_new_physical_state"
    assert by_board["GateMate"]["hardware_command_count"] == 0
    assert by_board["PolarFire"]["terminal_criterion_met"] is True
    assert by_board["PolarFire"]["processor_class"] == "cpu"
    assert by_board["PolarFire"]["programmable_logic_sampling_observed"] is False
    assert by_board["PolarFire"]["smoke_repeated"] is False
    assert all(row["latest_receipt_authenticated"] for row in rows)


def test_scenario_ising_7244_placement_retains_memory_and_unknowns() -> None:
    """SCENARIO-ISING-7244-PLACEMENT: measured sizes do not prove device fit."""

    memory = json.loads((ROOT / experiment.MEMORY_PATH).read_text(encoding="utf-8"))
    mapping = experiment.operation_map(memory)
    by_target = {row["target"]: row for row in mapping}
    assert by_target["cpu_rust"]["archive_mask_bytes_at_capacity_four"] == 128
    assert by_target["cpu_rust"]["packed_active_masks_bytes"] == 32
    assert by_target["cpu_rust"]["controller_state_bytes"] == 28669
    assert "validation_counter_updates" in by_target["cpu_rust"]["operations"]
    assert by_target["fpga_bram"]["prospective"] is True
    assert by_target["tsu"]["prospective"] is True
    assert all(row["topology"] == "unknown" for row in mapping)
    assert all(row["bandwidth"] == "unknown" for row in mapping)
    assert all(row["power"] == "unknown" for row in mapping)
    assert all(row["latency"] == "unknown" for row in mapping)
    assert all(row["degree_16_connectivity_establishes_fit"] is False for row in mapping)


def test_scenario_ising_7244_placement_allows_missing_memory() -> None:
    """SCENARIO-ISING-7244-PLACEMENT: absent optional memory does not block rows."""

    mapping = experiment.operation_map({})
    assert all(row["memory_footprint_available"] is False for row in mapping)
    assert all(row["controller_state_bytes"] is None for row in mapping)


def test_scenario_ising_7244_artifact_is_complete_and_read_only(tmp_path: Path) -> None:
    """SCENARIO-ISING-7244-ARTIFACT: three rows complete this read-only review."""

    paths = experiment.ExperimentPaths.under(tmp_path)
    artifact = experiment.build_artifact(ROOT, paths)
    assert paths.checkpoint.is_file()
    assert not paths.artifact.exists()
    assert paths.gatemate_search.is_file()
    assert experiment.validate_artifact(artifact) == []
    assert artifact["status"] == "complete"
    assert artifact["board_disposition_complete_score"] == 1
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["model_invocation_count"] == 0
    assert artifact["hardware_operations_issued"] == []
    assert artifact["memory_footprint"]["available"] is True
    assert artifact["acceptance_gate_results"]["gatemate_disposition_recorded"]["pass"] is True
    assert artifact["honest_verdict"].startswith("complete:")


def test_scenario_ising_7244_preflight_blocks_without_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-ISING-7244-PREFLIGHT: a failed contract stops aggregation."""

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
        ({"board_disposition_complete_score": 0}, "completion_score"),
        ({"hardware_operations_issued": ["jtag"]}, "hardware_operations"),
        ({"MODEL_SPECS": ["model"]}, "model_declaration"),
        ({"inference_substrate": "cpu_exact_solver_or_simulator"}, "substrate"),
        ({"reproducibility_checksum": "sha256:bad"}, "reproducibility_checksum"),
    ),
)
def test_req_ising_7244_reducer_rejects_claim_mutations(
    tmp_path: Path, mutation: dict[str, Any], error: str
) -> None:
    """REQ-ISING-7244: the independent reducer rejects unsafe mutations."""

    artifact = experiment.build_artifact(ROOT, experiment.ExperimentPaths.under(tmp_path))
    changed = deepcopy(artifact)
    changed.update(mutation)
    if "reproducibility_checksum" not in mutation:
        changed["reproducibility_checksum"] = experiment.artifact_checksum(changed)
    assert error in experiment.validate_artifact(changed)


def test_req_ising_7244_atomic_write_and_cli_validation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-ISING-7244: producer and read-only reducer support private replay."""

    paths = experiment.ExperimentPaths.under(tmp_path)
    artifact = experiment.run_experiment(ROOT, paths)
    assert json.loads(paths.artifact.read_text(encoding="utf-8")) == artifact
    assert experiment.main(["--validate", str(paths.artifact)]) == 0
    assert "validation_passed" in capsys.readouterr().out
    assert experiment.main(["--date", "20260911", "--root", str(ROOT)]) == 2


def test_req_ising_7244_thin_entrypoint() -> None:
    """REQ-ISING-7244: the experiment script delegates to the tested module."""

    source = (ROOT / experiment.ENTRYPOINT_PATH).read_text(encoding="utf-8")
    assert "experiment_7244_v637_board_disposition import main" in source
    assert source.count("main()") == 1


def test_req_ising_7244_defensive_input_and_path_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ISING-7244: malformed inputs and failed writes remain explicit."""

    assert experiment.ExperimentPaths.defaults(tmp_path) == experiment.ExperimentPaths.under(
        tmp_path
    )
    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    assert experiment._read_json(missing) == {}
    assert experiment._read_json(malformed) == {}
    assert experiment._read_json(scalar) == {}

    monkeypatch.setattr(
        experiment.tempfile, "mkstemp", lambda **_kwargs: (_ for _ in ()).throw(OSError("no"))
    )
    assert experiment._writable_destination(tmp_path / "blocked" / "out.json") is False


def test_req_ising_7244_atomic_temp_is_removed_after_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ISING-7244: a failed atomic replace leaves no provisional terminal file."""

    monkeypatch.setattr(
        experiment.os,
        "replace",
        lambda _source, _destination: (_ for _ in ()).throw(OSError("replace failed")),
    )
    with pytest.raises(OSError, match="replace failed"):
        experiment._atomic_json(tmp_path / "artifact.json", {"status": "complete"})
    assert list(tmp_path.glob(".artifact.json.*.tmp")) == []


def test_req_ising_7244_task_contract_fail_closed_shapes(tmp_path: Path) -> None:
    """REQ-ISING-7244: missing, malformed, and unrelated roadmaps do not match."""

    assert experiment._task_contract(tmp_path) is None
    (tmp_path / experiment.ROADMAP_PATH).write_text("[]\n", encoding="utf-8")
    assert experiment._task_contract(tmp_path) is None
    (tmp_path / experiment.ROADMAP_PATH).write_text("tasks: []\n", encoding="utf-8")
    assert experiment._task_contract(tmp_path) is None


def test_req_ising_7244_manifest_parse_failure_is_a_failed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ISING-7244: an unreadable exclusion manifest cannot authenticate inputs."""

    original = experiment.yaml.safe_load
    calls = 0

    def fail_second_load(value: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise experiment.yaml.YAMLError("bad manifest")
        return original(value)

    monkeypatch.setattr(experiment.yaml, "safe_load", fail_second_load)
    checks, _, _ = experiment.collect_preconditions(
        ROOT, experiment.ExperimentPaths.under(tmp_path)
    )
    assert any(row["check"] == "continuity_not_quarantined" for row in checks)
    assert any(row["passed"] is False for row in checks)


def test_req_ising_7244_reducer_handles_schema_and_encoding_failures(
    tmp_path: Path,
) -> None:
    """REQ-ISING-7244: missing fields, bad timestamps, and bad JSON values fail."""

    assert experiment.validate_artifact({})[0].startswith("missing_fields:")
    artifact = experiment.build_artifact(ROOT, experiment.ExperimentPaths.under(tmp_path))
    invalid_time = deepcopy(artifact)
    invalid_time["started_at_utc"] = "not-a-time"
    invalid_time["reproducibility_checksum"] = experiment.artifact_checksum(invalid_time)
    assert "started_at_utc" in experiment.validate_artifact(invalid_time)
    invalid_json = deepcopy(artifact)
    invalid_json["validation_receipts"] = {object()}
    assert "reproducibility_checksum" in experiment.validate_artifact(invalid_json)


def test_scenario_ising_7244_gatemate_positive_receipt_validates(
    tmp_path: Path,
) -> None:
    """SCENARIO-ISING-7244-GATEMATE: a complete receipt scopes later detection."""

    paths = experiment.ExperimentPaths.under(tmp_path)
    artifact = experiment.build_artifact(ROOT, paths)
    candidate = tmp_path / "operator_receipt.json"
    candidate.write_text(
        json.dumps(
            {
                "physical_state_receipt": {
                    "exists": True,
                    "authored_by": "operator",
                    "receipt_timestamp": "2026-09-12T16:00:00+00:00",
                    "changed_conditions": {
                        "cable": "reseated",
                        "port": None,
                        "power": None,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    operator = experiment.search_gatemate_operator_receipts(
        ROOT, tmp_path / "positive.json", candidate_paths=[candidate]
    )
    _, _, upstreams = experiment.collect_preconditions(ROOT, paths)
    rows = experiment.build_board_rows(ROOT, upstreams, operator)
    artifact["rows"] = rows
    artifact["board_rows"] = rows
    artifact["operator_state_receipt"] = operator
    artifact["reproducibility_checksum"] = experiment.artifact_checksum(artifact)
    assert experiment.validate_artifact(artifact) == []


def test_req_ising_7244_error_routes_do_not_publish_invalid_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ISING-7244: invalid reducer results fail writes and CLI validation."""

    with pytest.raises(ValueError, match="invalid Exp7244 artifact"):
        experiment.atomic_write(tmp_path / "invalid.json", {})
    assert experiment.main(["--validate", str(tmp_path / "missing.json")]) == 2

    monkeypatch.setattr(experiment, "build_artifact", lambda _root, _paths: {})
    monkeypatch.setattr(experiment, "validate_artifact", lambda _artifact, **_kwargs: ["bad"])
    with pytest.raises(ValueError, match="invalid Exp7244 artifact"):
        experiment.run_experiment(ROOT, experiment.ExperimentPaths.under(tmp_path))


def test_req_ising_7244_cli_resolves_private_relative_outputs(tmp_path: Path) -> None:
    """REQ-ISING-7244: relative outputs stay under the selected private root."""

    assert (
        experiment.main(
            [
                "--root",
                str(tmp_path),
                "--output",
                "private/result.json",
                "--raw-search",
                "private/search.json",
                "--checkpoint",
                "private/checkpoint.json",
            ]
        )
        == 0
    )
    saved = json.loads((tmp_path / "private/result.json").read_text(encoding="utf-8"))
    assert saved["status"] == "blocked_external_precondition"
