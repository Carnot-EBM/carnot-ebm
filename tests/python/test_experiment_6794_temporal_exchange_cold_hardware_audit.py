"""Tests for the independent temporal-exchange cold hardware audit.

Spec: REQ-SAMPLE-100, SCENARIO-SAMPLE-100, SCENARIO-SAMPLE-101,
SCENARIO-SAMPLE-102.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import pytest

from carnot import experiment_6794_temporal_exchange_cold_hardware_audit as mod


@pytest.fixture(scope="module")
def source() -> dict:
    """Load the frozen source once because its rows are large."""

    return mod.load_json(mod.REPO_ROOT / mod.SOURCE_PATH)


@pytest.fixture(scope="module")
def completed_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run one full cold audit for all terminal artifact assertions."""

    output = tmp_path_factory.mktemp("exp6794") / "audit.json"
    return mod.run(output_path=output, run_date="20260830")


def test_req_sample_100_preconditions_bind_source_rows_code_and_board_receipts(
    source: dict,
) -> None:
    """REQ-SAMPLE-100: The cold launch gate checks every required receipt."""

    checks = mod.check_preconditions(source)
    assert {row["check"] for row in checks} == {
        "source_comparison_completed",
        "source_code_identity",
        "raw_or_sufficient_statistic_hashes",
        "exact_target_hashes",
        "update_receipts",
        "existing_board_receipts",
    }
    assert all(row["passed"] for row in checks)


def test_scenario_sample_100_evaluator_has_no_source_reducer_import() -> None:
    """SCENARIO-SAMPLE-100: The evaluator does not import Exp6793 code."""

    tree = ast.parse((mod.REPO_ROOT / mod.MODULE_PATH).read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("experiment_6793" in name for name in imported)
    assert not any("sampling.temporal_exchange" in name for name in imported)


def test_scenario_sample_100_exact_target_and_replay_match_one_source_row(
    source: dict,
) -> None:
    """SCENARIO-SAMPLE-100: Independent enumeration and replay match receipts."""

    row = source["rows"][0]
    graph = next(item for item in source["graph_families"] if item["graph_id"] == row["graph_id"])
    target = mod.independent_exact_target(graph, row["temperature"])
    replay = mod.replay_row(row, graph, target)
    cold = mod.recompute_source_row(row, graph, target, replay)
    assert target["target_sha256"] == row["exact_target_sha256"]
    assert replay["trajectory_sha256"] == row["trajectory_sha256"]
    assert replay["accepted_state_changes"] > 0
    assert replay["attempted_conditional_updates"] == row["update_count"]
    assert cold["target_total_variation"] == pytest.approx(row["target_total_variation"])
    assert cold["autocorrelation"]["energy"]["effective_samples"] == pytest.approx(
        row["effective_samples"]["energy"]
    )
    assert cold["optimum_hitting_updates"] == row["optimum_hitting_updates"]


def test_req_sample_100_work_accounting_exposes_alternative_denominators(
    source: dict,
) -> None:
    """REQ-SAMPLE-100: Recounted work does not hide temporal overhead."""

    graph = source["graph_families"][0]
    matched = [
        row
        for row in source["rows"]
        if row["graph_id"] == graph["graph_id"]
        and row["temperature"] == 0.75
        and row["seed"] == 679300
    ]
    target = mod.independent_exact_target(graph, 0.75)
    replays = [mod.replay_row(row, graph, target) for row in matched]
    accounting = {row["arm"]: replay for row, replay in zip(matched, replays, strict=True)}
    ordinary = accounting["ordinary_gibbs"]
    temporal = accounting["temporal_exchange"]
    assert ordinary["attempted_conditional_updates"] == temporal["attempted_conditional_updates"]
    assert ordinary["temporal_coupling_operations"] == 0
    assert temporal["temporal_coupling_operations"] == temporal["attempted_conditional_updates"]
    assert temporal["logical_stored_state_reads"] > ordinary["logical_stored_state_reads"]
    assert temporal["random_uniform_draws"] == temporal["attempted_conditional_updates"]


def test_scenario_sample_101_sensitivity_zero_coupling_and_stationarity(
    source: dict,
) -> None:
    """SCENARIO-SAMPLE-101: The bounded panel covers every declared axis."""

    panel = mod.run_sensitivity_panel(source)
    assert {row["sensitivity_axis"] for row in panel["rows"]} == {
        "burn_in_sweeps",
        "thinning_sweeps",
        "initial_previous_state",
        "coupling",
        "temperature",
        "run_length",
        "stationarity_window",
    }
    assert panel["zero_coupling_equivalence"]["bit_identical"] is True
    assert panel["zero_coupling_equivalence"]["trajectory_hash_equal"] is True
    assert len(panel["stationarity_checks"]) == 2
    assert all(len(item["windows"]) == 4 for item in panel["stationarity_checks"])


def test_scenario_sample_102_static_mapping_uses_only_typed_receipts(source: dict) -> None:
    """SCENARIO-SAMPLE-102: Board mapping is static and evidence classes are closed."""

    mapping = mod.derive_hardware_mapping(source)
    assert mapping["physical_hardware_invoked"] is False
    assert mapping["estimated_state_cost"]["extra_previous_state_bits_formula"] == "N"
    assert mapping["estimated_arithmetic_cost"]["extra_adds_per_update"] == 1
    assert mapping["estimated_memory_traffic"]["extra_previous_state_reads_per_update"] == 1
    assert mapping["coefficient_precision_range"]["kv260_format"] == "signed Q8.8, 16-bit"
    assert mapping["coefficient_precision_range"]["gatemate_format"] == "signed 8-bit"
    assert set(mapping["evidence_class_by_hardware_field"].values()) <= mod.EVIDENCE_CLASSES
    assert mapping["board_envelope_comparison"]["kv260"]["existing_clb_lut_capacity"] == 117120
    assert mapping["board_envelope_comparison"]["gatemate"]["existing_cpe_ff_capacity"] == 40960


def test_req_sample_100_complete_artifact_has_required_schema_and_terminal_null(
    completed_artifact: dict,
) -> None:
    """REQ-SAMPLE-100: A complete cold audit remains terminal for a null source."""

    artifact = completed_artifact
    assert mod.REQUIRED_FIELDS <= set(artifact)
    assert set(artifact["field_principles"]) == set(artifact)
    assert artifact["temporal_exchange_audit_completed"] is True
    assert artifact["physical_hardware_invoked"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["source_verdict_supported"] is True
    assert artifact["gate_check_summary"] == []
    assert (
        len([row for row in artifact["rows"] if row["row_kind"] == "source_recomputation"]) == 360
    )
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)
    assert mod.validate_artifact(artifact) == []


def test_scenario_sample_102_failed_gate_writes_complete_blocked_schema(
    tmp_path: Path,
    source: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-SAMPLE-102: A missing receipt stops before any replay."""

    changed = deepcopy(source)
    changed["temporal_exchange_comparison_completed"] = False
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(changed), encoding="utf-8")
    monkeypatch.setattr(mod, "replay_all_source_rows", lambda _source: pytest.fail("replayed"))
    output = tmp_path / "blocked.json"
    artifact = mod.run(source_path=source_path, output_path=output, run_date="20260830")
    assert artifact["status"] == "complete_blocked_temporal_exchange_audit"
    assert artifact["temporal_exchange_audit_completed"] is False
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("complete_blocked_temporal_exchange_audit")
    assert artifact["gate_check_summary"]
    assert artifact["rows"] == []
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) == []


def test_req_sample_100_validation_rejects_tampering(completed_artifact: dict) -> None:
    """REQ-SAMPLE-100: Validation rejects changed evidence and claim boundaries."""

    changed = deepcopy(completed_artifact)
    changed["rows"][0]["target_total_variation"] += 0.1
    changed["physical_hardware_invoked"] = True
    changed["verdict_class"] = "invalid"
    changed["honest_verdict"] = "missing prefix"
    errors = mod.validate_artifact(changed)
    assert "row_hash_mismatch" in errors
    assert "physical_hardware_boundary_mismatch" in errors
    assert "verdict_class_mismatch" in errors
    assert "honest_verdict_prefix_mismatch" in errors
    assert "reproducibility_checksum_mismatch" in errors


def test_req_sample_100_cli_wrapper_and_main(
    tmp_path: Path,
    completed_artifact: dict,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-SAMPLE-100: The required script delegates and the CLI reports completion."""

    called: list[list[str] | None] = []

    def fake_main(argv: list[str] | None = None) -> int:
        called.append(argv)
        return 0

    real_main = mod.main
    monkeypatch.setattr(mod, "main", fake_main)
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(mod.REPO_ROOT / mod.SCRIPT_PATH), run_name="__main__")
    assert exc.value.code == 0
    assert called == [None]

    monkeypatch.setattr(mod, "run", lambda **_kwargs: completed_artifact)
    monkeypatch.setattr(mod, "main", real_main)
    assert mod.main(["--date", "20260830", "--output", str(tmp_path / "cli.json")]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["temporal_exchange_audit_completed"] is True
    assert printed["physical_hardware_invoked"] is False
