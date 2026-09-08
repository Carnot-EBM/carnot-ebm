"""Tests for REQ-REPORT-7147 and SCENARIO-REPORT-7147-* contracts."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts/experiments/experiment_7147_v627_capstone.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "experiment_7147_v627_capstone", MODULE_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
exp = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(exp)


def _base_payload(number: int, **updates: Any) -> dict[str, Any]:
    """Build a small upstream fixture with an explicit evidence class."""

    completion_field = exp.UPSTREAM_SPECS[number]["completion_field"]
    payload: dict[str, Any] = {
        "run_date": "20260908",
        "inference_substrate": "aggregation_from_upstream_artifacts: fixture",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "verifier_is_oracle": False,
        "verdict_class": "positive",
        "honest_verdict": "complete_positive_fixture",
        "gate_check_summary": {
            "passed": True,
            "failed_check": None,
            "expected_value": True,
            "observed_value": True,
        },
        "rows": [{"row_id": f"row-{number}", "passed": True}],
        completion_field: 1,
    }
    payload.update(updates)
    return payload


def _write_upstream(root: Path, number: int, payload: dict[str, Any]) -> Path:
    """Write a fixture only below pytest's temporary directory."""

    relative = Path(exp.UPSTREAM_SPECS[number]["path"])
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_complete_fixture(root: Path) -> None:
    """Write all 11 inputs without importing any upstream implementation."""

    for number in exp.UPSTREAM_ORDER:
        _write_upstream(root, number, _base_payload(number))


def test_req_report_7147_spec_defines_fields_and_scenarios() -> None:
    """REQ-REPORT-7147 owns the exact matrix fields and replay scenarios."""

    text = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-REPORT-7147") :]
    assert all(field in section for field in exp.REQUIRED_ARTIFACT_FIELDS)
    for name in ("INIT", "INVENTORY", "GATES", "METRICS", "VERDICTS", "DISPOSITIONS", "ARTIFACT"):
        assert f"SCENARIO-REPORT-7147-{name}" in section


def test_scenario_report_7147_init_has_complete_safe_schema(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7147-INIT initializes every field before loading."""

    artifact = exp.initialize_artifact("20260908")
    output = tmp_path / "initial.json"
    exp.write_atomic(output, artifact)
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert exp.REQUIRED_ARTIFACT_FIELDS <= loaded.keys()
    assert all(field in loaded["field_principles"] for field in exp.REQUIRED_ARTIFACT_FIELDS)
    assert loaded["verdict_class"] == "blocked"
    assert loaded["inference_substrate_class"] == "blocked_no_run"
    assert loaded["gate_check_summary"] == {
        "passed": False,
        "failed_check": "capstone_initialized",
        "expected_value": "upstream_checks_complete",
        "observed_value": "not_started",
    }


def test_scenario_report_7147_inventory_hashes_exact_paths(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7147-INVENTORY records ordered bytes and hashes."""

    _write_complete_fixture(tmp_path)
    records = exp.load_upstreams(tmp_path)
    inventories = exp.build_inventory_rows(records, exp.recompute_gates(records))

    assert [row["experiment"] for row in inventories] == list(exp.UPSTREAM_ORDER)
    assert len(inventories) == 11
    assert all(row["present"] and row["size_bytes"] > 0 for row in inventories)
    assert all(row["sha256"].startswith("sha256:") for row in inventories)
    assert all(row["validation_status"] == "valid" for row in inventories)


def test_scenario_report_7147_missing_stays_blocked_without_invention(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7147-INIT preserves one row for a missing input."""

    _write_complete_fixture(tmp_path)
    missing_path = tmp_path / exp.UPSTREAM_SPECS[7140]["path"]
    missing_path.unlink()
    records = exp.load_upstreams(tmp_path)
    gates = exp.recompute_gates(records)
    rows = exp.build_inventory_rows(records, gates)
    missing = next(row for row in rows if row["experiment"] == 7140)

    assert missing["present"] is False
    assert missing["size_bytes"] is None
    assert missing["sha256"] is None
    assert missing["recomputed_verdict_class"] == "blocked"
    assert missing["row_count"] is None
    assert gates[1]["passed"] is True
    assert gates[1]["consumer_present"] is False


def test_scenario_report_7147_gates_use_bare_producer_fields(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7147-GATES rejects unequal and nested producer values."""

    _write_complete_fixture(tmp_path)
    producer = _base_payload(7138, source_grounding_fixture_ready_score=0)
    _write_upstream(tmp_path, 7138, producer)
    records = exp.load_upstreams(tmp_path)
    first = exp.recompute_gates(records)[0]
    assert first["observed_value"] == 0
    assert first["passed"] is False

    producer.pop("source_grounding_fixture_ready_score")
    producer["nested"] = {"source_grounding_fixture_ready_score": 1}
    _write_upstream(tmp_path, 7138, producer)
    nested = exp.recompute_gates(exp.load_upstreams(tmp_path))[0]
    assert nested["observed_value"] is None
    assert nested["passed"] is False


def test_scenario_report_7147_verdicts_enforce_oracle_and_gate_ceilings() -> None:
    """SCENARIO-REPORT-7147-VERDICTS prevents positive over-claims."""

    oracle = _base_payload(7139, verifier_is_oracle=True)
    gated = _base_payload(7139)
    blocked = _base_payload(
        7139,
        verdict_class="partial",
        honest_verdict="blocked_native_server",
        rows=[],
    )

    assert exp.recompute_verdict(oracle, gate_passed=True) == "circular_positive"
    assert exp.recompute_verdict(gated, gate_passed=False) == "blocked"
    assert exp.recompute_verdict(blocked, gate_passed=True) == "blocked"
    assert exp.recompute_verdict(None, gate_passed=None) == "blocked"


def test_scenario_report_7147_metric_reducers_use_per_unit_rows() -> None:
    """SCENARIO-REPORT-7147-METRICS derives each branch from row evidence."""

    source = _base_payload(
        7138,
        fixture_rows=[{"id": 1}, {"id": 2}],
        model_view_rows=[{"id": 1}, {"id": 2}],
        sealed_scorer_rows=[{"id": 1}, {"id": 2}],
        independent_loader_rows=[{"exact_match": True}, {"exact_match": True}],
        label_exposure_count=0,
    )
    symbolic = _base_payload(
        7140,
        useful_intervention_rows=[{"model": "a"}, {"model": "b"}],
        harmful_intervention_rows=[{"model": "a"}],
        null_intervention_rows=[{"model": "c"}],
    )
    stream = _base_payload(
        7141,
        event_rows=[
            {"split": "future"},
            {"split": "future"},
            {"split": "protected_retention"},
        ],
    )
    learning = _base_payload(
        7142,
        event_rows=[{"event": 1}],
        future_success_rows=[
            {"arm": "no_memory", "exact_success": 0},
            {"arm": "verifier_balanced", "exact_success": 1},
        ],
        protected_retention_rows=[
            {"arm": "no_memory", "exact_success": 1},
            {"arm": "verifier_balanced", "exact_success": 1},
        ],
    )
    cold = _base_payload(
        7143,
        protected_retention_rows=[{"arm": "verifier_balanced", "exact_success": 1}],
        negative_transfer_rows=[{"arm": "verifier_balanced", "negative_transfer": 0}],
    )

    source_row = exp.recompute_source_grounding(source, None)[0]
    assert source_row["fixture_row_count"] == 2
    assert source_row["independent_loader_all_passed"] is True
    assert exp.recompute_symbolic_interventions(symbolic)[0]["useful_count"] == 2
    csl = exp.recompute_csl(stream, learning)
    assert csl[0]["split_counts"] == {"future": 2, "protected_retention": 1}
    assert csl[1]["future_success_rates"]["verifier_balanced"] == 1.0
    assert exp.recompute_cold_retention(cold)[0]["negative_transfer_rate"] == 0.0
    assert exp.recompute_symbolic_interventions(None) == []
    assert exp.recompute_cold_retention(None) == []


def test_scenario_report_7147_arc_rust_and_gatemate_recompute_rows() -> None:
    """SCENARIO-REPORT-7147-METRICS checks ARC, Rust, and hardware rows."""

    arc = _base_payload(
        7144,
        rows=[
            {"arm": "adapter_withheld", "levels": 0, "executed_transition_count": 5},
            {"arm": "adapter_visible_control", "levels": 1, "executed_transition_count": 3},
        ],
        request_rows=[
            {"arm": arm, "budget": {"actions": 25}, "model_hash": "same", "tools_hash": "same", "seed": 1, "executable_environment_hash": "same"}
            for arm in ("adapter_withheld", "adapter_visible_control")
        ],
        forbidden_read_rows=[
            {"arm": "adapter_withheld", "passed": False, "forbidden_reads": ["registry"]},
            {"arm": "adapter_visible_control", "passed": True, "forbidden_reads": []},
        ],
        import_rows=[
            {"arm": "adapter_withheld", "target_adapter_module_loaded": False, "target_recipe_symbols_loaded": False},
            {"arm": "adapter_visible_control", "target_adapter_module_loaded": True, "target_recipe_symbols_loaded": True},
        ],
        truncation_rows=[
            {"real_output": True, "truncated": True},
            {"real_output": True, "truncated": False},
        ],
        input_difference_rows=[{"measured_difference": True}],
        policy_difference_rows=[{"measured_difference": True}],
    )
    rust = _base_payload(
        7145,
        fixture_rows=[{"passed": True}],
        normalization_rows=[{"passed": True}],
        support_rows=[{"passed": True}],
        detailed_balance_rows=[{"passed": True}],
        stationarity_rows=[{"passed": True}],
        chain_rows=[{"passed": True}],
        benchmark_rows=[
            {"implementation": "python", "updates_per_s": 10.0},
            {"implementation": "rust", "updates_per_s": 25.0},
        ],
    )
    hardware = _base_payload(
        7146,
        receipt_rows=[{"valid": True}],
        command_rows=[{"argv": ["detect"], "returncode": 0}],
    )

    arc_row = exp.recompute_arc(arc)[0]
    assert arc_row["level_delta"] == -1
    assert arc_row["visible_control_nonzero"] is True
    assert arc_row["isolation_clean"] is False
    assert arc_row["all_real_outputs_truncated"] is False
    assert arc_row["common_arm_configuration"] is True
    rust_row = exp.recompute_rust(rust)[0]
    assert rust_row["exact_parity"] is True
    assert rust_row["rust_over_python_throughput"] == 2.5
    hardware_row = exp.recompute_gatemate(hardware)[0]
    assert hardware_row["valid_receipt_count"] == 1
    assert hardware_row["command_count"] == 1

    empty_rust = exp.recompute_rust(_base_payload(7145, rows=[], fixture_rows=[]))[0]
    assert empty_rust["exact_parity"] is None
    assert empty_rust["rust_over_python_throughput"] is None


def test_scenario_report_7147_current_matrix_is_complete_but_blocked(tmp_path: Path) -> None:
    """REQ-REPORT-7147 preserves current missing and blocked evidence."""

    output = tmp_path / "experiment_7147.json"
    artifact = exp.build_artifact(ROOT, "20260908", output)

    assert output.is_file()
    assert artifact["v627_capstone_complete_score"] == 1
    assert artifact["verdict_class"] == "blocked"
    assert artifact["inference_substrate_class"] == "blocked_no_run"
    assert artifact["rows"] == [
        {
            "matrix_slot_coverage_rate": 1.0,
            "present_artifact_rate": pytest.approx(9 / 11),
            "dependency_gate_pass_rate": 0.5,
            "scientific_branch_promotion_rate": 0.0,
        }
    ]
    assert len(artifact["artifact_inventory_rows"]) == 11
    assert len(artifact["gate_recompute_rows"]) == 4
    assert len(artifact["branch_disposition_rows"]) == 7
    assert len(artifact["prd_gap_rows"]) == 3
    assert len(artifact["deferral_rows"]) == len(exp.EXPLICIT_DEFERRALS)
    assert artifact["inference_rerun_count"] == 0
    assert artifact["hardware_command_count"] == 0
    assert {row["experiment"] for row in artifact["artifact_inventory_rows"] if not row["present"]} == {7140, 7143}
    assert {row["disposition"] for row in artifact["branch_disposition_rows"]} <= exp.DISPOSITIONS
    assert all(row["citation"]["field"] and row["citation"]["row_selector"] for row in artifact["branch_disposition_rows"])
    assert exp.validate_artifact(artifact) == []


def test_scenario_report_7147_initial_write_precedes_loader(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7147-INIT makes the initial block observable."""

    output = tmp_path / "experiment_7147.json"

    def observing_loader(_root: Path) -> dict[int, dict[str, Any]]:
        initial = json.loads(output.read_text(encoding="utf-8"))
        assert initial["gate_check_summary"]["failed_check"] == "capstone_initialized"
        return exp.load_upstreams(ROOT)

    artifact = exp.build_artifact(ROOT, "20260908", output, loader=observing_loader)
    assert artifact["gate_check_summary"]["failed_check"] == "upstream_terminal_availability"


def test_scenario_report_7147_validator_catches_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7147-ARTIFACT detects changed derived evidence."""

    artifact = exp.build_artifact(ROOT, "20260908", tmp_path / "experiment_7147.json")
    assert exp.validate_artifact(artifact) == []

    for field, value in (
        ("field_principles", {}),
        ("rows", []),
        ("gate_recompute_rows", []),
        ("branch_disposition_rows", []),
        ("inference_rerun_count", 1),
        ("hardware_command_count", 1),
        ("v627_capstone_complete_score", 0),
        ("honest_verdict", "complete_positive_wrong"),
    ):
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        assert exp.validate_artifact(changed), field

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:wrong"
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(changed)


def test_req_report_7147_cli_writes_and_validates(tmp_path: Path) -> None:
    """REQ-REPORT-7147 exposes writer and validation command paths."""

    _write_complete_fixture(tmp_path)
    output = tmp_path / "capstone.json"
    assert exp.main(["--root", str(tmp_path), "--date", "20260908", "--output", str(output)]) == 0
    assert exp.main(["--validate", "--output", str(output)]) == 0
    output.write_text("{}", encoding="utf-8")
    assert exp.main(["--validate", "--output", str(output)]) == 1


def test_req_report_7147_rejects_bad_date_and_unreadable_json(tmp_path: Path) -> None:
    """REQ-REPORT-7147 fails closed on malformed local evidence."""

    with pytest.raises(ValueError, match="YYYYMMDD"):
        exp.initialize_artifact("2026-09-08")
    path = tmp_path / exp.UPSTREAM_SPECS[7136]["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]", encoding="utf-8")
    record = exp.load_upstreams(tmp_path)[7136]
    assert record["present"] is True
    assert record["payload"] is None
    assert record["read_error"] == "json_root_not_object"


def test_scenario_report_7147_fail_closed_reducer_edges(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7147-VERDICTS exercises every evidence ceiling."""

    malformed = tmp_path / exp.UPSTREAM_SPECS[7136]["path"]
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text("{", encoding="utf-8")
    assert exp.load_upstreams(tmp_path)[7136]["read_error"].startswith("JSONDecodeError:")

    assert exp.recompute_verdict(
        _base_payload(7139, verdict_class="partial", honest_verdict="partial_fixture"),
        gate_passed=True,
    ) == "partial"
    assert exp.recompute_verdict(
        _base_payload(7139, verdict_class="invented", honest_verdict="invented"),
        gate_passed=True,
    ) == "disqualified"
    assert exp.recompute_verdict(
        _base_payload(7139, rows=[]), gate_passed=True
    ) == "disqualified"
    assert exp.recompute_arc(None) == []
    assert exp.recompute_rust(None) == []
    assert exp.recompute_gatemate(None) == []


def test_scenario_report_7147_upstream_validation_fails_closed() -> None:
    """SCENARIO-REPORT-7147-INVENTORY names malformed producer evidence."""

    status, findings = exp._upstream_validation(
        {"present": True, "payload": None, "read_error": "bad"}, None
    )
    assert status == "unreadable" and findings == ["bad"]

    payload = _base_payload(7139)
    payload.pop("run_date")
    payload["verdict_class"] = "invented"
    status, findings = exp._upstream_validation(
        {"present": True, "payload": payload}, True
    )
    assert status == "inconsistent"
    assert "missing_run_date" in findings
    assert "illegal_verdict_class" in findings
    assert "verdict_ceiling:invented->disqualified" in findings

    blocked = _base_payload(
        7139,
        verdict_class="blocked",
        honest_verdict="blocked_fixture",
        gate_check_summary=None,
    )
    assert "blocked_gate_summary_missing" in exp._upstream_validation(
        {"present": True, "payload": blocked}, True
    )[1]
    blocked["gate_check_summary"] = {
        "passed": False,
        "failed_check": "fixture",
        "expected_value": None,
        "observed_value": False,
    }
    assert "blocked_gate_summary_incomplete" in exp._upstream_validation(
        {"present": True, "payload": blocked}, True
    )[1]


def test_scenario_report_7147_validator_rejects_each_terminal_shape(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7147-ARTIFACT covers independent schema failures."""

    artifact = exp.build_artifact(ROOT, "20260908", tmp_path / "artifact.json")

    def failures(**changes: Any) -> list[str]:
        changed = deepcopy(artifact)
        changed.update(changes)
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        return exp.validate_artifact(changed)

    assert "inference_substrate_mismatch" in failures(inference_substrate="wrong")
    assert "inference_substrate_class_invalid" in failures(inference_substrate_class="wrong")
    assert "execution_venue_mismatch" in failures(execution_venue="device")
    assert "artifact_inventory_slots_invalid" in failures(artifact_inventory_rows={})

    branches = deepcopy(artifact["branch_disposition_rows"])
    branches[1]["branch"] = branches[0]["branch"]
    assert "branch_dispositions_not_unique" in failures(branch_disposition_rows=branches)
    branches = deepcopy(artifact["branch_disposition_rows"])
    branches[0]["disposition"] = "invented"
    assert "branch_disposition_invalid" in failures(branch_disposition_rows=branches)
    branches = deepcopy(artifact["branch_disposition_rows"])
    branches[0]["citation"]["field"] = ""
    assert "branch_citation_incomplete" in failures(branch_disposition_rows=branches)

    assert "verdict_class_invalid" in failures(verdict_class="invented")
    assert "blocked_gate_summary_invalid" in failures(
        gate_check_summary={"passed": True}
    )
    assert "blocked_gate_summary_incomplete" in failures(
        gate_check_summary={
            "passed": False,
            "failed_check": "fixture",
            "expected_value": None,
            "observed_value": False,
        }
    )
    assert "blocked_substrate_class_mismatch" in failures(
        inference_substrate_class="aggregation"
    )
    assert "positive_honest_verdict_prefix_mismatch" in failures(
        verdict_class="positive", honest_verdict="wrong"
    )
    assert "capstone_oracle_mismatch" in failures(verifier_is_oracle=True)
    assert exp.main(["--validate", "--output", str(tmp_path / "absent.json")]) == 1
