"""Tests for REQ-HW-7473 and SCENARIO-HW-7473-* evidence boundaries."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

import pytest

from carnot import experiment_7473_v654_board_continuity as exp
from carnot.reporting import current_work_receipt


ROOT = Path(__file__).resolve().parents[2]


def _sources() -> tuple[dict[str, object], dict[str, object]]:
    prior = json.loads((ROOT / exp.PRIOR_PATH).read_text(encoding="utf-8"))
    graduation = json.loads((ROOT / exp.GRADUATION_PATH).read_text(encoding="utf-8"))
    return prior, graduation


def _missing_changed_state(path: Path) -> dict[str, object]:
    return {
        "exists": False,
        "accepted_receipt_count": 0,
        "latest_receipt_date": "20260823",
        "receipt_timestamp": None,
        "changed_conditions": {},
        "evidence_path": None,
        "evidence_hash": None,
        "search_receipt_path": str(path),
        "search_receipt_hash": "sha256:" + "1" * 64,
        "eligibility_contract": deepcopy(exp.PHYSICAL_RECEIPT_CONTRACT),
        "hardware_operations_issued": [],
    }


def test_req_hw_7473_authenticates_exact_prior_rows_and_original_flags() -> None:
    """REQ-HW-7473 keeps producer identity and flags instead of rehabilitating them."""

    prior, graduation = _sources()
    gates = exp.authenticate_sources(prior, graduation, None, None)

    assert gates
    assert all(row["passed"] is True for row in gates)
    by_check = {row["check"]: row for row in gates}
    assert by_check["prior_verdict_class"]["observed"] == "null"
    assert by_check["prior_flagged_adversarial"]["observed"] is False
    assert by_check["graduation_verdict_class"]["observed"] == "positive"
    assert by_check["selector_optional"]["observed"] == "absent"
    assert by_check["prefix_service_optional"]["observed"] == "absent"


def test_scenario_hw_7473_unchanged_preserves_three_narrow_scopes(tmp_path: Path) -> None:
    """SCENARIO-HW-7473-UNCHANGED preserves claims and gives GateMate an exact block."""

    prior, _graduation = _sources()
    changed = _missing_changed_state(tmp_path / "gatemate_search.json")
    rows = exp.reduce_board_rows(prior, changed)
    by_board = {row["board"]: row for row in rows}

    assert set(by_board) == {"KV260", "PolarFire", "GateMate"}
    assert by_board["KV260"]["architecture_limit"] == "k_max<=5"
    assert by_board["KV260"]["future_access"] == "ssh kria only"
    assert by_board["KV260"]["fpga_sampling_claimed"] is True
    assert by_board["PolarFire"]["hash_matched_cpu_dispatch"] is True
    assert by_board["PolarFire"]["fpga_sampling_claimed"] is False
    gatemate = by_board["GateMate"]
    assert gatemate["honest_verdict"] == "blocked_unchanged_physical_prerequisite"
    assert gatemate["gate_check_summary"] == {
        "check": "dated_operator_cable_port_power_board_or_dirtyjtag_change",
        "upstream": "Exp6559 physical boundary",
        "path": str(tmp_path / "gatemate_search.json"),
        "field": "accepted_receipt_count",
        "expected": ">0",
        "observed": 0,
        "operator": ">",
        "passed": False,
    }
    assert all(row["hardware_operations_issued"] == [] for row in rows)
    for row in rows:
        expected = current_work_receipt.canonical_hash(
            {key: value for key, value in row.items() if key != "row_sha256"}
        )
        assert row["row_sha256"] == expected


def test_scenario_hw_7473_changed_is_future_probe_only(tmp_path: Path) -> None:
    """SCENARIO-HW-7473-CHANGED never converts a receipt into a current flash."""

    prior, _graduation = _sources()
    changed = _missing_changed_state(tmp_path / "search.json")
    changed.update(
        {
            "exists": True,
            "accepted_receipt_count": 1,
            "latest_receipt_date": "20260921",
            "receipt_timestamp": "2026-09-21T00:00:00Z",
            "changed_conditions": {"cable": True, "power": True},
            "evidence_path": "ops/operator-followup.md",
            "evidence_hash": "sha256:" + "2" * 64,
        }
    )

    rows = exp.reduce_board_rows(prior, changed)
    gatemate = next(row for row in rows if row["board"] == "GateMate")
    assert gatemate["honest_verdict"] == "complete_changed_physical_prerequisite_future_probe_only"
    assert gatemate["gate_check_summary"] is None
    assert gatemate["changed_state_evidence"]["changed_fields"] == ["cable", "power"]
    assert gatemate["future_probe"]["requires_separate_review"] is True
    assert gatemate["future_probe"]["flash_authorized"] is False
    assert gatemate["hardware_operations_issued"] == []
    assert exp.independent_reduce({"board_rows": rows})["gatemate_changed_state_score"] == 1


def test_scenario_hw_7473_wishlist_keeps_optional_evidence_and_prerequisites() -> None:
    """SCENARIO-HW-7473-WISHLIST keeps software and access gates explicit."""

    absent = exp.build_hardware_wishlist_disposition(None, None)
    assert absent["selector_evidence"]["availability"] == "not_produced"
    assert absent["prefix_service_evidence"]["availability"] == "not_produced"
    assert absent["npu_route"]["software_prerequisite_satisfied"] is False
    assert "VitisAI" in absent["npu_route"]["missing_prerequisite"]
    assert absent["tsu_route"]["authenticated_access"] is False
    assert absent["paper_or_sdk_proves_device_availability"] is False
    assert absent["purchase_count"] == absent["vendor_contact_count"] == 0

    selector = {
        "schema": "carnot.exp7466.v654.typed_energy_calibration.v1",
        "experiment_id": "exp7466-v654-typed-energy-calibration",
        "milestone": exp.MILESTONE,
        "run_date": exp.RUN_DATE,
        "status": "complete_null_selector_ties_simple_control",
        "honest_verdict": "complete_null_selector_ties_simple_control",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "typed_decision_value_score": 0,
    }
    prefix = {
        "schema": "carnot.exp7472.v654.prefix_service.v1",
        "experiment_id": "exp7472-v654-prefix-service",
        "milestone": exp.MILESTONE,
        "run_date": exp.RUN_DATE,
        "status": "complete_null_prefix_service_speed_gate_not_met",
        "honest_verdict": "complete_null_prefix_service_speed_gate_not_met",
        "verdict_class": "null",
        "flagged_adversarial": False,
        "prefix_parity_score": 1,
        "prefix_service_value_score": 0,
    }
    mapped = exp.build_hardware_wishlist_disposition(selector, prefix)
    assert mapped["selector_evidence"]["typed_decision_value_score"] == 0
    assert mapped["prefix_service_evidence"]["prefix_parity_score"] == 1
    assert mapped["prefix_service_evidence"]["prefix_service_value_score"] == 0
    assert mapped["hardware_speed_claimed"] is False
    prior, graduation = _sources()
    assert all(
        row["passed"] is True
        for row in exp.authenticate_sources(prior, graduation, selector, prefix)
    )


def test_req_hw_7473_defensive_input_and_receipt_boundaries(tmp_path: Path) -> None:
    """REQ-HW-7473 rejects malformed rows and incomplete command receipts."""

    assert datetime.fromisoformat(exp.utc_now()).tzinfo is not None
    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    assert exp._load_object(missing) == {}
    assert exp._load_object(malformed) == {}
    assert exp._load_object(scalar) == {}

    prior, _graduation = _sources()
    with pytest.raises(ValueError, match="prior_board_rows_missing"):
        exp.reduce_board_rows({}, {})
    with pytest.raises(ValueError, match="prior_board_rows_invalid"):
        exp.reduce_board_rows({"board_rows": prior["board_rows"][:2]}, {})

    names = ("one", "two")
    passing = [{"name": name, "passed": True} for name in names]
    assert exp._required_receipts(passing, names) is True
    assert exp._required_receipts(passing[:1], names) is False
    assert exp._required_receipts([*passing, passing[0]], names) is False


def test_req_hw_7473_fixture_validates_and_mutations_fail(tmp_path: Path) -> None:
    """REQ-HW-7473 cold validation rejects model, board, and gate drift."""

    artifact = exp.build_fixture_artifact(ROOT, tmp_path)
    assert exp.validate_artifact(artifact, root=ROOT, require_terminal=False) == []
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["verdict_class"] == "null"
    assert artifact["gatemate_changed_state_score"] == 0
    assert artifact["hardware_operations_issued"] == []
    assert set(artifact["field_principles"]) == set(artifact)

    mutations = {
        "lowercase_model_specs": ("model_specs", ["phantom/model"]),
        "gatemate_score": ("gatemate_changed_state_score", 1),
        "hardware_operations": ("hardware_operations_issued", ["detect"]),
    }
    for expected_error, (field, value) in mutations.items():
        broken = deepcopy(artifact)
        broken[field] = value
        assert expected_error in exp.validate_artifact(broken, root=ROOT, require_terminal=False)

    broken = deepcopy(artifact)
    kv260 = next(row for row in broken["board_rows"] if row["board"] == "KV260")
    kv260["architecture_limit"] = "unbounded"
    assert "board_claim_boundary" in exp.validate_artifact(
        broken, root=ROOT, require_terminal=False
    )

    broken = deepcopy(artifact)
    gatemate = next(row for row in broken["board_rows"] if row["board"] == "GateMate")
    gatemate["gate_check_summary"] = None
    assert "blocked_board_gate_summary" in exp.validate_artifact(
        broken, root=ROOT, require_terminal=False
    )

    scalar_mutations = {
        "identity": ("schema", "wrong"),
        "uppercase_model_specs": ("MODEL_SPECS", ["phantom/model"]),
        "current_inference_boundary": ("model_invoked", True),
        "field_principles": ("field_principles", {}),
        "hardware_operation_count": ("hardware_operation_count", 1),
        "hardware_scores": ("hardware_ready_score", 1),
        "wishlist_boundary": ("hardware_wishlist_disposition", {}),
        "affected_receipts": ("validation_required", True),
    }
    for expected_error, (field, value) in scalar_mutations.items():
        broken = deepcopy(artifact)
        broken[field] = value
        assert expected_error in exp.validate_artifact(broken, root=ROOT, require_terminal=False)

    assert "terminal_receipts" in exp.validate_artifact(artifact, root=ROOT, require_terminal=True)

    broken = deepcopy(artifact)
    broken["source_artifact_hashes"]["ignored"] = {"verify_on_replay": False}
    assert "source_hash:ignored" not in exp.validate_artifact(
        broken, root=ROOT, require_terminal=False
    )
    broken = deepcopy(artifact)
    source = next(iter(broken["source_artifact_hashes"].values()))
    source["sha256"] = "sha256:" + "0" * 64
    assert any(
        error.startswith("source_hash:")
        for error in exp.validate_artifact(broken, root=ROOT, require_terminal=False)
    )

    broken = deepcopy(artifact)
    broken["raw_evidence_reference"] = None
    assert "raw_reference" in exp.validate_artifact(broken, root=ROOT, require_terminal=False)
    broken = deepcopy(artifact)
    broken["raw_evidence_reference"] = {
        "path": str(tmp_path / "absent.json"),
        "sha256": "sha256:" + "0" * 64,
    }
    assert "raw_missing" in exp.validate_artifact(broken, root=ROOT, require_terminal=False)
    broken = deepcopy(artifact)
    broken["raw_evidence_reference"]["sha256"] = "sha256:" + "0" * 64
    assert "raw_hash" in exp.validate_artifact(broken, root=ROOT, require_terminal=False)


def test_req_hw_7473_validation_plan_is_affected_only(tmp_path: Path) -> None:
    """REQ-HW-7473 freezes the Exp7358 and Exp7303 affected-only command plan."""

    private = tmp_path / "validation"
    private.mkdir()
    commands = exp.build_validation_plan(ROOT, private)

    assert exp.validate_validation_plan(ROOT, commands) == []
    assert [command.name for command in commands] == [
        "worktree_imports",
        "focused_pytest",
        "changed_module_coverage",
        "changed_module_coverage_report",
        "ruff_check",
        "ruff_format",
        "changed_module_mypy",
        "scoped_spec_coverage",
    ]
    joined = "\n".join(" ".join(command.argv) for command in commands)
    assert "tests/python/test_experiment_7473_v654_board_continuity.py" in joined
    assert "pytest tests/python -q" not in joined
    assert "full_python_suite" not in joined
