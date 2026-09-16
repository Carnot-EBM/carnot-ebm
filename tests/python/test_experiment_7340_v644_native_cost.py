"""Tests for the V644 complete-boundary native cost experiment.

Spec refs: REQ-VERIFY-7340, SCENARIO-VERIFY-7340-*, REQ-PYBIND-7340,
and SCENARIO-PYBIND-7340-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7340_v644_native_cost as exp7340


ROOT = Path(__file__).resolve().parents[2]


def test_req_verify_7340_authenticates_exact_same_milestone_producer(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-7340: only the exact eligible Exp7339 result can supply scores."""

    checks, hashes, upstream = exp7340.collect_preconditions(ROOT)
    summary = exp7340.gate_check_summary(checks)
    assert summary["passed"] is True
    assert hashes["experiment_7339"] == exp7340.EXPECTED_EXP7339_SHA256
    assert upstream["status"] == "complete"
    assert upstream["native_binding_ready_score"] == 1

    missing_checks, missing_hashes, missing = exp7340.collect_preconditions(tmp_path)
    missing_summary = exp7340.gate_check_summary(missing_checks)
    assert missing == {}
    assert missing_hashes == {}
    assert missing_summary["passed"] is False
    assert missing_summary["first_failure"] == {
        "upstream": str(tmp_path / exp7340.UPSTREAM_RELATIVE_PATH),
        "check": "exp7339_available",
        "field": "path",
        "expected_value": True,
        "observed_value": False,
        "passed": False,
        "principle": "The exact same-milestone native binding result must exist.",
    }

    malformed_path = tmp_path / exp7340.UPSTREAM_RELATIVE_PATH
    malformed_path.parent.mkdir(parents=True, exist_ok=True)
    malformed_path.write_text("[]", encoding="utf-8")
    malformed_checks, malformed_hashes, malformed = exp7340.collect_preconditions(
        tmp_path,
        expected_upstream_sha256=exp7340.sha256_file(malformed_path),
    )
    assert malformed == {}
    assert malformed_hashes == {"experiment_7339": exp7340.sha256_file(malformed_path)}
    assert malformed_checks[-1]["check"] == "exp7339_parseable"
    assert malformed_checks[-1]["observed_value"] == "ValueError"


def test_scenario_verify_7340_preflight_rejects_quarantine_and_partial(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7340-PREFLIGHT: ineligible producers fail before rows."""

    upstream_path = tmp_path / exp7340.UPSTREAM_RELATIVE_PATH
    upstream_path.parent.mkdir(parents=True)
    payload = exp7340.eligible_upstream_fixture_for_test(tmp_path)
    payload["flagged_adversarial"] = True
    payload["verdict_class"] = "partial"
    upstream_path.write_text(json.dumps(payload), encoding="utf-8")

    checks, _, _ = exp7340.collect_preconditions(
        tmp_path,
        expected_upstream_sha256=exp7340.sha256_file(upstream_path),
    )
    failed = [row for row in checks if row["passed"] is False]
    assert [row["check"] for row in failed] == [
        "exp7339_eligible_class",
        "exp7339_not_quarantined",
    ]
    blocked = exp7340.blocked_artifact(checks, {})
    assert blocked["status"] == "complete"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["native_cost_complete_score"] == 0
    assert blocked["native_ten_x_score"] == 0
    assert blocked["rows"] == []
    assert blocked["honest_verdict"].startswith("blocked_exp7339_eligible_class:")
    assert exp7340.validate_artifact(blocked) == []


def test_scenario_verify_7340_validation_temp_parent_is_prepared(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7340-VALIDATION-TEMP: pytest receives an existing parent."""

    private_parent = tmp_path / "private" / "exp7340"

    assert exp7340.prepare_scoped_basetemp(private_parent) == private_parent
    assert private_parent.is_dir()


def test_scenario_pbind_7340_boundary_spans_are_complete_and_nonoverlapping() -> None:
    """SCENARIO-PYBIND-7340-BOUNDARY: all conversion and result work is charged."""

    rows = exp7340.synthetic_cost_rows(native_ratio=20.0)
    row = rows[1]
    assert row["arm"] == "rust_in_process"
    assert row["complete_boundary_ns"] == sum(
        row[name]
        for name in (
            "input_marshalling_ns",
            "evaluation_ns",
            "results_ns",
            "service_transport_ns",
        )
    )
    assert row["stage_overlap_check_passed"] is True
    assert row["parity_matched"] is True
    assert row["output_sha256"] == rows[0]["output_sha256"]


def test_scenario_verify_7340_cost_reducer_requires_all_ninety_paired_blocks() -> None:
    """SCENARIO-VERIFY-7340-COST: the denominator, parity, and stages cannot shrink."""

    rows = exp7340.synthetic_cost_rows(native_ratio=20.0)
    summary = exp7340.reduce_cost_rows(rows)
    assert summary["complete"] is True
    assert summary["paired_blocks"] == 90
    assert summary["row_count"] == 270
    assert summary["parity_mismatches"] == 0
    assert summary["stage_overlap_failures"] == 0
    assert summary["by_batch_size"]["32"]["latency_ns"]["rust_in_process"] == {
        "p50": 32_000.0,
        "p95": 32_000.0,
    }

    missing = exp7340.reduce_cost_rows(rows[:-1])
    assert missing["complete"] is False
    assert missing["paired_blocks"] == 89

    bad = deepcopy(rows)
    bad[0]["output_sha256"] = "sha256:different"
    bad[1]["stage_overlap_check_passed"] = False
    bad[2].pop("results_ns")
    reduced_bad = exp7340.reduce_cost_rows(bad)
    assert reduced_bad["complete"] is False
    assert reduced_bad["parity_mismatches"] == 1
    assert reduced_bad["stage_overlap_failures"] == 2


def test_scenario_verify_7340_ten_x_gate_uses_paired_bootstrap_lower_bounds() -> None:
    """SCENARIO-VERIFY-7340-GATE: every native CI95 lower bound must reach ten."""

    fast = exp7340.reduce_cost_rows(exp7340.synthetic_cost_rows(native_ratio=20.0))
    assert all(
        row["throughput_ratio_native_over_python"]["ci95_lower"] == 20.0
        for row in fast["by_batch_size"].values()
    )
    assert exp7340.derive_scores(fast) == {
        "native_cost_complete_score": 1,
        "native_ten_x_score": 1,
    }

    null = exp7340.reduce_cost_rows(exp7340.synthetic_cost_rows(native_ratio=2.0))
    assert exp7340.derive_scores(null) == {
        "native_cost_complete_score": 1,
        "native_ten_x_score": 0,
    }
    assert all(
        row["throughput_ratio_native_over_python"]["ci95_lower"] == 2.0
        for row in null["by_batch_size"].values()
    )
    assert exp7340.paired_bootstrap_ci95([]) == {
        "mean": None,
        "ci95_lower": None,
        "ci95_upper": None,
        "blocks": 0,
    }
    assert exp7340._percentile([0.0, 10.0], 0.5) == 5.0
    with pytest.raises(ValueError, match="percentile_requires_values"):
        exp7340._percentile([], 0.5)


def test_scenario_verify_7340_amortization_charges_setup_once() -> None:
    """SCENARIO-VERIFY-7340-AMORTIZATION: setup and steady costs never overlap."""

    assert exp7340.break_even_request_count(9_000, 2_000.0, 1_000.0) == 9
    assert exp7340.break_even_request_count(9_001, 2_000.0, 1_000.0) == 10
    assert exp7340.break_even_request_count(9_000, 1_000.0, 1_000.0) is None
    assert exp7340.break_even_request_count(9_000, 900.0, 1_000.0) is None

    amortization = exp7340.amortization_summary(
        setup_costs={"cold_import_ns": 4_000, "constraint_compilation_ns": 5_000},
        reduced=exp7340.reduce_cost_rows(exp7340.synthetic_cost_rows(native_ratio=2.0)),
    )
    assert amortization["setup_ns"] == 9_000
    assert amortization["break_even_requests_by_size"] == {
        "1": 9,
        "32": 9,
        "256": 9,
    }
    assert amortization["v643_whole_learning_upper_bound"] is None
    assert amortization["whole_learning_speedup_claimed"] is False


def test_scenario_pbind_7340_e2e003_rechecks_loaded_binary_identity() -> None:
    """SCENARIO-PYBIND-7340-E2E003: the measured extension hash stays bound."""

    checks, _, upstream = exp7340.collect_preconditions(ROOT)
    assert exp7340.gate_check_summary(checks)["passed"] is True
    extension = Path(upstream["binding_identity"]["module_file"])
    evidence = exp7340.e2e_003_identity(extension, upstream["binding_identity"])
    assert evidence["passed"] is True
    assert evidence["observed"]["module_sha256"] == upstream["binding_identity"]["module_sha256"]

    changed = deepcopy(upstream["binding_identity"])
    changed["module_sha256"] = "sha256:stale"
    assert exp7340.e2e_003_identity(extension, changed)["passed"] is False


def test_scenario_verify_7340_terminal_null_is_complete_and_tamper_evident(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7340-TERMINAL: a sub-10x result is a complete scoped null."""

    artifact = exp7340.complete_artifact_fixture_for_test(tmp_path, native_ratio=2.0)
    assert exp7340.validate_artifact(artifact) == []
    assert artifact["schema"] == "carnot.experiment_7340.v644_native_cost.v1"
    assert artifact["status"] == "complete"
    assert artifact["run_date"] == "20260916"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["native_cost_complete_score"] == 1
    assert artifact["native_ten_x_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null:")
    assert artifact["retirement"]["scope"] == "this_exact_complete_boundary"
    assert artifact["whole_learning_speedup_claimed"] is False

    changed = deepcopy(artifact)
    changed["rows"][0]["complete_boundary_ns"] += 1
    assert "row_reduction" in exp7340.validate_artifact(changed)

    output = tmp_path / "terminal.json"
    exp7340.write_artifact(output, artifact)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact

    positive = exp7340.complete_artifact_fixture_for_test(tmp_path, native_ratio=20.0)
    assert positive["verdict_class"] == "positive"
    assert positive["native_ten_x_score"] == 1

    disqualified = exp7340._base_artifact([], {})
    rows = exp7340.synthetic_cost_rows(native_ratio=2.0)
    disqualified = exp7340._complete_artifact(
        disqualified,
        rows,
        exp7340.reduce_cost_rows(rows),
        {"cold_import_ns": 1},
        affected_validation_passed=False,
        e2e_passed=True,
        adverse_passed=True,
    )
    assert disqualified["verdict_class"] == "disqualified"
    assert exp7340.validate_artifact(disqualified) == []

    changed["reproducibility_checksum"] = exp7340.reproducibility_checksum(changed)
    with pytest.raises(ValueError, match="artifact_validation_failed:.*row_reduction"):
        exp7340.write_artifact(tmp_path / "invalid.json", changed)
