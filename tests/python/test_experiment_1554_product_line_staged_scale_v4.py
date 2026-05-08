"""Tests for Exp1554 product-line staged scale v4.

Spec: REQ-VERIFY-1554, SCENARIO-VERIFY-1554.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot.verify import product_line_staged_scale_v4 as exp


def test_req_verify_1554_validates_stage_manifest_solver_fields() -> None:
    """REQ-VERIFY-1554: manifest rows must retain deterministic solver fields."""

    rows = exp.build_stage_manifest(target_count=8)
    validation = exp.validate_stage_manifest_rows(rows)

    assert validation["valid"] is True
    assert set(validation["stages_tested"]) == {
        "syntax_only",
        "feasibility",
        "objective",
        "natural_language",
    }
    assert validation["cases_total"] == 8

    broken = [copy.deepcopy(row) for row in rows]
    broken[0].pop("oracle_result")
    broken_validation = exp.validate_stage_manifest_rows(broken)

    assert broken_validation["valid"] is False
    assert broken_validation["deterministic_checks_available"] is False
    assert any("missing_required_field:oracle_result" in err for err in broken_validation["errors"])

    broken_number = [copy.deepcopy(row) for row in rows]
    broken_number[1]["oracle_label"].pop("optimal_cost")
    number_validation = exp.validate_stage_manifest_rows(broken_number)
    assert any(
        "missing_deterministic_field:oracle_label.optimal_cost" in err
        for err in number_validation["errors"]
    )


def test_scenario_verify_1554_aggregates_solver_grounded_metrics() -> None:
    """SCENARIO-VERIFY-1554: metrics are derived from structured row fields."""

    rows = exp.build_stage_manifest(target_count=8)
    metrics = exp.aggregate_manifest_metrics(rows)

    assert metrics["parse_rate"] == pytest.approx(1.0)
    assert metrics["feasibility_rate"] == pytest.approx(1.0)
    assert metrics["oracle_agreement_rate"] == pytest.approx(1.0)
    assert metrics["objective_gap_mean"] == pytest.approx(0.0)
    assert metrics["entity_hallucination_rate"] > 0.0
    assert metrics["false_accept_rate"] == pytest.approx(0.0)

    mismatched = [copy.deepcopy(row) for row in rows]
    oracle_label = mismatched[0]["oracle_label"]
    mismatched[0]["oracle_result"]["oracle_agrees"] = False
    mismatched[0]["oracle_result"]["classification"] = "wrong_or_suboptimal"
    mismatched[0]["oracle_result"]["selection_value"] = oracle_label["optimal_value"] - 2
    mismatched[0]["policy_result"]["accepted"] = True
    mismatched[0]["policy_result"]["false_accept"] = True
    min_cost_row = next(row for row in mismatched if row["operation"]["kind"] == "min_cost")
    min_cost_label = min_cost_row["oracle_label"]
    min_cost_row["oracle_result"]["oracle_agrees"] = False
    min_cost_row["oracle_result"]["classification"] = "wrong_or_suboptimal"
    min_cost_row["oracle_result"]["selection_cost"] = min_cost_label["optimal_cost"] + 3

    mismatch_metrics = exp.aggregate_manifest_metrics(mismatched)
    decision = exp.decide_scale_v4_readiness(
        cases_total=len(mismatched),
        validation=exp.validate_stage_manifest_rows(mismatched),
        metrics=mismatch_metrics,
        unified_contract_gate_ready=True,
        focused_tests_passed=True,
        blockers=[],
    )

    assert mismatch_metrics["objective_gap_mean"] > 0.0
    assert mismatch_metrics["false_accept_rate"] > 0.0
    assert exp._entity_hallucination_detected({"stages": [None]}) is False  # noqa: SLF001
    assert decision["product_line_scale_v4_ready"] is False
    assert decision["branch_retired"] is True
    assert "false_accept_rate" in decision["retirement_reason"]


def test_scenario_verify_1554_runner_writes_required_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1554: runner writes a ready zero-false-accept artifact."""

    output = tmp_path / "experiment_1554.json"
    manifest = tmp_path / "product_line_v4.jsonl"
    predecessors = _write_predecessors(tmp_path)

    artifact = exp.run_experiment(
        project_root=tmp_path,
        output_path=output,
        manifest_path=manifest,
        predecessor_paths=predecessors,
        target_count=12,
        focused_tests_passed=True,
        model_probe_fn=lambda _root: {
            "live_sota_model_inference_used": False,
            "models_used": [],
            "availability_blockers": ["no_mandated_sota_gguf_runtime"],
            "legacy_small_models_excluded_from_headline_metrics": True,
        },
    )
    persisted = json.loads(output.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert artifact == persisted
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["product_line_scale_v4_ready"] is True
    assert artifact["branch_retired"] is False
    assert artifact["cases_total"] == 12
    assert set(artifact["stages_tested"]) == set(exp.STAGE_VARIANTS)
    assert artifact["parse_rate"] == pytest.approx(1.0)
    assert artifact["feasibility_rate"] == pytest.approx(1.0)
    assert artifact["oracle_agreement_rate"] == pytest.approx(1.0)
    assert artifact["objective_gap_mean"] == pytest.approx(0.0)
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["focused_tests_passed"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(rows) == artifact["cases_total"]


def test_req_verify_1554_blocks_without_gate_and_retires_nondeterministic_rows(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1554: Exp1551 readiness and deterministic rows gate readiness."""

    blocked_predecessors = _write_predecessors(tmp_path / "blocked", gate_ready=False)
    blocked = exp.run_experiment(
        project_root=tmp_path,
        output_path=tmp_path / "blocked.json",
        manifest_path=tmp_path / "blocked.jsonl",
        predecessor_paths=blocked_predecessors,
        target_count=4,
        focused_tests_passed=True,
        model_probe_fn=lambda _root: {"availability_blockers": []},
    )

    assert blocked["status"] == "blocked"
    assert blocked["product_line_scale_v4_ready"] is False
    assert blocked["branch_retired"] is False
    assert "exp1551_unified_contract_gate_not_ready" in blocked["blockers"]

    empty = exp.run_experiment(
        project_root=tmp_path,
        output_path=tmp_path / "empty.json",
        manifest_path=tmp_path / "empty.jsonl",
        predecessor_paths=_write_predecessors(tmp_path / "empty"),
        target_count=4,
        focused_tests_passed=True,
        row_builder_fn=lambda _count: [],
        model_probe_fn=lambda _root: {"availability_blockers": []},
    )
    assert empty["status"] == "blocked"
    assert "no_product_line_v4_cases" in empty["blockers"]

    unfocused = exp.run_experiment(
        project_root=tmp_path,
        output_path=tmp_path / "unfocused.json",
        manifest_path=tmp_path / "unfocused.jsonl",
        predecessor_paths=_write_predecessors(tmp_path / "unfocused"),
        target_count=4,
        focused_tests_passed=False,
        model_probe_fn=lambda _root: {"availability_blockers": []},
    )
    assert "focused_tests_not_passed" in unfocused["blockers"]

    missing_loaded, missing_blockers = exp.load_predecessor_artifacts(
        exp.PredecessorPaths(
            exp1540_artifact=tmp_path / "missing1540.json",
            exp1551_artifact=tmp_path / "missing1551.json",
        ),
        project_root=tmp_path,
    )
    assert missing_loaded == {}
    assert len(missing_blockers) == 2

    bad_predecessors = _write_predecessors(tmp_path / "bad")
    bad_predecessors.exp1540_artifact.write_text(
        json.dumps(
            {
                "status": "complete",
                "product_line_scale_ready": False,
                "false_accept_rate": 0.5,
                "oracle_agreement_rate": 0.5,
            }
        ),
        encoding="utf-8",
    )
    bad_predecessors.exp1551_artifact.write_text(
        json.dumps(
            {
                "status": "complete",
                "unified_contract_gate_ready": True,
                "false_accept_rate": 0.5,
                "product_line_oracle_used": False,
            }
        ),
        encoding="utf-8",
    )
    _bad_loaded, bad_blockers = exp.load_predecessor_artifacts(
        bad_predecessors,
        project_root=tmp_path,
    )
    assert "exp1540_product_line_scale_not_ready" in bad_blockers
    assert "exp1540_false_accept_rate_nonzero:0.5" in bad_blockers
    assert "exp1540_oracle_agreement_below_one:0.5" in bad_blockers
    assert "exp1551_false_accept_rate_nonzero:0.5" in bad_blockers
    assert "exp1551_product_line_oracle_not_used" in bad_blockers

    predecessors = _write_predecessors(tmp_path / "nondeterministic")
    rows = exp.build_stage_manifest(target_count=4)
    rows[0].pop("oracle_result")
    retired = exp.run_experiment(
        project_root=tmp_path,
        output_path=tmp_path / "retired.json",
        manifest_path=tmp_path / "retired.jsonl",
        predecessor_paths=predecessors,
        target_count=4,
        focused_tests_passed=True,
        row_builder_fn=lambda _count: rows,
        model_probe_fn=lambda _root: {"availability_blockers": []},
    )

    assert retired["status"] == "complete"
    assert retired["product_line_scale_v4_ready"] is False
    assert retired["branch_retired"] is True
    assert "deterministic_manifest_fields_missing" in retired["retirement_reason"]


def _write_predecessors(tmp_path: Path, *, gate_ready: bool = True) -> exp.PredecessorPaths:
    tmp_path.mkdir(parents=True, exist_ok=True)
    exp1540 = tmp_path / "experiment_1540.json"
    exp1551 = tmp_path / "experiment_1551.json"
    exp1540.write_text(
        json.dumps(
            {
                "status": "complete",
                "product_line_scale_ready": True,
                "false_accept_rate": 0.0,
                "oracle_agreement_rate": 1.0,
            }
        ),
        encoding="utf-8",
    )
    exp1551.write_text(
        json.dumps(
            {
                "status": "complete",
                "unified_contract_gate_ready": gate_ready,
                "false_accept_rate": 0.0,
                "product_line_oracle_used": True,
            }
        ),
        encoding="utf-8",
    )
    return exp.PredecessorPaths(exp1540_artifact=exp1540, exp1551_artifact=exp1551)
