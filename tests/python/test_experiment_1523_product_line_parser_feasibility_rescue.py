"""Tests for Exp 1523 product-line staged feedback rescue.

Spec: REQ-BENCH-1523, SCENARIO-BENCH-1523.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import product_line_parser_feasibility_rescue as rescue
from carnot.eval import product_line_solver_oracle_benchmark as exp1511


def _baseline_rows() -> list[dict[str, Any]]:
    return rescue.load_jsonl(Path("results/product_line_solver_oracle_1511.jsonl"))


def test_req_bench_1523_reproduces_exp1511_baseline_metrics() -> None:
    """REQ-BENCH-1523: baseline metrics are reproduced before rescue."""

    metrics = rescue.reproduce_baseline_metrics(_baseline_rows())

    assert metrics["baseline_parse_rate"] == pytest.approx(0.333333)
    assert metrics["baseline_feasibility_rate"] == pytest.approx(0.0)
    assert metrics["baseline_oracle_agreement_rate"] == pytest.approx(0.0)
    assert metrics["baseline_false_accept_rate"] == pytest.approx(0.0)


def test_req_bench_1523_staged_feedback_repairs_parse_failure_with_audit_trail() -> None:
    """REQ-BENCH-1523: syntax, feature-model, solver, and policy stages are recorded."""

    row = _baseline_rows()[0]
    case = {case.case_id: case for case in exp1511.build_feature_model_cases()}[row["case_id"]]

    rescued = rescue.apply_staged_feedback(case, row)

    assert [stage["stage"] for stage in rescued["stages"]] == [
        "syntax_parse_feedback",
        "feature_model_consistency_feedback",
        "solver_feasibility_feedback",
        "policy_compliance_feedback",
    ]
    assert rescued["baseline_result"]["parse_ok"] is False
    assert rescued["parse_result"]["parse_ok"] is True
    assert rescued["oracle_result"]["feasible"] is True
    assert rescued["oracle_result"]["oracle_agrees"] is True
    assert rescued["policy_result"]["accepted"] is True
    assert rescued["policy_result"]["false_accept"] is False
    assert (
        rescued["final_answer"]["selected_features"] == rescued["oracle_result"]["optimal_features"]
    )


def test_req_bench_1523_feature_feedback_normalizes_selection_before_solver() -> None:
    """REQ-BENCH-1523: feature-model feedback removes unknowns and closes requirements."""

    case = exp1511.build_feature_model_cases()[0]
    noisy_row = {
        "case_id": case.case_id,
        "model_hf_id": exp1511.MANDATED_MODEL_SPECS[0]["hf_id"],
        "model_name": "Qwen3.6-35B-A3B",
        "generation_source": "live_sota_llamacpp",
        "model_output": json.dumps(
            {
                "selected_features": ["Coupons", "Bogus"],
                "objective_cost": 0,
                "objective_value": 0,
                "verifier": {"accept": False},
            }
        ),
        "parse_result": {"parse_ok": True},
        "oracle_result": {
            "classification": "wrong_or_suboptimal",
            "feasible": False,
            "oracle_agrees": False,
        },
        "verifier_result": {"self_verifier_false_accept": False},
    }

    rescued = rescue.apply_staged_feedback(case, noisy_row)
    feature_stage = rescued["stages"][1]

    assert feature_stage["status"] == "repaired"
    assert "removed_unknown:Bogus" in feature_stage["feedback"]
    assert "added_required:Catalog,Checkout,Store" in feature_stage["feedback"]
    assert "closed_requires:Coupons->Loyalty" in feature_stage["feedback"]
    assert rescued["oracle_result"]["oracle_agrees"] is True


def test_req_bench_1523_solver_stage_passes_when_answer_is_already_optimal() -> None:
    """REQ-BENCH-1523: solver feedback records pass-through optimal rows."""

    case = exp1511.build_feature_model_cases()[1]
    row = {
        "case_id": case.case_id,
        "model_hf_id": exp1511.MANDATED_MODEL_SPECS[0]["hf_id"],
        "model_name": "Qwen3.6-35B-A3B",
        "generation_source": "live_sota_llamacpp",
        "model_output": exp1511.compliant_answer_for_case(case),
        "parse_result": {"parse_ok": True},
        "oracle_result": {
            "classification": "oracle_agreement",
            "feasible": True,
            "oracle_agrees": True,
        },
        "verifier_result": {"self_verifier_false_accept": False},
    }

    rescued = rescue.apply_staged_feedback(case, row)

    assert rescued["stages"][2]["status"] == "passed"
    assert rescued["policy_result"]["accepted"] is True


def test_req_bench_1523_policy_rejects_non_oracle_agreement() -> None:
    """SCENARIO-BENCH-1523: policy feedback does not false-accept bad selections."""

    case = exp1511.build_feature_model_cases()[0]
    payload = rescue.finalize_policy_payload(case, ["Store", "Catalog", "Checkout"])

    assert payload["verifier"]["accept"] is False
    assert payload["policy_result"]["accepted"] is False
    assert payload["policy_result"]["false_accept"] is False
    assert payload["oracle_result"]["classification"] == "wrong_or_suboptimal"


def test_scenario_bench_1523_runner_writes_ready_manifest_and_artifact(tmp_path: Path) -> None:
    """SCENARIO-BENCH-1523: runner persists required fields and zero false accepts."""

    artifact_path = tmp_path / "experiment_1523.json"
    manifest_path = tmp_path / "product_line_rescue_1523.jsonl"

    artifact = rescue.run_rescue(
        baseline_path=Path("results/product_line_solver_oracle_1511.jsonl"),
        output_path=artifact_path,
        manifest_path=manifest_path,
        cached_pair_fn=lambda gpu_indices=(0, 1): [  # noqa: ARG005
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": exp1511.MANDATED_MODEL_SPECS[0]["hf_id"],
                "gpu": 0,
                "model_path": "/tmp/qwen.gguf",
            }
        ],
        gpu_probe_fn=lambda: {"gpu_count": 1},
        tests_run=["focused pytest"],
    )
    persisted = json.loads(artifact_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert artifact == persisted
    assert set(rescue.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["product_line_rescue_ready"] is True
    assert artifact["product_line_branch_retired"] is False
    assert artifact["baseline_parse_rate"] == pytest.approx(0.333333)
    assert artifact["rescue_parse_rate"] > artifact["baseline_parse_rate"]
    assert artifact["rescue_feasibility_rate"] > artifact["baseline_feasibility_rate"]
    assert artifact["rescue_oracle_agreement_rate"] > artifact["baseline_oracle_agreement_rate"]
    assert artifact["false_accept_count"] == 0
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["models_used"] == [exp1511.MANDATED_MODEL_SPECS[0]["hf_id"]]
    assert artifact["honest_verdict"].startswith("complete:")
    assert len(rows) == 6
    assert all(row["policy_result"]["false_accept"] is False for row in rows)


def test_req_bench_1523_runner_blocks_without_mandated_sota_cache(tmp_path: Path) -> None:
    """REQ-BENCH-1523: no legacy tiny model fallback is used for headline rescue."""

    def cache_error(**_kwargs: Any) -> None:
        raise RuntimeError("cache probe failed")

    artifact = rescue.run_rescue(
        baseline_path=Path("results/product_line_solver_oracle_1511.jsonl"),
        output_path=tmp_path / "blocked.json",
        manifest_path=tmp_path / "blocked.jsonl",
        cached_pair_fn=cache_error,
        gpu_probe_fn=lambda: {"gpu_count": 0},
    )

    assert artifact["status"] == "blocked"
    assert artifact["live_sota_model_inference_used"] is False
    assert artifact["product_line_rescue_ready"] is False
    assert artifact["product_line_branch_retired"] is True
    assert artifact["models_used"] == []
    assert "cached_sota_pair_error:RuntimeError: cache probe failed" in artifact["blockers"]
    assert "cached_sota_pair_not_available" in artifact["blockers"]
    assert artifact["honest_verdict"].startswith("complete_blocked:")


def test_req_bench_1523_runner_blocks_without_live_exp1511_rows(tmp_path: Path) -> None:
    """REQ-BENCH-1523: cached models alone are not enough without live SOTA source rows."""

    rows = _baseline_rows()
    for row in rows:
        row["generation_source"] = "fixture_not_headline"
    baseline_path = tmp_path / "no_live.jsonl"
    baseline_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    artifact = rescue.run_rescue(
        baseline_path=baseline_path,
        output_path=tmp_path / "blocked_no_live.json",
        manifest_path=tmp_path / "blocked_no_live.jsonl",
        cached_pair_fn=lambda gpu_indices=(0, 1): [  # noqa: ARG005
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": exp1511.MANDATED_MODEL_SPECS[0]["hf_id"],
                "gpu": 0,
                "model_path": "/tmp/qwen.gguf",
            }
        ],
        gpu_probe_fn=lambda: {"gpu_count": 1},
    )

    assert artifact["status"] == "blocked"
    assert artifact["product_line_branch_retired"] is True
    assert artifact["blockers"] == ["exp1511_live_sota_rows_not_available"]
