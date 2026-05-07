"""Tests for Exp 1481 Semantic Energy feasibility audit.

Spec: REQ-VERIFY-1481, SCENARIO-VERIFY-1481.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting.semantic_energy_feasibility_audit import (
    REQUIRED_ARTIFACT_FIELDS,
    build_case_feature_row,
    build_semantic_energy_payload,
    evaluate_feature_signals,
    extract_semantic_energy_features,
    run_experiment,
)


QWEN = "unsloth/Qwen3.6-35B-A3B-GGUF"


def _topk(expected_lp: float, wrong_lp: float) -> list[dict[str, float]]:
    return [
        {"\n\n": -0.02, "<think>": -4.0, "The": -5.0},
        {"1": expected_lp, "0": wrong_lp, "2": -4.0, " yes": -5.0, " no": -5.5},
    ]


def _row(
    *,
    case_id: str,
    label: int,
    expected_lp: float,
    wrong_lp: float,
    response_length: int = 10,
    overlap: float = 0.0,
    schema_valid: bool = True,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "hf_id": QWEN,
        "expected_answer": "1",
        "adversarial_wrong_answer": "0",
        "known_verifier_label": label,
        "correct": bool(label),
        "correctness_label": "correct" if label else "incorrect",
        "format_valid": schema_valid,
        "logits_available": True,
        "topk_alternatives_available": True,
        "token_logprobs_available": True,
        "top_logprobs": _topk(expected_lp, wrong_lp),
        "token_logprobs": [-0.1, max(expected_lp, wrong_lp)],
        "response_text": "1" if label else "0",
        "superficial_baselines": {
            "response_length": response_length,
            "token_count": 2,
            "json_valid": False,
            "schema_valid": schema_valid,
            "prompt_family": "fover_claim",
            "answer_lexical_overlap": overlap,
            "model_family": "qwen_moe",
        },
    }


def test_req_verify_1481_extracts_bounded_semantic_features() -> None:
    """REQ-VERIFY-1481: feature rows include all bounded Semantic Energy proxies."""
    row = _row(
        case_id="correct-confident",
        label=1,
        expected_lp=-0.05,
        wrong_lp=-3.0,
    )

    features = extract_semantic_energy_features(row)
    feature_row = build_case_feature_row(row)

    assert features["final_logit_entropy"] > 0.0
    assert features["topk_semantic_cluster_proxy"] > 0.0
    assert features["answer_choice_energy_gap"] < 0.0
    assert features["per_case_uncertainty_spread"] >= 0.0
    assert feature_row["semantic_failure_label"] == 0
    assert feature_row["label_source"] == "known_binary_verifier_label"
    assert set(features) == {
        "final_logit_entropy",
        "topk_semantic_cluster_proxy",
        "answer_choice_energy_gap",
        "per_case_uncertainty_spread",
    }


def test_req_verify_1481_edge_cases_fail_closed_without_inventing_labels() -> None:
    """REQ-VERIFY-1481: malformed telemetry and missing labels stay bounded."""
    empty_features = extract_semantic_energy_features(
        {
            "expected_answer": "",
            "adversarial_wrong_answer": "",
            "top_logprobs": [{"bad": "not-a-float"}],
        }
    )
    fallback_row = build_case_feature_row(
        {
            "case_id": "fallback-label",
            "correct": False,
            "expected_answer": "",
            "adversarial_wrong_answer": "",
            "top_logprobs": [{"\n\n": -0.1, "<think>": -0.2, "maybe": -0.3, "yes": -0.4}],
            "superficial_baselines": {"response_length": "long"},
        }
    )
    unlabeled_row = build_case_feature_row({"case_id": "unlabeled", "top_logprobs": []})

    assert empty_features == {
        "final_logit_entropy": None,
        "topk_semantic_cluster_proxy": None,
        "answer_choice_energy_gap": None,
        "per_case_uncertainty_spread": None,
    }
    assert fallback_row["semantic_failure_label"] == 1
    assert fallback_row["label_source"] == "response_correctness_label"
    assert fallback_row["response_length"] is None
    assert fallback_row["topk_semantic_cluster_proxy"] > 0.0
    assert unlabeled_row["semantic_failure_label"] is None
    assert unlabeled_row["label_source"] == "unlabeled"


def test_req_verify_1481_semantic_signal_must_strictly_beat_baselines() -> None:
    """REQ-VERIFY-1481: semantic claim is allowed only above all superficial baselines."""
    rows = [
        _row(case_id="correct-1", label=1, expected_lp=-0.05, wrong_lp=-3.0),
        _row(case_id="correct-2", label=1, expected_lp=-0.08, wrong_lp=-2.8),
        _row(case_id="wrong-1", label=0, expected_lp=-3.0, wrong_lp=-0.05),
        _row(case_id="wrong-2", label=0, expected_lp=-2.8, wrong_lp=-0.08),
    ]

    payload = build_semantic_energy_payload(
        rows,
        source_artifact={"status": "complete", "logits_available": True, "model_specs": [QWEN]},
        run_date="20260507",
        diagnostic_path=Path("results/semantic_energy_features_1481.json"),
    )

    assert payload["semantic_energy_features_computed"] is True
    assert payload["baseline_features_computed"] is True
    assert payload["best_semantic_signal"]["name"] == "answer_choice_energy_gap"
    assert payload["best_semantic_signal"]["oriented_auroc"] == pytest.approx(1.0)
    assert payload["best_superficial_baseline"]["oriented_auroc"] == pytest.approx(0.5)
    assert payload["signal_beats_superficial_baselines"] is True
    assert payload["claim_allowed"] is True
    assert payload["diagnostic_lineage_retired"] is False


def test_scenario_verify_1481_matching_baseline_blocks_claim() -> None:
    """SCENARIO-VERIFY-1481: a matched superficial baseline retires headline lineage."""
    rows = [
        _row(case_id="correct-1", label=1, expected_lp=-0.05, wrong_lp=-3.0, response_length=10),
        _row(case_id="correct-2", label=1, expected_lp=-0.08, wrong_lp=-2.8, response_length=11),
        _row(case_id="wrong-1", label=0, expected_lp=-3.0, wrong_lp=-0.05, response_length=100),
        _row(case_id="wrong-2", label=0, expected_lp=-2.8, wrong_lp=-0.08, response_length=101),
    ]
    feature_rows = [build_case_feature_row(row) for row in rows]

    semantic = evaluate_feature_signals(
        feature_rows,
        ("answer_choice_energy_gap",),
        feature_source="semantic_energy",
    )
    baseline = evaluate_feature_signals(
        feature_rows,
        ("response_length",),
        feature_source="superficial_baseline",
    )
    payload = build_semantic_energy_payload(
        rows,
        source_artifact={"status": "complete", "logits_available": True, "model_specs": [QWEN]},
        run_date="20260507",
        diagnostic_path=Path("results/semantic_energy_features_1481.json"),
    )

    assert semantic["best_signal"]["oriented_auroc"] == pytest.approx(1.0)
    assert baseline["best_signal"]["oriented_auroc"] == pytest.approx(1.0)
    assert payload["best_superficial_baseline"]["name"] == "response_length"
    assert payload["signal_beats_superficial_baselines"] is False
    assert payload["claim_allowed"] is False
    assert payload["diagnostic_lineage_retired"] is True
    assert payload["honest_verdict"] == "retired_semantic_energy_confounded_by_superficial_baseline"


def test_req_verify_1481_run_writes_artifact_and_feature_file(tmp_path: Path) -> None:
    """REQ-VERIFY-1481: runner writes complete artifact and per-case diagnostics."""
    results = tmp_path / "results"
    source_path = results / "experiment_1480_live_sota_balanced_telemetry_v2.json"
    manifest_path = results / "live_sota_balanced_telemetry_manifest_1480.jsonl"
    output_path = results / "experiment_1481_semantic_energy_feasibility_audit.json"
    diagnostic_path = results / "semantic_energy_features_1481.json"
    results.mkdir(parents=True)
    source_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "logits_available": True,
                "topk_logprobs_available": True,
                "model_specs": [QWEN],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = [
        _row(case_id="correct-1", label=1, expected_lp=-0.05, wrong_lp=-3.0),
        _row(case_id="wrong-1", label=0, expected_lp=-3.0, wrong_lp=-0.05, response_length=40),
    ]
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    artifact = run_experiment(
        project_root=tmp_path,
        run_date="20260507",
        source_artifact_path=source_path,
        manifest_path=manifest_path,
        output_path=output_path,
        diagnostic_path=diagnostic_path,
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["semantic_energy_audit_complete"] is True
    assert artifact["telemetry_rows_loaded"] == 2
    diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert diagnostics["run_date"] == "20260507"
    assert len(diagnostics["case_features"]) == 2
    assert diagnostics["case_features"][0]["case_id"] == "correct-1"


def test_req_verify_1481_missing_logits_fails_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-1481: missing Exp 1480 logits blocks without fresh inference."""
    output_path = tmp_path / "results" / "experiment_1481.json"

    artifact = run_experiment(
        project_root=tmp_path,
        run_date="20260507",
        source_artifact_path=tmp_path / "missing_source.json",
        manifest_path=tmp_path / "missing_manifest.jsonl",
        output_path=output_path,
        diagnostic_path=tmp_path / "results" / "diagnostics.json",
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["telemetry_rows_loaded"] == 0
    assert artifact["semantic_energy_audit_complete"] is False
    assert artifact["claim_allowed"] is False
    assert artifact["diagnostic_lineage_retired"] is True
    assert artifact["honest_verdict"] == "blocked_no_reusable_exp1480_logits"
