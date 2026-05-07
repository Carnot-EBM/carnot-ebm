"""Tests for Exp 1469 HALT and Spilled-Energy telemetry diagnostics.

Spec: REQ-VERIFY-1469, SCENARIO-VERIFY-1469
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting.halt_spilled_energy_telemetry_diagnostic import (
    REQUIRED_ARTIFACT_FIELDS,
    build_case_diagnostic,
    build_diagnostic_payload,
    binary_auroc,
    evaluate_rank_signals,
    extract_final_answer,
    extract_telemetry_features,
    run_experiment,
)


def _row(
    *,
    case_id: str,
    expected_answer: str,
    token_logprobs: list[float],
    top_logprobs: list[dict[str, float]],
    response_text: str,
    family: str = "fover_style",
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "family": family,
        "expected_answer": expected_answer,
        "response_text": response_text,
        "completion_tokens": len(token_logprobs),
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "topk_alternatives_available": bool(top_logprobs),
        "token_logprobs_available": bool(token_logprobs),
    }


def _topk_rows(gap: float = 2.0) -> list[dict[str, float]]:
    return [
        {"A": -0.1, "B": -0.1 - gap, "C": -3.0},
        {"A": -0.2, "B": -0.2 - gap / 2.0, "C": -3.0},
        {"A": -0.4, "B": -0.4 - gap / 4.0, "C": -3.0},
    ]


def test_exp1469_feature_extractor_computes_halt_and_spilled_proxies() -> None:
    """REQ-VERIFY-1469: feature rows include HALT trends and spill proxies."""
    row = _row(
        case_id="fover_gsm8k_verified_008",
        expected_answer="0",
        token_logprobs=[-0.1, -0.2, -0.4],
        top_logprobs=_topk_rows(gap=2.0),
        response_text="\n\n<think>\n\n</think>\n\n0",
    )

    features = extract_telemetry_features(row)
    diagnostic = build_case_diagnostic(row)

    assert features["token_logprob_trend"] < 0.0
    assert features["topk_entropy_mean"] > 0.0
    assert features["topk_gap_trend"] < 0.0
    assert features["spilled_energy_proxy_mean"] >= 0.0
    assert features["marginal_energy_proxy_mean"] >= 0.0
    assert diagnostic["known_verifier_label"] == 1
    assert diagnostic["response_correct_label"] == 1
    assert diagnostic["label_source"] == "known_binary_verifier_label"


def test_exp1469_final_answer_extraction_covers_common_completion_shapes() -> None:
    """REQ-VERIFY-1469: correctness labels use deterministic final-answer parsing."""
    assert extract_final_answer("\n\n<think>\n\n</think>\n\n17") == "17"
    assert extract_final_answer("Reasoning\n\n**Answer:** 1") == "1"
    assert extract_final_answer("No terminal numeric answer here") is None

    gsm_row = _row(
        case_id="fover_gsm8k_verified_005",
        expected_answer="14",
        token_logprobs=[-0.1],
        top_logprobs=[{"14": -0.1, "21": -2.0}],
        response_text="\n\n<think>\n\n</think>\n\n21",
        family="gsm8k_style",
    )
    diagnostic = build_case_diagnostic(gsm_row)
    assert diagnostic["known_verifier_label"] is None
    assert diagnostic["response_correct_label"] == 0
    assert diagnostic["label_source"] == "response_correctness_label"

    no_answer = build_case_diagnostic(
        _row(
            case_id="fover_no_answer",
            expected_answer="1",
            token_logprobs=[-0.1],
            top_logprobs=[{"1": -0.1, "0": -2.0}],
            response_text="No terminal numeric answer here",
        )
    )
    assert no_answer["response_correct_label"] is None


def test_exp1469_rank_signals_use_oriented_auroc() -> None:
    """SCENARIO-VERIFY-1469: AUROC reports both raw and oriented directions."""
    assert binary_auroc([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.8]) == pytest.approx(1.0)
    assert binary_auroc([0, 1], [0.4, 0.4]) == pytest.approx(0.5)

    diagnostics = [
        {"known_verifier_label": 0, "features": {"spilled_energy_proxy_mean": 0.1}},
        {"known_verifier_label": 0, "features": {"spilled_energy_proxy_mean": 0.2}},
        {"known_verifier_label": 1, "features": {"spilled_energy_proxy_mean": 0.8}},
        {"known_verifier_label": 1, "features": {"spilled_energy_proxy_mean": 0.9}},
    ]

    summary = evaluate_rank_signals(
        diagnostics,
        candidate_features=("spilled_energy_proxy_mean",),
    )

    assert summary["best_signal_name"] == "spilled_energy_proxy_mean"
    assert summary["best_signal"]["auroc"] == pytest.approx(1.0)
    assert summary["best_signal"]["direction"] == "higher"
    assert summary["best_signal"]["n"] == 4
    assert summary["small_sample_caveat"] is True


def test_exp1469_edge_cases_cover_degenerate_labels_and_blocked_artifact(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1469: malformed telemetry and missing manifests fail closed."""
    malformed = _row(
        case_id="malformed",
        expected_answer="",
        token_logprobs=[True, "bad"],
        top_logprobs=[None, {}, {"A": True, "B": "bad"}],
        response_text="No terminal numeric answer here",
    )
    malformed_diagnostic = build_case_diagnostic(malformed)

    assert malformed_diagnostic["known_verifier_label"] is None
    assert malformed_diagnostic["response_correct_label"] is None
    assert malformed_diagnostic["label_source"] == "unlabeled"
    assert malformed_diagnostic["features"]["topk_entropy_mean"] == pytest.approx(0.0)
    assert binary_auroc([0, 0], [0.1, 0.2]) is None

    no_pairs = evaluate_rank_signals(
        [{"known_verifier_label": None, "features": {"token_logprob_mean": 0.1}}],
        candidate_features=("token_logprob_mean",),
    )
    one_class = evaluate_rank_signals(
        [{"known_verifier_label": 0, "features": {"token_logprob_mean": 0.1}}],
        candidate_features=("token_logprob_mean",),
    )
    assert no_pairs["best_signal"] is None
    assert one_class["best_signal"] is None

    fallback_payload = build_diagnostic_payload(
        [
            _row(
                case_id="gsm_ok",
                expected_answer="3",
                token_logprobs=[-0.1],
                top_logprobs=[{"3": -0.1, "4": -2.0}],
                response_text="3",
                family="gsm8k_style",
            ),
            _row(
                case_id="gsm_bad",
                expected_answer="4",
                token_logprobs=[-0.2],
                top_logprobs=[{"3": -0.2, "4": -1.0}],
                response_text="3",
                family="gsm8k_style",
            ),
        ],
        run_date="20260507",
    )
    assert fallback_payload["label_key"] == "response_failure_label"

    output = tmp_path / "results" / "blocked.json"
    artifact = run_experiment(
        project_root=tmp_path,
        source_artifact_path=tmp_path / "missing_source.json",
        manifest_path=tmp_path / "missing_manifest.jsonl",
        output_path=output,
        diagnostic_path=tmp_path / "results" / "blocked_diagnostics.json",
    )
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["telemetry_rows_loaded"] == 0
    assert artifact["telemetry_diagnostic_complete"] is False
    assert artifact["diagnostic_lineage_retired"] is True
    assert artifact["honest_verdict"] == "blocked_no_reusable_topk_telemetry"


def test_exp1469_run_experiment_writes_artifact_and_retires_flat_signal(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1469: flat logprob signal retires as non-headline telemetry."""
    source_artifact = tmp_path / "results" / "experiment_1468.json"
    manifest = tmp_path / "results" / "manifest.jsonl"
    output = tmp_path / "results" / "experiment_1469.json"
    diagnostic_path = tmp_path / "results" / "diagnostics.json"
    source_artifact.parent.mkdir(parents=True)
    source_artifact.write_text(
        json.dumps(
            {
                "status": "complete",
                "topk_logprobs_available": True,
                "model_specs": [{"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF"}],
            }
        ),
        encoding="utf-8",
    )
    rows = [
        _row(
            case_id="true_short_1",
            expected_answer="1",
            token_logprobs=[-0.1, -0.1, -0.1],
            top_logprobs=_topk_rows(gap=1.0),
            response_text="1",
        ),
        _row(
            case_id="true_short_2",
            expected_answer="1",
            token_logprobs=[-0.1, -0.1, -0.1],
            top_logprobs=_topk_rows(gap=1.0),
            response_text="1",
        ),
        _row(
            case_id="false_long_1",
            expected_answer="0",
            token_logprobs=[-0.1, -0.1, -0.1],
            top_logprobs=_topk_rows(gap=1.0),
            response_text="long reasoning before 0",
        ),
        _row(
            case_id="false_long_2",
            expected_answer="0",
            token_logprobs=[-0.1, -0.1, -0.1],
            top_logprobs=_topk_rows(gap=1.0),
            response_text="another long reasoning before 0",
        ),
    ]
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    writes: list[dict[str, Any]] = []

    artifact = run_experiment(
        project_root=tmp_path,
        run_date="20260507",
        source_artifact_path=source_artifact,
        manifest_path=manifest,
        output_path=output,
        diagnostic_path=diagnostic_path,
        write_json_fn=lambda path, payload: (
            writes.append(dict(payload)),
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8"),
        ),
    )

    diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert writes[0]["status"] == "in_progress"
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["telemetry_rows_loaded"] == 4
    assert artifact["halt_features_computed"] is True
    assert artifact["spilled_energy_features_computed"] is True
    assert artifact["telemetry_diagnostic_complete"] is True
    assert artifact["length_or_format_confound_checked"] is True
    assert artifact["diagnostic_lineage_preserved"] is False
    assert artifact["diagnostic_lineage_retired"] is True
    assert artifact["honest_verdict"] == "retired_non_headline_telemetry_flat_or_confounded"
    assert diagnostics["case_count"] == 4
    assert len(diagnostics["cases"]) == 4
