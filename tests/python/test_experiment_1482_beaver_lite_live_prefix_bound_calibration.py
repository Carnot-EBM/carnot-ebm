"""Tests for Exp 1482 BEAVER-lite live-prefix bound calibration.

Spec: REQ-VERIFY-1482, SCENARIO-VERIFY-1482.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.reporting.beaver_lite_live_prefix_bound_calibration import (  # noqa: E402
    MANDATED_SOTA_GGUF_MODELS,
    REQUIRED_ARTIFACT_FIELDS,
    build_artifact,
    compatible_exp1480_rows,
    evaluate_live_prefix_row,
    run,
    validate_artifact,
    write_in_progress_artifact,
)


def _live_row(
    case_id: str,
    family: str,
    answer: str,
    *,
    hf_id: str = "unsloth/Qwen3.6-35B-A3B-GGUF",
    format_valid: bool = True,
    topk_available: bool = True,
) -> dict[str, object]:
    token_texts = ["\n\n", "<think>", "\n\n", "</think>", "\n\n", answer]
    return {
        "case_id": case_id,
        "family": family,
        "generation_source": "live_sota_llamacpp",
        "hf_id": hf_id,
        "model_name": "Qwen3.6-35B-A3B",
        "model_path": "/cache/qwen.gguf",
        "prompt": f"Prompt for {case_id}. Return the final integer only.",
        "expected_answer": answer,
        "response_text": "".join(token_texts),
        "response_text_available": True,
        "format_valid": format_valid,
        "token_texts": token_texts,
        "token_logprobs": [-0.01, -0.02, -0.03, 0.0, -0.001, -0.1],
        "token_logprobs_available": True,
        "top_logprobs": [
            {"\n\n": -0.01},
            {"<think>": -0.02},
            {"\n\n": -0.03},
            {"</think>": 0.0},
            {"\n\n": -0.001},
            {answer: -0.1, "7": -2.0, "<think>": -3.0, ".": -4.0, " words": -5.0},
        ],
        "topk_alternatives_available": topk_available,
        "topk_position_count": 6 if topk_available else 0,
    }


def _write_live_inputs(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    exp_id: str = "1480",
) -> tuple[Path, Path]:
    artifact_path = tmp_path / f"experiment_{exp_id}_live_sota_telemetry.json"
    manifest_path = tmp_path / f"live_sota_telemetry_manifest_{exp_id}.jsonl"
    artifact_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "honest_verdict": "balanced_live_sota_telemetry_ready",
                "live_sota_model_inference_used": True,
                "topk_logprobs_available": True,
                "logits_available": True,
                "models_used": ["unsloth/Qwen3.6-35B-A3B-GGUF"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return artifact_path, manifest_path


def test_req_verify_1482_writes_in_progress_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-1482-1: seed the calibration artifact before loading rows."""

    out_path = tmp_path / "experiment_1482_beaver_lite_live_prefix_bound_calibration.json"

    artifact = write_in_progress_artifact(out_path)

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "in_progress"
    assert artifact["constraints_evaluated"] == 0
    assert artifact["mock_or_live_logprobs"] == "pending"
    assert artifact["broad_benchmark_deferred"] is True


def test_req_verify_1482_live_row_bound_uses_terminal_topk_prefix_mass() -> None:
    """REQ-VERIFY-1482-4: live top-k terminal alternatives produce a sound bound."""

    evaluation = evaluate_live_prefix_row(
        _live_row("case-001", "arithmetic_word_problem", "3"),
        constraint_index=0,
    )

    assert evaluation["constraint"]["constraint_id"] == "exp1482_case_001"
    assert evaluation["constraint"]["prefix_closed"] is True
    assert evaluation["empirical_violation_rate"] == 0.0
    assert evaluation["unsafe_mass_bound"] == pytest.approx(
        math.exp(-3.0) + math.exp(-4.0) + math.exp(-5.0)
    )
    assert evaluation["bound_gap"] == pytest.approx(evaluation["unsafe_mass_bound"])


def test_scenario_verify_1482_uses_exp1480_live_prefix_rows(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1482: compatible Exp 1480 rows are labeled live."""

    families = ["fover_claim", "arithmetic_word_problem", "constraint_check"]
    rows = [
        _live_row(f"case-{index:03d}", families[index % len(families)], str(index % 10))
        for index in range(14)
    ]
    exp1480_artifact_path, exp1480_manifest_path = _write_live_inputs(tmp_path, rows)
    out_path = tmp_path / "experiment_1482_beaver_lite_live_prefix_bound_calibration.json"

    artifact = run(
        output_path=out_path,
        exp1480_artifact_path=exp1480_artifact_path,
        exp1480_manifest_path=exp1480_manifest_path,
        exp1468_artifact_path=tmp_path / "missing-1468.json",
        exp1468_manifest_path=tmp_path / "missing-1468.jsonl",
        limit=14,
    )

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["constraints_evaluated"] == 14
    assert len(artifact["prefix_closed_constraints"]) == 14
    assert artifact["mock_or_live_logprobs"] == "live_exp1480"
    assert artifact["model_specs"][0]["hf_id"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
    assert artifact["bound_is_sound"] is True
    assert artifact["bound_violations"] == []
    assert all(0.0 <= value <= 1.0 for value in artifact["unsafe_mass_bounds"])
    assert artifact["empirical_violation_rates"] == [0.0] * 14
    assert artifact["calibration_tightness_summary"]["p50_slack"] >= 0.0
    assert artifact["calibration_tightness_summary"]["p90_slack"] >= 0.0
    assert artifact["calibration_tightness_summary"]["max_slack"] >= 0.0
    assert artifact["broad_benchmark_deferred"] is True
    assert artifact["honest_verdict"] == "sound_bound_live_exp1480_calibrated"


def test_req_verify_1482_filters_incompatible_live_rows(tmp_path: Path) -> None:
    """REQ-VERIFY-1482-2/3: live rows must be mandated, top-k, and terminal valid."""

    valid = _live_row("valid", "fover_claim", "1")
    bad_model = _live_row("bad-model", "fover_claim", "1", hf_id="not/a-mandated-model")
    bad_format = _live_row("bad-format", "fover_claim", "1", format_valid=False)
    bad_topk = _live_row("bad-topk", "fover_claim", "1", topk_available=False)
    artifact_path, manifest_path = _write_live_inputs(
        tmp_path,
        [valid, bad_model, bad_format, bad_topk],
    )

    rows = compatible_exp1480_rows(artifact_path, manifest_path, limit=12)

    assert [row["case_id"] for row in rows] == ["valid"]
    assert "unsloth/Qwen3.6-35B-A3B-GGUF" in MANDATED_SOTA_GGUF_MODELS

    blocked_artifact = tmp_path / "blocked-exp1480.json"
    blocked_artifact.write_text(
        json.dumps(
            {
                "status": "blocked",
                "live_sota_model_inference_used": False,
                "topk_logprobs_available": False,
                "models_used": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert compatible_exp1480_rows(blocked_artifact, manifest_path, limit=12) == []


def test_req_verify_1482_extends_short_exp1480_with_compatible_exp1468_rows(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1482-2: Exp 1468 live rows extend a short Exp 1480 set."""

    exp1480_rows = [
        _live_row(f"exp1480-case-{index:03d}", "arithmetic_word_problem", str(index % 10))
        for index in range(9)
    ]
    exp1468_rows = [
        _live_row(f"exp1468-case-{index:03d}", "gsm8k_style", str(index % 10))
        for index in range(5)
    ]
    exp1480_artifact_path, exp1480_manifest_path = _write_live_inputs(
        tmp_path,
        exp1480_rows,
        exp_id="1480",
    )
    exp1468_artifact_path, exp1468_manifest_path = _write_live_inputs(
        tmp_path,
        exp1468_rows,
        exp_id="1468",
    )

    artifact = run(
        output_path=tmp_path / "experiment_1482.json",
        exp1480_artifact_path=exp1480_artifact_path,
        exp1480_manifest_path=exp1480_manifest_path,
        exp1468_artifact_path=exp1468_artifact_path,
        exp1468_manifest_path=exp1468_manifest_path,
        limit=12,
    )

    assert artifact["constraints_evaluated"] == 12
    assert artifact["mock_or_live_logprobs"] == "live_exp1480_plus_exp1468"
    assert artifact["source_artifacts"] == [
        "results/experiment_1480_live_sota_balanced_telemetry_v2.json",
        "results/experiment_1468_live_sota_logprob_telemetry_preflight.json",
    ]


def test_req_verify_1482_uses_exp1468_when_it_is_the_only_live_source(tmp_path: Path) -> None:
    """REQ-VERIFY-1482-2: Exp 1468 can be the sole compatible live lineage."""

    exp1468_rows = [
        _live_row(f"exp1468-only-{index:03d}", "gsm8k_style", str(index % 10))
        for index in range(12)
    ]
    exp1468_artifact_path, exp1468_manifest_path = _write_live_inputs(
        tmp_path,
        exp1468_rows,
        exp_id="1468",
    )

    artifact = run(
        output_path=tmp_path / "experiment_1482.json",
        exp1480_artifact_path=tmp_path / "missing-1480.json",
        exp1480_manifest_path=tmp_path / "missing-1480.jsonl",
        exp1468_artifact_path=exp1468_artifact_path,
        exp1468_manifest_path=exp1468_manifest_path,
        limit=12,
    )

    assert artifact["constraints_evaluated"] == 12
    assert artifact["mock_or_live_logprobs"] == "live_exp1468"
    assert artifact["honest_verdict"] == "sound_bound_live_exp1468_calibrated"


def test_req_verify_1482_falls_back_to_mock_logprobs_when_live_rows_are_unavailable(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1482-2: missing live telemetry uses explicit mock lineage."""

    out_path = tmp_path / "experiment_1482_beaver_lite_live_prefix_bound_calibration.json"

    artifact = run(
        output_path=out_path,
        exp1480_artifact_path=tmp_path / "missing-1480.json",
        exp1480_manifest_path=tmp_path / "missing-1480.jsonl",
        exp1468_artifact_path=tmp_path / "missing-1468.json",
        exp1468_manifest_path=tmp_path / "missing-1468.jsonl",
        limit=12,
        mock_top_k=10,
    )

    assert artifact["status"] == "complete"
    assert artifact["constraints_evaluated"] == 12
    assert artifact["mock_or_live_logprobs"] == "mock_logprobs"
    assert artifact["unsafe_mass_bounds"] == pytest.approx([0.4] * 12)
    assert artifact["empirical_violation_rates"] == pytest.approx([0.4] * 12)
    assert artifact["bound_is_sound"] is True
    assert artifact["honest_verdict"] == "sound_bound_mock_logprobs_calibrated"


def test_req_verify_1482_validation_rejects_unsound_or_mislabeled_artifacts() -> None:
    """REQ-VERIFY-1482-4/5: schema validation catches unsafe calibration output."""

    valid = build_artifact(
        constraints=[
            {
                "constraint_id": f"c{index}",
                "prefix_closed": True,
                "terminal_only": True,
                "source_case_id": f"case-{index}",
                "expected_answer": str(index),
            }
            for index in range(12)
        ],
        unsafe_mass_bounds=[0.1] * 12,
        empirical_violation_rates=[0.0] * 12,
        mock_or_live_logprobs="live_exp1480",
        models_used=["unsloth/Qwen3.6-35B-A3B-GGUF"],
    )
    validate_artifact(valid)

    with pytest.raises(ValueError, match="missing required fields"):
        validate_artifact({key: value for key, value in valid.items() if key != "honest_verdict"})
    with pytest.raises(ValueError, match="12 to 20"):
        validate_artifact(valid | {"constraints_evaluated": 11})
    with pytest.raises(ValueError, match="counts must match"):
        validate_artifact(valid | {"prefix_closed_constraints": valid["prefix_closed_constraints"][:-1]})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_artifact(valid | {"unsafe_mass_bounds": [1.1] + [0.1] * 11})
    with pytest.raises(ValueError, match="exceeds bound"):
        validate_artifact(valid | {"empirical_violation_rates": [0.2] + [0.0] * 11})
    with pytest.raises(ValueError, match="bound_violations"):
        validate_artifact(valid | {"bound_violations": [{"constraint_id": "c0"}]})
    with pytest.raises(ValueError, match="bound_is_sound"):
        validate_artifact(valid | {"bound_is_sound": False})
    with pytest.raises(ValueError, match="mock_or_live_logprobs"):
        validate_artifact(valid | {"mock_or_live_logprobs": "unclear"})
    with pytest.raises(ValueError, match="broad_benchmark_deferred"):
        validate_artifact(valid | {"broad_benchmark_deferred": False})
    with pytest.raises(ValueError, match="exceeds bound"):
        build_artifact(
            constraints=valid["prefix_closed_constraints"],
            unsafe_mass_bounds=[0.0] + [0.1] * 11,
            empirical_violation_rates=[0.2] + [0.0] * 11,
            mock_or_live_logprobs="live_exp1480",
            models_used=["unsloth/Qwen3.6-35B-A3B-GGUF"],
        )
    with pytest.raises(ValueError, match="12 to 20"):
        build_artifact(
            constraints=[valid["prefix_closed_constraints"][0]],
            unsafe_mass_bounds=[0.1],
            empirical_violation_rates=[0.0],
            mock_or_live_logprobs="live_exp1480",
            models_used=["unsloth/Qwen3.6-35B-A3B-GGUF"],
        )
