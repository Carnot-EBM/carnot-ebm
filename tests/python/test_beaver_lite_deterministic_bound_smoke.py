"""Tests for Exp 1470 BEAVER-lite deterministic-bound smoke.

Spec: REQ-VERIFY-1470, SCENARIO-VERIFY-1470.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting.beaver_lite_deterministic_bound_smoke import (
    REQUIRED_ARTIFACT_FIELDS,
    MANDATED_SOTA_GGUF_MODELS,
    build_artifact,
    compatible_exp1468_rows,
    run,
    validate_artifact,
    write_in_progress_artifact,
)


def _live_row(case_id: str, prompt: str, answer: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "family": "gsm8k_style",
        "generation_source": "live_sota_llamacpp",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "model_name": "Qwen3.6-35B-A3B",
        "model_path": "/cache/qwen.gguf",
        "prompt": prompt,
        "expected_answer": answer,
        "response_text": f"\n\n<think>\n\n</think>\n\n{answer}",
        "response_text_available": True,
        "token_texts": ["\n\n", "<think>", "\n\n", "</think>", "\n\n", answer],
        "token_logprobs": [-0.01, -0.02, -0.03, -0.0, -0.001, -0.002],
        "token_logprobs_available": True,
        "topk_alternatives_available": True,
    }


def _write_exp1468_inputs(tmp_path: Path, rows: list[dict[str, object]]) -> tuple[Path, Path]:
    artifact_path = tmp_path / "experiment_1468_live_sota_logprob_telemetry_preflight.json"
    manifest_path = tmp_path / "live_sota_telemetry_manifest_1468.jsonl"
    artifact_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "honest_verdict": "live_sota_topk_telemetry_ready",
                "live_sota_model_inference_used": True,
                "topk_logprobs_available": True,
                "telemetry_manifest_path": str(manifest_path),
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


def test_req_verify_1470_writes_in_progress_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-1470-1: seed the deliverable before the smoke evaluates rows."""

    out_path = tmp_path / "experiment_1470_beaver_lite_deterministic_bound_smoke.json"

    artifact = write_in_progress_artifact(out_path)

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "in_progress"
    assert artifact["bound_is_sound"] is False
    assert artifact["mock_or_live_logprobs"] == "pending"


def test_scenario_verify_1470_uses_compatible_exp1468_live_rows(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1470: compatible mandated GGUF telemetry is labeled live."""

    rows = [
        _live_row("case-1", "Mia has 1 marble and gets 2 more.", "3"),
        _live_row("case-2", "A bus has 9 riders. 4 get off and 6 get on.", "11"),
        _live_row("case-3", "Noah buys 3 packs with 4 pencils each.", "12"),
    ]
    exp1468_artifact_path, exp1468_manifest_path = _write_exp1468_inputs(tmp_path, rows)
    out_path = tmp_path / "experiment_1470_beaver_lite_deterministic_bound_smoke.json"

    artifact = run(
        output_path=out_path,
        exp1468_artifact_path=exp1468_artifact_path,
        exp1468_manifest_path=exp1468_manifest_path,
    )

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["questions_evaluated"] == [row["prompt"] for row in rows]
    assert [item["source_case_id"] for item in artifact["prefix_closed_constraint"]] == [
        "case-1",
        "case-2",
        "case-3",
    ]
    assert artifact["unsafe_mass_bounds"] == [0.0, 0.0, 0.0]
    assert artifact["empirical_violation_rates"] == [0.0, 0.0, 0.0]
    assert artifact["bound_is_sound"] is True
    assert artifact["mock_or_live_logprobs"] == "live_exp1468"
    assert artifact["model_used"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
    assert artifact["external_fit_verdict"] == "adopted_minimal_beaver_smoke_fit"
    assert artifact["broad_benchmark_deferred"] is True
    assert artifact["honest_verdict"] == "sound_bound_live_exp1468"


def test_req_verify_1470_falls_back_to_mock_logprobs_when_live_rows_are_incompatible(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1470-4: incompatible telemetry uses the existing mock path."""

    rows = [
        _live_row("case-1", "bad model row", "3") | {"hf_id": "not/a-mandated-model"},
    ]
    exp1468_artifact_path, exp1468_manifest_path = _write_exp1468_inputs(tmp_path, rows)
    out_path = tmp_path / "experiment_1470_beaver_lite_deterministic_bound_smoke.json"

    artifact = run(
        output_path=out_path,
        exp1468_artifact_path=exp1468_artifact_path,
        exp1468_manifest_path=exp1468_manifest_path,
        top_k=10,
    )

    assert artifact["status"] == "complete"
    assert artifact["questions_evaluated"] == [
        "Janet has 10 marbles and gives away 3. How many remain?",
        "A box has 4 red balls and 5 blue balls. How many balls are in the box?",
        "Luis read 12 pages on Monday and 8 on Tuesday. How many pages did he read?",
    ]
    assert artifact["mock_or_live_logprobs"] == "mock_logprobs"
    assert artifact["model_used"] is None
    assert artifact["unsafe_mass_bounds"] == pytest.approx([0.4, 0.4, 0.4])
    assert artifact["empirical_violation_rates"] == pytest.approx([0.4, 0.4, 0.4])
    assert artifact["bound_is_sound"] is True
    assert artifact["honest_verdict"] == "sound_bound_mock_logprobs"
    assert compatible_exp1468_rows(tmp_path / "missing.json", tmp_path / "missing.jsonl") == []

    incompatible_summary = tmp_path / "incompatible-summary.json"
    incompatible_manifest = tmp_path / "incompatible-manifest.jsonl"
    incompatible_summary.write_text(
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
    incompatible_manifest.write_text("", encoding="utf-8")
    assert compatible_exp1468_rows(incompatible_summary, incompatible_manifest) == []


def test_req_verify_1470_validation_rejects_unsound_or_mislabeled_artifacts() -> None:
    """REQ-VERIFY-1470-3/5: schema validation catches unsound smoke artifacts."""

    valid = build_artifact(
        questions=["q1", "q2", "q3"],
        constraints=[
            {"constraint_id": "c1", "prefix_closed": True},
            {"constraint_id": "c2", "prefix_closed": True},
            {"constraint_id": "c3", "prefix_closed": True},
        ],
        unsafe_mass_bounds=[0.0, 0.5, 1.0],
        empirical_violation_rates=[0.0, 0.5, 1.0],
        mock_or_live_logprobs="mock_logprobs",
        model_used=None,
    )
    validate_artifact(valid)

    with pytest.raises(ValueError, match="below empirical"):
        build_artifact(
            questions=["q1", "q2", "q3"],
            constraints=[
                {"constraint_id": "c1", "prefix_closed": True},
                {"constraint_id": "c2", "prefix_closed": True},
                {"constraint_id": "c3", "prefix_closed": True},
            ],
            unsafe_mass_bounds=[0.0, 0.1, 0.0],
            empirical_violation_rates=[0.0, 0.2, 0.0],
            mock_or_live_logprobs="mock_logprobs",
            model_used=None,
        )
    with pytest.raises(ValueError, match="missing required fields"):
        validate_artifact({key: value for key, value in valid.items() if key != "honest_verdict"})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_artifact(valid | {"unsafe_mass_bounds": [0.0, 0.5, 1.1]})
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_artifact(valid | {"empirical_violation_rates": [0.0, 0.5, 1.1]})
    with pytest.raises(ValueError, match="below empirical"):
        validate_artifact(valid | {"unsafe_mass_bounds": [0.0, 0.4, 0.9]})
    with pytest.raises(ValueError, match="mock_or_live_logprobs"):
        validate_artifact(valid | {"mock_or_live_logprobs": "unclear"})
    with pytest.raises(ValueError, match="exactly three"):
        validate_artifact(valid | {"questions_evaluated": ["q1"]})
    assert "unsloth/Qwen3.6-35B-A3B-GGUF" in MANDATED_SOTA_GGUF_MODELS
