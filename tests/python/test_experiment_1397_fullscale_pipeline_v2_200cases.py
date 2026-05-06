"""Tests for Exp 1397 full-scale pipeline v2 at 200 cases.

Spec: REQ-VERIFY-1397, SCENARIO-VERIFY-1397
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import fullscale_pipeline_v2_200cases as mod


QWEN_SPEC = {
    "name": "Qwen3.6-35B-A3B",
    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "gpu": 0,
    "model_path": "/cache/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
}
GEMMA_SPEC = {
    "name": "Gemma4-31B-it",
    "hf_id": "unsloth/gemma-4-31B-it-GGUF",
    "gpu": 1,
    "model_path": "/cache/gemma-4-31B-it-Q4_K_M.gguf",
}


def _cases(n: int = 200) -> list[mod.FoVerPipelineCase]:
    cases: list[mod.FoVerPipelineCase] = []
    for index in range(n):
        incorrect = index % 2 == 1
        cases.append(
            mod.FoVerPipelineCase(
                case_id=f"case_{index}",
                question="",
                response=(
                    "2 + 2 = 4, therefore the arithmetic step is supported."
                    if not incorrect
                    else "2 + 2 = 5, therefore the arithmetic step needs repair."
                ),
                label=1 if incorrect else 0,
                source="fover_v4",
            )
        )
    return cases


def _generation(
    spec: dict[str, Any],
    case: mod.CertificateCase,
    prompts: mod.CranePrompts,
) -> mod.CraneGenerationResult:
    del prompts
    return mod.CraneGenerationResult(
        model_hf_id=spec["hf_id"],
        case_id=case.case_id,
        reasoning_text="bounded CRANE reasoning",
        reasoning_token_count=7,
        certificate_prefix=mod.structural_tag(case.expected_state) + "\n",
        certificate_body=mod.certificate_body_for_state(case.expected_state),
        generation_source="live_sota_llamacpp",
        certificate_token_count=8,
    )


def _dvi_predictor(case: mod.FoVerPipelineCase) -> float:
    return 0.91 if case.label == 1 else 0.11


def test_req1397_pipeline_metrics_improvements_and_headline_gate() -> None:
    """REQ-VERIFY-1397: 200-case v2 metrics are adapted from the live pipeline."""

    artifact = mod.build_fullscale_pipeline_v2_artifact(
        cases=_cases(200),
        model_specs=[QWEN_SPEC, GEMMA_SPEC],
        dvi_checkpoint_path="/tmp/dvi_checkpoint_v1.pt",
        exp1396_artifact={"semantic_validation_improvement_measured": True},
        dvi_predictor=_dvi_predictor,
        generation_fn=_generation,
        run_date="20260506",
        project_root="/repo",
        checkpoint_path=None,
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["cases_evaluated"] == 200
    assert artifact["certificate_extract_count"] == 200
    assert artifact["certificate_parse_rate"] == pytest.approx(1.0)
    assert artifact["semantic_validation_pass_rate"] == pytest.approx(1.0)
    assert artifact["full_pipeline_pass_rate"] == pytest.approx(0.5)
    assert artifact["semantic_validation_improvement_vs_exp1382"] == pytest.approx(0.41)
    assert artifact["full_pipeline_improvement_vs_exp1382"] == pytest.approx(0.21)
    assert artifact["headline_result_allowed"] is True
    assert artifact["MODEL_SPECS"][0]["hf_id"] == QWEN_SPEC["hf_id"]
    assert artifact["models_used"][0]["hf_id"] == QWEN_SPEC["hf_id"]


def test_req1397_headline_gate_requires_metric_thresholds() -> None:
    """REQ-VERIFY-1397: provenance alone cannot allow a below-threshold result."""

    artifact = mod.finalize_exp1397_artifact(
        {
            "status": "complete",
            "total_fover_cases": 200,
            "certificate_extract_count": 200,
            "certificate_parse_rate": 1.0,
            "semantic_validation_pass_rate": 0.69,
            "full_pipeline_pass_rate": 0.5,
            "models_used": [{"hf_id": QWEN_SPEC["hf_id"]}],
            "headline_gate_evidence": {
                "headline_result_allowed": True,
                "mandated_live_generation_case_count": 200,
            },
        },
        model_specs=[QWEN_SPEC, GEMMA_SPEC],
        exp1396_artifact={"semantic_validation_improvement_measured": True},
        run_date="20260506",
        project_root="/repo",
    )

    assert artifact["headline_result_allowed"] is False
    assert artifact["semantic_validation_improvement_vs_exp1382"] == pytest.approx(0.1)
    assert artifact["honest_verdict"] == "not_headline_semantic_validation_below_0_70"


def test_scenario1397_run_experiment_writes_progress_and_terminal_json(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1397: runner gates on exp1396 and writes the final JSON."""

    fover_path = tmp_path / "fover.jsonl"
    rows = [
        {
            "question_id": f"q{i}",
            "step_text": "2 + 2 = 4" if i % 2 == 0 else "2 + 2 = 5",
            "label": "correct" if i % 2 == 0 else "incorrect",
            "source": "fover_v4",
        }
        for i in range(220)
    ]
    fover_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    checkpoint_file = tmp_path / "dvi_checkpoint_v1.pt"
    checkpoint_file.write_text("fake checkpoint", encoding="utf-8")
    exp1381_path = tmp_path / "exp1381.json"
    exp1381_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "dvi_deployed": True,
                "dvi_checkpoint_path": str(checkpoint_file),
            }
        ),
        encoding="utf-8",
    )
    exp1396_path = tmp_path / "exp1396.json"
    exp1396_path.write_text(
        json.dumps({"status": "complete", "semantic_validation_improvement_measured": True}),
        encoding="utf-8",
    )

    output_path = tmp_path / "exp1397.json"
    writes: list[dict[str, Any]] = []
    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260506",
        fover_path=fover_path,
        exp1381_path=exp1381_path,
        exp1396_path=exp1396_path,
        output_path=output_path,
        checkpoint_path=None,
        cached_pair_fn=lambda **_kwargs: [QWEN_SPEC, GEMMA_SPEC],
        generation_fn=_generation,
        dvi_predictor=_dvi_predictor,
        write_observer=lambda _path, payload: writes.append(dict(payload)),
    )

    assert [payload["status"] for payload in writes] == ["in_progress", "complete"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["cases_evaluated"] == 200
    assert artifact["exp1396_fix_confirmed"] is True
