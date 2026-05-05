"""Tests for Exp 1382 full-scale certificate to repair pipeline.

Spec: REQ-VERIFY-1382, SCENARIO-VERIFY-1382
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import fullscale_certificate_semantic_repair_100cases as mod


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


def _cases(n: int = 50) -> list[mod.FoVerPipelineCase]:
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
                source="unit",
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


def test_req1382_pipeline_metrics_and_headline_gate() -> None:
    """REQ-VERIFY-1382: full pipeline metrics are computed from live SOTA rows."""

    artifact = mod.build_fullscale_pipeline_artifact(
        cases=_cases(50),
        model_specs=[QWEN_SPEC, GEMMA_SPEC],
        dvi_checkpoint_path="/tmp/dvi_checkpoint_v1.pt",
        dvi_predictor=_dvi_predictor,
        generation_fn=_generation,
        run_date="20260505",
        project_root="/repo",
        checkpoint_path=None,
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["total_fover_cases"] == 50
    assert artifact["certificate_extract_count"] == 50
    assert artifact["certificate_parse_rate"] == pytest.approx(1.0)
    assert artifact["semantic_validation_pass_rate"] == pytest.approx(1.0)
    assert artifact["mcs_repair_localization_rate"] == pytest.approx(1.0)
    assert artifact["repair_hint_precision"] == pytest.approx(1.0)
    assert artifact["scheduler_accept_rate"] == pytest.approx(0.5)
    assert artifact["scheduler_false_acceptance_rate"] == pytest.approx(0.0)
    assert artifact["full_pipeline_pass_rate"] == pytest.approx(0.5)
    assert artifact["headline_result_allowed"] is True
    assert artifact["models_used"][0]["hf_id"] == QWEN_SPEC["hf_id"]
    assert artifact["honest_verdict"] == "fullscale_pipeline_headline_allowed_parse_rate_1_0"


def test_req1382_headline_gate_requires_live_mandated_sota_rows() -> None:
    """REQ-VERIFY-1382: synthetic or under-sized rows cannot pass headline gate."""

    def replay_generation(
        spec: dict[str, Any],
        case: mod.CertificateCase,
        prompts: mod.CranePrompts,
    ) -> mod.CraneGenerationResult:
        row = _generation(spec, case, prompts)
        return mod.CraneGenerationResult(
            model_hf_id=row.model_hf_id,
            case_id=row.case_id,
            reasoning_text=row.reasoning_text,
            reasoning_token_count=row.reasoning_token_count,
            certificate_prefix=row.certificate_prefix,
            certificate_body=row.certificate_body,
            generation_source="deterministic_replay",
            certificate_token_count=row.certificate_token_count,
        )

    artifact = mod.build_fullscale_pipeline_artifact(
        cases=_cases(50),
        model_specs=[QWEN_SPEC, GEMMA_SPEC],
        dvi_checkpoint_path="/tmp/dvi_checkpoint_v1.pt",
        dvi_predictor=_dvi_predictor,
        generation_fn=replay_generation,
        run_date="20260505",
        project_root="/repo",
        checkpoint_path=None,
    )

    assert artifact["certificate_parse_rate"] == pytest.approx(1.0)
    assert artifact["headline_result_allowed"] is False
    assert artifact["terminal_blocker"] == "no_live_mandated_sota_generation_rows"


def test_scenario1382_run_experiment_writes_progress_and_checkpoints(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1382: runner persists bootstrap, checkpoints, and final JSON."""

    fover_path = tmp_path / "fover.jsonl"
    rows = [
        {
            "question_id": f"q{i}",
            "step_text": "2 + 2 = 4" if i % 2 == 0 else "2 + 2 = 5",
            "label": "correct" if i % 2 == 0 else "incorrect",
            "source": "unit",
        }
        for i in range(60)
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

    output_path = tmp_path / "exp1382.json"
    ckpt_path = tmp_path / "exp1382_ckpt.json"
    writes: list[dict[str, Any]] = []

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260505",
        fover_path=fover_path,
        exp1381_path=exp1381_path,
        output_path=output_path,
        checkpoint_path=ckpt_path,
        cached_pair_fn=lambda **_kwargs: [QWEN_SPEC, GEMMA_SPEC],
        generation_fn=_generation,
        dvi_predictor=_dvi_predictor,
        write_observer=lambda _path, payload: writes.append(dict(payload)),
    )

    assert [payload["status"] for payload in writes] == ["in_progress", "complete"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    checkpoint = json.loads(ckpt_path.read_text(encoding="utf-8"))
    assert checkpoint["processed_cases"] in {50, 60}
    assert checkpoint["checkpoint_interval_cases"] == 25
    assert artifact["total_fover_cases"] == 60
    assert artifact["dvi_checkpoint_used"] == str(checkpoint_file)
