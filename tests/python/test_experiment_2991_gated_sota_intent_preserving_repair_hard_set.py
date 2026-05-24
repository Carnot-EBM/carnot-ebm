"""Tests for Exp 2991 gated SOTA intent-preserving hard-set repair.

Spec: REQ-CODE-2991, SCENARIO-CODE-2991.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import gated_sota_intent_preserving_repair_hard_set as exp
from carnot.eval import hard_code_stress_manifest as hard


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/code-verification/spec.md"
REQUIRED_FIELDS = {
    "repair_rerun_clean",
    "headline_result",
    "n_tasks",
    "model_specs",
    "headline_models_used",
    "pass_at_1_delta",
    "pass_at_k_delta",
    "schema_failure_rate_delta",
    "syntax_failure_rate_delta",
    "verifier_false_accept_delta",
    "trace_coverage",
    "transcript_paths",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_hard_sources(tmp_path: Path, *, n_items: int = 24) -> None:
    hard.write_artifact(
        hard.ExperimentConfig(
            repo_root=tmp_path,
            manifest_items=hard.default_items()[:n_items],
            started_at=10.0,
            clock=lambda: 11.0,
            tests_run=("focused-exp2990",),
        )
    )


def _ready_report() -> exp.PreconditionReport:
    return exp.PreconditionReport(
        checks=[
            {"resource": "exp2989_sota_preflight", "available": True},
            {"resource": "exp2990_hard_stress_artifact", "available": True},
            {"resource": "hard_manifest_integrity", "available": True},
            {"resource": "cuda_available", "available": True},
            {"resource": "headline_model_cache_available", "available": True},
        ],
        model_specs={
            "headline_models": list(exp.HEADLINE_MODEL_IDS),
            "smoke_only_models": list(exp.SMOKE_ONLY_MODEL_IDS),
            "runnable_headline_models": [
                {
                    "name": "Gemma4-26B-A4B-it",
                    "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                    "gpu": 0,
                    "model_path": "/models/gemma.gguf",
                    "cached": True,
                }
            ],
        },
        runnable_model_specs=[
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "gpu": 0,
                "model_path": "/models/gemma.gguf",
                "cached": True,
            }
        ],
    )


def _config(tmp_path: Path, *, n_tasks: int = 20) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        n_tasks=n_tasks,
        started_at=20.0,
        clock=lambda: 25.0,
        tests_run=("focused-exp2991",),
    )


def _reference_generator(
    item: dict[str, Any],
    prompt: str,
    seed: int,
    max_tokens: int,
    model_spec: dict[str, Any],
) -> exp.GenerationOutcome:
    assert item["baseline_verification"]["failing_test_ids"][0] in prompt
    assert item["expected_behavior"] in prompt
    assert "Do not hard-code the visible tests" in prompt
    return exp.GenerationOutcome(
        text=json.dumps(
            {
                "draft_intent": item["expected_behavior"],
                "final_patch": item["reference_solution"],
            }
        ),
        tokens_generated=64,
        duration_s=1.25,
        backend="fake-live-llama",
        backend_detail=str(model_spec["model_path"]),
    )


def _schema_syntax_bad_generator(
    _item: dict[str, Any],
    _prompt: str,
    _seed: int,
    _max_tokens: int,
    model_spec: dict[str, Any],
) -> exp.GenerationOutcome:
    return exp.GenerationOutcome(
        text='{"draft_intent": "broken", "final_patch": "def bad(:\\n"}',
        tokens_generated=9,
        duration_s=0.5,
        backend="fake-live-llama",
        backend_detail=str(model_spec["model_path"]),
    )


def test_req_code_2991_spec_anchor_exists() -> None:
    """REQ-CODE-2991: the hard-set repair rerun is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CODE-2991" in spec
    assert "SCENARIO-CODE-2991" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert 'inference_substrate="live_llm_inference"' in spec


def test_scenario_code_2991_clean_headline_run_writes_replayable_evidence(
    tmp_path: Path,
) -> None:
    """SCENARIO-CODE-2991: clean promotion requires live headline evidence and gates."""

    _write_hard_sources(tmp_path, n_items=24)
    artifact = exp.write_artifact(
        _config(tmp_path, n_tasks=20),
        generator=_reference_generator,
        precondition_probe=lambda _config: _ready_report(),
    )
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["repair_rerun_clean"] is True
    assert artifact["headline_result"] is True
    assert artifact["honest_verdict"] == "clean: hard-set intent-preserving repair gates passed"
    assert artifact["n_tasks"] == 20
    assert artifact["headline_models_used"] == ["unsloth/gemma-4-26B-A4B-it-GGUF"]
    assert artifact["model_specs"]["smoke_only_models"] == list(exp.SMOKE_ONLY_MODEL_IDS)
    assert artifact["baseline_pass_at_1"] == pytest.approx(0.0)
    assert artifact["repair_pass_at_1"] == pytest.approx(1.0)
    assert artifact["pass_at_1_delta"] == pytest.approx(1.0)
    assert artifact["pass_at_k_delta"] == pytest.approx(1.0)
    assert artifact["schema_failure_rate_delta"] == pytest.approx(0.0)
    assert artifact["syntax_failure_rate_delta"] == pytest.approx(0.0)
    assert artifact["verifier_false_accept_delta"] == pytest.approx(0.0)
    assert artifact["trace_coverage"] == pytest.approx(1.0)
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert len(artifact["transcript_paths"]) == 20
    assert all((tmp_path / path).is_file() for path in artifact["transcript_paths"])
    assert all((tmp_path / row["candidate_patch_path"]).is_file() for row in artifact["candidate_evaluations"])
    assert all((tmp_path / row["verifier_log_path"]).is_file() for row in artifact["candidate_evaluations"])
    assert all(row["runtime_trace_present"] for row in artifact["candidate_evaluations"])


def test_scenario_code_2991_flagged_when_schema_or_syntax_regresses(tmp_path: Path) -> None:
    """SCENARIO-CODE-2991: parser regressions block headline promotion."""

    _write_hard_sources(tmp_path, n_items=24)
    artifact = exp.build_artifact(
        _config(tmp_path, n_tasks=20),
        generator=_schema_syntax_bad_generator,
        precondition_probe=lambda _config: _ready_report(),
    )

    assert artifact["headline_result"] is True
    assert artifact["repair_rerun_clean"] is False
    assert artifact["honest_verdict"] == "flagged: hard-set repair did not clear promotion gates"
    assert artifact["pass_at_1_delta"] == pytest.approx(0.0)
    assert artifact["syntax_failure_rate_delta"] == pytest.approx(1.0)
    assert artifact["trace_coverage"] == pytest.approx(1.0)


def test_req_code_2991_default_preconditions_block_missing_sota_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-2991: Exp 2989 and cached headline GGUFs are hard preconditions."""

    _write_hard_sources(tmp_path, n_items=24)
    _write_json(
        tmp_path / "results" / exp.PREFLIGHT_FILENAME,
        {"sota_headline_ready": False, "model_specs": {}},
    )
    monkeypatch.setattr(exp, "cached_sota_pair", lambda **_kwargs: None)
    monkeypatch.setattr(exp, "resolve_cached_gguf", lambda _hf_id: None)

    artifact = exp.build_artifact(_config(tmp_path, n_tasks=20))

    assert artifact["honest_verdict"] == "blocked: preconditions not met"
    assert artifact["headline_result"] is False
    assert artifact["repair_rerun_clean"] is False
    assert artifact["n_tasks"] == 0
    assert any(
        row["resource"] == "exp2989_sota_preflight" and row["available"] is False
        for row in artifact["preconditions_checked"]
    )


def test_req_code_2991_blocks_when_hard_manifest_is_too_small(tmp_path: Path) -> None:
    """REQ-CODE-2991: headline repair cannot run below the minimum hard-set size."""

    _write_hard_sources(tmp_path, n_items=3)

    artifact = exp.build_artifact(
        _config(tmp_path, n_tasks=3),
        generator=_reference_generator,
        precondition_probe=lambda _config: _ready_report(),
    )

    assert artifact["honest_verdict"] == "blocked: preconditions not met"
    assert artifact["n_tasks"] == 0
    assert any(
        row["resource"] == "hard_manifest_minimum_task_count" and row["available"] is False
        for row in artifact["preconditions_checked"]
    )


def test_req_code_2991_blocks_when_hard_manifest_is_missing(tmp_path: Path) -> None:
    """REQ-CODE-2991: missing hard manifest emits a terminal blocked artifact."""

    artifact = exp.build_artifact(
        _config(tmp_path, n_tasks=20),
        generator=_reference_generator,
        precondition_probe=lambda _config: _ready_report(),
    )

    assert artifact["honest_verdict"] == "blocked: preconditions not met"
    assert any(
        row["resource"] == "hard_manifest_integrity" and row["available"] is False
        for row in artifact["preconditions_checked"]
    )


def test_req_code_2991_repair_output_parser_preserves_diagnostics() -> None:
    """REQ-CODE-2991: malformed model output is diagnosable, not auto-accepted."""

    parsed = exp.parse_repair_output(
        '{"draft_intent": "keep signature", "final_patch": "def f(x):\\n    return x\\n"}'
    )
    prefixed_json = exp.parse_repair_output(
        'Here is the patch: {"draft_intent": "x", "final_patch": "def f(x):\\n    return x\\n"}'
    )
    json_fenced = exp.parse_repair_output(
        '```json\n{"draft_intent": "x", "final_patch": "def f(x):\\n    return x\\n"}\n```'
    )
    missing_fields = exp.parse_repair_output('{"draft_intent": "", "final_patch": ""}')
    fenced = exp.parse_repair_output("```python\ndef f(x):\n    return x\n```")
    broken = exp.parse_repair_output("not json and no function")
    invalid_json = exp.parse_repair_output('{"draft_intent": ')

    assert parsed.schema_valid is True
    assert parsed.draft_intent == "keep signature"
    assert parsed.final_patch.startswith("def f")
    assert prefixed_json.schema_valid is True
    assert json_fenced.schema_valid is True
    assert missing_fields.schema_valid is False
    assert 'missing non-empty "draft_intent"' in missing_fields.schema_errors
    assert 'missing non-empty "final_patch"' in missing_fields.schema_errors
    assert fenced.schema_valid is False
    assert fenced.final_patch.startswith("def f")
    assert "no JSON object found" in fenced.schema_errors
    assert broken.final_patch == "not json and no function"
    assert invalid_json.schema_errors == ["invalid JSON object: Expecting value"]
    assert exp.syntax_diagnostics("") == (False, ["empty candidate"])
    assert exp.syntax_diagnostics("def bad(:") == (False, ["SyntaxError: invalid syntax"])


def test_req_code_2991_small_helpers_cover_cache_and_path_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-2991: cache helper fallbacks keep model and path evidence stable."""

    def no_arg_cached_pair() -> list[dict[str, Any]]:
        return [{"hf_id": exp.HEADLINE_MODEL_IDS[0], "model_path": "/models/qwen.gguf"}]

    monkeypatch.setattr(exp, "cached_sota_pair", no_arg_cached_pair)

    assert exp._call_cached_sota_pair() == [  # noqa: SLF001 - focused branch coverage.
        {"hf_id": exp.HEADLINE_MODEL_IDS[0], "model_path": "/models/qwen.gguf"}
    ]
    assert exp._model_name("unsloth/gemma-4-26B-A4B-it-GGUF") == "gemma-4-26B-A4B-it"  # noqa: SLF001
    assert exp._relative_or_absolute(tmp_path, tmp_path.parent / "outside.txt").is_absolute()  # noqa: SLF001
