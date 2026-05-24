"""Tests for Exp 3016 SOTA repair rerun with acceptance controller.

Spec: REQ-CODE-3016, SCENARIO-CODE-3016.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import hard_code_stress_manifest as hard
from carnot.eval import metamorphic_repair_oracle_audit as metamorphic
from carnot.eval import sota_repair_rerun_with_acceptance_controller as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/code-verification/spec.md"
HEADLINE_MODEL = "unsloth/gemma-4-26B-A4B-it-GGUF"
REQUIRED_FIELDS = {
    "repair_controller_clean",
    "headline_result",
    "preconditions_checked",
    "n_tasks",
    "n_metamorphic_variants",
    "model_specs",
    "headline_models_used",
    "model_checksums",
    "acceptance_controller_config_path",
    "pass_at_1_delta",
    "pass_at_k_delta",
    "false_accept_delta",
    "tautology_gate_clean",
    "syntax_failure_rate_delta",
    "schema_failure_rate_delta",
    "live_transcript_paths",
    "verifier_log_paths",
    "inference_substrate",
    "duration_s",
    "honest_verdict",
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_ready_sources(tmp_path: Path, *, n_items: int = 24, tautology: bool = True) -> None:
    hard.write_artifact(
        hard.ExperimentConfig(
            repo_root=tmp_path,
            manifest_items=hard.default_items()[:n_items],
            started_at=10.0,
            clock=lambda: 11.0,
            tests_run=("focused-exp2990",),
        )
    )
    metamorphic.write_artifact(
        metamorphic.ExperimentConfig(
            repo_root=tmp_path,
            started_at=12.0,
            clock=lambda: 13.0,
            tests_run=("focused-exp3002",),
        )
    )
    controller_path = tmp_path / exp.CONFIG_REL_PATH
    _write_json(
        controller_path,
        {
            "policy_type": "transparent_grid_rule",
            "selected_rule": {
                "require_schema_valid": True,
                "require_syntax_success": True,
                "require_entry_point_present": True,
                "require_false_accept_probe_clean": True,
                "require_no_intent_drift": True,
                "require_original_passed": True,
                "require_metamorphic_passed_all": True,
                "require_tautology_probe_clean": True,
            },
            "llm_judge_used": False,
        },
    )
    _write_json(
        tmp_path / "results" / exp.EXP3015_FILENAME,
        {
            "artifact": "experiment_3015_cactus_style_repair_acceptance_controller_v1",
            "acceptance_controller_ready": True,
            "controller_config_path": str(exp.CONFIG_REL_PATH),
            "honest_verdict": "complete: offline repair acceptance controller ready",
        },
    )
    _write_json(
        tmp_path / "results" / exp.EXP3013_FILENAME,
        {
            "artifact": "experiment_3013_sota_gguf_logprob_telemetry_preflight_v1",
            "sota_headline_ready": True,
            "sota_logprob_ready": True,
            "preconditions_checked": True,
            "model_checksums": {
                HEADLINE_MODEL: {
                    "status": "available",
                    "path": "/models/gemma.gguf",
                    "bounded_sha256": "checksum",
                }
            },
            "cache_paths": {"headline_models": {HEADLINE_MODEL: "/models/gemma.gguf"}},
            "headline_models_attempted": [
                {
                    "hf_id": HEADLINE_MODEL,
                    "transcript_path": "/tmp/exp3013-live.json",
                    "transcript_sha256": "live-sha",
                    "telemetry_observation": {"topk": True},
                    "token_logprobs_exposed": True,
                    "topk_logprobs_exposed": True,
                    "duration_s": 1.25,
                }
            ],
            "precondition_evidence": {
                "gpu_inventory": {"available": True},
                "torch_cuda": {"cuda_available": True, "cuda_device_count": 2},
                "llama_cpp": {
                    "llama_cpp_import_ok": True,
                    "llama_cpp_supports_gpu_offload": True,
                },
            },
        },
    )
    if not tautology:
        exp3002_path = tmp_path / "results" / exp.EXP3002_FILENAME
        payload = json.loads(exp3002_path.read_text(encoding="utf-8"))
        payload["tautology_probe_ready"] = False
        payload["rejected_variants"] = []
        _write_json(exp3002_path, payload)


def _config(tmp_path: Path, *, n_tasks: int = 20) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        n_tasks=n_tasks,
        max_headline_models=1,
        started_at=20.0,
        clock=lambda: 26.5,
        tests_run=("focused-exp3016",),
    )


def _reference_generator(
    item: dict[str, Any],
    _prompt: str,
    _seed: int,
    _max_tokens: int,
    _model_spec: dict[str, Any],
) -> exp.GenerationOutcome:
    return exp.GenerationOutcome(
        text=json.dumps(
            {
                "draft_intent": item["expected_behavior"],
                "final_patch": item["reference_solution"],
            }
        ),
        tokens_generated=32,
        duration_s=0.75,
        backend="fixture_llama_cpp",
        backend_detail="/models/gemma.gguf",
    )


def _overfit_first_generator(
    item: dict[str, Any],
    prompt: str,
    seed: int,
    max_tokens: int,
    model_spec: dict[str, Any],
) -> exp.GenerationOutcome:
    if item["item_id"] != "repair-hard-0001":
        return _reference_generator(item, prompt, seed, max_tokens, model_spec)
    return exp.GenerationOutcome(
        text=json.dumps(
            {
                "draft_intent": item["expected_behavior"],
                "final_patch": (
                    "def clamp_score(x, lo, hi):\n"
                    "    if (x, lo, hi) == (12, 0, 10):\n"
                    "        return 10\n"
                    "    if (x, lo, hi) == (-3, 0, 10):\n"
                    "        return 0\n"
                    "    if (x, lo, hi) == (5, 0, 10):\n"
                    "        return 5\n"
                    "    return None\n"
                ),
            }
        ),
        tokens_generated=48,
        duration_s=0.8,
        backend="fixture_llama_cpp",
        backend_detail="/models/gemma.gguf",
    )


def _bad_schema_generator(
    _item: dict[str, Any],
    _prompt: str,
    _seed: int,
    _max_tokens: int,
    _model_spec: dict[str, Any],
) -> exp.GenerationOutcome:
    return exp.GenerationOutcome(
        text="```python\ndef broken(:\n    pass\n```",
        tokens_generated=8,
        duration_s=0.1,
        backend="fixture_llama_cpp",
        backend_detail="/models/gemma.gguf",
    )


def test_req_code_3016_spec_anchor_exists() -> None:
    """REQ-CODE-3016: the controller rerun is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-CODE-3016" in spec
    assert "SCENARIO-CODE-3016" in spec
    assert exp.ARTIFACT_FILENAME in spec
    assert "acceptance_controller_config_path" in spec


def test_scenario_code_3016_clean_headline_run_writes_required_evidence(tmp_path: Path) -> None:
    """SCENARIO-CODE-3016: accepted headline candidates produce replayable evidence."""

    _write_ready_sources(tmp_path)

    artifact = exp.write_artifact(_config(tmp_path), generator=_reference_generator)
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert REQUIRED_FIELDS <= set(artifact)
    assert artifact["repair_controller_clean"] is True
    assert artifact["headline_result"] is True
    assert artifact["preconditions_checked"] is True
    assert artifact["n_tasks"] == 20
    assert artifact["n_metamorphic_variants"] > 20
    assert artifact["headline_models_used"] == [HEADLINE_MODEL]
    assert artifact["model_checksums"][HEADLINE_MODEL]["bounded_sha256"] == "checksum"
    assert artifact["acceptance_controller_config_path"] == str(exp.CONFIG_REL_PATH)
    assert artifact["pass_at_1_delta"] == pytest.approx(1.0)
    assert artifact["pass_at_k_delta"] == pytest.approx(1.0)
    assert artifact["false_accept_delta"] == pytest.approx(0.0)
    assert artifact["syntax_failure_rate_delta"] == pytest.approx(0.0)
    assert artifact["schema_failure_rate_delta"] == pytest.approx(0.0)
    assert artifact["tautology_gate_clean"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == 6.5
    assert len(artifact["live_transcript_paths"]) == 20
    assert len(artifact["verifier_log_paths"]) == 20
    assert all((tmp_path / path).is_file() for path in artifact["candidate_patch_paths"])
    assert all((tmp_path / path).is_file() for path in artifact["live_transcript_paths"])
    assert all(row["controller_accepted"] is True for row in artifact["candidate_evaluations"])
    assert all(row["deterministic_tests_executed"] is True for row in artifact["candidate_evaluations"])
    assert artifact["accept_all_metrics"]["pass_at_1"] == pytest.approx(1.0)


def test_scenario_code_3016_controller_rejects_false_accept_before_promotion(
    tmp_path: Path,
) -> None:
    """SCENARIO-CODE-3016: controller rejection runs deterministic tests and removes overfit."""

    _write_ready_sources(tmp_path)

    artifact = exp.build_artifact(_config(tmp_path), generator=_overfit_first_generator)
    rejected = next(row for row in artifact["candidate_evaluations"] if row["item_id"] == "repair-hard-0001")

    assert artifact["repair_controller_clean"] is True
    assert artifact["headline_result"] is True
    assert artifact["pass_at_1_delta"] == pytest.approx(0.95)
    assert artifact["accept_all_metrics"]["false_accept_rate"] > 0.0
    assert artifact["repair_metrics"]["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["accept_all_comparison"]["false_accept_delta_vs_accept_all"] < 0.0
    assert rejected["original_passed"] is True
    assert rejected["metamorphic_passed_all"] is False
    assert rejected["false_accept"] is True
    assert rejected["controller_accepted"] is False
    assert "metamorphic_passed_all" in rejected["controller_rejection_reasons"]
    assert "false_accept" in rejected["controller_rejection_reasons"]
    assert rejected["deterministic_tests_executed"] is True


def test_req_code_3016_schema_syntax_and_tautology_gates_flag_run(tmp_path: Path) -> None:
    """REQ-CODE-3016: unsafe syntax/schema and tautology evidence cannot promote."""

    _write_ready_sources(tmp_path, tautology=False)

    artifact = exp.build_artifact(_config(tmp_path), generator=_bad_schema_generator)

    assert artifact["repair_controller_clean"] is False
    assert artifact["headline_result"] is True
    assert artifact["tautology_gate_clean"] is False
    assert artifact["repair_metrics"]["candidate_count"] == 0
    assert artifact["accept_all_metrics"]["syntax_failure_rate"] == pytest.approx(1.0)
    assert artifact["accept_all_metrics"]["schema_failure_rate"] == pytest.approx(1.0)
    assert artifact["honest_verdict"].startswith("complete_flagged:")


def test_req_code_3016_blocks_when_preconditions_are_missing(tmp_path: Path) -> None:
    """REQ-CODE-3016: missing telemetry/controller gates emit a terminal blocked artifact."""

    _write_ready_sources(tmp_path)
    _write_json(
        tmp_path / "results" / exp.EXP3013_FILENAME,
        {
            "sota_headline_ready": True,
            "sota_logprob_ready": False,
            "preconditions_checked": True,
            "model_checksums": {},
            "precondition_evidence": {"gpu_inventory": {"available": True}},
        },
    )

    artifact = exp.build_artifact(_config(tmp_path), generator=_reference_generator)

    assert artifact["preconditions_checked"] is True
    assert artifact["headline_result"] is False
    assert artifact["repair_controller_clean"] is False
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["n_tasks"] == 0
    assert artifact["n_metamorphic_variants"] == 0
    assert any(
        row["resource"] == "exp3013_sota_logprob_telemetry"
        and row["available"] is False
        for row in artifact["precondition_checks"]
    )


def test_req_code_3016_sample_and_helper_branches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-3016: sample, path, parser, and controller helper branches are deterministic."""

    _write_ready_sources(tmp_path)

    too_small = exp.build_artifact(_config(tmp_path, n_tasks=3), generator=_reference_generator)
    assert too_small["honest_verdict"].startswith("blocked:")
    assert any(row["resource"] == "exp3016_sample_size" for row in too_small["precondition_checks"])

    rule = {"require_schema_valid": True, "require_tautology_probe_clean": True}
    assert exp._controller_rejection_reasons(
        {"schema_valid": False, "tautology_probe_clean": False}, rule
    ) == ["schema_valid", "tautology_probe_clean"]
    assert exp._entry_point_present("def f():\n    return 1\n", "f") is True
    assert exp._entry_point_present("def bad(:\n", "f") is False
    assert exp._relative_or_absolute(tmp_path, tmp_path.parent / "outside.txt").is_absolute()
    assert exp._resolve_repo_path(tmp_path, tmp_path / "absolute.json").is_absolute()
    explicit_controller = tmp_path / "controller.json"
    explicit_meta = tmp_path / "metamorphic.jsonl"
    explicit_config = exp.ExperimentConfig(
        repo_root=tmp_path,
        controller_config_path=explicit_controller,
        metamorphic_manifest_path=explicit_meta,
    )
    assert explicit_config.resolved_controller_config_path({}) == explicit_controller
    assert explicit_config.resolved_metamorphic_manifest_path({}) == explicit_meta
    assert exp._validate_hard_manifest(exp.ExperimentConfig(repo_root=tmp_path / "missing"))[
        "checks"
    ][0]["available"] is False
    assert exp._validate_metamorphic_manifest(
        exp.ExperimentConfig(repo_root=tmp_path, metamorphic_manifest_path=tmp_path / "missing.jsonl"),
        {"metamorphic_oracle_ready": True},
    )["checks"][0]["available"] is False
    monkeypatch.setattr(
        exp,
        "cached_sota_pair",
        lambda **_kwargs: [{"hf_id": HEADLINE_MODEL, "model_path": "/cache/gemma.gguf"}],
    )
    assert exp._runnable_model_specs({})[0]["model_path"] == "/cache/gemma.gguf"
    monkeypatch.setattr(
        exp,
        "cached_sota_pair",
        lambda **_kwargs: [{"hf_id": "not-a-headline-model", "model_path": "/cache/nope.gguf"}],
    )
    monkeypatch.setattr(exp, "resolve_cached_gguf", lambda *_args, **_kwargs: None)
    assert exp._runnable_model_specs({}) == []
    monkeypatch.setattr(
        exp,
        "cached_sota_pair",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert exp._call_cached_sota_pair() is None
    assert exp._telemetry_by_model({"headline_models_attempted": [{}]}) == {}
    assert exp._unique_paths(["a", "a", ""]) == ["a"]
    parsed = exp._parse_args(["--n-tasks", "21", "--max-tokens", "64", "--test-run", "focused"])
    assert parsed.n_tasks == 21
    assert parsed.max_tokens == 64
    assert parsed.test_run == ["focused"]
