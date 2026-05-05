"""Tests for Exp 1353 triggered certificate v7 SOTA terminal run.

Spec: REQ-VERIFY-1353, SCENARIO-VERIFY-1353
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import triggered_certificate_v7_truncproof_sota as mod


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


def _exp1324() -> dict[str, Any]:
    return {
        "status": "complete",
        "minimum_parseable_attempts_to_recover": 6,
        "formalizer_failure_modes": [{"class": "unknown_state_mishandling", "count": 4}],
        "source_metrics": {"exp1312_certificate_parse_rate": 0.71223},
    }


def _exp1339(*, ready: bool = True) -> dict[str, Any]:
    return {
        "status": "complete",
        "dynamic_grammar_ready": ready,
        "certificate_states_supported": ["REPAIR_HINT", "SAT", "UNKNOWN", "UNSAT"],
        "unknown_state_supported": ready,
        "state_transition_error_count": 0 if ready else 1,
    }


def _exp1351() -> dict[str, Any]:
    return {
        "status": "complete",
        "honest_verdict": "handoff_state_missing_exp1340_terminal_certificate_semantic_scheduler_dvi_grpo_closed",
    }


def _exp1352(*, allowed: bool = True, max_tokens: int = 96) -> dict[str, Any]:
    return {
        "status": "complete",
        "sota_run_allowed": allowed,
        "blocker_if_not_allowed": None if allowed else "max_token_budget_insufficient",
        "runtime_settings_used": {"max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0},
        "min_completion_tokens_by_state": {"SAT": 6, "UNSAT": 6, "UNKNOWN": 6, "REPAIR_HINT": 10},
        "max_token_budget_sufficient": allowed,
        "dynamic_dispatch_preserved": allowed,
        "structural_tag_supported": allowed,
    }


def _source_artifacts(*, allowed: bool = True) -> dict[str, dict[str, Any]]:
    return {
        "exp1324": _exp1324(),
        "exp1339": _exp1339(),
        "exp1351": _exp1351(),
        "exp1352": _exp1352(allowed=allowed),
    }


def _perfect_generation(spec: dict[str, Any], case: mod.CertificateCase, prompt: str) -> mod.GenerationResult:
    del prompt
    return mod.GenerationResult(
        model_hf_id=spec["hf_id"],
        case_id=case.case_id,
        text=f"{mod.structural_tag(case.expected_state)}\n{mod.json_certificate_text(case.expected_state)}",
        generation_source="live_sota_llamacpp",
        token_count=18,
    )


def test_req1353_headline_sota_rows_compute_certificate_metrics() -> None:
    """REQ-VERIFY-1353-3/4/5/6/7: SOTA rows produce headline metrics."""

    artifact = mod.build_experiment_artifact(
        source_artifacts=_source_artifacts(),
        model_specs=[QWEN_SPEC, GEMMA_SPEC],
        gpu_health=mod.GPUHealth(True, 2, []),
        generation_fn=_perfect_generation,
        run_date="20260505",
        project_root="/repo",
        max_models=1,
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["terminal_blocker"] is None
    assert artifact["headline_result_allowed"] is True
    assert artifact["certificate_case_count"] == 4
    assert artifact["trigger_token_hit_rate"] == pytest.approx(1.0)
    assert artifact["certificate_parse_rate"] == pytest.approx(1.0)
    assert artifact["certificate_truthfulness_rate"] == pytest.approx(1.0)
    assert artifact["unknown_preservation_rate"] == pytest.approx(1.0)
    assert artifact["parse_rate_delta_over_exp1312"] == pytest.approx(0.28777)
    assert artifact["min_completion_budget_respected"] is True
    assert artifact["completion_preflight_used"]["sota_run_allowed"] is True
    assert artifact["models_used"][0]["hf_id"] == QWEN_SPEC["hf_id"]
    assert artifact["models_used"][0]["gpu"] == 0
    assert artifact["models_used"][0]["model_path"] == QWEN_SPEC["model_path"]
    assert artifact["honest_verdict"] == "sota_triggered_certificate_v7_measured"


def test_req1353_cpu_smoke_terminal_blocker_when_cached_pair_missing() -> None:
    """REQ-VERIFY-1353-3/7: missing cached SOTA pair allows only non-headline smoke."""

    artifact = mod.build_experiment_artifact(
        source_artifacts=_source_artifacts(),
        model_specs=None,
        gpu_health=mod.GPUHealth(True, 2, []),
        generation_fn=lambda *_args, **_kwargs: pytest.fail("SOTA generation must not run"),
        run_date="20260505",
        project_root="/repo",
    )

    assert artifact["status"] == "complete"
    assert artifact["terminal_blocker"] == "cached_sota_pair_unavailable"
    assert artifact["headline_result_allowed"] is False
    assert artifact["models_used"][0]["headline_result_allowed"] is False
    assert artifact["models_used"][0]["generation_source"] == "legacy_cpu_smoke"
    assert artifact["certificate_case_count"] == 4
    assert artifact["certificate_parse_rate"] == pytest.approx(1.0)
    assert artifact["honest_verdict"] == "blocked_cached_sota_pair_unavailable_cpu_smoke_complete"

    legacy = mod.build_experiment_artifact(
        source_artifacts=_source_artifacts(),
        model_specs=[{"name": "tiny", "hf_id": "legacy/small", "gpu": "cpu"}],
        gpu_health=mod.GPUHealth(True, 2, []),
        run_date="20260505",
        project_root="/repo",
    )
    no_path = mod.build_experiment_artifact(
        source_artifacts=_source_artifacts(),
        model_specs=[{"name": "Qwen", "hf_id": QWEN_SPEC["hf_id"], "gpu": 0}],
        gpu_health=mod.GPUHealth(True, 2, []),
        run_date="20260505",
        project_root="/repo",
    )

    assert legacy["terminal_blocker"] == "cached_sota_pair_unavailable"
    assert no_path["terminal_blocker"] == "cached_sota_pair_unavailable"


def test_req1353_completion_preflight_blocks_before_generation() -> None:
    """REQ-VERIFY-1353-5/6: insufficient completion budget is terminal."""

    artifact = mod.build_experiment_artifact(
        source_artifacts=_source_artifacts(allowed=False),
        model_specs=[QWEN_SPEC, GEMMA_SPEC],
        gpu_health=mod.GPUHealth(True, 2, []),
        generation_fn=lambda *_args, **_kwargs: pytest.fail("preflight block must stop generation"),
        run_date="20260505",
        project_root="/repo",
    )

    assert artifact["status"] == "complete"
    assert artifact["terminal_blocker"] == "completion_preflight_blocked:max_token_budget_insufficient"
    assert artifact["headline_result_allowed"] is False
    assert artifact["certificate_case_count"] == 4
    assert artifact["trigger_token_hit_rate"] == pytest.approx(0.0)
    assert artifact["certificate_parse_rate"] == pytest.approx(0.0)
    assert artifact["min_completion_budget_respected"] is False
    assert artifact["honest_verdict"] == "blocked_completion_preflight_cpu_smoke_not_run"


def test_req1353_terminal_blocker_branches_stay_explicit() -> None:
    """REQ-VERIFY-1353-2/6/7: dynamic grammar, GPU, and generation blockers are terminal."""

    dynamic = mod.build_experiment_artifact(
        source_artifacts=_source_artifacts() | {"exp1339": _exp1339(ready=False)},
        model_specs=[QWEN_SPEC, GEMMA_SPEC],
        gpu_health=mod.GPUHealth(True, 2, []),
        generation_fn=lambda *_args, **_kwargs: pytest.fail("grammar block must stop generation"),
        run_date="20260505",
        project_root="/repo",
    )
    missing_state = mod.build_experiment_artifact(
        source_artifacts=_source_artifacts()
        | {"exp1339": _exp1339() | {"certificate_states_supported": ["SAT", "UNSAT", "UNKNOWN"]}},
        model_specs=[QWEN_SPEC, GEMMA_SPEC],
        gpu_health=mod.GPUHealth(True, 2, []),
        generation_fn=lambda *_args, **_kwargs: pytest.fail("grammar block must stop generation"),
        run_date="20260505",
        project_root="/repo",
    )
    gpu = mod.build_experiment_artifact(
        source_artifacts=_source_artifacts(),
        model_specs=[QWEN_SPEC, GEMMA_SPEC],
        gpu_health=mod.GPUHealth(False, 2, ["busy"]),
        generation_fn=lambda *_args, **_kwargs: pytest.fail("GPU block must stop generation"),
        run_date="20260505",
        project_root="/repo",
    )

    def raises(_spec: dict[str, Any], _case: mod.CertificateCase, _prompt: str) -> mod.GenerationResult:
        raise RuntimeError("llama load failed")

    failed_generation = mod.build_experiment_artifact(
        source_artifacts=_source_artifacts(),
        model_specs=[QWEN_SPEC, GEMMA_SPEC],
        gpu_health=mod.GPUHealth(True, 2, []),
        generation_fn=raises,
        run_date="20260505",
        project_root="/repo",
    )
    non_headline_rows = mod.build_experiment_artifact(
        source_artifacts=_source_artifacts(),
        model_specs=[QWEN_SPEC, GEMMA_SPEC],
        gpu_health=mod.GPUHealth(True, 2, []),
        generation_fn=lambda spec, case, prompt: mod.GenerationResult(
            model_hf_id=spec["hf_id"],
            case_id=case.case_id,
            text=f"missing tag {prompt}",
            generation_source="synthetic_not_headline",
            token_count=2,
        ),
        run_date="20260505",
        project_root="/repo",
    )

    assert dynamic["terminal_blocker"] == "dynamic_grammar_not_ready"
    assert dynamic["honest_verdict"] == "blocked_dynamic_grammar_not_ready"
    assert missing_state["terminal_blocker"] == "dynamic_grammar_missing_required_state"
    assert gpu["terminal_blocker"] == "gpu_health_failed"
    assert gpu["models_used"][-1]["generation_source"] == "legacy_cpu_smoke"
    assert gpu["honest_verdict"] == "blocked_gpu_health_failed_cpu_smoke_complete"
    assert failed_generation["terminal_blocker"].startswith("sota_generation_failed:RuntimeError")
    assert failed_generation["honest_verdict"] == "blocked_sota_generation_failed_cpu_smoke_complete"
    assert non_headline_rows["terminal_blocker"] == "no_mandated_sota_generation_rows"
    assert non_headline_rows["trigger_token_hit_rate"] == pytest.approx(0.0)


def test_req1353_live_generator_adapter_and_helpers_are_testable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1353-3/4: live adapter records llama.cpp output deterministically."""

    class FakeLlama:
        calls: list[dict[str, Any]] = []

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            self.calls.append({"prompt": prompt, "kwargs": kwargs, "init": self.kwargs})
            return {
                "choices": [{"text": f"{mod.structural_tag('SAT')}\n{mod.json_certificate_text('SAT')}"}],
                "usage": {"completion_tokens": 7},
            }

    generator = mod.LlamaCppCertificateGenerator(
        {"max_tokens": 9, "temperature": 0.0, "top_p": 1.0, "stop": ["</s>"], "n_ctx": 128},
        llama_importer=lambda: FakeLlama,
    )
    case = mod.bounded_certificate_suite()[0]
    first = generator(QWEN_SPEC, case, "prompt")
    second = generator(QWEN_SPEC, case, "prompt")

    assert first.text.startswith("<CARNOT_CERT_STATE:SAT>")
    assert second.token_count == 7
    assert len(FakeLlama.calls) == 2
    assert FakeLlama.calls[0]["init"]["model_path"] == QWEN_SPEC["model_path"]
    assert FakeLlama.calls[0]["kwargs"]["max_tokens"] == 9
    with pytest.raises(RuntimeError, match="model_path missing"):
        generator({"hf_id": QWEN_SPEC["hf_id"]}, case, "prompt")

    assert mod._response_text({"choices": [{"text": "ok"}]}) == "ok"
    assert mod._response_text("raw") == "raw"
    assert mod._completion_token_count({"usage": {"completion_tokens": 3}}, "ignored") == 3
    assert mod._completion_token_count({}, "A+B") == 3
    assert mod._quantization_from_path("model-Q8_0.gguf") == "Q8_0"
    assert mod._quantization_from_path("model.gguf") is None
    assert mod._normalised_state("UNSATISFIABLE") == "UNSAT"
    assert mod._normalised_state("SATISFIABLE") == "SAT"
    assert mod._normalised_state("ABSTAIN") == "ABSTAIN"
    assert mod._normalised_state("MAYBE") == "MAYBE"
    assert mod._truthful("REPAIR_HINT", {"final_answer": "ABSTAIN"}) is True
    assert mod._rate(1, 0) == 0.0

    monkeypatch.setattr(mod, "_CUDA_LIB_ROOT", tmp_path / "missing")
    mod._add_venv_cuda_libs_to_ld_path()
    lib_root = tmp_path / "nvidia"
    (lib_root / "cublas/lib").mkdir(parents=True)
    monkeypatch.setattr(mod, "_CUDA_LIB_ROOT", lib_root)
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path / "old"))
    mod._add_venv_cuda_libs_to_ld_path()
    assert str((lib_root / "cublas/lib").resolve()) in (mod.os.environ["LD_LIBRARY_PATH"])

    no_min_tokens = mod.build_experiment_artifact(
        source_artifacts=_source_artifacts() | {"exp1352": {"sota_run_allowed": False}},
        model_specs=[QWEN_SPEC, GEMMA_SPEC],
        gpu_health=mod.GPUHealth(True, 2, []),
        run_date="20260505",
        project_root="/repo",
    )
    no_baseline = mod.build_experiment_artifact(
        source_artifacts=_source_artifacts() | {"exp1324": {"status": "complete"}},
        model_specs=None,
        gpu_health=mod.GPUHealth(True, 2, []),
        run_date="20260505",
        project_root="/repo",
    )
    assert no_min_tokens["completion_preflight_used"]["required_min_completion_tokens"] == 10
    assert no_baseline["parse_rate_delta_over_exp1312"] == pytest.approx(0.28777)


def test_req1353_gpu_health_probe_and_cached_pair_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-1353-3/7: local environment probes report exact blockers."""

    class Result:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *_args, **_kwargs: Result(0, "0, RTX 3090, 24576, 0, 0\n1, RTX 3090, 24576, 0, 0\n"),
    )
    assert mod.check_gpu_health() == mod.GPUHealth(True, 2, [])
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *_args, **_kwargs: Result(0, "0, RTX 3090, 24576, 101, 0\n1, RTX 3090, 24576, bad, 0\n"),
    )
    unhealthy = mod.check_gpu_health()
    assert "gpu0_vram_used_101mb" in unhealthy.issues
    assert any(issue.startswith("gpu_vram_used_unparseable") for issue in unhealthy.issues)
    monkeypatch.setattr(mod.subprocess, "run", lambda *_args, **_kwargs: Result(0, "0, RTX 3090\n"))
    assert mod.check_gpu_health().issues == ["fewer_than_two_gpus_visible"]
    monkeypatch.setattr(mod.subprocess, "run", lambda *_args, **_kwargs: Result(9, "", "bad driver"))
    assert mod.check_gpu_health().issues[0].startswith("nvidia_smi_exit_9")

    def raise_oserror(*_args: Any, **_kwargs: Any) -> Result:
        raise OSError("missing nvidia-smi")

    monkeypatch.setattr(mod.subprocess, "run", raise_oserror)
    assert mod.check_gpu_health().issues[0].startswith("nvidia_smi_error")
    monkeypatch.setattr(mod, "_load_cached_sota_pair", lambda **_kwargs: [QWEN_SPEC, GEMMA_SPEC])

    results = tmp_path / "results"
    results.mkdir()
    paths = {
        "exp1324_path": results / "exp1324.json",
        "exp1339_path": results / "exp1339.json",
        "exp1351_path": results / "exp1351.json",
        "exp1352_path": results / "exp1352.json",
    }
    paths["exp1324_path"].write_text(json.dumps(_exp1324()), encoding="utf-8")
    paths["exp1339_path"].write_text(json.dumps(_exp1339()), encoding="utf-8")
    paths["exp1351_path"].write_text(json.dumps(_exp1351()), encoding="utf-8")
    paths["exp1352_path"].write_text(json.dumps(_exp1352()), encoding="utf-8")
    output_path = results / "exp1353.json"

    artifact = mod.run_experiment(
        output_path=output_path,
        run_date="20260505",
        project_root=tmp_path,
        cached_pair_fn=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("cache broken")),
        gpu_health_fn=lambda: mod.GPUHealth(True, 2, []),
        **paths,
    )

    assert artifact["terminal_blocker"] == "cached_sota_pair_unavailable"
    assert artifact["headline_result_allowed"] is False
    assert mod._load_json(output_path) == artifact

    default_cached = mod.run_experiment(
        output_path=output_path,
        run_date="20260505",
        project_root=tmp_path,
        gpu_health_fn=lambda: mod.GPUHealth(True, 2, []),
        generation_fn=_perfect_generation,
        max_models=1,
        **paths,
    )
    assert default_cached["headline_result_allowed"] is True


def test_scenario1353_run_experiment_writes_in_progress_then_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-1353: runner persists bootstrap and terminal artifact."""

    results = tmp_path / "results"
    results.mkdir()
    paths = {
        "exp1324_path": results / "exp1324.json",
        "exp1339_path": results / "exp1339.json",
        "exp1351_path": results / "exp1351.json",
        "exp1352_path": results / "exp1352.json",
    }
    paths["exp1324_path"].write_text(json.dumps(_exp1324()), encoding="utf-8")
    paths["exp1339_path"].write_text(json.dumps(_exp1339()), encoding="utf-8")
    paths["exp1351_path"].write_text(json.dumps(_exp1351()), encoding="utf-8")
    paths["exp1352_path"].write_text(json.dumps(_exp1352()), encoding="utf-8")
    output_path = results / "exp1353.json"
    writes: list[dict[str, Any]] = []
    real_write = mod._write_json

    def recording_write(path: Path, payload: dict[str, Any]) -> None:
        writes.append(payload)
        real_write(path, payload)

    monkeypatch.setattr(mod, "_write_json", recording_write)

    artifact = mod.run_experiment(
        output_path=output_path,
        run_date="20260505",
        project_root=tmp_path,
        cached_pair_fn=lambda **_kwargs: [QWEN_SPEC, GEMMA_SPEC],
        gpu_health_fn=lambda: mod.GPUHealth(True, 2, []),
        generation_fn=_perfect_generation,
        max_models=1,
        **paths,
    )

    assert [write["status"] for write in writes] == ["in_progress", "complete"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["artifact_metadata"]["project_root"] == str(tmp_path)
    assert artifact["artifact_metadata"]["run_date"] == "20260505"
