"""Tests for Exp 2905 bounded-budget SOTA code-generation expansion.

Spec: REQ-CODE-2905, SCENARIO-CODE-2905.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import sota_code_generation_bounded_budget_expansion as exp


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _stage_repo(
    tmp_path: Path,
    *,
    sota_runtime_clean: bool = True,
    gpu_available: bool = True,
    gpu_offload: bool = True,
) -> Path:
    model_path = tmp_path / "models" / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"fake-gguf")

    mbpp_path = tmp_path / "data" / "eval_manifests" / "mbpp_20260522.jsonl"
    humaneval_path = tmp_path / "data" / "eval_manifests" / "humaneval_20260522.jsonl"
    _write_jsonl(
        mbpp_path,
        [
            {
                "canonical_code": "def add_one(x):\n    return x + 1\n",
                "dataset": "MBPP",
                "prompt": "Write a function that adds one.",
                "stable_id": "mbpp-add-one",
                "test_imports": [],
                "tests": ["assert add_one(1) == 2", "assert add_one(-1) == 0"],
            }
        ],
    )
    _write_jsonl(
        humaneval_path,
        [
            {
                "canonical_solution": "    return x * 2\n",
                "dataset": "HumanEval",
                "entry_point": "double",
                "prompt": "def double(x: int) -> int:\n    \"\"\"Return x doubled.\"\"\"\n",
                "stable_id": "HumanEval/0",
                "tests": "\ndef check(candidate):\n    assert candidate(3) == 6\n",
            }
        ],
    )
    _write_json(
        tmp_path / exp.MANIFEST_CONTRACT_REL_PATH,
        {
            "artifact": "experiment_2863_eval_manifest_contract_v2",
            "honest_verdict": "complete: eval manifest contract ready",
            "manifest_contract_ready": True,
            "mbpp_ready": True,
            "humaneval_ready": True,
            "resolved_manifest_counts": {"mbpp": 1, "humaneval": 1},
            "resolved_manifest_paths": {
                "mbpp": str(mbpp_path),
                "humaneval": str(humaneval_path),
            },
            "resolved_manifest_sha256": {
                "mbpp": _sha256(mbpp_path),
                "humaneval": _sha256(humaneval_path),
            },
        },
    )
    _write_json(
        tmp_path / exp.CROSS_CORPUS_MATRIX_REL_PATH,
        {
            "artifact": "experiment_2865_cross_corpus_matrix_v5",
            "honest_verdict": "complete: cross-corpus matrix built",
        },
    )
    _write_json(
        tmp_path / exp.EXP2874_REL_PATH,
        {
            "artifact": "experiment_2874_sota_runtime_clean_corrigendum_v4",
            "sota_runtime_clean": sota_runtime_clean,
            "selected_model_hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "selected_model_path": str(model_path),
            "selected_model_checksum_or_fingerprint": "sha256:deadbeef;size_bytes=9",
            "model_specs": [
                {"name": "Gemma4-26B-A4B-it", "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF"}
            ],
            "llama_cpp_supports_gpu_offload": gpu_offload,
            "gpu_inventory": {"available": gpu_available},
        },
    )
    return model_path


def _config(tmp_path: Path, **overrides: Any) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        tests_run=("focused-pytest",),
        started_at=overrides.pop("started_at", 10.0),
        clock=overrides.pop("clock", lambda: 75.0),
        n_tasks_per_corpus=overrides.pop("n_tasks_per_corpus", 1),
        k_candidates_per_task=overrides.pop("k_candidates_per_task", 8),
        max_tokens=overrides.pop("max_tokens", 768),
        random_seed=overrides.pop("random_seed", 2905),
        sandbox_timeout_s=overrides.pop("sandbox_timeout_s", 5.0),
        duration_floor_s=overrides.pop("duration_floor_s", 60.0),
        **overrides,
    )


def test_scenario_code_2905_records_k8_pass_at_k_gain(tmp_path: Path) -> None:
    """SCENARIO-CODE-2905: k=8 sampling records per-task pass@1 and pass@k."""

    _stage_repo(tmp_path)
    seeds: list[int] = []
    scripts: list[str] = []

    def generator(corpus: str, row: dict[str, Any], seed: int, max_tokens: int) -> exp.GenerationOutcome:
        seeds.append(seed)
        assert max_tokens == 768
        candidate_ix = (seed - 2905) % 8
        if corpus == "mbpp":
            text = (
                "```python\ndef add_one(x):\n    return x + 1\n```"
                if candidate_ix == 3
                else "```python\ndef add_one(x):\n    return x\n```"
            )
        else:
            text = (
                "```python\ndef double(x):\n    return x * 2\n```"
                if candidate_ix == 0
                else "```python\ndef double(x):\n    return x\n```"
            )
        return exp.GenerationOutcome(
            text=text,
            tokens_generated=len(text.split()),
            duration_s=0.25,
            backend="fake-live",
            backend_detail=str(row["stable_id"]),
        )

    def executor(script: str, timeout_s: float) -> exp.ExecutionOutcome:
        scripts.append(script)
        assert timeout_s == pytest.approx(5.0)
        passed = "return x + 1" in script or "return x * 2" in script
        return exp.ExecutionOutcome(
            passed=passed,
            error_type=None if passed else "AssertionError",
            error_message="" if passed else "assertion failed",
        )

    artifact = exp.write_experiment_artifact(
        _config(tmp_path),
        generator=generator,
        executor=executor,
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text())

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["model_specs"]["principle"] == exp.LIVE_MODEL_PRINCIPLE
    assert artifact["n_tasks_per_corpus"] == 1
    assert artifact["k_candidates_per_task"] == 8
    assert artifact["max_tokens_per_candidate"] == 768
    assert artifact["per_task_pass_at_1"] == [0.0, 1.0]
    assert artifact["per_task_pass_at_k"] == [1.0, 1.0]
    assert artifact["pass_at_k_exceeds_pass_at_1"] is True
    assert artifact["aggregate_pass_at_1"] == pytest.approx(0.5)
    assert artifact["aggregate_pass_at_k"] == pytest.approx(1.0)
    assert artifact["random_seeds_used"] == list(range(2905, 2921))
    assert artifact["random_seeds_used"] == seeds
    assert len(artifact["task_results"]) == 2
    assert len(artifact["candidate_results"]) == 16
    assert len(scripts) == 16
    mbpp_task = artifact["task_results"][0]
    assert mbpp_task["stable_id"] == "mbpp-add-one"
    assert mbpp_task["first_passing_candidate_index"] == 3
    assert artifact["duration_s"] == pytest.approx(65.0)
    assert artifact["reproducibility_checksum"]


def test_req_code_2905_blocks_before_generation_when_sandbox_missing(tmp_path: Path) -> None:
    """REQ-CODE-2905: missing runsc writes terminal blocked artifact."""

    _stage_repo(tmp_path)
    invoked: list[str] = []

    def generator(*_args: Any, **_kwargs: Any) -> exp.GenerationOutcome:
        invoked.append("generator")
        return exp.GenerationOutcome(text="", tokens_generated=0, duration_s=0.0, backend="fake")

    artifact = exp.build_experiment_artifact(
        _config(tmp_path),
        generator=generator,
        executor=lambda _script, _timeout_s: exp.ExecutionOutcome(passed=True),
        sandbox_status_provider=lambda: {"available": False, "runtime": "none"},
    )

    assert artifact["honest_verdict"] == "blocked_sandbox"
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["candidate_results"] == []
    assert artifact["per_task_pass_at_1"] == []
    assert invoked == []


def test_req_code_2905_blocks_for_missing_or_unclean_exp2874(tmp_path: Path) -> None:
    """REQ-CODE-2905: Exp 2874 clean runtime evidence is a hard gate."""

    _stage_repo(tmp_path, sota_runtime_clean=False)
    artifact = exp.build_experiment_artifact(
        _config(tmp_path),
        generator=lambda *_args, **_kwargs: pytest.fail("generator should not run"),
        executor=lambda _script, _timeout_s: exp.ExecutionOutcome(passed=True),
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )
    assert artifact["honest_verdict"] == "blocked_exp2874_sota_runtime_not_clean"

    (tmp_path / exp.EXP2874_REL_PATH).unlink()
    missing = exp.build_experiment_artifact(
        _config(tmp_path),
        generator=lambda *_args, **_kwargs: pytest.fail("generator should not run"),
        executor=lambda _script, _timeout_s: exp.ExecutionOutcome(passed=True),
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )
    assert missing["honest_verdict"] == "blocked_exp2874_sota_runtime_not_clean"


def test_req_code_2905_blocks_model_gpu_and_manifest_preconditions(tmp_path: Path) -> None:
    """REQ-CODE-2905: model path, GPU offload, and manifest checksum gates are enforced."""

    model_path = _stage_repo(tmp_path)
    model_path.unlink()
    missing_model = exp.build_experiment_artifact(
        _config(tmp_path),
        generator=lambda *_args, **_kwargs: pytest.fail("generator should not run"),
        executor=lambda _script, _timeout_s: exp.ExecutionOutcome(passed=True),
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )
    assert missing_model["honest_verdict"] == "blocked_selected_model_path_missing"

    model_path.write_bytes(b"fake-gguf")
    exp2874_path = tmp_path / exp.EXP2874_REL_PATH
    exp2874_payload = json.loads(exp2874_path.read_text(encoding="utf-8"))
    exp2874_payload["gpu_inventory"] = {"available": False}
    _write_json(exp2874_path, exp2874_payload)
    no_gpu = exp.build_experiment_artifact(
        _config(tmp_path),
        generator=lambda *_args, **_kwargs: pytest.fail("generator should not run"),
        executor=lambda _script, _timeout_s: exp.ExecutionOutcome(passed=True),
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )
    assert no_gpu["honest_verdict"] == "blocked_gpu_offload_unavailable"

    exp2874_payload["gpu_inventory"] = {"available": True}
    _write_json(exp2874_path, exp2874_payload)
    contract_path = tmp_path / exp.MANIFEST_CONTRACT_REL_PATH
    contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_payload["resolved_manifest_sha256"]["mbpp"] = "0" * 64
    _write_json(contract_path, contract_payload)
    bad_manifest = exp.build_experiment_artifact(
        _config(tmp_path),
        generator=lambda *_args, **_kwargs: pytest.fail("generator should not run"),
        executor=lambda _script, _timeout_s: exp.ExecutionOutcome(passed=True),
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )
    assert bad_manifest["honest_verdict"] == "blocked_manifest_contract"


def test_req_code_2905_candidate_generation_failure_and_duration_floor(tmp_path: Path) -> None:
    """REQ-CODE-2905: candidate failures are recorded and short live runs cannot complete."""

    _stage_repo(tmp_path)

    def generator(corpus: str, row: dict[str, Any], seed: int, max_tokens: int) -> exp.GenerationOutcome:
        if seed == 2905:
            return exp.GenerationOutcome(
                text="",
                tokens_generated=0,
                duration_s=0.01,
                backend="fake-live",
                error="empty_output",
            )
        text = (
            "```python\ndef add_one(x):\n    return x + 1\n```"
            if corpus == "mbpp"
            else "```python\ndef double(x):\n    return x * 2\n```"
        )
        return exp.GenerationOutcome(text=text, tokens_generated=8, duration_s=0.01, backend="fake-live")

    artifact = exp.build_experiment_artifact(
        _config(tmp_path, clock=lambda: 20.0),
        generator=generator,
        executor=lambda _script, _timeout_s: exp.ExecutionOutcome(passed=True),
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )

    assert artifact["honest_verdict"] == "blocked_duration_floor_not_met"
    assert artifact["candidate_generation_clean"] is False
    assert artifact["candidate_results"][0]["row_status"] == "blocked_generation"
    assert artifact["candidate_results"][0]["executed"] is False
    assert artifact["per_task_pass_at_1"] == [0.0, 1.0]
    assert artifact["per_task_pass_at_k"] == [1.0, 1.0]
    assert artifact["blocked_reason"].startswith("Live inference duration")

    unclean = exp.build_experiment_artifact(
        _config(tmp_path, duration_floor_s=0.0),
        generator=generator,
        executor=lambda _script, _timeout_s: exp.ExecutionOutcome(passed=True),
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )
    assert unclean["honest_verdict"] == "blocked_generation_or_execution_unclean"


def test_req_code_2905_blocks_when_no_eligible_rows(tmp_path: Path) -> None:
    """REQ-CODE-2905: no eligible MBPP/HumanEval tasks blocks before generation."""

    mbpp_manifest = tmp_path / "data" / "eval_manifests" / "mbpp_20260522.jsonl"
    humaneval_manifest = tmp_path / "data" / "eval_manifests" / "humaneval_20260522.jsonl"
    _stage_repo(tmp_path)
    _write_jsonl(mbpp_manifest, [])
    _write_jsonl(humaneval_manifest, [])
    contract_path = tmp_path / exp.MANIFEST_CONTRACT_REL_PATH
    contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_payload["resolved_manifest_sha256"]["mbpp"] = _sha256(mbpp_manifest)
    contract_payload["resolved_manifest_sha256"]["humaneval"] = _sha256(humaneval_manifest)
    _write_json(contract_path, contract_payload)

    artifact = exp.build_experiment_artifact(
        _config(tmp_path),
        generator=lambda *_args, **_kwargs: pytest.fail("generator should not run"),
        executor=lambda _script, _timeout_s: exp.ExecutionOutcome(passed=True),
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )

    assert artifact["honest_verdict"] == "blocked_no_eligible_code_rows"
    assert artifact["selected_task_ids"] == []


def test_extract_python_candidate_prefers_function_fence_and_raw_text() -> None:
    """REQ-CODE-2905: extraction avoids non-code fences when possible."""

    text = "```text\nnot code\n```\n```python\ndef f():\n    return 1\n```\n"
    assert exp.extract_python_candidate(text) == "def f():\n    return 1"
    assert exp.extract_python_candidate("```\nvalue = 1\n```") == "value = 1"
    assert exp.extract_python_candidate("def g():\n    return 2\n") == "def g():\n    return 2"


def test_run_experiment_uses_live_generator_factory_when_not_injected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-2905: default entrypoint wires the selected model into the live generator."""

    model_path = _stage_repo(tmp_path)
    seen_model_paths: list[str] = []

    def fake_factory(*, model_path: str, temperature: float) -> exp.Generator:
        seen_model_paths.append(model_path)
        assert temperature == pytest.approx(exp.DEFAULT_TEMPERATURE)

        def generator(corpus: str, row: dict[str, Any], seed: int, max_tokens: int) -> exp.GenerationOutcome:
            text = (
                "```python\ndef add_one(x):\n    return x + 1\n```"
                if corpus == "mbpp"
                else "```python\ndef double(x):\n    return x * 2\n```"
            )
            return exp.GenerationOutcome(text=text, tokens_generated=8, duration_s=0.1, backend="fake-live")

        return generator

    monkeypatch.setattr(exp, "llama_cpp_generator", fake_factory)
    artifact = exp.run_experiment(
        _config(tmp_path, k_candidates_per_task=1),
        executor=lambda _script, _timeout_s: exp.ExecutionOutcome(passed=True),
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )

    assert seen_model_paths == [str(model_path)]
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["k_candidates_per_task"] == 1
