"""Tests for Exp 2910 SOTA code-generation corrigendum.

Spec: REQ-CODE-2910, SCENARIO-CODE-2910.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import sota_code_generation_corrigendum as exp


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


def _stage_repo(tmp_path: Path, *, n_rows: int = 20) -> None:
    mbpp_path = tmp_path / "data" / "eval_manifests" / "mbpp_20260522.jsonl"
    humaneval_path = tmp_path / "data" / "eval_manifests" / "humaneval_20260522.jsonl"
    _write_jsonl(
        mbpp_path,
        [
            {
                "canonical_code": "def add_one(x):\n    return x + 1\n",
                "dataset": "MBPP",
                "prompt": f"Write a function that adds one. Case {index}.",
                "stable_id": f"mbpp-add-one-{index}",
                "test_imports": [],
                "tests": ["assert add_one(1) == 2", "assert add_one(-1) == 0"],
            }
            for index in range(n_rows)
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
                "stable_id": f"HumanEval/{index}",
                "tests": "\ndef check(candidate):\n    assert candidate(3) == 6\n",
            }
            for index in range(n_rows)
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
            "resolved_manifest_counts": {"mbpp": n_rows, "humaneval": n_rows},
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


def _config(tmp_path: Path, **overrides: Any) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        raw_response_dir=tmp_path / "results" / "raw" / "exp2910",
        tests_run=("focused-pytest",),
        started_at=overrides.pop("started_at", 10.0),
        clock=overrides.pop("clock", lambda: 75.0),
        n_tasks_per_corpus=overrides.pop("n_tasks_per_corpus", 20),
        k_candidates_per_task=overrides.pop("k_candidates_per_task", 8),
        max_tokens=overrides.pop("max_tokens", 256),
        random_seed=overrides.pop("random_seed", 2910),
        sandbox_timeout_s=overrides.pop("sandbox_timeout_s", 5.0),
        duration_floor_s=overrides.pop("duration_floor_s", 60.0),
        **overrides,
    )


def _sota_pair(tmp_path: Path) -> list[dict[str, Any]]:
    model_a = tmp_path / "models" / "qwen.gguf"
    model_b = tmp_path / "models" / "gemma.gguf"
    model_a.parent.mkdir(parents=True, exist_ok=True)
    model_a.write_bytes(b"qwen")
    model_b.write_bytes(b"gemma")
    return [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "gpu": 0,
            "model_path": str(model_a),
        },
        {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "gpu": 1,
            "model_path": str(model_b),
        },
    ]


def test_scenario_code_2910_writes_corrigendum_schema_with_k8_gain(tmp_path: Path) -> None:
    """SCENARIO-CODE-2910: k=8 corrected row records seeds, raw text, and pass metrics."""

    _stage_repo(tmp_path, n_rows=20)
    seeds: list[int] = []
    scripts: list[str] = []

    def generator(
        corpus: str,
        row: dict[str, Any],
        seed: int,
        max_tokens: int,
        model_spec: dict[str, Any],
    ) -> exp.GenerationOutcome:
        seeds.append(seed)
        assert max_tokens == 256
        candidate_ix = (seed - 2910) % 8
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
            duration_s=0.01,
            backend="fake-live",
            backend_detail=str(model_spec["model_path"]),
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
        cached_pair_provider=lambda gpu_indices=(0, 1): _sota_pair(tmp_path),
        mandated_model_resolver=lambda: [],
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text())

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["codegen_corrigendum_ready"] is True
    assert artifact["flagged_adversarial"] is False
    assert artifact["random_seed"] == 2910
    assert artifact["random_seeds_used"] == list(range(2910, 3230))
    assert artifact["random_seeds_used"] == seeds
    assert artifact["cached_sota_pair_used"] is True
    assert artifact["legacy_smoke_only"] is False
    assert artifact["models_used"] == ["unsloth/Qwen3.6-35B-A3B-GGUF"]
    assert artifact["n_tasks_per_corpus"] == 20
    assert artifact["k_candidates_per_task"] == 8
    assert artifact["aggregate_pass_at_1"] == pytest.approx(0.5)
    assert artifact["aggregate_pass_at_k"] == pytest.approx(1.0)
    assert artifact["pass_at_k_exceeds_pass_at_1"] is True
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["duration_s"] == pytest.approx(65.0)
    assert len(artifact["per_task_results"]) == 40
    assert len(artifact["candidate_results"]) == 320
    assert len(scripts) == 320
    first_task = artifact["per_task_results"][0]
    assert first_task["pass_vector"] == [False, False, False, True, False, False, False, False]
    assert first_task["first_passing_candidate_index"] == 3
    first_candidate = artifact["candidate_results"][0]
    assert first_candidate["model_path"].endswith("qwen.gguf")
    assert first_candidate["gpu_index"] == 0
    assert first_candidate["prompt_sha256"]
    raw_path = tmp_path / first_candidate["raw_response_path"]
    assert raw_path.read_text(encoding="utf-8") == first_candidate["raw_response"]
    assert artifact["methodology_check"] == "pass_at_k_exceeds_pass_at_1"


def test_req_code_2910_blocks_without_mandated_sota_cache(tmp_path: Path) -> None:
    """REQ-CODE-2910: no mandated GGUF cache writes blocked artifact without pass rates."""

    _stage_repo(tmp_path, n_rows=20)
    invoked: list[str] = []

    def generator(*_args: Any, **_kwargs: Any) -> exp.GenerationOutcome:
        invoked.append("generator")
        return exp.GenerationOutcome(text="", tokens_generated=0, duration_s=0.0, backend="fake")

    artifact = exp.build_experiment_artifact(
        _config(tmp_path),
        generator=generator,
        executor=lambda _script, _timeout_s: exp.ExecutionOutcome(passed=True),
        cached_pair_provider=lambda gpu_indices=(0, 1): None,
        mandated_model_resolver=lambda: [],
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )

    assert artifact["honest_verdict"] == "blocked_sota_gguf_cache_missing"
    assert artifact["codegen_corrigendum_ready"] is False
    assert artifact["flagged_adversarial"] is True
    assert artifact["cached_sota_pair_used"] is False
    assert artifact["legacy_smoke_only"] is True
    assert artifact["models_used"] == []
    assert artifact["aggregate_pass_at_1"] is None
    assert artifact["aggregate_pass_at_k"] is None
    assert artifact["candidate_results"] == []
    assert invoked == []


def test_req_code_2910_falls_back_to_single_mandated_model_and_explains_equal_metrics(
    tmp_path: Path,
) -> None:
    """REQ-CODE-2910: pair miss may use one SOTA model and explain equal metrics."""

    _stage_repo(tmp_path, n_rows=1)
    model_path = tmp_path / "models" / "gemma.gguf"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"gemma")

    def generator(
        corpus: str,
        row: dict[str, Any],
        seed: int,
        max_tokens: int,
        model_spec: dict[str, Any],
    ) -> exp.GenerationOutcome:
        text = (
            "```python\ndef add_one(x):\n    return x + 1\n```"
            if corpus == "mbpp"
            else "```python\ndef double(x):\n    return x * 2\n```"
        )
        return exp.GenerationOutcome(text=text, tokens_generated=8, duration_s=0.01, backend="fake")

    artifact = exp.build_experiment_artifact(
        _config(tmp_path, n_tasks_per_corpus=1, duration_floor_s=0.0),
        generator=generator,
        executor=lambda _script, _timeout_s: exp.ExecutionOutcome(passed=True),
        cached_pair_provider=lambda gpu_indices=(0, 1): None,
        mandated_model_resolver=lambda: [
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "gpu": 0,
                "model_path": str(model_path),
            }
        ],
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["cached_sota_pair_used"] is False
    assert artifact["models_used"] == ["unsloth/gemma-4-26B-A4B-it-GGUF"]
    assert artifact["aggregate_pass_at_1"] == pytest.approx(1.0)
    assert artifact["aggregate_pass_at_k"] == pytest.approx(1.0)
    assert artifact["pass_at_k_exceeds_pass_at_1"] is False
    assert artifact["methodology_check"] == "equal_metrics_explained_per_task"
    assert all(task["pass_metric_explanation"] for task in artifact["per_task_results"])


def test_extract_python_candidate_classifies_failures() -> None:
    """REQ-CODE-2910: code extraction classifies non-code and syntax failures."""

    fenced = exp.extract_python_candidate("note\n```python\ndef f():\n    return 1\n```")
    raw = exp.extract_python_candidate("def g():\n    return 2\n")
    fenced_without_function = exp.extract_python_candidate("```python\nvalue = 1\n```")
    missing = exp.extract_python_candidate("no function here")
    syntax = exp.extract_python_candidate("```python\ndef bad(:\n    pass\n```")

    assert fenced.code == "def f():\n    return 1"
    assert fenced.extraction_status == "python_fence"
    assert fenced.extraction_success is True
    assert fenced.syntax_success is True
    assert raw.extraction_status == "raw_function"
    assert raw.syntax_success is True
    assert fenced_without_function.extraction_status == "no_function_found"
    assert missing.extraction_success is False
    assert missing.extraction_status == "no_function_found"
    assert syntax.extraction_success is True
    assert syntax.syntax_success is False
    assert syntax.extraction_status == "syntax_error"


def test_req_code_2910_blocks_manifest_sandbox_and_insufficient_rows(tmp_path: Path) -> None:
    """REQ-CODE-2910: manifest, sample-size, and sandbox gates block before generation."""

    _stage_repo(tmp_path, n_rows=20)
    model_specs = _sota_pair(tmp_path)
    config = _config(tmp_path)

    contract_path = tmp_path / exp.MANIFEST_CONTRACT_REL_PATH
    contract_payload = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_payload["resolved_manifest_sha256"]["mbpp"] = "0" * 64
    _write_json(contract_path, contract_payload)
    bad_manifest = exp.build_experiment_artifact(
        config,
        generator=lambda *_args, **_kwargs: pytest.fail("generator should not run"),
        cached_pair_provider=lambda gpu_indices=(0, 1): model_specs,
        mandated_model_resolver=lambda: [],
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )
    assert bad_manifest["honest_verdict"] == "blocked_manifest_contract"

    _stage_repo(tmp_path, n_rows=1)
    insufficient = exp.build_experiment_artifact(
        config,
        generator=lambda *_args, **_kwargs: pytest.fail("generator should not run"),
        cached_pair_provider=lambda gpu_indices=(0, 1): model_specs,
        mandated_model_resolver=lambda: [],
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )
    assert insufficient["honest_verdict"] == "blocked_insufficient_local_code_rows"

    _stage_repo(tmp_path, n_rows=20)
    no_sandbox = exp.build_experiment_artifact(
        config,
        generator=lambda *_args, **_kwargs: pytest.fail("generator should not run"),
        cached_pair_provider=lambda gpu_indices=(0, 1): model_specs,
        mandated_model_resolver=lambda: [],
        sandbox_status_provider=lambda: {"available": False, "runtime": "none"},
    )
    assert no_sandbox["honest_verdict"] == "blocked_sandbox"


def test_req_code_2910_records_candidate_failures_and_duration_floor(
    tmp_path: Path,
) -> None:
    """REQ-CODE-2910: generation, extraction, syntax, and short-duration failures are explicit."""

    _stage_repo(tmp_path, n_rows=1)

    def generator(
        corpus: str,
        row: dict[str, Any],
        seed: int,
        max_tokens: int,
        model_spec: dict[str, Any],
    ) -> exp.GenerationOutcome:
        candidate_ix = (seed - 2910) % 8
        if candidate_ix == 0:
            return exp.GenerationOutcome(
                text="",
                tokens_generated=0,
                duration_s=0.01,
                backend="fake",
                error="timeout",
            )
        if candidate_ix == 1:
            text = "this is prose without code"
        elif candidate_ix == 2:
            text = "```python\ndef bad(:\n    pass\n```"
        else:
            text = (
                "```python\ndef add_one(x):\n    return x\n```"
                if corpus == "mbpp"
                else "```python\ndef double(x):\n    return x\n```"
            )
        return exp.GenerationOutcome(text=text, tokens_generated=4, duration_s=0.01, backend="fake")

    def executor(script: str, timeout_s: float) -> exp.ExecutionOutcome:
        if "def bad(:" in script:
            return exp.ExecutionOutcome(passed=False, error_type="SyntaxError", error_message="bad")
        return exp.ExecutionOutcome(passed=False, error_type="AssertionError", error_message="failed")

    artifact = exp.build_experiment_artifact(
        _config(tmp_path, n_tasks_per_corpus=1, clock=lambda: 12.0),
        generator=generator,
        executor=executor,
        cached_pair_provider=lambda gpu_indices=(0, 1): _sota_pair(tmp_path),
        mandated_model_resolver=lambda: [],
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )

    assert artifact["honest_verdict"] == "blocked_duration_floor_not_met"
    first, second, third = artifact["candidate_results"][:3]
    assert first["row_status"] == "candidate_generation_failed"
    assert first["executed"] is False
    assert second["row_status"] == "candidate_extraction_failed"
    assert second["extraction_status"] == "no_function_found"
    assert third["row_status"] == "candidate_syntax_failed"
    assert third["syntax_success"] is False
    assert third["runtime_success"] is False
    assert artifact["per_task_results"][0]["pass_metric_explanation"] == (
        "no_candidate_in_the_bounded_k_set_passed_for_this_task"
    )


def test_req_code_2910_blocks_equal_metrics_without_explanations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-2910: equal aggregate metrics need explicit per-task explanations."""

    _stage_repo(tmp_path, n_rows=1)
    monkeypatch.setattr(exp, "_pass_metric_explanation", lambda _pass_vector: "")

    def generator(
        corpus: str,
        row: dict[str, Any],
        seed: int,
        max_tokens: int,
        model_spec: dict[str, Any],
    ) -> exp.GenerationOutcome:
        text = (
            "```python\ndef add_one(x):\n    return x\n```"
            if corpus == "mbpp"
            else "```python\ndef double(x):\n    return x\n```"
        )
        return exp.GenerationOutcome(text=text, tokens_generated=4, duration_s=0.01, backend="fake")

    artifact = exp.build_experiment_artifact(
        _config(tmp_path, n_tasks_per_corpus=1, duration_floor_s=0.0),
        generator=generator,
        executor=lambda _script, _timeout_s: exp.ExecutionOutcome(
            passed=False,
            error_type="AssertionError",
            error_message="failed",
        ),
        cached_pair_provider=lambda gpu_indices=(0, 1): _sota_pair(tmp_path),
        mandated_model_resolver=lambda: [],
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )

    assert artifact["honest_verdict"] == "blocked_equal_metrics_without_per_task_explanation"
    assert artifact["flagged_adversarial"] is True


def test_default_model_resolver_and_llama_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-2910: default SOTA resolver and llama wrapper preserve model provenance."""

    model_path = tmp_path / "models" / "qwen.gguf"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"qwen")

    def fake_resolve(hf_id: str) -> str | None:
        return str(model_path) if hf_id == "unsloth/Qwen3.6-35B-A3B-GGUF" else None

    monkeypatch.setattr(exp, "resolve_cached_gguf", fake_resolve)
    resolved = exp._default_mandated_model_resolver()
    assert resolved == [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "gpu": 0,
            "model_path": str(model_path),
        }
    ]

    calls: list[tuple[str, float]] = []

    def fake_llama_cpp_generator(*, model_path: str, temperature: float) -> exp.Generator:
        calls.append((model_path, temperature))

        def base(corpus: str, row: dict[str, Any], seed: int, max_tokens: int) -> exp.GenerationOutcome:
            return exp.GenerationOutcome(
                text=f"{corpus}:{row['stable_id']}:{seed}:{max_tokens}",
                tokens_generated=1,
                duration_s=0.01,
                backend="fake-llama",
            )

        return base

    monkeypatch.setattr(exp.exp2889, "llama_cpp_generator", fake_llama_cpp_generator)
    wrapped = exp.llama_cpp_model_generator(model_path=str(model_path), temperature=0.2)
    outcome = wrapped("mbpp", {"stable_id": "row"}, 7, 11, {"model_path": str(model_path)})
    assert calls == [(str(model_path), 0.2)]
    assert outcome.text == "mbpp:row:7:11"


def test_run_experiment_uses_selected_model_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CODE-2910: default entrypoint wires the selected model into generation."""

    _stage_repo(tmp_path, n_rows=1)
    model_path = tmp_path / "models" / "qwen.gguf"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"qwen")
    seen: list[tuple[str, float]] = []

    def fake_factory(*, model_path: str, temperature: float) -> exp.Generator:
        seen.append((model_path, temperature))

        def generator(
            corpus: str,
            row: dict[str, Any],
            seed: int,
            max_tokens: int,
            model_spec: dict[str, Any],
        ) -> exp.GenerationOutcome:
            text = (
                "```python\ndef add_one(x):\n    return x + 1\n```"
                if corpus == "mbpp"
                else "```python\ndef double(x):\n    return x * 2\n```"
            )
            return exp.GenerationOutcome(text=text, tokens_generated=8, duration_s=0.01, backend="fake")

        return generator

    monkeypatch.setattr(exp, "llama_cpp_model_generator", fake_factory)
    artifact = exp.run_experiment(
        _config(tmp_path, n_tasks_per_corpus=1, duration_floor_s=0.0),
        executor=lambda _script, _timeout_s: exp.ExecutionOutcome(passed=True),
        cached_pair_provider=lambda gpu_indices=(0, 1): [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "gpu": 0,
                "model_path": str(model_path),
            }
        ],
        mandated_model_resolver=lambda: [],
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )

    assert seen == [(str(model_path), exp.DEFAULT_TEMPERATURE)]
    assert artifact["honest_verdict"].startswith("complete:")
