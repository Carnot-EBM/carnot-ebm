"""Tests for Exp 2889 smallest defensible MBPP/HumanEval generated-code row.

Spec: REQ-CODE-2889, SCENARIO-CODE-2889.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import mbpp_humaneval_generated_code_clean_row as exp


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
    model_path_override: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Lay out the manifest contract + Exp 2874 evidence the runner expects."""

    model_path = model_path_override
    if model_path is None:
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
                "stable_id": "mbpp-eligible-a",
                "test_imports": [],
                "tests": ["assert add_one(1) == 2", "assert add_one(-1) == 0"],
            },
            {
                "canonical_code": "def negate(x):\n    return -x\n",
                "dataset": "MBPP",
                "prompt": "Write a function that negates the input.",
                "stable_id": "mbpp-eligible-b",
                "test_imports": [],
                "tests": ["assert negate(3) == -3"],
            },
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
                "stable_id": "HumanEval/eligible",
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
            "resolved_manifest_counts": {"mbpp": 2, "humaneval": 1},
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
            "llama_cpp_supports_gpu_offload": True,
            "gpu_inventory": {"available": True},
        },
    )
    return mbpp_path, humaneval_path, model_path


def _config(tmp_path: Path, **overrides: Any) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        tests_run=("focused-pytest",),
        started_at=10.0,
        clock=lambda: 12.5,
        target_per_corpus=overrides.pop("target_per_corpus", 2),
        max_tokens=overrides.pop("max_tokens", 64),
        random_seed=overrides.pop("random_seed", 4242),
        sandbox_timeout_s=overrides.pop("sandbox_timeout_s", 5.0),
        **overrides,
    )


def _passing_generator(seed_log: list[int]) -> exp.Generator:
    canned = {
        ("mbpp", "mbpp-eligible-a"): "```python\ndef add_one(x):\n    return x + 1\n```",
        ("mbpp", "mbpp-eligible-b"): "```python\ndef negate(x):\n    return -x\n```",
        ("humaneval", "HumanEval/eligible"): "```python\ndef double(x):\n    return x * 2\n```",
    }

    def _generate(corpus, row, seed, max_tokens):
        seed_log.append(seed)
        text = canned[(corpus, str(row["stable_id"]))]
        return exp.GenerationOutcome(
            text=text,
            tokens_generated=len(text.split()),
            duration_s=0.42,
            backend="fake",
            backend_detail="canned",
        )

    return _generate


def _ok_executor(record: list[str]) -> exp.Executor:
    def _exec(script: str, timeout_s: float) -> exp.ExecutionOutcome:
        record.append(script)
        assert timeout_s == pytest.approx(5.0)
        return exp.ExecutionOutcome(passed=True)

    return _exec


def test_scenario_code_2889_clean_pilot_row_with_passes(tmp_path: Path) -> None:
    """SCENARIO-CODE-2889: clean SOTA GGUF generation + sandbox execution + passing tests."""

    _stage_repo(tmp_path)
    seeds: list[int] = []
    scripts: list[str] = []
    config = _config(tmp_path)

    artifact = exp.write_experiment_artifact(
        config,
        generator=_passing_generator(seeds),
        executor=_ok_executor(scripts),
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text())

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["row_status"] == "pilot_only_clean_with_passes"
    assert artifact["generated_code_row_clean"] is True
    assert artifact["deterministic_execution_used"] is True
    assert artifact["sandbox_status"] == "available: runsc"
    assert artifact["n_mbpp_rows"] == 2
    assert artifact["n_humaneval_rows"] == 1
    assert artifact["n_generated_outputs"] == 3
    assert artifact["pass_rate_if_computable"] == pytest.approx(1.0)
    assert artifact["headline_metric_claim_made"] is False
    assert artifact["blocked_reason"] == ""
    assert artifact["random_seed"] == 4242
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["selected_model_hf_id"] == "unsloth/gemma-4-26B-A4B-it-GGUF"
    assert artifact["model_specs"][0]["hf_id"] == "unsloth/gemma-4-26B-A4B-it-GGUF"
    assert artifact["reproducibility_checksum"] != ""
    assert len(scripts) == 3
    assert any("assert add_one(1) == 2" in s for s in scripts)
    assert any("check(double)" in s for s in scripts)
    assert seeds == [4242, 4243, 4244]
    rows_by_id = {row["stable_id"]: row for row in artifact["row_results"]}
    assert rows_by_id["mbpp-eligible-a"]["passed"] is True
    assert rows_by_id["mbpp-eligible-a"]["extracted_code"].startswith("def add_one")
    assert rows_by_id["mbpp-eligible-a"]["n_tests"] == 2
    assert rows_by_id["HumanEval/eligible"]["n_tests"] == 1


def test_req_code_2889_blocks_when_exp2874_not_clean(tmp_path: Path) -> None:
    """REQ-CODE-2889: missing Exp 2874 clean-runtime evidence writes blocked_*."""

    _stage_repo(tmp_path, sota_runtime_clean=False)

    def forbidden_generator(*_args, **_kwargs):
        raise AssertionError("generator should not be invoked when preconditions fail")

    artifact = exp.build_experiment_artifact(
        _config(tmp_path),
        generator=forbidden_generator,
        executor=lambda script, timeout_s: exp.ExecutionOutcome(passed=True),
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )

    assert artifact["honest_verdict"] == "blocked_exp2874_sota_runtime_not_clean"
    assert artifact["row_status"] == "blocked_preconditions"
    assert artifact["generated_code_row_clean"] is False
    assert artifact["row_results"] == []
    assert artifact["n_generated_outputs"] == 0
    assert artifact["blocked_reason"].startswith("Exp 2874")


def test_req_code_2889_blocks_when_sandbox_missing(tmp_path: Path) -> None:
    """REQ-CODE-2889: runsc unavailable refuses any generation or execution."""

    _stage_repo(tmp_path)
    invoked: list[str] = []

    def gen(*_args, **_kwargs):
        invoked.append("gen")
        return exp.GenerationOutcome(text="", tokens_generated=0, duration_s=0.0, backend="fake")

    artifact = exp.build_experiment_artifact(
        _config(tmp_path),
        generator=gen,
        executor=lambda script, timeout_s: exp.ExecutionOutcome(passed=True),
        sandbox_status_provider=lambda: {"available": False, "runtime": "none"},
    )

    assert artifact["honest_verdict"] == "blocked_sandbox"
    assert artifact["row_status"] == "blocked_preconditions"
    assert artifact["sandbox_status"] == "blocked_sandbox: runsc unavailable"
    assert artifact["generated_code_row_clean"] is False
    assert invoked == []


def test_req_code_2889_blocks_when_model_path_missing(tmp_path: Path) -> None:
    """REQ-CODE-2889: Exp 2874 referencing a vanished model file blocks the row."""

    _, _, model_path = _stage_repo(tmp_path)
    model_path.unlink()

    artifact = exp.build_experiment_artifact(
        _config(tmp_path),
        generator=lambda *_args, **_kwargs: exp.GenerationOutcome(
            text="x", tokens_generated=1, duration_s=0.1, backend="fake"
        ),
        executor=lambda script, timeout_s: exp.ExecutionOutcome(passed=True),
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )

    assert artifact["honest_verdict"] == "blocked_selected_model_path_missing"
    assert artifact["generated_code_row_clean"] is False


def test_req_code_2889_blocks_when_manifest_checksums_drift(tmp_path: Path) -> None:
    """REQ-CODE-2889: manifest checksum mismatch refuses to call the generator."""

    _stage_repo(tmp_path)
    payload_path = tmp_path / exp.MANIFEST_CONTRACT_REL_PATH
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["resolved_manifest_sha256"]["mbpp"] = "0" * 64
    _write_json(payload_path, payload)

    def forbidden_generator(*_args, **_kwargs):
        raise AssertionError("generator should not run when manifest checksum drifts")

    artifact = exp.build_experiment_artifact(
        _config(tmp_path),
        generator=forbidden_generator,
        executor=lambda script, timeout_s: exp.ExecutionOutcome(passed=True),
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )

    assert artifact["honest_verdict"] == "blocked_manifest_contract"
    assert artifact["generated_code_row_clean"] is False
    assert artifact["manifest_contract_ready"] is False


def test_req_code_2889_row_unclean_when_generation_empty(tmp_path: Path) -> None:
    """REQ-CODE-2889: empty generation marks the row blocked_generation but executes others."""

    _stage_repo(tmp_path)
    scripts: list[str] = []

    def patchy_generator(corpus, row, seed, max_tokens):
        if str(row["stable_id"]) == "mbpp-eligible-a":
            return exp.GenerationOutcome(
                text="", tokens_generated=0, duration_s=0.05, backend="fake",
                error="empty_output",
            )
        text = "```python\ndef negate(x):\n    return -x\n```" if corpus == "mbpp" \
            else "```python\ndef double(x):\n    return x * 2\n```"
        return exp.GenerationOutcome(
            text=text,
            tokens_generated=3,
            duration_s=0.1,
            backend="fake",
            backend_detail="canned",
        )

    def executor(script, timeout_s):
        scripts.append(script)
        return exp.ExecutionOutcome(passed=True)

    artifact = exp.build_experiment_artifact(
        _config(tmp_path),
        generator=patchy_generator,
        executor=executor,
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )

    assert artifact["honest_verdict"] == "blocked_generation_or_execution_unclean"
    assert artifact["generated_code_row_clean"] is False
    assert artifact["row_status"] == "blocked_generation_or_execution_unclean"
    assert artifact["pass_rate_if_computable"] == pytest.approx(1.0)
    assert artifact["n_generated_outputs"] == 2
    by_id = {row["stable_id"]: row for row in artifact["row_results"]}
    assert by_id["mbpp-eligible-a"]["executed"] is False
    assert by_id["mbpp-eligible-a"]["error_type"] == "GenerationFailed"
    assert by_id["mbpp-eligible-b"]["passed"] is True
    assert len(scripts) == 2


def test_req_code_2889_blocks_when_no_eligible_rows(tmp_path: Path) -> None:
    """REQ-CODE-2889: empty manifests block the row before any generation runs."""

    mbpp_path, humaneval_path, _ = _stage_repo(tmp_path)
    _write_jsonl(mbpp_path, [])
    _write_jsonl(humaneval_path, [])
    payload_path = tmp_path / exp.MANIFEST_CONTRACT_REL_PATH
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["resolved_manifest_sha256"]["mbpp"] = _sha256(mbpp_path)
    payload["resolved_manifest_sha256"]["humaneval"] = _sha256(humaneval_path)
    _write_json(payload_path, payload)

    def forbidden_generator(*_args, **_kwargs):
        raise AssertionError("generator should not run when no rows exist")

    artifact = exp.build_experiment_artifact(
        _config(tmp_path),
        generator=forbidden_generator,
        executor=lambda script, timeout_s: exp.ExecutionOutcome(passed=True),
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )

    assert artifact["honest_verdict"] == "blocked_no_eligible_code_rows"
    assert artifact["row_results"] == []


def test_extract_python_code_block_handles_fenced_and_raw_output() -> None:
    """REQ-CODE-2889: fenced ```python``` blocks extract; raw text is returned as-is."""

    fenced = "noise before\n```python\ndef f():\n    return 1\n```\nafter"
    assert exp.extract_python_code_block(fenced) == "def f():\n    return 1"
    assert exp.extract_python_code_block("def g():\n    return 2\n") == "def g():\n    return 2"
    triple_no_lang = "```\ndef h():\n    return 3\n```"
    assert exp.extract_python_code_block(triple_no_lang) == "def h():\n    return 3"


def test_clean_pilot_row_without_passes_records_pilot_only_clean_no_passes(tmp_path: Path) -> None:
    """REQ-CODE-2889: clean generation + clean sandbox execution but failed tests
    stays pilot_only_clean_no_passes (still complete, never headline)."""

    _stage_repo(tmp_path)
    scripts: list[str] = []

    def generator(corpus, row, seed, max_tokens):
        text = "```python\ndef wrong(x):\n    return x\n```"
        return exp.GenerationOutcome(
            text=text, tokens_generated=4, duration_s=0.2, backend="fake"
        )

    def executor(script, timeout_s):
        scripts.append(script)
        return exp.ExecutionOutcome(
            passed=False,
            error_type="AssertionError",
            error_message="assertion failed",
        )

    artifact = exp.build_experiment_artifact(
        _config(tmp_path),
        generator=generator,
        executor=executor,
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["row_status"] == "pilot_only_clean_no_passes"
    assert artifact["generated_code_row_clean"] is True
    assert artifact["pass_rate_if_computable"] == pytest.approx(0.0)
    assert artifact["headline_metric_claim_made"] is False
    assert all(row["row_status"] == "pilot_only_failed" for row in artifact["row_results"])
    assert len(scripts) == 3


def test_llama_cpp_generator_factory_returns_callable_without_loading(tmp_path: Path) -> None:
    """REQ-CODE-2889: the llama_cpp factory builds a generator lazily — no GGUF I/O at construction."""

    model_path = tmp_path / "ghost.gguf"
    # File deliberately missing; we should only see a failure if the closure runs.
    generator = exp.llama_cpp_generator(model_path=str(model_path))
    assert callable(generator)


def test_req_code_2889_blocks_when_exp2874_artifact_missing(tmp_path: Path) -> None:
    """REQ-CODE-2889: a missing Exp 2874 file is treated as not-clean and blocks the row."""

    _stage_repo(tmp_path)
    (tmp_path / exp.EXP2874_REL_PATH).unlink()

    artifact = exp.build_experiment_artifact(
        _config(tmp_path),
        generator=lambda *_args, **_kwargs: pytest.fail("generator should not run"),
        executor=lambda script, timeout_s: exp.ExecutionOutcome(passed=True),
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )

    assert artifact["honest_verdict"] == "blocked_exp2874_sota_runtime_not_clean"
    assert artifact["generated_code_row_clean"] is False


def test_req_code_2889_blocks_when_llama_cpp_gpu_offload_unavailable(tmp_path: Path) -> None:
    """REQ-CODE-2889: missing llama.cpp GPU offload or GPU inventory blocks the row."""

    _stage_repo(tmp_path)
    payload_path = tmp_path / exp.EXP2874_REL_PATH
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["llama_cpp_supports_gpu_offload"] = False
    _write_json(payload_path, payload)

    artifact = exp.build_experiment_artifact(
        _config(tmp_path),
        generator=lambda *_args, **_kwargs: pytest.fail("generator should not run"),
        executor=lambda script, timeout_s: exp.ExecutionOutcome(passed=True),
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
    )

    assert artifact["honest_verdict"] == "blocked_gpu_offload_unavailable"
    assert artifact["generated_code_row_clean"] is False


def test_llama_cpp_generator_drives_fake_llama_returns_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-CODE-2889: the llama_cpp_generator factory loads lazily then drives a real call path."""

    class FakeLlama:
        loaded: list[dict[str, Any]] = []

        def __init__(self, **kwargs: Any) -> None:
            FakeLlama.loaded.append(kwargs)
            self.kwargs = kwargs

        def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            return {
                "choices": [{"text": "```python\ndef add_one(x):\n    return x + 1\n```"}],
                "usage": {"completion_tokens": 11},
                "prompt_seen": prompt[:32],
                "kwargs": kwargs,
            }

    fake_module = type("FakeLlamaCpp", (), {"Llama": FakeLlama})
    monkeypatch.setitem(__import__("sys").modules, "llama_cpp", fake_module)

    generator = exp.llama_cpp_generator(model_path="/nowhere.gguf", main_gpu=1)
    row = {
        "canonical_code": "def add_one(x):\n    return x + 1\n",
        "prompt": "Add one.",
        "stable_id": "mbpp-fake",
        "test_imports": [],
        "tests": ["assert add_one(1) == 2"],
    }

    outcome_first = generator("mbpp", row, seed=11, max_tokens=32)
    outcome_second = generator("mbpp", row, seed=12, max_tokens=32)

    assert outcome_first.text.startswith("```python")
    assert outcome_first.tokens_generated == 11
    assert outcome_first.duration_s >= 0.0
    assert outcome_first.backend == "llama_cpp"
    assert outcome_first.error is None
    # Loaded exactly once across two calls — load is cached.
    assert len(FakeLlama.loaded) == 1
    assert FakeLlama.loaded[0]["main_gpu"] == 1
    assert outcome_second.text == outcome_first.text
