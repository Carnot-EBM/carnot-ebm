"""Tests for Exp 3086 Dafny/Z3 formal-feedback pilot.

Spec refs: REQ-VERIFY-3086,
           SCENARIO-VERIFY-3086,
           SCENARIO-VERIFY-3086-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import dafny_z3_formal_feedback_pilot_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"
SCRIPT_PATH = REPO_ROOT / "scripts" / exp.SCRIPT_FILENAME


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        self.value += 2.0
        return self.value


class FakeLlama:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = False

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["temperature"] == 0.0
        assert kwargs["seed"] == exp.DEFAULT_SEED
        fixture_id = _field(prompt, "Fixture")
        repairs = {
            "abs-identity-invalid": {"kind": "function_contract", "operation": "abs"},
            "increment-add-two-invalid": {
                "kind": "function_contract",
                "operation": "increment",
            },
            "sum-total-missing": {"kind": "record_sum", "a": 2, "b": 3, "total": 5},
            "vacuous-precondition": {
                "kind": "function_contract",
                "operation": "increment",
                "precondition": "0 <= x <= 3",
                "postcondition": "result == x + 1",
            },
            "weak-postcondition": {
                "kind": "function_contract",
                "operation": "increment",
                "precondition": "0 <= x <= 3",
                "postcondition": "result == x + 1",
            },
        }
        return {"choices": [{"text": json.dumps({"repair": repairs[fixture_id]})}]}

    def close(self) -> None:
        self.closed = True


class BadLlama(FakeLlama):
    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        return {"choices": [{"text": "not json"}]}


class RaisingLlama:
    def __init__(self, **_kwargs: Any) -> None:
        raise RuntimeError("load failed")


def _field(prompt: str, name: str) -> str:
    prefix = f"{name}: "
    for line in prompt.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    raise AssertionError(f"missing {name} in prompt")


def _model_path(tmp_path: Path) -> Path:
    path = tmp_path / "models" / "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"unit-test gguf placeholder")
    return path


def _resolve_one_model(path: Path) -> exp.ResolveGgufFn:
    def resolve(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
        assert preferred_quant == "Q4_K_M"
        if hf_id == "unsloth/gemma-4-26B-A4B-it-GGUF":
            return str(path)
        return None

    return resolve


def _command_resolver(
    dafny: str | None = None, z3: str | None = "/usr/bin/z3"
) -> exp.CommandResolver:
    def resolve(command: str) -> str | None:
        if command == "dafny":
            return dafny
        if command == "z3":
            return z3
        return None

    return resolve


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / exp.OUTPUT_REL_PATH,
        tests_run=("pytest focused",),
    )


def _successful_artifact(tmp_path: Path, llama_factory: Any = FakeLlama) -> dict[str, Any]:
    return exp.run_experiment(
        _config(tmp_path),
        command_resolver=_command_resolver(),
        resolve_gguf_func=_resolve_one_model(_model_path(tmp_path)),
        llama_factory=llama_factory,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
        cuda_probe_func=lambda: {"cuda_available": True, "gpu_count": 2},
        gpu_inventory_func=lambda: {
            "available": True,
            "gpus": [{"index": 0, "memory_free_mib": 1}],
        },
        python_environment_func=lambda: {"executable": "python-test"},
    )


def test_req_verify_3086_spec_and_script_anchor_exists() -> None:
    """REQ-VERIFY-3086: OpenSpec declares the pilot and required fields."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3086" in spec
    assert "SCENARIO-VERIFY-3086" in spec
    assert "SCENARIO-VERIFY-3086-BLOCKED" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    assert "formal_feedback_ready" in spec
    assert "blocked_formal_toolchain_missing" in spec
    assert SCRIPT_PATH.exists()


def test_scenario_verify_3086_z3_fallback_reports_formal_feedback_delta(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3086: Z3-only diagnostics guide exact model repairs."""
    artifact = _successful_artifact(tmp_path)
    saved = json.loads((tmp_path / exp.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["formal_feedback_ready"] is True
    assert artifact["formal_feedback_delta"] == pytest.approx(1.0)
    assert artifact["dafny_available"] is False
    assert artifact["z3_available"] is True
    assert artifact["vacuity_guard_passed"] is True
    assert artifact["guided_success_count"] == 5
    assert artifact["solver_only_success_count"] == 0
    assert artifact["exact_ground_truth_count"] == 5
    assert artifact["models_used"] == ["unsloth/gemma-4-26B-A4B-it-GGUF"]
    assert artifact["model_specs"][0]["model_path"].endswith(".gguf")
    assert artifact["legacy_smoke_only_used"] is False
    assert artifact["preconditions_checked"]["dafny_command"]["ok"] is False
    assert artifact["preconditions_checked"]["z3_command"]["ok"] is True
    assert artifact["preconditions_checked"]["selected_model_load"]["ok"] is True
    assert artifact["prompt_hash_count"] == len(artifact["prompt_hashes"]) == 5
    assert artifact["inference_substrate"]["kind"] == "live_llm_inference_plus_z3"
    assert artifact["inference_substrate"]["repo_commit"] == "test-commit"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_or_checks_run"] == ["pytest focused"]

    fixture_ids = {row["fixture_id"] for row in artifact["fixture_results"]}
    assert fixture_ids == {
        "abs-identity-invalid",
        "increment-add-two-invalid",
        "sum-total-missing",
        "vacuous-precondition",
        "weak-postcondition",
    }
    assert all(row["diagnostics"]["non_vacuous"] for row in artifact["fixture_results"])
    assert any(row["diagnostics"]["vacuity_detected"] for row in artifact["fixture_results"])
    assert any(
        row["diagnostics"]["weak_postcondition_detected"] for row in artifact["fixture_results"]
    )
    assert all(row["guided_validation"]["valid"] for row in artifact["fixture_results"])

    exp.validate_artifact(artifact)


def test_scenario_verify_3086_blocks_when_formal_toolchain_missing(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3086-BLOCKED: no Dafny and no Z3 fails before inference."""
    artifact = exp.run_experiment(
        _config(tmp_path),
        command_resolver=_command_resolver(z3=None),
        resolve_gguf_func=_resolve_one_model(_model_path(tmp_path)),
        llama_factory=FakeLlama,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
        cuda_probe_func=lambda: {"cuda_available": True, "gpu_count": 2},
        gpu_inventory_func=lambda: {"available": True, "gpus": []},
        python_environment_func=lambda: {"executable": "python-test"},
    )

    assert artifact["formal_feedback_ready"] is False
    assert artifact["dafny_available"] is False
    assert artifact["z3_available"] is False
    assert artifact["models_used"] == []
    assert artifact["model_specs"] == []
    assert artifact["prompt_hashes"] == []
    assert artifact["guided_success_count"] == 0
    assert artifact["solver_only_success_count"] == 0
    assert artifact["preconditions_checked"]["formal_toolchain"]["ok"] is False
    assert artifact["honest_verdict"].startswith("blocked_formal_toolchain_missing")
    exp.validate_artifact(artifact)


def test_req_verify_3086_exact_diagnostics_detect_vacuity_and_weak_posts() -> None:
    """REQ-VERIFY-3086: formal diagnostics flag vacuity and weak postconditions."""
    fixtures = exp.default_fixtures()
    diagnostics = {fx.fixture_id: exp.diagnose_fixture(fx) for fx in fixtures}

    assert diagnostics["abs-identity-invalid"]["counterexample"]
    assert diagnostics["sum-total-missing"]["missing_fields"] == ["total"]
    assert diagnostics["vacuous-precondition"]["vacuity_detected"] is True
    assert diagnostics["weak-postcondition"]["weak_postcondition_detected"] is True
    assert exp.vacuity_guard_passed(fixtures, list(diagnostics.values())) is True

    repaired = {
        "kind": "function_contract",
        "operation": "increment",
        "precondition": "0 <= x <= 3",
        "postcondition": "result == x + 1",
    }
    weak_fixture = next(fx for fx in fixtures if fx.fixture_id == "weak-postcondition")
    assert exp.validate_candidate(weak_fixture, repaired)["valid"] is True
    assert exp.validate_candidate(weak_fixture, weak_fixture.candidate)["valid"] is False


def test_req_verify_3086_model_load_and_bad_output_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-3086: load failures and unparseable model output are not promoted."""
    missing_cuda = exp.run_experiment(
        _config(tmp_path),
        command_resolver=_command_resolver(),
        resolve_gguf_func=_resolve_one_model(_model_path(tmp_path)),
        llama_factory=FakeLlama,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
        cuda_probe_func=lambda: {"cuda_available": False, "gpu_count": 0},
        gpu_inventory_func=lambda: {"available": False, "gpus": []},
        python_environment_func=lambda: {"executable": "python-test"},
    )
    assert missing_cuda["runtime_blocker"] == "cuda_gpu_unavailable"
    assert missing_cuda["formal_feedback_ready"] is False
    exp.validate_artifact(missing_cuda)

    load_failed = exp.run_experiment(
        _config(tmp_path),
        command_resolver=_command_resolver(),
        resolve_gguf_func=_resolve_one_model(_model_path(tmp_path)),
        llama_factory=RaisingLlama,
        monotonic=FakeClock(),
        repo_commit_func=lambda _: "test-commit",
        cuda_probe_func=lambda: {"cuda_available": True, "gpu_count": 2},
        gpu_inventory_func=lambda: {"available": True, "gpus": []},
        python_environment_func=lambda: {"executable": "python-test"},
    )
    assert load_failed["formal_feedback_ready"] is False
    assert load_failed["runtime_blocker"].startswith("model_load_failed:")
    assert load_failed["preconditions_checked"]["selected_model_load"]["ok"] is False
    assert load_failed["honest_verdict"].startswith("blocked_sota_or_model_precondition_failed")
    exp.validate_artifact(load_failed)

    bad_output = _successful_artifact(tmp_path, llama_factory=BadLlama)
    assert bad_output["formal_feedback_ready"] is False
    assert bad_output["guided_success_count"] == 0
    assert bad_output["formal_feedback_delta"] == pytest.approx(0.0)
    assert bad_output["honest_verdict"].startswith("complete:")
    exp.validate_artifact(bad_output)


def test_req_verify_3086_artifact_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3086: terminal artifacts cannot overstate readiness."""
    good = _successful_artifact(tmp_path)

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="legacy"):
        exp.validate_artifact(good | {"legacy_smoke_only_used": True})
    with pytest.raises(ValueError, match="model_specs"):
        exp.validate_artifact(good | {"model_specs": []})
    with pytest.raises(ValueError, match="prompt_hashes"):
        exp.validate_artifact(good | {"prompt_hashes": []})
    with pytest.raises(ValueError, match="exact_ground_truth_count"):
        exp.validate_artifact(good | {"exact_ground_truth_count": 0})
    with pytest.raises(ValueError, match="vacuity_guard_passed"):
        exp.validate_artifact(good | {"vacuity_guard_passed": False})
    with pytest.raises(ValueError, match="formal_feedback_delta"):
        exp.validate_artifact(good | {"formal_feedback_delta": 0.0})
    with pytest.raises(ValueError, match="guided_success_count"):
        exp.validate_artifact(
            good
            | {
                "formal_feedback_delta": 1.0,
                "guided_success_count": 1,
                "solver_only_success_count": 1,
            }
        )
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(good | {"honest_verdict": "ready"})
    with pytest.raises(ValueError, match="blocked_formal_toolchain_missing"):
        exp.validate_artifact(
            good
            | {
                "formal_feedback_ready": False,
                "runtime_blocker": "formal_toolchain_missing",
                "honest_verdict": "complete: wrong blocked prefix",
            }
        )
    with pytest.raises(ValueError, match="model precondition"):
        exp.validate_artifact(
            good
            | {
                "formal_feedback_ready": False,
                "runtime_blocker": "cuda_gpu_unavailable",
                "honest_verdict": "complete: wrong blocked prefix",
            }
        )
    with pytest.raises(ValueError, match="terminal success prefix"):
        exp.validate_artifact(
            good
            | {
                "formal_feedback_ready": False,
                "runtime_blocker": None,
                "honest_verdict": "ready",
            }
        )


def test_req_verify_3086_parsing_and_helper_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3086: parser, hashing, and precondition helpers are deterministic."""
    config = exp.ExperimentConfig(
        repo_root=tmp_path,
        decode_config={"max_tokens": 3},
        load_config={"n_batch": 8},
    )
    assert config.effective_decode_config()["max_tokens"] == 3
    assert config.effective_load_config(1)["n_batch"] == 8
    assert config.effective_load_config(1)["main_gpu"] == 1

    parsed = exp.parse_repair_response('```json\n{"repair":{"kind":"record_sum","total":5}}\n```')
    assert parsed["repair"]["total"] == 5
    assert (
        exp.parse_repair_response('{"kind":"record_sum","total":5}')["repair"]["kind"]
        == "record_sum"
    )
    assert exp.parse_repair_response("no json")["parse_error"]
    assert exp._extract_text({"choices": []}) == ""
    assert exp._sha256_text("x") == exp._sha256_text("x")
    assert exp._relative_path(tmp_path, tmp_path / "out.json") == "out.json"
    assert exp._relative_path(tmp_path, Path("/outside/out.json")) == "/outside/out.json"
    assert exp._model_family("unsloth/Qwen3.6-35B-A3B-GGUF") == "qwen"
    assert exp._model_family("vendor/Other-GGUF") == "vendor"
    assert exp._selected_model({}) is None
    assert (
        exp._first_precondition_failure(
            {
                "formal_toolchain": {"ok": True},
                "cuda_gpu": {"ok": False},
                "gguf_cache": {"ok": True},
                "selected_model_load": {"ok": True},
            }
        )
        == "cuda_gpu_unavailable"
    )
    assert (
        exp._first_precondition_failure(
            {
                "formal_toolchain": {"ok": False},
                "cuda_gpu": {"ok": True},
                "gguf_cache": {"ok": True},
            }
        )
        == "formal_toolchain_missing"
    )
    assert (
        exp._first_precondition_failure(
            {
                "formal_toolchain": {"ok": True},
                "cuda_gpu": {"ok": True},
                "gguf_cache": {"ok": False},
            }
        )
        == "mandated_gguf_unavailable"
    )

    large = tmp_path / "large.gguf"
    large.write_bytes(b"a" * 2048)
    evidence = exp._file_evidence(str(large), full_limit_bytes=1)
    assert evidence["checksum_feasibility"]["method"] == "bounded_head_tail_sha256"
    assert evidence["checksum_feasibility"]["full_sha256_feasible"] is False

    small = tmp_path / "small.gguf"
    small.write_bytes(b"small")
    assert exp._file_evidence(str(small))["checksum_feasibility"]["method"] == "full_sha256"
    assert exp._file_evidence(str(tmp_path / "missing.gguf"))["checksum_feasibility"]["method"] == (
        "missing_file"
    )

    fixtures = exp.default_fixtures()
    record_fixture = next(fx for fx in fixtures if fx.fixture_id == "sum-total-missing")
    wrong_total = {"kind": "record_sum", "a": 2, "b": 3, "total": 7}
    wrong_total_diag = exp.diagnose_fixture(record_fixture, wrong_total)
    assert wrong_total_diag["counterexample"]["expected_total"] == 5

    increment_fixture = next(fx for fx in fixtures if fx.fixture_id == "increment-add-two-invalid")
    bad_postcondition = {
        "kind": "function_contract",
        "operation": "increment",
        "precondition": "true",
        "postcondition": "result == x + 1 and x >0",
    }
    assert exp.diagnose_fixture(increment_fixture, bad_postcondition)["postcondition_violation"]
    with pytest.raises(ValueError, match="unsupported precondition"):
        exp.diagnose_fixture(
            increment_fixture,
            bad_postcondition | {"precondition": "x != 0"},
        )
    with pytest.raises(ValueError, match="unsupported postcondition"):
        exp.diagnose_fixture(
            increment_fixture,
            bad_postcondition | {"postcondition": "result >= x"},
        )
    with pytest.raises(ValueError, match="unsupported operation"):
        exp.diagnose_fixture(increment_fixture, bad_postcondition | {"operation": "triple"})
    with pytest.raises(ValueError, match="unsupported operation"):
        exp._eval_operation("triple", 1)
    with pytest.raises(RuntimeError, match="z3 Python module unavailable"):
        original_z3 = exp._z3
        try:
            exp._z3 = None
            exp._require_z3()
        finally:
            exp._z3 = original_z3
