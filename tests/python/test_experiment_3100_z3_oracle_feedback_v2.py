"""Tests for Exp 3100 Z3/test-oracle formal-feedback v2.

Spec refs: REQ-VERIFY-3100,
           SCENARIO-VERIFY-3100,
           SCENARIO-VERIFY-3100-BLOCKED-HEADLINE.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import z3_oracle_feedback_v2 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"


class FakeClock:
    def __init__(self) -> None:
        self.value = 10.0

    def __call__(self) -> float:
        self.value += 1.25
        return self.value


class FakeLlama:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.closed = False

    def __call__(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["temperature"] == 0.0
        fixture_id = _field(prompt, "Fixture")
        repairs = {
            "repair-json": {
                "candidate": json.dumps({"mode": "bounded", "limit": 2}),
            },
            "repair-smt": {
                "candidate_assignment": {"rx": 10, "ry": 1},
            },
            "repair-py": {
                "candidate_assertion": "assert ((7 * 2) - 2) == 12",
            },
        }
        return {"choices": [{"text": json.dumps({"repair": repairs[fixture_id]})}]}

    def close(self) -> None:
        self.closed = True


class ShouldNotLoadLlama:
    def __init__(self, **_kwargs: Any) -> None:
        raise AssertionError("llama should not load when cached_sota_pair is unavailable")


class RaisingLlama:
    def __init__(self, **_kwargs: Any) -> None:
        raise RuntimeError("load failed")


def _field(prompt: str, name: str) -> str:
    prefix = f"{name}: "
    for line in prompt.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    raise AssertionError(f"missing {name} in prompt")


def _write_json(root: Path, rel_path: Path | str, payload: dict[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(root: Path, rel_path: Path | str, rows: list[dict[str, Any]]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _model_file(root: Path, name: str) -> Path:
    path = root / "models" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"unit test model {name}".encode())
    return path


def _manifest_row(
    *,
    fixture_id: str,
    perturbation_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "carnot.exact_fixture_eval_manifest.v1",
        "source_fixture_id": fixture_id,
        "source_prompt_payload_sha256": f"hash-{fixture_id}",
        "task_family": "repairable_invalid_candidates",
        "task_axis": "repairing",
        "perturbation_type": perturbation_type,
        "expected_answer": "REPAIRABLE",
        "solver_label": "repairable",
        "label_source": "unit_exact_authority",
        "exact_label_kind": "repairability",
        "leakage_safe_prompt_payload": payload,
        "verifier_target": {"expected_action": "reject", "expected_reject": True},
        "repair_target": {
            "applicable": True,
            "candidate_valid": False,
            "repairable": True,
            "failure_kind": "unit_failure",
            "repair_validation": "passed",
        },
        "evaluation_tasks": ["formal_feedback_v2"],
        "stratum_key": f"repairable_invalid_candidates|{perturbation_type}",
    }


def _manifest_rows() -> list[dict[str, Any]]:
    return [
        _manifest_row(
            fixture_id="repair-json",
            perturbation_type="json_syntax_repair",
            payload={
                "task": "Repair the candidate so the object parses and preserves fields.",
                "candidate": '{"mode": "bounded" "limit": 2}',
                "required_fields": ["mode", "limit"],
            },
        ),
        _manifest_row(
            fixture_id="repair-smt",
            perturbation_type="numeric_bound_repair",
            payload={
                "task": "Repair the candidate integer assignment while preserving variable names.",
                "candidate_assignment": {"rx": 2, "ry": 1},
                "constraints": ["rx >= 0", "ry >= 0", "rx + ry == 11"],
                "variables": ["rx", "ry"],
            },
        ),
        _manifest_row(
            fixture_id="repair-py",
            perturbation_type="python_assertion_repair",
            payload={
                "task": "Repair the candidate assertion while keeping the expression fixed.",
                "candidate_assertion": "assert ((7 * 2) - 2) == 13",
                "expression": "(7 * 2) - 2",
            },
        ),
    ]


def _write_protocol(root: Path, rows: list[dict[str, Any]] | None = None) -> None:
    active_rows = rows or _manifest_rows()
    _write_jsonl(root, exp.STRATIFIED_MANIFEST_REL_PATH, active_rows)
    _write_json(
        root,
        exp.EXP3097_REL_PATH,
        {
            "eval_protocol_ready": True,
            "minimum_live_eval_count": 48,
            "usable_fixture_count": len(active_rows),
            "stratified_eval_manifest_path": exp.STRATIFIED_MANIFEST_REL_PATH.as_posix(),
            "downstream_usage": {
                "formal_feedback_v2": {
                    "available_repair_fixtures": len(active_rows),
                    "ready_for_headline": True,
                }
            },
        },
    )


def _resolve_from(paths: dict[str, Path]) -> exp.ResolveGgufFn:
    def resolve(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
        assert preferred_quant == "Q4_K_M"
        path = paths.get(hf_id)
        return str(path) if path is not None else None

    return resolve


def _cached_pair(paths: dict[str, Path]) -> exp.CachedPairFn:
    def cached_pair(*, gpu_indices: tuple[int, int] = (0, 1), preferred_quant: str = "Q4_K_M"):
        assert preferred_quant == "Q4_K_M"
        return [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "gpu": gpu_indices[0],
                "model_path": str(paths["unsloth/Qwen3.6-35B-A3B-GGUF"]),
            },
            {
                "name": "Gemma4-26B-A4B-it",
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "gpu": gpu_indices[1],
                "model_path": str(paths["unsloth/gemma-4-26B-A4B-it-GGUF"]),
            },
        ]

    return cached_pair


def _command_resolver(dafny: str | None = None, z3: str | None = "/usr/bin/z3"):
    def resolve(command: str) -> str | None:
        if command == "dafny":
            return dafny
        if command == "z3":
            return z3
        return None

    return resolve


def _config(tmp_path: Path) -> exp.FeedbackConfig:
    return exp.FeedbackConfig(
        repo_root=tmp_path,
        output_path=tmp_path / exp.OUTPUT_REL_PATH,
        panel_size=3,
        clock=FakeClock(),
        tests_run=("pytest focused",),
    )


def test_req_verify_3100_spec_anchor_and_required_fields() -> None:
    """REQ-VERIFY-3100: OpenSpec declares the v2 pilot and artifact fields."""
    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3100" in spec
    assert "SCENARIO-VERIFY-3100" in spec
    assert "SCENARIO-VERIFY-3100-BLOCKED-HEADLINE" in spec
    assert exp.OUTPUT_REL_PATH.as_posix() in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_verify_3100_z3_oracle_feedback_runs_without_dafny(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3100: Z3/test-oracle feedback beats solver-only fallback."""
    _write_protocol(tmp_path)
    paths = {
        "unsloth/Qwen3.6-35B-A3B-GGUF": _model_file(tmp_path, "qwen.gguf"),
        "unsloth/gemma-4-26B-A4B-it-GGUF": _model_file(tmp_path, "gemma26.gguf"),
    }

    artifact = exp.run_experiment(
        _config(tmp_path),
        command_resolver=_command_resolver(),
        resolve_gguf_func=_resolve_from(paths),
        cached_pair_func=_cached_pair(paths),
        llama_factory=FakeLlama,
        repo_commit_func=lambda _: "test-commit",
        python_environment_func=lambda: {"executable": "python-test"},
    )
    saved = json.loads((tmp_path / exp.OUTPUT_REL_PATH).read_text(encoding="utf-8"))

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["formal_feedback_v2_ready"] is True
    assert artifact["dafny_available"] is False
    assert artifact["z3_available"] is True
    assert artifact["exact_ground_truth_count"] == 3
    assert artifact["guided_success_count"] == 3
    assert artifact["solver_only_success_count"] == 1
    assert artifact["no_feedback_success_count"] == 0
    assert artifact["formal_feedback_delta"] == pytest.approx(2 / 3)
    assert artifact["vacuity_guard_passed"] is True
    assert artifact["test_oracle_count"] >= 9
    assert artifact["guided_evaluation_feasible"] is True
    assert artifact["headline_blocked_reason"] is None
    assert artifact["model_specs"][0]["hf_id"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
    assert artifact["model_specs"][0]["cached"] is True
    assert artifact["inference_substrate"]["kind"] == "live_llm_inference_plus_z3_test_oracle"
    assert artifact["inference_substrate"]["repo_commit"] == "test-commit"
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_or_checks_run"] == ["pytest focused"]
    assert all(row["guided_validation"]["valid"] for row in artifact["fixture_results"])
    assert any(row["solver_only_validation"]["valid"] for row in artifact["fixture_results"])
    assert all(
        row["empty_repair_validation"]["valid"] is False for row in artifact["fixture_results"]
    )

    exp.validate_artifact(artifact)


def test_scenario_verify_3100_blocked_headline_when_cached_pair_missing(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-3100-BLOCKED-HEADLINE: missing SOTA pair does not promote."""
    _write_protocol(tmp_path)
    one_model = {
        "unsloth/gemma-4-26B-A4B-it-GGUF": _model_file(tmp_path, "gemma26.gguf"),
    }

    artifact = exp.run_experiment(
        _config(tmp_path),
        command_resolver=_command_resolver(),
        resolve_gguf_func=_resolve_from(one_model),
        cached_pair_func=lambda **_: None,
        llama_factory=ShouldNotLoadLlama,
        repo_commit_func=lambda _: "test-commit",
        python_environment_func=lambda: {"executable": "python-test"},
    )

    assert artifact["formal_feedback_v2_ready"] is False
    assert artifact["headline_blocked_reason"] == "cached_sota_pair_unavailable"
    assert artifact["guided_evaluation_feasible"] is False
    assert artifact["guided_success_count"] == 0
    assert artifact["solver_only_success_count"] == 1
    assert artifact["exact_ground_truth_count"] == 3
    assert artifact["model_specs"][1]["cached"] is True
    assert artifact["model_specs"][0]["cached"] is False
    assert artifact["inference_substrate"]["live_llm_inference"] is False
    assert artifact["honest_verdict"].startswith("complete_blocked_headline:")
    exp.validate_artifact(artifact)


def test_req_verify_3100_oracles_and_vacuity_guards_reject_empty_repairs() -> None:
    """REQ-VERIFY-3100: empty proofs/specifications cannot pass exact oracles."""
    rows = _manifest_rows()

    json_empty = exp.validate_candidate(rows[0], {}, z3_module=exp._z3)
    smt_empty = exp.validate_candidate(rows[1], {}, z3_module=exp._z3)
    py_empty = exp.validate_candidate(rows[2], {}, z3_module=exp._z3)

    assert json_empty["valid"] is False
    assert smt_empty["valid"] is False
    assert py_empty["valid"] is False
    assert exp.vacuity_guard_passed(
        [
            {"empty_repair_validation": json_empty},
            {"empty_repair_validation": smt_empty},
            {"empty_repair_validation": py_empty},
        ]
    )

    assert exp.solver_only_repair(rows[1], z3_module=exp._z3)["candidate_assignment"] == {
        "rx": 10,
        "ry": 1,
    }
    assert exp.validate_candidate(
        rows[2],
        {"candidate_assertion": "assert ((7 * 2) - 2) == 12"},
        z3_module=exp._z3,
    )["valid"]
    assert (
        exp.validate_candidate(
            rows[2],
            {"candidate_assertion": "assert ((8 * 2) - 2) == 14"},
            z3_module=exp._z3,
        )["valid"]
        is False
    )


def test_req_verify_3100_missing_z3_or_protocol_writes_blocked_artifact(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-3100: blocked preconditions still write a terminal artifact."""
    artifact = exp.run_experiment(
        _config(tmp_path),
        command_resolver=_command_resolver(z3=None),
        resolve_gguf_func=_resolve_from({}),
        cached_pair_func=lambda **_: None,
        llama_factory=ShouldNotLoadLlama,
        z3_module=None,
        repo_commit_func=lambda _: "test-commit",
        python_environment_func=lambda: {"executable": "python-test"},
    )

    assert (tmp_path / exp.OUTPUT_REL_PATH).is_file()
    assert artifact["formal_feedback_v2_ready"] is False
    assert artifact["z3_available"] is False
    assert artifact["exact_ground_truth_count"] == 0
    assert artifact["test_oracle_count"] == 0
    assert artifact["honest_verdict"].startswith("blocked_z3_or_protocol_precondition_failed:")
    exp.validate_artifact(artifact)


def test_req_verify_3100_artifact_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-3100: validation prevents overstated formal-feedback lift."""
    _write_protocol(tmp_path)
    paths = {
        "unsloth/Qwen3.6-35B-A3B-GGUF": _model_file(tmp_path, "qwen.gguf"),
        "unsloth/gemma-4-26B-A4B-it-GGUF": _model_file(tmp_path, "gemma26.gguf"),
    }
    good = exp.run_experiment(
        _config(tmp_path),
        command_resolver=_command_resolver(),
        resolve_gguf_func=_resolve_from(paths),
        cached_pair_func=_cached_pair(paths),
        llama_factory=FakeLlama,
        repo_commit_func=lambda _: "test-commit",
        python_environment_func=lambda: {"executable": "python-test"},
    )

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact({"honest_verdict": "complete: incomplete"})
    with pytest.raises(ValueError, match="model_specs"):
        exp.validate_artifact(good | {"model_specs": []})
    with pytest.raises(ValueError, match="exact_ground_truth_count"):
        exp.validate_artifact(good | {"exact_ground_truth_count": 0})
    with pytest.raises(ValueError, match="vacuity_guard_passed"):
        exp.validate_artifact(good | {"vacuity_guard_passed": False})
    with pytest.raises(ValueError, match="formal_feedback_delta"):
        exp.validate_artifact(good | {"formal_feedback_delta": 0.0})
    with pytest.raises(ValueError, match="guided_success_count"):
        exp.validate_artifact(good | {"guided_success_count": 1, "solver_only_success_count": 1})
    with pytest.raises(ValueError, match="live guided"):
        exp.validate_artifact(good | {"guided_evaluation_feasible": False})
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(good | {"honest_verdict": "ready"})
    with pytest.raises(ValueError, match="blocked precondition"):
        exp.validate_artifact(
            good
            | {
                "formal_feedback_v2_ready": False,
                "runtime_blocker": "z3_or_protocol_precondition_failed",
                "honest_verdict": "complete: wrong",
            }
        )
    with pytest.raises(ValueError, match="complete_blocked_headline"):
        exp.validate_artifact(
            good
            | {
                "formal_feedback_v2_ready": False,
                "headline_blocked_reason": "cached_sota_pair_unavailable",
                "honest_verdict": "complete: wrong",
            }
        )
    with pytest.raises(ValueError, match="terminal success prefix"):
        exp.validate_artifact(
            good
            | {
                "formal_feedback_v2_ready": False,
                "headline_blocked_reason": None,
                "runtime_blocker": None,
                "honest_verdict": "ready",
            }
        )


def test_req_verify_3100_helper_edges_and_load_failure(tmp_path: Path) -> None:
    """REQ-VERIFY-3100: helper edge cases fail closed without promoted lift."""
    _write_protocol(tmp_path)
    paths = {
        "unsloth/Qwen3.6-35B-A3B-GGUF": _model_file(tmp_path, "qwen.gguf"),
        "unsloth/gemma-4-26B-A4B-it-GGUF": _model_file(tmp_path, "gemma26.gguf"),
    }
    load_failed = exp.run_experiment(
        _config(tmp_path),
        command_resolver=_command_resolver(),
        resolve_gguf_func=_resolve_from(paths),
        cached_pair_func=_cached_pair(paths),
        llama_factory=RaisingLlama,
        repo_commit_func=lambda _: "test-commit",
        python_environment_func=lambda: {"executable": "python-test"},
    )
    assert load_failed["formal_feedback_v2_ready"] is False
    assert load_failed["headline_blocked_reason"].startswith("model_load_failed:")
    assert load_failed["honest_verdict"].startswith("complete_blocked_headline:")

    assert (
        exp.first_runtime_blocker(z3_available=True, exp3097={}, selected_rows=_manifest_rows())
        == "z3_or_protocol_precondition_failed"
    )
    assert (
        exp.first_runtime_blocker(
            z3_available=True,
            exp3097={"eval_protocol_ready": True},
            selected_rows=[],
        )
        == "z3_or_protocol_precondition_failed"
    )
    assert exp.honest_verdict(
        ready=False,
        runtime_blocker=None,
        headline_blocked_reason=None,
        z3_available=True,
        dafny_available=False,
        guided=1,
        solver_only=1,
        exact_count=3,
    ).startswith("complete:")

    def no_arg_cached_pair() -> list[dict[str, Any]]:
        return []

    assert exp.probe_cached_pair(no_arg_cached_pair, "Q4_K_M")["available"] is False

    def broken_cached_pair(**_kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("cache probe failed")

    assert exp.probe_cached_pair(broken_cached_pair, "Q4_K_M")["error"].startswith("RuntimeError:")
    assert exp.parse_guided_repair_response("not json")["valid_parse"] is False

    rows = _manifest_rows()
    assert (
        exp.solver_only_repair(
            rows[1]
            | {
                "leakage_safe_prompt_payload": rows[1]["leakage_safe_prompt_payload"]
                | {"constraints": ["rx >= 0"]}
            },
            z3_module=exp._z3,
        )
        is None
    )
    assert (
        exp.validate_candidate(
            rows[0],
            {"candidate": json.dumps({"mode": "bounded"})},
            z3_module=exp._z3,
        )["failure_reason"]
        == "missing_required_fields"
    )
    assert (
        exp.validate_candidate(
            rows[1],
            {"candidate_assignment": {"rx": 10, "ry": 1}},
            z3_module=None,
        )["failure_reason"]
        == "z3_python_unavailable"
    )
    assert (
        exp.validate_candidate(
            rows[1],
            {"candidate_assignment": "not a mapping"},
            z3_module=exp._z3,
        )["failure_reason"]
        == "empty_or_incomplete_assignment"
    )
    assert (
        exp.validate_candidate(
            rows[2],
            {"candidate_assertion": "assert ("},
            z3_module=exp._z3,
        )["failure_reason"]
        == "syntax_error"
    )
    assert (
        exp.validate_candidate(
            rows[2],
            {"candidate_assertion": "x = 1"},
            z3_module=exp._z3,
        )["failure_reason"]
        == "missing_assert_statement"
    )
    assert (
        exp.validate_candidate(
            rows[2],
            {"candidate_assertion": "assert 12"},
            z3_module=exp._z3,
        )["failure_reason"]
        == "unsupported_assertion_shape"
    )
    assert (
        exp.validate_candidate(
            rows[2],
            {"candidate_assertion": "assert ((8 * 2) - 4) == 12"},
            z3_module=exp._z3,
        )["failure_reason"]
        == "expression_changed"
    )

    assert exp._sum_constraint_target([], ["rx"]) is None
    assert exp._sum_constraint_target(["rx >= 0"], ["rx", "ry"]) is None
    symbols = {"rx": exp._z3.Int("rx"), "ry": exp._z3.Int("ry")}
    assert str(exp._constraint_to_z3("rx <= 3", symbols, exp._z3)) == "rx <= 3"
    assert str(exp._constraint_to_z3("rx == 3", symbols, exp._z3)) == "rx == 3"
    assert str(exp._constraint_to_z3("unsupported", symbols, exp._z3)) == "False"
    assert exp._safe_eval_ast(__import__("ast").parse("-3", mode="eval").body) == -3
    assert exp._safe_eval_ast(__import__("ast").parse("1 + 2", mode="eval").body) == 3
    with pytest.raises(ValueError, match="unsupported expression"):
        exp._safe_eval_ast(__import__("ast").parse("2 / 1", mode="eval").body)

    assert exp.extract_text({"choices": []}) == ""
    assert exp.file_evidence(str(tmp_path / "missing.gguf"))["checksum_feasibility"]["method"] == (
        "missing_file"
    )
    assert exp._relative_path(tmp_path, Path("/outside/file.json")) == "/outside/file.json"
