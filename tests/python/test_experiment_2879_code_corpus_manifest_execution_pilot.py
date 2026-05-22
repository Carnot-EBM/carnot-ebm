"""Tests for Exp 2879 MBPP/HumanEval manifest-only execution pilot.

Spec: REQ-CODE-2879, SCENARIO-CODE-2879.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import code_corpus_manifest_execution_pilot as exp


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


def _write_manifest_contract(tmp_path: Path) -> tuple[Path, Path]:
    mbpp_path = tmp_path / "data" / "eval_manifests" / "mbpp_20260522.jsonl"
    humaneval_path = tmp_path / "data" / "eval_manifests" / "humaneval_20260522.jsonl"
    _write_jsonl(
        mbpp_path,
        [
            {
                "canonical_code": "",
                "dataset": "MBPP",
                "prompt": "Missing code row should be skipped.",
                "stable_id": "mbpp-skip",
                "tests": ["assert missing() is None"],
            },
            {
                "canonical_code": "def add_one(x):\n    return x + 1\n",
                "dataset": "MBPP",
                "prompt": "Write a function that adds one.",
                "stable_id": "mbpp-eligible",
                "test_imports": [],
                "tests": ["assert add_one(1) == 2", "assert add_one(-1) == 0"],
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
    return mbpp_path, humaneval_path


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
        tests_run=("focused-pytest",),
        started_at=10.0,
        clock=lambda: 12.5,
    )


def test_scenario_code_2879_executes_deterministic_manifest_rows(tmp_path: Path) -> None:
    """SCENARIO-CODE-2879: canonical MBPP/HumanEval rows run through sandbox wrapper."""

    mbpp_path, humaneval_path = _write_manifest_contract(tmp_path)
    scripts: list[str] = []

    def fake_executor(script: str, timeout_s: float) -> exp.ExecutionOutcome:
        scripts.append(script)
        assert timeout_s == pytest.approx(10.0)
        return exp.ExecutionOutcome(passed=True)

    artifact = exp.write_experiment_artifact(
        _config(tmp_path),
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
        executor=fake_executor,
    )
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text())

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["code_manifest_pilot_ready"] is True
    assert artifact["deterministic_execution_used"] is True
    assert artifact["sandbox_status"] == "available: runsc"
    assert artifact["n_mbpp_rows"] == 1
    assert artifact["n_humaneval_rows"] == 1
    assert artifact["headline_metric_claim_made"] is False
    assert artifact["tests_run"] == ["focused-pytest"]
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["manifest_paths"] == {
        "mbpp": str(mbpp_path),
        "humaneval": str(humaneval_path),
    }
    assert artifact["source_artifacts"] == [
        "results/experiment_2863_eval_manifest_contract_v2.json",
        "results/experiment_2865_cross_corpus_matrix_v5.json",
        "data/eval_manifests/mbpp_20260522.jsonl",
        "data/eval_manifests/humaneval_20260522.jsonl",
    ]
    assert "first eligible row" in artifact["selection_rule"]
    assert set(artifact["selection_checksums"]) == {"mbpp-eligible", "HumanEval/eligible"}

    by_id = {row["stable_id"]: row for row in artifact["pilot_rows"]}
    assert by_id["mbpp-eligible"]["passed"] is True
    assert by_id["mbpp-eligible"]["n_tests"] == 2
    assert by_id["mbpp-eligible"]["verifier_feature_coverage"]["mbpp_assert_tests"] is True
    assert by_id["HumanEval/eligible"]["passed"] is True
    assert by_id["HumanEval/eligible"]["n_tests"] == 1
    assert by_id["HumanEval/eligible"]["verifier_feature_coverage"]["official_check"] is True
    assert all(row["verifier_feature_coverage"]["no_llm_generation"] for row in artifact["pilot_rows"])
    assert any("assert add_one(1) == 2" in script for script in scripts)
    assert any("check(double)" in script for script in scripts)


def test_req_code_2879_blocks_before_execution_when_sandbox_missing(tmp_path: Path) -> None:
    """REQ-CODE-2879: missing gVisor/runsc writes blocked_sandbox and no pilot rows."""

    _write_manifest_contract(tmp_path)

    def forbidden_executor(script: str, timeout_s: float) -> exp.ExecutionOutcome:
        raise AssertionError(f"execution should be blocked, got {script[:40]} {timeout_s}")

    artifact = exp.build_experiment_artifact(
        _config(tmp_path),
        sandbox_status_provider=lambda: {"available": False, "runtime": "none"},
        executor=forbidden_executor,
    )

    assert artifact["honest_verdict"] == "blocked_sandbox"
    assert artifact["code_manifest_pilot_ready"] is False
    assert artifact["deterministic_execution_used"] is False
    assert artifact["sandbox_status"] == "blocked_sandbox: runsc unavailable"
    assert artifact["pilot_rows"] == []
    assert artifact["n_mbpp_rows"] == 0
    assert artifact["n_humaneval_rows"] == 0


def test_req_code_2879_blocks_when_contract_checksums_do_not_verify(tmp_path: Path) -> None:
    """REQ-CODE-2879: manifest checksum mismatch prevents pilot execution."""

    _write_manifest_contract(tmp_path)
    payload_path = tmp_path / exp.MANIFEST_CONTRACT_REL_PATH
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["resolved_manifest_sha256"]["mbpp"] = "0" * 64
    _write_json(payload_path, payload)

    artifact = exp.build_experiment_artifact(
        _config(tmp_path),
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
        executor=lambda script, timeout_s: exp.ExecutionOutcome(passed=True),
    )

    assert artifact["honest_verdict"] == "blocked_manifest_contract"
    assert artifact["code_manifest_pilot_ready"] is False
    assert artifact["pilot_rows"] == []
    assert artifact["manifest_contract_ready"] is False
    assert artifact["manifest_checksum_verified"] == {"mbpp": False, "humaneval": True}


def test_req_code_2879_blocks_when_no_eligible_code_rows_exist(tmp_path: Path) -> None:
    """REQ-CODE-2879: manifests without canonical code/tests do not produce inferred rows."""

    mbpp_path, _ = _write_manifest_contract(tmp_path)
    _write_jsonl(
        mbpp_path,
        [
            {
                "canonical_code": "",
                "dataset": "MBPP",
                "prompt": "No eligible MBPP row.",
                "stable_id": "mbpp-empty",
                "tests": [],
            }
        ],
    )
    payload_path = tmp_path / exp.MANIFEST_CONTRACT_REL_PATH
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["resolved_manifest_sha256"]["mbpp"] = _sha256(mbpp_path)
    _write_json(payload_path, payload)

    artifact = exp.build_experiment_artifact(
        _config(tmp_path),
        sandbox_status_provider=lambda: {"available": True, "runtime": "runsc"},
        executor=lambda script, timeout_s: exp.ExecutionOutcome(passed=True),
    )

    assert artifact["honest_verdict"] == "blocked_no_eligible_code_rows"
    assert artifact["code_manifest_pilot_ready"] is False
    assert artifact["pilot_rows"] == []
    assert set(artifact["selection_checksums"]) == {"HumanEval/eligible"}


def test_req_code_2879_execution_wrapper_reports_sandbox_outcomes() -> None:
    """REQ-CODE-2879: the existing sandbox wrapper is used without unsafe fallback."""

    calls: list[dict[str, Any]] = []
    responses: list[tuple[Any, Exception | None]] = [
        (True, None),
        ("not true", None),
        (None, TimeoutError("too slow")),
    ]

    def fake_sandbox(
        code: str,
        func_name: str,
        args: tuple[Any, ...],
        timeout: float,
        allow_fallback: bool,
    ) -> tuple[Any, Exception | None]:
        calls.append(
            {
                "code": code,
                "func_name": func_name,
                "args": args,
                "timeout": timeout,
                "allow_fallback": allow_fallback,
            }
        )
        return responses.pop(0)

    script = "def __carnot_pilot__():\n    return True\n"

    assert exp.execute_script_in_sandbox(script, 3.0, fake_sandbox) == exp.ExecutionOutcome(
        passed=True,
    )
    assert exp.execute_script_in_sandbox(script, 3.0, fake_sandbox) == exp.ExecutionOutcome(
        passed=False,
        error_type="AssertionError",
        error_message="pilot harness returned 'not true'",
    )
    assert exp.execute_script_in_sandbox(script, 3.0, fake_sandbox) == exp.ExecutionOutcome(
        passed=False,
        error_type="TimeoutError",
        error_message="too slow",
        timed_out=True,
    )
    assert calls == [
        {
            "code": script,
            "func_name": "__carnot_pilot__",
            "args": (),
            "timeout": 3.0,
            "allow_fallback": False,
        },
        {
            "code": script,
            "func_name": "__carnot_pilot__",
            "args": (),
            "timeout": 3.0,
            "allow_fallback": False,
        },
        {
            "code": script,
            "func_name": "__carnot_pilot__",
            "args": (),
            "timeout": 3.0,
            "allow_fallback": False,
        }
    ]
