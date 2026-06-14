"""Tests for Exp 4175 executable headroom gate census.

Spec refs: REQ-VERIFY-4175, SCENARIO-VERIFY-4175.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "headroom_gate.py"


def _load_headroom_gate():
    spec = importlib.util.spec_from_file_location("headroom_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HG = _load_headroom_gate()


def test_req_4175_spec_declared() -> None:
    # REQ-VERIFY-4175: OpenSpec declares the runner, artifact, fields, and oracles.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    for marker in (
        "REQ-VERIFY-4175",
        "SCENARIO-VERIFY-4175",
        "scripts/headroom_gate.py",
        "results/experiment_4175_headroom_gate_executable_census.py",
        "experiment_4175_headroom_gate_executable_census.json",
        "oracle_at_k",
        "baseline_pass1",
        "sc_vote_pass1",
        "selectable_headroom=oracle_at_k - sc_vote_pass1",
        "blocked_no_multicandidate_pool",
        "arXiv:2605.07395",
    ):
        assert marker in spec
    for principle in HG.FIELD_PRINCIPLES.values():
        assert principle in spec


def test_headroom_counts_code_baseline_and_repair_candidates() -> None:
    # REQ-VERIFY-4175: code pools use executable unit-test pass flags, no LLM judge.
    stats = HG.headroom(
        {
            "domain": "code",
            "source": "experiment_1999_code_verification_humaneval.json",
            "results": [
                {"task_id": "a", "baseline_passed": False, "repair_passed": True},
                {"task_id": "b", "baseline_passed": True, "repair_passed": True},
                {"task_id": "c", "baseline_passed": False, "repair_passed": False},
                {"task_id": "d", "baseline_passed": False, "repair_passed": True},
            ],
        }
    )
    assert stats["n"] == 4
    assert stats["oracle_at_k"] == pytest.approx(0.75)
    assert stats["baseline_pass1"] == pytest.approx(0.25)
    assert stats["sc_vote_pass1"] == pytest.approx(0.25)
    assert stats["selectable_headroom"] == pytest.approx(0.5)
    assert stats["artifact_flags"]["objective_oracle"] == "unit_test_pass_flags"
    assert stats["artifact_flags"]["census_incomplete"] is False


def test_headroom_sanitizes_math_candidates_before_oracle_counting() -> None:
    # SCENARIO-VERIFY-4175: truncated/unparseable candidates are artifact inflation.
    stats = HG.headroom(
        {
            "domain": "math",
            "source": "math-fixture",
            "tasks": [
                {
                    "task_id": "m0",
                    "gold_answer": "4",
                    "candidates": [
                        {"answer": "7", "votes": 4},
                        {"answer": "4", "votes": 1},
                        {"text": "reasoning then \\boxed{4", "votes": 100},
                    ],
                },
                {
                    "task_id": "m1",
                    "gold_answer": "8",
                    "candidates": [
                        {"answer": "8", "votes": 5},
                        {"answer": "9", "votes": 1},
                    ],
                },
            ],
        }
    )
    assert stats["n"] == 2
    assert stats["oracle_at_k"] == pytest.approx(1.0)
    assert stats["sc_vote_pass1"] == pytest.approx(0.5)
    assert stats["selectable_headroom"] == pytest.approx(0.5)
    assert stats["artifact_flags"]["artifact_inflation_flagged"] == 1
    assert stats["artifact_flags"]["excluded_reasons"]["truncated"] == 1


def test_headroom_counts_sudoku_vote_and_oracle_from_candidate_table() -> None:
    # REQ-VERIFY-4175: Sudoku/ARC pools use exact candidate correctness flags.
    stats = HG.headroom(
        {
            "domain": "sudoku",
            "source": "arc3_gap3_stage0_candidate_table.json",
            "tasks": [
                {
                    "task": "s0",
                    "cands": [
                        {"correct": False, "votes": 5},
                        {"correct": True, "votes": 1},
                    ],
                },
                {
                    "task": "s1",
                    "cands": [
                        {"correct": True, "votes": 3},
                        {"correct": False, "votes": 2},
                    ],
                },
                {
                    "task": "s2",
                    "cands": [
                        {"correct": False, "votes": 4},
                        {"correct": False, "votes": 1},
                    ],
                },
            ],
        }
    )
    assert stats["n"] == 3
    assert stats["oracle_at_k"] == pytest.approx(2 / 3)
    assert stats["baseline_pass1"] == pytest.approx(1 / 3)
    assert stats["sc_vote_pass1"] == pytest.approx(1 / 3)
    assert stats["selectable_headroom"] == pytest.approx(1 / 3)
    assert stats["artifact_flags"]["objective_oracle"] == "exact_candidate_correctness"


def test_run_census_writes_bare_gated_fields_and_incomplete_math(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4175: terminal artifact chooses the largest sanitized headroom.
    results = tmp_path / "results"
    results.mkdir()
    (results / "experiment_1999_code_verification_humaneval.json").write_text(
        json.dumps(
            {
                "honest_verdict": "complete: fixture",
                "results": [
                    {"task_id": "a", "baseline_passed": False, "repair_passed": True},
                    {"task_id": "b", "baseline_passed": True, "repair_passed": True},
                    {"task_id": "c", "baseline_passed": False, "repair_passed": False},
                    {"task_id": "d", "baseline_passed": False, "repair_passed": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    (results / "experiment_1816_gsm8k_baseline.json").write_text(
        json.dumps({"honest_verdict": "blocked_gate_check_failed", "gates_evaluated": []}),
        encoding="utf-8",
    )
    (results / "adversarial_gsm8k_data_400.json").write_text(
        json.dumps({"datasets": {"control": [{"correct_answer": 4}]}}),
        encoding="utf-8",
    )
    (results / "arc3_gap3_stage0_candidate_table.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task": "s0",
                        "cands": [
                            {"correct": False, "votes": 5},
                            {"correct": True, "votes": 1},
                        ],
                    },
                    {
                        "task": "s1",
                        "cands": [
                            {"correct": True, "votes": 3},
                            {"correct": False, "votes": 2},
                        ],
                    },
                    {
                        "task": "s2",
                        "cands": [
                            {"correct": False, "votes": 4},
                            {"correct": False, "votes": 1},
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    artifact = HG.run_census(tmp_path)
    HG.validate_artifact(artifact)

    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["headroom_present_domain"] == "code"
    assert artifact["max_selectable_headroom"] == pytest.approx(0.5)
    assert isinstance(artifact["max_selectable_headroom"], float)
    assert artifact["per_domain_headroom"]["math"]["artifact_flags"]["census_incomplete"] is True
    assert artifact["artifact_inflation_flagged"] == 0
    assert artifact["field_principles"] == HG.FIELD_PRINCIPLES
    written = json.loads(
        (results / "experiment_4175_headroom_gate_executable_census.json").read_text(
            encoding="utf-8"
        )
    )
    assert written == artifact

    with pytest.raises(ValueError, match="bare float"):
        HG.validate_artifact({**artifact, "max_selectable_headroom": {"value": 0.5}})
    with pytest.raises(ValueError, match="headroom_present_domain"):
        HG.validate_artifact({**artifact, "headroom_present_domain": {}})


def test_run_census_blocks_when_no_multicandidate_pool_exists(tmp_path: Path) -> None:
    # REQ-VERIFY-4175: no executable K-candidate pool yields an honest blocked verdict.
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "experiment_1816_gsm8k_baseline.json").write_text(
        json.dumps({"honest_verdict": "blocked_gate_check_failed"}),
        encoding="utf-8",
    )

    artifact = HG.run_census(tmp_path)

    HG.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_no_multicandidate_pool"
    assert artifact["max_selectable_headroom"] == 0.0
    assert artifact["headroom_present_domain"] == ""
