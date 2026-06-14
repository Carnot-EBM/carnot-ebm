"""Tests for Exp 4187 GAP-4 graded execution gate hardening.

REQ-VERIFY-4187 / SCENARIO-VERIFY-4187: the replay artifact must be produced
from cached evidence, carry the required schema fields, and report the
vote-aware guard against the 25094a63 high-vote promotion.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
MODULE_PATH = ROOT / "results" / "experiment_4187_gap4_graded_execution_gate_hardening.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("experiment_4187", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_pool(path: Path) -> None:
    near_pred = [[0 for _ in range(10)] for _ in range(10)]
    near_wrong = [[0 for _ in range(10)] for _ in range(10)]
    near_wrong[0][0] = 1
    near_gold = [[0 for _ in range(10)] for _ in range(10)]
    near_gold[0][0] = 2
    entries = [
        {
            "task": "recover_task",
            "candidates": [
                {"grid": [[0]], "votes": 10, "correct": False},
                {"grid": [[8]], "votes": 9, "correct": False},
                {"grid": [[1]], "votes": 1, "correct": True},
            ],
        },
        {
            "task": "25094a63",
            "candidates": [
                {"grid": [[7]], "votes": 945, "correct": True},
                {"grid": [[3]], "votes": 32, "correct": False},
                {"grid": [[4]], "votes": 1, "correct": False},
            ],
        },
        {
            "task": "near_miss_task",
            "candidates": [
                {"grid": near_wrong, "votes": 6, "correct": False},
                {"grid": near_gold, "votes": 5, "correct": True},
                {"grid": [[9]], "votes": 1, "correct": False},
            ],
        },
    ]
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"entries": entries}, f)


def _write_programs(path: Path) -> None:
    programs = [
        {
            "task": "recover_task",
            "demo_fit": 1.0,
            "demo_perfect": True,
            "pred_grid": [[1]],
        },
        {
            "task": "25094a63",
            "demo_fit": 1.0,
            "demo_perfect": True,
            "pred_grid": [[3]],
        },
        {
            "task": "near_miss_task",
            "demo_fit": 1.0,
            "demo_perfect": True,
            "pred_grid": [[0 for _ in range(10)] for _ in range(10)],
        },
    ]
    path.write_text(json.dumps({"programs": programs}), encoding="utf-8")


def test_req_verify_4187_blocked_when_cached_pool_missing(tmp_path: Path) -> None:
    exp = _load_module()
    artifact_path = tmp_path / "experiment_4187.json"

    artifact = exp.run(
        pool_path=tmp_path / "missing_pool.json.gz",
        programs_path=tmp_path / "missing_programs.json",
        rule_exec_path=tmp_path / "missing_rule_exec.json",
        artifact_path=artifact_path,
    )

    assert artifact["honest_verdict"] == "blocked_gap4_arc1_pool_missing"
    assert artifact_path.exists()


def test_req_verify_4187_artifact_schema_and_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exp = _load_module()
    monkeypatch.setattr(exp, "EXACT_MATCH_BASELINE_RECOVERED", 1)
    pool_path = tmp_path / "pool.json.gz"
    programs_path = tmp_path / "programs.json"
    rule_exec_path = tmp_path / "rule_exec.json"
    artifact_path = tmp_path / "experiment_4187.json"
    _write_pool(pool_path)
    _write_programs(programs_path)
    rule_exec_path.write_text(json.dumps({"random_seed": 12345}), encoding="utf-8")

    artifact = exp.run(
        pool_path=pool_path,
        programs_path=programs_path,
        rule_exec_path=rule_exec_path,
        artifact_path=artifact_path,
        high_vote_guard_threshold=900,
    )

    required = {
        "honest_verdict",
        "graded_gate_pass2_vs_vote",
        "vote_aware_guard_blocked_mispromotion",
        "gross_recovery_ledger",
        "band_precision_at_tau",
        "random_seed",
        "reproducibility_checksum",
        "gate_fire_count",
        "pass_at_1",
        "pass_at_2",
        "pass2_vote_wins_lost",
        "agreement_confidence_label_only",
        "duration_s",
        "inference_substrate",
    }
    assert required <= artifact.keys()
    assert artifact["vote_aware_guard_blocked_mispromotion"] is True
    assert artifact["pass2_vote_wins_lost"] == 0
    assert artifact["gross_recovery_ledger"] == {"recovered": 1, "lost": 0}
    assert artifact["band_precision_at_tau"]["total"] == 1
    assert artifact["band_precision_at_tau"]["correct"] == 0
    assert artifact["agreement_confidence_label_only"] is True
    assert artifact_path.exists()


def test_req_verify_4187_rejects_task_mismatch(tmp_path: Path) -> None:
    exp = _load_module()
    entries = [{"task": "a", "candidates": [{"grid": [[1]], "votes": 1, "correct": True}]}]
    programs = [{"task": "b", "demo_fit": 1.0, "pred_grid": [[1]]}]

    with pytest.raises(ValueError, match="task mismatch"):
        exp.build_artifact(
            entries=entries,
            programs=programs,
            rule_exec={"random_seed": 12345},
            pool_path=tmp_path / "pool.json.gz",
            programs_path=tmp_path / "programs.json",
            started=0.0,
        )


def test_req_verify_4187_cli_main_writes_artifact(tmp_path: Path) -> None:
    exp = _load_module()
    pool_path = tmp_path / "pool.json.gz"
    programs_path = tmp_path / "programs.json"
    rule_exec_path = tmp_path / "rule_exec.json"
    artifact_path = tmp_path / "experiment_4187.json"
    _write_pool(pool_path)
    _write_programs(programs_path)
    rule_exec_path.write_text(json.dumps({"random_seed": 12345}), encoding="utf-8")

    assert exp.main(
        [
            "--pool",
            str(pool_path),
            "--programs",
            str(programs_path),
            "--rule-exec",
            str(rule_exec_path),
            "--artifact",
            str(artifact_path),
        ]
    ) == 0

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["honest_verdict"].startswith("complete: gap4_graded_gate_bounded_arc1")
