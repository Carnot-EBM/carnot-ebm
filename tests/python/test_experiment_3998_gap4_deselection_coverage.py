"""Tests for Exp 3998 GAP-4 de-selection coverage.

Spec refs: REQ-VERIFY-3998, SCENARIO-VERIFY-3998.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import experiment_3998_gap4_deselection_coverage as exp


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_gzip_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _entry(task: str, value: int) -> dict[str, Any]:
    return {
        "task": task,
        "demos": [{"input": [[value]], "output": [[value]]}],
        "test_input": [[value]],
        "candidates": [],
    }


def test_req_verify_3998_spec_anchor_exists() -> None:
    """REQ-VERIFY-3998: OpenSpec declares the de-selection coverage contract."""

    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")

    assert "REQ-VERIFY-3998" in spec
    assert "SCENARIO-VERIFY-3998" in spec
    assert "fresh_chain_demo_perfect_rate_nonselected" in spec
    assert "blocked_eval_pool_unreadable" in spec


def test_req_verify_3998_identifies_raw_11_task_complement() -> None:
    """REQ-VERIFY-3998: the real de-selection set is the raw complement of the 12 selected tasks."""

    pool = exp.load_eval_pool(Path("results/arc3_gap4_arc2_eval_pool.json.gz"))
    chain_artifact = json.loads(Path("results/arc3_gap4_arc2_chain_ensemble.json").read_text())

    assert exp.never_chained_tasks(pool["entries"], chain_artifact) == [
        "16b78196",
        "21897d95",
        "269e22fb",
        "28a6681f",
        "2c181942",
        "3a25b0d8",
        "3dc255db",
        "a6f40cea",
        "aa4ec2a5",
        "b9e38dc0",
        "dd6b8c4b",
    ]


def test_req_verify_3998_cp95_uses_exact_clopper_pearson() -> None:
    """REQ-VERIFY-3998: small-n coverage is reported with exact CP95 bounds."""

    low, high = exp.clopper_pearson_95(20, 24)
    assert low == pytest.approx(0.6262)
    assert high == pytest.approx(0.9526)

    assert exp.clopper_pearson_95(0, 10) == pytest.approx((0.0, 0.3085))
    assert exp.clopper_pearson_95(10, 10) == pytest.approx((0.6915, 1.0))


def test_req_verify_3998_leak_audit_uses_word_boundaries(tmp_path: Path) -> None:
    """REQ-VERIFY-3998: astype/pos false positives are avoided while bare leak vectors fail."""

    clean = tmp_path / "clean.txt"
    clean.write_text(
        "===== PROMPT =====\nTEST INPUT\n===== RAW OUTPUT =====\n"
        "Fix the corrected function.\n"
        "def transform(grid):\n    pos = np.argwhere(grid > 0)\n    return grid.astype(np.int64)\n",
        encoding="utf-8",
    )
    dirty = tmp_path / "dirty.txt"
    dirty.write_text(
        "===== PROMPT =====\nTEST INPUT\n===== RAW OUTPUT =====\n"
        "def transform(grid):\n    return type(grid)\n",
        encoding="utf-8",
    )

    assert exp.audit_transcripts([clean]) == {"clean": True, "n_transcripts": 1, "violations": []}

    report = exp.audit_transcripts([dirty])
    assert report["clean"] is False
    assert report["n_transcripts"] == 1
    assert report["violations"][0]["token"] == "type("


def test_scenario_verify_3998_run_writes_mocked_complete_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-3998: a complete run reports bare rates, CI, gold precision, and provenance."""

    pool_path = tmp_path / "results" / "arc3_gap4_arc2_eval_pool.json.gz"
    chain_path = tmp_path / "results" / "arc3_gap4_arc2_chain_ensemble.json"
    output_path = tmp_path / "results" / "experiment_3998_gap4_deselection_coverage.json"
    transcript_dir = tmp_path / "results" / "experiment_3998_gap4_deselection_transcripts"
    challenges_path = tmp_path / "arc-agi_evaluation2_challenges.json"
    solutions_path = tmp_path / "arc-agi_evaluation2_solutions.json"

    _write_gzip_json(
        pool_path,
        {"entries": [_entry("selected", 7), _entry("n0", 1), _entry("n1", 2)]},
    )
    _write_json(
        chain_path,
        {
            "preregistration": {"tasks": ["selected"]},
            "fresh_chain_arms_demo_perfect": "1/2",
        },
    )
    _write_json(
        challenges_path,
        {
            "n0": {"test": [{"input": [[1]]}]},
            "n1": {"test": [{"input": [[2]]}]},
        },
    )
    _write_json(solutions_path, {"n0": [[[1]]], "n1": [[[9]]]})

    def fake_induce(
        task_name: str,
        demos: list[dict[str, Any]],
        test_input: list[list[int]],
        iters: int,
        timeout: int,
        transcripts_dir: str,
    ) -> dict[str, Any]:
        del demos, timeout
        assert iters == 3
        arm = Path(transcripts_dir).name
        Path(transcripts_dir).mkdir(parents=True, exist_ok=True)
        if task_name == "n0" and arm == "arm2":
            history = [
                {"iter": 0, "status": "graded", "demo_fit": 0.0, "codex_s": 1.0, "code_len": 30},
                {"iter": 1, "status": "graded", "demo_fit": 1.0, "codex_s": 2.0, "code_len": 30},
            ]
            demo_perfect = True
            code = "def transform(grid):\n    return grid.copy()\n"
        elif task_name == "n0":
            history = [
                {"iter": 0, "status": "graded", "demo_fit": 1.0, "codex_s": 1.5, "code_len": 30}
            ]
            demo_perfect = True
            code = "def transform(grid):\n    return grid.copy()\n"
        else:
            history = [
                {"iter": 0, "status": "graded", "demo_fit": 0.0, "codex_s": 1.0, "code_len": 30}
            ]
            demo_perfect = False
            code = "def transform(grid):\n    return grid * 0\n"
        for row in history:
            (Path(transcripts_dir) / f"{task_name}_iter{row['iter']}.txt").write_text(
                "===== PROMPT =====\nDemo only\n===== RAW OUTPUT =====\n```python\n"
                + code
                + "```\n",
                encoding="utf-8",
            )
        pred = np.asarray(test_input).tolist() if demo_perfect else None
        return {
            "task": task_name,
            "demo_fit": 1.0 if demo_perfect else 0.0,
            "demo_perfect": demo_perfect,
            "pred_hash": "unused",
            "pred_grid": pred,
            "n_calls": len(history),
            "codex_seconds": sum(row["codex_s"] for row in history),
            "history": history,
            "code": code,
        }

    monkeypatch.setattr(exp, "induce_program", fake_induce)

    artifact = exp.run(
        root=tmp_path,
        pool_path=pool_path,
        chain_artifact_path=chain_path,
        output_path=output_path,
        transcripts_dir=transcript_dir,
        challenges_path=challenges_path,
        solutions_path=solutions_path,
        codex_available_override=True,
        workers=1,
        expected_nonselected_count=2,
    )

    assert output_path.exists()
    assert artifact["honest_verdict"].startswith("complete: gap4_deselection_coverage_0.5")
    assert artifact["fresh_chain_demo_perfect_rate_nonselected"] == 0.5
    assert artifact["debiased_coverage_combined"] == 0.5
    assert artifact["cp95_low"] == pytest.approx(0.0676)
    assert artifact["cp95_high"] == pytest.approx(0.9324)
    assert artifact["per_arm_gold_given_perfect"]["fresh_chain1"] == {
        "gold": 1,
        "n": 1,
        "rate": 1.0,
    }
    assert artifact["per_arm_gold_given_perfect"]["fresh_chain2"] == {
        "gold": 1,
        "n": 1,
        "rate": 1.0,
    }
    assert artifact["iter0_vs_chainfinal"]["iter0_demo_perfect"] == 1
    assert artifact["iter0_vs_chainfinal"]["chain_final_demo_perfect"] == 2
    assert artifact["iter0_vs_chainfinal"]["recovered_by_chain"] == 1
    assert artifact["leak_clean"] is True
    assert artifact["n_tasks_chained"] == 2
    assert artifact["total_codex_calls"] == 5
    assert artifact["total_codex_seconds"] == 6.5
    assert artifact["preconditions_checked"] == [
        {"resource": "codex", "available": True},
        {"resource": "eval_pool", "available": True},
    ]
    assert len(list(transcript_dir.glob("arm*/*.txt"))) == 5
    exp.validate_artifact(artifact)


def test_scenario_verify_3998_blocks_without_codex(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3998: missing codex exits as blocked without fabricated metrics."""

    pool_path = tmp_path / "results" / "arc3_gap4_arc2_eval_pool.json.gz"
    _write_gzip_json(pool_path, {"entries": [_entry("n0", 1)]})

    artifact = exp.run(
        root=tmp_path,
        pool_path=pool_path,
        output_path=tmp_path / "results" / "experiment_3998_gap4_deselection_coverage.json",
        codex_available_override=False,
        write=False,
    )

    assert artifact["honest_verdict"] == "blocked_codex_unavailable"
    assert artifact["fresh_chain_demo_perfect_rate_nonselected"] is None
    assert artifact["total_codex_calls"] == 0
    assert artifact["preconditions_checked"][0] == {"resource": "codex", "available": False}


def test_req_verify_3998_defensive_helpers_and_guards(tmp_path: Path) -> None:
    """REQ-VERIFY-3998: defensive branches preserve honest blocking and schema checks."""

    with pytest.raises(ValueError, match="total"):
        exp.clopper_pearson_95(0, 0)
    with pytest.raises(ValueError, match="successes"):
        exp.clopper_pearson_95(3, 2)

    fallback_artifact = {
        "per_task": [
            {
                "task": "selected_b",
                "arms": [
                    {"source": "fresh_chain1", "demo_perfect": True},
                    {"source": "fresh_chain2", "demo_perfect": False},
                    {"source": "probe_chain", "demo_perfect": True},
                ],
            },
            {
                "task": "selected_a",
                "arms": [{"source": "fresh_chain1", "demo_perfect": True}],
            },
        ]
    }
    assert exp.selected_tasks_from_chain_artifact(fallback_artifact) == [
        "selected_a",
        "selected_b",
    ]
    assert exp.selected_fresh_counts(fallback_artifact) == (2, 3)
    assert exp._history_iter0_demo_perfect([]) is False

    missing_pool = tmp_path / "missing.json.gz"
    preconditions = exp.check_preconditions(missing_pool, codex_available_override=True)
    assert preconditions == [
        {"resource": "codex", "available": True},
        {"resource": "eval_pool", "available": False},
    ]
    assert exp.blocker_from_preconditions(preconditions) == "blocked_eval_pool_unreadable"

    assert exp.gold_for("missing", [[1]], {}, {}) is None
    assert exp.gold_for(
        "task",
        [[9]],
        {"task": {"test": [{"input": [[1]]}]}},
        {"task": [[[1]]]},
    ) is None
    assert exp._score_gold_for_arm(
        {"demo_perfect": True, "code": "def transform(grid):\n    return type(grid)\n"},
        [_entry("task", 1)],
        {"task": {"test": [{"input": [[1]]}]}},
        {"task": [[[1]]]},
    ) == (0, 0)
    assert exp._rate(0, 0) is None

    output = tmp_path / "results" / "blocked.json"
    pool_path = tmp_path / "results" / "arc3_gap4_arc2_eval_pool.json.gz"
    _write_gzip_json(pool_path, {"entries": [_entry("n0", 1)]})
    blocked = exp.run(
        root=tmp_path,
        pool_path=pool_path,
        output_path=output,
        codex_available_override=False,
    )
    assert output.exists()
    assert blocked["honest_verdict"] == "blocked_codex_unavailable"

    chain_path = tmp_path / "results" / "chain.json"
    _write_json(chain_path, {"preregistration": {"tasks": []}, "fresh_chain_arms_demo_perfect": "0/0"})
    with pytest.raises(ValueError, match="expected 11"):
        exp.run(
            root=tmp_path,
            pool_path=pool_path,
            chain_artifact_path=chain_path,
            output_path=tmp_path / "results" / "unused.json",
            codex_available_override=True,
            write=False,
        )

    valid = exp.blocked_artifact(
        "blocked_codex_unavailable",
        [{"resource": "codex", "available": False}, {"resource": "eval_pool", "available": True}],
        1.0,
    )
    missing = dict(valid)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required"):
        exp.validate_artifact(missing)

    bad = dict(valid, honest_verdict="maybe")
    with pytest.raises(ValueError, match="terminal prefix"):
        exp.validate_artifact(bad)
    bad = dict(valid, fresh_chain_demo_perfect_rate_nonselected=True)
    with pytest.raises(ValueError, match="bare float"):
        exp.validate_artifact(bad)
    bad = dict(valid, cp95_low=True)
    with pytest.raises(ValueError, match="cp95_low"):
        exp.validate_artifact(bad)
    bad = dict(valid, leak_clean="yes")
    with pytest.raises(ValueError, match="leak_clean"):
        exp.validate_artifact(bad)
    bad = dict(valid, n_tasks_chained=True)
    with pytest.raises(ValueError, match="n_tasks_chained"):
        exp.validate_artifact(bad)
    bad = dict(valid, total_codex_seconds=0)
    with pytest.raises(ValueError, match="total_codex_seconds"):
        exp.validate_artifact(bad)
    bad = dict(valid, preconditions_checked={})
    with pytest.raises(ValueError, match="preconditions_checked"):
        exp.validate_artifact(bad)
