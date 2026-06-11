"""Tests for Exp 4013 GAP-4 verifier-vs-Codex-judge efficiency.

Spec refs: REQ-VERIFY-4013, SCENARIO-VERIFY-4013.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

import experiment_4013_verifier_vs_judge_efficiency as exp


def _cand(grid: list[list[int]], votes: int, correct: bool = False) -> dict[str, Any]:
    return {"grid": grid, "votes": votes, "q_mean": 0.5, "correct": correct}


def _entry(
    task: str,
    test_input: list[list[int]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "task": task,
        "demos": [
            {"input": [[1]], "output": [[2]]},
            {"input": [[3]], "output": [[4]]},
        ],
        "test_input": test_input,
        "candidates": candidates,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_pool(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"entries": entries}, handle)


def _write_fixture_files(root: Path) -> dict[str, Path]:
    paths = {
        "rule": root / "rule.json",
        "arc1_programs": root / "arc1_programs.json",
        "arc1_pool": root / "arc1_pool.json.gz",
        "arc2_induced": root / "arc2_induced.json",
        "arc2_chain": root / "arc2_chain.json",
        "arc2_pool": root / "arc2_pool.json.gz",
        "output": root / "out.json",
    }
    arc1_entries = [
        _entry("A1", [[9]], [_cand([[2]], 9, False), _cand([[10]], 1, True)]),
        _entry("A2", [[8]], [_cand([[7]], 6, False), _cand([[9]], 2, True)]),
    ]
    arc2_entries = [
        _entry("B1", [[5]], [_cand([[6]], 5, True), _cand([[0]], 9, False)]),
        _entry("B2", [[6]], [_cand([[1]], 7, False), _cand([[8]], 3, True)]),
    ]
    _write_pool(paths["arc1_pool"], arc1_entries)
    _write_pool(paths["arc2_pool"], arc2_entries)
    _write_json(
        paths["rule"],
        {
            "experiment": "arc3_gap4_rule_exec_verifier",
            "n_tasks": 2,
            "random_seed": 12345,
            "per_task": [
                {"task": "A1", "i": 0, "demo_perfect": True, "pred_is_gold": True},
                {"task": "A2", "i": 1, "demo_perfect": True, "pred_is_gold": True},
            ],
        },
    )
    _write_json(
        paths["arc1_programs"],
        {
            "programs": [
                {
                    "task": "A1",
                    "entry_i": 0,
                    "demo_fit": 1.0,
                    "demo_perfect": True,
                    "pred_grid": [[10]],
                    "pred_hash": "unused",
                    "code": "def transform(grid):\n    return grid\n",
                },
                {
                    "task": "A2",
                    "entry_i": 1,
                    "demo_fit": 1.0,
                    "demo_perfect": True,
                    "pred_grid": [[9]],
                    "pred_hash": "unused",
                    "code": "def transform(grid):\n    return grid\n",
                },
            ]
        },
    )
    _write_json(
        paths["arc2_induced"],
        {
            "programs": [
                {
                    "task": "B1",
                    "entry_i": 0,
                    "demo_fit": 1.0,
                    "demo_perfect": True,
                    "pred_grid": [[6]],
                    "pred_hash": "unused",
                    "code": "def transform(grid):\n    return grid\n",
                },
                {
                    "task": "B2",
                    "entry_i": 1,
                    "demo_fit": 1.0,
                    "demo_perfect": True,
                    "pred_grid": [[1]],
                    "pred_hash": "unused",
                    "code": "def transform(grid):\n    return grid\n",
                },
            ]
        },
    )
    _write_json(
        paths["arc2_chain"],
        {
            "per_task": [
                {
                    "task": "B1",
                    "arms": [
                        {
                            "source": "probe_chain",
                            "demo_fit": 1.0,
                            "demo_perfect": True,
                            "code": "def transform(grid):\n    import numpy as np\n    return np.array([[6]])\n",
                        },
                        {
                            "source": "fresh_chain1",
                            "demo_fit": 1.0,
                            "demo_perfect": True,
                            "code": "def transform(grid):\n    import numpy as np\n    return np.array([[6]])\n",
                        },
                    ],
                },
                {
                    "task": "B2",
                    "arms": [
                        {
                            "source": "probe_chain",
                            "demo_fit": 1.0,
                            "demo_perfect": True,
                            "code": "def transform(grid):\n    import numpy as np\n    return np.array([[1]])\n",
                        },
                        {
                            "source": "fresh_chain1",
                            "demo_fit": 1.0,
                            "demo_perfect": True,
                            "code": "def transform(grid):\n    import numpy as np\n    return np.array([[8]])\n",
                        },
                    ],
                },
            ],
            "total_codex_seconds": 10.0,
        },
    )
    return paths


def test_req_4013_spec_declared() -> None:
    # REQ-VERIFY-4013: OpenSpec declares Exp 4013 before implementation.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    assert "REQ-VERIFY-4013" in spec
    assert "SCENARIO-VERIFY-4013" in spec
    assert "selection_accuracy_parity" in spec


def test_preconditions_block_codex_before_candidate_sets(tmp_path: Path) -> None:
    # REQ-VERIFY-4013: missing Codex is the terminal blocker when the judge cannot run.
    paths = _write_fixture_files(tmp_path)
    preconditions = exp.check_preconditions(paths=paths, codex_available_override=False)
    assert exp.blocker_from_preconditions(preconditions) == "blocked_codex_unavailable"


def test_preconditions_block_missing_candidate_sets(tmp_path: Path) -> None:
    # REQ-VERIFY-4013: missing candidate artifacts block honestly.
    paths = _write_fixture_files(tmp_path)
    paths["arc2_chain"].unlink()
    preconditions = exp.check_preconditions(paths=paths, codex_available_override=True)
    assert exp.blocker_from_preconditions(preconditions) == "blocked_candidate_sets_missing"


def test_assemble_candidate_sets_and_verifier_selection(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4013: Arm A selects from the same candidate IDs the judge will see.
    paths = _write_fixture_files(tmp_path)
    loaded = exp.load_sources(paths)
    candidate_sets = exp.assemble_candidate_sets(loaded, top_pool_candidates=1)
    by_key = {row["task_key"]: row for row in candidate_sets}
    assert len(candidate_sets) == 4

    b1 = by_key["arc2:0:B1"]
    chosen = exp.select_with_verifier(b1)
    assert chosen == "C0"
    assert b1["candidates"][0]["is_gold"] is True
    assert b1["candidates"][0]["program_source_count"] == 3
    assert {candidate["choice_id"] for candidate in b1["candidates"]} == {"C0", "C1"}


def test_parse_judge_payload_accepts_json_and_fallback_text() -> None:
    # SCENARIO-VERIFY-4013: judge parsing maps task keys to candidate IDs.
    payload = '{"decisions":[{"task_key":"T","choice_id":"C1"}]}'
    assert exp.parse_judge_payload(payload, ["T"]) == {"T": "C1"}
    assert exp.parse_judge_payload('{"T":"C2"}', ["T"]) == {"T": "C2"}
    assert exp.parse_judge_payload("Task T -> C0", ["T"]) == {"T": "C0"}


def test_build_artifact_metrics_and_verdict() -> None:
    # REQ-VERIFY-4013: required scalar metrics and terminal verdict are derived from selections.
    candidate_sets = [
        {
            "task_key": "T1",
            "verifier_choice_id": "C0",
            "judge_choice_id": "C0",
            "candidates": [{"choice_id": "C0", "is_gold": True}],
        },
        {
            "task_key": "T2",
            "verifier_choice_id": "C0",
            "judge_choice_id": "C1",
            "candidates": [
                {"choice_id": "C0", "is_gold": False},
                {"choice_id": "C1", "is_gold": True},
            ],
        },
    ]
    artifact = exp.build_artifact(
        candidate_sets=candidate_sets,
        judge_seconds_total=20.0,
        judge_tokens_total=200,
        n_judge_calls=2,
        verifier_seconds_per_task=0.1,
        preconditions=[{"resource": "codex", "available": True}],
        started_s=0.0,
        now_s=21.0,
    )
    assert artifact["selection_agreement_rate"] == 0.5
    assert artifact["verifier_gold_rate"] == 0.5
    assert artifact["judge_gold_rate"] == 1.0
    assert artifact["cost_judge_seconds"] == 10.0
    assert artifact["cost_ratio_judge_over_verifier"] == 100.0
    assert artifact["n_tasks"] == 2
    exp.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("selection_accuracy_parity", "yes", "bare bool"),
        ("verifier_gold_rate", "1.0", "bare float"),
        ("n_tasks", 1.5, "bare int"),
        ("honest_verdict", "done", "terminal prefix"),
        ("inference_substrate", 7, "must be a string"),
    ],
)
def test_validate_artifact_rejects_typed_fields(field: str, value: Any, message: str) -> None:
    artifact = exp.blocked_artifact(
        "blocked_codex_unavailable",
        [{"resource": "codex", "available": False}],
        duration_s=0.1,
    )
    artifact[field] = value
    with pytest.raises(ValueError, match=message):
        exp.validate_artifact(artifact)


def test_helper_edges_are_deterministic(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4013: helper edge cases are deterministic and non-live.
    paths = _write_fixture_files(tmp_path)
    loaded = exp.load_sources(paths)
    limited = exp.assemble_candidate_sets(loaded, top_pool_candidates=1, limit=1)
    assert len(limited) == 1
    prompt = exp.build_judge_prompt(limited)
    assert "TASK arc1:0:A1" in prompt
    assert "Candidate C0" in prompt
    assert exp._execute_chain_arm({"demo_perfect": False, "code": "def transform(grid): return grid"}, [[1]]) is None
    assert exp._finalize_candidate_set(corpus="x", entry_idx=0, entry={"task": "T"}, clusters={}) is None
    clusters: dict[str, dict[str, Any]] = {}
    exp._add_candidate(clusters, grid=[], source="bad", is_gold=False)
    assert clusters == {}
    assert exp._top_pool_candidates({"candidates": [{"grid": [], "votes": 99}]}, 1) == []
    assert exp._choice_is_gold(limited[0], "missing") is False
    assert exp._ci95(0.0, 0) == (0.0, 0.0)
    assert exp._verdict(False, 12.0, 0.25).startswith("complete:")
    assert exp._sanitize_choice(limited[0], "not-a-choice") == "C0"
    with pytest.raises(ValueError, match="missing required field"):
        exp.validate_artifact({})


def test_run_blocked_writes_valid_artifact(tmp_path: Path) -> None:
    # REQ-VERIFY-4013: blocked preconditions still write the required schema fields.
    paths = _write_fixture_files(tmp_path)
    artifact = exp.run(
        paths=paths,
        output_path=paths["output"],
        codex_available_override=False,
        write=True,
    )
    assert artifact["honest_verdict"] == "blocked_codex_unavailable"
    assert paths["output"].exists()
    exp.validate_artifact(artifact)


def test_run_blocks_when_no_candidate_sets_materialize(tmp_path: Path) -> None:
    # REQ-VERIFY-4013: loadable but empty artifacts are still blocked as missing candidate sets.
    paths = {
        "rule": tmp_path / "rule.json",
        "arc1_programs": tmp_path / "arc1_programs.json",
        "arc1_pool": tmp_path / "arc1_pool.json.gz",
        "arc2_chain": tmp_path / "arc2_chain.json",
        "arc2_induced": tmp_path / "arc2_induced.json",
        "arc2_pool": tmp_path / "arc2_pool.json.gz",
        "output": tmp_path / "out.json",
    }
    _write_json(paths["rule"], {"per_task": []})
    _write_json(paths["arc1_programs"], {"programs": []})
    _write_json(paths["arc2_chain"], {"per_task": []})
    _write_json(paths["arc2_induced"], {"programs": []})
    _write_pool(paths["arc1_pool"], [])
    _write_pool(paths["arc2_pool"], [])
    artifact = exp.run(
        paths=paths,
        output_path=paths["output"],
        codex_available_override=True,
        write=True,
    )
    assert artifact["honest_verdict"] == "blocked_candidate_sets_missing"


def test_run_complete_with_fake_judge(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4013: fake judge completes the full metric path without live Codex.
    paths = _write_fixture_files(tmp_path)

    def fake_judge(batch: list[dict[str, Any]]) -> exp.JudgeBatchResult:
        choices = {}
        for task in batch:
            gold = next((c["choice_id"] for c in task["candidates"] if c["is_gold"]), None)
            choices[task["task_key"]] = gold or task["candidates"][0]["choice_id"]
        return exp.JudgeBatchResult(choices=choices, seconds=2.5, tokens=111, raw="fake")

    artifact = exp.run(
        paths=paths,
        output_path=paths["output"],
        codex_available_override=True,
        judge_batch_func=fake_judge,
        judge_batch_size=2,
        verifier_seconds_per_task=0.1,
        write=True,
    )
    exp.validate_artifact(artifact)
    assert artifact["n_tasks"] == 4
    assert artifact["n_judge_calls"] == 2
    assert artifact["cost_judge_seconds"] == pytest.approx(1.25)
    assert artifact["judge_gold_rate"] == 1.0
    assert artifact["honest_verdict"].startswith(("success:", "complete:"))
    assert paths["output"].exists()
