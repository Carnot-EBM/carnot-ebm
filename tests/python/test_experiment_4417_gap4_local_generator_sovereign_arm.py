"""Tests for Exp 4417 GAP-4 local generator sovereign forward arm.

REQ-VERIFY-4417 / SCENARIO-VERIFY-4417: the runner must fail closed on missing
cached resources and otherwise report the local-generator coverage separately
from the k-consistent guarded graded-gate pass@2 effect.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).parents[2]
MODULE_PATH = ROOT / "results" / "experiment_4417_gap4_local_generator_sovereign_arm.py"

PLUS_ONE = "def transform(grid):\n    return grid + 1\n"
IDENTITY = "def transform(grid):\n    return grid\n"
WRONG = "def transform(grid):\n    return grid + 9\n"


def _load_module():
    spec = importlib.util.spec_from_file_location("experiment_4417", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_pool(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"entries": entries}, handle)


def _demo(value_in: int, value_out: int) -> dict[str, Any]:
    return {"input": [[value_in]], "output": [[value_out]]}


def _candidate(grid: list[list[int]], *, votes: int, correct: bool) -> dict[str, Any]:
    return {"grid": grid, "votes": votes, "correct": correct}


def _entry(task: str, demos: list[dict[str, Any]], test_input: list[list[int]]) -> dict[str, Any]:
    return {
        "task": task,
        "demos": demos,
        "test_input": test_input,
        "candidates": [
            _candidate([[99]], votes=10, correct=False),
            _candidate([[88]], votes=9, correct=False),
            _candidate([[test_input[0][0] + 1]], votes=1, correct=True),
        ],
    }


def _entries() -> list[dict[str, Any]]:
    return [
        _entry("recover", [_demo(1, 2), _demo(3, 4)], [[5]]),
        {
            "task": "vote_win",
            "demos": [_demo(7, 7), _demo(8, 8)],
            "test_input": [[2]],
            "candidates": [
                _candidate([[2]], votes=10, correct=True),
                _candidate([[0]], votes=1, correct=False),
            ],
        },
        _entry("singleton", [_demo(4, 5), _demo(5, 6)], [[9]]),
    ]


def _sample(task: str, code: str, *, draw_index: int) -> dict[str, Any]:
    return {
        "task": task,
        "draw_index": draw_index,
        "status": "graded",
        "demo_fit": 1.0,
        "demo_perfect": True,
        "local_s": 0.1,
        "code": code,
    }


def _write_checkpoint(path: Path, *, disagree: bool = False) -> None:
    recover_second = WRONG if disagree else PLUS_ONE
    _write_json(
        path,
        {
            "schema": "carnot.test.local_checkpoint.v1",
            "k_samples_per_task": 2,
            "local_model_used": "Qwen3.6-35B-A3B",
            "tasks": {
                "recover": [
                    _sample("recover", PLUS_ONE, draw_index=0),
                    _sample("recover", recover_second, draw_index=1),
                ],
                "vote_win": [_sample("vote_win", IDENTITY, draw_index=0)],
                "singleton": [_sample("singleton", PLUS_ONE, draw_index=0)],
            },
        },
    )


def _write_vote_baseline(path: Path) -> None:
    _write_json(
        path,
        {
            "experiment": "arc3_trm_verifier_rerank",
            "trm_vote_pass2": 0.3333,
            "rankers": {"TRM_VOTE": {"pass@2": 0.3333}},
        },
    )


def _gguf_path(tmp_path: Path) -> Path:
    path = (
        tmp_path
        / "models--unsloth--Qwen3.6-35B-A3B-GGUF"
        / "snapshots"
        / "abc"
        / "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    )
    path.parent.mkdir(parents=True)
    path.write_text("gguf\n", encoding="utf-8")
    return path


def test_req_verify_4417_spec_declared() -> None:
    # REQ-VERIFY-4417: OpenSpec declares the sovereign forward arm and blockers.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    assert "REQ-VERIFY-4417" in spec
    assert "SCENARIO-VERIFY-4417" in spec
    assert "experiment_4417_gap4_local_generator_sovereign_arm.py" in spec
    assert "blocked_cached_pool_unavailable" in spec
    assert "blocked_local_generator_not_cached" in spec


def test_req_verify_4417_blocks_before_scoring_when_pool_or_vote_missing(tmp_path: Path) -> None:
    # REQ-VERIFY-4417: missing cached pool/baseline produces the mandated blocked verdict.
    exp = _load_module()
    artifact_path = tmp_path / "artifact.json"

    artifact = exp.run(
        artifact_path=artifact_path,
        pool_path=tmp_path / "missing_pool.json.gz",
        vote_baseline_path=tmp_path / "missing_vote.json",
        checkpoint_path=tmp_path / "missing_checkpoint.json",
        cache_root=tmp_path,
        gguf_preflight_fn=lambda _path: True,
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_cached_pool_unavailable"
    assert artifact["sovereign_gap4_gate_holds"] is False
    assert artifact["local_generator_coverage"] == 0.0
    assert artifact_path.exists()


def test_req_verify_4417_blocks_when_local_generator_not_cached(tmp_path: Path) -> None:
    # REQ-VERIFY-4417: the local GGUF path is checked before local sample replay.
    exp = _load_module()
    pool_path = tmp_path / "pool.json.gz"
    vote_path = tmp_path / "vote.json"
    checkpoint_path = tmp_path / "checkpoint.json"
    artifact_path = tmp_path / "artifact.json"
    _write_pool(pool_path, _entries())
    _write_vote_baseline(vote_path)
    _write_checkpoint(checkpoint_path)

    artifact = exp.run(
        artifact_path=artifact_path,
        pool_path=pool_path,
        vote_baseline_path=vote_path,
        checkpoint_path=checkpoint_path,
        cache_root=tmp_path / "empty",
        gguf_preflight_fn=lambda _path: False,
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_local_generator_not_cached"
    assert artifact["preconditions_checked"][1]["resource"] == "local_open_weight_generator_gguf"
    assert artifact["preconditions_checked"][1]["available"] is False


def test_scenario_verify_4417_k_consistent_gate_beats_vote(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4417: two agreeing local inductions may drive a guarded promotion.
    exp = _load_module()
    pool_path = tmp_path / "pool.json.gz"
    vote_path = tmp_path / "vote.json"
    checkpoint_path = tmp_path / "checkpoint.json"
    artifact_path = tmp_path / "artifact.json"
    _write_pool(pool_path, _entries())
    _write_vote_baseline(vote_path)
    _write_checkpoint(checkpoint_path)

    artifact = exp.run(
        artifact_path=artifact_path,
        pool_path=pool_path,
        vote_baseline_path=vote_path,
        checkpoint_path=checkpoint_path,
        cache_root=tmp_path,
        gguf_preflight_fn=lambda _path: True,
        gguf_resolver=lambda _root: {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "model_path": str(_gguf_path(tmp_path)),
        },
        n_bootstrap=64,
    )

    exp.validate_artifact(artifact)
    assert artifact["sovereign_gap4_gate_holds"] is True
    assert artifact["local_generator_coverage"] == 1.0
    assert artifact["pass2_vs_vote"]["vote_pass2"] == 0.3333
    assert artifact["pass2_vs_vote"]["gated_pass2"] == 0.6667
    assert artifact["pass2_vs_vote"]["pass2_vote_wins_lost"] == 0
    assert artifact["pass2_vs_vote"]["graded_gate_fires"] == 1
    assert artifact["verifier_is_oracle"] is True
    assert artifact["model_specs"]["k_consistency"] == 2


def test_scenario_verify_4417_disagreement_falls_back_to_vote(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4417: demo-perfect local samples without k-consistency do not promote.
    exp = _load_module()
    pool_path = tmp_path / "pool.json.gz"
    vote_path = tmp_path / "vote.json"
    checkpoint_path = tmp_path / "checkpoint.json"
    _write_pool(pool_path, _entries())
    _write_vote_baseline(vote_path)
    _write_checkpoint(checkpoint_path, disagree=True)

    artifact = exp.run(
        artifact_path=tmp_path / "artifact.json",
        pool_path=pool_path,
        vote_baseline_path=vote_path,
        checkpoint_path=checkpoint_path,
        cache_root=tmp_path,
        gguf_preflight_fn=lambda _path: True,
        gguf_resolver=lambda _root: {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "model_path": str(_gguf_path(tmp_path)),
        },
        n_bootstrap=64,
    )

    exp.validate_artifact(artifact)
    assert artifact["pass2_vs_vote"]["gated_pass2"] == artifact["pass2_vs_vote"]["vote_pass2"]
    assert artifact["pass2_vs_vote"]["graded_gate_fires"] == 0
    assert artifact["pass2_vs_vote"]["pass2_vote_wins_lost"] == 0
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_verify_4417_rejects_schema_poison() -> None:
    # REQ-VERIFY-4417: capstone-facing booleans and floats remain bare primitives.
    exp = _load_module()
    artifact = exp.blocked_artifact(
        verdict="blocked_cached_pool_unavailable",
        preconditions=[{"resource": "cached_trm_pool_and_vote_baseline", "available": False}],
        duration_s=0.1,
    )
    artifact["sovereign_gap4_gate_holds"] = "false"
    with pytest.raises(ValueError, match="sovereign_gap4_gate_holds"):
        exp.validate_artifact(artifact)


def test_req_verify_4417_helper_edges_and_validation_errors(tmp_path: Path) -> None:
    # REQ-VERIFY-4417: malformed resources and schema poison fail closed.
    exp = _load_module()

    bad_pool = tmp_path / "bad_pool.json.gz"
    with gzip.open(bad_pool, "wt", encoding="utf-8") as handle:
        json.dump({"entries": {}}, handle)
    with pytest.raises(ValueError, match="entries list"):
        exp._load_pool(bad_pool)

    bad_vote = tmp_path / "bad_vote.json"
    _write_json(bad_vote, {"rankers": {"TRM_VOTE": {}}})
    with pytest.raises(ValueError, match="TRM_VOTE pass@2"):
        exp._load_vote_baseline(bad_vote)
    ranker_vote = tmp_path / "ranker_vote.json"
    _write_json(ranker_vote, {"rankers": {"TRM_VOTE": {"pass@2": 0.25}}})
    assert exp._load_vote_baseline(ranker_vote)["artifact_pass2"] == 0.25

    bad_checkpoint = tmp_path / "bad_checkpoint.json"
    _write_json(bad_checkpoint, {"tasks": []})
    with pytest.raises(ValueError, match="tasks map"):
        exp._load_checkpoint(bad_checkpoint)

    assert exp._sha256_file(tmp_path / "missing.txt") is None
    assert exp.bootstrap_delta_ci95([], []) == [0.0, 0.0]
    with pytest.raises(ValueError, match="equal length"):
        exp.bootstrap_delta_ci95([1], [1, 0])
    assert exp._verdict(holds=False, coverage=0.1, delta=-0.1, fires=0, lost=1).startswith(
        "complete: clean_null"
    )

    model_path = _gguf_path(tmp_path)
    no_exist = model_path.parents[2] / ".no_exist" / "abc"
    no_exist.mkdir(parents=True)
    (no_exist / "fake.gguf").write_text("fake\n", encoding="utf-8")
    resolved = exp.resolve_cached_local_gguf(tmp_path)
    assert resolved and resolved["model_path"] == str(model_path)
    assert exp.resolve_cached_local_gguf(tmp_path / "none") is None

    entry = _entries()[0]
    records, perfect = exp._verified_predictions_for_task(
        entry,
        [
            {"draw_index": True, "code": ""},
            {"draw_index": 1, "code": "def transform(grid):\n    return open('x')\n"},
        ],
    )
    assert records == []
    assert perfect is False

    records_by_task, perfect_tasks = exp.build_local_program_records(
        [entry],
        {"tasks": {"recover": {"not": "a-list"}}},
    )
    assert records_by_task == {"recover": []}
    assert perfect_tasks == set()

    pool_path = tmp_path / "pool.json.gz"
    vote_path = tmp_path / "vote.json"
    _write_pool(pool_path, _entries())
    _write_vote_baseline(vote_path)
    checkpoint_missing = exp.run(
        artifact_path=tmp_path / "checkpoint_missing.json",
        pool_path=pool_path,
        vote_baseline_path=vote_path,
        checkpoint_path=tmp_path / "missing_checkpoint.json",
        cache_root=tmp_path,
        gguf_preflight_fn=lambda _path: True,
        gguf_resolver=lambda _root: {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "model_path": str(model_path),
        },
    )
    assert checkpoint_missing["honest_verdict"] == "blocked_local_generator_not_cached"

    valid = exp.blocked_artifact(
        verdict="blocked_cached_pool_unavailable",
        preconditions=[{"resource": "cached_trm_pool_and_vote_baseline", "available": False}],
        duration_s=0.1,
    )
    mutations = [
        ("missing required fields", lambda a: a.pop("model_specs")),
        ("honest_verdict", lambda a: a.__setitem__("honest_verdict", "bad")),
        ("verifier_is_oracle", lambda a: a.__setitem__("verifier_is_oracle", "true")),
        ("local_generator_coverage", lambda a: a.__setitem__("local_generator_coverage", 0)),
        ("pass2_vs_vote", lambda a: a.__setitem__("pass2_vs_vote", [])),
        ("pass2_vs_vote.vote_pass2", lambda a: a["pass2_vs_vote"].__setitem__("vote_pass2", 0)),
        ("pass2_vs_vote.delta_ci95", lambda a: a["pass2_vs_vote"].__setitem__("delta_ci95", [0.0])),
        (
            "pass2_vs_vote.graded_gate_fires",
            lambda a: a["pass2_vs_vote"].__setitem__("graded_gate_fires", True),
        ),
        ("preconditions_checked", lambda a: a.__setitem__("preconditions_checked", {})),
        (
            "precondition availability",
            lambda a: a.__setitem__("preconditions_checked", [{"available": "yes"}]),
        ),
        ("random_seed", lambda a: a.__setitem__("random_seed", True)),
        ("reproducibility_checksum", lambda a: a.__setitem__("reproducibility_checksum", 123)),
        ("model_specs", lambda a: a.__setitem__("model_specs", [])),
        ("duration_s", lambda a: a.__setitem__("duration_s", "0.1")),
    ]
    for message, mutate in mutations:
        candidate = json.loads(json.dumps(valid))
        mutate(candidate)
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(candidate)
