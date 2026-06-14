"""Tests for Exp 4200 GAP-4 certified ARC corpus and distill-lift read.

REQ-VERIFY-4200 / SCENARIO-VERIFY-4200: the runner must build a guarded
GAP-4-certified Codex corpus first, then compare local cold induction with a
certified-exemplar seeded local checkpoint using a paired bootstrap CI.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).parents[2]
MODULE_PATH = ROOT / "results" / "experiment_4200_certified_arc_corpus_distill_lift.py"

PLUS_ONE = "def transform(grid):\n    return grid + 1\n"
IDENTITY = "def transform(grid):\n    return grid\n"
WRONG = "def transform(grid):\n    return grid + 9\n"


def _load_module():
    spec = importlib.util.spec_from_file_location("experiment_4200", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cached_pair() -> list[dict[str, Any]]:
    return [
        {
            "name": "Qwen3.6-35B-A3B",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "gpu": 0,
            "model_path": "/models/qwen.gguf",
        },
        {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "gpu": 1,
            "model_path": "/models/gemma26.gguf",
        },
    ]


def _demo(value_in: int, value_out: int) -> dict[str, Any]:
    return {"input": [[value_in]], "output": [[value_out]]}


def _candidate(grid: list[list[int]], *, votes: int, correct: bool) -> dict[str, Any]:
    return {"grid": grid, "votes": votes, "correct": correct, "q_mean": 0.5}


def _entry(
    task: str,
    demos: list[dict[str, Any]],
    test_input: list[list[int]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "task": task,
        "demos": demos,
        "test_input": test_input,
        "candidates": candidates,
    }


def _synthetic_entries() -> list[dict[str, Any]]:
    return [
        _entry(
            "cert_correct",
            [_demo(1, 2), _demo(3, 4)],
            [[5]],
            [_candidate([[9]], votes=10, correct=False), _candidate([[6]], votes=1, correct=True)],
        ),
        _entry(
            "identity_correct",
            [_demo(2, 2), _demo(4, 4)],
            [[7]],
            [_candidate([[7]], votes=8, correct=True), _candidate([[0]], votes=1, correct=False)],
        ),
        _entry(
            "cert_wrong",
            [_demo(1, 2), _demo(3, 4)],
            [[5]],
            [_candidate([[6]], votes=4, correct=False), _candidate([[7]], votes=3, correct=True)],
        ),
        _entry(
            "guard_blocked",
            [_demo(1, 2), _demo(3, 4)],
            [[5]],
            [
                _candidate([[7]], votes=1000, correct=True),
                _candidate([[6]], votes=1, correct=False),
            ],
        ),
    ]


def _write_pool(path: Path, entries: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"entries": entries}, handle)


def _write_programs(path: Path) -> None:
    programs = [
        {"task": "cert_correct", "code": PLUS_ONE, "demo_fit": 1.0, "demo_perfect": True},
        {"task": "identity_correct", "code": IDENTITY, "demo_fit": 1.0, "demo_perfect": True},
        {"task": "cert_wrong", "code": PLUS_ONE, "demo_fit": 1.0, "demo_perfect": True},
        {"task": "guard_blocked", "code": PLUS_ONE, "demo_fit": 1.0, "demo_perfect": True},
    ]
    path.write_text(json.dumps({"programs": programs}, indent=2), encoding="utf-8")


def _sample(task: str, code: str, draw_index: int = 0) -> dict[str, Any]:
    return {
        "task": task,
        "draw_index": draw_index,
        "status": "graded",
        "demo_fit": 1.0,
        "demo_perfect": True,
        "local_s": 0.1,
        "code": code,
    }


def _bad_sample(task: str, draw_index: int = 0) -> dict[str, Any]:
    return {
        "task": task,
        "draw_index": draw_index,
        "status": "graded",
        "demo_fit": 0.0,
        "demo_perfect": False,
        "local_s": 0.1,
        "code": WRONG,
    }


def _write_checkpoint(path: Path, tasks: dict[str, list[dict[str, Any]]]) -> None:
    payload = {
        "schema": "carnot.exp4200.synthetic_checkpoint.v1",
        "k_samples_per_task": 1,
        "local_model_used": "Qwen3.6-35B-A3B",
        "tasks": tasks,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_req_verify_4200_spec_declared() -> None:
    # REQ-VERIFY-4200: OpenSpec declares the certified corpus + distill-lift artifact first.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    assert "REQ-VERIFY-4200" in spec
    assert "SCENARIO-VERIFY-4200" in spec
    assert "experiment_4200_certified_arc_corpus_distill_lift.py" in spec
    assert "blocked_gap4_arc1_pool_missing" in spec
    assert "distill_lift_ci95" in spec


def test_req_verify_4200_blocks_missing_pool_before_model(tmp_path: Path) -> None:
    # REQ-VERIFY-4200: missing Codex ARC-1 program pool stops as blocked_gap4_arc1_pool_missing.
    exp = _load_module()
    artifact_path = tmp_path / "artifact.json"

    artifact = exp.run(
        artifact_path=artifact_path,
        corpus_path=tmp_path / "corpus.jsonl",
        pool_path=tmp_path / "pool.json.gz",
        programs_path=tmp_path / "missing_programs.json",
        cold_checkpoint_path=tmp_path / "cold.json",
        seeded_checkpoint_path=tmp_path / "seeded.json",
        cached_pair_fn=lambda: None,
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_gap4_arc1_pool_missing"
    assert artifact["certified_corpus_size"] == 0
    assert artifact_path.exists()


def test_req_verify_4200_blocks_missing_sota_gguf(tmp_path: Path) -> None:
    # REQ-VERIFY-4200: cached_sota_pair() is the local-base precondition.
    exp = _load_module()
    pool_path = tmp_path / "pool.json.gz"
    programs_path = tmp_path / "programs.json"
    _write_pool(pool_path, _synthetic_entries())
    _write_programs(programs_path)

    artifact = exp.run(
        artifact_path=tmp_path / "artifact.json",
        corpus_path=tmp_path / "corpus.jsonl",
        pool_path=pool_path,
        programs_path=programs_path,
        cold_checkpoint_path=tmp_path / "cold.json",
        seeded_checkpoint_path=tmp_path / "seeded.json",
        cached_pair_fn=lambda: [],
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_model_not_cached_sota_gguf"
    assert artifact["model_specs"]["local_ggufs"] == []


def test_scenario_verify_4200_builds_corpus_and_seeded_lift(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4200: guarded gate corpus plus paired seeded-vs-cold lift CI.
    exp = _load_module()
    entries = _synthetic_entries()
    pool_path = tmp_path / "pool.json.gz"
    programs_path = tmp_path / "programs.json"
    cold_path = tmp_path / "cold.json"
    seeded_path = tmp_path / "seeded.json"
    corpus_path = tmp_path / "corpus.jsonl"
    artifact_path = tmp_path / "artifact.json"
    _write_pool(pool_path, entries)
    _write_programs(programs_path)
    _write_checkpoint(
        cold_path,
        {entry["task"]: [_bad_sample(entry["task"])] for entry in entries},
    )
    _write_checkpoint(
        seeded_path,
        {
            "cert_correct": [_sample("cert_correct", PLUS_ONE)],
            "identity_correct": [_sample("identity_correct", IDENTITY)],
            "cert_wrong": [_sample("cert_wrong", PLUS_ONE)],
            "guard_blocked": [_sample("guard_blocked", PLUS_ONE)],
        },
    )

    artifact = exp.run(
        artifact_path=artifact_path,
        corpus_path=corpus_path,
        pool_path=pool_path,
        programs_path=programs_path,
        cold_checkpoint_path=cold_path,
        seeded_checkpoint_path=seeded_path,
        cached_pair_fn=_cached_pair,
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["certified_corpus_size"] == 3
    assert artifact["certification_precision"] == {
        "correct": 2,
        "certified": 3,
        "rate": 0.6667,
    }
    assert artifact["local_induction_cold"]["rate"] == 0.0
    assert artifact["local_induction_with_certified_exemplars"]["rate"] == 1.0
    assert artifact["distill_lift_ci95"][0] > 0.0
    assert artifact["invisible_leash_diagnosis"] == "latent"
    lines = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    assert [line["task"] for line in lines] == ["cert_correct", "identity_correct", "cert_wrong"]
    assert all(line["verifier_label"] == "gap4_guarded_demo_perfect" for line in lines)
    assert all("correct" not in line and "test_output" not in line for line in lines)


def test_req_verify_4200_conservative_no_seeded_checkpoint(tmp_path: Path) -> None:
    # REQ-VERIFY-4200: missing seeded checkpoint is reported as flat/no-lift, not fabricated.
    exp = _load_module()
    entries = _synthetic_entries()
    pool_path = tmp_path / "pool.json.gz"
    programs_path = tmp_path / "programs.json"
    cold_path = tmp_path / "cold.json"
    _write_pool(pool_path, entries)
    _write_programs(programs_path)
    _write_checkpoint(
        cold_path,
        {"cert_correct": [_sample("cert_correct", PLUS_ONE)]},
    )

    artifact = exp.run(
        artifact_path=tmp_path / "artifact.json",
        corpus_path=tmp_path / "corpus.jsonl",
        pool_path=pool_path,
        programs_path=programs_path,
        cold_checkpoint_path=cold_path,
        seeded_checkpoint_path=tmp_path / "missing_seeded.json",
        cached_pair_fn=_cached_pair,
    )

    assert artifact["seeded_generation_status"] == "missing_seeded_checkpoint_conservative_flat"
    assert artifact["local_induction_with_certified_exemplars"]["rate"] == artifact[
        "local_induction_cold"
    ]["rate"]
    assert artifact["distill_lift_ci95"] == [0.0, 0.0]
    assert artifact["invisible_leash_diagnosis"] == "uninformative"


def test_req_verify_4200_schema_guards_and_helpers(tmp_path: Path) -> None:
    # REQ-VERIFY-4200: helpers reject malformed scalar/list schema and mismatched tasks.
    exp = _load_module()
    artifact = exp.blocked_artifact(
        verdict="blocked_gap4_arc1_pool_missing",
        preconditions=[{"resource": "arc1_codex_induced_programs", "available": False}],
        model_specs={},
        duration_s=0.1,
    )
    invalid_size_artifact = dict(artifact)
    invalid_size_artifact["certified_corpus_size"] = True
    with pytest.raises(ValueError, match="certified_corpus_size"):
        exp.validate_artifact(invalid_size_artifact)

    with pytest.raises(ValueError, match="task mismatch"):
        exp.build_certified_corpus(
            [_entry("a", [_demo(1, 2)], [[1]], [_candidate([[2]], votes=1, correct=True)])],
            [{"task": "b", "code": PLUS_ONE, "demo_fit": 1.0}],
        )

    assert exp._gguf_pair_available([{"model_path": "one.gguf"}]) is False
    assert exp._sample_draw_index({"draw_index": True}) == 0
    assert exp._rate(1, 3) == {"demo_perfect": 1, "total": 3, "rate": 0.3333}

    assert exp._numpy_only_import("numpy").__name__ == "numpy"
    with pytest.raises(ImportError):
        exp._numpy_only_import("os")
    assert exp.safe_transform_from_code("def transform(grid):\n    return open('x')\n") is None
    assert exp.safe_transform_from_code("def transform(:\n") is None
    assert exp.safe_transform_from_code("x = 1\n") is None
    assert exp.safe_transform_from_code("def transform(grid):\n    raise RuntimeError('boom')\n")(
        [[1]]
    ) is None
    assert exp.safe_transform_from_code("def transform(grid):\n    return []\n")([[1]]) is None
    assert exp.safe_transform_from_code("def transform(grid):\n    return [[99]]\n")([[1]]) is None
    assert exp._to_grid_list(None) is None
    assert exp._to_grid_list([1, 2, 3]) is None

    bad_pool = tmp_path / "bad_pool.json.gz"
    with gzip.open(bad_pool, "wt", encoding="utf-8") as handle:
        json.dump({"entries": {}}, handle)
    with pytest.raises(ValueError, match="entries list"):
        exp._load_pool(bad_pool)

    bad_programs = tmp_path / "bad_programs.json"
    bad_programs.write_text(json.dumps({"programs": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="programs list"):
        exp._load_programs(bad_programs)

    bad_checkpoint = tmp_path / "bad_checkpoint.json"
    bad_checkpoint.write_text(json.dumps({"tasks": []}), encoding="utf-8")
    assert exp._load_checkpoint(bad_checkpoint) == {}

    branch_entries = [
        _entry("missing", [_demo(1, 2)], [[1]], [_candidate([[2]], votes=1, correct=True)]),
        _entry("unsafe", [_demo(1, 2)], [[1]], [_candidate([[2]], votes=1, correct=True)]),
        _entry("bad_fit", [_demo(1, 2)], [[1]], [_candidate([[2]], votes=1, correct=True)]),
    ]
    branch_programs = [
        {"task": "missing", "code": ""},
        {"task": "unsafe", "code": "def transform(grid):\n    return open('x')\n"},
        {"task": "bad_fit", "code": IDENTITY},
    ]
    corpus, precision, audit = exp.build_certified_corpus(branch_entries, branch_programs)
    assert corpus == []
    assert precision == {"correct": 0, "certified": 0, "rate": 0.0}
    assert [row["reason"] for row in audit] == [
        "missing_code",
        "unsafe_code",
        "demo_fit_not_exact",
    ]

    malformed_checkpoint = {"tasks": {"missing": "not-list"}}
    successes, summaries = exp._checkpoint_task_successes(branch_entries, malformed_checkpoint)
    assert successes == [0, 0, 0]
    assert summaries[0]["n_samples"] == 0
    skipped_checkpoint = {
        "tasks": {
            "missing": [
                {"code": ""},
                {"code": "def transform(grid):\n    return open('x')\n"},
            ]
        }
    }
    skipped_successes, skipped_summaries = exp._checkpoint_task_successes(
        branch_entries, skipped_checkpoint
    )
    assert skipped_successes == [0, 0, 0]
    assert skipped_summaries[0]["n_samples"] == 2
    empty_successes, _empty_summary = exp._checkpoint_task_successes(branch_entries, {"tasks": []})
    assert empty_successes == [0, 0, 0]

    with pytest.raises(ValueError, match="equal length"):
        exp.bootstrap_lift_ci([1], [1, 0])
    assert exp.bootstrap_lift_ci([], []) == [0.0, 0.0]
    assert exp.bootstrap_lift_ci([1, 0], [1, 0]) == [0.0, 0.0]
    assert exp._diagnosis([0.0, 0.2], "seeded_checkpoint_replay") == "absent"
    assert exp._verdict("absent", 3, {"rate": 0.5}, [0.0, 0.2]).startswith("complete:")

    for field, bad_value, message in [
        ("honest_verdict", "not-terminal", "honest_verdict"),
        ("certification_precision", [], "certification_precision"),
        ("distill_lift_ci95", [0.0], "distill_lift_ci95"),
        ("random_seed", True, "random_seed"),
        ("reproducibility_checksum", 7, "reproducibility_checksum"),
        ("preconditions_checked", {}, "preconditions_checked"),
        ("duration_s", 1, "duration_s"),
        ("inference_substrate", 3, "inference_substrate"),
    ]:
        bad_artifact = dict(artifact)
        bad_artifact[field] = bad_value
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(bad_artifact)

    missing_field = dict(artifact)
    del missing_field["model_specs"]
    with pytest.raises(ValueError, match="missing required field"):
        exp.validate_artifact(missing_field)
