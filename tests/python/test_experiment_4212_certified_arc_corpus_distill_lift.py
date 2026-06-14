"""Tests for Exp 4212 certified ARC corpus distill-lift read.

REQ-VERIFY-4212 / SCENARIO-VERIFY-4212: the runner resumes the Exp 4200
certified corpus, rebuilds the GAP-4-certified Codex corpus, reports
verifier_is_oracle=true, and compares cold vs certified-exemplar seeded local
induction with a paired bootstrap CI.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_4212_certified_arc_corpus_distill_lift as exp4212


PLUS_ONE = "def transform(grid):\n    return grid + 1\n"
IDENTITY = "def transform(grid):\n    return grid\n"
WRONG = "def transform(grid):\n    return grid + 9\n"


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
        "schema": "carnot.exp4212.synthetic_checkpoint.v1",
        "k_samples_per_task": 1,
        "local_model_used": "Qwen3.6-35B-A3B",
        "tasks": tasks,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_prior_corpus(path: Path) -> None:
    row = {
        "task": "cert_correct",
        "demo_pairs": [_demo(1, 2), _demo(3, 4)],
        "program": PLUS_ONE.strip(),
        "verifier_label": "gap4_guarded_demo_perfect",
        "demo_fit": 1.0,
    }
    path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")


def test_req_verify_4212_spec_declared() -> None:
    # REQ-VERIFY-4212: OpenSpec declares the v2 certified corpus and lift artifact.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    assert "REQ-VERIFY-4212" in spec
    assert "SCENARIO-VERIFY-4212" in spec
    assert "experiment_4212_certified_arc_corpus_distill_lift.py" in spec
    assert "verifier_is_oracle=true" in spec
    assert "distill_lift_delta" in spec


def test_req_verify_4212_blocks_missing_pool_before_model(tmp_path: Path) -> None:
    # REQ-VERIFY-4212: missing ARC-1 Codex pool stops as blocked_gap4_arc1_pool_missing.
    artifact = exp4212.run(
        artifact_path=tmp_path / "artifact.json",
        corpus_path=tmp_path / "corpus.jsonl",
        prior_corpus_path=tmp_path / "prior.jsonl",
        pool_path=tmp_path / "missing_pool.json.gz",
        programs_path=tmp_path / "missing_programs.json",
        cold_checkpoint_path=tmp_path / "cold.json",
        seeded_checkpoint_path=tmp_path / "seeded.json",
        cached_pair_fn=lambda: pytest.fail("cached_sota_pair should not run"),
    )

    exp4212.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_gap4_arc1_pool_missing"
    assert artifact["certified_corpus_size"] == 0
    assert artifact["verifier_is_oracle"] is True


def test_req_verify_4212_blocks_missing_sota_gguf(tmp_path: Path) -> None:
    # REQ-VERIFY-4212: cached_sota_pair() is the local GGUF precondition.
    pool_path = tmp_path / "pool.json.gz"
    programs_path = tmp_path / "programs.json"
    _write_pool(pool_path, _synthetic_entries())
    _write_programs(programs_path)

    artifact = exp4212.run(
        artifact_path=tmp_path / "artifact.json",
        corpus_path=tmp_path / "corpus.jsonl",
        prior_corpus_path=tmp_path / "prior.jsonl",
        pool_path=pool_path,
        programs_path=programs_path,
        cold_checkpoint_path=tmp_path / "cold.json",
        seeded_checkpoint_path=tmp_path / "seeded.json",
        cached_pair_fn=lambda: [],
    )

    exp4212.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_model_not_cached_sota_gguf"
    assert artifact["model_specs"]["local_ggufs"] == []


def test_scenario_verify_4212_builds_corpus_and_latent_seeded_lift(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4212: seeded checkpoint lift excludes zero and diagnoses latent.
    entries = _synthetic_entries()
    pool_path = tmp_path / "pool.json.gz"
    programs_path = tmp_path / "programs.json"
    cold_path = tmp_path / "cold.json"
    seeded_path = tmp_path / "seeded.json"
    prior_path = tmp_path / "prior.jsonl"
    corpus_path = tmp_path / "corpus.jsonl"
    _write_pool(pool_path, entries)
    _write_programs(programs_path)
    _write_prior_corpus(prior_path)
    _write_checkpoint(cold_path, {entry["task"]: [_bad_sample(entry["task"])] for entry in entries})
    _write_checkpoint(
        seeded_path,
        {
            "cert_correct": [_sample("cert_correct", PLUS_ONE)],
            "identity_correct": [_sample("identity_correct", IDENTITY)],
            "cert_wrong": [_sample("cert_wrong", PLUS_ONE)],
            "guard_blocked": [_sample("guard_blocked", PLUS_ONE)],
        },
    )

    artifact = exp4212.run(
        artifact_path=tmp_path / "artifact.json",
        corpus_path=corpus_path,
        prior_corpus_path=prior_path,
        pool_path=pool_path,
        programs_path=programs_path,
        cold_checkpoint_path=cold_path,
        seeded_checkpoint_path=seeded_path,
        cached_pair_fn=_cached_pair,
        bootstrap_n=300,
    )

    exp4212.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["certified_corpus_size"] == 3
    assert artifact["certification_precision"] == {
        "correct": 2,
        "certified": 3,
        "rate": 0.6667,
    }
    assert artifact["prior_corpus_resume"]["prior_rows"] == 1
    assert artifact["prior_corpus_resume"]["retained_rows"] == 1
    assert artifact["local_induction_cold"]["rate"] == 0.0
    assert artifact["local_induction_with_certified_exemplars"]["rate"] == 1.0
    assert artifact["distill_lift_delta"] == 1.0
    assert artifact["distill_lift_ci95"][0] > 0.0
    assert artifact["invisible_leash_diagnosis"] == "latent"
    assert artifact["verifier_is_oracle"] is True
    assert "verifier_is_oracle" in artifact["field_principles"]
    lines = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    assert [line["task"] for line in lines] == ["cert_correct", "identity_correct", "cert_wrong"]
    assert all(line["verifier_label"] == "gap4_guarded_demo_perfect" for line in lines)
    assert all("correct" not in line and "test_output" not in line for line in lines)


def test_scenario_verify_4212_missing_seeded_checkpoint_is_absent_flat(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4212: no seeded sample source is reported as flat/absent, not latent.
    entries = _synthetic_entries()
    pool_path = tmp_path / "pool.json.gz"
    programs_path = tmp_path / "programs.json"
    cold_path = tmp_path / "cold.json"
    _write_pool(pool_path, entries)
    _write_programs(programs_path)
    _write_checkpoint(
        cold_path,
        {
            "cert_correct": [_sample("cert_correct", PLUS_ONE)],
            "identity_correct": [_bad_sample("identity_correct")],
            "cert_wrong": [_bad_sample("cert_wrong")],
            "guard_blocked": [_bad_sample("guard_blocked")],
        },
    )

    artifact = exp4212.run(
        artifact_path=tmp_path / "artifact.json",
        corpus_path=tmp_path / "corpus.jsonl",
        prior_corpus_path=tmp_path / "missing_prior.jsonl",
        pool_path=pool_path,
        programs_path=programs_path,
        cold_checkpoint_path=cold_path,
        seeded_checkpoint_path=tmp_path / "missing_seeded.json",
        cached_pair_fn=_cached_pair,
        bootstrap_n=300,
    )

    exp4212.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["seeded_generation_status"] == "missing_seeded_checkpoint_conservative_flat"
    assert artifact["local_induction_cold"]["rate"] == 0.25
    assert artifact["local_induction_with_certified_exemplars"]["rate"] == 0.25
    assert artifact["distill_lift_delta"] == 0.0
    assert artifact["distill_lift_ci95"] == [0.0, 0.0]
    assert artifact["invisible_leash_diagnosis"] == "absent"


def test_req_verify_4212_validation_rejects_nonbare_oracle() -> None:
    # REQ-VERIFY-4212: verifier_is_oracle must be the bare bool true.
    artifact = exp4212.blocked_artifact(
        verdict="blocked_gap4_arc1_pool_missing",
        preconditions=[],
        model_specs=exp4212.model_specs_from_inputs(None),
        duration_s=0.0,
    )
    artifact["verifier_is_oracle"] = "true"
    with pytest.raises(ValueError, match="verifier_is_oracle"):
        exp4212.validate_artifact(artifact)


def test_req_verify_4212_helper_edges_and_validation_errors(tmp_path: Path) -> None:
    # REQ-VERIFY-4212: helper edges remain deterministic and schema poison is rejected.
    assert exp4212._numpy_only_import("numpy").__name__ == "numpy"
    with pytest.raises(ImportError):
        exp4212._numpy_only_import("os")

    assert exp4212.safe_transform_from_code("def transform(grid):\n    return open('x')\n") is None
    assert exp4212.safe_transform_from_code("def transform(:\n") is None
    assert exp4212.safe_transform_from_code("x = 1\n") is None
    assert exp4212.safe_transform_from_code("def transform(grid):\n    raise RuntimeError()\n")(
        [[1]]
    ) is None
    assert exp4212.safe_transform_from_code("def transform(grid):\n    return [1, 2, 3]\n")(
        [[1]]
    ) is None
    assert exp4212.safe_transform_from_code("def transform(grid):\n    return [[99]]\n")(
        [[1]]
    ) is None
    assert exp4212._to_grid_list(None) is None
    assert exp4212._to_grid_list([1, 2, 3]) is None
    assert exp4212._sample_draw_index({"draw_index": True}) == 0

    bad_pool = tmp_path / "bad_pool.json.gz"
    with gzip.open(bad_pool, "wt", encoding="utf-8") as handle:
        json.dump({"entries": {}}, handle)
    with pytest.raises(ValueError, match="entries list"):
        exp4212._load_pool(bad_pool)

    bad_programs = tmp_path / "bad_programs.json"
    bad_programs.write_text(json.dumps({"programs": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="programs list"):
        exp4212._load_programs(bad_programs)

    bad_checkpoint = tmp_path / "bad_checkpoint.json"
    bad_checkpoint.write_text(json.dumps({"tasks": []}), encoding="utf-8")
    assert exp4212._load_checkpoint(bad_checkpoint) == {}

    prior = tmp_path / "prior.jsonl"
    prior.write_text("\n" + json.dumps({"task": "x", "program": "p"}) + "\n", encoding="utf-8")
    assert exp4212._load_prior_corpus(prior) == [{"task": "x", "program": "p"}]

    entries = _synthetic_entries()
    missing_code_programs = [
        {"task": entry["task"], "code": "", "demo_fit": 0.0, "demo_perfect": False}
        for entry in entries
    ]
    corpus, precision, audit = exp4212.build_certified_corpus(entries, missing_code_programs)
    assert corpus == []
    assert precision == {"correct": 0, "certified": 0, "rate": 0.0}
    assert audit[0]["reason"] == "missing_code"

    unsafe_programs = [
        {"task": entry["task"], "code": "def transform(grid):\n    return open('x')\n"}
        for entry in entries
    ]
    assert exp4212.build_certified_corpus(entries, unsafe_programs)[2][0]["reason"] == "unsafe_code"

    low_fit_programs = [
        {"task": entry["task"], "code": IDENTITY}
        for entry in entries
    ]
    assert (
        exp4212.build_certified_corpus(entries, low_fit_programs)[2][0]["reason"]
        == "demo_fit_not_exact"
    )

    with pytest.raises(ValueError, match="task mismatch"):
        exp4212.build_certified_corpus(
            [entries[0]],
            [{"task": "different", "code": PLUS_ONE}],
        )

    successes, summaries = exp4212._checkpoint_task_successes(
        entries,
        {
            "tasks": {
                "cert_correct": "not-a-list",
                "identity_correct": [{"code": ""}, {"code": "def transform(grid):\n    return open('x')\n"}],
            }
        },
    )
    assert successes == [0, 0, 0, 0]
    assert summaries[0]["n_samples"] == 0
    assert summaries[1]["n_samples"] == 2
    assert exp4212._checkpoint_task_successes(entries, {"tasks": []})[0] == [0, 0, 0, 0]

    assert exp4212.bootstrap_lift_ci([], []) == [0.0, 0.0]
    with pytest.raises(ValueError, match="equal length"):
        exp4212.bootstrap_lift_ci([1], [])

    base = exp4212.blocked_artifact(
        verdict="blocked_gap4_arc1_pool_missing",
        preconditions=[],
        model_specs=exp4212.model_specs_from_inputs(None),
        duration_s=0.0,
    )

    def poisoned(**updates: Any) -> dict[str, Any]:
        artifact = json.loads(json.dumps(base))
        artifact.update(updates)
        return artifact

    bad = json.loads(json.dumps(base))
    bad.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required field"):
        exp4212.validate_artifact(bad)
    with pytest.raises(ValueError, match="terminal prefix"):
        exp4212.validate_artifact(poisoned(honest_verdict="maybe"))
    with pytest.raises(ValueError, match="certified_corpus_size"):
        exp4212.validate_artifact(poisoned(certified_corpus_size=True))
    with pytest.raises(ValueError, match="certification_precision"):
        exp4212.validate_artifact(poisoned(certification_precision=[]))
    with pytest.raises(ValueError, match="distill_lift_delta"):
        exp4212.validate_artifact(poisoned(distill_lift_delta=0))
    with pytest.raises(ValueError, match="distill_lift_ci95"):
        exp4212.validate_artifact(poisoned(distill_lift_ci95=[0.0]))
    with pytest.raises(ValueError, match="random_seed"):
        exp4212.validate_artifact(poisoned(random_seed=True))
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp4212.validate_artifact(poisoned(reproducibility_checksum=123))
    with pytest.raises(ValueError, match="preconditions_checked"):
        exp4212.validate_artifact(poisoned(preconditions_checked={}))
    with pytest.raises(ValueError, match="duration_s"):
        exp4212.validate_artifact(poisoned(duration_s=0))
    with pytest.raises(ValueError, match="inference_substrate"):
        exp4212.validate_artifact(poisoned(inference_substrate=[]))
