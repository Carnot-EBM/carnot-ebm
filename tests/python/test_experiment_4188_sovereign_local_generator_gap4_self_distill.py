"""Tests for Exp 4188 sovereign local GAP-4 generator self-distillation.

REQ-VERIFY-4188 / SCENARIO-VERIFY-4188: the runner must either block honestly
when the mandated SOTA GGUF pair is absent, or replay local GGUF-generated
programs through the hardened GAP-4 gate and bank only verifier-labeled
demo-perfect programs.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).parents[2]
MODULE_PATH = ROOT / "results" / "experiment_4188_sovereign_local_generator_gap4_self_distill.py"

PLUS_ONE = "def transform(grid):\n    return grid + 1\n"
IDENTITY = "def transform(grid):\n    return grid\n"
WRONG = "def transform(grid):\n    return grid + 9\n"


def _load_module():
    spec = importlib.util.spec_from_file_location("experiment_4188", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _demo(value_in: int, value_out: int) -> dict[str, Any]:
    return {"input": [[value_in]], "output": [[value_out]]}


def _candidate(grid: list[list[int]], *, votes: int, correct: bool) -> dict[str, Any]:
    return {"grid": grid, "votes": votes, "correct": correct}


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
            "recover",
            [_demo(1, 2), _demo(3, 4)],
            [[5]],
            [
                _candidate([[9]], votes=10, correct=False),
                _candidate([[8]], votes=9, correct=False),
                _candidate([[6]], votes=1, correct=True),
            ],
        ),
        _entry(
            "vote_keeps",
            [_demo(4, 4), _demo(7, 7)],
            [[2]],
            [
                _candidate([[2]], votes=10, correct=True),
                _candidate([[0]], votes=1, correct=False),
            ],
        ),
        _entry(
            "no_verified_program",
            [_demo(1, 3), _demo(2, 4)],
            [[3]],
            [
                _candidate([[0]], votes=10, correct=False),
                _candidate([[1]], votes=9, correct=False),
                _candidate([[5]], votes=1, correct=True),
            ],
        ),
    ]


def _write_pool(path: Path, entries: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"entries": entries}, handle)


def _sample(task: str, code: str, *, draw_index: int, claimed_perfect: bool = True) -> dict[str, Any]:
    return {
        "task": task,
        "draw_index": draw_index,
        "status": "graded",
        "demo_fit": 1.0 if claimed_perfect else 0.0,
        "demo_perfect": claimed_perfect,
        "local_s": 0.25,
        "code": code,
    }


def _write_checkpoint(path: Path) -> None:
    payload = {
        "schema": "carnot.exp4188.synthetic_checkpoint.v1",
        "k_samples_per_task": 2,
        "local_model_used": "Qwen3.6-35B-A3B",
        "tasks": {
            "recover": [_sample("recover", PLUS_ONE, draw_index=0)],
            "vote_keeps": [_sample("vote_keeps", IDENTITY, draw_index=0)],
            "no_verified_program": [
                _sample("no_verified_program", WRONG, draw_index=0, claimed_perfect=True)
            ],
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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


def test_req_verify_4188_spec_declared() -> None:
    # REQ-VERIFY-4188: OpenSpec declares the sovereign local generator artifact first.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    assert "REQ-VERIFY-4188" in spec
    assert "SCENARIO-VERIFY-4188" in spec
    assert "experiment_4188_sovereign_local_generator_gap4_self_distill.py" in spec
    assert "blocked_model_not_cached_sota_gguf" in spec
    assert "self_distillation_corpus_size" in spec


def test_req_verify_4188_blocks_when_sota_gguf_pair_missing(tmp_path: Path) -> None:
    # REQ-VERIFY-4188: missing SOTA GGUF pair blocks before pool reads or inference.
    exp = _load_module()
    artifact_path = tmp_path / "artifact.json"

    artifact = exp.run(
        artifact_path=artifact_path,
        corpus_path=tmp_path / "corpus.jsonl",
        pool_path=tmp_path / "missing_pool.json.gz",
        checkpoint_path=tmp_path / "missing_checkpoint.json",
        hardened_gate_path=tmp_path / "missing_4187.json",
        cached_pair_fn=lambda: None,
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_model_not_cached_sota_gguf"
    assert artifact["no_closed_weight_call"] is True
    assert artifact["local_induction_rate"]["total"] == 0
    assert artifact_path.exists()


def test_req_verify_4188_hardened_replay_and_corpus(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4188: local demo-perfect programs drive hardened gate and corpus banking.
    exp = _load_module()
    pool_path = tmp_path / "pool.json.gz"
    checkpoint_path = tmp_path / "checkpoint.json"
    hardened_path = tmp_path / "experiment_4187.json"
    artifact_path = tmp_path / "artifact.json"
    corpus_path = tmp_path / "corpus.jsonl"
    _write_pool(pool_path, _synthetic_entries())
    _write_checkpoint(checkpoint_path)
    hardened_path.write_text(json.dumps({"honest_verdict": "complete: hardened"}), encoding="utf-8")

    artifact = exp.run(
        artifact_path=artifact_path,
        corpus_path=corpus_path,
        pool_path=pool_path,
        checkpoint_path=checkpoint_path,
        hardened_gate_path=hardened_path,
        cached_pair_fn=_cached_pair,
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["local_induction_rate"] == {
        "demo_perfect": 2,
        "total": 3,
        "rate": 0.6667,
        "codex_reference": {"demo_perfect": 29, "total": 31, "rate": 0.9355},
    }
    assert artifact["sovereign_pool_pass2"]["TRM_VOTE"] == 0.3333
    assert artifact["sovereign_pool_pass2"]["LOCAL_HARDENED_GATE"] == 0.6667
    assert artifact["sovereign_pool_pass2"]["recovered"] == 1
    assert artifact["sovereign_pool_pass2"]["lost"] == 0
    assert artifact["self_distillation_corpus_size"] == 2
    assert artifact["no_closed_weight_call"] is True
    assert artifact["model_specs"]["generator_hf_id"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
    lines = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
    assert [line["task"] for line in lines] == ["recover", "vote_keeps"]
    assert all(line["verifier_label"] == "demo_perfect" for line in lines)
    assert all("correct" not in line and "test_output" not in line for line in lines)


def test_req_verify_4188_rejects_schema_poison(tmp_path: Path) -> None:
    # REQ-VERIFY-4188: no_closed_weight_call must remain a bare bool.
    exp = _load_module()
    artifact = exp.blocked_artifact(
        verdict="blocked_model_not_cached_sota_gguf",
        preconditions=[{"resource": "sota_gguf_pair_cached", "available": False}],
        duration_s=0.1,
    )
    artifact["no_closed_weight_call"] = "true"
    with pytest.raises(ValueError, match="no_closed_weight_call"):
        exp.validate_artifact(artifact)


def test_req_verify_4188_helper_edges_and_error_branches(tmp_path: Path) -> None:
    # REQ-VERIFY-4188: deterministic helpers reject unsafe code and malformed replay inputs.
    exp = _load_module()

    assert exp._numpy_only_import("numpy").__name__ == "numpy"
    with pytest.raises(ImportError):
        exp._numpy_only_import("os")
    assert exp.safe_transform_from_code("def transform(grid):\n    return open('x')\n") is None
    assert exp.safe_transform_from_code("def transform(:\n") is None
    assert exp.safe_transform_from_code("x = 1\n") is None
    assert exp.safe_transform_from_code("def transform(grid):\n    return 99\n")([[1]]) is None
    assert (
        exp.safe_transform_from_code("def transform(grid):\n    raise RuntimeError('boom')\n")(
            [[1]]
        )
        is None
    )
    assert exp._sample_draw_index({"draw_index": True}) == 0
    assert exp._to_grid_list(None) is None
    assert exp._to_grid_list([1, 2, 3]) is None

    bad_pool = tmp_path / "bad_pool.json.gz"
    with gzip.open(bad_pool, "wt", encoding="utf-8") as handle:
        json.dump({"entries": {}}, handle)
    with pytest.raises(ValueError, match="entries list"):
        exp._load_pool(bad_pool)

    bad_checkpoint = tmp_path / "bad_checkpoint.json"
    bad_checkpoint.write_text(json.dumps({"tasks": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="tasks map"):
        exp._load_checkpoint(bad_checkpoint)

    assert exp._gguf_pair_available([{"model_path": "one.gguf"}]) is False
    assert exp._select_generator_spec(
        [
            {"name": "Gemma", "hf_id": "gemma", "model_path": "g.gguf"},
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "model_path": "q.gguf",
            },
        ],
        checkpoint_model=None,
    )["name"] == "Qwen3.6-35B-A3B"
    assert exp._select_generator_spec(
        [{"name": "Only", "hf_id": "only", "model_path": "only.gguf"}],
        checkpoint_model="missing",
    )["name"] == "Only"
    assert exp._blocker([{"resource": "sota_gguf_pair_cached", "available": True}]) == (
        "blocked_arc1_candidate_pool_missing"
    )
    assert exp._blocker(
        [
            {"resource": "sota_gguf_pair_cached", "available": True},
            {"resource": "arc1_candidate_pool", "available": True},
            {"resource": "hardened_gap4_gate_artifact", "available": False},
        ]
    ) == "blocked_hardened_gap4_gate_missing"

    duplicate_records, best_fit = exp._verified_records_for_task(
        "dup",
        [_demo(1, 2)],
        [
            {"code": "", "draw_index": 0},
            {"code": "import os\ndef transform(grid):\n    return grid\n", "draw_index": 1},
            _sample("dup", PLUS_ONE, draw_index=2),
            _sample("dup", PLUS_ONE, draw_index=3),
        ],
    )
    assert len(duplicate_records) == 1
    assert best_fit == 1.0

    entries = _synthetic_entries()
    with pytest.raises(ValueError, match="length mismatch"):
        exp.score_hardened_gate(entries, [])
    with pytest.raises(ValueError, match="task mismatch"):
        exp.score_hardened_gate(
            [entries[0]],
            [{"task": "other", "demo_fit": 1.0, "pred_grid": [[6]]}],
        )

    empty_corpus = tmp_path / "empty.jsonl"
    exp._write_corpus(empty_corpus, [])
    assert empty_corpus.read_text(encoding="utf-8") == ""
    assert exp._file_sha(tmp_path / "missing.txt") is None
    assert exp._verdict(0.1, 0.3, 0.5, 0).startswith("complete:")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("honest_verdict", "done", "terminal prefix"),
        ("no_closed_weight_call", False, "must be true"),
        ("self_distillation_corpus_size", True, "bare int"),
        ("random_seed", True, "bare int"),
        ("local_induction_rate", [], "must be a dict"),
        ("preconditions_checked", {}, "must be a list"),
        ("reproducibility_checksum", 123, "must be a string"),
        ("duration_s", 1, "bare float"),
        ("inference_substrate", 123, "must be a string"),
    ],
)
def test_req_verify_4188_validate_rejects_typed_fields(
    field: str,
    value: Any,
    message: str,
) -> None:
    # REQ-VERIFY-4188: terminal artifacts keep bare typed schema fields.
    exp = _load_module()
    artifact = exp.blocked_artifact(
        verdict="blocked_model_not_cached_sota_gguf",
        preconditions=[{"resource": "sota_gguf_pair_cached", "available": False}],
        duration_s=0.1,
    )
    artifact[field] = value
    with pytest.raises(ValueError, match=message):
        exp.validate_artifact(artifact)


def test_req_verify_4188_validate_rejects_missing_field() -> None:
    # REQ-VERIFY-4188: every required artifact field must be present.
    exp = _load_module()
    with pytest.raises(ValueError, match="missing required field"):
        exp.validate_artifact({})
