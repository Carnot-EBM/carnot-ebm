"""Tests for Exp 4069 synchronous MoE resume-accumulate runner.

Spec refs: REQ-VERIFY-4069, SCENARIO-VERIFY-4069.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

import exp4069_decentralization_moe_sync as exp


PLUS_ONE = "```python\ndef transform(grid):\n    return grid + 1\n```"


class _BatchSampler:
    def __init__(self, responses: list[str], seconds: float = 0.1) -> None:
        self.responses = responses
        self.seconds = seconds
        self.calls: list[tuple[str, list[int]]] = []

    def sample_many(self, prompt: str, draw_indices: list[int]) -> list[tuple[int, str, float]]:
        self.calls.append((prompt, list(draw_indices)))
        return [
            (draw_index, self.responses[draw_index % len(self.responses)], self.seconds)
            for draw_index in draw_indices
        ]


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
    return {"grid": grid, "votes": votes, "correct": correct, "q_mean": 0.5}


def _entry(task: str, rule_delta: int, test_in: int) -> dict[str, Any]:
    return {
        "task": task,
        "demos": [_demo(1, 1 + rule_delta), _demo(3, 3 + rule_delta)],
        "test_input": [[test_in]],
        "candidates": [
            _candidate([[test_in + rule_delta]], votes=1, correct=True),
            _candidate([[99]], votes=9, correct=False),
        ],
    }


def _synthetic_pool() -> list[dict[str, Any]]:
    return [
        _entry("A", 1, 5),
        _entry("B", 2, 4),
        _entry("C", 1, 6),
        _entry("D", 1, 7),
    ]


def _sample(task: str, *, perfect: bool, code: str | None = PLUS_ONE) -> dict[str, Any]:
    return {
        "task": task,
        "draw_index": 0,
        "status": "graded" if perfect else "no_code",
        "demo_fit": 1.0 if perfect else 0.0,
        "demo_perfect": perfect,
        "local_s": 0.2,
        "code": code if perfect else None,
    }


def _checkpoint(path: Path, tasks: dict[str, list[dict[str, Any]]], *, k: int = 2) -> None:
    _write_json(
        path,
        {
            "schema": "carnot.experiment_4048_decentralization_moe_base_raw.checkpoint.v1",
            "k_samples_per_task": k,
            "local_model_used": exp.MOE_MODEL_NAME,
            "tasks": tasks,
        },
    )


def _cache_dir(tmp_path: Path) -> Path:
    cache = tmp_path / "models--unsloth--Qwen3.6-35B-A3B-GGUF"
    cache.mkdir(parents=True)
    (cache / "marker").write_text("cached\n", encoding="utf-8")
    return cache


def _references(oracle: float = 0.6129) -> dict[str, float]:
    return {
        "coverage_12b": 0.2581,
        "pass2_12b": 0.4516,
        "oracle_coverage": oracle,
        "codex_pass2": 0.5806,
        "codex_seconds": 46.24,
    }


def test_req_4069_spec_declared() -> None:
    # REQ-VERIFY-4069: OpenSpec declares the synchronous terminal runner first.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    assert "REQ-VERIFY-4069" in spec
    assert "SCENARIO-VERIFY-4069" in spec
    assert "exp4069_decentralization_moe_sync.py" in spec
    assert "single_synchronous_resume_accumulate_no_background" in spec


def test_blocked_missing_moe_cache_writes_terminal_artifact(tmp_path: Path) -> None:
    # REQ-VERIFY-4069: missing MoE cache blocks before llama.cpp or inference.
    output = tmp_path / "result.json"

    artifact = exp.run(
        pool_path=tmp_path / "missing_pool.json.gz",
        output_path=output,
        source_checkpoint_path=tmp_path / "missing_source.json",
        stable_checkpoint_path=tmp_path / "stable.json",
        gaps_path=tmp_path / "gaps.md",
        cache_dir=tmp_path / "empty_cache",
        resolver=lambda _hf_id: None,
        llama_available_override=False,
        expected_unique_tasks=0,
        run_summarizer=False,
    )

    exp.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_moe_base_not_cached"
    assert artifact["accumulated_n_tasks"] == 0
    assert artifact["mechanism"] == exp.MECHANISM
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert output.exists()


def test_preconditions_require_exp4012_pool_and_exp4048_checkpoint(tmp_path: Path) -> None:
    # REQ-VERIFY-4069: pool and checkpoint are checked before inference.
    pool = tmp_path / "pool.json.gz"
    source = tmp_path / "source.json"
    model_path = tmp_path / "qwen.gguf"
    model_path.write_text("model\n", encoding="utf-8")
    _write_pool(pool, _synthetic_pool())
    _checkpoint(source, {"A": [_sample("A", perfect=True)]}, k=2)

    preconditions, chosen, resumed = exp.check_preconditions(
        pool_path=pool,
        source_checkpoint_path=source,
        cache_dir=_cache_dir(tmp_path),
        resolver=lambda _hf_id: str(model_path),
        llama_available_override=True,
        k=2,
        expected_unique_tasks=0,
    )

    assert chosen and chosen["name"] == exp.MOE_MODEL_NAME
    assert resumed == 1
    assert exp.blocker_from_preconditions(preconditions) is None
    assert {row["resource"]: row["available"] for row in preconditions}[
        "exp4048_checkpoint"
    ] is True


def test_resume_accumulates_one_task_without_regenerating_checkpointed_tasks(
    tmp_path: Path,
) -> None:
    # SCENARIO-VERIFY-4069: source tasks seed stable checkpoint and foreground progress is emitted.
    pool = tmp_path / "pool.json.gz"
    source = tmp_path / "source.json"
    stable = tmp_path / "stable.json"
    output = tmp_path / "result.json"
    gaps = tmp_path / "ops" / "verifier_gaps.md"
    model_path = tmp_path / "qwen.gguf"
    model_path.write_text("model\n", encoding="utf-8")
    _write_pool(pool, _synthetic_pool())
    _checkpoint(
        source,
        {
            "A": [_sample("A", perfect=True)],
            "B": [_sample("B", perfect=False), _sample("B", perfect=False)],
        },
        k=2,
    )
    progress: list[str] = []
    summaries: list[Path] = []
    sampler = _BatchSampler([PLUS_ONE])

    artifact = exp.run(
        pool_path=pool,
        output_path=output,
        source_checkpoint_path=source,
        stable_checkpoint_path=stable,
        gaps_path=gaps,
        cache_dir=_cache_dir(tmp_path),
        resolver=lambda _hf_id: str(model_path),
        llama_available_override=True,
        expected_unique_tasks=0,
        k=2,
        batch_size=1,
        max_new_tasks=1,
        sampler=sampler,
        progress_fn=progress.append,
        n_bootstrap=64,
        run_summarizer=True,
        summarizer_fn=lambda path: summaries.append(path) or {"returncode": 0},
    )

    exp.validate_artifact(artifact)
    stable_payload = json.loads(stable.read_text(encoding="utf-8"))
    assert sorted(stable_payload["tasks"]) == ["A", "B", "C"]
    assert len(sampler.calls) == 1
    assert artifact["resumed_from_n"] == 2
    assert artifact["new_tasks_processed"] == 1
    assert artifact["accumulated_n_tasks"] == 3
    assert artifact["local_support_diagnosis"] == "accumulating"
    assert artifact["honest_verdict"] == "complete: decentralization_moe_accumulating_n_3"
    assert progress == ["[moe] task 3/4 demo_perfect=True cov=0.6667 elapsed=0s"]
    assert summaries == [output]
    assert "GAP-DECENTRALIZATION-MOE-SYNC-4069" in gaps.read_text(encoding="utf-8")


def test_terminal_diagnosis_branches_latent_absent_and_uninformative(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4069: Invisible-Leash diagnosis follows accumulated N, CI, and oracle headroom.
    base_kwargs = {
        "output_path": tmp_path / "result.json",
        "preconditions": [{"resource": "moe_base_gguf_cached", "available": True}],
        "model_specs": {"generator_model": exp.MOE_MODEL_NAME},
        "random_seed": exp.SEED,
        "duration_s": 1.0,
        "resumed_from_n": 14,
        "new_tasks_processed": 16,
        "pass2": 0.7,
        "references": _references(),
        "stable_checkpoint_path": tmp_path / "stable.json",
        "source_checkpoint_path": tmp_path / "source.json",
        "n_bootstrap": 64,
    }
    latent_rows = [
        {"task": f"T{i}", "best_of_n_demo_perfect": True, "n_demo_perfect_samples": 1, "local_seconds": 1.0}
        for i in range(30)
    ]
    absent_rows = [
        {"task": f"T{i}", "best_of_n_demo_perfect": False, "n_demo_perfect_samples": 0, "local_seconds": 1.0}
        for i in range(30)
    ]

    latent = exp.build_terminal_artifact(rows=latent_rows, **base_kwargs)
    absent = exp.build_terminal_artifact(rows=absent_rows, **base_kwargs)
    saturated = exp.build_terminal_artifact(
        rows=latent_rows,
        **{**base_kwargs, "references": _references(oracle=0.2581)},
    )

    assert latent["local_support_diagnosis"] == "latent"
    assert latent["honest_verdict"] == "complete: decentralization_moe_cov_1_latent_distill_viable"
    assert absent["local_support_diagnosis"] == "absent"
    assert absent["honest_verdict"] == "complete: decentralization_moe_cov_0_absent_leash_holds_n30"
    assert saturated["local_support_diagnosis"] == "uninformative"
    assert saturated["honest_verdict"] == (
        "complete: decentralization_moe_cov_1_uninformative_saturated_pool"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("honest_verdict", "done", "terminal prefix"),
        ("moe_base_demo_perfect_coverage", "1.0", "bare float"),
        ("accumulated_n_tasks", True, "bare int"),
        ("coverage_delta_vs_12b", "0.1", "bare float"),
        ("bootstrap_ci95", [0.1], "2-element"),
        ("oracle_coverage", "0.6", "bare float"),
        ("local_support_diagnosis", "retired", "latent, absent, uninformative, or accumulating"),
        ("local_seconds_per_task", "1", "bare float"),
        ("mechanism", "background", "single synchronous"),
        ("model_specs", [], "must be a dict"),
        ("random_seed", False, "bare int"),
        ("reproducibility_checksum", 123, "must be a string"),
        ("missing_verifier_gaps", {}, "must be a list"),
        ("preconditions_checked", {}, "must be a list"),
        ("inference_substrate", "cached", "live_llm_inference"),
    ],
)
def test_validate_artifact_rejects_bad_terminal_schema(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    # REQ-VERIFY-4069: terminal artifacts expose typed bare fields for downstream audit.
    artifact = exp.blocked_artifact(
        "blocked_moe_base_not_cached",
        preconditions=[{"resource": "moe_base_gguf_cached", "available": False}],
        references=_references(),
        rows=[],
        output_path=tmp_path / "result.json",
        duration_s=0.1,
    )
    artifact[field] = value
    with pytest.raises(ValueError, match=message):
        exp.validate_artifact(artifact)


def test_validate_artifact_rejects_missing_required_field() -> None:
    # REQ-VERIFY-4069: malformed terminal JSON is not considered complete.
    with pytest.raises(ValueError, match="missing required field"):
        exp.validate_artifact({})
