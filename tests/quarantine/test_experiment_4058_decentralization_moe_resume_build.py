"""Tests for Exp 4058/4059 MoE resume-and-accumulate build.

Spec refs: REQ-VERIFY-4058, SCENARIO-VERIFY-4058.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

import exp4058_decentralization_moe_resume_build as build
import experiment_4059_decentralization_moe_resume_best_of_n as run4059


PLUS_ONE = "```python\ndef transform(grid):\n    return grid + 1\n```"
PLUS_TWO = "```python\ndef transform(grid):\n    return grid + 2\n```"
IDENTITY = "```python\ndef transform(grid):\n    return grid\n```"


class _BatchSampler:
    def __init__(self, responses: list[str], seconds: float = 0.2) -> None:
        self.responses = responses
        self.seconds = seconds
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def sample_many(self, prompt: str, draw_indices: list[int]) -> list[tuple[int, str, float]]:
        self.calls.append((prompt, tuple(draw_indices)))
        rows = []
        for draw_index in draw_indices:
            rows.append((draw_index, self.responses[draw_index % len(self.responses)], self.seconds))
        return rows


class _BoomBatchSampler:
    def sample_many(self, prompt: str, draw_indices: list[int]) -> list[tuple[int, str, float]]:
        raise RuntimeError("smoke failed")


def _demo(value_in: int, value_out: int) -> dict[str, Any]:
    return {"input": [[value_in]], "output": [[value_out]]}


def _cand(grid: list[list[int]], votes: int, correct: bool, q: float = 0.5) -> dict[str, Any]:
    return {"grid": grid, "votes": votes, "correct": correct, "q_mean": q}


def _entry(
    task: str,
    rule_delta: int,
    test_in: int,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "task": task,
        "demos": [_demo(1, 1 + rule_delta), _demo(3, 3 + rule_delta)],
        "test_input": [[test_in]],
        "candidates": candidates,
    }


def _synthetic_pool() -> list[dict[str, Any]]:
    return [
        _entry("T1", 1, 5, [_cand([[6]], votes=1, correct=True), _cand([[0]], 9, False)]),
        _entry("T2", 2, 4, [_cand([[6]], votes=1, correct=True), _cand([[0]], 9, False)]),
        _entry("T3", 1, 8, [_cand([[9]], votes=1, correct=True), _cand([[0]], 9, False)]),
        _entry("T4", 2, 2, [_cand([[4]], votes=1, correct=True), _cand([[0]], 9, False)]),
    ]


def _write_pool(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"entries": entries}, fh)


def _cache_dir(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "models--unsloth--Qwen3.6-35B-A3B-GGUF"
    cache_dir.mkdir()
    (cache_dir / "marker").write_text("cached", encoding="utf-8")
    return cache_dir


def _source_checkpoint(path: Path, *, tasks: dict[str, list[dict[str, Any]]] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "carnot.experiment_4048_decentralization_moe_base_raw.checkpoint.v1",
        "k_samples_per_task": 8,
        "local_model_used": "Qwen3.6-35B-A3B",
        "tasks": tasks
        or {
            "T1": [
                {
                    "task": "T1",
                    "draw_index": 0,
                    "status": "graded",
                    "demo_fit": 1.0,
                    "demo_perfect": True,
                    "local_s": 1.0,
                    "code": "def transform(grid):\n    return grid + 1\n",
                }
            ]
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_req_4058_spec_declared() -> None:
    # REQ-VERIFY-4058: OpenSpec declares resume-and-accumulate before code.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    assert "REQ-VERIFY-4058" in spec
    assert "SCENARIO-VERIFY-4058" in spec
    assert "decentralization_moe_qwen35a3b_arc1_k8.checkpoint.json" in spec
    assert "experiment_4058_decentralization_moe_resume_build.json" in spec
    assert "ACCUMULATED-N" in spec


def test_preconditions_require_exp4048_checkpoint_after_pool(tmp_path: Path) -> None:
    # REQ-VERIFY-4058: missing source checkpoints block before inference.
    pool = tmp_path / "pool.json.gz"
    _write_pool(pool, _synthetic_pool())
    cache_dir = _cache_dir(tmp_path)

    precs, chosen, resumed = run4059.check_preconditions(
        pool_path=pool,
        source_checkpoint_path=tmp_path / "missing.ckpt.json",
        resolver=lambda _hf_id: "/cache/qwen.gguf",
        cache_dir=cache_dir,
        llama_available_override=True,
    )
    assert chosen["name"] == "Qwen3.6-35B-A3B"
    assert resumed == 0
    assert run4059.blocker_from_preconditions(precs) == "blocked_exp4048_checkpoint_unreadable"

    precs, _chosen, _resumed = run4059.check_preconditions(
        pool_path=tmp_path / "missing_pool.json.gz",
        source_checkpoint_path=tmp_path / "missing.ckpt.json",
        resolver=lambda _hf_id: "/cache/qwen.gguf",
        cache_dir=cache_dir,
        llama_available_override=True,
    )
    assert run4059.blocker_from_preconditions(precs) == "blocked_exp4012_pool_unreadable"


def test_checkpoint_merge_resumes_source_without_restart(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4058: stable checkpoint is initialized from Exp 4048 and then merged.
    source = tmp_path / "exp4048.ckpt.json"
    stable = tmp_path / "stable.ckpt.json"
    _source_checkpoint(source)
    stable.write_text(
        json.dumps(
            {
                "schema": "carnot.experiment_4059_decentralization_moe_resume.checkpoint.v1",
                "k_samples_per_task": 8,
                "local_model_used": "Qwen3.6-35B-A3B",
                "tasks": {
                    "T2": [
                        {
                            "task": "T2",
                            "draw_index": 0,
                            "status": "graded",
                            "demo_fit": 1.0,
                            "demo_perfect": True,
                            "local_s": 2.0,
                            "code": "def transform(grid):\n    return grid + 2\n",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    merged, resumed_from_n = run4059.ensure_stable_checkpoint(
        source_checkpoint_path=source,
        stable_checkpoint_path=stable,
        k=8,
        model_name="Qwen3.6-35B-A3B",
    )

    assert resumed_from_n == 1
    assert sorted(merged) == ["T1", "T2"]
    on_disk = json.loads(stable.read_text(encoding="utf-8"))
    assert sorted(on_disk["tasks"]) == ["T1", "T2"]
    assert on_disk["stable_checkpoint_key"] == "arc1:qwen35a3b:k8"


def test_run4059_accumulates_next_unscored_tasks_and_writes_raw(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4058: resume preserves old tasks and smokes two new tasks only.
    pool = tmp_path / "pool.json.gz"
    source = tmp_path / "source.ckpt.json"
    stable = tmp_path / "stable.ckpt.json"
    out = tmp_path / "raw.json"
    _write_pool(pool, _synthetic_pool())
    _source_checkpoint(source)
    sampler = _BatchSampler([PLUS_TWO, PLUS_ONE], seconds=0.5)

    artifact = run4059.run(
        pool_path=pool,
        output_path=out,
        source_checkpoint_path=source,
        stable_checkpoint_path=stable,
        k=8,
        max_new_tasks=2,
        sampler=sampler,
        resolver=lambda _hf_id: "/cache/qwen.gguf",
        cache_dir=_cache_dir(tmp_path),
        llama_available_override=True,
        write=True,
    )

    run4059.validate_raw_artifact(artifact)
    assert artifact["runner_ready"] is True
    assert artifact["resumed_from_n"] == 1
    assert artifact["ACCUMULATED-N"] == 3
    assert artifact["stable_checkpoint_path"] == str(stable)
    assert sorted(json.loads(stable.read_text(encoding="utf-8"))["tasks"]) == ["T1", "T2", "T3"]
    assert len(sampler.calls) == 2
    assert artifact["best_of_n_coverage"] > 0.0
    assert out.exists()


def test_run4059_blocked_writes_required_raw_fields(tmp_path: Path) -> None:
    # REQ-VERIFY-4058: blocked resume runs still emit a typed raw artifact.
    artifact = run4059.run(
        output_path=tmp_path / "raw.json",
        source_checkpoint_path=tmp_path / "missing_source.ckpt.json",
        stable_checkpoint_path=tmp_path / "stable.ckpt.json",
        resolver=lambda _hf_id: None,
        cache_dir=tmp_path / "missing-cache",
        llama_available_override=True,
        write=False,
    )
    assert artifact["honest_verdict"] == "blocked_moe_base_not_cached"
    assert artifact["runner_ready"] is False
    assert artifact["ACCUMULATED-N"] == 0
    run4059.validate_raw_artifact(artifact)


def test_run_build_blocks_before_smoke_or_launch(tmp_path: Path) -> None:
    # REQ-VERIFY-4058: Exp 4058 stops before inference when a precondition is missing.
    launched: list[object] = []
    artifact = build.run_build(
        output_path=tmp_path / "build.json",
        smoke_output_path=tmp_path / "smoke.json",
        raw_output_path=tmp_path / "raw.json",
        source_checkpoint_path=tmp_path / "missing_source.ckpt.json",
        stable_checkpoint_path=tmp_path / "stable.ckpt.json",
        log_path=tmp_path / "run.log",
        resolver=lambda _hf_id: None,
        cache_dir=tmp_path / "missing-cache",
        llama_available_override=True,
        launcher=lambda spec: launched.append(spec) or 4059,
        write=True,
    )

    build.validate_build_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_moe_base_not_cached"
    assert artifact["runner_ready"] is False
    assert artifact["resumed_from_n"] == 0
    assert artifact["smoke_passed"] is False
    assert artifact["launched_pid"] == 0
    assert launched == []


def test_run_build_blocks_when_smoke_fails(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4058: failed smoke prevents the resume background launch.
    pool = tmp_path / "pool.json.gz"
    source = tmp_path / "source.ckpt.json"
    _write_pool(pool, _synthetic_pool())
    _source_checkpoint(source)
    launched: list[object] = []

    artifact = build.run_build(
        pool_path=pool,
        output_path=tmp_path / "build.json",
        smoke_output_path=tmp_path / "smoke.json",
        raw_output_path=tmp_path / "raw.json",
        source_checkpoint_path=source,
        stable_checkpoint_path=tmp_path / "stable.ckpt.json",
        log_path=tmp_path / "run.log",
        resolver=lambda _hf_id: "/cache/qwen.gguf",
        cache_dir=_cache_dir(tmp_path),
        llama_available_override=True,
        smoke_sampler=_BoomBatchSampler(),
        launcher=lambda spec: launched.append(spec) or 4059,
        write=True,
    )

    build.validate_build_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_smoke_failed"
    assert artifact["smoke_passed"] is False
    assert artifact["launched_pid"] == 0
    assert "smoke failed" in artifact["smoke_error"]
    assert launched == []


def test_run_build_smokes_and_launches_resume_runner(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4058: a valid build writes the launch receipt and records the PID.
    pool = tmp_path / "pool.json.gz"
    source = tmp_path / "source.ckpt.json"
    stable = tmp_path / "stable.ckpt.json"
    _write_pool(pool, _synthetic_pool())
    _source_checkpoint(source)
    launched: list[build.LaunchSpec] = []

    def fake_launcher(spec: build.LaunchSpec) -> int:
        launched.append(spec)
        return 4059

    artifact = build.run_build(
        pool_path=pool,
        output_path=tmp_path / "build.json",
        smoke_output_path=tmp_path / "smoke.json",
        raw_output_path=tmp_path / "raw.json",
        source_checkpoint_path=source,
        stable_checkpoint_path=stable,
        log_path=tmp_path / "run.log",
        resolver=lambda _hf_id: "/cache/qwen.gguf",
        cache_dir=_cache_dir(tmp_path),
        llama_available_override=True,
        smoke_sampler=_BatchSampler([PLUS_TWO, PLUS_ONE], seconds=0.5),
        launcher=fake_launcher,
        write=True,
    )

    build.validate_build_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "success: decentralization_moe_resume_runner_launched_qwen35moe"
    )
    assert artifact["runner_ready"] is True
    assert artifact["moe_base_model"] == "Qwen3.6-35B-A3B"
    assert artifact["stable_checkpoint_path"] == str(stable)
    assert artifact["resumed_from_n"] == 1
    assert artifact["smoke_passed"] is True
    assert artifact["smoke_per_task_seconds"] == pytest.approx(0.5)
    assert artifact["launched_pid"] == 4059
    assert launched and launched[0].k == 8
    assert "setsid" in artifact["launch_command"]
    assert "nohup" in artifact["launch_command"]
    assert "experiment_4059_decentralization_moe_resume_best_of_n.py" in " ".join(
        launched[0].argv
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("honest_verdict", "done", "terminal prefix"),
        ("runner_ready", "true", "bare bool"),
        ("ACCUMULATED-N", True, "bare int"),
        ("resumed_from_n", True, "bare int"),
        ("best_of_n_coverage", "0.5", "bare float"),
        ("stable_checkpoint_path", 123, "must be a string"),
        ("preconditions_checked", {}, "must be a list"),
        ("model_specs", [], "must be a dict"),
    ],
)
def test_validate_raw_artifact_rejects_bad_schema(
    field: str, value: Any, message: str
) -> None:
    # REQ-VERIFY-4058: the raw artifact exposes bare fields for accumulation gates.
    artifact = run4059.blocked_raw_artifact(
        "blocked_moe_base_not_cached",
        chosen_model=None,
        preconditions=[{"resource": "moe_base_gguf_cached", "available": False}],
        duration_s=0.1,
        stable_checkpoint_path=Path("results/decentralization_moe_qwen35a3b_arc1_k8.checkpoint.json"),
    )
    artifact[field] = value
    with pytest.raises(ValueError, match=message):
        run4059.validate_raw_artifact(artifact)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("honest_verdict", "done", "terminal prefix"),
        ("runner_ready", "true", "bare bool"),
        ("resumed_from_n", True, "bare int"),
        ("smoke_per_task_seconds", "0.1", "bare float"),
        ("launched_pid", "4059", "bare int"),
        ("stable_checkpoint_path", 123, "must be a string"),
    ],
)
def test_validate_build_artifact_rejects_bad_schema(
    field: str, value: Any, message: str
) -> None:
    # REQ-VERIFY-4058: the build artifact keeps conductor-gated values bare.
    artifact = build.blocked_build_artifact(
        "blocked_moe_base_not_cached",
        chosen_model=None,
        preconditions=[{"resource": "moe_base_gguf_cached", "available": False}],
        duration_s=0.1,
        output_path=Path("results/experiment_4058_decentralization_moe_resume_build.json"),
        stable_checkpoint_path=Path("results/decentralization_moe_qwen35a3b_arc1_k8.checkpoint.json"),
    )
    artifact[field] = value
    with pytest.raises(ValueError, match=message):
        build.validate_build_artifact(artifact)

