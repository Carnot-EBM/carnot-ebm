"""Tests for Exp 4047/4048 MoE-base decentralization build.

Spec refs: REQ-VERIFY-4047, SCENARIO-VERIFY-4047.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

import exp4047_decentralization_moe_base_build as build
import experiment_4048_decentralization_moe_base_best_of_n as run4048


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
        out: list[tuple[int, str, float]] = []
        for draw_index in draw_indices:
            out.append((draw_index, self.responses[draw_index % len(self.responses)], self.seconds))
        return out


class _FakeLlama:
    def __init__(self, content: str = PLUS_ONE) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def create_chat_completion(self, messages, **kwargs):  # noqa: ANN001
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return {"choices": [{"message": {"content": self.content}}]}


class _BoomBatchSampler:
    def sample_many(self, prompt: str, draw_indices: list[int]) -> list[tuple[int, str, float]]:
        raise RuntimeError("smoke exploded")


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
        _entry(
            "T1",
            1,
            5,
            [
                _cand([[6]], votes=1, correct=True),
                _cand([[9]], votes=9, correct=False),
                _cand([[8]], votes=8, correct=False),
            ],
        ),
        _entry("T2", 2, 4, [_cand([[12]], votes=9, correct=True), _cand([[0]], 1, False)]),
        _entry(
            "T3",
            3,
            2,
            [
                _cand([[5]], votes=1, correct=True),
                _cand([[7]], votes=9, correct=False),
                _cand([[1]], votes=8, correct=False),
            ],
        ),
        _entry("T4", 1, 5, [_cand([[3]], votes=5, correct=False), _cand([[2]], 4, False)]),
    ]


def _write_pool(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"entries": entries}, fh)


def _write_codex_ref(path: Path) -> None:
    path.write_text(
        json.dumps(
            {"generator": {"total_codex_seconds": 20.0}, "n_unique_tasks": 4, "per_task": []}
        ),
        encoding="utf-8",
    )


def _cache_dir(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "models--unsloth--Qwen3.6-35B-A3B-GGUF"
    cache_dir.mkdir()
    (cache_dir / "marker").write_text("cached", encoding="utf-8")
    return cache_dir


def test_req_4047_spec_declared() -> None:
    # REQ-VERIFY-4047: OpenSpec declares the MoE build gate before code.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    assert "REQ-VERIFY-4047" in spec
    assert "SCENARIO-VERIFY-4047" in spec
    assert "experiment_4047_decentralization_moe_base_build.json" in spec
    assert "experiment_4048_decentralization_moe_base_raw.json" in spec
    assert "smoke_per_task_seconds" in spec


def test_select_moe_model_is_qwen_only(tmp_path: Path) -> None:
    # REQ-VERIFY-4047: dense Gemma is not eligible for the throughput rerun.
    chosen = run4048.select_moe_model(
        resolver=lambda hf_id: "/cache/qwen.gguf" if "Qwen3.6" in hf_id else None
    )
    assert chosen["name"] == "Qwen3.6-35B-A3B"
    assert chosen["model_key"] == "qwen35moe"
    assert chosen["model_path"] == "/cache/qwen.gguf"
    assert run4048.select_moe_model("gemma31", resolver=lambda _hf_id: "/cache/gemma.gguf") is None


def test_preconditions_block_in_order(tmp_path: Path) -> None:
    # REQ-VERIFY-4047: missing resources map to explicit blocked_<resource> verdicts.
    pool = tmp_path / "pool.json.gz"
    _write_pool(pool, _synthetic_pool())
    precs, chosen = run4048.check_preconditions(
        pool_path=pool,
        resolver=lambda _hf_id: None,
        cache_dir=tmp_path / "missing-cache",
        llama_available_override=False,
    )
    assert chosen is None
    assert run4048.blocker_from_preconditions(precs) == "blocked_moe_base_not_cached"

    cache_dir = _cache_dir(tmp_path)
    precs, chosen = run4048.check_preconditions(
        pool_path=pool,
        resolver=lambda _hf_id: "/cache/qwen.gguf",
        cache_dir=cache_dir,
        llama_available_override=False,
    )
    assert chosen["name"] == "Qwen3.6-35B-A3B"
    assert run4048.blocker_from_preconditions(precs) == "blocked_llama_cpp_unavailable"

    precs, _chosen = run4048.check_preconditions(
        pool_path=tmp_path / "missing.json.gz",
        resolver=lambda _hf_id: "/cache/qwen.gguf",
        cache_dir=cache_dir,
        llama_available_override=True,
    )
    assert run4048.blocker_from_preconditions(precs) == "blocked_exp4012_pool_unreadable"


def test_preconditions_real_llama_import_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REQ-VERIFY-4047: the live import check is recorded when no override is supplied.
    pool = tmp_path / "pool.json.gz"
    _write_pool(pool, _synthetic_pool())
    cache_dir = _cache_dir(tmp_path)
    precs, _chosen = run4048.check_preconditions(
        pool_path=pool,
        resolver=lambda _hf_id: "/cache/qwen.gguf",
        cache_dir=cache_dir,
    )
    assert {row["resource"]: row["available"] for row in precs}["llama_cpp"] is True

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # noqa: ANN001
        if name == "llama_cpp":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    precs, _chosen = run4048.check_preconditions(
        pool_path=pool,
        resolver=lambda _hf_id: "/cache/qwen.gguf",
        cache_dir=cache_dir,
    )
    assert {row["resource"]: row["available"] for row in precs}["llama_cpp"] is False


def test_batched_llama_sampler_varies_seed_and_temperature() -> None:
    # SCENARIO-VERIFY-4047: batched independent draws still vary seed and temperature.
    fake = _FakeLlama()
    sampler = run4048.BatchedIndependentLocalSampler(fake, base_seed=100, base_temperature=0.2)
    rows = sampler.sample_many("prompt", [0, 1])
    assert [row[0] for row in rows] == [0, 1]
    first, second = fake.calls
    assert first["kwargs"]["seed"] != second["kwargs"]["seed"]
    assert first["kwargs"]["temperature"] != second["kwargs"]["temperature"]
    assert first["messages"][1]["content"] == "prompt"


def test_induce_task_samples_batch_early_stops_after_demo_perfect() -> None:
    # SCENARIO-VERIFY-4047: RoBoN-style sampling stops the task once a demo-perfect rule appears.
    sampler = _BatchSampler([PLUS_ONE, IDENTITY, IDENTITY])
    samples = run4048.induce_task_samples_batched(
        "T1",
        [_demo(1, 2), _demo(3, 4)],
        sampler,
        k=8,
        batch_size=3,
    )
    assert len(samples) == 1
    assert samples[0]["demo_perfect"] is True
    assert len(sampler.calls) == 1
    assert sampler.calls[0][1] == (0, 1, 2)


def test_callable_sampler_and_grade_edges() -> None:
    # REQ-VERIFY-4047: sample grading preserves no-code and unsafe failure rows.
    unsafe = "```python\nimport os\ndef transform(grid):\n    return grid\n```"
    responses = ["no code here", unsafe, PLUS_ONE]

    def sampler(prompt: str, draw_index: int) -> tuple[str, float]:
        return responses[draw_index], 0.1

    samples = run4048.induce_task_samples_batched(
        "T1",
        [_demo(1, 2), _demo(3, 4)],
        sampler,
        k=3,
        batch_size=1,
    )
    assert [sample["status"] for sample in samples] == [
        "no_code",
        "unsafe_or_uncompilable",
        "graded",
    ]
    assert samples[-1]["demo_perfect"] is True


def test_checkpoint_resume_and_task_early_stop(tmp_path: Path) -> None:
    # REQ-VERIFY-4047: every completed task is checkpointed and reused after interruption.
    entries = [_entry("T1", 1, 5, [_cand([[6]], 1, True)])]
    checkpoint = tmp_path / "ckpt.json"
    sampler = _BatchSampler([PLUS_ONE, IDENTITY], seconds=0.1)
    first = run4048.induce_pool_best_of_n_batched(
        entries,
        sampler,
        k=8,
        checkpoint_path=checkpoint,
        model_name="Qwen3.6-35B-A3B",
    )
    assert len(first["T1"]) == 1
    assert first["T1"][0]["demo_perfect"] is True
    assert checkpoint.exists()

    class _Boom:
        def sample_many(self, prompt: str, draw_indices: list[int]) -> list[tuple[int, str, float]]:
            raise AssertionError("checkpoint was not used")

    second = run4048.induce_pool_best_of_n_batched(
        entries,
        _Boom(),
        k=8,
        checkpoint_path=checkpoint,
        model_name="Qwen3.6-35B-A3B",
    )
    assert second == first


def test_checkpoint_mismatch_none_save_and_timeout(tmp_path: Path) -> None:
    # REQ-VERIFY-4047: stale checkpoints are ignored and timeout preserves partial progress.
    entries = [_entry("T1", 1, 5, [_cand([[6]], 1, True)])]
    checkpoint = tmp_path / "ckpt.json"
    checkpoint.write_text(
        json.dumps({"k_samples_per_task": 99, "local_model_used": "old", "tasks": {"T1": []}}),
        encoding="utf-8",
    )
    timed_out = run4048.induce_pool_best_of_n_batched(
        entries,
        _BatchSampler([PLUS_ONE]),
        k=8,
        checkpoint_path=checkpoint,
        model_name="Qwen3.6-35B-A3B",
        started_s=0.0,
        max_wall_s=0.0,
    )
    assert timed_out == {}
    run4048._save_checkpoint(None, {"T1": []}, k=8, model_name="Qwen3.6-35B-A3B")


def test_run4048_blocked_writes_required_raw_fields(tmp_path: Path) -> None:
    # REQ-VERIFY-4047: blocked raw runs still emit the required schema.
    out = tmp_path / "raw.json"
    artifact = run4048.run(
        output_path=out,
        checkpoint_path=tmp_path / "ckpt.json",
        resolver=lambda _hf_id: None,
        cache_dir=tmp_path / "missing-cache",
        llama_available_override=True,
        write=True,
    )
    assert artifact["honest_verdict"] == "blocked_moe_base_not_cached"
    assert artifact["runner_ready"] is False
    assert artifact["moe_base_model"] == "none"
    assert artifact["inference_substrate"] == run4048.INFERENCE_SUBSTRATE
    run4048.validate_raw_artifact(artifact)
    assert out.exists()


def test_run4048_blocked_no_write(tmp_path: Path) -> None:
    # REQ-VERIFY-4047: blocked raw runs can be inspected without writing.
    out = tmp_path / "raw.json"
    artifact = run4048.run(
        output_path=out,
        checkpoint_path=tmp_path / "ckpt.json",
        resolver=lambda _hf_id: None,
        cache_dir=tmp_path / "missing-cache",
        llama_available_override=True,
        write=False,
    )
    assert artifact["honest_verdict"] == "blocked_moe_base_not_cached"
    assert not out.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("honest_verdict", "done", "terminal prefix"),
        ("runner_ready", "yes", "bare bool"),
        ("best_of_n_coverage", "0.5", "bare float"),
        ("launched_pid", 1.2, "bare int"),
        ("moe_base_model", 35, "must be a string"),
        ("per_task", "bad", "must be a list"),
        ("model_specs", [], "must be a dict"),
    ],
)
def test_validate_raw_artifact_rejects_bad_schema(field: str, value: Any, message: str) -> None:
    # REQ-VERIFY-4047: downstream gates receive bare, typed raw fields.
    artifact = run4048.blocked_raw_artifact(
        "blocked_moe_base_not_cached",
        chosen_model=None,
        preconditions=[{"resource": "moe_base_gguf_cached", "available": False}],
        duration_s=0.1,
    )
    artifact[field] = value
    with pytest.raises(ValueError, match=message):
        run4048.validate_raw_artifact(artifact)


def test_validate_raw_artifact_rejects_missing_field() -> None:
    # REQ-VERIFY-4047: raw artifacts cannot omit required fields.
    with pytest.raises(ValueError, match="missing required field"):
        run4048.validate_raw_artifact({})


def test_run4048_complete_with_fake_batch_sampler_writes_raw_artifact(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4047: the MoE runner mirrors 4012 scoring with unchanged verifier.
    pool = tmp_path / "pool.json.gz"
    _write_pool(pool, _synthetic_pool())
    codex = tmp_path / "codex.json"
    _write_codex_ref(codex)
    out = tmp_path / "raw.json"
    artifact = run4048.run(
        pool_path=pool,
        output_path=out,
        codex_ref_path=codex,
        checkpoint_path=tmp_path / "ckpt.json",
        k=8,
        sampler=_BatchSampler([IDENTITY, PLUS_ONE], seconds=0.25),
        resolver=lambda _hf_id: "/cache/qwen.gguf",
        cache_dir=_cache_dir(tmp_path),
        llama_available_override=True,
        write=True,
    )
    run4048.validate_raw_artifact(artifact)
    assert artifact["experiment"] == "experiment_4048_decentralization_moe_base_raw"
    assert artifact["moe_base_model"] == "Qwen3.6-35B-A3B"
    assert artifact["best_of_n_coverage"] == pytest.approx(0.5)
    assert artifact["gated_pass_at_2"] == artifact["local_gated_pass2"]
    assert artifact["k_samples_per_task"] == 8
    assert artifact["model_specs"]["generator_hf_id"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
    assert len(artifact["reproducibility_checksum"]) == 16
    assert all("local_seconds" in row for row in artifact["per_task"])
    assert out.exists()


def test_run4048_complete_no_write_and_complete_verdict_branch(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4047: write=False leaves only the returned raw artifact.
    pool = tmp_path / "pool.json.gz"
    _write_pool(pool, _synthetic_pool())
    codex = tmp_path / "codex.json"
    _write_codex_ref(codex)
    out = tmp_path / "raw.json"
    artifact = run4048.run(
        pool_path=pool,
        output_path=out,
        codex_ref_path=codex,
        checkpoint_path=tmp_path / "ckpt.json",
        k=1,
        limit=2,
        sampler=_BatchSampler([IDENTITY], seconds=0.1),
        resolver=lambda _hf_id: "/cache/qwen.gguf",
        cache_dir=_cache_dir(tmp_path),
        llama_available_override=True,
        write=False,
    )
    assert artifact["honest_verdict"].startswith("complete: decentralization_moe_base_cov0")
    assert not out.exists()


def test_run_build_blocks_before_smoke_or_launch(tmp_path: Path) -> None:
    # REQ-VERIFY-4047: Exp 4047 stops before inference when a precondition is missing.
    launched: list[object] = []
    artifact = build.run_build(
        output_path=tmp_path / "build.json",
        smoke_output_path=tmp_path / "smoke.json",
        raw_output_path=tmp_path / "raw.json",
        smoke_checkpoint_path=tmp_path / "smoke.ckpt.json",
        full_checkpoint_path=tmp_path / "full.ckpt.json",
        log_path=tmp_path / "run.log",
        resolver=lambda _hf_id: None,
        cache_dir=tmp_path / "missing-cache",
        llama_available_override=True,
        launcher=lambda spec: launched.append(spec) or 99,
        write=True,
    )
    build.validate_build_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_moe_base_not_cached"
    assert artifact["runner_ready"] is False
    assert artifact["smoke_passed"] is False
    assert artifact["smoke_per_task_seconds"] == 0.0
    assert artifact["launched_pid"] == 0
    assert launched == []
    assert Path(artifact["build_artifact_path"]).exists()
    assert not (tmp_path / "smoke.json").exists()


def test_run_build_blocks_without_writing_when_requested(tmp_path: Path) -> None:
    # REQ-VERIFY-4047: tests can exercise the blocked path without mutating an artifact.
    out = tmp_path / "build.json"
    artifact = build.run_build(
        output_path=out,
        smoke_output_path=tmp_path / "smoke.json",
        raw_output_path=tmp_path / "raw.json",
        smoke_checkpoint_path=tmp_path / "smoke.ckpt.json",
        full_checkpoint_path=tmp_path / "full.ckpt.json",
        log_path=tmp_path / "run.log",
        resolver=lambda _hf_id: None,
        cache_dir=tmp_path / "missing-cache",
        llama_available_override=True,
        write=False,
    )
    assert artifact["honest_verdict"] == "blocked_moe_base_not_cached"
    assert not out.exists()


def test_run_build_blocks_when_smoke_runner_fails(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4047: failed smoke prevents the full background launch.
    pool = tmp_path / "pool.json.gz"
    _write_pool(pool, _synthetic_pool())
    launched: list[object] = []
    artifact = build.run_build(
        pool_path=pool,
        output_path=tmp_path / "build.json",
        smoke_output_path=tmp_path / "smoke.json",
        raw_output_path=tmp_path / "raw.json",
        smoke_checkpoint_path=tmp_path / "smoke.ckpt.json",
        full_checkpoint_path=tmp_path / "full.ckpt.json",
        log_path=tmp_path / "run.log",
        resolver=lambda _hf_id: "/cache/qwen.gguf",
        cache_dir=_cache_dir(tmp_path),
        llama_available_override=True,
        smoke_sampler=_BoomBatchSampler(),
        launcher=lambda spec: launched.append(spec) or 99,
        write=True,
    )
    build.validate_build_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_smoke_failed"
    assert artifact["smoke_passed"] is False
    assert artifact["launched_pid"] == 0
    assert "smoke exploded" in artifact["smoke_error"]
    assert launched == []


def test_run_build_smoke_failure_no_write(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4047: smoke failures also support no-write callers.
    pool = tmp_path / "pool.json.gz"
    _write_pool(pool, _synthetic_pool())
    out = tmp_path / "build.json"
    artifact = build.run_build(
        pool_path=pool,
        output_path=out,
        smoke_output_path=tmp_path / "smoke.json",
        raw_output_path=tmp_path / "raw.json",
        smoke_checkpoint_path=tmp_path / "smoke.ckpt.json",
        full_checkpoint_path=tmp_path / "full.ckpt.json",
        log_path=tmp_path / "run.log",
        resolver=lambda _hf_id: "/cache/qwen.gguf",
        cache_dir=_cache_dir(tmp_path),
        llama_available_override=True,
        smoke_sampler=_BoomBatchSampler(),
        write=False,
    )
    assert artifact["honest_verdict"] == "blocked_smoke_failed"
    assert not out.exists()


def test_run_build_smokes_and_launches_full_runner(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4047: a valid build writes the MoE launch receipt and records the PID.
    pool = tmp_path / "pool.json.gz"
    _write_pool(pool, _synthetic_pool())
    launched: list[build.LaunchSpec] = []

    def fake_launcher(spec: build.LaunchSpec) -> int:
        launched.append(spec)
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        spec.log_path.write_text("launched\n", encoding="utf-8")
        return 4048

    artifact = build.run_build(
        pool_path=pool,
        output_path=tmp_path / "build.json",
        smoke_output_path=tmp_path / "smoke.json",
        raw_output_path=tmp_path / "raw.json",
        smoke_checkpoint_path=tmp_path / "smoke.ckpt.json",
        full_checkpoint_path=tmp_path / "full.ckpt.json",
        log_path=tmp_path / "run.log",
        resolver=lambda _hf_id: "/cache/qwen.gguf",
        cache_dir=_cache_dir(tmp_path),
        llama_available_override=True,
        smoke_sampler=_BatchSampler([PLUS_ONE, PLUS_TWO], seconds=0.1),
        launcher=fake_launcher,
        smoke_k=8,
        full_k=8,
        full_time_budget_s=4500.0,
        write=True,
    )
    build.validate_build_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "success: decentralization_moe_base_runner_launched_qwen35moe"
    )
    assert artifact["runner_ready"] is True
    assert artifact["moe_base_model"] == "Qwen3.6-35B-A3B"
    assert artifact["smoke_passed"] is True
    assert artifact["smoke_per_task_seconds"] == pytest.approx(0.15)
    assert artifact["launched_pid"] == 4048
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert Path(artifact["smoke_artifact_path"]).exists()
    assert launched and launched[0].k == 8
    assert launched[0].max_wall_s == 4500.0
    assert "nohup" in artifact["launch_command"]
    assert "experiment_4048_decentralization_moe_base_best_of_n.py" in " ".join(launched[0].argv)


def test_run_build_success_no_write_and_smoke_seconds_fallback(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4047: build success supports no-write callers and fallback timing math.
    assert build._smoke_seconds_per_task({"local_seconds": 3.0, "n_unique_tasks": 2}) == 1.5
    pool = tmp_path / "pool.json.gz"
    _write_pool(pool, _synthetic_pool())
    out = tmp_path / "build.json"
    artifact = build.run_build(
        pool_path=pool,
        output_path=out,
        smoke_output_path=tmp_path / "smoke.json",
        raw_output_path=tmp_path / "raw.json",
        smoke_checkpoint_path=tmp_path / "smoke.ckpt.json",
        full_checkpoint_path=tmp_path / "full.ckpt.json",
        log_path=tmp_path / "run.log",
        resolver=lambda _hf_id: "/cache/qwen.gguf",
        cache_dir=_cache_dir(tmp_path),
        llama_available_override=True,
        smoke_sampler=_BatchSampler([PLUS_ONE, PLUS_TWO], seconds=0.1),
        launcher=lambda _spec: 4049,
        write=False,
    )
    assert artifact["runner_ready"] is True
    assert not out.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("honest_verdict", "done", "terminal prefix"),
        ("runner_ready", "true", "bare bool"),
        ("smoke_passed", "true", "bare bool"),
        ("smoke_per_task_seconds", "0.1", "bare float"),
        ("launched_pid", "4048", "bare int"),
        ("moe_base_model", 35, "must be a string"),
        ("preconditions_checked", {}, "must be a list"),
        ("duration_s", "0.1", "bare float"),
    ],
)
def test_validate_build_artifact_rejects_bad_schema(field: str, value: Any, message: str) -> None:
    # REQ-VERIFY-4047: the build gate emits bare scalar fields for conductor gating.
    artifact = build.blocked_build_artifact(
        "blocked_moe_base_not_cached",
        chosen_model=None,
        preconditions=[{"resource": "moe_base_gguf_cached", "available": False}],
        duration_s=0.1,
        output_path=Path("results/experiment_4047_decentralization_moe_base_build.json"),
    )
    artifact[field] = value
    with pytest.raises(ValueError, match=message):
        build.validate_build_artifact(artifact)


def test_validate_build_artifact_rejects_missing_field() -> None:
    # REQ-VERIFY-4047: build artifacts cannot omit conductor-gated fields.
    with pytest.raises(ValueError, match="missing required field"):
        build.validate_build_artifact({})
