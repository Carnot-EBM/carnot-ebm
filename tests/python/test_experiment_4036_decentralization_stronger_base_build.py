"""Tests for Exp 4036/4037 stronger-base decentralization build.

Spec refs: REQ-VERIFY-4036, SCENARIO-VERIFY-4036.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

import exp4036_decentralization_stronger_base_build as build
import experiment_4037_decentralization_stronger_base_best_of_n as run4037


PLUS_ONE = "```python\ndef transform(grid):\n    return grid + 1\n```"
IDENTITY = "```python\ndef transform(grid):\n    return grid\n```"


class _SeqSampler:
    def __init__(self, responses: list[str], seconds: float = 0.2) -> None:
        self.responses = responses
        self.seconds = seconds
        self.calls: list[tuple[str, int]] = []

    def __call__(self, prompt: str, draw_index: int) -> tuple[str, float]:
        self.calls.append((prompt, draw_index))
        return self.responses[draw_index % len(self.responses)], self.seconds


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
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"entries": entries}, fh)


def _resolver(hits: dict[str, str]):
    return lambda hf_id: hits.get(hf_id)


def test_req_4036_spec_declared() -> None:
    # REQ-VERIFY-4036: OpenSpec declares the stronger-base build gate before code.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    assert "REQ-VERIFY-4036" in spec
    assert "SCENARIO-VERIFY-4036" in spec
    assert "experiment_4036_decentralization_stronger_base_build.json" in spec
    assert "experiment_4037_decentralization_stronger_base_raw.json" in spec
    assert "runner_ready" in spec


def test_select_stronger_model_prefers_gemma31_then_qwen35() -> None:
    # REQ-VERIFY-4036: choose the cached faster stronger base before the Qwen fallback.
    hits = {
        "unsloth/gemma-4-31B-it-GGUF": "/cache/gemma31.gguf",
        "unsloth/Qwen3.6-35B-A3B-GGUF": "/cache/qwen35.gguf",
    }
    chosen = run4037.select_stronger_model("auto", resolver=_resolver(hits))
    assert chosen["name"] == "Gemma4-31B-it"
    assert chosen["model_path"] == "/cache/gemma31.gguf"

    chosen = run4037.select_stronger_model(
        "auto",
        resolver=_resolver({"unsloth/Qwen3.6-35B-A3B-GGUF": "/cache/qwen35.gguf"}),
    )
    assert chosen["name"] == "Qwen3.6-35B-A3B"
    assert run4037.select_stronger_model("bogus", resolver=lambda _hf_id: "/cache/x.gguf") is None


def test_preconditions_block_in_order(tmp_path: Path) -> None:
    # REQ-VERIFY-4036: missing resources map to explicit blocked_<resource> verdicts.
    pool = tmp_path / "pool.json.gz"
    _write_pool(pool, _synthetic_pool())
    precs, chosen = run4037.check_preconditions(
        model_key="auto",
        pool_path=pool,
        resolver=lambda _hf_id: None,
        llama_available_override=False,
    )
    assert chosen is None
    assert run4037.blocker_from_preconditions(precs) == "blocked_stronger_base_not_cached"

    precs, chosen = run4037.check_preconditions(
        model_key="gemma31",
        pool_path=pool,
        resolver=lambda _hf_id: "/cache/gemma31.gguf",
        llama_available_override=False,
    )
    assert chosen["name"] == "Gemma4-31B-it"
    assert run4037.blocker_from_preconditions(precs) == "blocked_llama_cpp_unavailable"

    precs, _chosen = run4037.check_preconditions(
        model_key="gemma31",
        pool_path=tmp_path / "missing.json.gz",
        resolver=lambda _hf_id: "/cache/gemma31.gguf",
        llama_available_override=True,
    )
    assert run4037.blocker_from_preconditions(precs) == "blocked_exp4012_pool_unreadable"


def test_preconditions_real_llama_import_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REQ-VERIFY-4036: the live import check is recorded when no override is supplied.
    pool = tmp_path / "pool.json.gz"
    _write_pool(pool, _synthetic_pool())
    precs, _chosen = run4037.check_preconditions(
        model_key="gemma31",
        pool_path=pool,
        resolver=lambda _hf_id: "/cache/gemma31.gguf",
    )
    assert {row["resource"]: row["available"] for row in precs}["llama_cpp"] is True

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # noqa: ANN001
        if name == "llama_cpp":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    precs, _chosen = run4037.check_preconditions(
        model_key="gemma31",
        pool_path=pool,
        resolver=lambda _hf_id: "/cache/gemma31.gguf",
    )
    assert {row["resource"]: row["available"] for row in precs}["llama_cpp"] is False


def test_run4037_blocked_writes_required_raw_fields(tmp_path: Path) -> None:
    # REQ-VERIFY-4036: blocked raw runs still emit the required schema.
    out = tmp_path / "raw.json"
    artifact = run4037.run(
        output_path=out,
        checkpoint_path=tmp_path / "ckpt.json",
        resolver=lambda _hf_id: None,
        llama_available_override=True,
        write=True,
    )
    assert artifact["honest_verdict"] == "blocked_stronger_base_not_cached"
    assert artifact["runner_ready"] is False
    assert artifact["stronger_base_model"] == "none"
    assert artifact["inference_substrate"] == run4037.INFERENCE_SUBSTRATE
    run4037.validate_raw_artifact(artifact)
    assert out.exists()


def test_run4037_complete_with_fake_sampler_writes_raw_artifact(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4036: the stronger-base runner mirrors 4012 scoring with unchanged verifier.
    pool = tmp_path / "pool.json.gz"
    _write_pool(pool, _synthetic_pool())
    codex = tmp_path / "codex.json"
    codex.write_text(
        json.dumps(
            {"generator": {"total_codex_seconds": 20.0}, "n_unique_tasks": 4, "per_task": []}
        )
    )
    out = tmp_path / "raw.json"
    artifact = run4037.run(
        pool_path=pool,
        output_path=out,
        codex_ref_path=codex,
        checkpoint_path=tmp_path / "ckpt.json",
        k=2,
        sampler=_SeqSampler([IDENTITY, PLUS_ONE], seconds=0.25),
        resolver=lambda _hf_id: "/cache/gemma31.gguf",
        llama_available_override=True,
        write=True,
    )
    run4037.validate_raw_artifact(artifact)
    assert artifact["experiment"] == "experiment_4037_decentralization_stronger_base_raw"
    assert artifact["stronger_base_model"] == "Gemma4-31B-it"
    assert artifact["local_model_used"] == "Gemma4-31B-it"
    assert artifact["best_of_n_coverage"] == pytest.approx(0.5)
    assert artifact["local_demo_perfect_coverage_bestofn"] == pytest.approx(0.5)
    assert artifact["gated_pass_at_2"] == artifact["local_gated_pass2"]
    assert artifact["k_samples_per_task"] == 2
    assert artifact["model_specs"]["generator_hf_id"] == "unsloth/gemma-4-31B-it-GGUF"
    assert len(artifact["reproducibility_checksum"]) == 16
    assert all("local_seconds" in row for row in artifact["per_task"])
    assert out.exists()


def test_raw_success_verdict_names_latent_support() -> None:
    # SCENARIO-VERIFY-4036: positive raw runs report latent support in the terminal verdict.
    verdict = run4037._raw_verdict(True, 0.9, 0.58, "Gemma4-31B-it")
    assert verdict.startswith("success: decentralization_stronger_base_latent_support_cov0.9")
    complete = run4037._raw_verdict(False, 0.25, 0.45, "Gemma4-31B-it")
    assert complete == "complete: decentralization_stronger_base_cov0.25_pass20.45_below_codex"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("honest_verdict", "done", "terminal prefix"),
        ("runner_ready", "yes", "bare bool"),
        ("best_of_n_coverage", "0.5", "bare float"),
        ("launched_pid", 1.2, "bare int"),
        ("stronger_base_model", 31, "must be a string"),
        ("per_task", "bad", "must be a list"),
        ("model_specs", [], "must be a dict"),
    ],
)
def test_validate_raw_artifact_rejects_bad_schema(field: str, value: Any, message: str) -> None:
    # REQ-VERIFY-4036: downstream gates receive bare, typed schema fields.
    artifact = run4037.blocked_raw_artifact(
        "blocked_stronger_base_not_cached",
        chosen_model=None,
        preconditions=[{"resource": "stronger_base_gguf_cached", "available": False}],
        duration_s=0.1,
    )
    artifact[field] = value
    with pytest.raises(ValueError, match=message):
        run4037.validate_raw_artifact(artifact)


def test_validate_raw_artifact_rejects_missing_field() -> None:
    # REQ-VERIFY-4036: raw artifacts cannot omit required fields.
    with pytest.raises(ValueError, match="missing required field"):
        run4037.validate_raw_artifact({})


def test_run_build_blocks_before_smoke_or_launch(tmp_path: Path) -> None:
    # REQ-VERIFY-4036: Exp 4036 stops before inference when a precondition is missing.
    launched: list[object] = []
    artifact = build.run_build(
        output_path=tmp_path / "build.json",
        smoke_output_path=tmp_path / "smoke.json",
        raw_output_path=tmp_path / "raw.json",
        smoke_checkpoint_path=tmp_path / "smoke.ckpt.json",
        full_checkpoint_path=tmp_path / "full.ckpt.json",
        log_path=tmp_path / "run.log",
        resolver=lambda _hf_id: None,
        llama_available_override=True,
        launcher=lambda spec: launched.append(spec) or 99,
        write=True,
    )
    build.validate_build_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_stronger_base_not_cached"
    assert artifact["runner_ready"] is False
    assert artifact["smoke_passed"] is False
    assert artifact["launched_pid"] == 0
    assert launched == []
    assert Path(artifact["build_artifact_path"]).exists()
    assert not (tmp_path / "smoke.json").exists()


def test_run_build_blocks_when_smoke_runner_fails(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4036: failed smoke prevents the full background launch.
    pool = tmp_path / "pool.json.gz"
    _write_pool(pool, _synthetic_pool())

    class _BoomSampler:
        def __call__(self, prompt: str, draw_index: int) -> tuple[str, float]:
            raise RuntimeError("smoke exploded")

    launched: list[object] = []
    artifact = build.run_build(
        pool_path=pool,
        output_path=tmp_path / "build.json",
        smoke_output_path=tmp_path / "smoke.json",
        raw_output_path=tmp_path / "raw.json",
        smoke_checkpoint_path=tmp_path / "smoke.ckpt.json",
        full_checkpoint_path=tmp_path / "full.ckpt.json",
        log_path=tmp_path / "run.log",
        resolver=lambda _hf_id: "/cache/gemma31.gguf",
        llama_available_override=True,
        smoke_sampler=_BoomSampler(),
        launcher=lambda spec: launched.append(spec) or 99,
        write=True,
    )
    build.validate_build_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_smoke_failed"
    assert artifact["smoke_passed"] is False
    assert artifact["launched_pid"] == 0
    assert "smoke exploded" in artifact["smoke_error"]
    assert launched == []


def test_run_build_smokes_and_launches_full_runner(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4036: a valid build writes the build artifact and records the nohup PID.
    pool = tmp_path / "pool.json.gz"
    _write_pool(pool, _synthetic_pool())
    launched: list[build.LaunchSpec] = []

    def fake_launcher(spec: build.LaunchSpec) -> int:
        launched.append(spec)
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        spec.log_path.write_text("launched\n", encoding="utf-8")
        return 4242

    artifact = build.run_build(
        pool_path=pool,
        output_path=tmp_path / "build.json",
        smoke_output_path=tmp_path / "smoke.json",
        raw_output_path=tmp_path / "raw.json",
        smoke_checkpoint_path=tmp_path / "smoke.ckpt.json",
        full_checkpoint_path=tmp_path / "full.ckpt.json",
        log_path=tmp_path / "run.log",
        resolver=lambda _hf_id: "/cache/gemma31.gguf",
        llama_available_override=True,
        smoke_sampler=_SeqSampler([PLUS_ONE], seconds=0.1),
        launcher=fake_launcher,
        smoke_k=1,
        full_k=8,
        full_time_budget_s=4500.0,
        write=True,
    )
    build.validate_build_artifact(artifact)
    assert (
        artifact["honest_verdict"]
        == "success: decentralization_stronger_base_runner_launched_Gemma4-31B-it"
    )
    assert artifact["runner_ready"] is True
    assert artifact["stronger_base_model"] == "Gemma4-31B-it"
    assert artifact["smoke_passed"] is True
    assert artifact["launched_pid"] == 4242
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert Path(artifact["smoke_artifact_path"]).exists()
    assert launched and launched[0].k == 8
    assert launched[0].max_wall_s == 4500.0
    assert "--checkpoint" in launched[0].argv
    assert "nohup" in artifact["launch_command"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("honest_verdict", "done", "terminal prefix"),
        ("runner_ready", "true", "bare bool"),
        ("smoke_passed", "true", "bare bool"),
        ("launched_pid", "4242", "bare int"),
        ("stronger_base_model", 31, "must be a string"),
        ("preconditions_checked", {}, "must be a list"),
        ("duration_s", "0.1", "bare float"),
    ],
)
def test_validate_build_artifact_rejects_bad_schema(field: str, value: Any, message: str) -> None:
    # REQ-VERIFY-4036: the build gate emits bare scalar fields for conductor gating.
    artifact = build.blocked_build_artifact(
        "blocked_stronger_base_not_cached",
        chosen_model=None,
        preconditions=[{"resource": "stronger_base_gguf_cached", "available": False}],
        duration_s=0.1,
        output_path=Path("results/experiment_4036_decentralization_stronger_base_build.json"),
    )
    artifact[field] = value
    with pytest.raises(ValueError, match=message):
        build.validate_build_artifact(artifact)


def test_validate_build_artifact_rejects_missing_field() -> None:
    # REQ-VERIFY-4036: build artifacts cannot omit conductor-gated fields.
    with pytest.raises(ValueError, match="missing required field"):
        build.validate_build_artifact({})
