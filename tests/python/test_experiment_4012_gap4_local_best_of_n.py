"""Tests for Exp 4012 GAP-4 LOCAL best-of-N generator arm.

Spec refs: REQ-VERIFY-4012, SCENARIO-VERIFY-4012.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pytest

import experiment_4012_gap4_local_best_of_n as exp


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


class _FakeLlama:
    def __init__(self, content: str = PLUS_ONE) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def create_chat_completion(self, messages, **kwargs):  # noqa: ANN001
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return {"choices": [{"message": {"content": self.content}}]}


def _demo(value_in: int, value_out: int) -> dict[str, Any]:
    return {"input": [[value_in]], "output": [[value_out]]}


def _cand(grid: list[list[int]], votes: int, correct: bool, q: float = 0.5) -> dict[str, Any]:
    return {"grid": grid, "votes": votes, "correct": correct, "q_mean": q}


def _entry(task: str, rule_delta: int, test_in: int, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task": task,
        "demos": [_demo(1, 1 + rule_delta), _demo(3, 3 + rule_delta)],
        "test_input": [[test_in]],
        "candidates": candidates,
    }


def _synthetic_pool() -> list[dict[str, Any]]:
    return [
        _entry("T1", 1, 5, [
            _cand([[6]], votes=1, correct=True),
            _cand([[9]], votes=9, correct=False),
            _cand([[8]], votes=8, correct=False),
        ]),
        _entry("T2", 2, 4, [
            _cand([[12]], votes=9, correct=True),
            _cand([[0]], votes=1, correct=False),
        ]),
        _entry("T3", 3, 2, [
            _cand([[5]], votes=1, correct=True),
            _cand([[7]], votes=9, correct=False),
            _cand([[1]], votes=8, correct=False),
        ]),
        _entry("T4", 1, 5, [
            _cand([[3]], votes=5, correct=False),
            _cand([[2]], votes=4, correct=False),
        ]),
    ]


def _write_pool(path: Path, entries: list[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"entries": entries}, fh)


def test_req_4012_spec_declared() -> None:
    # REQ-VERIFY-4012: OpenSpec declares the best-of-N local generator arm before code.
    spec = Path("openspec/capabilities/verification/spec.md").read_text(encoding="utf-8")
    assert "REQ-VERIFY-4012" in spec
    assert "SCENARIO-VERIFY-4012" in spec
    assert "local_demo_perfect_coverage_bestofn" in spec


def test_select_local_model_prefers_gemma12_then_fallback() -> None:
    # REQ-VERIFY-4012: prefer 12B, fall back only to the 26B local GGUF.
    hits = {
        "unsloth/gemma-4-12B-it-GGUF": "/cache/gemma12.gguf",
        "unsloth/gemma-4-26B-A4B-it-GGUF": "/cache/gemma26.gguf",
    }
    chosen = exp.select_local_model("auto", resolver=lambda hf_id: hits.get(hf_id))
    assert chosen["name"] == "gemma-4-12B"
    assert chosen["model_path"] == "/cache/gemma12.gguf"

    chosen = exp.select_local_model(
        "auto",
        resolver=lambda hf_id: "/cache/gemma26.gguf" if "26B" in hf_id else None,
    )
    assert chosen["name"] == "gemma-4-26B-A4B"


def test_select_local_model_unknown_key_returns_none() -> None:
    # REQ-VERIFY-4012: only the registered local Gemma GGUFs are eligible.
    assert exp.select_local_model("bogus", resolver=lambda _hf_id: "/cache/x.gguf") is None


def test_preconditions_block_missing_gguf_before_llama(tmp_path: Path) -> None:
    # REQ-VERIFY-4012: absent GGUF emits blocked_local_gguf_not_cached, no fallback.
    pool = tmp_path / "pool.json.gz"
    _write_pool(pool, [])
    precs, chosen = exp.check_preconditions(
        model_key="auto",
        pool_path=pool,
        resolver=lambda _hf_id: None,
        llama_available_override=False,
    )
    assert chosen is None
    assert exp.blocker_from_preconditions(precs) == "blocked_local_gguf_not_cached"


def test_preconditions_block_llama_after_gguf(tmp_path: Path) -> None:
    pool = tmp_path / "pool.json.gz"
    _write_pool(pool, [])
    precs, chosen = exp.check_preconditions(
        model_key="gemma12",
        pool_path=pool,
        resolver=lambda _hf_id: "/cache/model.gguf",
        llama_available_override=False,
    )
    assert chosen["model_path"] == "/cache/model.gguf"
    assert exp.blocker_from_preconditions(precs) == "blocked_llama_cpp_unavailable"


def test_preconditions_block_pool_or_verifier_load(tmp_path: Path) -> None:
    # REQ-VERIFY-4012: pool/verifier load failure is terminal, after GGUF and llama pass.
    precs, chosen = exp.check_preconditions(
        model_key="gemma12",
        pool_path=tmp_path / "missing.json.gz",
        resolver=lambda _hf_id: "/cache/model.gguf",
        llama_available_override=True,
    )
    assert chosen["name"] == "gemma-4-12B"
    assert exp.blocker_from_preconditions(precs) == "blocked_eval_pool_unreadable"


def test_preconditions_real_llama_import_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # SCENARIO-VERIFY-4012: the real import path records llama availability without overrides.
    pool = tmp_path / "pool.json.gz"
    _write_pool(pool, [])
    precs, _chosen = exp.check_preconditions(
        model_key="gemma12",
        pool_path=pool,
        resolver=lambda _hf_id: "/cache/model.gguf",
    )
    assert {p["resource"]: p["available"] for p in precs}["llama_cpp"] is True

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # noqa: ANN001
        if name == "llama_cpp":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    precs, _chosen = exp.check_preconditions(
        model_key="gemma12",
        pool_path=pool,
        resolver=lambda _hf_id: "/cache/model.gguf",
    )
    assert {p["resource"]: p["available"] for p in precs}["llama_cpp"] is False


def test_independent_sampler_varies_seed_and_temperature() -> None:
    # SCENARIO-VERIFY-4012: independent draws vary seed and temperature.
    fake = _FakeLlama()
    sampler = exp.IndependentLocalSampler(fake, base_seed=100, base_temperature=0.2)
    sampler("prompt", 0)
    sampler("prompt", 1)
    first, second = fake.calls
    assert first["kwargs"]["seed"] != second["kwargs"]["seed"]
    assert first["kwargs"]["temperature"] != second["kwargs"]["temperature"]
    assert first["messages"][1]["content"] == "prompt"


def test_demo_only_prompt_excludes_task_id_candidates_and_test_input() -> None:
    # REQ-VERIFY-4012: generation prompt uses demo pairs only.
    prompt = exp.demo_only_prompt([_demo(1, 2)], task_name="T_SECRET", test_input=[[99]])
    assert "T_SECRET" not in prompt
    assert "TEST INPUT" not in prompt
    assert "99" not in prompt
    assert "candidate" not in prompt.lower()


def test_induce_task_samples_keeps_every_demo_perfect_program() -> None:
    # SCENARIO-VERIFY-4012: k independent samples are graded and every demo-perfect one is kept.
    samples = exp.induce_task_samples(
        "T1",
        [_demo(1, 2), _demo(3, 4)],
        _SeqSampler(["no code", IDENTITY, PLUS_ONE]),
        k=3,
    )
    assert len(samples) == 3
    assert [s["status"] for s in samples] == ["no_code", "graded", "graded"]
    assert [s["demo_perfect"] for s in samples] == [False, False, True]


def test_induce_task_samples_rejects_unsafe_code() -> None:
    # REQ-VERIFY-4012: samples still use the restricted GAP-4 sandbox.
    unsafe = "```python\nimport os\ndef transform(grid):\n    return grid\n```"
    samples = exp.induce_task_samples("T", [_demo(1, 2)], _SeqSampler([unsafe]), k=1)
    assert samples[0]["status"] == "unsafe_or_uncompilable"
    assert samples[0]["demo_perfect"] is False


def test_build_entry_programs_snaps_near_match_and_marks_coverage() -> None:
    # SCENARIO-VERIFY-4012: demo-perfect predictions snap to an exact/near candidate before gating.
    entry = _entry("T1", 1, 5, [_cand([[6]], 1, True), _cand([[9]], 9, False)])
    samples = exp.induce_task_samples("T1", entry["demos"], _SeqSampler([PLUS_ONE]), k=1)
    progs = exp.build_entry_programs([entry], {"T1": samples}, tau=0.005)
    prog = progs[id(entry)]
    assert prog["demo_perfect"] is True
    assert prog["pred_grid"] == [[6]]
    assert prog["snap_hamming"] == 0.0


def test_reexecute_sample_handles_nonperfect_and_illegal_prediction() -> None:
    entry = _entry("T", 1, 5, [_cand([[6]], 1, True)])
    nonperfect = {"demo_perfect": False, "code": None}
    assert exp._reexecute_sample_for_entry(nonperfect, entry)["pred_grid"] is None
    illegal = {
        "demo_perfect": True,
        "code": "def transform(grid):\n    return []\n",
        "draw_index": 0,
    }
    assert exp._reexecute_sample_for_entry(illegal, entry)["pred_grid"] is None


def test_checkpoint_resume_preserves_completed_task(tmp_path: Path) -> None:
    # REQ-VERIFY-4012: per-task checkpointing avoids redoing completed local samples.
    entries = [_entry("T1", 1, 5, [_cand([[6]], 1, True)])]
    checkpoint = tmp_path / "ckpt.json"
    sampler = _SeqSampler([PLUS_ONE])
    first = exp.induce_pool_best_of_n(
        entries,
        sampler,
        k=2,
        checkpoint_path=checkpoint,
        model_name="gemma-4-12B",
    )
    assert len(sampler.calls) == 2
    assert checkpoint.exists()

    class _Boom:
        def __call__(self, prompt: str, draw_index: int) -> tuple[str, float]:
            raise AssertionError("checkpoint was not used")

    second = exp.induce_pool_best_of_n(
        entries,
        _Boom(),
        k=2,
        checkpoint_path=checkpoint,
        model_name="gemma-4-12B",
    )
    assert second == first


def test_checkpoint_mismatch_is_ignored(tmp_path: Path) -> None:
    entries = [_entry("T1", 1, 5, [_cand([[6]], 1, True)])]
    checkpoint = tmp_path / "ckpt.json"
    checkpoint.write_text(
        json.dumps({"k_samples_per_task": 99, "local_model_used": "old", "tasks": {"T1": []}}),
        encoding="utf-8",
    )
    sampler = _SeqSampler([PLUS_ONE])
    samples = exp.induce_pool_best_of_n(
        entries,
        sampler,
        k=1,
        checkpoint_path=checkpoint,
        model_name="gemma-4-12B",
    )
    assert len(sampler.calls) == 1
    assert samples["T1"][0]["demo_perfect"] is True


def test_score_best_of_n_gate_beats_vote() -> None:
    # SCENARIO-VERIFY-4012: best-of-N coverage drives the unchanged GAP-4 gated pass@2 scorer.
    entries = _synthetic_pool()
    samples = exp.induce_pool_best_of_n(
        entries,
        _SeqSampler([IDENTITY, PLUS_ONE]),
        k=2,
        checkpoint_path=None,
        model_name="gemma-4-12B",
    )
    progs = exp.build_entry_programs(entries, samples)
    scored = exp.score_best_of_n_pool(entries, progs)
    assert scored["g2"] > scored["vote2"]
    assert scored["n_perfect"] == 2
    assert "T1" in scored["headroom_recovered"]


def test_missing_verifier_gaps_are_task_ids(tmp_path: Path) -> None:
    entries = _synthetic_pool()
    samples = exp.induce_pool_best_of_n(
        entries,
        _SeqSampler([PLUS_ONE]),
        k=1,
        checkpoint_path=None,
        model_name="gemma-4-12B",
    )
    ref = tmp_path / "codex.json"
    ref.write_text(
        json.dumps({"per_task": [{"task": "T1", "demo_perfect": True}, {"task": "T3", "demo_perfect": True}]}),
        encoding="utf-8",
    )
    assert exp.missing_verifier_gap_tasks(entries, samples, ref) == ["T3"]


def test_missing_verifier_gaps_falls_back_when_reference_missing(tmp_path: Path) -> None:
    entries = _synthetic_pool()
    samples = exp.induce_pool_best_of_n(
        entries,
        _SeqSampler([PLUS_ONE]),
        k=1,
        checkpoint_path=None,
        model_name="gemma-4-12B",
    )
    assert exp.missing_verifier_gap_tasks(entries, samples, tmp_path / "missing.json") == ["T2", "T3"]


def test_verdict_success_and_complete_branches() -> None:
    assert exp._verdict(True, 0.91, 0.58, "gemma-4-12B").startswith(
        "success: gap4_local_bestofn_beats_vote_pass2"
    )
    assert exp._verdict(False, 0.5, 0.45, "gemma-4-12B").startswith(
        "complete: gap4_local_bestofn_cov"
    )


def test_validate_artifact_rejects_schema_poison() -> None:
    art = exp.blocked_artifact(
        "blocked_local_gguf_not_cached",
        None,
        [{"resource": "local_gguf_cached", "available": False}],
        duration_s=0.1,
    )
    art["k_samples_per_task"] = 8.0
    with pytest.raises(ValueError, match="bare int"):
        exp.validate_artifact(art)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("honest_verdict", "done", "terminal prefix"),
        ("local_beats_vote", "yes", "bare bool"),
        ("local_gated_pass2", "0.5", "bare float"),
        ("random_seed", 1.5, "bare int"),
        ("ci95_local_minus_vote", [0.0], "2-element list"),
        ("local_model_used", 12, "must be a string"),
        ("missing_verifier_gaps", "none", "must be a list"),
    ],
)
def test_validate_artifact_rejects_typed_fields(field: str, value: Any, message: str) -> None:
    art = exp.blocked_artifact(
        "blocked_local_gguf_not_cached",
        None,
        [{"resource": "local_gguf_cached", "available": False}],
        duration_s=0.1,
    )
    art[field] = value
    with pytest.raises(ValueError, match=message):
        exp.validate_artifact(art)


def test_validate_artifact_rejects_missing_field() -> None:
    with pytest.raises(ValueError, match="missing required field"):
        exp.validate_artifact({})


def test_run_blocked_writes_valid_artifact(tmp_path: Path) -> None:
    # REQ-VERIFY-4012: blocked precondition writes the required schema fields.
    out = tmp_path / "artifact.json"
    art = exp.run(
        output_path=out,
        checkpoint_path=tmp_path / "ckpt.json",
        resolver=lambda _hf_id: None,
        llama_available_override=True,
        write=True,
    )
    assert art["honest_verdict"] == "blocked_local_gguf_not_cached"
    for field in exp.REQUIRED_FIELDS:
        assert field in art
    assert out.exists()


def test_run_complete_with_fake_sampler(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4012: complete pipeline with fake local samples and unchanged verifier scoring.
    pool = tmp_path / "pool.json.gz"
    _write_pool(pool, _synthetic_pool())
    ref = tmp_path / "codex.json"
    ref.write_text(json.dumps({"generator": {"total_codex_seconds": 20.0}, "n_unique_tasks": 4, "per_task": []}))
    out = tmp_path / "artifact.json"
    art = exp.run(
        pool_path=pool,
        output_path=out,
        codex_ref_path=ref,
        checkpoint_path=tmp_path / "ckpt.json",
        k=2,
        sampler=_SeqSampler([IDENTITY, PLUS_ONE], seconds=0.25),
        resolver=lambda _hf_id: "/cache/gemma12.gguf",
        llama_available_override=True,
        write=True,
    )
    exp.validate_artifact(art)
    assert art["experiment"] == "experiment_4012_gap4_local_best_of_n"
    assert art["local_model_used"] == "gemma-4-12B"
    assert art["k_samples_per_task"] == 2
    assert art["local_demo_perfect_coverage_bestofn"] == 0.5
    assert art["coverage_gain_vs_3attempt"] == pytest.approx(0.2419)
    assert art["verifier_side_unchanged"] is True
    assert art["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert art["honest_verdict"].startswith(("success:", "complete:"))
    assert out.exists()


def test_run_complete_with_limit_no_write(tmp_path: Path) -> None:
    pool = tmp_path / "pool.json.gz"
    _write_pool(pool, _synthetic_pool())
    out = tmp_path / "artifact.json"
    art = exp.run(
        pool_path=pool,
        output_path=out,
        codex_ref_path=tmp_path / "no_codex.json",
        checkpoint_path=tmp_path / "ckpt.json",
        k=1,
        limit=2,
        sampler=_SeqSampler([PLUS_ONE], seconds=0.1),
        resolver=lambda _hf_id: "/cache/gemma12.gguf",
        llama_available_override=True,
        write=False,
    )
    assert art["n_entries"] == 2
    assert not out.exists()
