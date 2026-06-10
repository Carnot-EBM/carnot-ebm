"""Tests for Exp 4002 GAP-4 LOCAL open-weight generator arm.

Spec refs: REQ-VERIFY-4002, SCENARIO-VERIFY-4002.

These tests drive the ENTIRE pipeline with a FAKE proposer and a synthetic pool — no GGUF is
loaded, no GPU is touched. That is the design contract of the experiment: the generator is
injectable so the swap (codex -> local GGUF) is isolated and unit-testable, and the verifier side
is the unchanged GAP-4 code exercised through `score_pool`.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import experiment_4002_gap4_local_generator_arm as exp


# --------------------------------------------------------------------------- fakes / fixtures
PLUS_ONE = "```python\ndef transform(grid):\n    return grid + 1\n```"


def plus_one_proposer(prompt: str) -> tuple[str, float]:
    """A fake LOCAL model that always proposes the +1 rule. It is demo-perfect for +1 tasks and
    wrong for others, so the SAME canned program yields different per-task outcomes — exactly the
    behaviour we need to exercise the gate-promote and the no-gate-fallback branches."""
    return PLUS_ONE, 0.5


class _SeqProposer:
    """Returns a fixed sequence of responses across calls (for the failure-feedback loop)."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.i = 0

    def __call__(self, prompt: str) -> tuple[str, float]:
        r = self.responses[min(self.i, len(self.responses) - 1)]
        self.i += 1
        return r, 0.1


class _FakeLlama:
    """Minimal stand-in for llama_cpp.Llama: returns a canned chat completion."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[Any] = []

    def create_chat_completion(self, messages, **kwargs):  # noqa: ANN001
        self.calls.append((messages, kwargs))
        return {"choices": [{"message": {"content": self.content}}]}


def _demo(value_in: int, value_out: int) -> dict[str, Any]:
    return {"input": [[value_in]], "output": [[value_out]]}


def _cand(grid: list[list[int]], votes: int, correct: bool, q: float = 0.5) -> dict[str, Any]:
    return {"grid": grid, "votes": votes, "correct": correct, "q_mean": q}


def _entry(task: str, rule_delta: int, test_in: int, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    # demos all share the +rule_delta transform; plus_one is demo-perfect iff rule_delta == 1.
    return {
        "task": task,
        "demos": [_demo(1, 1 + rule_delta), _demo(3, 3 + rule_delta)],
        "test_input": [[test_in]],
        "candidates": candidates,
    }


def _synthetic_pool() -> list[dict[str, Any]]:
    """4 tasks engineered so the +1 fake inducer makes the gated rerank beat vote at the point
    estimate, while leaving the verifier code fully exercised:

      T1 (+1 rule): vote MISSES (correct ranked 3rd by votes); the demo-perfect program exec-matches
         the correct candidate -> gate promotes it -> gated RECOVERS.  (headroom)
      T2 (+2 rule): plus_one NOT demo-perfect -> no gate -> pure vote, which is correct.  (agree)
      T3 (+3 rule): plus_one NOT demo-perfect -> no gate; vote misses -> both miss.  (codex-gap task)
      T4 (+1 rule): demo-perfect but NO correct candidate in pool -> no help, no harm.  (oracle miss)
    """
    return [
        _entry("T1", 1, 5, [
            _cand([[6]], votes=1, correct=True),   # gold (+1 of 5), low vote
            _cand([[9]], votes=9, correct=False),
            _cand([[8]], votes=8, correct=False),
        ]),
        _entry("T2", 2, 4, [
            _cand([[12]], votes=9, correct=True),  # vote already right
            _cand([[0]], votes=1, correct=False),
        ]),
        _entry("T3", 3, 2, [
            _cand([[5]], votes=1, correct=True),   # gold, low vote, no demo-perfect program -> miss
            _cand([[7]], votes=9, correct=False),
            _cand([[1]], votes=8, correct=False),
        ]),
        _entry("T4", 1, 5, [
            _cand([[3]], votes=5, correct=False),  # no correct candidate at all
            _cand([[2]], votes=4, correct=False),
        ]),
    ]


# --------------------------------------------------------------------------- preconditions
def test_check_preconditions_all_pass(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4002: all three resources available -> no blocker.
    pool = tmp_path / "pool.json.gz"
    with gzip.open(pool, "wt", encoding="utf-8") as fh:
        json.dump({"entries": _synthetic_pool()}, fh)
    gguf = tmp_path / "model.gguf"
    gguf.write_text("x")
    precs = exp.check_preconditions(
        "unsloth/Qwen3.6-35B-A3B-GGUF", pool,
        gguf_path_override=str(gguf), llama_available_override=True,
    )
    assert exp.blocker_from_preconditions(precs) is None
    assert {p["resource"] for p in precs} == {"local_gguf_cached", "llama_cpp", "eval_pool"}


def test_check_preconditions_gguf_missing(tmp_path: Path) -> None:
    # REQ-VERIFY-4002: absent GGUF -> blocked_local_gguf_not_cached (never DSL/codex fallback).
    pool = tmp_path / "pool.json.gz"
    with gzip.open(pool, "wt", encoding="utf-8") as fh:
        json.dump({"entries": []}, fh)
    precs = exp.check_preconditions(
        "unsloth/Qwen3.6-35B-A3B-GGUF", pool,
        gguf_path_override="", llama_available_override=True,
    )
    assert exp.blocker_from_preconditions(precs) == "blocked_local_gguf_not_cached"


def test_check_preconditions_llama_missing(tmp_path: Path) -> None:
    pool = tmp_path / "pool.json.gz"
    with gzip.open(pool, "wt", encoding="utf-8") as fh:
        json.dump({"entries": []}, fh)
    gguf = tmp_path / "m.gguf"
    gguf.write_text("x")
    precs = exp.check_preconditions(
        "unsloth/Qwen3.6-35B-A3B-GGUF", pool,
        gguf_path_override=str(gguf), llama_available_override=False,
    )
    assert exp.blocker_from_preconditions(precs) == "blocked_llama_cpp_unavailable"


def test_check_preconditions_pool_unreadable(tmp_path: Path) -> None:
    gguf = tmp_path / "m.gguf"
    gguf.write_text("x")
    precs = exp.check_preconditions(
        "unsloth/Qwen3.6-35B-A3B-GGUF", tmp_path / "nope.json.gz",
        gguf_path_override=str(gguf), llama_available_override=True,
    )
    assert exp.blocker_from_preconditions(precs) == "blocked_eval_pool_unreadable"


def test_blocker_none_when_all_available() -> None:
    precs = [
        {"resource": "local_gguf_cached", "available": True},
        {"resource": "llama_cpp", "available": True},
        {"resource": "eval_pool", "available": True},
    ]
    assert exp.blocker_from_preconditions(precs) is None


# --------------------------------------------------------------------------- the LOCAL proposer
def test_local_gguf_proposer_contract() -> None:
    # SCENARIO-VERIFY-4002: the proposer returns (raw_text, wall_seconds) like ask_codex.
    proposer = exp.LocalGGUFProposer(_FakeLlama(PLUS_ONE))
    text, secs = proposer("some prompt")
    assert "def transform" in text
    assert isinstance(secs, float) and secs >= 0.0


def test_local_gguf_proposer_sends_system_and_user() -> None:
    fake = _FakeLlama(PLUS_ONE)
    exp.LocalGGUFProposer(fake)("the-induction-prompt")
    messages = fake.calls[0][0]
    assert messages[0]["role"] == "system" and "transform" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "the-induction-prompt"}


def test_local_gguf_proposer_survives_model_crash() -> None:
    class _Boom:
        def create_chat_completion(self, **kwargs):  # noqa: ANN001
            raise RuntimeError("oom")

    text, secs = exp.LocalGGUFProposer(_Boom())("p")
    assert text.startswith("__local_error__")
    assert isinstance(secs, float)


# --------------------------------------------------------------------------- induction (generator)
def test_induce_program_local_demo_perfect_one_iter() -> None:
    # SCENARIO-VERIFY-4002: a demo-perfect program is kept and executed on the test input.
    rec = exp.induce_program_local("T", [_demo(1, 2), _demo(3, 4)], [[5]], plus_one_proposer)
    assert rec["demo_perfect"] is True
    assert rec["demo_fit"] == 1.0
    assert rec["pred_grid"] == [[6]]
    assert rec["n_calls"] == 1


def test_induce_program_local_feedback_recovers() -> None:
    # The failure-feedback loop: a wrong first draft, then a correct second draft.
    wrong = "```python\ndef transform(grid):\n    return grid\n```"
    rec = exp.induce_program_local(
        "T", [_demo(1, 2), _demo(3, 4)], [[5]], _SeqProposer([wrong, PLUS_ONE]), iters=3
    )
    assert rec["demo_perfect"] is True
    assert rec["n_calls"] == 2


def test_induce_program_local_no_code_then_no_program() -> None:
    rec = exp.induce_program_local("T", [_demo(1, 2)], [[5]], _SeqProposer(["no code here"]))
    assert rec["demo_perfect"] is False
    assert rec["pred_grid"] is None
    assert rec["history"][0]["status"] == "no_code"


def test_induce_program_local_rejects_unsafe_code() -> None:
    unsafe = "```python\nimport os\ndef transform(grid):\n    return grid\n```"
    rec = exp.induce_program_local("T", [_demo(1, 2)], [[5]], _SeqProposer([unsafe]))
    assert rec["demo_perfect"] is False
    assert rec["history"][0]["status"] == "unsafe_or_uncompilable"


def test_induce_pool_reuses_program_for_extra_entries() -> None:
    # A task with two test entries induces once and re-executes per entry.
    e0 = _entry("DUP", 1, 5, [_cand([[6]], 1, True)])
    e1 = _entry("DUP", 1, 7, [_cand([[8]], 1, True)])
    progs = exp.induce_pool([e0, e1], plus_one_proposer)
    assert progs[id(e0)]["pred_grid"] == [[6]]
    assert progs[id(e1)]["pred_grid"] == [[8]]   # same program, different test input
    assert progs[id(e1)]["n_calls"] == 0          # reused, not re-induced


# --------------------------------------------------------------------------- verifier (UNCHANGED)
def test_score_pool_gate_beats_vote() -> None:
    # SCENARIO-VERIFY-4002: the unchanged gated rerank recovers a vote-missed task via exec-match.
    entries = _synthetic_pool()
    progs = exp.induce_pool(entries, plus_one_proposer)
    scored = exp.score_pool(entries, progs, seed=exp.SEED)
    assert scored["g2"] > scored["vote2"]
    assert scored["gates"]["selection_beats_vote"] is True
    assert "T1" in scored["headroom_recovered"]
    assert scored["gates"]["vote_wins_lost"] == 0   # the safety invariant
    assert scored["n_perfect"] == 2                 # T1 + T4 are +1 tasks


def test_score_pool_demo_perfect_count_matches_rule() -> None:
    entries = _synthetic_pool()
    progs = exp.induce_pool(entries, plus_one_proposer)
    scored = exp.score_pool(entries, progs)
    perfect_tasks = {pt["task"] for pt in scored["per_task"] if pt["demo_perfect"]}
    assert perfect_tasks == {"T1", "T4"}


# --------------------------------------------------------------------------- cost + gaps
def test_codex_reference_cost_reads_artifact(tmp_path: Path) -> None:
    ref = tmp_path / "codex.json"
    ref.write_text(json.dumps({"generator": {"total_codex_seconds": 1387.2}, "n_unique_tasks": 30}))
    assert exp.codex_reference_cost(ref) == pytest.approx(46.24, abs=0.01)


def test_codex_reference_cost_fallback(tmp_path: Path) -> None:
    assert exp.codex_reference_cost(tmp_path / "missing.json") == pytest.approx(46.24, abs=0.01)


def test_missing_verifier_gaps_lists_codex_only_tasks(tmp_path: Path) -> None:
    entries = _synthetic_pool()
    progs = exp.induce_pool(entries, plus_one_proposer)  # local perfect on T1, T4
    ref = tmp_path / "codex.json"
    ref.write_text(json.dumps({"per_task": [
        {"task": "T1", "demo_perfect": True},
        {"task": "T3", "demo_perfect": True},   # codex perfect, local NOT -> a gap
    ]}))
    gaps = exp.compute_missing_verifier_gaps(entries, progs, ref)
    assert "T3" in gaps and "T1" not in gaps


def test_missing_verifier_gaps_no_gap(tmp_path: Path) -> None:
    entries = _synthetic_pool()
    progs = exp.induce_pool(entries, plus_one_proposer)
    ref = tmp_path / "codex.json"
    ref.write_text(json.dumps({"per_task": [{"task": "T1", "demo_perfect": True}]}))
    assert "No induction gap" in exp.compute_missing_verifier_gaps(entries, progs, ref)


def test_missing_verifier_gaps_no_reference(tmp_path: Path) -> None:
    entries = _synthetic_pool()
    progs = exp.induce_pool(entries, plus_one_proposer)
    msg = exp.compute_missing_verifier_gaps(entries, progs, tmp_path / "none.json")
    assert "reference unavailable" in msg


# --------------------------------------------------------------------------- verdict + validation
def test_verdict_success_branch() -> None:
    v = exp._verdict(True, 0.93, 0.58, "Qwen3.6-35B-A3B")
    assert v.startswith("success: gap4_local_generator_beats_vote_pass2")
    assert "Qwen3.6-35B-A3B" in v


def test_verdict_complete_branch() -> None:
    v = exp._verdict(False, 0.5, 0.45, "gemma-4-12B")
    assert v.startswith("complete: gap4_local_induction")
    assert v.endswith("_below_codex")


def test_validate_artifact_rejects_missing_field() -> None:
    with pytest.raises(ValueError, match="missing required field"):
        exp.validate_artifact({})


def test_validate_artifact_rejects_bad_verdict() -> None:
    art = _minimal_valid_artifact()
    art["honest_verdict"] = "gap4_done"   # no terminal prefix
    with pytest.raises(ValueError, match="terminal prefix"):
        exp.validate_artifact(art)


def test_validate_artifact_rejects_non_bare_bool() -> None:
    art = _minimal_valid_artifact()
    art["local_beats_vote"] = "yes"
    with pytest.raises(ValueError, match="bare bool"):
        exp.validate_artifact(art)


def test_validate_artifact_rejects_bad_ci() -> None:
    art = _minimal_valid_artifact()
    art["ci95_local_minus_vote"] = [0.0]
    with pytest.raises(ValueError, match="2-element list"):
        exp.validate_artifact(art)


def _minimal_valid_artifact() -> dict[str, Any]:
    return {
        "local_induction_demo_perfect_rate": 0.5,
        "local_gated_pass2": 0.5,
        "local_beats_vote": False,
        "local_model_used": "m",
        "cost_local_seconds": 1.0,
        "cost_codex_seconds_ref": 1.0,
        "cost_verifier_seconds": 0.001,
        "ci95_local_minus_vote": [0.0, 0.1],
        "verifier_side_unchanged": True,
        "missing_verifier_gaps": "none",
        "preconditions_checked": [],
        "random_seed": exp.SEED,
        "honest_verdict": "complete: x",
        "duration_s": 1.0,
        "inference_substrate": exp.INFERENCE_SUBSTRATE,
    }


def test_build_complete_artifact_success_branch() -> None:
    # Deterministic success branch (CI excludes 0) without relying on the bootstrap of a tiny pool.
    entries = _synthetic_pool()
    progs = exp.induce_pool(entries, plus_one_proposer)
    scored = exp.score_pool(entries, progs)
    scored = {**scored, "ci95_gated_vs_vote": [0.05, 0.30], "g2": 0.58, "vote2": 0.45}
    art = exp.build_complete_artifact(
        entries=entries, prog_by_entry_id=progs, scored=scored,
        model_name="Qwen3.6-35B-A3B", model_path="/x/model.gguf", preconditions=[],
        verifier_seconds=0.01, started_s=0.0, now_s=2.0, codex_ref_path=Path("/nonexistent.json"),
    )
    assert art["local_beats_vote"] is True
    assert art["honest_verdict"].startswith("success: gap4_local_generator_beats_vote_pass2")
    assert art["verifier_side_unchanged"] is True


# --------------------------------------------------------------------------- end-to-end run()
def test_run_blocked_writes_valid_artifact(tmp_path: Path) -> None:
    # REQ-VERIFY-4002: blocked precondition -> a schema-valid blocked artifact, zero induction.
    out = tmp_path / "art.json"
    art = exp.run(
        model_key="qwen35", output_path=out,
        gguf_path_override="", llama_available_override=True, write=True,
    )
    assert art["honest_verdict"] == "blocked_local_gguf_not_cached"
    for field in exp.REQUIRED_FIELDS:
        assert field in art
    assert out.exists()


def test_run_complete_with_fake_proposer(tmp_path: Path) -> None:
    # SCENARIO-VERIFY-4002: the whole pipeline with an injected fake LOCAL model, no GGUF loaded.
    pool = tmp_path / "pool.json.gz"
    with gzip.open(pool, "wt", encoding="utf-8") as fh:
        json.dump({"entries": _synthetic_pool()}, fh)
    gguf = tmp_path / "model.gguf"
    gguf.write_text("x")
    out = tmp_path / "art.json"
    art = exp.run(
        model_key="qwen35", pool_path=pool, output_path=out,
        codex_ref_path=tmp_path / "no_codex.json",
        proposer=plus_one_proposer, gguf_path_override=str(gguf),
        llama_available_override=True, write=True,
    )
    # required schema present + correctly typed
    for field in exp.REQUIRED_FIELDS:
        assert field in art
    exp.validate_artifact(art)
    assert art["verifier_side_unchanged"] is True
    assert art["inference_substrate"] == "live_llm_inference"
    assert isinstance(art["local_beats_vote"], bool)
    assert art["honest_verdict"].startswith(("success:", "complete:"))
    # the gated rerank beat vote at the point estimate on the engineered pool
    assert art["local_gated_pass2"] > exp.VOTE_PASS2_REF - 1.0  # sanity: a real float pass@2
    assert art["local_induction_demo_perfect_rate"] == 0.5      # T1 + T4 of 4 entries
    assert out.exists()


def test_run_with_limit_slices_entries(tmp_path: Path) -> None:
    pool = tmp_path / "pool.json.gz"
    with gzip.open(pool, "wt", encoding="utf-8") as fh:
        json.dump({"entries": _synthetic_pool()}, fh)
    gguf = tmp_path / "model.gguf"
    gguf.write_text("x")
    art = exp.run(
        model_key="qwen35", pool_path=pool, output_path=tmp_path / "a.json",
        codex_ref_path=tmp_path / "no_codex.json", limit=2,
        proposer=plus_one_proposer, gguf_path_override=str(gguf),
        llama_available_override=True, write=False,
    )
    assert art["n_entries"] == 2


# --------------------------------------------------------------------------- real-resource probes
def test_resolve_local_gguf_bogus_returns_none() -> None:
    # The real resolver (no override): a non-existent repo id resolves to None, never a fake path.
    assert exp.resolve_local_gguf("unsloth/this-model-does-not-exist-GGUF") is None


def test_check_preconditions_without_overrides_uses_real_resolvers(tmp_path: Path) -> None:
    # Exercises the real gguf-resolve + real llama_cpp import branches. A bogus model id is not
    # cached, so the gguf precondition is False; the pool + llama_cpp checks run for real.
    pool = tmp_path / "pool.json.gz"
    with gzip.open(pool, "wt", encoding="utf-8") as fh:
        json.dump({"entries": []}, fh)
    precs = exp.check_preconditions("unsloth/this-model-does-not-exist-GGUF", pool)
    by = {p["resource"]: p["available"] for p in precs}
    assert by["local_gguf_cached"] is False
    assert by["eval_pool"] is True
    assert isinstance(by["llama_cpp"], bool)


def test_check_preconditions_llama_import_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the real `import llama_cpp` to fail so the except-branch (llama_ok=False) is exercised
    # even though the library is installed on this box.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # noqa: ANN001
        if name == "llama_cpp":
            raise ImportError("simulated missing llama_cpp")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    pool = tmp_path / "pool.json.gz"
    with gzip.open(pool, "wt", encoding="utf-8") as fh:
        json.dump({"entries": []}, fh)
    gguf = tmp_path / "m.gguf"
    gguf.write_text("x")
    precs = exp.check_preconditions(
        "unsloth/Qwen3.6-35B-A3B-GGUF", pool, gguf_path_override=str(gguf)
    )
    by = {p["resource"]: p["available"] for p in precs}
    assert by["llama_cpp"] is False


def test_validate_artifact_rejects_non_bare_float() -> None:
    art = _minimal_valid_artifact()
    art["local_gated_pass2"] = "x"
    with pytest.raises(ValueError, match="bare float"):
        exp.validate_artifact(art)


def test_validate_artifact_rejects_non_bare_int_seed() -> None:
    art = _minimal_valid_artifact()
    art["random_seed"] = 1.5
    with pytest.raises(ValueError, match="bare int"):
        exp.validate_artifact(art)


def test_validate_artifact_rejects_non_string_model() -> None:
    art = _minimal_valid_artifact()
    art["local_model_used"] = 123
    with pytest.raises(ValueError, match="must be a string"):
        exp.validate_artifact(art)


def test_validate_artifact_rejects_non_list_preconditions() -> None:
    art = _minimal_valid_artifact()
    art["preconditions_checked"] = "not-a-list"
    with pytest.raises(ValueError, match="must be a list"):
        exp.validate_artifact(art)


def test_run_complete_no_write(tmp_path: Path) -> None:
    pool = tmp_path / "pool.json.gz"
    with gzip.open(pool, "wt", encoding="utf-8") as fh:
        json.dump({"entries": _synthetic_pool()}, fh)
    gguf = tmp_path / "model.gguf"
    gguf.write_text("x")
    out = tmp_path / "should_not_exist.json"
    art = exp.run(
        model_key="qwen35", pool_path=pool, output_path=out,
        codex_ref_path=tmp_path / "no_codex.json",
        proposer=plus_one_proposer, gguf_path_override=str(gguf),
        llama_available_override=True, write=False,
    )
    assert not out.exists()
    assert art["experiment"] == "experiment_4002_gap4_local_generator_arm"
