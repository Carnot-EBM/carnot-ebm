"""Tests for experiment_1077_fr11_alpha_t_sota_v4.

Covers the helpers added or repurposed by Exp 1077 (the SOTA-tier rerun of
Exp 1074):
  - _build_questions: deterministic, returns N items with int answers
  - _final_answer_correct: extracts the last numeric literal
  - _symcode_verdict: arithmetic claim eval (correct, incorrect, no-claims)
  - _ising_verdict: feature-energy thresholding
  - _length_verdict: short-response rejection
  - _temperature_verdict: top-50%-by-length partition
  - _resolve_sota_path: returns a real Qwen3.6-35B path or None
  - _run_experiment: produces a model_tier_violation / blocked artifact when
                     SOTA path resolution or GPU detection fails

These tests are CPU-only.  They cover the verifier helpers in full and the
``_run_experiment`` failure paths that do not require a real GPU.  The live
GPU path is exercised by the actual experiment run; the test suite stops
short of demanding a GPU because we want this file to remain green on CI.

Spec: REQ-PHI-001, REQ-PHI-002, REQ-PHI-003, REQ-VERIFY-083, REQ-INFER-SOTA-001.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_module_importable() -> None:
    """The experiment module imports cleanly without re-execing the process."""
    import scripts.experiment_1077_fr11_alpha_t_sota_v4 as mod  # noqa: F401

    assert mod.EXP_ID == 1077
    assert mod.N_QUESTIONS_TARGET == 100
    assert mod.SOTA_NAME == "Qwen3.6-35B-A3B"
    assert "Qwen3.6-35B-A3B-GGUF" in mod.SOTA_HF_ID
    assert mod.ALPHA_T_V1_COMPARISON == 0.78


def test_build_questions_deterministic_count() -> None:
    """_build_questions returns exactly n entries with the expected schema."""
    from scripts.experiment_1077_fr11_alpha_t_sota_v4 import _build_questions

    qs = _build_questions(100)
    assert len(qs) == 100
    for q in qs:
        assert {"question_id", "question", "answer"} <= set(q.keys())
        assert isinstance(q["answer"], int)


def test_build_questions_is_deterministic() -> None:
    """Repeated calls produce identical questions (no RNG leakage)."""
    from scripts.experiment_1077_fr11_alpha_t_sota_v4 import _build_questions

    a = _build_questions(10)
    b = _build_questions(10)
    assert a == b


def test_final_answer_correct_takes_last_number() -> None:
    """The last integer in the response is treated as the final answer."""
    from scripts.experiment_1077_fr11_alpha_t_sota_v4 import _final_answer_correct

    assert _final_answer_correct("Step 1: 5+3=8. Step 2: 8-1=7. Answer: 7", 7) is True
    assert _final_answer_correct("Answer: 42", 42) is True
    assert _final_answer_correct("Answer: 41", 42) is False


def test_final_answer_correct_no_numbers() -> None:
    """A response with no numbers is treated as incorrect, not a crash."""
    from scripts.experiment_1077_fr11_alpha_t_sota_v4 import _final_answer_correct

    assert _final_answer_correct("I do not know.", 7) is False


def test_symcode_verdict_paths() -> None:
    """_symcode_verdict covers all-correct, all-wrong, and no-claim paths."""
    from scripts.experiment_1077_fr11_alpha_t_sota_v4 import _symcode_verdict

    assert _symcode_verdict("So 2+2=4 and 3*5=15.")[0] == "correct"
    assert _symcode_verdict("So 2+2=5.")[0] == "incorrect"
    v_no, s_no = _symcode_verdict("There were no equations here.")
    assert v_no == "correct"
    assert s_no == 1.0


def test_ising_verdict_clean_response() -> None:
    """A clean response triggers no flags and the verdict is correct."""
    from scripts.experiment_1077_fr11_alpha_t_sota_v4 import _ising_verdict

    v, e = _ising_verdict("She has 4 muffins left.")
    assert v == "correct"
    assert e == 0.0


def test_ising_verdict_flags_runaway_and_hedging() -> None:
    """Hedging words plus runaway length flip the verdict to incorrect."""
    from scripts.experiment_1077_fr11_alpha_t_sota_v4 import _ising_verdict

    bad = "wait, that was a mistake. " + ("filler text " * 200)
    v, e = _ising_verdict(bad)
    assert v == "incorrect"
    assert e > 0.5


def test_length_verdict_rejects_short_responses() -> None:
    """A response shorter than 0.5x the question is rejected."""
    from scripts.experiment_1077_fr11_alpha_t_sota_v4 import _length_verdict

    long_q = "How many apples does Tom have if he started with 10 and gave away 3?"
    v_short, _ = _length_verdict("4.", long_q)
    assert v_short == "incorrect"
    v_long, _ = _length_verdict("Tom has 10 apples; after giving 3 away he has 7.", long_q)
    assert v_long == "correct"


def test_length_verdict_empty_question() -> None:
    """An empty question returns the safe default."""
    from scripts.experiment_1077_fr11_alpha_t_sota_v4 import _length_verdict

    v, s = _length_verdict("anything", "")
    assert v == "correct"
    assert s == 1.0


def test_temperature_verdict_partitions_by_length() -> None:
    """The baseline keeps responses at or above the median length."""
    from scripts.experiment_1077_fr11_alpha_t_sota_v4 import _temperature_verdict

    all_resp = ["a", "bb", "ccc", "dddd", "eeeee"]
    v_short, _ = _temperature_verdict("a", all_resp)
    v_long, _ = _temperature_verdict("eeeee", all_resp)
    assert v_short == "incorrect"
    assert v_long == "correct"


def test_temperature_verdict_empty_pool_safe_default() -> None:
    """An empty response pool returns the safe-default verdict."""
    from scripts.experiment_1077_fr11_alpha_t_sota_v4 import _temperature_verdict

    v, s = _temperature_verdict("anything", [])
    assert v == "correct"
    assert s == 0.0


def test_resolve_sota_path_returns_string_or_none() -> None:
    """_resolve_sota_path returns a string when the cache hits, else None."""
    from scripts.experiment_1077_fr11_alpha_t_sota_v4 import (
        SOTA_TOKEN,
        _resolve_sota_path,
    )

    p = _resolve_sota_path()
    assert p is None or (isinstance(p, str) and (SOTA_TOKEN in p or "3.6-35B" in p))


def test_resolve_sota_path_none_when_resolver_missing() -> None:
    """When the resolver is unimportable the helper returns None gracefully."""
    import builtins
    import scripts.experiment_1077_fr11_alpha_t_sota_v4 as mod

    real_import = builtins.__import__

    def _fake_import(name: str, *a, **kw):
        if name == "carnot.inference.sota_models":
            raise ImportError("simulated missing module")
        return real_import(name, *a, **kw)

    with patch.object(builtins, "__import__", side_effect=_fake_import):
        assert mod._resolve_sota_path() is None


def test_run_experiment_blocks_when_no_gpu(tmp_path: Path, monkeypatch) -> None:
    """When CUDA is unavailable the run writes a blocked_no_live_gpu artifact."""
    import scripts.experiment_1077_fr11_alpha_t_sota_v4 as mod
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    prev_deliverable = mod.DELIVERABLE
    mod.DELIVERABLE = str(tmp_path / "exp1077.json")
    try:
        artifact = mod._run_experiment()
    finally:
        mod.DELIVERABLE = prev_deliverable
    assert artifact["honest_verdict"] == "blocked_no_live_gpu"
    assert artifact["status"] == "blocked"
    assert artifact["model_tier"] == "sota_moe"
    assert artifact["fr11_loop_closed"] is False
    assert artifact["alpha_t"] == 0.0


def test_run_experiment_blocks_when_sota_missing(tmp_path: Path, monkeypatch) -> None:
    """When the SOTA GGUF is missing, write model_tier_violation and stop."""
    import scripts.experiment_1077_fr11_alpha_t_sota_v4 as mod
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(mod, "_resolve_sota_path", lambda: None)
    prev_deliverable = mod.DELIVERABLE
    mod.DELIVERABLE = str(tmp_path / "exp1077.json")
    try:
        artifact = mod._run_experiment()
    finally:
        mod.DELIVERABLE = prev_deliverable
    assert artifact["honest_verdict"] == "model_tier_violation"
    assert artifact["status"] == "blocked"
    assert artifact["model_path"] is None
    assert artifact["model_tier"] == "sota_moe"


def test_artifact_schema_has_required_fields(tmp_path: Path, monkeypatch) -> None:
    """Even the blocked-path artifact carries every required schema field."""
    import scripts.experiment_1077_fr11_alpha_t_sota_v4 as mod
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    prev = mod.DELIVERABLE
    mod.DELIVERABLE = str(tmp_path / "exp1077.json")
    try:
        artifact = mod._run_experiment()
    finally:
        mod.DELIVERABLE = prev
    required = {
        "experiment",
        "title",
        "run_date",
        "started_at",
        "finished_at",
        "duration_s",
        "status",
        "schema_version",
        "inference_mode",
        "n_questions_generated",
        "n_questions_target",
        "alpha_t",
        "alpha_t_v1_comparison",
        "phi_metric",
        "n_fr11_training_examples_appended",
        "fr11_loop_closed",
        "honest_verdict",
        "model_name",
        "model_tier",
    }
    missing = required - set(artifact)
    assert not missing, f"missing fields: {missing}"
