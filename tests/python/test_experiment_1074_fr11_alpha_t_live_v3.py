"""Tests for experiment_1074_fr11_alpha_t_live_v3.

Covers the helpers added by Exp 1074:
  - _build_questions: deterministic, returns N items with int answers
  - _final_answer_correct: extracts the last numeric literal as the answer
  - _symcode_verdict: arithmetic claim eval (correct, incorrect, no-claims)
  - _ising_verdict: feature-energy thresholding
  - _length_verdict: short-response rejection
  - _temperature_verdict: top-50%-by-length partition
  - _run_experiment: produces a blocked artifact when both live paths fail
  - Deliverable JSON schema fields are present after the experiment runs

These tests do not require a GPU — when no live path is available the
experiment writes a blocked_no_live_gpu artifact, which is a first-class
test path that exercises the full ExperimentTemplate.build_result wiring.

Spec: REQ-PHI-001, REQ-PHI-002, REQ-PHI-003, REQ-VERIFY-083.
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
    """The experiment module imports without side-effect failures."""
    import scripts.experiment_1074_fr11_alpha_t_live_v3 as mod  # noqa: F401

    assert mod.EXP_ID == 1074
    assert mod.N_QUESTIONS_TARGET == 50


def test_build_questions_deterministic_count() -> None:
    """_build_questions returns exactly n entries with the expected schema."""
    from scripts.experiment_1074_fr11_alpha_t_live_v3 import _build_questions

    qs = _build_questions(50)
    assert len(qs) == 50
    for q in qs:
        assert {"question_id", "question", "answer"} <= set(q.keys())
        assert isinstance(q["answer"], int)


def test_build_questions_is_deterministic() -> None:
    """Repeated calls produce identical questions (no RNG leakage)."""
    from scripts.experiment_1074_fr11_alpha_t_live_v3 import _build_questions

    a = _build_questions(10)
    b = _build_questions(10)
    assert a == b


def test_final_answer_correct_takes_last_number() -> None:
    """The last integer in the response is treated as the final answer."""
    from scripts.experiment_1074_fr11_alpha_t_live_v3 import _final_answer_correct

    assert _final_answer_correct("Step 1: 5+3=8. Step 2: 8-1=7. Answer: 7", 7) is True
    assert _final_answer_correct("Answer: 42", 42) is True
    assert _final_answer_correct("Answer: 41", 42) is False


def test_final_answer_correct_no_numbers() -> None:
    """A response with no numbers is treated as incorrect, not as a crash."""
    from scripts.experiment_1074_fr11_alpha_t_live_v3 import _final_answer_correct

    assert _final_answer_correct("I do not know.", 7) is False


def test_symcode_verdict_all_correct() -> None:
    """_symcode_verdict returns 'correct' when every arithmetic claim checks out."""
    from scripts.experiment_1074_fr11_alpha_t_live_v3 import _symcode_verdict

    verdict, score = _symcode_verdict("First, 2+3=5. Then 5*2=10.")
    assert verdict == "correct"
    assert score == pytest.approx(1.0)


def test_symcode_verdict_partial_wrong() -> None:
    """Any incorrect arithmetic claim flips the verdict to 'incorrect'."""
    from scripts.experiment_1074_fr11_alpha_t_live_v3 import _symcode_verdict

    verdict, score = _symcode_verdict("2+3=5 and 4*4=15")
    assert verdict == "incorrect"
    assert 0.0 < score < 1.0


def test_symcode_verdict_no_claims() -> None:
    """No arithmetic claims = no evidence of fault, default to 'correct' with score 1.0."""
    from scripts.experiment_1074_fr11_alpha_t_live_v3 import _symcode_verdict

    verdict, score = _symcode_verdict("The answer is forty-two.")
    assert verdict == "correct"
    assert score == pytest.approx(1.0)


def test_symcode_verdict_division_by_zero() -> None:
    """Division by zero in a claim is rejected, not raised."""
    from scripts.experiment_1074_fr11_alpha_t_live_v3 import _symcode_verdict

    verdict, _score = _symcode_verdict("9/0=0")
    assert verdict == "incorrect"


def test_ising_verdict_clean_response_correct() -> None:
    """A short response with no equations and no error words is 'correct'."""
    from scripts.experiment_1074_fr11_alpha_t_live_v3 import _ising_verdict

    verdict, energy = _ising_verdict("ok")
    assert verdict == "correct"
    assert energy == pytest.approx(0.0)


def test_ising_verdict_long_with_error_words_above_threshold() -> None:
    """Long error-laden CoT with many equations crosses the energy threshold."""
    from scripts.experiment_1074_fr11_alpha_t_live_v3 import _ising_verdict

    text = "wait, 1+1=2. mistake — 2+2=4. " * 80
    verdict, energy = _ising_verdict(text)
    assert verdict == "incorrect"
    assert energy > 0.5


def test_length_verdict_too_short() -> None:
    """A very short response relative to the question is rejected."""
    from scripts.experiment_1074_fr11_alpha_t_live_v3 import _length_verdict

    verdict, ratio = _length_verdict("42", "A long question " * 10)
    assert verdict == "incorrect"
    assert ratio < 0.5


def test_length_verdict_adequate() -> None:
    """An adequately-long response passes the length check."""
    from scripts.experiment_1074_fr11_alpha_t_live_v3 import _length_verdict

    verdict, ratio = _length_verdict("solution: " + "x" * 200, "Q?")
    assert verdict == "correct"
    assert ratio >= 0.5


def test_length_verdict_empty_question() -> None:
    """When the question is empty the length check is a no-op (always 'correct')."""
    from scripts.experiment_1074_fr11_alpha_t_live_v3 import _length_verdict

    verdict, ratio = _length_verdict("anything", "")
    assert verdict == "correct"
    assert ratio == pytest.approx(1.0)


def test_temperature_verdict_top_half_kept() -> None:
    """Top-half-by-length responses are 'correct'; bottom-half are 'incorrect'."""
    from scripts.experiment_1074_fr11_alpha_t_live_v3 import _temperature_verdict

    all_resp = ["a", "ab", "abc", "abcd", "abcde"]
    v_short, _ = _temperature_verdict("a", all_resp)
    v_long, _ = _temperature_verdict("abcde", all_resp)
    assert v_short == "incorrect"
    assert v_long == "correct"


def test_temperature_verdict_empty_pool() -> None:
    """Empty response pool yields a 'correct' fallback rather than IndexError."""
    from scripts.experiment_1074_fr11_alpha_t_live_v3 import _temperature_verdict

    verdict, score = _temperature_verdict("anything", [])
    assert verdict == "correct"
    assert score == 0.0


def test_run_experiment_writes_blocked_artifact_when_no_live_path(tmp_path, monkeypatch) -> None:
    """When both live paths return None, _run_experiment emits a blocked artifact.

    This exercises the build_result wiring end-to-end without a GPU: both the
    llama-cpp and transformers helpers are stubbed to return None, and we
    verify the artifact has the contracted schema fields.
    """
    import scripts.experiment_1074_fr11_alpha_t_live_v3 as mod

    monkeypatch.setattr(mod, "_try_llama_cpp", lambda _q: None)
    monkeypatch.setattr(mod, "_try_transformers_fallback", lambda _q: None)
    # Redirect FR-11 output so the test does not pollute the real data file.
    monkeypatch.setattr(mod, "FR11_OUTPUT", tmp_path / "fr11_test.jsonl")

    artifact = mod._run_experiment()
    required = {
        "inference_mode",
        "n_questions_generated",
        "n_questions_target",
        "alpha_t",
        "phi_metric",
        "n_fr11_training_examples_written",
        "fr11_loop_closed",
        "honest_verdict",
    }
    assert required <= set(artifact.keys())
    assert artifact["honest_verdict"] == "blocked_no_live_gpu"
    assert artifact["inference_mode"] == "blocked_no_live_gpu"
    assert artifact["fr11_loop_closed"] is False
    assert artifact["n_questions_generated"] == 0


def test_run_experiment_writes_live_artifact_with_stubbed_responses(tmp_path, monkeypatch) -> None:
    """Stub the live path to return canned responses and verify alpha_t is computed.

    We feed a mix of correct + degenerate responses so that the AND-composed
    verdict and the temperature-only verdict diverge on at least one example,
    which is the only condition that produces alpha_t > 0.
    """
    import scripts.experiment_1074_fr11_alpha_t_live_v3 as mod

    def _stub_llama(questions):
        # Mix: half are well-formed CoT with correct arithmetic, half are
        # degenerate "42." responses.  Length-based temperature baseline will
        # reject the short ones; AND-verifier will also reject the short ones
        # (length verdict + symcode mismatch), so we need a third pattern that
        # the two disagree on.  Insert a "long-but-wrong-arithmetic" response
        # that temperature accepts (long enough) but symcode rejects.
        out = []
        for i, q in enumerate(questions):
            if i % 3 == 0:
                out.append(f"Step 1: 2+3=5. Step 2: 5*2=10. Answer: {q['answer']}")
            elif i % 3 == 1:
                out.append("42")
            else:
                out.append(
                    "Long reasoning: 1+1=3. Then we proceed carefully and "
                    "carefully and carefully. Final answer: " + str(q["answer"])
                )
        return out, "stub-model", "stub://path"

    monkeypatch.setattr(mod, "_try_llama_cpp", _stub_llama)
    monkeypatch.setattr(mod, "_try_transformers_fallback", lambda _q: None)
    monkeypatch.setattr(mod, "FR11_OUTPUT", tmp_path / "fr11_test.jsonl")
    # Run on a small pool so the test is fast.
    monkeypatch.setattr(mod, "N_QUESTIONS_TARGET", 9)

    artifact = mod._run_experiment()
    assert artifact["inference_mode"] == "live_gpu"
    assert artifact["n_questions_generated"] == 9
    assert 0.0 <= artifact["alpha_t"] <= 1.0
    assert artifact["honest_verdict"] in (
        "fr11_loop_closed_alpha_t_positive",
        "fr11_loop_closed_alpha_t_zero",
    )
    assert artifact["k_verifiers"] == 5
    # FR-11 rows should have been appended.
    fr11_path = tmp_path / "fr11_test.jsonl"
    assert fr11_path.exists()
    rows = [json.loads(line) for line in fr11_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 9
    for row in rows:
        assert "alpha_t_contributes" in row
        assert row["filter_source"] == "carnot_and_compose_k5"


def test_main_writes_deliverable(tmp_path, monkeypatch) -> None:
    """The script's main() writes the deliverable JSON with required fields."""
    import scripts.experiment_1074_fr11_alpha_t_live_v3 as mod

    monkeypatch.setattr(mod, "_try_llama_cpp", lambda _q: None)
    monkeypatch.setattr(mod, "_try_transformers_fallback", lambda _q: None)
    monkeypatch.setattr(mod, "FR11_OUTPUT", tmp_path / "fr11_test.jsonl")
    deliverable_path = tmp_path / "exp1074.json"
    monkeypatch.setattr(mod, "DELIVERABLE", str(deliverable_path))

    rc = mod.main()
    assert rc == 0
    assert deliverable_path.exists()
    artifact = json.loads(deliverable_path.read_text())
    assert artifact["honest_verdict"] == "blocked_no_live_gpu"
    assert artifact["experiment"] == 1074
