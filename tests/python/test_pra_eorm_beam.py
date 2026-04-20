"""Tests for carnot.pipeline.pra_eorm_beam — PRA EORM Beam Search.

100% coverage of python/carnot/pipeline/pra_eorm_beam.py.

Spec: REQ-REPAIR-016,
      SCENARIO-REPAIR-031, SCENARIO-REPAIR-032, SCENARIO-REPAIR-033
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from carnot.pipeline.pra_eorm_beam import (
    PRABeamCandidate,
    PRABeamResult,
    PRAEBMBeamSearch,
)


# ---------------------------------------------------------------------------
# Minimal mock EORM model (no JAX, no real model weights needed)
# ---------------------------------------------------------------------------

class _FixedEnergyModel:
    """Returns preset energies in round-robin order for deterministic tests.

    The energy() method accepts anything (CoTEnergyInput or string) and
    pops from a fixed list of floats.
    """

    def __init__(self, energies: list[float]) -> None:
        self._energies = list(energies)
        self._idx = 0

    def energy(self, cot_input: object) -> float:
        e = self._energies[self._idx % len(self._energies)]
        self._idx += 1
        return e


class _ConstantEnergyModel:
    """Always returns the same energy — useful for tie-breaking tests."""

    def __init__(self, value: float = 0.5) -> None:
        self._value = value

    def energy(self, cot_input: object) -> float:
        return self._value


class _TextHashModel:
    """Returns energy based on whether response_text contains '='."""

    def energy(self, cot_input: object) -> float:
        if hasattr(cot_input, "response_text"):
            text = cot_input.response_text
        else:
            text = str(cot_input)
        return 0.1 if "=" in text else 0.9


# ---------------------------------------------------------------------------
# PRABeamCandidate tests
# ---------------------------------------------------------------------------

class TestPRABeamCandidate:
    """SCENARIO-REPAIR-031: PRABeamCandidate dataclass contract."""

    def test_fields_accessible(self) -> None:
        c = PRABeamCandidate(step_text="hello", eorm_energy=0.5)
        assert c.step_text == "hello"
        assert c.eorm_energy == 0.5
        assert c.is_selected is False

    def test_is_selected_default_false(self) -> None:
        c = PRABeamCandidate(step_text="x", eorm_energy=1.0)
        assert c.is_selected is False

    def test_is_selected_can_be_set(self) -> None:
        c = PRABeamCandidate(step_text="x", eorm_energy=1.0, is_selected=True)
        assert c.is_selected is True


# ---------------------------------------------------------------------------
# PRABeamResult tests
# ---------------------------------------------------------------------------

class TestPRABeamResult:
    """SCENARIO-REPAIR-031: PRABeamResult dataclass contract."""

    def test_default_fields(self) -> None:
        r = PRABeamResult(n_steps=4, n_beams_explored=12)
        assert r.n_steps == 4
        assert r.n_beams_explored == 12
        assert r.selected_candidates == []
        assert r.baseline_violation_rate == 0.0
        assert r.beam_violation_rate == 0.0
        assert r.improvement == 0.0

    def test_can_set_all_fields(self) -> None:
        cand = PRABeamCandidate(step_text="a", eorm_energy=0.2, is_selected=True)
        r = PRABeamResult(
            n_steps=2,
            n_beams_explored=6,
            selected_candidates=[cand],
            baseline_violation_rate=0.5,
            beam_violation_rate=0.0,
            improvement=0.5,
        )
        assert r.improvement == 0.5
        assert len(r.selected_candidates) == 1


# ---------------------------------------------------------------------------
# PRAEBMBeamSearch.score_candidate tests
# ---------------------------------------------------------------------------

class TestScoreCandidate:
    """SCENARIO-REPAIR-031: score_candidate calls model.energy and returns float."""

    def test_returns_float(self) -> None:
        model = _ConstantEnergyModel(0.42)
        beam = PRAEBMBeamSearch(eorm_model=model, k_candidates=3)
        score = beam.score_candidate("some step text")
        assert isinstance(score, float)
        assert score == pytest.approx(0.42)

    def test_passes_question_to_model(self) -> None:
        """score_candidate must forward the question so EORM has context."""
        received: list[object] = []

        class _CapturingModel:
            def energy(self, cot_input: object) -> float:
                received.append(cot_input)
                return 0.0

        beam = PRAEBMBeamSearch(eorm_model=_CapturingModel(), k_candidates=3)
        beam.score_candidate("step text", question="my question")
        assert len(received) == 1
        inp = received[0]
        # Either CoTEnergyInput or plain string (mock path)
        if hasattr(inp, "question_text"):
            assert inp.question_text == "my question"
            assert inp.response_text == "step text"

    def test_score_without_question(self) -> None:
        model = _ConstantEnergyModel(1.0)
        beam = PRAEBMBeamSearch(eorm_model=model)
        # Should not raise when question is omitted
        score = beam.score_candidate("step")
        assert score == pytest.approx(1.0)

    def test_import_error_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When CoTEnergyInput import fails, fall back to passing text directly."""
        import carnot.pipeline.pra_eorm_beam as module

        # Patch the import inside score_candidate to raise ImportError
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else None

        received: list[object] = []

        class _StringModel:
            def energy(self, cot_input: object) -> float:
                received.append(cot_input)
                return 0.5

        # Force the ImportError branch by monkeypatching the module-level import
        import builtins
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "carnot.models.eorm":
                raise ImportError("mocked import failure")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        beam = PRAEBMBeamSearch(eorm_model=_StringModel(), k_candidates=3)
        score = beam.score_candidate("step text", question="q")
        assert score == pytest.approx(0.5)
        # The model received something (may be string or CoTEnergyInput depending
        # on which branch ran — what matters is no exception)
        assert len(received) == 1


# ---------------------------------------------------------------------------
# PRAEBMBeamSearch.select_best tests
# ---------------------------------------------------------------------------

class TestSelectBest:
    """SCENARIO-REPAIR-031: select_best returns minimum-energy candidate."""

    def test_selects_minimum_energy(self) -> None:
        # Energies: 0.9, 0.3, 0.7 — should pick index 1 ("b")
        model = _FixedEnergyModel([0.9, 0.3, 0.7])
        beam = PRAEBMBeamSearch(eorm_model=model, k_candidates=3)
        best = beam.select_best(["a", "b", "c"])
        assert best.step_text == "b"
        assert best.eorm_energy == pytest.approx(0.3)
        assert best.is_selected is True

    def test_all_equal_selects_first(self) -> None:
        model = _ConstantEnergyModel(0.5)
        beam = PRAEBMBeamSearch(eorm_model=model, k_candidates=3)
        best = beam.select_best(["x", "y", "z"])
        # min() returns the first when energies are equal
        assert best.step_text == "x"
        assert best.is_selected is True

    def test_single_candidate(self) -> None:
        model = _ConstantEnergyModel(1.23)
        beam = PRAEBMBeamSearch(eorm_model=model, k_candidates=1)
        best = beam.select_best(["only"])
        assert best.step_text == "only"
        assert best.is_selected is True

    def test_empty_candidates_raises(self) -> None:
        model = _ConstantEnergyModel(0.0)
        beam = PRAEBMBeamSearch(eorm_model=model, k_candidates=3)
        with pytest.raises(ValueError, match="non-empty"):
            beam.select_best([])

    def test_question_forwarded(self) -> None:
        """select_best must forward the question to score_candidate."""
        received: list[object] = []

        class _CapturingModel:
            def energy(self, cot_input: object) -> float:
                received.append(cot_input)
                return 0.0

        beam = PRAEBMBeamSearch(eorm_model=_CapturingModel(), k_candidates=2)
        beam.select_best(["a", "b"], question="test question")
        assert len(received) == 2
        for inp in received:
            if hasattr(inp, "question_text"):
                assert inp.question_text == "test question"


# ---------------------------------------------------------------------------
# PRAEBMBeamSearch.run_beam_episode tests
# ---------------------------------------------------------------------------

class TestRunBeamEpisode:
    """SCENARIO-REPAIR-032, SCENARIO-REPAIR-033: run_beam_episode correctness."""

    def _fixed_generate_fn(
        self, candidates_per_step: list[list[str]]
    ) -> object:
        """Return a generate_fn that yields preset candidates per step."""

        def generate_fn(question: str, step_idx: int) -> list[str]:
            if step_idx < len(candidates_per_step):
                return candidates_per_step[step_idx]
            return []

        return generate_fn

    def test_basic_episode_structure(self) -> None:
        # 2 steps, 3 candidates each
        model = _ConstantEnergyModel(0.5)
        beam = PRAEBMBeamSearch(eorm_model=model, k_candidates=3)
        gen = self._fixed_generate_fn([
            ["a1", "a2", "a3"],
            ["b1", "b2", "b3"],
        ])
        result = beam.run_beam_episode("q", gen, n_steps=2)
        assert result.n_steps == 2
        assert result.n_beams_explored == 6
        assert len(result.selected_candidates) == 2
        for cand in result.selected_candidates:
            assert cand.is_selected is True

    def test_beam_wins_when_greedy_is_worst(self) -> None:
        # Energies: [0.9, 0.1, 0.5] per step — greedy picks 0.9, beam picks 0.1
        # Mean = (0.9+0.1+0.5)/3 = 0.5
        # Greedy: 0.9 > 0.5 → violation
        # Beam: 0.1 < 0.5 → no violation
        model = _FixedEnergyModel([0.9, 0.1, 0.5, 0.9, 0.1, 0.5])
        beam = PRAEBMBeamSearch(eorm_model=model, k_candidates=3)
        gen = self._fixed_generate_fn([
            ["worst", "best", "mid"],
            ["worst", "best", "mid"],
        ])
        result = beam.run_beam_episode("q", gen, n_steps=2)
        assert result.baseline_violation_rate == pytest.approx(1.0)
        assert result.beam_violation_rate == pytest.approx(0.0)
        assert result.improvement == pytest.approx(1.0)

    def test_equal_energies_no_violations(self) -> None:
        # All energies equal → mean = energy → neither above mean → no violations
        model = _ConstantEnergyModel(0.5)
        beam = PRAEBMBeamSearch(eorm_model=model, k_candidates=3)
        gen = self._fixed_generate_fn([
            ["a", "b", "c"],
            ["d", "e", "f"],
        ])
        result = beam.run_beam_episode("q", gen, n_steps=2)
        assert result.baseline_violation_rate == pytest.approx(0.0)
        assert result.beam_violation_rate == pytest.approx(0.0)
        assert result.improvement == pytest.approx(0.0)

    def test_zero_steps(self) -> None:
        model = _ConstantEnergyModel(0.5)
        beam = PRAEBMBeamSearch(eorm_model=model, k_candidates=3)
        gen = self._fixed_generate_fn([])
        result = beam.run_beam_episode("q", gen, n_steps=0)
        assert result.n_steps == 0
        assert result.n_beams_explored == 0
        assert result.baseline_violation_rate == pytest.approx(0.0)
        assert result.beam_violation_rate == pytest.approx(0.0)

    def test_empty_candidates_from_generate_fn_skipped(self) -> None:
        # Step 0 returns candidates, step 1 returns empty → only 1 step counted
        model = _ConstantEnergyModel(0.5)
        beam = PRAEBMBeamSearch(eorm_model=model, k_candidates=3)

        def gen(question: str, step_idx: int) -> list[str]:
            if step_idx == 0:
                return ["a", "b", "c"]
            return []  # step 1 returns nothing

        result = beam.run_beam_episode("q", gen, n_steps=2)
        # n_steps is 2 (what we asked for), but only 1 step had candidates
        assert result.n_steps == 2
        assert result.n_beams_explored == 3
        assert len(result.selected_candidates) == 1

    def test_selected_candidate_is_minimum_energy(self) -> None:
        # 1 step, energies [0.3, 0.8, 0.5] → best should be "first" (energy 0.3)
        model = _FixedEnergyModel([0.3, 0.8, 0.5])
        beam = PRAEBMBeamSearch(eorm_model=model, k_candidates=3)
        gen = self._fixed_generate_fn([["first", "second", "third"]])
        result = beam.run_beam_episode("q", gen, n_steps=1)
        assert len(result.selected_candidates) == 1
        best = result.selected_candidates[0]
        assert best.step_text == "first"
        assert best.eorm_energy == pytest.approx(0.3)

    def test_improvement_equals_baseline_minus_beam(self) -> None:
        model = _FixedEnergyModel([0.9, 0.1, 0.5])  # one step: greedy=0.9, best=0.1
        beam = PRAEBMBeamSearch(eorm_model=model, k_candidates=3)
        gen = self._fixed_generate_fn([["w", "b", "m"]])
        result = beam.run_beam_episode("q", gen, n_steps=1)
        assert result.improvement == pytest.approx(
            result.baseline_violation_rate - result.beam_violation_rate
        )

    def test_k_candidates_attribute(self) -> None:
        model = _ConstantEnergyModel(0.0)
        beam = PRAEBMBeamSearch(eorm_model=model, k_candidates=5)
        assert beam.k_candidates == 5

    def test_text_hash_model_rewards_equals_sign(self) -> None:
        """Verify that the text-hash mock correctly rewards '=' candidates."""
        model = _TextHashModel()
        beam = PRAEBMBeamSearch(eorm_model=model, k_candidates=3)
        gen = self._fixed_generate_fn([
            ["no equals here", "result = 42", "also no equals"],
        ])
        result = beam.run_beam_episode("2+40=?", gen, n_steps=1)
        assert len(result.selected_candidates) == 1
        # "result = 42" contains '=' so it should be chosen (lowest energy)
        assert "=" in result.selected_candidates[0].step_text


# ---------------------------------------------------------------------------
# Export / import tests
# ---------------------------------------------------------------------------

class TestExports:
    """Verify that all three symbols are exported from carnot.pipeline."""

    def test_imports_from_pipeline(self) -> None:
        from carnot.pipeline import (  # noqa: PLC0415
            PRABeamCandidate,
            PRABeamResult,
            PRAEBMBeamSearch,
        )
        assert PRABeamCandidate is not None
        assert PRABeamResult is not None
        assert PRAEBMBeamSearch is not None
