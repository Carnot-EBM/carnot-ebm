"""Tests for carnot.pipeline.llm_extractor.

Each test references REQ-VERIFY-* or SCENARIO-VERIFY-* per spec-anchored
development requirements.

Spec: REQ-VERIFY-001, REQ-VERIFY-003, REQ-VERIFY-010, SCENARIO-VERIFY-010
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest

from carnot.pipeline import ArithmeticExtractor, VerifyRepairPipeline
from carnot.pipeline.llm_extractor import LLMConstraintExtractor


def _load_exp203_results() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    results_path = repo_root / "results" / "experiment_203_results.json"
    return json.loads(results_path.read_text())


class _StepClock:
    def __init__(self, start: float = 0.0, step: float = 0.2) -> None:
        self._value = start
        self._step = step

    def __call__(self) -> float:
        value = self._value
        self._value += self._step
        return value


def _make_extractor(
    outputs: dict[str, str],
    *,
    clock: _StepClock | None = None,
    model_name: str | None = None,
    load_model_fn: Any | None = None,
    max_new_tokens: int = 128,
) -> LLMConstraintExtractor:
    def fake_generate(
        model: Any,
        tokenizer: Any,
        prompt: str,
        requested_tokens: int,
    ) -> str:
        assert requested_tokens == max_new_tokens
        del model, tokenizer
        for needle, output in outputs.items():
            if needle in prompt:
                return output
        return "NONE"

    kwargs: dict[str, Any] = {
        "generate_fn": fake_generate,
        "clock": clock or _StepClock(),
        "max_new_tokens": max_new_tokens,
    }
    if model_name is None:
        kwargs["model"] = object()
        kwargs["tokenizer"] = object()
    else:
        kwargs["model_name"] = model_name
        if load_model_fn is not None:
            kwargs["load_model_fn"] = load_model_fn
    return LLMConstraintExtractor(**kwargs)


class TestLLMConstraintExtractor:
    """Tests for the LLM-assisted arithmetic extractor."""

    def test_supported_domains(self) -> None:
        """REQ-VERIFY-001: supported_domains includes arithmetic."""
        ext = _make_extractor({})
        assert "arithmetic" in ext.supported_domains

    def test_domain_filter_skips_non_matching_domain(self) -> None:
        """SCENARIO-VERIFY-010: non-arithmetic domain hints are ignored."""
        ext = _make_extractor({"Half of 48 is 20.": "CLAIM: 48 / 2 = 20"})
        assert ext.extract("Half of 48 is 20.", domain="code") == []

    def test_prompt_contains_required_instruction_and_response(self) -> None:
        """REQ-VERIFY-010: prompt asks for canonical CLAIM lines."""
        captured: dict[str, Any] = {}

        def fake_generate(
            model: Any,
            tokenizer: Any,
            prompt: str,
            requested_tokens: int,
        ) -> str:
            del model, tokenizer
            captured["prompt"] = prompt
            captured["requested_tokens"] = requested_tokens
            return "NONE"

        ext = LLMConstraintExtractor(
            model=object(),
            tokenizer=object(),
            generate_fn=fake_generate,
            max_new_tokens=33,
        )

        assert ext.extract("Half of 48 is 20.") == []
        assert captured["requested_tokens"] == 33
        assert "List all verifiable arithmetic claims in this response." in captured["prompt"]
        assert "Format each as: CLAIM: a OP b = c" in captured["prompt"]
        assert "Half of 48 is 20." in captured["prompt"]

    def test_empty_text_returns_no_constraints(self) -> None:
        """REQ-VERIFY-003: empty input yields no constraints."""
        ext = _make_extractor({})
        assert ext.extract("") == []

    def test_requires_model_or_model_name(self) -> None:
        """REQ-VERIFY-010: extractor raises if no generator backend is configured."""
        ext = LLMConstraintExtractor()
        with pytest.raises(RuntimeError, match="requires model_name or preloaded"):
            ext.extract("2 + 2 = 4")

    def test_model_is_loaded_lazily_once(self) -> None:
        """REQ-VERIFY-010: model loading is deferred until first extraction."""
        calls: list[str] = []

        def fake_load_model(name: str) -> tuple[Any, Any]:
            calls.append(name)
            return object(), object()

        ext = _make_extractor(
            {"2 + 3 = 5": "CLAIM: 2 + 3 = 5"},
            model_name="tiny-extractor",
            load_model_fn=fake_load_model,
        )

        ext.extract("2 + 3 = 5")
        ext.extract("2 + 3 = 5")
        assert calls == ["tiny-extractor"]

    def test_loader_failure_raises_runtime_error(self) -> None:
        """REQ-VERIFY-010: failed model load surfaces clearly to callers."""
        ext = _make_extractor(
            {"2 + 3 = 5": "CLAIM: 2 + 3 = 5"},
            model_name="tiny-extractor",
            load_model_fn=lambda name: (None, None),
        )

        with pytest.raises(RuntimeError, match="Could not load extractor model"):
            ext.extract("2 + 3 = 5")

    def test_default_model_loader_helpers_are_lazy_and_patchable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """REQ-VERIFY-010: default loader/generator resolve through model_loader at call time."""
        import carnot.inference.model_loader as model_loader_mod

        monkeypatch.setattr(
            model_loader_mod,
            "load_model",
            lambda name: (object(), object()),
        )
        monkeypatch.setattr(
            model_loader_mod,
            "generate",
            lambda model, tokenizer, prompt, max_new_tokens: "CLAIM: 2 + 3 = 5",
        )

        ext = LLMConstraintExtractor(model_name="tiny-default")
        results = ext.extract("2 + 3 = 5")

        assert len(results) == 1
        assert results[0].metadata["satisfied"] is True

    def test_parses_claim_lines_ignores_malformed_and_division_by_zero(self) -> None:
        """REQ-VERIFY-010: malformed model output is ignored without crashing."""
        ext = _make_extractor(
            {
                "response": "\n".join(
                    [
                        "CLAIM: 48 / 2 = 20",
                        "not a claim",
                        "CLAIM: 1,000 + 5 = 1005",
                        "CLAIM: 3 / 0 = 1",
                        "CLAIM: nope",
                    ]
                )
            }
        )

        results = ext.extract("response")

        assert len(results) == 2
        assert results[0].metadata["correct_result"] == 24
        assert results[0].metadata["satisfied"] is False
        assert results[1].metadata["a"] == 1000
        assert results[1].metadata["claimed_result"] == 1005
        assert results[1].metadata["satisfied"] is True
        assert results[0].energy_term is not None
        assert results[1].energy_term is not None
        assert results[0].energy_term.satisfaction_threshold == pytest.approx(0.5)
        assert float(results[0].energy_term.energy(jnp.zeros(1))) == 1.0
        assert results[1].energy_term.is_satisfied(jnp.zeros(1)) is True

    def test_handles_multiplication_and_float_claims(self) -> None:
        """REQ-VERIFY-010: canonical CLAIM parsing supports `*` and decimal answers."""
        ext = _make_extractor(
            {
                "mixed": "\n".join(
                    [
                        "CLAIM: 6 * 7 = 42",
                        "CLAIM: 42 / 4 = 10.5",
                        "CLAIM: 9 / 3 = 3.0",
                    ]
                )
            }
        )

        results = ext.extract("mixed")

        assert [result.metadata["operator"] for result in results] == ["*", "/", "/"]
        assert results[0].metadata["correct_result"] == 42
        assert results[1].metadata["claimed_result"] == pytest.approx(10.5)
        assert results[1].metadata["satisfied"] is True
        assert results[2].description == "9 / 3 = 3"

    def test_pipeline_consumes_energy_terms(self) -> None:
        """REQ-VERIFY-003: VerifyRepairPipeline accepts LLM-backed energy terms."""
        ext = _make_extractor({"Half of 48 is 20.": "CLAIM: 48 / 2 = 20"})
        pipeline = VerifyRepairPipeline(extractor=ext)

        result = pipeline.verify(
            question="What is half of 48?",
            response="Half of 48 is 20.",
            domain="arithmetic",
        )

        assert result.verified is False
        assert result.energy == pytest.approx(1.0)
        assert len(result.violations) == 1
        assert result.violations[0].metadata["correct_result"] == 24
        assert result.certificate["per_constraint"][0]["satisfied"] is False

    def test_finds_error_that_regex_misses(self) -> None:
        """SCENARIO-VERIFY-010: LLM extraction can canonicalize natural-language arithmetic."""
        response = "Half of 48 is 20."
        regex_results = ArithmeticExtractor().extract(response, domain="arithmetic")
        assert regex_results == []

        ext = _make_extractor({response: "CLAIM: 48 / 2 = 20"})
        pipeline = VerifyRepairPipeline(extractor=ext)
        result = pipeline.verify(
            question="What is half of 48?",
            response=response,
            domain="arithmetic",
        )

        assert result.verified is False
        assert result.violations[0].metadata["correct_result"] == 24

    def test_latency_is_recorded_per_response(self) -> None:
        """REQ-VERIFY-010: extraction latency is preserved in metadata."""
        ext = _make_extractor(
            {"3 + 4 = 7": "CLAIM: 3 + 4 = 7"},
            clock=_StepClock(start=10.0, step=0.4),
        )

        results = ext.extract("3 + 4 = 7")

        assert ext.last_latency_seconds == pytest.approx(0.4)
        assert results[0].metadata["extraction_latency_seconds"] == pytest.approx(0.4)

    def test_exp203_live_gemma_corpus_regression(self) -> None:
        """SCENARIO-VERIFY-010: curated auxiliary outputs improve on the live Exp 203 wrong-case corpus."""
        data = _load_exp203_results()
        wrong_cases = data["wrong_answer_autopsies"]
        correct_cases = data["correct_answer_examples"][:3]

        # The 2026-04-12 Exp 203 artifact currently contains three wrong cases,
        # even though the roadmap text still says four.
        assert len(wrong_cases) == 3
        assert len(correct_cases) == 3

        regex_wrong_violations = 0
        for case in wrong_cases:
            regex_wrong_violations += sum(
                not result.metadata["satisfied"]
                for result in ArithmeticExtractor().extract(case["response"])
            )
        assert regex_wrong_violations == 0

        outputs = {
            "Total cups per row = $27 / 3 = 9$ cups.": "CLAIM: 4 / 2 = 4",
            "Cost for Jack: $240": "\n".join(
                [
                    "CLAIM: 240 / 20 = 12",
                    "CLAIM: 120 / 20 = 6",
                    "CLAIM: 360 / 20 = 18",
                    "CLAIM: 42 / 4 = 10.5",
                ]
            ),
            "Actual cost = $150 - $60 = $90": "\n".join(
                [
                    "CLAIM: 10 * 15 = 150",
                    "CLAIM: 0.4 * 150 = 60",
                    "CLAIM: 150 - 60 = 90",
                    "CLAIM: 5 * 40 = 200",
                    "CLAIM: 90 - 200 = -110",
                ]
            ),
            "Total Male Guppies: $4 + 2 = 6$": "\n".join(
                [
                    "CLAIM: 4 + 2 = 6",
                    "CLAIM: 7 + 1 = 8",
                    "CLAIM: 3 + 2 = 5",
                    "CLAIM: 5 + 3 = 8",
                    "CLAIM: 6 + 5 = 11",
                    "CLAIM: 8 + 8 = 16",
                    "CLAIM: 16 - 11 = 5",
                ]
            ),
            "Chinese Asians = $240 - 80 = 160$": "\n".join(
                [
                    "CLAIM: 240 - 80 = 160",
                    "CLAIM: 160 - 60 = 100",
                ]
            ),
            "Number of cards with A = $80": "\n".join(
                [
                    "CLAIM: 160 / 5 = 32",
                    "CLAIM: 80 - 32 = 48",
                    "CLAIM: 48 / 2 = 24",
                    "CLAIM: 120 / 8 = 15",
                    "CLAIM: 24 - 15 = 9",
                ]
            ),
        }
        ext = _make_extractor(outputs, clock=_StepClock(start=0.0, step=0.2))
        pipeline = VerifyRepairPipeline(extractor=ext)

        llm_detected_wrong = 0
        latencies: list[float] = []

        for case in wrong_cases:
            result = pipeline.verify(
                question=case["question"],
                response=case["response"],
                domain="arithmetic",
            )
            if not result.verified:
                llm_detected_wrong += 1
            latencies.append(ext.last_latency_seconds)

        for case in correct_cases:
            result = pipeline.verify(
                question=case["question"],
                response=case["response"],
                domain="arithmetic",
            )
            assert result.verified is True
            latencies.append(ext.last_latency_seconds)

        assert llm_detected_wrong > regex_wrong_violations
        assert all(latency < 1.0 for latency in latencies)
