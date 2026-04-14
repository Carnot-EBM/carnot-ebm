"""Tests for Exp 272: FactualExtractor on Gemma4-E4B-it responses.

**Detailed explanation for engineers:**
    Tests the ``gemma4_factual_extraction`` module, which measures whether
    FactualExtractor's Wikidata-backed extraction pipeline (proven at 96%
    coverage in Exp 158 on synthetic sentences) generalises to natural,
    prose-style responses representative of google/gemma-4-E4B-it output.

    All tests run offline — no live Gemma4 inference and no Wikidata network
    calls. Network calls are either replaced by a stub FactualExtractor or
    tested using the pre-built GEMMA4_RESPONSES that contain verifiable
    claim patterns (e.g. "Paris is the capital of France").

    Test coverage:
    - QUESTION_BANK and GEMMA4_RESPONSES have exactly 20 entries (REQ-VERIFY-001)
    - run_extraction_on_responses returns one ExtractionResult per pair (REQ-VERIFY-001)
    - run_extraction_on_responses sets covered=True when ≥1 constraint (REQ-VERIFY-002)
    - run_extraction_on_responses sets covered=False when extractor returns [] (REQ-VERIFY-002)
    - ExtractionResult.has_verified is True for factual_verified constraints (REQ-VERIFY-001)
    - ExtractionResult.has_contradicted is True for factual_contradicted constraints (REQ-VERIFY-001)
    - compute_metrics coverage_pct = n_covered / n * 100 (REQ-VERIFY-001)
    - compute_metrics coverage_target_met when coverage_pct >= 70 (REQ-VERIFY-001)
    - compute_metrics accuracy_pct = n_covered_verified_only / n_covered * 100 (REQ-VERIFY-002)
    - compute_metrics accuracy_target_met when accuracy_pct >= 75 (REQ-VERIFY-002)
    - compute_metrics delta_coverage_vs_158 is negative for lower coverage (REQ-VERIFY-001)
    - compute_metrics domain_breakdown per-domain keys (REQ-VERIFY-001)
    - build_results_payload schema has required top-level keys (REQ-VERIFY-001)
    - build_results_payload per_question length == n_results (REQ-VERIFY-001)
    - build_results_payload includes exp158_baseline (REQ-VERIFY-002)
    - run_exp272 with stub extractor returns correct payload keys (REQ-VERIFY-001)
    - GEMMA4_RESPONSES contain claim patterns that FactualExtractor can detect (SCENARIO-VERIFY-002)

Spec: REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-002
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from carnot.pipeline.extract import ConstraintResult
from carnot.pipeline.gemma4_factual_extraction import (
    EXP158_ACCURACY_PCT,
    EXP158_COVERAGE_PCT,
    GEMMA4_RESPONSES,
    QUESTION_BANK,
    ExtractionResult,
    build_results_payload,
    compute_metrics,
    run_exp272,
    run_extraction_on_responses,
)


# ---------------------------------------------------------------------------
# Shared helpers / stubs
# ---------------------------------------------------------------------------


def _make_verified_constraint() -> ConstraintResult:
    """Return a ConstraintResult with constraint_type 'factual_verified'.

    Spec: REQ-VERIFY-001
    """
    return ConstraintResult(
        constraint_type="factual_verified",
        description="KB confirms: France capital Paris",
        metadata={"subject": "France", "predicate": "capital", "object": "Paris", "verified": True},
    )


def _make_contradicted_constraint() -> ConstraintResult:
    """Return a ConstraintResult with constraint_type 'factual_contradicted'.

    Spec: REQ-VERIFY-001
    """
    return ConstraintResult(
        constraint_type="factual_contradicted",
        description="KB contradicts: France capital London",
        metadata={"subject": "France", "predicate": "capital", "object": "London", "verified": False},
    )


class _StubExtractor:
    """Offline stub that returns pre-configured constraints without Wikidata calls.

    Spec: REQ-VERIFY-001
    """

    def __init__(self, constraints_per_call: list[list[ConstraintResult]]) -> None:
        """Each call to extract() pops the next entry from constraints_per_call."""
        self._queue = list(constraints_per_call)

    @property
    def supported_domains(self) -> list[str]:
        return ["factual"]

    def extract(self, text: str, domain: str | None = None) -> list[ConstraintResult]:
        if not self._queue:
            return []
        return self._queue.pop(0)


def _make_extraction_result(
    *,
    idx: int = 0,
    covered: bool = True,
    has_verified: bool = True,
    has_contradicted: bool = False,
    n_constraints: int = 1,
    domain: str = "geography",
) -> ExtractionResult:
    """Build a minimal ExtractionResult for metric tests.

    Spec: REQ-VERIFY-001
    """
    constraints = [_make_verified_constraint()] * n_constraints if n_constraints else []
    return ExtractionResult(
        question_idx=idx,
        question="What is the capital of France?",
        domain=domain,
        response="Paris is the capital of France.",
        constraints=constraints,
        covered=covered,
        has_verified=has_verified,
        has_contradicted=has_contradicted,
        elapsed_s=0.1,
    )


# ---------------------------------------------------------------------------
# Bank / response size checks
# ---------------------------------------------------------------------------


class TestBankSizes:
    """REQ-VERIFY-001: QUESTION_BANK and GEMMA4_RESPONSES have 20 entries each."""

    def test_question_bank_has_20_entries(self) -> None:
        """REQ-VERIFY-001: QUESTION_BANK length is exactly 20."""
        assert len(QUESTION_BANK) == 20

    def test_gemma4_responses_has_20_entries(self) -> None:
        """REQ-VERIFY-001: GEMMA4_RESPONSES length matches QUESTION_BANK."""
        assert len(GEMMA4_RESPONSES) == len(QUESTION_BANK)

    def test_question_bank_entries_have_required_keys(self) -> None:
        """REQ-VERIFY-001: every entry has question, domain, expected_claim_substring."""
        required = {"question", "domain", "expected_claim_substring"}
        for entry in QUESTION_BANK:
            assert required <= set(entry.keys()), f"Missing keys in {entry}"

    def test_question_bank_domains_are_valid(self) -> None:
        """REQ-VERIFY-001: domain is one of the four recognised categories."""
        valid_domains = {"geography", "history", "science", "person"}
        for entry in QUESTION_BANK:
            assert entry["domain"] in valid_domains

    def test_gemma4_responses_are_nonempty_strings(self) -> None:
        """REQ-VERIFY-001: every pre-built response is a non-empty string."""
        for response in GEMMA4_RESPONSES:
            assert isinstance(response, str) and len(response) > 10


# ---------------------------------------------------------------------------
# run_extraction_on_responses tests
# ---------------------------------------------------------------------------


class TestRunExtractionOnResponses:
    """REQ-VERIFY-001, REQ-VERIFY-002: extraction over QA pairs."""

    def test_returns_one_result_per_pair(self) -> None:
        """REQ-VERIFY-001: result list length equals input length."""
        qa_pairs = [("Q1", "R1"), ("Q2", "R2")]
        domains = ["geography", "geography"]
        stub = _StubExtractor([[], []])

        results = run_extraction_on_responses(qa_pairs, domains, extractor=stub)

        assert len(results) == 2

    def test_covered_true_when_constraints_returned(self) -> None:
        """REQ-VERIFY-002: covered=True iff extractor returns ≥1 constraint."""
        qa_pairs = [("What is the capital of France?", "Paris is the capital of France.")]
        domains = ["geography"]
        stub = _StubExtractor([[_make_verified_constraint()]])

        results = run_extraction_on_responses(qa_pairs, domains, extractor=stub)

        assert results[0].covered is True

    def test_covered_false_when_no_constraints_returned(self) -> None:
        """REQ-VERIFY-002: covered=False when extractor returns empty list."""
        qa_pairs = [("What is the meaning of life?", "42.")]
        domains = ["science"]
        stub = _StubExtractor([[]])  # empty

        results = run_extraction_on_responses(qa_pairs, domains, extractor=stub)

        assert results[0].covered is False

    def test_has_verified_true_for_factual_verified_type(self) -> None:
        """REQ-VERIFY-001: has_verified set when any constraint is factual_verified."""
        qa_pairs = [("What is the capital of France?", "Paris is the capital of France.")]
        domains = ["geography"]
        stub = _StubExtractor([[_make_verified_constraint()]])

        results = run_extraction_on_responses(qa_pairs, domains, extractor=stub)

        assert results[0].has_verified is True
        assert results[0].has_contradicted is False

    def test_has_contradicted_true_for_factual_contradicted_type(self) -> None:
        """REQ-VERIFY-001: has_contradicted set when any constraint is factual_contradicted."""
        qa_pairs = [("What is the capital of France?", "London is the capital of France.")]
        domains = ["geography"]
        stub = _StubExtractor([[_make_contradicted_constraint()]])

        results = run_extraction_on_responses(qa_pairs, domains, extractor=stub)

        assert results[0].has_contradicted is True
        assert results[0].has_verified is False

    def test_question_idx_matches_input_order(self) -> None:
        """REQ-VERIFY-001: question_idx is the 0-based index into qa_pairs."""
        qa_pairs = [("Q0", "R0"), ("Q1", "R1"), ("Q2", "R2")]
        domains = ["geography", "history", "science"]
        stub = _StubExtractor([[], [], []])

        results = run_extraction_on_responses(qa_pairs, domains, extractor=stub)

        assert [r.question_idx for r in results] == [0, 1, 2]

    def test_domain_stored_in_result(self) -> None:
        """REQ-VERIFY-001: ExtractionResult.domain matches input domains list."""
        qa_pairs = [("Q", "R")]
        domains = ["history"]
        stub = _StubExtractor([[]])

        results = run_extraction_on_responses(qa_pairs, domains, extractor=stub)

        assert results[0].domain == "history"

    def test_elapsed_s_is_non_negative(self) -> None:
        """REQ-VERIFY-001: elapsed time is measured and ≥0."""
        qa_pairs = [("Q", "R")]
        domains = ["geography"]
        stub = _StubExtractor([[]])

        results = run_extraction_on_responses(qa_pairs, domains, extractor=stub)

        assert results[0].elapsed_s >= 0.0

    def test_creates_default_extractor_when_none(self) -> None:
        """REQ-VERIFY-001: extractor=None creates a default FactualExtractor.

        We use a text with no verifiable claims so no network call is made.
        """
        qa_pairs = [("What is 2 plus 2?", "The answer is 4.")]
        domains = ["science"]

        # No network calls expected for this text (no named entities / claim patterns)
        results = run_extraction_on_responses(qa_pairs, domains, extractor=None)

        # Should return 1 result (may or may not be covered — we just check length)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# compute_metrics tests
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """REQ-VERIFY-001, REQ-VERIFY-002: metric computation."""

    def test_coverage_pct_all_covered(self) -> None:
        """REQ-VERIFY-001: 100% coverage when all results are covered."""
        results = [_make_extraction_result(idx=i) for i in range(10)]
        metrics = compute_metrics(results)
        assert metrics["coverage_pct"] == 100.0
        assert metrics["n_covered"] == 10

    def test_coverage_pct_none_covered(self) -> None:
        """REQ-VERIFY-001: 0% coverage when no results are covered."""
        results = [
            _make_extraction_result(idx=i, covered=False, has_verified=False, n_constraints=0)
            for i in range(5)
        ]
        metrics = compute_metrics(results)
        assert metrics["coverage_pct"] == 0.0
        assert metrics["n_covered"] == 0

    def test_coverage_pct_half_covered(self) -> None:
        """REQ-VERIFY-001: 50% coverage when half are covered."""
        covered = [_make_extraction_result(idx=i) for i in range(5)]
        not_covered = [
            _make_extraction_result(idx=i + 5, covered=False, has_verified=False, n_constraints=0)
            for i in range(5)
        ]
        metrics = compute_metrics(covered + not_covered)
        assert metrics["coverage_pct"] == 50.0

    def test_coverage_target_met_above_70(self) -> None:
        """REQ-VERIFY-001: coverage_target_met is True for ≥70%."""
        results = [_make_extraction_result(idx=i) for i in range(10)]
        metrics = compute_metrics(results)
        assert metrics["coverage_target_met"] is True

    def test_coverage_target_not_met_below_70(self) -> None:
        """REQ-VERIFY-001: coverage_target_met is False for <70%."""
        covered = [_make_extraction_result(idx=i) for i in range(6)]
        not_covered = [
            _make_extraction_result(idx=i + 6, covered=False, has_verified=False, n_constraints=0)
            for i in range(4)
        ]
        metrics = compute_metrics(covered + not_covered)
        assert metrics["coverage_pct"] == 60.0
        assert metrics["coverage_target_met"] is False

    def test_accuracy_pct_all_verified_no_contradictions(self) -> None:
        """REQ-VERIFY-002: 100% accuracy when all covered results are verified-only."""
        results = [_make_extraction_result(idx=i) for i in range(5)]
        metrics = compute_metrics(results)
        assert metrics["accuracy_pct"] == 100.0

    def test_accuracy_pct_none_verified(self) -> None:
        """REQ-VERIFY-002: 0% accuracy when covered but none are verified-only."""
        results = [
            _make_extraction_result(idx=i, has_verified=False, has_contradicted=True)
            for i in range(5)
        ]
        metrics = compute_metrics(results)
        assert metrics["accuracy_pct"] == 0.0

    def test_accuracy_pct_half_verified_no_contradictions(self) -> None:
        """REQ-VERIFY-002: 50% accuracy when half of covered have verified-only."""
        verified = [_make_extraction_result(idx=i) for i in range(5)]
        contradicted = [
            _make_extraction_result(idx=i + 5, has_verified=False, has_contradicted=True)
            for i in range(5)
        ]
        metrics = compute_metrics(verified + contradicted)
        assert metrics["accuracy_pct"] == 50.0

    def test_accuracy_target_met_above_75(self) -> None:
        """REQ-VERIFY-002: accuracy_target_met is True when accuracy ≥ 75%."""
        results = [_make_extraction_result(idx=i) for i in range(8)]
        metrics = compute_metrics(results)
        assert metrics["accuracy_target_met"] is True

    def test_delta_coverage_vs_158_negative_for_lower_coverage(self) -> None:
        """REQ-VERIFY-001: delta is negative when coverage is less than Exp 158's 96%."""
        # 50% coverage < 96%
        covered = [_make_extraction_result(idx=i) for i in range(5)]
        not_covered = [
            _make_extraction_result(idx=i + 5, covered=False, has_verified=False, n_constraints=0)
            for i in range(5)
        ]
        metrics = compute_metrics(covered + not_covered)
        assert metrics["delta_coverage_vs_158"] < 0

    def test_delta_coverage_vs_158_is_coverage_minus_exp158(self) -> None:
        """REQ-VERIFY-001: delta equals coverage_pct minus EXP158_COVERAGE_PCT."""
        results = [_make_extraction_result(idx=i) for i in range(10)]
        metrics = compute_metrics(results)
        expected_delta = metrics["coverage_pct"] - EXP158_COVERAGE_PCT
        assert abs(metrics["delta_coverage_vs_158"] - expected_delta) < 1e-9

    def test_domain_breakdown_keys_present(self) -> None:
        """REQ-VERIFY-001: domain_breakdown contains an entry per unique domain."""
        geo = _make_extraction_result(idx=0, domain="geography")
        hist = _make_extraction_result(idx=1, domain="history")
        sci = _make_extraction_result(idx=2, domain="science")
        metrics = compute_metrics([geo, hist, sci])
        assert "geography" in metrics["domain_breakdown"]
        assert "history" in metrics["domain_breakdown"]
        assert "science" in metrics["domain_breakdown"]

    def test_domain_breakdown_coverage_pct_correct(self) -> None:
        """REQ-VERIFY-001: per-domain coverage_pct is computed over domain subset."""
        covered = _make_extraction_result(idx=0, domain="geography")
        not_covered = _make_extraction_result(
            idx=1, domain="geography", covered=False, has_verified=False, n_constraints=0
        )
        metrics = compute_metrics([covered, not_covered])
        assert metrics["domain_breakdown"]["geography"]["coverage_pct"] == 50.0


# ---------------------------------------------------------------------------
# build_results_payload tests
# ---------------------------------------------------------------------------


class TestBuildResultsPayload:
    """REQ-VERIFY-001: results JSON schema."""

    def _metrics_and_results(self) -> tuple[dict[str, Any], list[ExtractionResult]]:
        results = [_make_extraction_result(idx=i) for i in range(4)]
        metrics = compute_metrics(results)
        return metrics, results

    def test_required_top_level_keys_present(self) -> None:
        """REQ-VERIFY-001: payload has experiment, model_name, and metric keys."""
        metrics, results = self._metrics_and_results()
        payload = build_results_payload(metrics, results)
        required_keys = {
            "experiment",
            "model_name",
            "live_model_used",
            "exp158_baseline",
            "coverage_pct",
            "accuracy_pct",
            "per_question",
        }
        assert required_keys <= set(payload.keys())

    def test_per_question_length_matches_results(self) -> None:
        """REQ-VERIFY-001: per_question list has one entry per ExtractionResult."""
        metrics, results = self._metrics_and_results()
        payload = build_results_payload(metrics, results)
        assert len(payload["per_question"]) == len(results)

    def test_exp158_baseline_populated(self) -> None:
        """REQ-VERIFY-002: payload includes Exp 158 baseline for comparison."""
        metrics, results = self._metrics_and_results()
        payload = build_results_payload(metrics, results)
        baseline = payload["exp158_baseline"]
        assert baseline["coverage_pct"] == EXP158_COVERAGE_PCT
        assert baseline["accuracy_pct"] == EXP158_ACCURACY_PCT

    def test_live_model_used_false_by_default(self) -> None:
        """REQ-VERIFY-001: live_model_used=False when not specified."""
        metrics, results = self._metrics_and_results()
        payload = build_results_payload(metrics, results)
        assert payload["live_model_used"] is False

    def test_live_model_used_true_when_specified(self) -> None:
        """REQ-VERIFY-001: live_model_used=True passes through."""
        metrics, results = self._metrics_and_results()
        payload = build_results_payload(metrics, results, live_model_used=True)
        assert payload["live_model_used"] is True

    def test_per_question_entry_has_required_keys(self) -> None:
        """REQ-VERIFY-001: each per_question entry has idx, covered, elapsed_s."""
        metrics, results = self._metrics_and_results()
        payload = build_results_payload(metrics, results)
        for entry in payload["per_question"]:
            assert "idx" in entry
            assert "question" in entry
            assert "domain" in entry
            assert "covered" in entry
            assert "elapsed_s" in entry

    def test_model_name_passed_through(self) -> None:
        """REQ-VERIFY-001: model_name kwarg appears in payload."""
        metrics, results = self._metrics_and_results()
        payload = build_results_payload(
            metrics, results, model_name="google/gemma-4-E4B-it"
        )
        assert payload["model_name"] == "google/gemma-4-E4B-it"


# ---------------------------------------------------------------------------
# run_exp272 integration test (offline)
# ---------------------------------------------------------------------------


class TestRunExp272:
    """REQ-VERIFY-001, REQ-VERIFY-002: end-to-end offline run."""

    def test_returns_dict_with_required_keys(self) -> None:
        """REQ-VERIFY-001: run_exp272 returns a payload dict with expected keys."""
        # Use stub extractor that returns one verified constraint for every call
        stub = _StubExtractor(
            [[_make_verified_constraint()] for _ in range(20)]
        )
        payload = run_exp272(use_live_model=False, extractor=stub)
        assert "coverage_pct" in payload
        assert "accuracy_pct" in payload
        assert "per_question" in payload

    def test_per_question_has_20_entries(self) -> None:
        """REQ-VERIFY-001: 20 questions produce 20 per_question entries."""
        stub = _StubExtractor([[] for _ in range(20)])
        payload = run_exp272(use_live_model=False, extractor=stub)
        assert len(payload["per_question"]) == 20

    def test_all_covered_when_stub_returns_constraints(self) -> None:
        """REQ-VERIFY-002: coverage_pct=100 when stub always returns ≥1 constraint."""
        stub = _StubExtractor(
            [[_make_verified_constraint()] for _ in range(20)]
        )
        payload = run_exp272(use_live_model=False, extractor=stub)
        assert payload["coverage_pct"] == 100.0
        assert payload["coverage_target_met"] is True

    def test_zero_coverage_when_stub_returns_empty(self) -> None:
        """REQ-VERIFY-002: coverage_pct=0 when stub never returns constraints."""
        stub = _StubExtractor([[] for _ in range(20)])
        payload = run_exp272(use_live_model=False, extractor=stub)
        assert payload["coverage_pct"] == 0.0

    def test_use_live_model_reads_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """REQ-VERIFY-001: use_live_model=None reads CARNOT_LIVE_MODEL env var (=0 → False)."""
        monkeypatch.setenv("CARNOT_LIVE_MODEL", "0")
        stub = _StubExtractor([[] for _ in range(20)])
        payload = run_exp272(use_live_model=None, extractor=stub)
        # live_model_used=False because env var is "0"
        assert payload["live_model_used"] is False

    def test_use_live_model_true_calls_generate_responses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-VERIFY-001: use_live_model=True calls generate_responses_with_gemma4."""
        import carnot.pipeline.gemma4_factual_extraction as mod

        fake_responses = GEMMA4_RESPONSES  # reuse pre-built for the mock
        monkeypatch.setattr(mod, "generate_responses_with_gemma4", lambda qs, **kw: fake_responses)
        stub = _StubExtractor([[] for _ in range(20)])
        payload = run_exp272(use_live_model=True, extractor=stub)
        assert payload["live_model_used"] is True


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-002: representative responses contain extractable claims
# ---------------------------------------------------------------------------


class TestGemma4ResponsesContainExtractableClaims:
    """SCENARIO-VERIFY-002: pre-built responses have claim patterns FactualExtractor parses."""

    @pytest.mark.parametrize(
        "response_idx,expected_substring",
        [
            (0, "paris"),       # "Paris is the capital of France."
            (1, "tokyo"),       # "The capital of Japan is Tokyo."
            (2, "brasília"),    # "The capital of Brazil is Brasília."
            (3, "canberra"),    # "Canberra is the capital of Australia."
            (4, "ottawa"),      # "Ottawa is the capital of Canada."
            (5, "new delhi"),   # "New Delhi is the capital of India."
        ],
    )
    def test_geography_responses_contain_capital_claim(
        self, response_idx: int, expected_substring: str
    ) -> None:
        """SCENARIO-VERIFY-002: capital claim appears in pre-built response."""
        response = GEMMA4_RESPONSES[response_idx]
        assert expected_substring in response.lower(), (
            f"Response {response_idx!r} does not contain {expected_substring!r}: {response!r}"
        )

    def test_year_responses_contain_year_digits(self) -> None:
        """SCENARIO-VERIFY-002: history responses contain the expected year."""
        year_map = {
            6: "1945",  # WWII
            7: "1989",  # Berlin Wall
            8: "1945",  # UN
            9: "1969",  # Moon landing
        }
        for idx, year in year_map.items():
            assert year in GEMMA4_RESPONSES[idx], (
                f"Response {idx} missing year {year}: {GEMMA4_RESPONSES[idx]!r}"
            )

    def test_einstein_response_contains_german(self) -> None:
        """SCENARIO-VERIFY-002: Einstein response mentions German nationality."""
        assert "german" in GEMMA4_RESPONSES[15].lower()

    def test_armstrong_response_contains_moon(self) -> None:
        """SCENARIO-VERIFY-002: Armstrong response contains 'Moon'."""
        assert "moon" in GEMMA4_RESPONSES[16].lower()

    def test_hawking_response_contains_black_hole(self) -> None:
        """SCENARIO-VERIFY-002: Hawking response mentions black holes."""
        response_lower = GEMMA4_RESPONSES[19].lower()
        assert "black hole" in response_lower


# ---------------------------------------------------------------------------
# generate_responses_with_gemma4 coverage (mocked — no live model required)
# ---------------------------------------------------------------------------


class TestGenerateResponsesWithGemma4:
    """REQ-VERIFY-001: generate_responses_with_gemma4 calls load_model + generate."""

    def test_calls_load_model_and_generate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """REQ-VERIFY-001: function loads model and calls generate once per question."""
        import carnot.pipeline.gemma4_factual_extraction as mod

        # Stub load_model to return fake model/tokenizer objects
        fake_model = object()
        fake_tokenizer = object()
        load_model_calls: list[str] = []
        generate_calls: list[tuple[Any, Any, str]] = []

        def fake_load_model(model_name: str) -> tuple[object, object]:
            load_model_calls.append(model_name)
            return fake_model, fake_tokenizer

        def fake_generate(model: object, tokenizer: object, prompt: str, **kwargs: Any) -> str:
            generate_calls.append((model, tokenizer, prompt))
            return f"Response to: {prompt}"

        # Patch the module-level import inside the function by patching the module
        import carnot.inference.model_loader as ml_mod

        monkeypatch.setattr(ml_mod, "load_model", fake_load_model)
        monkeypatch.setattr(ml_mod, "generate", fake_generate)

        from carnot.pipeline.gemma4_factual_extraction import generate_responses_with_gemma4

        questions = ["What is the capital of France?", "What is 1 + 1?"]
        responses = generate_responses_with_gemma4(questions)

        assert len(responses) == 2
        assert len(generate_calls) == 2
        assert generate_calls[0][2] == "What is the capital of France?"
        assert generate_calls[1][2] == "What is 1 + 1?"
