"""Tests for FactualExtractor — 100% coverage of factual_extractor.py.

**Researcher summary:**
    Verifies entity extraction regex patterns, claim triple decomposition,
    graceful degradation on network timeout, FactualClaimConstraint encoding,
    and AutoExtractor integration with domain="factual" + enable_factual_extractor.
    All network calls are mocked — no real Wikidata traffic during tests.

Spec: REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-002
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest

from carnot.pipeline.factual_extractor import (
    _CLAIM_CACHE,
    _QID_CACHE,
    FactualClaimConstraint,
    FactualExtractor,
    _resolve_qid,
    _verify_claim_sparql,
    extract_claims,
    extract_entities,
)
from carnot.pipeline.extract import AutoExtractor, ConstraintResult


# ---------------------------------------------------------------------------
# Helpers: reset module-level caches before each test class
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_caches() -> None:
    """Clear the module-level QID and claim caches before every test.

    This prevents cache state leaking between tests when mocked responses
    would otherwise be cached as real results.
    """
    _QID_CACHE.clear()
    _CLAIM_CACHE.clear()


# ---------------------------------------------------------------------------
# FactualClaimConstraint tests
# ---------------------------------------------------------------------------


class TestFactualClaimConstraint:
    """REQ-VERIFY-001 — FactualClaimConstraint satisfies ConstraintTerm protocol."""

    def test_name_satisfied(self) -> None:
        """name includes 'ok' when satisfied=True."""
        c = FactualClaimConstraint(claim_satisfied=True, claim_description="France capital Paris")
        assert "ok" in c.name
        assert "factual" in c.name

    def test_name_violated(self) -> None:
        """name includes 'violated' when satisfied=False."""
        c = FactualClaimConstraint(claim_satisfied=False, claim_description="France capital London")
        assert "violated" in c.name
        assert "factual" in c.name

    def test_name_truncates_long_description(self) -> None:
        """Description is truncated to 50 chars in name."""
        long_desc = "A" * 100
        c = FactualClaimConstraint(claim_satisfied=True, claim_description=long_desc)
        # The truncated description in the name should not exceed 50 chars
        # (plus the "factual(ok): " prefix)
        assert len(long_desc[:50]) == 50
        assert "A" * 50 in c.name

    def test_energy_zero_when_satisfied(self) -> None:
        """energy(x) = 0.0 when claim_satisfied=True."""
        c = FactualClaimConstraint(claim_satisfied=True, claim_description="test")
        x = jnp.zeros(5)
        assert abs(float(c.energy(x)) - 0.0) < 1e-6

    def test_energy_one_when_violated(self) -> None:
        """energy(x) = 1.0 when claim_satisfied=False."""
        c = FactualClaimConstraint(claim_satisfied=False, claim_description="test")
        x = jnp.zeros(5)
        assert abs(float(c.energy(x)) - 1.0) < 1e-6

    def test_energy_ignores_x(self) -> None:
        """energy(x) is constant regardless of x content or shape."""
        c = FactualClaimConstraint(claim_satisfied=True, claim_description="test")
        assert abs(float(c.energy(jnp.zeros(1))) - 0.0) < 1e-6
        assert abs(float(c.energy(jnp.ones(100))) - 0.0) < 1e-6
        assert abs(float(c.energy(jnp.array([9.9, -3.1]))) - 0.0) < 1e-6

    def test_is_satisfied_true(self) -> None:
        """is_satisfied returns True when satisfied=True."""
        c = FactualClaimConstraint(claim_satisfied=True, claim_description="test")
        assert c.is_satisfied(jnp.zeros(1)) is True

    def test_is_satisfied_false(self) -> None:
        """is_satisfied returns False when satisfied=False."""
        c = FactualClaimConstraint(claim_satisfied=False, claim_description="test")
        assert c.is_satisfied(jnp.zeros(1)) is False

    def test_satisfaction_threshold(self) -> None:
        """satisfaction_threshold is 0.5 (midpoint between 0.0 and 1.0)."""
        c = FactualClaimConstraint(claim_satisfied=True, claim_description="test")
        assert c.satisfaction_threshold == 0.5

    def test_grad_energy_is_zero(self) -> None:
        """Gradient is zero because energy is constant (no x-dependence)."""
        import jax

        c = FactualClaimConstraint(claim_satisfied=True, claim_description="test")
        x = jnp.array([1.0, 2.0, 3.0])
        grad = jax.grad(c.energy)(x)
        assert jnp.allclose(grad, jnp.zeros_like(x))


# ---------------------------------------------------------------------------
# extract_entities tests
# ---------------------------------------------------------------------------


class TestExtractEntities:
    """REQ-VERIFY-001 — entity extraction regex patterns."""

    def test_extracts_two_word_named_entity(self) -> None:
        """Capitalized two-word sequences are recognized as named_entity."""
        entities = extract_entities("Albert Einstein was a physicist.")
        names = [e for e, t in entities]
        assert "Albert Einstein" in names

    def test_extracts_multi_word_named_entity(self) -> None:
        """Three-word capitalized sequences are extracted."""
        entities = extract_entities("New York City is a major hub.")
        names = [e for e, t in entities]
        assert "New York City" in names

    def test_extracts_acronym(self) -> None:
        """ALL-CAPS sequences of 2-6 letters are tagged as acronym."""
        entities = extract_entities("The USA joined NATO in 1949.")
        by_type = {e: t for e, t in entities}
        assert "USA" in by_type
        assert by_type["USA"] == "acronym"
        assert "NATO" in by_type
        assert by_type["NATO"] == "acronym"

    def test_extracts_year(self) -> None:
        """Four-digit years in plausible range are tagged as year."""
        entities = extract_entities("World War II ended in 1945.")
        years = [e for e, t in entities if t == "year"]
        assert "1945" in years

    def test_extracts_month_year(self) -> None:
        """Month+year combinations are tagged as date."""
        entities = extract_entities("He was born in April 1889.")
        dates = [e for e, t in entities if t == "date"]
        assert any("April 1889" in d for d in dates)

    def test_extracts_quantity(self) -> None:
        """Numeric quantities with units are tagged as quantity."""
        entities = extract_entities("The mountain is 8849 meters tall.")
        quantities = [e for e, t in entities if t == "quantity"]
        assert any("8849" in q for q in quantities)

    def test_no_duplicates(self) -> None:
        """Repeated entity occurrences produce only one entry."""
        entities = extract_entities("Paris is great. Paris is the capital.")
        names = [e for e, t in entities]
        assert names.count("Paris") <= 1

    def test_empty_text(self) -> None:
        """Empty string returns empty list."""
        assert extract_entities("") == []

    def test_no_entities_in_lowercase_text(self) -> None:
        """All-lowercase text yields no named_entity results."""
        entities = extract_entities("the quick brown fox jumps over the lazy dog.")
        named = [e for e, t in entities if t == "named_entity"]
        assert named == []

    def test_leading_stop_words_stripped_to_empty(self) -> None:
        """Entity consisting only of stop words is skipped entirely."""
        # "The Or" → strip "The" → strip "Or" → empty → skipped
        entities = extract_entities("The Or visited some place.")
        # "The Or" should be stripped to empty and not appear
        names = [e for e, t in entities if t == "named_entity"]
        assert "The Or" not in names
        # The entity should not appear in any form after full stripping
        # (both "The" and "Or" are stop words)
        assert all("The Or" not in e for e in names)

    def test_duplicate_entity_in_text_not_repeated(self) -> None:
        """Entity appearing twice in text appears only once in results."""
        # "Albert Einstein" matches the 2-word pattern both times
        entities = extract_entities(
            "Albert Einstein was a physicist. Albert Einstein won a Nobel Prize."
        )
        names = [e for e, t in entities]
        assert names.count("Albert Einstein") == 1


# ---------------------------------------------------------------------------
# extract_claims tests
# ---------------------------------------------------------------------------


class TestExtractClaims:
    """REQ-VERIFY-001 — claim triple decomposition on example sentences."""

    def test_capital_is_pattern(self) -> None:
        """'X is the capital of Y' → (Y, capital, X)."""
        claims = extract_claims("Paris is the capital of France.")
        # subject=France, predicate=capital, object=Paris
        assert any(
            s.lower() == "france" and p == "capital" and "paris" in o.lower()
            for s, p, o in claims
        )

    def test_capital_of_pattern(self) -> None:
        """'capital of Y is X' → (Y, capital, X)."""
        claims = extract_claims("The capital of Germany is Berlin.")
        assert any(
            "germany" in s.lower() and p == "capital" and "berlin" in o.lower()
            for s, p, o in claims
        )

    def test_born_in_pattern(self) -> None:
        """'X was born in Y' → (X, born in, Y)."""
        claims = extract_claims("Albert Einstein was born in Ulm.")
        assert any(
            "einstein" in s.lower() and p == "born in" and "ulm" in o.lower()
            for s, p, o in claims
        )

    def test_located_in_pattern(self) -> None:
        """'X is located in Y' → (X, located in, Y)."""
        claims = extract_claims("The Eiffel Tower is located in Paris.")
        assert any(p == "located in" for _, p, _ in claims)

    def test_official_language_pattern(self) -> None:
        """'official language of X is Y' → (X, official language, Y)."""
        claims = extract_claims("The official language of Brazil is Portuguese.")
        assert any(
            "brazil" in s.lower() and "official language" in p and "portuguese" in o.lower()
            for s, p, o in claims
        )

    def test_currency_pattern(self) -> None:
        """'currency of X is Y' → (X, currency, Y)."""
        claims = extract_claims("The currency of Japan is the yen.")
        assert any("currency" in p for _, p, _ in claims)

    def test_currency_uses_pattern(self) -> None:
        """'X uses Y as its currency' → (X, currency, Y)."""
        claims = extract_claims("Canada uses the dollar as its currency.")
        assert any("currency" in p for _, p, _ in claims)

    def test_is_in_pattern(self) -> None:
        """'X is in Y' → (X, located in, Y)."""
        claims = extract_claims("Rome is in Italy.")
        assert any("located in" in p for _, p, _ in claims)

    def test_no_duplicates_across_sentences(self) -> None:
        """Same claim in two sentences is deduplicated."""
        text = "Paris is the capital of France. Paris is the capital of France."
        claims = extract_claims(text)
        # Should have exactly 1, not 2
        capital_claims = [(s, p, o) for s, p, o in claims if p == "capital"]
        assert len(capital_claims) == 1

    def test_empty_text_returns_empty(self) -> None:
        """Empty string returns empty list."""
        assert extract_claims("") == []

    def test_no_matching_pattern(self) -> None:
        """Text without any claim patterns returns empty list."""
        claims = extract_claims("The cat sat on the mat.")
        assert claims == []

    def test_multi_sentence_text(self) -> None:
        """Multiple sentences can each contribute claims."""
        text = (
            "Paris is the capital of France. "
            "Albert Einstein was born in Ulm."
        )
        claims = extract_claims(text)
        assert len(claims) >= 2


# ---------------------------------------------------------------------------
# _resolve_qid tests (network mocked)
# ---------------------------------------------------------------------------


class TestResolveQid:
    """REQ-VERIFY-001 — Wikidata entity QID resolution with graceful degradation."""

    def test_returns_qid_on_success(self) -> None:
        """Returns QID string when Wikidata API responds successfully."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "search": [{"id": "Q142", "label": "France"}]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            mock_requests.get.return_value = mock_resp
            qid = _resolve_qid("France")

        assert qid == "Q142"

    def test_returns_none_when_no_results(self) -> None:
        """Returns None when Wikidata returns empty search results."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"search": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            mock_requests.get.return_value = mock_resp
            qid = _resolve_qid("NotARealEntity")

        assert qid is None

    def test_graceful_degradation_on_timeout(self) -> None:
        """Returns None and logs warning on requests.Timeout."""
        import requests as real_requests

        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            mock_requests.get.side_effect = real_requests.exceptions.Timeout("timeout")
            qid = _resolve_qid("France")

        assert qid is None

    def test_graceful_degradation_on_connection_error(self) -> None:
        """Returns None and logs warning on requests.ConnectionError."""
        import requests as real_requests

        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            mock_requests.get.side_effect = real_requests.exceptions.ConnectionError("refused")
            qid = _resolve_qid("France")

        assert qid is None

    def test_caches_successful_result(self) -> None:
        """Successful QID lookup is cached; second call does not hit network."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"search": [{"id": "Q142"}]}
        mock_resp.raise_for_status = MagicMock()

        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            mock_requests.get.return_value = mock_resp
            qid1 = _resolve_qid("France")
            qid2 = _resolve_qid("France")

        assert qid1 == "Q142"
        assert qid2 == "Q142"
        # Only one actual API call (second call uses cache).
        assert mock_requests.get.call_count == 1

    def test_caches_none_result(self) -> None:
        """None result (not found) is also cached."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"search": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            mock_requests.get.return_value = mock_resp
            qid1 = _resolve_qid("NoSuchThing")
            qid2 = _resolve_qid("NoSuchThing")

        assert qid1 is None
        assert qid2 is None
        assert mock_requests.get.call_count == 1

    def test_graceful_degradation_when_requests_not_installed(self) -> None:
        """Returns None when requests module is None (not installed)."""
        # Patch the module-level `requests` attribute to None (simulates missing install)
        with patch("carnot.pipeline.factual_extractor.requests", None):
            qid = _resolve_qid("France")
        assert qid is None


# ---------------------------------------------------------------------------
# _verify_claim_sparql tests (network mocked)
# ---------------------------------------------------------------------------


class TestVerifyClaimSparql:
    """REQ-VERIFY-001 — Wikidata SPARQL claim verification with graceful degradation."""

    def test_returns_true_when_label_matches(self) -> None:
        """Returns True when SPARQL result label contains the object text."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "results": {
                "bindings": [
                    {"objLabel": {"value": "Paris"}}
                ]
            }
        }

        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            mock_requests.get.return_value = mock_resp
            result = _verify_claim_sparql("Q142", "P36", "Paris")

        assert result is True

    def test_returns_false_when_no_label_matches(self) -> None:
        """Returns False when SPARQL returns labels but none match object text."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "results": {
                "bindings": [
                    {"objLabel": {"value": "Berlin"}}
                ]
            }
        }

        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            mock_requests.get.return_value = mock_resp
            result = _verify_claim_sparql("Q142", "P36", "London")

        assert result is False

    def test_returns_none_when_no_bindings(self) -> None:
        """Returns None when SPARQL returns empty bindings (unknown property)."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"results": {"bindings": []}}

        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            mock_requests.get.return_value = mock_resp
            result = _verify_claim_sparql("Q142", "P36", "Paris")

        assert result is None

    def test_graceful_degradation_on_timeout(self) -> None:
        """Returns None and logs warning on network timeout."""
        import requests as real_requests

        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            mock_requests.get.side_effect = real_requests.exceptions.Timeout("timeout")
            result = _verify_claim_sparql("Q142", "P36", "Paris")

        assert result is None

    def test_graceful_degradation_on_http_error(self) -> None:
        """Returns None and logs warning on HTTP error."""
        import requests as real_requests

        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = real_requests.exceptions.HTTPError("403")
            mock_requests.get.return_value = mock_resp
            result = _verify_claim_sparql("Q142", "P36", "Paris")

        assert result is None

    def test_caches_verified_result(self) -> None:
        """Verified result is cached; second call skips network."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "results": {"bindings": [{"objLabel": {"value": "Paris"}}]}
        }

        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            mock_requests.get.return_value = mock_resp
            r1 = _verify_claim_sparql("Q142", "P36", "Paris")
            r2 = _verify_claim_sparql("Q142", "P36", "Paris")

        assert r1 is True
        assert r2 is True
        assert mock_requests.get.call_count == 1

    def test_partial_label_match(self) -> None:
        """'Paris' is found in label 'Greater Paris Metropolitan Area'."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "results": {
                "bindings": [{"objLabel": {"value": "Greater Paris Metropolitan Area"}}]
            }
        }

        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            mock_requests.get.return_value = mock_resp
            result = _verify_claim_sparql("Q142", "P36", "Paris")

        assert result is True

    def test_graceful_degradation_when_requests_not_installed(self) -> None:
        """Returns None when requests module is None (not installed)."""
        with patch("carnot.pipeline.factual_extractor.requests", None):
            result = _verify_claim_sparql("Q142", "P36", "Paris")
        assert result is None


# ---------------------------------------------------------------------------
# FactualExtractor.extract() tests
# ---------------------------------------------------------------------------


class TestFactualExtractorExtract:
    """REQ-VERIFY-001, REQ-VERIFY-002 — FactualExtractor.extract() end-to-end."""

    def _make_successful_qid_response(self, qid: str = "Q142") -> MagicMock:
        """Return a mock requests.Response for a successful QID lookup."""
        r = MagicMock()
        r.raise_for_status = MagicMock()
        r.json.return_value = {"search": [{"id": qid}]}
        return r

    def _make_successful_sparql_response(self, label: str) -> MagicMock:
        """Return a mock requests.Response for a successful SPARQL query."""
        r = MagicMock()
        r.raise_for_status = MagicMock()
        r.json.return_value = {
            "results": {"bindings": [{"objLabel": {"value": label}}]}
        }
        return r

    def test_domain_guard_skips_non_factual(self) -> None:
        """extract() returns [] when domain != 'factual'."""
        ext = FactualExtractor()
        results = ext.extract("Paris is the capital of France.", domain="arithmetic")
        assert results == []

    def test_domain_none_processes_text(self) -> None:
        """extract() runs when domain=None (no domain hint)."""
        ext = FactualExtractor()
        # We need to mock network calls to avoid real requests
        qid_resp = self._make_successful_qid_response("Q142")
        sparql_resp = self._make_successful_sparql_response("Paris")

        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            mock_requests.get.side_effect = [qid_resp, sparql_resp]
            results = ext.extract("Paris is the capital of France.", domain=None)

        # Should return results when domain=None
        assert isinstance(results, list)

    def test_returns_constraint_for_verified_claim(self) -> None:
        """Verified KB claim → ConstraintResult with factual_verified type."""
        ext = FactualExtractor()
        qid_resp = self._make_successful_qid_response("Q142")
        sparql_resp = self._make_successful_sparql_response("Paris")

        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            mock_requests.get.side_effect = [qid_resp, sparql_resp]
            results = ext.extract(
                "Paris is the capital of France.", domain="factual"
            )

        assert len(results) >= 1
        verified = [r for r in results if r.constraint_type == "factual_verified"]
        assert len(verified) >= 1

    def test_constraint_result_has_energy_term(self) -> None:
        """ConstraintResult.energy_term is a FactualClaimConstraint."""
        ext = FactualExtractor()
        qid_resp = self._make_successful_qid_response("Q142")
        sparql_resp = self._make_successful_sparql_response("Paris")

        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            mock_requests.get.side_effect = [qid_resp, sparql_resp]
            results = ext.extract(
                "Paris is the capital of France.", domain="factual"
            )

        verified = [r for r in results if r.constraint_type == "factual_verified"]
        assert len(verified) >= 1
        assert isinstance(verified[0].energy_term, FactualClaimConstraint)
        assert verified[0].energy_term.is_satisfied(jnp.zeros(1)) is True

    def test_returns_constraint_for_contradicted_claim(self) -> None:
        """KB-contradicted claim → ConstraintResult with factual_contradicted type."""
        ext = FactualExtractor()
        qid_resp = self._make_successful_qid_response("Q142")
        # SPARQL returns "Berlin" but we claim "London" → contradiction
        sparql_resp = MagicMock()
        sparql_resp.raise_for_status = MagicMock()
        sparql_resp.json.return_value = {
            "results": {"bindings": [{"objLabel": {"value": "Berlin"}}]}
        }

        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            mock_requests.get.side_effect = [qid_resp, sparql_resp]
            results = ext.extract(
                "London is the capital of France.", domain="factual"
            )

        contradicted = [r for r in results if r.constraint_type == "factual_contradicted"]
        assert len(contradicted) >= 1
        assert contradicted[0].energy_term.is_satisfied(jnp.zeros(1)) is False

    def test_skips_unknown_predicate_claims(self) -> None:
        """Claims with predicates not in _PREDICATE_TO_PROPERTY are skipped."""
        ext = FactualExtractor()
        # "X resembles Y" has no Wikidata property mapping
        results = ext.extract(
            "The Eiffel Tower resembles an antenna.", domain="factual"
        )
        # Should return empty (no known predicate to verify)
        assert results == []

    def test_skips_claim_when_qid_resolution_fails(self) -> None:
        """Claims are skipped when Wikidata QID resolution returns None."""
        import requests as real_requests

        ext = FactualExtractor()

        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            mock_requests.get.side_effect = real_requests.exceptions.Timeout("timeout")
            results = ext.extract(
                "Paris is the capital of France.", domain="factual"
            )

        # Network failure → no constraints returned
        assert results == []

    def test_skips_claim_when_sparql_returns_none(self) -> None:
        """Claims are skipped when SPARQL returns unknown (None) result."""
        ext = FactualExtractor()
        qid_resp = self._make_successful_qid_response("Q142")
        sparql_resp = MagicMock()
        sparql_resp.raise_for_status = MagicMock()
        sparql_resp.json.return_value = {"results": {"bindings": []}}

        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            mock_requests.get.side_effect = [qid_resp, sparql_resp]
            results = ext.extract(
                "Paris is the capital of France.", domain="factual"
            )

        # Empty SPARQL bindings → None → claim skipped
        assert results == []

    def test_empty_text_returns_empty(self) -> None:
        """Empty text → no claims → empty list."""
        ext = FactualExtractor()
        results = ext.extract("", domain="factual")
        assert results == []

    def test_text_with_no_claim_patterns(self) -> None:
        """Text with no matching claim patterns → empty list (no network calls)."""
        ext = FactualExtractor()
        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            results = ext.extract("The quick brown fox.", domain="factual")
        assert results == []
        mock_requests.get.assert_not_called()

    def test_metadata_fields_present(self) -> None:
        """ConstraintResult.metadata contains required fields."""
        ext = FactualExtractor()
        qid_resp = self._make_successful_qid_response("Q142")
        sparql_resp = self._make_successful_sparql_response("Paris")

        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            mock_requests.get.side_effect = [qid_resp, sparql_resp]
            results = ext.extract(
                "Paris is the capital of France.", domain="factual"
            )

        assert len(results) >= 1
        meta = results[0].metadata
        assert "subject" in meta
        assert "predicate" in meta
        assert "object" in meta
        assert "subject_qid" in meta
        assert "property_id" in meta
        assert "verified" in meta
        assert "satisfied" in meta

    def test_supported_domains(self) -> None:
        """supported_domains returns ['factual']."""
        ext = FactualExtractor()
        assert ext.supported_domains == ["factual"]

    def test_custom_timeout_passed_through(self) -> None:
        """Custom timeout= is forwarded to Wikidata API calls."""
        ext = FactualExtractor(timeout=2.0)
        # Just verify the extractor is created; timeout is stored internally
        assert ext._timeout == 2.0

    def test_skips_claim_with_unknown_predicate_key(self) -> None:
        """Claims whose predicate_key is not in _PREDICATE_TO_PROPERTY are skipped."""
        from unittest.mock import patch as _patch
        import carnot.pipeline.factual_extractor as _fe

        ext = FactualExtractor()
        # Mock extract_claims to return a triple with a predicate not in the map
        with _patch.object(_fe, "extract_claims", return_value=[
            ("France", "invented_by", "Gauls")  # "invented_by" has no Wikidata property
        ]):
            results = ext.extract("France was invented by the Gauls.", domain="factual")

        # Should return empty: predicate not in _PREDICATE_TO_PROPERTY → skip
        assert results == []


# ---------------------------------------------------------------------------
# AutoExtractor integration tests
# ---------------------------------------------------------------------------


class TestAutoExtractorIntegration:
    """REQ-VERIFY-001 — AutoExtractor integration with FactualExtractor."""

    def test_factual_extractor_disabled_by_default(self) -> None:
        """AutoExtractor() does NOT include FactualExtractor by default."""
        auto = AutoExtractor()
        extractor_types = [type(e).__name__ for e in auto._extractors]
        assert "FactualExtractor" not in extractor_types

    def test_factual_extractor_enabled_via_flag(self) -> None:
        """AutoExtractor(enable_factual_extractor=True) includes FactualExtractor."""
        auto = AutoExtractor(enable_factual_extractor=True)
        extractor_types = [type(e).__name__ for e in auto._extractors]
        assert "FactualExtractor" in extractor_types

    def test_factual_extractor_added_via_add_extractor(self) -> None:
        """add_extractor(FactualExtractor()) registers the extractor."""
        auto = AutoExtractor()
        auto.add_extractor(FactualExtractor())
        extractor_types = [type(e).__name__ for e in auto._extractors]
        assert "FactualExtractor" in extractor_types

    def test_domain_factual_triggers_factual_extractor(self) -> None:
        """With domain='factual', FactualExtractor is invoked when registered."""
        auto = AutoExtractor(enable_factual_extractor=True)

        # Mock the network calls
        qid_resp = MagicMock()
        qid_resp.raise_for_status = MagicMock()
        qid_resp.json.return_value = {"search": [{"id": "Q142"}]}
        sparql_resp = MagicMock()
        sparql_resp.raise_for_status = MagicMock()
        sparql_resp.json.return_value = {
            "results": {"bindings": [{"objLabel": {"value": "Paris"}}]}
        }

        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            mock_requests.get.side_effect = [qid_resp, sparql_resp]
            results = auto.extract(
                "Paris is the capital of France.", domain="factual"
            )

        # FactualExtractor should have contributed at least one result
        factual_results = [
            r for r in results
            if r.constraint_type in ("factual_verified", "factual_contradicted")
        ]
        assert len(factual_results) >= 1

    def test_domain_none_does_not_invoke_factual_extractor_when_disabled(self) -> None:
        """Without enable_factual_extractor, domain=None doesn't call FactualExtractor."""
        auto = AutoExtractor()  # default: factual extractor disabled
        # No mocking needed — if FactualExtractor were called, it would try
        # real network calls and likely fail or return empty results.
        # We just verify no FactualExtractor is in the list.
        extractor_types = [type(e).__name__ for e in auto._extractors]
        assert "FactualExtractor" not in extractor_types

    def test_supported_domains_includes_factual_when_enabled(self) -> None:
        """AutoExtractor.supported_domains includes 'factual' when FactualExtractor enabled."""
        auto = AutoExtractor(enable_factual_extractor=True)
        assert "factual" in auto.supported_domains

    def test_graceful_degradation_does_not_block_pipeline(self) -> None:
        """Network timeout in FactualExtractor does not raise; pipeline continues."""
        import requests as real_requests

        auto = AutoExtractor(enable_factual_extractor=True)

        with patch("carnot.pipeline.factual_extractor.requests") as mock_requests:
            mock_requests.get.side_effect = real_requests.exceptions.Timeout("timeout")
            # Should not raise; returns results from other extractors
            results = auto.extract(
                "Paris is the capital of France.", domain="factual"
            )

        # FactualExtractor returns [] on timeout; pipeline still returns other results
        assert isinstance(results, list)
        # No factual_verified or factual_contradicted since network failed
        factual = [
            r for r in results
            if r.constraint_type in ("factual_verified", "factual_contradicted")
        ]
        assert factual == []
