"""Tests for the leak-free factual grounding verifier.

Spec: REQ-VERIFY-3642, SCENARIO-VERIFY-3642.
"""

from carnot.verify.retrieval_nli_grounding_verifier import RetrievalNLIGroundingVerifier


def test_split_into_claims():
    verifier = RetrievalNLIGroundingVerifier()
    assert verifier.split_into_claims("H1. H2!") == ["H1", "H2"]
    assert verifier.split_into_claims("No punctuation") == ["No punctuation"]
    assert verifier.split_into_claims("") == [""]


def test_compute_entailment_proxy():
    verifier = RetrievalNLIGroundingVerifier()
    assert (
        verifier.compute_entailment_proxy(
            "Arthur's Magazine",
            "Arthur's Magazine was an American literary periodical.",
        )
        == 0.0
    )
    assert (
        verifier.compute_entailment_proxy(
            "Mumbai",
            "The hotel company has its head office in Delhi.",
        )
        > 0.5
    )
    assert verifier.compute_entailment_proxy("H1", "H1 is supported evidence") < (
        verifier.compute_entailment_proxy("H1", "R1 is different evidence")
    )


def test_verify():
    verifier = RetrievalNLIGroundingVerifier()
    assert (
        verifier.verify(
            "Delhi. Mumbai.",
            "The Oberoi Group has its head office in Delhi.",
        )
        == 0.5
    )
    assert (
        verifier.verify(
            "President Richard Nixon.",
            "Matt Groening named the character after President Richard Nixon.",
        )
        == 0.0
    )
    assert verifier.verify("", "context") == 0.0
    # test punctuation without valid claims if any
    assert verifier.verify(".", "context") == 0.0
