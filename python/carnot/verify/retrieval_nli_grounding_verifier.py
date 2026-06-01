"""Leak-free retrieval NLI grounding verifier.

**What this is and why it exists:**
    Factual hallucination checks should judge whether the model's answer is
    grounded in the evidence passage, not whether the answer resembles a label
    string in the corpus. The intended HalluSearch/VeriScore shape is:

    1. Decompose the model answer into atomic claims.
    2. Retrieve evidence for each claim.
    3. Score whether the evidence entails each claim.
    4. Return energy as the fraction of unentailed claims.

**What we're approximating here:**
    This implementation is a disclosed text-statistical proxy rather than a
    DeBERTa/MiniLM NLI checkpoint. It measures content-token support: a claim is
    treated as grounded when its non-stopword tokens are present in the supplied
    evidence passage. This is weaker than model-based NLI because it cannot
    reason over paraphrase or contradiction, but it is honest about that gap and
    it never reads gold answers or hallucination labels.

Spec: REQ-VERIFY-3642, SCENARIO-VERIFY-3642.
"""

from __future__ import annotations

import re


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "with",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")


class RetrievalNLIGroundingVerifier:
    """Verifier combining atomic claim splitting, evidence retrieval, and NLI entailment.

    The current substrate is a disclosed token-support proxy. It is deliberately
    less capable than a real NLI checkpoint, but it preserves the critical leak
    guard: the score depends only on the model answer and the evidence text.
    """

    def split_into_claims(self, answer: str) -> list[str]:
        """Decompose the answer into atomic claims."""
        claims = [c.strip() for c in re.split(r"[.!?]+", answer) if c.strip()]
        if not claims:
            claims = [answer.strip()]
        return claims

    def retrieve_evidence(self, claim: str, context: str) -> str:
        """Retrieve evidence for a given claim.

        In a full system, this would query a document index using the claim.
        Here, we use the provided context as the retrieved evidence.
        """
        return context

    def compute_entailment_proxy(self, claim: str, evidence: str) -> float:
        """Compute the NLI proxy score.

        0.0 = Entailed (grounded), 1.0 = Contradiction/Neutral (ungrounded).
        The proxy is ``1 - content_token_coverage`` over the evidence passage.
        A real NLI model should replace this when a checkpoint is available.
        """
        claim_tokens = set(_content_tokens(claim))
        if not claim_tokens:
            return 0.0
        evidence_tokens = set(_content_tokens(evidence))
        supported = len(claim_tokens & evidence_tokens)
        return float(1.0 - supported / len(claim_tokens))

    def verify(self, answer: str, context: str) -> float:
        """Compute the overall grounding energy for the answer.

        Energy is the fraction of claims that are not entailed by the evidence.
        """
        claims = self.split_into_claims(answer)
        # Filter out completely empty claims if any
        claims = [c for c in claims if c]
        if not claims:
            return 0.0

        unentailed_count = 0.0
        for claim in claims:
            evidence = self.retrieve_evidence(claim, context)
            entailment_score = self.compute_entailment_proxy(claim, evidence)
            if entailment_score >= 0.5:
                unentailed_count += 1.0

        return float(unentailed_count / len(claims))


def _content_tokens(text: str) -> list[str]:
    """Return normalized content tokens for the disclosed grounding proxy."""

    tokens = []
    for token in _TOKEN_RE.findall(str(text).lower()):
        if token in _STOPWORDS or len(token) <= 1:
            continue
        tokens.append(token)
    return tokens
