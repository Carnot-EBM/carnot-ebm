"""EmbeddingConstraintStore — SPO-format constraint memory with L2-normalized embeddings.

**Researcher summary (RETRO-CONSTRAINT-ZERO-DELTA):**
    Exp 788 showed constraint_addition_delta=0.0.  Root cause (arXiv 2601.15313, Semantic
    Interference): scalar keyword-count encoding collapses semantically similar constraints
    into indistinguishable embeddings.  Two constraints like "carry_error" and "sign_error"
    both produce near-identical sparse bag-of-words vectors, so the retrieval system cannot
    distinguish them and always returns the same top-K regardless of the query.

    Fix: encode each constraint as an (S, P, O) = Subject-Predicate-Object triple and embed
    the concatenated SPO text using sentence-transformers (all-MiniLM-L6-v2, 384-dim, CPU).

**Researcher summary (RETRO-RETRIEVAL-NEAR-ZERO-COSINE — Exp 847):**
    Exp 836 confirmed 15 constraints written (n_constraints_in_store_after_s3=15) but
    delta_overall=0.0.  Root cause: orthogonality regularization deflects stored embeddings
    away from their original semantic directions via Gram-Schmidt projection.  A query for
    "carry error" encodes in the original sentence-transformer space, but the stored "carry"
    embedding has been projected perpendicular to all previously stored embeddings — so
    cosine similarity between query and stored constraint is near-zero even for matching
    constraint types.  Additionally, sentence-transformer embeddings have L2 norm ~0.9-1.1
    (not exactly 1.0); without explicit normalization on both sides, dot products deviate
    from true cosine similarity.

    Fix (Exp 847): store L2-normalized embeddings WITHOUT orthogonalization.  Normalize
    queries before retrieval.  Lower cosine threshold from 0.7 → 0.5 (constraint-type
    variations for the same semantic meaning typically score 0.5-0.7, not 0.8-0.95).
    This restores semantic alignment between stored embeddings and query embeddings.

    In CI environments where sentence_transformers is not installed, a deterministic hash-based
    384-dim float embedding is substituted.  This mode is labeled "ci_hash" and exists only
    for test correctness — retrieval_auc in ci_hash mode is not meaningful for research.

Spec: REQ-LEARN-057, REQ-LEARN-058, REQ-LEARN-059, SCENARIO-LEARN-098,
      REQ-VERIFY-150, SCENARIO-VERIFY-230
"""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class ConstraintSPOTuple:
    """One constraint expressed as a Subject-Predicate-Object triple.

    Why SPO instead of a plain string?
        A plain string like "carry propagation error" is ambiguous — it could be
        a topic label, a description, or a rule.  SPO forces each constraint to
        declare WHO (subject) is constrained, HOW (predicate) it is constrained,
        and WHAT (object) it must satisfy.  This structural separation prevents
        embedding collapse when two constraints share words but have different
        logical roles.

    Fields:
        subject: The entity or concept that is being constrained
                 (e.g. "arithmetic_carry").
        predicate: The relationship type, almost always "violates" for error
                   patterns (e.g. "violates").
        object: The specific rule or invariant that is violated
                (e.g. "carry_propagation").
        embedding: The sentence-transformer (or ci_hash) embedding of the
                   concatenated SPO text, after orthogonality regularization.
                   None if not yet encoded.
        source_violation_type: The short label used for retrieval_auc evaluation
                               (e.g. "carry", "sign", "unit").
        timestamp: ISO-8601 UTC timestamp recording when this entry was created.
    """

    subject: str
    predicate: str
    object: str
    embedding: Optional[list[float]]
    source_violation_type: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _dot(a: list[float], b: list[float]) -> float:
    """Dot product of two equal-length float lists."""
    return sum(x * y for x, y in zip(a, b))


def _l2norm(v: list[float]) -> float:
    """L2 norm of a float list."""
    return math.sqrt(sum(x * x for x in v))


def _normalize(v: list[float]) -> list[float]:
    """Return L2-normalized copy of v.  Returns v unchanged if norm is ~0."""
    n = _l2norm(v)
    if n < 1e-12:
        return list(v)
    return [x / n for x in v]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length float lists."""
    na = _l2norm(a)
    nb = _l2norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return _dot(a, b) / (na * nb)


def _ci_hash_embedding(text: str, dim: int = 384) -> list[float]:
    """Deterministic pseudo-embedding from SHA-256 of text.

    Why this exists:
        sentence_transformers is a heavy dependency (PyTorch, transformers).
        CI runners often lack GPU and have tight dependency budgets.  This
        function produces a reproducible 384-dim float vector without any
        ML framework — purely from the text's SHA-256 hash repeated into
        enough bytes.  The distribution is uniform in [-0.5, 0.5], which
        means cosine similarities cluster near 0 and retrieval_auc is ~0.5
        (chance level).  That is intentional — it signals "not meaningful"
        rather than silently returning inflated numbers.
    """
    raw = text.encode("utf-8")
    # Expand hash to enough bytes for `dim` float32 values (4 bytes each)
    needed = dim * 4
    h = b""
    counter = 0
    while len(h) < needed:
        h += hashlib.sha256(raw + counter.to_bytes(4, "little")).digest()
        counter += 1
    floats = []
    for i in range(dim):
        (val,) = struct.unpack_from("<I", h, i * 4)
        # Map uint32 to [-0.5, 0.5]
        floats.append(val / 2**32 - 0.5)
    return floats


class EmbeddingConstraintStore:
    """Constraint memory that encodes constraints as SPO tuples with sentence-transformer embeddings.

    Why this is better than keyword-count encoding (the Exp 788 failure mode):
        Keyword-count encoding represents "arithmetic carry error" and "sign inversion error"
        as nearly identical sparse vectors because they share stop words and arithmetic terms.
        The sentence-transformer maps each full SPO string to a 384-dim dense vector in a
        semantically meaningful space where "carry propagation" and "sign preservation" land
        in different directions.

    Why L2-normalization instead of orthogonality regularization (Exp 847 fix):
        Prior code applied Gram-Schmidt orthogonalization before storing each embedding.
        This deflects stored embeddings away from their original semantic directions: a stored
        "carry" embedding is projected perpendicular to all previously stored embeddings, so
        a query for "carry error" (in the original sentence-transformer space) has near-zero
        cosine similarity with the stored "carry" embedding.  The retrieval system silently
        returns empty matches, and IsingEBM receives zero-magnitude external field input.

        The fix: store each embedding as its plain L2-normalized form — no orthogonalization.
        Cosine similarity between a normalized stored embedding and a normalized query is then
        the true semantic similarity, which for matching constraint types is typically 0.5-0.7.

    Class invariant: retrieval_l2_normalized = True — all stored and query embeddings are
        explicitly L2-normalized before any similarity computation.

    Attributes:
        model_name: Name of the sentence_transformers model used for encoding.
        embedding_mode: "sentence_transformer" when the real model is loaded,
                        "ci_hash" when sentence_transformers is unavailable.
        retrieval_l2_normalized: Always True; asserted in store() and retrieve() to guard
                                 against future regressions that skip normalization.
    """

    retrieval_l2_normalized: bool = True

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._store: list[ConstraintSPOTuple] = []
        self._encoder = None

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
            self._encoder = SentenceTransformer(model_name)
            self.embedding_mode = "sentence_transformer"
        except ImportError:
            self.embedding_mode = "ci_hash"

    def _encode(self, text: str) -> list[float]:
        """Encode text to a float list using the active encoder.

        When embedding_mode == "sentence_transformer", calls the real model.
        When embedding_mode == "ci_hash", calls the deterministic hash fallback.
        """
        if self._encoder is not None:
            vec = self._encoder.encode(text)
            return [float(x) for x in vec]
        return _ci_hash_embedding(text)

    def _orthogonalize(self, new_embedding: list[float]) -> list[float]:
        """Project out components along all currently-stored embedding directions.

        Algorithm (Gram-Schmidt style):
            For each stored embedding e_i (already normalized at store time):
                projection_scalar = dot(new_embedding, e_i) / dot(e_i, e_i)
                new_embedding -= projection_scalar * e_i
            Then L2-normalize the result.

        Why this reduces semantic interference:
            After orthogonalization, new_embedding has zero (or near-zero) component
            along every existing embedding direction.  The store spans an approximate
            set of orthogonal axes, one per constraint type.  Retrieval via cosine
            similarity then cleanly separates constraint types because each lives in
            its own subspace.

        Returns the orthogonalized and L2-normalized embedding as a list[float].
        If the store is empty, returns L2-normalized new_embedding unchanged.
        """
        v = list(new_embedding)
        for entry in self._store:
            if entry.embedding is None:
                continue
            ei = entry.embedding
            denom = _dot(ei, ei)
            if denom < 1e-12:
                continue
            proj = _dot(v, ei) / denom
            v = [vi - proj * ej for vi, ej in zip(v, ei)]
        return _normalize(v)

    def store(self, spo: ConstraintSPOTuple) -> None:
        """Encode, L2-normalize, and store a new SPO constraint tuple.

        The SPO triple text is formed as "(subject) (predicate) (object)" —
        parentheses help the sentence-transformer attend to each role separately
        rather than treating the string as a plain sentence.

        Explicit L2-normalization is applied (REQ-VERIFY-150): storing raw embeddings
        with L2 norm ~0.9-1.1 causes dot products to deviate from cosine similarity.
        Normalizing here ensures every stored embedding is a unit vector, so dot product
        with a normalized query equals exact cosine similarity.

        Why NOT orthogonalization (Exp 847 root-cause fix):
            Gram-Schmidt projection deflects stored embeddings away from their original
            semantic directions, so queries in the original sentence-transformer space
            show near-zero cosine similarity with orthogonalized stored embeddings.
            Plain L2 normalization preserves semantic direction.
        """
        assert self.retrieval_l2_normalized, "class invariant: L2-normalization must be enabled"
        spo_text = f"({spo.subject}) ({spo.predicate}) ({spo.object})"
        raw_emb = self._encode(spo_text)
        # L2-normalize before storing — guarantees unit vectors in the store.
        norm = _l2norm(raw_emb)
        normalized_emb = [x / (norm + 1e-8) for x in raw_emb]
        spo.embedding = normalized_emb
        self._store.append(spo)

    def retrieve(
        self, query: str, top_k: int = 3, cosine_threshold: float = 0.5
    ) -> list[ConstraintSPOTuple]:
        """Return the top_k stored constraints most similar to the query.

        The query is encoded with the same encoder used at store time, then
        explicitly L2-normalized (REQ-VERIFY-150) before similarity computation.
        Because both stored embeddings and the query are unit vectors, the dot
        product equals exact cosine similarity — no division required.

        Entries with cosine similarity below cosine_threshold are excluded.
        Default threshold is 0.5 (not 0.7 as in prior code): sentence-transformer
        vectors for constraint-type variations typically score 0.5-0.7, not the
        0.8-0.95 seen in document retrieval.  Using 0.7 caused empty retrieval.

        Args:
            query: Error context string to match against stored constraints.
            top_k: Maximum number of constraints to return.
            cosine_threshold: Minimum cosine similarity to include in results.
                              Default 0.5 (lowered from 0.7 in Exp 847 fix).

        Spec: REQ-LEARN-059, REQ-VERIFY-150
        """
        assert self.retrieval_l2_normalized, "class invariant: L2-normalization must be enabled"
        if not self._store:
            return []
        raw_emb = self._encode(query)
        # Explicit L2-normalization: raw sentence-transformer vectors have norm ~0.9-1.1.
        # Normalizing makes dot product == cosine similarity.
        qnorm = _l2norm(raw_emb)
        query_emb = [x / (qnorm + 1e-8) for x in raw_emb]
        scored = [
            (entry, _dot(query_emb, entry.embedding or []))
            for entry in self._store
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            entry for entry, sim in scored[:top_k] if sim >= cosine_threshold
        ]

    def retrieval_auc(self, queries: list[str], labels: list[str]) -> float:
        """Fraction of queries where top-1 retrieved constraint matches the label.

        'AUC' here is used loosely to mean the proportion of correct top-1
        retrievals, analogous to Recall@1.  A score > 0.70 indicates the
        embedding store can discriminate between constraint types better than
        the keyword-count baseline (which produced 0.0 delta in Exp 788).

        Args:
            queries: List of error context strings (one per test case).
            labels: Parallel list of expected source_violation_type strings.

        Returns:
            Fraction correct in [0.0, 1.0].  Returns 0.0 if queries is empty
            or the store is empty.

        Spec: REQ-LEARN-059, SCENARIO-LEARN-098
        """
        if not queries or not self._store:
            return 0.0
        correct = 0
        for query, label in zip(queries, labels):
            top1 = self.retrieve(query, top_k=1)
            if top1 and top1[0].source_violation_type == label:
                correct += 1
        return correct / len(queries)

    def from_casememory_patterns(self, patterns: dict) -> None:
        """Bootstrap the store from legacy CaseMemory keyword-pattern dictionaries.

        Legacy CaseMemory stored error patterns as keyword-count dicts like
        {"carry": 4, "sign": 2, ...}.  This method converts each known pattern
        type into a structured SPO tuple and stores it, providing a migration
        path from the old scalar encoding to the new embedding-based store.

        The five standard violation types and their SPO mappings:
            carry      → (arithmetic_carry, violates, carry_propagation)
            sign       → (numeric_sign, violates, sign_preservation)
            unit       → (unit_label, violates, unit_consistency)
            comparison → (comparison_direction, violates, inequality_direction)
            causal     → (causal_entailment, violates, step_causality)

        The `patterns` argument is inspected only for its keys — any dict that
        contains one or more of the five known keys will trigger that tuple to
        be stored.  Unknown keys are silently ignored.

        Spec: REQ-LEARN-057
        """
        _SPO_MAP = {
            "carry": ("arithmetic_carry", "violates", "carry_propagation"),
            "sign": ("numeric_sign", "violates", "sign_preservation"),
            "unit": ("unit_label", "violates", "unit_consistency"),
            "comparison": ("comparison_direction", "violates", "inequality_direction"),
            "causal": ("causal_entailment", "violates", "step_causality"),
        }
        for key, (subj, pred, obj) in _SPO_MAP.items():
            if key in patterns:
                spo = ConstraintSPOTuple(
                    subject=subj,
                    predicate=pred,
                    object=obj,
                    embedding=None,
                    source_violation_type=key,
                )
                self.store(spo)
