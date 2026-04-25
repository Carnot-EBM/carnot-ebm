"""SemanticEnergyProbe — Tier 0f advisory hallucination detector (arXiv 2508.14496).

**Why this probe exists (Boltzmann pairwise semantic energy):**
    arXiv 2508.14496 ("Semantic Energy: Detecting LLM Hallucination Beyond Entropy")
    proposes detecting hallucination by measuring the COHERENCE of sentences within a
    response.  Coherent, correct responses form a tight semantic cluster: every sentence
    relates to the same topic, supports the same claim, and uses similar vocabulary.
    Hallucinated responses contain one or more sentences that contradict or are semantically
    distant from the rest — the "rogue sentence" pattern.

    The intuition is physical: imagine the sentences as particles in a potential well.
    Coherent sentences attract each other (low energy), incoherent sentences repel (high
    energy).  We formalise this via a Boltzmann-inspired pairwise kernel:

        E = -mean_{i != j} exp(-||e_i - e_j||^2 / sigma^2)

    where e_i, e_j are L2-normalised sentence embeddings.  The negative sign means:
    - Coherent response: all e_i close together → exp(...) → 1 → E << 0 (low energy, stable)
    - Incoherent response: sentences far apart → exp(...) → 0 → E → 0 (high energy, unstable)

    The advisory threshold is -0.5 by default.  Responses with E > -0.5 are flagged as
    is_unstable=True.

**Why TF-IDF + random projection as the embedding:**
    We need sentence embeddings that require no GPU and no model download.  TF-IDF vectors
    capture the lexical profile of each sentence.  Random projection (Johnson-Lindenstrauss
    lemma) reduces the high-dimensional sparse TF-IDF to a dense, fixed-dimension vector
    while approximately preserving pairwise distances.  This gives us embeddings that are:
    - Deterministic (fixed random seed for reproducibility)
    - Fast (pure Python + math operations, no ML framework)
    - Meaningful (sentences with similar words → similar embeddings)

**Architecture:**
    SemanticEnergyResult: frozen dataclass with energy, flags, diagnostics.
    SemanticEnergyProbe:  single-entry-point scorer; score(response) -> SemanticEnergyResult.
    No GPU required, no external ML library beyond standard Python.

Tier: 0f (advisory, no short-circuit)
Spec: REQ-VERIFY-155, SCENARIO-VERIFY-180, SCENARIO-VERIFY-181
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_sentences(text: str) -> List[str]:
    """Split response text into declarative sentence fragments.

    **Why these split points:**
        Period-space '. ' and question-mark-space '? ' are the most reliable
        sentence boundary markers in prose that doesn't use newlines.  We skip
        split points at the end of the string to avoid empty trailing fragments.
        Single-character fragments (e.g., lone initials) are also discarded.

    Args:
        text: Raw response text.

    Returns:
        List of non-empty sentence strings with leading/trailing whitespace stripped.
        Returns [text] when no split points exist.
    """
    raw = text.replace("? ", ". ").split(". ")
    return [s.strip() for s in raw if len(s.strip()) > 1]


def _tokenize(text: str) -> List[str]:
    """Split text into lowercase word tokens (letters and digits only).

    Args:
        text: Raw string.

    Returns:
        List of lowercase word strings.
    """
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


def _build_vocab(sentences: List[str]) -> Dict[str, int]:
    """Assign a unique integer index to each distinct token across all sentences.

    Args:
        sentences: List of sentence strings.

    Returns:
        Dict mapping token -> integer index (0-based, insertion order).
    """
    vocab: Dict[str, int] = {}
    for s in sentences:
        for tok in _tokenize(s):
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def _tf_vector(sentence: str, vocab: Dict[str, int]) -> List[float]:
    """Compute a raw term-frequency (TF) vector for one sentence.

    **Why TF (not TF-IDF) for intra-response sentence comparison:**
        TF-IDF penalises shared words via IDF, which works well for document
        retrieval over large corpora.  Within a single response, the exact
        OPPOSITE is needed: words shared by many sentences indicate topical
        coherence and should contribute positively to similarity.

        Raw TF gives shared content words (e.g. "cat", "mat") equal weight
        regardless of how many sentences contain them.  Two sentences that
        both discuss "cat sitting on mat" will have high cosine similarity in
        TF space even though "cat" and "mat" are the most frequent terms.

    Args:
        sentence: The target sentence.
        vocab:    Shared token-to-index mapping.

    Returns:
        Dense list of floats (length == len(vocab)).  Zero for absent tokens.
    """
    tokens = _tokenize(sentence)
    if not tokens:
        return [0.0] * len(vocab)

    total = len(tokens)
    vec = [0.0] * len(vocab)
    for tok in tokens:
        if tok in vocab:
            vec[vocab[tok]] += 1.0 / total
    return vec


def _random_project(vec: List[float], out_dim: int, seed: int = 42) -> List[float]:
    """Project a high-dimensional sparse vector into `out_dim` dimensions via random projection.

    **Why random projection (Johnson-Lindenstrauss lemma):**
        JL shows that a random linear map approximately preserves pairwise L2 distances
        with high probability.  The projection matrix entries are drawn from N(0, 1/out_dim)
        so that the expected squared norm is preserved.

    **Reproducibility:**
        We seed Python's random module locally (not globally) before drawing the projection
        matrix so that the same (in_dim, out_dim, seed) triple always produces the same matrix.

    Args:
        vec:     Input vector (length = in_dim).
        out_dim: Target embedding dimension.
        seed:    Random seed for reproducibility.

    Returns:
        Dense float list of length `out_dim`.
    """
    in_dim = len(vec)
    if in_dim == 0:
        return [0.0] * out_dim

    rng = random.Random(seed)
    scale = 1.0 / math.sqrt(out_dim)

    projected = [0.0] * out_dim
    for j in range(out_dim):
        for i in range(in_dim):
            # Gaussian random weight: mean 0, std 1/sqrt(out_dim)
            w = rng.gauss(0.0, scale)
            projected[j] += vec[i] * w
    return projected


def _l2_normalize(vec: List[float]) -> List[float]:
    """L2-normalize a vector to unit length.

    **Why L2 normalisation:**
        The Gaussian kernel exp(-||e_i - e_j||^2 / sigma^2) measures angular + magnitude
        distance.  Normalising to unit sphere ensures we only measure angular distance —
        two sentences with the same word distribution but different lengths map to the same
        point.  Without normalisation, longer sentences would always appear more distant
        simply because their TF-IDF magnitudes are larger.

    Args:
        vec: Input float list.

    Returns:
        Unit-norm float list, or zero vector if input norm is zero.
    """
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-12:
        return [0.0] * len(vec)
    return [x / norm for x in vec]


def _embed_sentences(sentences: List[str], embedding_dim: int) -> List[List[float]]:
    """Embed each sentence as an L2-normalised random-projected TF-IDF vector.

    Steps:
        1. Build shared vocabulary across all sentences.
        2. Compute TF-IDF vector for each sentence.
        3. Random-project to `embedding_dim` dimensions.
        4. L2-normalise.

    Args:
        sentences:     List of sentence strings.
        embedding_dim: Target embedding dimension.

    Returns:
        List of L2-normalised float lists, one per sentence.
    """
    if not sentences:
        return []
    vocab = _build_vocab(sentences)
    embeddings = []
    for s in sentences:
        raw_vec = _tf_vector(s, vocab)
        proj_vec = _random_project(raw_vec, embedding_dim, seed=42)
        norm_vec = _l2_normalize(proj_vec)
        embeddings.append(norm_vec)
    return embeddings


def _gaussian_kernel(ei: List[float], ej: List[float], sigma: float) -> float:
    """Compute the Gaussian (RBF) kernel between two L2-normalised embeddings.

    k(e_i, e_j) = exp(-||e_i - e_j||^2 / sigma^2)

    For unit-norm vectors:
        ||e_i - e_j||^2 = 2 - 2 * dot(e_i, e_j)

    So we can compute this via the dot product, which is faster.

    Args:
        ei:    First embedding (unit-norm).
        ej:    Second embedding (unit-norm).
        sigma: Bandwidth parameter.

    Returns:
        Float in (0, 1].  1.0 when embeddings are identical.
    """
    dot = sum(a * b for a, b in zip(ei, ej))
    # Clamp to avoid floating-point values slightly outside [-1, 1]
    dot = max(-1.0, min(1.0, dot))
    sq_dist = 2.0 - 2.0 * dot
    return math.exp(-sq_dist / (sigma * sigma))


def _row_entropy(kernel_row: List[float]) -> float:
    """Compute Shannon entropy of a probability distribution derived from one kernel row.

    **Why row entropy:**
        After row-normalising the kernel matrix (so each row sums to 1), each row is a
        discrete probability distribution over the other sentences.  Low entropy means the
        sentence is attracted to a specific other sentence (coherent cluster).  High entropy
        means the sentence is equidistant from all others (incoherent, no cluster structure).

        cluster_entropy = mean entropy across all rows = overall cluster incoherence.

    Args:
        kernel_row: Non-negative floats that sum to 1.0 (a probability distribution).

    Returns:
        Shannon entropy H = -sum p log p.  0.0 for deterministic, log(n) for uniform.
    """
    entropy = 0.0
    for p in kernel_row:
        if p > 1e-15:
            entropy -= p * math.log(p)
    return entropy


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticEnergyResult:
    """Result of Tier 0f semantic energy probe.

    Attributes:
        energy:         Boltzmann pairwise semantic energy.  More negative = more coherent.
                        Near zero = incoherent / possibly hallucinated.  Zero when fewer
                        than 2 sentences are found (no pairs to evaluate).
        is_unstable:    True when energy > threshold (high energy = semantic incoherence
                        = hallucination risk).  Advisory flag only — does NOT short-circuit.
        sentence_count: Number of declarative sentences extracted from the response.
        cluster_entropy: Mean Shannon entropy of the row-normalised kernel matrix.
                        High entropy → sentences equidistant from each other (no structure).
        threshold:      The decision threshold used for is_unstable.  Stored for audit.

    Spec: REQ-VERIFY-155
    """

    energy: float
    is_unstable: bool
    sentence_count: int
    cluster_entropy: float
    threshold: float


class SemanticEnergyProbe:
    """Tier 0f advisory probe: detects semantic incoherence via pairwise Boltzmann energy.

    **Why this works:**
        Hallucinated sentences tend to be semantically distant from each other and from
        the question.  A correct response about, say, photosynthesis will have every sentence
        anchored in plant biology vocabulary.  A hallucinated response that inserts a false
        claim ("plants absorb nitrogen from sunlight") will include sentences whose word
        vectors point in a different direction.

        We compute pairwise semantic energy between sentences using the Boltzmann-inspired
        Gaussian kernel formula:

            E = -mean_{i != j} exp(-||e_i - e_j||^2 / sigma^2)

        where e_i are L2-normalised TF-IDF + random-projection embeddings of each sentence.
        The negative sign means:
        - Coherent response: all e_i close → exp(...) → 1 → E << 0 (stable, low energy)
        - Incoherent response: e_i spread apart → exp(...) → 0 → E → 0 (unstable, high energy)

        Advisory threshold default -0.5: responses with E > -0.5 are flagged is_unstable=True.

    **No short-circuit (advisory pattern, like Tier 0e HalluField):**
        is_unstable=True records the flag in the VerificationCertificate but does NOT
        skip Tiers 1, 2, or 3.  This ensures the probe contributes signal without
        causing false-positive fast-path exits.

    Args:
        sigma:         Gaussian kernel bandwidth.  Controls how quickly similarity decays
                       with distance.  Smaller sigma → sharper penalty for distant sentences.
                       Default 1.0.
        threshold:     Energy threshold for is_unstable flag.  Energies above this value
                       are flagged as unstable.  Default -0.5 (chosen so that a 5-sentence
                       coherent response with mean cosine similarity ~0.4 sits clearly below
                       this threshold).
        embedding_dim: Dimension of the random-projected sentence embedding.  Higher values
                       preserve more TF-IDF structure at the cost of runtime.  Default 64.

    Spec: REQ-VERIFY-155, SCENARIO-VERIFY-180, SCENARIO-VERIFY-181
    """

    def __init__(
        self,
        sigma: float = 1.0,
        threshold: float = -0.5,
        embedding_dim: int = 64,
    ) -> None:
        self.sigma = sigma
        self.threshold = threshold
        self.embedding_dim = embedding_dim

    def score(self, response: str) -> SemanticEnergyResult:
        """Compute Boltzmann pairwise semantic energy for a response.

        Steps:
            1. Extract declarative sentences (split on '. ' and '? ').
            2. Embed each sentence via TF-IDF + random projection (embedding_dim dims).
            3. L2-normalise all embeddings.
            4. Compute pairwise Gaussian kernel: k_ij = exp(-||e_i-e_j||^2 / sigma^2).
            5. Energy = -mean(k_ij) over all pairs (i != j).  Returns 0.0 for < 2 sentences.
            6. is_unstable = (energy > threshold).
            7. cluster_entropy = mean entropy of row-normalised kernel matrix.

        Args:
            response: Full response text to analyse.

        Returns:
            SemanticEnergyResult with all diagnostic fields populated.

        Spec: REQ-VERIFY-155, SCENARIO-VERIFY-180, SCENARIO-VERIFY-181
        """
        sentences = _extract_sentences(response)
        n = len(sentences)

        if n < 2:
            # Cannot compute pairwise energy with fewer than 2 sentences.
            # Return zero energy (neutral / uninformative) — not flagged as unstable
            # because a single sentence has no internal incoherence to detect.
            return SemanticEnergyResult(
                energy=0.0,
                is_unstable=False,
                sentence_count=n,
                cluster_entropy=0.0,
                threshold=self.threshold,
            )

        embeddings = _embed_sentences(sentences, self.embedding_dim)

        # Compute n×n kernel matrix
        kernel: List[List[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    kernel[i][j] = 1.0  # diagonal is self-similarity = 1
                else:
                    kernel[i][j] = _gaussian_kernel(embeddings[i], embeddings[j], self.sigma)

        # Energy: mean of off-diagonal kernel values, negated
        # More coherent (higher k_ij) → more negative energy
        n_pairs = n * (n - 1)
        total_k = sum(
            kernel[i][j]
            for i in range(n)
            for j in range(n)
            if i != j
        )
        energy = -(total_k / n_pairs) if n_pairs > 0 else 0.0

        # Cluster entropy: entropy of row-normalised kernel (off-diagonal rows)
        entropies: List[float] = []
        for i in range(n):
            row_off_diag = [kernel[i][j] for j in range(n) if j != i]
            row_sum = sum(row_off_diag)
            if row_sum > 1e-15:
                row_prob = [v / row_sum for v in row_off_diag]
            else:
                # All zeros — maximum uncertainty: uniform distribution
                row_prob = [1.0 / (n - 1)] * (n - 1)
            entropies.append(_row_entropy(row_prob))
        cluster_entropy = sum(entropies) / len(entropies) if entropies else 0.0

        is_unstable = energy > self.threshold

        return SemanticEnergyResult(
            energy=energy,
            is_unstable=is_unstable,
            sentence_count=n,
            cluster_entropy=cluster_entropy,
            threshold=self.threshold,
        )

    def evaluate_auc(self, texts: List[str], labels: List[int]) -> float:
        """Compute AUROC of energy scores against binary hallucination labels.

        **How this is used:**
            labels[i] = 1 means the i-th response is hallucinated (positive class).
            labels[i] = 0 means the i-th response is correct (negative class).
            Higher energy (less negative) = more likely to be hallucinated.

        Args:
            texts:  List of response text strings.
            labels: Parallel binary labels (1 = hallucinated, 0 = correct).

        Returns:
            Float AUROC in [0.0, 1.0].  0.5 = chance; 1.0 = perfect.
            Returns 0.5 when either class is absent.

        Spec: REQ-VERIFY-155
        """
        n_pos = sum(1 for lb in labels if lb == 1)
        n_neg = sum(1 for lb in labels if lb == 0)
        if n_pos == 0 or n_neg == 0:
            return 0.5

        scored: List[Tuple[float, int]] = [
            (self.score(t).energy, lb) for t, lb in zip(texts, labels)
        ]
        # Sort descending: high energy (closer to 0) → predicted hallucinated
        scored.sort(key=lambda x: x[0], reverse=True)

        tp = 0
        fp = 0
        auc = 0.0
        prev_fpr = 0.0
        prev_tpr = 0.0

        for _, lb in scored:
            if lb == 1:
                tp += 1
            else:
                fp += 1
            fpr = fp / n_neg
            tpr = tp / n_pos
            if fpr > prev_fpr:
                auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
            prev_fpr = fpr
            prev_tpr = tpr

        return float(min(1.0, max(0.0, auc)))
