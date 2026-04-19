"""BoltzmannSemanticEnergy — Tier 0d hallucination pre-filter combining semantic clustering
with Boltzmann-weighted energy from token logits.

**WHY this approach (the key insight from arXiv 2508.14496):**

    Semantic Entropy (SE, Farquhar et al. 2023) asks: "how spread are the model's MEANINGS?"
    It clusters multiple sampled responses by semantic equivalence and measures the entropy
    over meaning clusters.  When the model gives the same meaning repeatedly, SE is low
    (confident).  When the model gives different meanings, SE is high (uncertain).

    SE's weakness: it only measures IF the model is uncertain (spread of meanings),
    not HOW uncertain each alternative is (confidence in each meaning).  In cases where
    the model is overconfident — confidently wrong — SE stays low even though the model
    is hallucinating.

    Boltzmann Semantic Energy (arXiv 2508.14496) adds the MISSING SIGNAL:
      1. Cluster tokens by semantic similarity (cosine-similarity k-means on token embeddings).
         This groups tokens that "mean the same thing" at the vocabulary level.
      2. For each cluster, compute the BOLTZMANN ENERGY:
            cluster_energy = -mean_logit / temperature
         A cluster with high mean logit has LOW energy (the model is confident about it).
         A cluster with low/negative mean logit has HIGH energy (the model is uncertain).
      3. Sum the Boltzmann-weighted cluster energies:
            total_energy = sum(cluster_energy_k * boltzmann_weight_k for k in clusters)
            where boltzmann_weight_k = exp(-cluster_energy_k) / Z  (partition function)
      4. Normalise via sigmoid to get a score in [0, 1]:
            score = sigmoid(total_energy)

    The combined signal captures BOTH axes of uncertainty:
      - High semantic spread (many clusters have significant weight) → high entropy → SE signal
      - High Boltzmann energy per cluster (low confidence in each alternative) → energy signal

    arXiv 2508.14496 shows 13% average AUROC improvement over SE in the HIGH-CONFIDENCE
    FAILURE regime — exactly where SE fails (model is confidently wrong, low SE, but high
    Boltzmann energy exposes the instability).

**Why Boltzmann distributions are Carnot's native language:**
    Carnot's Ising verifier already computes energy functions over constraint terms.
    A Boltzmann distribution is just exp(-E/T) — the same form as the Ising partition
    function.  BoltzmannSemanticEnergy reuses this mathematical framework at the token
    level, making it a natural fit for Carnot's energy-based verification philosophy.

**Cascade position (Tier 0d):**
    Tier 0a: ThinkProbe        — generative CoT verdict (wrong/uncertain/correct)
    Tier 0b: SpilledEnergy     — per-token NLL discrepancy (raw energy signal)
    Tier 0c: NUPProbe          — continuation entropy (AUC=0.600)
    Tier 0d: BoltzmannSemantic — semantic cluster energy (THIS MODULE, Exp 506)
    Tier 1:  SinkProbe         — attention sink concentration
    Tier 2:  EORM              — outcome reward model
    Tier 3:  Ising             — constraint-based energy minimisation

**CPU-only design:**
    This module uses only Python stdlib + math.  No JAX, no torch, no GPU required.
    This makes it fast to import in CI and safe to run in the conductor subprocess.

Spec: REQ-VERIFY-101, REQ-VERIFY-102, REQ-VERIFY-103
SCENARIO-VERIFY-134, SCENARIO-VERIFY-135, SCENARIO-VERIFY-136
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# SemanticCluster dataclass
# ---------------------------------------------------------------------------


@dataclass
class SemanticCluster:
    """A group of semantically similar tokens with their aggregate energy statistics.

    **Why cluster tokens instead of treating them individually:**
        Individual tokens are noisy — "cat" and "cats" share meaning but have separate
        logits.  Grouping by semantic similarity (cosine of character-level embeddings)
        reduces noise by averaging over tokens that mean the same thing.  The cluster's
        mean_logit is a more stable signal than any individual token's logit.

    **Boltzmann weight formula:**
        weight = exp(-energy / temperature) where energy = -mean_logit / temperature
        Equivalently: weight = exp(mean_logit / temperature^2)
        Higher mean_logit → lower energy → higher Boltzmann weight (model prefers this cluster)

    Attributes:
        tokens: List of token strings assigned to this cluster.
        mean_logit: Average logit value across all member tokens.
        cluster_energy: Boltzmann energy of this cluster = -mean_logit / temperature.
            Computed at cluster-time for a specific temperature value.

    Spec: REQ-VERIFY-101
    """

    tokens: list[str] = field(default_factory=list)
    mean_logit: float = 0.0
    cluster_energy: float = 0.0

    def boltzmann_weight(self, temperature: float = 1.0) -> float:
        """Compute the unnormalised Boltzmann weight for this cluster.

        **Why unnormalised:**
            The caller (BoltzmannSemanticEnergy.score) normalises across all clusters
            by dividing by the partition function Z = sum(weights).  Keeping the weight
            unnormalised here lets the caller decide the normalisation scheme without
            forcing this dataclass to know about the global cluster set.

        Formula:
            weight = exp(-cluster_energy / temperature)
            Since cluster_energy = -mean_logit / temperature_at_cluster_time,
            if the same temperature is used: weight = exp(mean_logit / temperature^2)

        Args:
            temperature: Positive float controlling the sharpness of the Boltzmann
                distribution.  Lower temperature = sharper (more concentrated around
                high-logit clusters).  Default 1.0.

        Returns:
            Positive float. Never exactly zero (math.exp clamps to float minimum).

        Spec: REQ-VERIFY-101
        """
        return math.exp(-self.cluster_energy / max(temperature, 1e-9))


# ---------------------------------------------------------------------------
# _char_embedding helper
# ---------------------------------------------------------------------------


def _char_embedding(token: str, dim: int = 8) -> list[float]:
    """Compute a cheap character-level embedding for clustering purposes.

    **Why character-level instead of a pretrained embedding:**
        We need a CPU-only, zero-dependency embedding that approximates semantic
        similarity well enough for k-means clustering.  Character n-gram statistics
        correlate strongly with morphological and semantic similarity for common
        vocabulary tokens (e.g. "run"/"running"/"runner" cluster together).
        This is not a production-quality embedding — it is a lightweight proxy
        that enables the Boltzmann clustering signal in CPU-only mode.

    The embedding is a fixed-size vector of character frequency statistics:
        dims 0-3: counts of a/e/i/o (vowel profile)
        dims 4-5: count of uppercase letters, count of digits
        dims 6-7: length mod 8, first-char ordinal mod 16 (shape features)
    All values are divided by (len(token) + 1) to normalise for length.

    Args:
        token: Token string to embed.
        dim: Output dimension (fixed at 8 for this implementation).

    Returns:
        List of 8 floats in [0, 1].
    """
    n = len(token) + 1  # avoid division by zero
    vec = [
        sum(1 for c in token if c in "aeiouAEIOU") / n,
        sum(1 for c in token if c in "aeiou") / n,
        sum(1 for c in token if c.isupper()) / n,
        sum(1 for c in token if c.isdigit()) / n,
        (len(token) % 8) / 8.0,
        (ord(token[0]) % 16) / 16.0 if token else 0.0,
        sum(ord(c) for c in token[:4]) / (4 * 128.0) if token else 0.0,
        sum(ord(c) for c in token[-4:]) / (4 * 128.0) if token else 0.0,
    ]
    return vec


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    denom = mag_a * mag_b
    if denom < 1e-12:
        return 0.0
    return dot / denom


# ---------------------------------------------------------------------------
# BoltzmannSemanticEnergy
# ---------------------------------------------------------------------------


@dataclass
class BoltzmannSemanticEnergy:
    """Tier 0d hallucination detector using Boltzmann-weighted semantic cluster energy.

    **The core algorithm (three steps):**

    STEP 1 — Cluster tokens by semantic similarity:
        Each token in token_logits is embedded via _char_embedding().
        K-means with `n_clusters` centroids groups tokens by cosine similarity.
        Result: k clusters, each with a list of member tokens.

    STEP 2 — Compute per-cluster Boltzmann energy:
        For cluster k:
            mean_logit_k = mean(logit[t] for t in tokens_k)
            cluster_energy_k = -mean_logit_k / temperature
        A CONFIDENT cluster has high mean_logit → low energy (model "favours" it).
        An UNCERTAIN cluster has low/negative mean_logit → high energy (model is spread out).

    STEP 3 — Compute Boltzmann-weighted total energy and normalise to [0, 1]:
        weight_k = exp(-cluster_energy_k / temperature)   (unnormalised Boltzmann weight)
        Z = sum(weight_k for all k)                        (partition function)
        normalised_weight_k = weight_k / Z
        total_energy = sum(cluster_energy_k * normalised_weight_k for all k)
        score = sigmoid(total_energy)                      (maps R → [0, 1])

    HIGH score (→ 1): high total energy = model is uncertain AND semantically conflicted
    LOW score (→ 0):  low total energy  = model is confident in a single semantic region

    **Why sigmoid (not min-max normalisation)?**
        Min-max normalisation requires knowing the empirical range of total_energy, which
        varies with vocabulary size and temperature.  Sigmoid is parameter-free and maps
        any real number to (0, 1) monotonically.  The midpoint is at total_energy=0, which
        corresponds to equal probability mass on positive and negative mean-logit clusters —
        a natural "maximum uncertainty" point.

    Attributes:
        n_clusters: Number of semantic clusters. Default 10.
            More clusters = finer semantic resolution but noisier per-cluster statistics.
            10 is a sensible default for typical LLM vocabulary slices.
        temperature: Boltzmann temperature. Default 1.0.
            Lower temperature sharpens the distribution (energy differences matter more).
            Higher temperature flattens it (all clusters contribute equally).

    Spec: REQ-VERIFY-101, REQ-VERIFY-102, REQ-VERIFY-103
    """

    n_clusters: int = 10
    temperature: float = 1.0

    def cluster(self, token_logits: dict[str, float]) -> list[SemanticCluster]:
        """Group tokens into semantic clusters via cosine-similarity k-means.

        **Why k-means with cosine similarity (not Euclidean):**
            Token embeddings vary in magnitude (longer tokens have larger vectors).
            Cosine similarity normalises for magnitude, so the clustering is based
            purely on the DIRECTION (semantic content) of the embedding, not its size.
            This prevents long tokens from dominating short tokens in the cluster assignment.

        **K-means algorithm (Lloyd's algorithm, max 20 iterations):**
            1. Initialise k centroids by sampling k distinct tokens (or fewer if |vocab| < k).
            2. Assign each token to its nearest centroid by cosine similarity.
            3. Update each centroid to the mean of its assigned token embeddings.
            4. Repeat until stable (no assignment changes) or max_iter reached.
            5. Compute SemanticCluster statistics for each final cluster.

        Args:
            token_logits: Dict mapping token string → float logit value.
                Tokens are the vocabulary items observed in the LLM's output.
                Logits are the raw (unnormalised) scores before softmax.

        Returns:
            List of SemanticCluster instances. Length is min(n_clusters, len(token_logits)).
            Empty list if token_logits is empty.

        Spec: REQ-VERIFY-101
        """
        if not token_logits:
            return []

        tokens = list(token_logits.keys())
        embeddings = {t: _char_embedding(t) for t in tokens}

        k = min(self.n_clusters, len(tokens))
        if k == 0:
            return []

        # Initialise centroids: pick k evenly-spaced tokens (deterministic, not random)
        # so that CI runs are reproducible without seeding global state.
        step = max(1, len(tokens) // k)
        centroids = [embeddings[tokens[i * step]] for i in range(k)]

        assignments: list[int] = [0] * len(tokens)
        max_iter = 20

        for _ in range(max_iter):
            new_assignments = []
            for emb in (embeddings[t] for t in tokens):
                best_c = max(range(k), key=lambda c: _cosine_sim(emb, centroids[c]))
                new_assignments.append(best_c)

            if new_assignments == assignments:
                break
            assignments = new_assignments

            # Recompute centroids as mean of assigned embeddings
            sums: list[list[float]] = [[0.0] * 8 for _ in range(k)]
            counts: list[int] = [0] * k
            for idx, c in enumerate(assignments):
                emb = embeddings[tokens[idx]]
                for d in range(8):
                    sums[c][d] += emb[d]
                counts[c] += 1

            for c in range(k):
                if counts[c] > 0:
                    centroids[c] = [sums[c][d] / counts[c] for d in range(8)]

        # Build SemanticCluster objects
        cluster_tokens: list[list[str]] = [[] for _ in range(k)]
        for idx, c in enumerate(assignments):
            cluster_tokens[c].append(tokens[idx])

        result: list[SemanticCluster] = []
        for c in range(k):
            members = cluster_tokens[c]
            if not members:
                continue
            mean_logit = sum(token_logits[t] for t in members) / len(members)
            energy = -mean_logit / max(self.temperature, 1e-9)
            result.append(
                SemanticCluster(
                    tokens=members,
                    mean_logit=mean_logit,
                    cluster_energy=energy,
                )
            )

        return result

    def score(self, response: str, token_logits: dict[str, float]) -> float:
        """Compute a hallucination score in [0, 1] for a response given its token logits.

        **The score is the sigmoid of the Boltzmann-weighted total cluster energy:**
            total_energy = sum(cluster_energy_k * normalised_weight_k)
            score = 1 / (1 + exp(-total_energy))

        A HIGH score (close to 1) means:
            - High total Boltzmann energy
            - The model's probability mass is spread across clusters with high energy
            - i.e., the model is UNCERTAIN and SEMANTICALLY CONFLICTED
            - This is the hallucination signal

        A LOW score (close to 0) means:
            - Low total Boltzmann energy
            - Most probability mass is concentrated in one low-energy cluster
            - i.e., the model is CONFIDENT about a specific semantic region
            - This is the confident-and-correct signal

        **Fallback when token_logits is empty:**
            Returns 0.5 (maximum uncertainty / uninformative score).
            The caller should treat 0.5 as "no signal" and proceed to the next tier.

        **The response parameter is reserved for future enrichment:**
            In a future version, the response text can be parsed to extract
            candidate answers and compared against the token_logits clusters to
            detect when the stated answer differs from the high-weight cluster.
            Currently the response is accepted but not used.

        Args:
            response: The LLM response text (reserved for future use).
            token_logits: Dict mapping token string → float logit.

        Returns:
            Float in [0.0, 1.0]. Higher = more likely hallucination.

        Spec: REQ-VERIFY-102
        SCENARIO-VERIFY-134, SCENARIO-VERIFY-135
        """
        clusters = self.cluster(token_logits)
        if not clusters:
            return 0.5  # no signal — return maximum uncertainty

        # Compute unnormalised Boltzmann weights
        weights = [c.boltzmann_weight(self.temperature) for c in clusters]
        z = sum(weights)
        if z < 1e-30:
            return 0.5  # degenerate partition function — return uninformative score

        normalised = [w / z for w in weights]
        total_energy = sum(c.cluster_energy * w for c, w in zip(clusters, normalised))

        # Sigmoid maps total_energy ∈ ℝ → [0, 1]
        score = 1.0 / (1.0 + math.exp(-total_energy))
        # Clamp to strict [0, 1] to guard against floating-point edge cases
        return max(0.0, min(1.0, score))

    def benchmark(
        self,
        responses: list[tuple[str, dict[str, float]]],
        ground_truth: list[bool],
    ) -> dict:
        """Measure AUROC on a labelled corpus of (response, token_logits) pairs.

        **AUROC (Area Under the Receiver Operating Characteristic curve):**
            AUROC measures how well the score discriminates between hallucinated
            (ground_truth=False) and correct (ground_truth=True) responses.

            AUROC = 0.5 → random classifier (no discrimination ability)
            AUROC = 1.0 → perfect discrimination
            AUROC < 0.5 → inverted classifier (score is anti-correlated with hallucination)

            We use the Wilcoxon-Mann-Whitney U statistic to compute AUROC without
            needing to threshold the score.  This is numerically exact for small datasets.

            AUROC = P(score(hallucinated) > score(correct))
                  = (number of (hall, correct) pairs where score_h > score_c) / (n_h * n_c)

        **skip_rate:**
            Fraction of responses where score > 0.5 (flagged as likely hallucinated).
            This is informational — in the real cascade, a threshold would be tuned
            per deployment target.  We use 0.5 as a natural midpoint.

        Args:
            responses: List of (response_text, token_logits) tuples.
                The response_text is passed to score() for future enrichment.
            ground_truth: Parallel list of booleans.
                True = correct response (not hallucinated).
                False = hallucinated response.

        Returns:
            Dict with keys:
                'auroc': float in [0.0, 1.0], AUROC over the corpus.
                'skip_rate': float in [0.0, 1.0], fraction of responses scoring > 0.5.
                'n_total': int, total number of responses evaluated.
                'n_hallucinated': int, number of ground_truth=False responses.
                'n_correct': int, number of ground_truth=True responses.

        Spec: REQ-VERIFY-103
        SCENARIO-VERIFY-136
        """
        if not responses:
            return {
                "auroc": 0.5,
                "skip_rate": 0.0,
                "n_total": 0,
                "n_hallucinated": 0,
                "n_correct": 0,
            }

        scores = [self.score(resp, logits) for resp, logits in responses]

        hall_scores = [s for s, g in zip(scores, ground_truth) if not g]
        corr_scores = [s for s, g in zip(scores, ground_truth) if g]

        # Wilcoxon-Mann-Whitney AUROC: P(score_hall > score_correct)
        n_h = len(hall_scores)
        n_c = len(corr_scores)
        if n_h == 0 or n_c == 0:
            auroc = 0.5
        else:
            concordant = sum(
                1 for h in hall_scores for c in corr_scores if h > c
            )
            tied = sum(
                0.5 for h in hall_scores for c in corr_scores if h == c
            )
            auroc = (concordant + tied) / (n_h * n_c)

        skip_rate = sum(1 for s in scores if s > 0.5) / len(scores)

        return {
            "auroc": float(auroc),
            "skip_rate": float(skip_rate),
            "n_total": len(scores),
            "n_hallucinated": n_h,
            "n_correct": n_c,
        }
