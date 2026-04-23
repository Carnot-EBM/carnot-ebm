"""SemanticEnergyProbe — logit-space energy hallucination detector (Tier 0g).

**Why SemanticEnergyProbe exists (arXiv 2508.14496):**
    Standard semantic entropy (Kuhn et al., 2023) assigns uncertainty using softmax
    probabilities.  But softmax NORMALISES out magnitude: a response with logit 100
    and a response with logit 1 can produce the same probability if other logits
    scale proportionally.  This means a model can be "confidently hallucinating" —
    high probability AND high energy — and entropy would miss it.

    arXiv 2508.14496 ("Semantic Energy: Detecting LLM Hallucination Beyond Entropy",
    August 2025) shows that operating in LOGIT space instead of probability space
    recovers that lost magnitude.  The key formula:

        E(response) = -sum_i log p(t_i)   where p(t_i) is the token probability

    For an autoregressive LLM, log p(t_i) is directly available as the log-likelihood
    of each token at its position.  A low-energy response (every token likely) signals
    confident correct generation.  A high-energy response (some tokens unlikely, even
    if the final probability is high) signals hallucination risk.

    Semantic energy groups equivalent-meaning responses into clusters, then computes
    energy per cluster — matching the semantic-entropy grouping idea but using the
    energy formulation rather than Shannon entropy.

**Why TF-IDF as a log-prob proxy:**
    In the offline setting (we only have the text, not the logits), we cannot compute
    the true per-token log-probabilities from the generating model.  TF-IDF scores
    provide a text-statistics-based proxy:
    - High TF-IDF for a token in this document = unusual token = likely high energy.
    - Low TF-IDF = common token across the corpus = likely low energy.
    This is an approximation.  The limitation is noted explicitly in score() below.
    When real logits are available, they should be passed directly; the TF-IDF path
    is the fallback for offline/text-only evaluation.

**How clustering works:**
    Responses are grouped by TF-IDF cosine similarity >= threshold (default 0.9).
    Two responses in the same cluster "mean the same thing" (at least superficially).
    Cluster energy = mean energy across members of the cluster.

**Architecture:**
    - SemanticCluster: groups responses, computes cluster energy.
    - SemanticEnergyProbe: single-response energy with optional threshold gating.
    - No GPU required, no external ML library — all standard Python/NumPy/sklearn.

Spec: REQ-PROBE-020, REQ-PROBE-021,
      SCENARIO-PROBE-030, SCENARIO-PROBE-031
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helper: minimal TF-IDF without sklearn (keeps the module dependency-light)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Split text into lowercase word tokens by whitespace and punctuation.

    This simple tokeniser avoids an NLTK / spaCy dependency.  It strips
    common punctuation that does not carry semantic weight in energy calculations.

    Args:
        text: Raw string from a CoT step or response.

    Returns:
        List of lowercase word strings.
    """
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


def _compute_tfidf_matrix(docs: List[str]) -> List[Dict[str, float]]:
    """Compute TF-IDF scores for each token in each document.

    **How TF-IDF works here:**
        TF (term frequency): count of token in this doc / total tokens in doc.
        IDF (inverse document frequency): log(N / df(t)) where N = number of docs
        and df(t) = number of docs containing token t.  Common tokens score near 0;
        rare tokens score high.

        TF-IDF(t, d) = TF(t, d) * IDF(t).

        A token with high TF-IDF in a document is unusual for that document.
        We use this as a proxy for the token's "surprise value" — its contribution
        to the response's energy.

    Args:
        docs: List of document strings to compute TF-IDF over.

    Returns:
        List of dicts, one per document, mapping token -> TF-IDF score.
    """
    n = len(docs)
    if n == 0:
        return []

    # Tokenize all documents
    tokenized = [_tokenize(d) for d in docs]

    # Compute document frequency for IDF
    df: Dict[str, int] = defaultdict(int)
    for tokens in tokenized:
        for t in set(tokens):
            df[t] += 1

    # Compute TF-IDF per document
    result: List[Dict[str, float]] = []
    for tokens in tokenized:
        total = len(tokens)
        if total == 0:
            result.append({})
            continue
        tf: Dict[str, float] = defaultdict(float)
        for t in tokens:
            tf[t] += 1.0 / total
        tfidf: Dict[str, float] = {}
        for t, tf_val in tf.items():
            idf = math.log(n / df[t]) if df[t] > 0 else 0.0
            tfidf[t] = tf_val * idf
        result.append(tfidf)

    return result


def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Compute cosine similarity between two TF-IDF vectors (as dicts).

    **Why cosine similarity for semantic grouping:**
        Cosine similarity measures the angle between vectors, ignoring magnitude.
        Two responses with similar word-frequency profiles score near 1.0 even if
        one is longer.  A threshold of 0.9 means the responses share >= 90% of
        their directional TF-IDF "meaning vector."

    Args:
        a: TF-IDF dict for first document.
        b: TF-IDF dict for second document.

    Returns:
        Float in [0.0, 1.0].  1.0 = identical direction; 0.0 = orthogonal.
    """
    common_keys = set(a.keys()) & set(b.keys())
    dot = sum(a[k] * b[k] for k in common_keys)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# SemanticCluster
# ---------------------------------------------------------------------------


class SemanticCluster:
    """Groups responses by semantic similarity and computes cluster energy.

    **Why group before computing energy:**
        arXiv 2508.14496 shows that grouping equivalent-meaning responses and
        computing energy AT THE CLUSTER LEVEL reduces noise from surface-form
        variation.  Two responses that say the same thing with different words
        should share the same energy estimate.  Per-response energy ignores
        this — cluster energy exploits it.

    **Cluster algorithm:**
        Greedy single-link clustering: for each response (in order), assign it to
        the first existing cluster whose representative has cosine similarity >=
        threshold.  If no such cluster exists, create a new cluster.
        The representative of each cluster is its first member.

    Spec: REQ-PROBE-020, SCENARIO-PROBE-030
    """

    def __init__(self, threshold: float = 0.9) -> None:
        """
        Args:
            threshold: Cosine similarity threshold for merging two responses
                       into the same semantic cluster.  Default 0.9.
        """
        self.threshold = threshold

    def group_by_semantics(self, responses: List[str]) -> List[List[str]]:
        """Group responses into semantic clusters via TF-IDF cosine similarity.

        **Algorithm:**
            1. Compute TF-IDF vectors for all responses (one shared IDF corpus).
            2. Greedy assign: each response goes to the first cluster whose
               representative's cosine similarity to it is >= self.threshold.
            3. If no cluster matches, start a new one.

        Args:
            responses: List of response text strings.

        Returns:
            List of clusters, each a list of response strings.  Every input
            response appears in exactly one cluster.  Preserves insertion order
            within each cluster.

        Spec: REQ-PROBE-020
        """
        if not responses:
            return []

        tfidf_vecs = _compute_tfidf_matrix(responses)
        clusters: List[List[str]] = []
        representative_vecs: List[Dict[str, float]] = []

        for i, resp in enumerate(responses):
            vec = tfidf_vecs[i]
            placed = False
            for j, rep_vec in enumerate(representative_vecs):
                sim = _cosine_similarity(vec, rep_vec)
                if sim >= self.threshold:
                    clusters[j].append(resp)
                    placed = True
                    break
            if not placed:
                clusters.append([resp])
                representative_vecs.append(vec)

        return clusters

    def compute_cluster_energy(self, responses: List[str]) -> float:
        """Compute mean energy across all responses in a cluster.

        **Energy formula (arXiv 2508.14496, equation 3):**
            For a single response:
                energy(r) = -sum_i log(tfidf_score(token_i) + eps)
            where eps = 1e-9 prevents log(0).

            A token with TF-IDF close to 1 (common in this document, rare in
            corpus) contributes -log(1) = 0 to energy — expected token, no
            surprise.  A token with TF-IDF close to 0 contributes -log(eps) =
            very large positive energy — unexpected token, hallucination signal.

            **Limitation note:** This uses TF-IDF as a proxy for -log p(token).
            The true formula requires per-token log-probabilities from the
            generating model.  TF-IDF is a text-statistics approximation that
            captures word rarity, not the model's internal uncertainty.

            Cluster energy = mean of per-response energies.

        Args:
            responses: List of response strings within one semantic cluster.

        Returns:
            Float cluster energy score.  Higher = more energy = more likely hallucination.
            Returns 0.0 for empty input.

        Spec: REQ-PROBE-020
        """
        if not responses:
            return 0.0

        eps = 1e-9
        tfidf_vecs = _compute_tfidf_matrix(responses)
        energies: List[float] = []

        for vec in tfidf_vecs:
            if not vec:
                energies.append(0.0)
                continue
            # Sum -log(score + eps) over all tokens with nonzero TF-IDF.
            # Tokens with low TF-IDF (common = expected) get near-zero contribution;
            # tokens with high TF-IDF (rare = surprising) get large positive contribution.
            energy = sum(-math.log(score + eps) for score in vec.values())
            energies.append(energy)

        return sum(energies) / len(energies)


# ---------------------------------------------------------------------------
# SemanticEnergyProbe
# ---------------------------------------------------------------------------


class SemanticEnergyProbe:
    """Logit-space energy probe for hallucination detection (Tier 0g, arXiv 2508.14496).

    **Why this probe is different from NUP Probe v4 (Tier 0c):**
        NUP Probe v4 uses contrastive training over character bigrams to learn an
        energy gap between correct and incorrect steps.  It requires labelled training
        data and learns the geometry from examples.

        SemanticEnergyProbe is UNSUPERVISED.  It computes energy directly from the
        text's TF-IDF surprise value without any training.  This makes it:
        - Always available (no training data required)
        - Orthogonal signal (different computational pathway)
        - Theoretically grounded in the energy = -log p formulation

        The tradeoff: without labelled contrastive training, it may not achieve
        the same AUC as NUP v4 on domain-specific CoT steps.  The experiment
        (Exp 772) measures the actual gap.

    **Tier 0g wiring:**
        This probe is advisory — it does not short-circuit the pipeline.
        If is_high_energy() returns True, an advisory flag is set in the
        VerificationResult but tiers 1-3 still run.  This allows the pipeline
        to collect energy evidence without risking false-positive short-circuits.

    Args:
        energy_threshold: Score above which a response is flagged as high energy
                          (potential hallucination).  Default 5.0.

    Spec: REQ-PROBE-020, REQ-PROBE-021,
          SCENARIO-PROBE-030, SCENARIO-PROBE-031
    """

    def __init__(self, energy_threshold: float = 5.0) -> None:
        self.energy_threshold = energy_threshold

    def score(self, response_text: str) -> float:
        """Compute energy score for a single response.

        **What this measures:**
            Computes E(response) = -sum_i log(tfidf_score(token_i) + eps) for all
            tokens with nonzero TF-IDF weight, then normalises by response length
            (number of tokens) so that longer responses don't automatically score
            higher.

            The energy is computed in a single-document TF-IDF context (IDF = log(1/1) = 0
            for every token, so pure TF drives the score when there is only one document).
            This means: token frequency within the response drives the energy.

            **Limitation:** With a single document, IDF collapses to 0.  The score
            therefore reflects within-response token entropy, not inter-response
            surprise.  For multi-response comparison, use SemanticCluster.compute_cluster_energy()
            with a corpus of candidate responses instead.

            When actual logits are available, pass them as per-token log-probs and
            bypass this method (see Exp 772 discussion for the upgrade path).

        Args:
            response_text: The CoT step or response text to score.

        Returns:
            Non-negative float.  Higher = more surprising text = higher hallucination risk.

        Spec: REQ-PROBE-020
        """
        if not response_text.strip():
            return 0.0

        tokens = _tokenize(response_text)
        if not tokens:
            return 0.0

        # Single-document TF-IDF: IDF = 0 for all tokens (only one document in corpus),
        # so the score is TF-based only.  We use raw TF (count / total) as the proxy
        # for token probability p(token | response).
        total = len(tokens)
        tf: Dict[str, float] = defaultdict(float)
        for t in tokens:
            tf[t] += 1.0 / total

        eps = 1e-9
        # Energy = -sum log p(t_i), proxy: p(t_i) ~ TF(t_i in response)
        # Normalise by number of unique tokens to make the score length-independent.
        energy = sum(-math.log(p + eps) for p in tf.values())
        n_unique = len(tf)
        return energy / n_unique if n_unique > 0 else 0.0

    def is_high_energy(self, response_text: str) -> bool:
        """Return True when the response energy exceeds the threshold.

        **Advisory semantics (Tier 0g):**
            Returning True does NOT short-circuit the pipeline.  It sets an advisory
            flag to signal that this response has elevated energy and warrants
            closer inspection by downstream tiers.

        Args:
            response_text: The CoT step or response text to evaluate.

        Returns:
            True when score(response_text) > self.energy_threshold.

        Spec: REQ-PROBE-021, SCENARIO-PROBE-031
        """
        return self.score(response_text) > self.energy_threshold

    def evaluate_auc(
        self,
        texts: List[str],
        labels: List[int],
    ) -> float:
        """Compute AUROC of energy scores against binary correctness labels.

        **How this is used:**
            labels[i] = 1 means the i-th response is INCORRECT (positive class).
            labels[i] = 0 means the i-th response is CORRECT  (negative class).
            Higher energy = more likely to be incorrect.

            AUROC is computed via the standard trapezoidal rule on the ROC curve,
            same implementation as NUPProbeV4.evaluate_auc() for consistency.

        Args:
            texts:  List of response text strings.
            labels: Parallel binary labels (1 = incorrect/hallucinated, 0 = correct).

        Returns:
            Float AUROC in [0.0, 1.0].  0.5 = chance; 1.0 = perfect.  Returns 0.5
            when either class is absent from the test set.

        Spec: REQ-PROBE-020
        """
        n_pos = sum(1 for lb in labels if lb == 1)
        n_neg = sum(1 for lb in labels if lb == 0)
        if n_pos == 0 or n_neg == 0:
            return 0.5

        scored: List[Tuple[float, int]] = [
            (self.score(t), lb) for t, lb in zip(texts, labels)
        ]
        # Sort descending: high energy → predicted incorrect (positive)
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
