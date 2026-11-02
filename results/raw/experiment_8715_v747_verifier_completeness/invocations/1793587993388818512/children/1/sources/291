"""Multi-concept contrastive vectors for fine-grained hallucination analysis.

**Researcher summary:**
    Inspired by Anthropic's emotion/concept vector research, this module
    generates multiple concept vectors (certain, uncertain, confabulating,
    reasoning, memorized) by prompting a model, extracting activations,
    and computing contrastive directions. Per-concept energy scores
    enable fine-grained diagnosis of *why* a model is hallucinating.

**Detailed explanation for engineers:**
    The hallucination direction (from ``hallucination_direction.py``) gives
    a single binary signal: "is this hallucinating?" But hallucinations
    have different causes:

    - **Confabulation**: The model makes up facts it was never trained on.
    - **Uncertainty**: The model doesn't know the answer but generates
      one anyway instead of expressing doubt.
    - **Memorization failure**: The model has the fact in its training
      data but fails to retrieve it accurately.

    This module builds concept-specific direction vectors by:

    1. **Prompting**: For each concept (certain, uncertain, confabulating,
       reasoning, memorized), generate text from the model using a
       concept-specific prompt. For example, the "confabulating" prompt
       asks the model to write a made-up fact, while the "certain" prompt
       asks for a confident factual statement.

    2. **Extracting**: Run each generated text through the model's forward
       pass and extract activations (using ``extract_layer_activations``).
       Average over layers and sequence positions to get a single vector
       per generated sample.

    3. **Contrasting**: For each concept, compute the mean-difference
       direction between that concept's activations and the activations
       from all other concepts combined. This gives a direction that is
       uniquely associated with each concept.

    4. **Scoring**: Given a new activation, compute its projection onto
       each concept vector to get per-concept energy scores. High energy
       on the "confabulating" vector means the model is likely making
       things up; high energy on "uncertain" means it's generating
       despite not knowing.

    **Integration with Carnot:**
    The concept energies can be composed with other constraints via
    ``ComposedEnergy``. For instance, you might weight the "confabulating"
    energy 2x higher than "uncertain" because confabulation is harder
    to detect downstream.

    **Graceful fallback:**
    Like other embedding modules, this lazy-imports ``torch`` and
    ``transformers``. If unavailable, functions return ``None``.

Spec: REQ-INFER-016
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

# Default concept prompts for multi-vector analysis.
# Each prompt is designed to elicit a specific generation mode from the
# model. The model's activations while generating each type of text
# will form distinct clusters in activation space.
CONCEPT_PROMPTS: dict[str, str] = {
    "certain": ("Write a confident factual statement about a well-known historical event."),
    "uncertain": ("Write an uncertain guess about something you are not sure about."),
    "confabulating": ("Write a made-up fact that sounds plausible but is entirely fictional."),
    "reasoning": ("Write a step-by-step derivation of a simple mathematical result."),
    "memorized": ("Recite a well-known fact about the solar system."),
}


def find_concept_vectors(
    model: Any,
    tokenizer: Any,
    concept_prompts: dict[str, str] | None = None,
    n_samples: int = 10,
) -> dict[str, jnp.ndarray] | None:
    """Generate text for each concept, extract activations, compute contrastive vectors.

    **Researcher summary:**
        For each concept, generates ``n_samples`` texts, extracts their
        mean activations, and computes the contrastive direction
        (concept mean minus all-others mean) as the concept vector.

    **Detailed explanation for engineers:**
        The algorithm has three phases:

        **Phase 1: Generation**
        For each concept in ``concept_prompts``, call ``model.generate()``
        ``n_samples`` times with the concept's prompt. This produces
        diverse text samples that represent the model operating in each
        "mode" (certain, uncertain, confabulating, etc.).

        **Phase 2: Activation extraction**
        For each generated text, run the full text through the model's
        forward pass, extract hidden states from all layers (using
        ``extract_layer_activations``), and compute the mean activation
        across all layers and sequence positions. This gives a single
        vector per sample.

        **Phase 3: Contrastive direction**
        For each concept, compute:
        - ``mu_concept``: Mean activation over that concept's samples.
        - ``mu_others``: Mean activation over all other concepts' samples.
        - ``direction = mu_concept - mu_others``: The contrastive vector.
        - Normalize to unit length.

        The result is a dict mapping concept name to its contrastive
        direction vector.

    Args:
        model: A Hugging Face causal language model.
        tokenizer: The corresponding tokenizer.
        concept_prompts: Dict mapping concept names to generation prompts.
            Defaults to ``CONCEPT_PROMPTS``.
        n_samples: Number of text samples to generate per concept.

    Returns:
        Dict mapping concept name to JAX array of shape ``(hidden_dim,)``
        representing the concept's contrastive direction vector. Returns
        None if torch/transformers are unavailable or activation
        extraction fails.

    Spec: REQ-INFER-016
    """
    if concept_prompts is None:
        concept_prompts = CONCEPT_PROMPTS

    try:
        import torch
    except ImportError:  # pragma: no cover
        return None

    from carnot.embeddings.activation_extractor import extract_layer_activations

    # Phase 1 & 2: Generate and extract activations for each concept.
    concept_activations: dict[str, list[jnp.ndarray]] = {}

    for concept_name, prompt in concept_prompts.items():
        concept_acts: list[jnp.ndarray] = []

        for _i in range(n_samples):
            # Generate text from the concept prompt.
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            param = next(model.parameters(), None)
            dev = param.device if param is not None else "cpu"
            inputs = {k: v.to(dev) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=50)

            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Extract activations from the generated text.
            layer_acts = extract_layer_activations(generated_text)
            if layer_acts is None:
                continue

            # Average across all layers and sequence positions to get
            # a single vector per sample.
            all_layer_means = []
            for _layer_idx, act in layer_acts.items():
                # act shape: (seq_len, hidden_dim) -> mean over seq_len
                all_layer_means.append(jnp.mean(act, axis=0))

            if all_layer_means:
                # Mean across all layers: shape (hidden_dim,)
                sample_mean = jnp.mean(jnp.stack(all_layer_means), axis=0)
                concept_acts.append(sample_mean)

        if concept_acts:
            concept_activations[concept_name] = concept_acts

    if not concept_activations:
        return None

    # Phase 3: Compute contrastive directions.
    return _compute_contrastive_vectors(concept_activations)


def _compute_contrastive_vectors(
    concept_activations: dict[str, list[jnp.ndarray]],
) -> dict[str, jnp.ndarray]:
    """Compute contrastive direction vectors from per-concept activation sets.

    **Detailed explanation for engineers:**
        For each concept, we compute:

            direction = mean(concept_activations) - mean(all_other_activations)

        This gives a vector that points from the "average other concept"
        toward the target concept. We normalize it to unit length so that
        projection-based energy scores are comparable across concepts.

        If a concept has only one sample or the direction has zero norm
        (all concepts have identical activations, which is degenerate),
        we still include it as a zero vector rather than failing.

    Args:
        concept_activations: Dict mapping concept name to list of
            activation vectors (each shape ``(hidden_dim,)``).

    Returns:
        Dict mapping concept name to unit-normalized contrastive
        direction vector of shape ``(hidden_dim,)``.

    Spec: REQ-INFER-016
    """
    # Compute per-concept means.
    concept_means: dict[str, jnp.ndarray] = {}
    all_activations: list[jnp.ndarray] = []

    for concept_name, acts in concept_activations.items():
        stacked = jnp.stack(acts)
        concept_means[concept_name] = jnp.mean(stacked, axis=0)
        all_activations.extend(acts)

    global_mean = jnp.mean(jnp.stack(all_activations), axis=0)

    # Compute contrastive directions: concept mean minus "everything else" mean.
    result: dict[str, jnp.ndarray] = {}
    n_concepts = len(concept_activations)

    for concept_name, mu_concept in concept_means.items():
        if n_concepts <= 1:
            # Only one concept: direction is just the concept mean minus global.
            # This is zero by definition, but we handle it gracefully.
            direction = mu_concept - global_mean
        else:
            # mean_others = (global_mean * n_total - concept_sum) / n_others
            # Simplified: use weighted combination of concept means.
            n_concept = len(concept_activations[concept_name])
            n_total = len(all_activations)
            n_others = n_total - n_concept

            if n_others > 0:
                # mu_others = (global_mean * n_total - mu_concept * n_concept) / n_others
                mu_others = (global_mean * n_total - mu_concept * n_concept) / n_others
                direction = mu_concept - mu_others
            else:
                direction = jnp.zeros_like(mu_concept)  # pragma: no cover

        # Normalize to unit length.
        norm = jnp.linalg.norm(direction)
        direction = direction / jnp.maximum(norm, 1e-8)
        result[concept_name] = direction

    return result


def concept_energy(
    activation: jnp.ndarray,
    concept_vectors: dict[str, jnp.ndarray],
) -> dict[str, float]:
    """Compute per-concept energy scores for an activation vector.

    **Researcher summary:**
        Projects the activation onto each concept vector and returns
        the signed projection as a per-concept energy score.

    **Detailed explanation for engineers:**
        For each concept, the energy is the dot product of the activation
        with the concept's unit direction vector. This is equivalent to
        the signed scalar projection:

            E_concept = dot(activation, concept_vector)

        Interpretation:
        - **High positive energy** on "confabulating": the activation
          resembles confabulated text activations.
        - **High positive energy** on "certain": the activation
          resembles confident factual statements.
        - **Negative energy**: the activation is moving *away* from
          that concept.

        The energies are comparable across concepts because all concept
        vectors are unit-normalized. A higher absolute value means
        stronger alignment with the concept.

    Args:
        activation: A single activation vector, shape ``(hidden_dim,)``.
        concept_vectors: Dict mapping concept name to unit direction
            vector, as returned by ``find_concept_vectors``.

    Returns:
        Dict mapping concept name to energy score (float).

    Spec: REQ-INFER-016
    """
    result: dict[str, float] = {}
    for concept_name, direction in concept_vectors.items():
        energy = float(jnp.dot(activation, direction))
        result[concept_name] = energy
    return result


def best_concept_for_detection(
    concept_vectors: dict[str, jnp.ndarray],
    correct_acts: list[jnp.ndarray],
    hallucinated_acts: list[jnp.ndarray],
) -> str:
    """Find which concept vector best separates correct from hallucinated activations.

    **Researcher summary:**
        Computes the Fisher-style discriminant ratio for each concept
        vector and returns the concept with the highest separation
        between correct and hallucinated activation distributions.

    **Detailed explanation for engineers:**
        For each concept vector ``v``, we project all correct and
        hallucinated activations onto ``v`` to get two sets of scalar
        values. Then we compute a separation metric:

            separation = |mean_halluc_proj - mean_correct_proj|
                         / (std_correct_proj + std_halluc_proj + eps)

        This is a 1-D Fisher discriminant ratio. The concept with the
        highest ratio is the best single-concept detector for
        hallucination.

        **Why not use all concepts together?**
        You can (and should, for production systems). This function
        tells you which *single* concept is most informative, which
        is useful for interpretability ("the model hallucinates by
        confabulating, not by being uncertain") and for feature
        selection in multi-concept energy scoring.

    Args:
        concept_vectors: Dict mapping concept name to direction vector.
        correct_acts: List of activation vectors from correct outputs.
        hallucinated_acts: List of activation vectors from hallucinated
            outputs.

    Returns:
        Name of the concept with the best separation (string).

    Raises:
        ValueError: If concept_vectors, correct_acts, or
            hallucinated_acts is empty.

    Spec: REQ-INFER-016
    """
    if not concept_vectors:
        raise ValueError("concept_vectors must not be empty")
    if not correct_acts:
        raise ValueError("correct_acts must not be empty")
    if not hallucinated_acts:
        raise ValueError("hallucinated_acts must not be empty")

    correct_mat = jnp.stack(correct_acts)
    halluc_mat = jnp.stack(hallucinated_acts)

    best_name = ""
    best_score = -1.0
    eps = 1e-8

    for concept_name, direction in concept_vectors.items():
        # Project all activations onto this concept direction.
        correct_proj = correct_mat @ direction  # shape (n_correct,)
        halluc_proj = halluc_mat @ direction  # shape (n_halluc,)

        # Fisher discriminant ratio on the 1-D projections.
        mean_diff = float(jnp.abs(jnp.mean(halluc_proj) - jnp.mean(correct_proj)))
        std_sum = float(jnp.std(correct_proj) + jnp.std(halluc_proj))
        score = mean_diff / (std_sum + eps)

        if score > best_score:
            best_score = score
            best_name = concept_name

    return best_name
