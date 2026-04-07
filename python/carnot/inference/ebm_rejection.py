"""EBM-guided rejection sampling: combine per-token activation energy with logprobs.

**Researcher summary:**
    Generates N candidate responses, extracts per-token hidden states from
    each, scores them through a trained Gibbs EBM, and combines with logprob
    scores to select the lowest-energy (most likely correct) candidate.

    This is the practical payoff of experiments 19-22: use the per-token EBM
    hallucination detector as a real-time candidate filter.

**Detailed explanation for engineers:**
    The pipeline:
    1. Generate N candidates using model.generate() with sampling.
    2. For each candidate, run a forward pass to get hidden states from the
       last transformer layer (or a configurable layer).
    3. For each generated token, compute EBM energy using the trained
       GibbsModel.energy() function.
    4. The mean EBM energy per candidate is a hallucination risk score:
       lower energy = more likely correct.
    5. Combine with the model's own logprobs (experiment 13) as a weighted
       composite: composite = ebm_weight * mean_ebm_energy - logprob_weight * mean_logprob
    6. Select candidate with lowest composite energy.

Spec: REQ-INFER-015
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

logger = logging.getLogger(__name__)


@dataclass
class EBMRejectionConfig:
    """Configuration for EBM-guided rejection sampling.

    **Detailed explanation for engineers:**
        Controls the balance between the model's own logprob confidence
        and the external EBM hallucination detector. Higher ebm_weight
        trusts the EBM more; higher logprob_weight trusts the model more.

        The hidden_layer parameter selects which transformer layer's
        activations to score. -1 means the last layer (default). Middle
        layers may retain more hallucination signal on instruction-tuned
        models (Principle 8).

    Spec: REQ-INFER-015
    """

    ebm_weight: float = 1.0
    logprob_weight: float = 1.0
    hidden_layer: int = -1  # Which layer to extract (-1 = last)
    n_candidates: int = 5
    temperature: float = 0.8
    max_new_tokens: int = 80

    def __post_init__(self) -> None:
        if self.ebm_weight < 0:
            raise ValueError(f"ebm_weight must be >= 0, got {self.ebm_weight}")
        if self.logprob_weight < 0:
            raise ValueError(f"logprob_weight must be >= 0, got {self.logprob_weight}")
        if self.n_candidates < 1:
            raise ValueError(f"n_candidates must be >= 1, got {self.n_candidates}")


@dataclass
class EBMCandidateScore:
    """Score for a single candidate response.

    Spec: REQ-INFER-015
    """

    response: str
    mean_logprob: float
    mean_ebm_energy: float
    composite_energy: float
    n_tokens: int


@dataclass
class EBMRejectionResult:
    """Result of EBM-guided rejection sampling.

    **Detailed explanation for engineers:**
        Contains the best candidate (lowest composite energy) and all
        candidates sorted by composite energy ascending (best first).

    Spec: REQ-INFER-015
    """

    best: EBMCandidateScore
    all_candidates: list[EBMCandidateScore] = field(default_factory=list)


def _generate_candidate_with_hidden_states(  # pragma: no cover
    model: Any,
    tokenizer: Any,
    prompt: str,
    do_sample: bool = True,
    temperature: float = 0.8,
    max_new_tokens: int = 80,
    hidden_layer: int = -1,
    strip_thinking: bool = True,
) -> tuple[str, float, Any, Any]:
    """Generate one candidate and return response, logprob, hidden states, and generated ids.

    **Detailed explanation for engineers:**
        Does two passes:
        1. model.generate() with output_scores=True → gets response + logprobs.
        2. model(full_sequence, output_hidden_states=True) → gets per-token
           hidden states from the specified layer.

        Returns the decoded response, mean logprob, hidden state tensor
        (shape: [n_generated_tokens, hidden_dim]), and generated token ids.

    Spec: REQ-INFER-015
    """
    import torch

    # Use chat template if tokenizer supports it
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt

    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    # Generate with logprobs
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **gen_kwargs,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences[0, prompt_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Strip thinking tags if present
    if strip_thinking and "</think>" in response:
        response = response.split("</think>")[-1].strip()

    # Compute mean logprob from scores
    total_logprob = 0.0
    n_tokens = 0
    if hasattr(outputs, "scores") and outputs.scores:
        for step_idx, scores in enumerate(outputs.scores):
            if step_idx >= len(generated_ids):
                break
            token_id = generated_ids[step_idx].item()
            log_probs = torch.log_softmax(scores[0], dim=-1)
            total_logprob += log_probs[token_id].item()
            n_tokens += 1
    mean_logprob = total_logprob / max(n_tokens, 1)

    # Get hidden states from forward pass
    with torch.no_grad():
        hidden_out = model(outputs.sequences, output_hidden_states=True)
        hs = hidden_out.hidden_states

    # Extract the specified layer's activations for generated tokens only
    layer_hidden = hs[hidden_layer][0, prompt_len:, :].float().cpu().numpy()

    return response, mean_logprob, layer_hidden, generated_ids.cpu()


def score_activations_with_ebm(
    ebm: Any,
    activations: Any,
) -> float:
    """Score per-token activations through a trained EBM.

    **Detailed explanation for engineers:**
        Takes a numpy array of shape (n_tokens, hidden_dim) and computes
        the mean energy across all tokens using the EBM's energy() method.
        Lower energy = more likely correct (the EBM was trained to assign
        low energy to correct-answer activations).

    Args:
        ebm: A trained GibbsModel (or any model with .energy() method).
        activations: numpy array of shape (n_tokens, hidden_dim).

    Returns:
        Mean energy across all tokens (float).

    Spec: REQ-INFER-015
    """
    total_energy = 0.0
    n_tokens = len(activations)
    for t in range(n_tokens):
        token_act = jnp.array(activations[t])
        total_energy += float(ebm.energy(token_act))
    return total_energy / max(n_tokens, 1)


def ebm_rejection_sample(  # pragma: no cover
    model: Any,
    tokenizer: Any,
    ebm: Any,
    prompt: str,
    config: EBMRejectionConfig | None = None,
) -> EBMRejectionResult:
    """Generate N candidates and select the one with lowest composite energy.

    **Researcher summary:**
        Combines two complementary signals for candidate selection:
        1. Logprobs (model's own confidence) — experiment 13 showed +10%
        2. Per-token EBM energy (external hallucination detector) — experiment 19-22

        The composite energy is: ebm_weight * mean_ebm + (-logprob_weight * mean_logprob)
        Lower is better. If the signals are complementary, the composite
        should outperform either alone.

    **Detailed explanation for engineers:**
        For each of N candidates:
        1. Generate response with sampling (temperature > 0)
        2. Compute mean logprob from the generation scores
        3. Extract hidden states from the specified layer
        4. Score each token's activation through the trained EBM
        5. Compute composite: ebm_weight * mean_ebm - logprob_weight * mean_logprob
        6. Select candidate with lowest composite energy

    Args:
        model: HuggingFace causal LM (loaded and on device).
        tokenizer: Corresponding tokenizer.
        ebm: Trained GibbsModel for scoring activations.
        prompt: The user's question/task.
        config: Optional configuration. Defaults to EBMRejectionConfig().

    Returns:
        EBMRejectionResult with best candidate and all scored candidates.

    Spec: REQ-INFER-015
    """
    if config is None:
        config = EBMRejectionConfig()

    candidates: list[EBMCandidateScore] = []
    use_sampling = config.n_candidates > 1

    for i in range(config.n_candidates):
        response, mean_logprob, activations, gen_ids = _generate_candidate_with_hidden_states(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            do_sample=use_sampling,
            temperature=config.temperature,
            max_new_tokens=config.max_new_tokens,
            hidden_layer=config.hidden_layer,
        )

        mean_ebm = score_activations_with_ebm(ebm, activations)

        # Composite: lower is better
        # -logprob because higher logprob = better, but we want lower = better
        composite = config.ebm_weight * mean_ebm - config.logprob_weight * mean_logprob

        candidates.append(EBMCandidateScore(
            response=response,
            mean_logprob=mean_logprob,
            mean_ebm_energy=mean_ebm,
            composite_energy=composite,
            n_tokens=len(activations),
        ))

        logger.debug(
            "Candidate %d: logprob=%.3f, ebm=%.3f, composite=%.3f, tokens=%d",
            i, mean_logprob, mean_ebm, composite, len(activations),
        )

    # Sort by composite energy ascending (lowest = best)
    candidates.sort(key=lambda c: c.composite_energy)

    return EBMRejectionResult(
        best=candidates[0],
        all_candidates=candidates,
    )
