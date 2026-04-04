"""ARM-EBM bijection: extract per-token energy from LLM logits.

**Researcher summary:**
    Every autoregressive LLM is an EBM in disguise (arxiv 2512.15605).
    The bijection extracts per-token immediate reward (energy contribution)
    from logits: r(s,y) = logit(y) - logsumexp(logits).

    This is exactly the log-probability, but framed as energy it reveals:
    - Which tokens the model is confident about (reward near 0)
    - Which tokens are uncertain (large negative reward) — hallucination points

**Detailed explanation for engineers:**
    A language model assigns probabilities to each next token given the context.
    Internally, it produces a vector of "logits" — one number per vocabulary
    token — and then applies softmax to get probabilities:

        P(token_i | context) = exp(logit_i) / sum(exp(logit_j) for all j)

    Taking the log of this probability gives:

        log P(token_i | context) = logit_i - logsumexp(all logits)

    The paper arxiv 2512.15605 shows this log-probability IS an energy
    contribution: each token adds r(s,y) = log P(y|s) to the total sequence
    "reward". The total sequence energy is E = -sum(r_t), so a high-probability
    sequence has low energy (which is exactly the EBM framework).

    **Why this matters for hallucination detection:**
    When the model is uncertain about a token, its logit is not much higher
    than alternatives, so log P is very negative (say -5 or -10). These tokens
    contribute a lot of "energy" and are exactly where hallucinations hide.
    By thresholding per-token rewards, we can pinpoint which specific tokens
    in a generation are most likely to be wrong.

    **The bijection identity:**
    reward = logit(chosen) - logsumexp(all logits) = log P(chosen | context)

    This means if you have logprobs from an API (like OpenAI's), those ARE
    the per-token rewards — no extra computation needed. If you have raw logits
    (from running the model locally), you compute logsumexp yourself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp


@dataclass
class TokenEnergyAnalysis:
    """Complete energy analysis of a token sequence.

    **What each field means (in plain English):**

    - ``token_rewards``: For each token in the sequence, how "expected" was it?
      Values near 0 mean very expected (confident). Very negative values (like
      -5 or -10) mean the model was surprised by that token — it did not expect
      it, which is a hallucination signal.

    - ``sequence_energy``: A single number summarizing total confidence. Lower
      (more negative) means the model was confident overall. Higher means the
      model was uncertain about many tokens.

    - ``hallucination_indices``: Positions in the sequence where the model was
      especially uncertain (reward below the threshold). These are the specific
      tokens most likely to be wrong.

    - ``mean_reward``: Average per-token confidence. Useful for comparing
      different generations of the same prompt.

    - ``min_reward``: The worst (most uncertain) token in the sequence. If this
      is very negative, at least one token is highly suspicious.
    """

    token_rewards: list[float] = field(default_factory=list)
    sequence_energy: float = 0.0
    hallucination_indices: list[int] = field(default_factory=list)
    mean_reward: float = 0.0
    min_reward: float = 0.0


def extract_token_rewards(
    logits: list[list[float]],
    token_ids: list[int],
) -> list[float]:
    """Extract per-token immediate reward via the ARM-EBM bijection.

    **The bijection formula:**

        r(s_t, y_t) = logit(y_t) - logsumexp(logits_t)

    This equals log P(y_t | s_t), the log-probability of the chosen token.
    In the EBM framework from arxiv 2512.15605, this is the "immediate reward"
    that the token contributes to the total sequence energy.

    **Detailed explanation:**
        For each position t in the sequence, the model outputs a logit vector
        (one value per vocabulary token). We pick the logit for the token that
        was actually generated (token_ids[t]) and subtract the logsumexp of
        all logits at that position. The logsumexp is the "soft value function"
        V_q(s) from the paper — it measures how much total probability mass
        the model has at that position.

        If the chosen token had a high logit relative to others, the reward
        is near 0 (very confident). If the chosen token had a low logit
        relative to others, the reward is very negative (uncertain).

    Args:
        logits: List of logit vectors, one per sequence position. Each inner
            list has one float per vocabulary token. Shape: [seq_len, vocab_size].
        token_ids: Indices of the chosen tokens at each position.
            Shape: [seq_len]. Must have same length as logits.

    Returns:
        List of per-token rewards. Each value is the log-probability of the
        chosen token at that position. Shape: [seq_len].

    Raises:
        ValueError: If logits and token_ids have different lengths.
        IndexError: If any token_id is out of range for its logit vector.
    """
    if len(logits) != len(token_ids):
        msg = (
            f"logits has {len(logits)} positions but token_ids has "
            f"{len(token_ids)} — they must match (one token_id per position)"
        )
        raise ValueError(msg)

    rewards: list[float] = []
    for t, (logit_vec, token_id) in enumerate(zip(logits, token_ids)):
        logit_array = jnp.array(logit_vec)
        # The bijection: r(s,y) = logit(y) - logsumexp(logits)
        # This equals log P(y|s), the log-probability of the chosen token.
        chosen_logit = logit_array[token_id]
        log_normalizer = jnp.log(jnp.sum(jnp.exp(logit_array)))
        reward = float(chosen_logit - log_normalizer)
        rewards.append(reward)

    return rewards


def extract_token_rewards_from_logprobs(
    logprobs: list[float],
) -> list[float]:
    """Simplified reward extraction using pre-computed log-probabilities.

    **Why this exists:**
        Many LLM APIs (OpenAI, Anthropic) return log-probabilities directly
        instead of raw logit vectors. Since the bijection gives:

            r(s,y) = logit(y) - logsumexp(logits) = log P(y|s) = logprob

        the reward IS the logprob. No further computation is needed. This
        function exists for API-friendly workflows where you only have logprobs.

    **When to use this vs. extract_token_rewards:**
        - Use this function when you get logprobs from an API call.
        - Use extract_token_rewards when you have full logit vectors (e.g.,
          from running a model locally with output_hidden_states=True).

    Args:
        logprobs: Pre-computed log-probabilities for each chosen token.
            These come directly from the API response. Shape: [seq_len].

    Returns:
        The same logprobs as a list of floats (identity function, but typed
        consistently with the rest of the API for pipeline composability).
    """
    # The reward IS the logprob — the bijection identity.
    # We return a new list (not a reference) for consistency and immutability.
    return list(logprobs)


def compute_sequence_energy(token_rewards: list[float]) -> float:
    """Compute total sequence energy from per-token rewards.

    **The energy formula:**

        E(sequence) = -sum(r(s_t, y_t) for all t)

    **Detailed explanation:**
        In the EBM framework, probability is proportional to exp(-E). So a
        sequence with high probability (all tokens were expected) has low energy,
        and a sequence with low probability (many surprising tokens) has high
        energy.

        Since each reward r_t = log P(y_t | s_t) is negative or zero, the
        energy E = -sum(r_t) is always non-negative. A perfectly confident
        sequence (all rewards = 0) has energy = 0. A sequence full of
        surprises has high energy.

    Args:
        token_rewards: Per-token rewards from extract_token_rewards or
            extract_token_rewards_from_logprobs. Shape: [seq_len].

    Returns:
        Total sequence energy. Lower = more confident generation.
        Always >= 0 for valid log-probabilities.
    """
    if not token_rewards:
        return 0.0
    return -sum(token_rewards)


def identify_hallucination_tokens(
    token_rewards: list[float],
    threshold: float = -2.0,
) -> list[int]:
    """Find positions where the model was especially uncertain.

    **What this does:**
        Scans per-token rewards and returns indices where the reward is below
        the threshold. These are positions where the model assigned low
        probability to the chosen token — the tokens most likely to be wrong.

    **Choosing a threshold:**
        - -1.0: Very aggressive — flags tokens with < 37% probability.
        - -2.0: Moderate (default) — flags tokens with < 13.5% probability.
        - -3.0: Conservative — flags tokens with < 5% probability.
        - -5.0: Only extreme uncertainty — flags tokens with < 0.7% probability.

        The default of -2.0 means "the model thought this token had less than
        a 1-in-7 chance of being correct", which is a reasonable hallucination
        signal for most applications.

    Args:
        token_rewards: Per-token rewards. Shape: [seq_len].
        threshold: Reward value below which a token is flagged. Default -2.0.

    Returns:
        Sorted list of indices where reward < threshold.
    """
    return [i for i, r in enumerate(token_rewards) if r < threshold]


def analyze_token_energy(
    logprobs: list[float],
    threshold: float = -2.0,
) -> TokenEnergyAnalysis:
    """Full energy analysis of a token sequence from log-probabilities.

    **What this does (step by step):**

    1. Converts logprobs to per-token rewards (identity via bijection).
    2. Computes total sequence energy = -sum(rewards).
    3. Identifies hallucination tokens (rewards below threshold).
    4. Computes summary statistics (mean, min reward).

    This is the main entry point for analyzing LLM outputs. Feed it the
    logprobs from any LLM API and get back a structured analysis.

    **Example usage:**

        logprobs = [-0.1, -0.3, -4.5, -0.2, -0.1]  # 5 tokens
        analysis = analyze_token_energy(logprobs)
        # analysis.hallucination_indices == [2]  # token 2 is suspicious
        # analysis.sequence_energy == 5.2  # total energy
        # analysis.min_reward == -4.5  # worst token

    Args:
        logprobs: Log-probabilities for each token in the sequence.
        threshold: Reward threshold for hallucination detection. Default -2.0.

    Returns:
        TokenEnergyAnalysis with all derived quantities.
    """
    token_rewards = extract_token_rewards_from_logprobs(logprobs)
    sequence_energy = compute_sequence_energy(token_rewards)
    hallucination_indices = identify_hallucination_tokens(token_rewards, threshold)

    mean_reward = sum(token_rewards) / len(token_rewards) if token_rewards else 0.0
    min_reward = min(token_rewards) if token_rewards else 0.0

    return TokenEnergyAnalysis(
        token_rewards=token_rewards,
        sequence_energy=sequence_energy,
        hallucination_indices=hallucination_indices,
        mean_reward=mean_reward,
        min_reward=min_reward,
    )
