"""
Negative Constraint Optimization (NCO) module.
"""

def compute_nco_rejection_rate(token_logprobs: list[float], threshold: float = -10.0) -> float:
    """
    Computes the rejection rate of tokens based on a log probability threshold.
    Tokens with a logprob strictly less than the threshold are rejected.
    """
    if not token_logprobs:
        return 0.0
    rejected = sum(1 for lp in token_logprobs if lp < threshold)
    return rejected / len(token_logprobs)
