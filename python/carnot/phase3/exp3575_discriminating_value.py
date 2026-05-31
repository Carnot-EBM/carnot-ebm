from typing import Any

from carnot.phase3.p01_energy_vote_scoring import mcnemar_exact


def compute_conditional_lift(
    baseline_caught: list[bool],
    ensemble_caught: list[bool],
) -> dict[str, Any]:
    """Compute conditional lift of the ensemble over the baseline on held-out errors.

    Parameters
    ----------
    baseline_caught : list[bool]
        Whether the baseline caught the error.
    ensemble_caught : list[bool]
        Whether the ensemble caught the error.

    Returns
    -------
    dict
        Dictionary containing the computed metrics.
    """
    if len(baseline_caught) != len(ensemble_caught):
        raise ValueError("Inputs must have the same length.")

    n_errors = len(baseline_caught)
    if n_errors == 0:
        return {
            "baseline_fraction": 0.0,
            "ensemble_fraction": 0.0,
            "conditional_catch_rate": 0.0,
            "mcnemar_p": 1.0,
            "n_errors": 0,
        }

    baseline_misses = sum(1 for b in baseline_caught if not b)
    ensemble_catches_when_baseline_misses = sum(
        1 for b, e in zip(baseline_caught, ensemble_caught) if (not b) and e
    )

    baseline_frac = sum(baseline_caught) / n_errors
    ensemble_frac = sum(ensemble_caught) / n_errors

    cond_catch_rate = 0.0
    if baseline_misses > 0:
        cond_catch_rate = ensemble_catches_when_baseline_misses / baseline_misses

    # mcnemar_exact uses a_correct, b_correct
    # In our context: discordance is when they differ.
    p_val = mcnemar_exact(baseline_caught, ensemble_caught)

    return {
        "baseline_fraction": baseline_frac,
        "ensemble_fraction": ensemble_frac,
        "conditional_catch_rate": cond_catch_rate,
        "mcnemar_p": p_val,
        "n_errors": n_errors,
    }
