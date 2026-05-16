"""
Fast-Slow Variant Codification Module.
Provides programmatic access to the codified metrics from the Fast-Slow Variant.
"""

def get_fast_slow_codified_metrics():
    """
    Returns the codified metrics for the Fast-Slow variant
    as established by exp1811 and exp1909.
    """
    return {
        "exp1811": {
            "sample_efficiency_ratio": 3.1,
            "kl_drift_ratio": 0.25,
            "seed": 172911
        },
        "exp1909": {
            "confirmation_sample_efficiency_ratio": 3.0,
            "confirmation_kl_drift_ratio": 0.2,
            "seed": 192737
        }
    }
