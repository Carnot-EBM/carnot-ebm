#!/usr/bin/env python3
"""Experiment 863: HalluSAEGeometricProbe Tier 0i — TF-IDF bigram SAE proxy.

**Researcher summary:**
    Evaluates HalluSAEGeometricProbe, a Tier 0i advisory hallucination detector
    based on arXiv 2604.16430 (HalluSAE).  Instead of a trained Sparse Autoencoder,
    we use TF-IDF bigrams as a no-GPU proxy for SAE feature activations.  The probe
    measures how far CoT steps drift from the centroid of correct reference steps in
    that feature space.  High drift = potential hallucination.

    Benchmark: 50 synthetic CoT pairs (25 correct, 25 with injected nonsense steps).
    We compute AUC-ROC using sklearn.metrics.roc_auc_score to evaluate discriminability.

    Honest verdicts:
        tier_0i_viable   if AUC_geometric > 0.65
        tier_0i_marginal if 0.55 < AUC_geometric <= 0.65
        tier_0i_fails    if AUC_geometric <= 0.55

Spec: REQ-PROBE-050, SCENARIO-PROBE-060
"""

import sys
from pathlib import Path

# Make project root importable regardless of CWD
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

from carnot.probes.hallusae_geometric_probe import HalluSAEGeometricProbe  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic CoT dataset construction
# ---------------------------------------------------------------------------
# We build 50 CoT pairs that are concrete enough to produce differentiated
# TF-IDF feature vectors, but short enough to run without GPU.

# 25 correct multi-step arithmetic problems.
# Each problem is represented as a list of CoT steps (the "trajectory").
CORRECT_COT_PAIRS: list[list[str]] = [
    # Problem 1
    [
        "Let x equal 5. We need to compute 2 times x.",
        "2 times 5 equals 10.",
        "Therefore x times 2 equals 10.",
    ],
    # Problem 2
    [
        "We have 12 apples and give away 4 apples.",
        "Subtracting 4 from 12 gives 12 minus 4 equals 8.",
        "Therefore 8 apples remain.",
    ],
    # Problem 3
    [
        "The perimeter of a square with side 7 is 4 times 7.",
        "4 times 7 equals 28.",
        "Therefore the perimeter equals 28 units.",
    ],
    # Problem 4
    [
        "We need the sum of integers from 1 to 4: 1 plus 2 plus 3 plus 4.",
        "1 plus 2 equals 3. Then 3 plus 3 equals 6. Then 6 plus 4 equals 10.",
        "Therefore the sum equals 10.",
    ],
    # Problem 5
    [
        "Speed equals distance divided by time. Distance is 60 km, time is 2 hours.",
        "60 divided by 2 equals 30.",
        "Therefore the speed equals 30 km per hour.",
    ],
    # Problem 6
    [
        "We want to compute 3 squared plus 4 squared.",
        "3 squared equals 9. 4 squared equals 16. 9 plus 16 equals 25.",
        "Therefore 3 squared plus 4 squared equals 25.",
    ],
    # Problem 7
    [
        "A rectangle has length 8 and width 3. Area equals length times width.",
        "8 times 3 equals 24.",
        "Therefore the area equals 24 square units.",
    ],
    # Problem 8
    [
        "We have 100 dollars and spend 37 dollars.",
        "100 minus 37 equals 63.",
        "Therefore 63 dollars remain.",
    ],
    # Problem 9
    [
        "Convert 3 kilometers to meters: multiply by 1000.",
        "3 times 1000 equals 3000.",
        "Therefore 3 kilometers equals 3000 meters.",
    ],
    # Problem 10
    [
        "Compute the average of 4, 8, and 12.",
        "Sum equals 4 plus 8 plus 12 equals 24. Divide by 3 values.",
        "24 divided by 3 equals 8. Therefore the average equals 8.",
    ],
    # Problem 11
    [
        "We need 5 factorial: 5 times 4 times 3 times 2 times 1.",
        "5 times 4 equals 20. 20 times 3 equals 60. 60 times 2 equals 120. 120 times 1 equals 120.",
        "Therefore 5 factorial equals 120.",
    ],
    # Problem 12
    [
        "A triangle has base 6 and height 4. Area equals half base times height.",
        "Half of 6 equals 3. 3 times 4 equals 12.",
        "Therefore the area equals 12 square units.",
    ],
    # Problem 13
    [
        "We have a 20 percent discount on a price of 50 dollars.",
        "20 percent of 50 equals 0.20 times 50 equals 10.",
        "Therefore the discounted price equals 50 minus 10 equals 40 dollars.",
    ],
    # Problem 14
    [
        "Compute 7 times 8.",
        "7 times 8 equals 56.",
        "Therefore the product is 56.",
    ],
    # Problem 15
    [
        "We have 3 groups of 9 students each.",
        "Total students equals 3 times 9 equals 27.",
        "Therefore there are 27 students in total.",
    ],
    # Problem 16
    [
        "Find the remainder of 17 divided by 5.",
        "17 divided by 5 is 3 with remainder 2 because 3 times 5 equals 15 and 17 minus 15 equals 2.",
        "Therefore the remainder equals 2.",
    ],
    # Problem 17
    [
        "Water boils at 100 degrees Celsius. Convert to Fahrenheit using F equals C times 1.8 plus 32.",
        "100 times 1.8 equals 180. 180 plus 32 equals 212.",
        "Therefore 100 degrees Celsius equals 212 degrees Fahrenheit.",
    ],
    # Problem 18
    [
        "Compute the square root of 144.",
        "12 times 12 equals 144.",
        "Therefore the square root of 144 equals 12.",
    ],
    # Problem 19
    [
        "A train travels 250 km in 5 hours. What is the average speed?",
        "Speed equals distance divided by time. 250 divided by 5 equals 50.",
        "Therefore the average speed equals 50 km per hour.",
    ],
    # Problem 20
    [
        "How many seconds are in 3 minutes?",
        "1 minute equals 60 seconds. 3 times 60 equals 180.",
        "Therefore there are 180 seconds in 3 minutes.",
    ],
    # Problem 21
    [
        "Compute 15 percent of 200.",
        "15 divided by 100 equals 0.15. 0.15 times 200 equals 30.",
        "Therefore 15 percent of 200 equals 30.",
    ],
    # Problem 22
    [
        "A cube has side length 4. Compute the volume.",
        "Volume equals side cubed. 4 cubed equals 4 times 4 times 4.",
        "4 times 4 equals 16. 16 times 4 equals 64. Therefore the volume equals 64 cubic units.",
    ],
    # Problem 23
    [
        "Find the greatest common divisor of 12 and 18.",
        "Factors of 12: 1, 2, 3, 4, 6, 12. Factors of 18: 1, 2, 3, 6, 9, 18.",
        "Common factors: 1, 2, 3, 6. Greatest common divisor equals 6.",
    ],
    # Problem 24
    [
        "Compute 2 raised to the power of 8.",
        "2^1=2, 2^2=4, 2^3=8, 2^4=16, 2^5=32, 2^6=64, 2^7=128, 2^8=256.",
        "Therefore 2 raised to the power of 8 equals 256.",
    ],
    # Problem 25
    [
        "A store sells 3 shirts at 15 dollars each and 2 pants at 25 dollars each.",
        "Shirts total: 3 times 15 equals 45 dollars. Pants total: 2 times 25 equals 50 dollars.",
        "Grand total equals 45 plus 50 equals 95 dollars.",
    ],
]

# 25 hallucinated CoT pairs — same problem setup, but one step is replaced with
# geometrically-distant nonsense.  The nonsense step uses entirely different
# vocabulary (colors, animals, abstract terms) that will be far from the
# arithmetic reference centroid in TF-IDF space.
HALLUCINATED_COT_PAIRS: list[list[str]] = [
    [
        "Let x equal 5. We need to compute 2 times x.",
        "Therefore x equals 42 because the purple elephant decided so.",
        "Therefore x times 2 equals 84.",
    ],
    [
        "We have 12 apples and give away 4 apples.",
        "Bananas are yellow and clouds float freely above the ocean waves.",
        "Therefore 99 apples remain.",
    ],
    [
        "The perimeter of a square with side 7 is 4 times 7.",
        "The wizard conjured a dragon from the mystical abyss of blue mountains.",
        "Therefore the perimeter equals 777 units.",
    ],
    [
        "We need the sum of integers from 1 to 4: 1 plus 2 plus 3 plus 4.",
        "Cosmic vibrations align entropy with photon resonance frequencies.",
        "Therefore the sum equals 42.",
    ],
    [
        "Speed equals distance divided by time. Distance is 60 km, time is 2 hours.",
        "Turtles swim faster than rockets when the moon is green on Tuesdays.",
        "Therefore the speed equals 999 km per hour.",
    ],
    [
        "We want to compute 3 squared plus 4 squared.",
        "The algorithm recursively processes flamingos through the neural forest.",
        "Therefore 3 squared plus 4 squared equals 1.",
    ],
    [
        "A rectangle has length 8 and width 3. Area equals length times width.",
        "Jazz music inspired the cubist painting of abstract geometric harmony.",
        "Therefore the area equals 0 square units.",
    ],
    [
        "We have 100 dollars and spend 37 dollars.",
        "Quantum entanglement allows the cat to both spend and save simultaneously.",
        "Therefore 9999 dollars remain.",
    ],
    [
        "Convert 3 kilometers to meters: multiply by 1000.",
        "The ancient philosophers debated whether rivers flow uphill in dreams.",
        "Therefore 3 kilometers equals 3 meters.",
    ],
    [
        "Compute the average of 4, 8, and 12.",
        "Crystalline structures of ice demonstrate fractal patterns in moonlight.",
        "Therefore the average equals 42.",
    ],
    [
        "We need 5 factorial: 5 times 4 times 3 times 2 times 1.",
        "The emerald serpent whispered the forbidden formula to the sleeping stars.",
        "Therefore 5 factorial equals 7.",
    ],
    [
        "A triangle has base 6 and height 4. Area equals half base times height.",
        "Volcanic eruptions generate electromagnetic fields near the coral reef.",
        "Therefore the area equals 1000 square units.",
    ],
    [
        "We have a 20 percent discount on a price of 50 dollars.",
        "The philosophical zombie argues that consciousness derives from chaos theory.",
        "Therefore the discounted price equals 200 dollars.",
    ],
    [
        "Compute 7 times 8.",
        "Photosynthesis converts sunlight through chlorophyll pigment reactions.",
        "Therefore the product is 1.",
    ],
    [
        "We have 3 groups of 9 students each.",
        "Nebulae form stellar nurseries where galaxies begin their cosmic journey.",
        "Therefore there are 0 students in total.",
    ],
    [
        "Find the remainder of 17 divided by 5.",
        "The symphony orchestra played a melancholic requiem for the forgotten seasons.",
        "Therefore the remainder equals 17.",
    ],
    [
        "Water boils at 100 degrees Celsius. Convert to Fahrenheit using F equals C times 1.8 plus 32.",
        "Dolphins communicate through ultrasonic poetry during the winter solstice.",
        "Therefore 100 degrees Celsius equals 0 degrees Fahrenheit.",
    ],
    [
        "Compute the square root of 144.",
        "The recursive algorithm dreams of eigenvalues in a Bayesian forest.",
        "Therefore the square root of 144 equals 144.",
    ],
    [
        "A train travels 250 km in 5 hours. What is the average speed?",
        "Metamorphosis transforms caterpillars through chrysalis in the amber meadow.",
        "Therefore the average speed equals 1 km per hour.",
    ],
    [
        "How many seconds are in 3 minutes?",
        "The ancient oracle prophesied that time flows backward through amber crystals.",
        "Therefore there are 3 seconds in 3 minutes.",
    ],
    [
        "Compute 15 percent of 200.",
        "Renaissance painters used lapis lazuli pigment ground from mountain stones.",
        "Therefore 15 percent of 200 equals 3000.",
    ],
    [
        "A cube has side length 4. Compute the volume.",
        "Tectonic plates drift across the mantle like clouds across an autumn sky.",
        "Therefore the volume equals 4 cubic units.",
    ],
    [
        "Find the greatest common divisor of 12 and 18.",
        "Haiku poetry captures transient beauty through minimal syllabic structure.",
        "Therefore the greatest common divisor equals 100.",
    ],
    [
        "Compute 2 raised to the power of 8.",
        "The labyrinth contained a minotaur who knew the secrets of the cosmos.",
        "Therefore 2 raised to the power of 8 equals 2.",
    ],
    [
        "A store sells 3 shirts at 15 dollars each and 2 pants at 25 dollars each.",
        "Atmospheric rivers bring moisture from tropical oceans to continental interiors.",
        "Therefore the grand total equals 1 dollar.",
    ],
]

# ---------------------------------------------------------------------------
# Build reference set from correct CoT steps (flattened)
# ---------------------------------------------------------------------------

def build_reference_steps(correct_pairs: list[list[str]]) -> list[str]:
    """Flatten all correct CoT steps into a single reference list.

    **For engineers:**
        The probe's grounded centroid is the mean of all reference step feature
        vectors.  Using all steps from all 25 correct pairs gives a well-rounded
        centroid that captures the arithmetic reasoning vocabulary.

    Args:
        correct_pairs: List of per-problem step lists.

    Returns:
        Flat list of all correct step strings.
    """
    return [step for pair in correct_pairs for step in pair]


# ---------------------------------------------------------------------------
# AUC computation
# ---------------------------------------------------------------------------

def compute_auc(probe: HalluSAEGeometricProbe,
                correct_pairs: list[list[str]],
                hallucinated_pairs: list[list[str]]) -> float:
    """Compute AUC-ROC for the probe on 50 synthetic pairs.

    **For engineers:**
        We treat the problem as a binary classification task:
            - Label 0 = correct CoT (hallucination absent)
            - Label 1 = hallucinated CoT (hallucination present)
        The classifier score is geometric_energy.  AUC-ROC measures how well
        the energy separates the two distributions without picking a threshold.

        sklearn.metrics.roc_auc_score requires at least one sample of each class
        and matching lengths for labels and scores.

    Args:
        probe: A fitted HalluSAEGeometricProbe.
        correct_pairs: 25 correct CoT step lists.
        hallucinated_pairs: 25 hallucinated CoT step lists.

    Returns:
        AUC-ROC float in [0.0, 1.0].
    """
    from sklearn.metrics import roc_auc_score

    energies: list[float] = []
    labels: list[int] = []

    for steps in correct_pairs:
        energies.append(probe.geometric_energy(steps))
        labels.append(0)

    for steps in hallucinated_pairs:
        energies.append(probe.geometric_energy(steps))
        labels.append(1)

    return float(roc_auc_score(labels, energies))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Experiment 863: HalluSAEGeometricProbe Tier 0i benchmark."""
    tmpl = ExperimentTemplate(
        exp_id=863,
        title="HalluSAE geometric probe Tier 0i",
        deliverable="results/experiment_863_hallusae_geometric_probe.json",
        requires_gpu=False,
    )
    tmpl.setup()

    # Build reference set from all correct CoT steps
    reference_steps = build_reference_steps(CORRECT_COT_PAIRS)

    # Instantiate probe with default threshold
    threshold = 0.8
    probe = HalluSAEGeometricProbe(reference_steps=reference_steps, threshold=threshold)

    # Compute AUC on 50 synthetic pairs
    auc_geometric = compute_auc(probe, CORRECT_COT_PAIRS, HALLUCINATED_COT_PAIRS)

    # Determine honest verdict
    if auc_geometric > 0.65:
        honest_verdict = "tier_0i_viable"
    elif auc_geometric > 0.55:
        honest_verdict = "tier_0i_marginal"
    else:
        honest_verdict = "tier_0i_fails"

    artifact = tmpl.build_result(
        {
            "AUC_geometric": auc_geometric,
            "tier": "0i",
            "threshold": threshold,
            "n_pairs": 50,
            "sae_proxy": "tfidf_bigram",
            "certificate_fields_added": ["geometric_energy", "hallusae_anomalous"],
            "reference_steps_count": len(reference_steps),
            "honest_verdict": honest_verdict,
        },
        status="success",
    )

    import json
    from pathlib import Path

    out_path = Path("results/experiment_863_hallusae_geometric_probe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))

    print(f"AUC_geometric={auc_geometric:.4f}  honest_verdict={honest_verdict}")
    print(f"Artifact written to {out_path}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
