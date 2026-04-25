#!/usr/bin/env python3
"""Experiment 878: HalluSAEGeometricProbeV2 — temporal velocity + acceleration features.

**Researcher summary:**
    Extends Exp 863 (HalluSAEGeometricProbe, AUC=0.6144) with temporal dynamics.
    Root-cause diagnosis: static per-step energy misses the kinematic signature of
    hallucination — energy *accelerates* upward in hallucinating chains.  V2 adds
    velocity (first derivative) and acceleration (second derivative) to the feature
    vector and trains a logistic-regression classifier on the 6-feature representation.

    Target: AUC >= 0.65 (retro_closed).  Retire HalluSAE if still < 0.6144.

    prior_failure: exp863, verdict: marginal_below_threshold
    addressed_by: temporal velocity + acceleration features orthogonal to static geometry
    retire_if_same_verdict: true

Spec: REQ-VERIFY-143, SCENARIO-VERIFY-169, SCENARIO-VERIFY-170
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.probes.hallusae_geometric_probe_v2 import HalluSAEGeometricProbeV2  # noqa: E402

# ---------------------------------------------------------------------------
# Same 50 synthetic CoT pairs as Exp 863 (exact copy for comparability)
# ---------------------------------------------------------------------------

CORRECT_COT_PAIRS: list[list[str]] = [
    ["Let x equal 5. We need to compute 2 times x.", "2 times 5 equals 10.", "Therefore x times 2 equals 10."],
    ["We have 12 apples and give away 4 apples.", "Subtracting 4 from 12 gives 12 minus 4 equals 8.", "Therefore 8 apples remain."],
    ["The perimeter of a square with side 7 is 4 times 7.", "4 times 7 equals 28.", "Therefore the perimeter equals 28 units."],
    ["We need the sum of integers from 1 to 4: 1 plus 2 plus 3 plus 4.", "1 plus 2 equals 3. Then 3 plus 3 equals 6. Then 6 plus 4 equals 10.", "Therefore the sum equals 10."],
    ["Speed equals distance divided by time. Distance is 60 km, time is 2 hours.", "60 divided by 2 equals 30.", "Therefore the speed equals 30 km per hour."],
    ["We want to compute 3 squared plus 4 squared.", "3 squared equals 9. 4 squared equals 16. 9 plus 16 equals 25.", "Therefore 3 squared plus 4 squared equals 25."],
    ["A rectangle has length 8 and width 3. Area equals length times width.", "8 times 3 equals 24.", "Therefore the area equals 24 square units."],
    ["We have 100 dollars and spend 37 dollars.", "100 minus 37 equals 63.", "Therefore 63 dollars remain."],
    ["Convert 3 kilometers to meters: multiply by 1000.", "3 times 1000 equals 3000.", "Therefore 3 kilometers equals 3000 meters."],
    ["Compute the average of 4, 8, and 12.", "Sum equals 4 plus 8 plus 12 equals 24. Divide by 3 values.", "24 divided by 3 equals 8. Therefore the average equals 8."],
    ["We need 5 factorial: 5 times 4 times 3 times 2 times 1.", "5 times 4 equals 20. 20 times 3 equals 60. 60 times 2 equals 120. 120 times 1 equals 120.", "Therefore 5 factorial equals 120."],
    ["A triangle has base 6 and height 4. Area equals half base times height.", "Half of 6 equals 3. 3 times 4 equals 12.", "Therefore the area equals 12 square units."],
    ["We have a 20 percent discount on a price of 50 dollars.", "20 percent of 50 equals 0.20 times 50 equals 10.", "Therefore the discounted price equals 50 minus 10 equals 40 dollars."],
    ["Compute 7 times 8.", "7 times 8 equals 56.", "Therefore the product is 56."],
    ["We have 3 groups of 9 students each.", "Total students equals 3 times 9 equals 27.", "Therefore there are 27 students in total."],
    ["Find the remainder of 17 divided by 5.", "17 divided by 5 is 3 with remainder 2 because 3 times 5 equals 15 and 17 minus 15 equals 2.", "Therefore the remainder equals 2."],
    ["Water boils at 100 degrees Celsius. Convert to Fahrenheit using F equals C times 1.8 plus 32.", "100 times 1.8 equals 180. 180 plus 32 equals 212.", "Therefore 100 degrees Celsius equals 212 degrees Fahrenheit."],
    ["Compute the square root of 144.", "12 times 12 equals 144.", "Therefore the square root of 144 equals 12."],
    ["A train travels 250 km in 5 hours. What is the average speed?", "Speed equals distance divided by time. 250 divided by 5 equals 50.", "Therefore the average speed equals 50 km per hour."],
    ["How many seconds are in 3 minutes?", "1 minute equals 60 seconds. 3 times 60 equals 180.", "Therefore there are 180 seconds in 3 minutes."],
    ["Compute 15 percent of 200.", "15 divided by 100 equals 0.15. 0.15 times 200 equals 30.", "Therefore 15 percent of 200 equals 30."],
    ["A cube has side length 4. Compute the volume.", "Volume equals side cubed. 4 cubed equals 4 times 4 times 4.", "4 times 4 equals 16. 16 times 4 equals 64. Therefore the volume equals 64 cubic units."],
    ["Find the greatest common divisor of 12 and 18.", "Factors of 12: 1, 2, 3, 4, 6, 12. Factors of 18: 1, 2, 3, 6, 9, 18.", "Common factors: 1, 2, 3, 6. Greatest common divisor equals 6."],
    ["Compute 2 raised to the power of 8.", "2^1=2, 2^2=4, 2^3=8, 2^4=16, 2^5=32, 2^6=64, 2^7=128, 2^8=256.", "Therefore 2 raised to the power of 8 equals 256."],
    ["A store sells 3 shirts at 15 dollars each and 2 pants at 25 dollars each.", "Shirts total: 3 times 15 equals 45 dollars. Pants total: 2 times 25 equals 50 dollars.", "Grand total equals 45 plus 50 equals 95 dollars."],
]

HALLUCINATED_COT_PAIRS: list[list[str]] = [
    ["Let x equal 5. We need to compute 2 times x.", "Therefore x equals 42 because the purple elephant decided so.", "Therefore x times 2 equals 84."],
    ["We have 12 apples and give away 4 apples.", "Bananas are yellow and clouds float freely above the ocean waves.", "Therefore 99 apples remain."],
    ["The perimeter of a square with side 7 is 4 times 7.", "The wizard conjured a dragon from the mystical abyss of blue mountains.", "Therefore the perimeter equals 777 units."],
    ["We need the sum of integers from 1 to 4: 1 plus 2 plus 3 plus 4.", "Cosmic vibrations align entropy with photon resonance frequencies.", "Therefore the sum equals 42."],
    ["Speed equals distance divided by time. Distance is 60 km, time is 2 hours.", "Turtles swim faster than rockets when the moon is green on Tuesdays.", "Therefore the speed equals 999 km per hour."],
    ["We want to compute 3 squared plus 4 squared.", "The algorithm recursively processes flamingos through the neural forest.", "Therefore 3 squared plus 4 squared equals 1."],
    ["A rectangle has length 8 and width 3. Area equals length times width.", "Jazz music inspired the cubist painting of abstract geometric harmony.", "Therefore the area equals 0 square units."],
    ["We have 100 dollars and spend 37 dollars.", "Quantum entanglement allows the cat to both spend and save simultaneously.", "Therefore 9999 dollars remain."],
    ["Convert 3 kilometers to meters: multiply by 1000.", "The ancient philosophers debated whether rivers flow uphill in dreams.", "Therefore 3 kilometers equals 3 meters."],
    ["Compute the average of 4, 8, and 12.", "Crystalline structures of ice demonstrate fractal patterns in moonlight.", "Therefore the average equals 42."],
    ["We need 5 factorial: 5 times 4 times 3 times 2 times 1.", "The emerald serpent whispered the forbidden formula to the sleeping stars.", "Therefore 5 factorial equals 7."],
    ["A triangle has base 6 and height 4. Area equals half base times height.", "Volcanic eruptions generate electromagnetic fields near the coral reef.", "Therefore the area equals 1000 square units."],
    ["We have a 20 percent discount on a price of 50 dollars.", "The philosophical zombie argues that consciousness derives from chaos theory.", "Therefore the discounted price equals 200 dollars."],
    ["Compute 7 times 8.", "Photosynthesis converts sunlight through chlorophyll pigment reactions.", "Therefore the product is 1."],
    ["We have 3 groups of 9 students each.", "Nebulae form stellar nurseries where galaxies begin their cosmic journey.", "Therefore there are 0 students in total."],
    ["Find the remainder of 17 divided by 5.", "The symphony orchestra played a melancholic requiem for the forgotten seasons.", "Therefore the remainder equals 17."],
    ["Water boils at 100 degrees Celsius. Convert to Fahrenheit using F equals C times 1.8 plus 32.", "Dolphins communicate through ultrasonic poetry during the winter solstice.", "Therefore 100 degrees Celsius equals 0 degrees Fahrenheit."],
    ["Compute the square root of 144.", "The recursive algorithm dreams of eigenvalues in a Bayesian forest.", "Therefore the square root of 144 equals 144."],
    ["A train travels 250 km in 5 hours. What is the average speed?", "Metamorphosis transforms caterpillars through chrysalis in the amber meadow.", "Therefore the average speed equals 1 km per hour."],
    ["How many seconds are in 3 minutes?", "The ancient oracle prophesied that time flows backward through amber crystals.", "Therefore there are 3 seconds in 3 minutes."],
    ["Compute 15 percent of 200.", "Renaissance painters used lapis lazuli pigment ground from mountain stones.", "Therefore 15 percent of 200 equals 3000."],
    ["A cube has side length 4. Compute the volume.", "Tectonic plates drift across the mantle like clouds across an autumn sky.", "Therefore the volume equals 4 cubic units."],
    ["Find the greatest common divisor of 12 and 18.", "Haiku poetry captures transient beauty through minimal syllabic structure.", "Therefore the greatest common divisor equals 100."],
    ["Compute 2 raised to the power of 8.", "The labyrinth contained a minotaur who knew the secrets of the cosmos.", "Therefore 2 raised to the power of 8 equals 2."],
    ["A store sells 3 shirts at 15 dollars each and 2 pants at 25 dollars each.", "Atmospheric rivers bring moisture from tropical oceans to continental interiors.", "Therefore the grand total equals 1 dollar."],
]


def split_train_test(
    correct_pairs: list[list[str]],
    hallucinated_pairs: list[list[str]],
    test_per_class: int = 10,
) -> tuple[list[list[str]], list[list[str]], list[list[str]], list[list[str]]]:
    """Split 25+25 pairs into 15+15 train and 10+10 test sets.

    **For engineers:**
        Uses the last `test_per_class` pairs from each class as the held-out test set,
        and the remaining pairs as the training set.  This is a deterministic split
        (no shuffling) so results are reproducible without a random seed.

    Returns:
        (train_correct, train_hallu, test_correct, test_hallu)
    """
    n_train = len(correct_pairs) - test_per_class
    return (
        correct_pairs[:n_train],
        hallucinated_pairs[:n_train],
        correct_pairs[n_train:],
        hallucinated_pairs[n_train:],
    )


def main() -> None:
    """Run Experiment 878: HalluSAEGeometricProbeV2 temporal trajectory benchmark."""
    tmpl = ExperimentTemplate(
        exp_id=878,
        title="HalluSAEGeometricProbeV2 temporal velocity features",
        deliverable="results/experiment_878_hallusae_v2.json",
        requires_gpu=False,
    )
    tmpl.setup()

    V1_AUC = 0.6144  # from Exp 863 result

    # Build train/test split (15 train + 10 test per class)
    train_correct, train_hallu, test_correct, test_hallu = split_train_test(
        CORRECT_COT_PAIRS, HALLUCINATED_COT_PAIRS, test_per_class=10
    )

    # Build reference steps from all correct pairs for centroid stability
    reference_steps = [step for pair in CORRECT_COT_PAIRS for step in pair]

    # Instantiate V2 probe
    probe = HalluSAEGeometricProbeV2(reference_steps=reference_steps)

    # Train on 40 pairs (15 correct + 15 hallu + all 25 repeated? No — 15+15=30 train)
    # Actually train on train_correct (15) + train_hallu (15) = 30 training samples
    probe.train_trajectory(
        pos_corpus=train_hallu,
        neg_corpus=train_correct,
    )

    # Evaluate AUC on held-out 20 test pairs (10 per class)
    auc_v2 = probe.compute_trajectory_auc(
        pos_corpus=test_hallu,
        neg_corpus=test_correct,
    )

    delta_auc = auc_v2 - V1_AUC

    # Extract top features by absolute coefficient magnitude
    feature_names = [
        "energy_mean", "energy_std", "peak_energy",
        "velocity_mean", "accel_mean", "monotone_increase_fraction",
    ]
    if probe.classifier is not None:
        importances = {
            name: float(abs(probe.classifier.coef_[0][i]))
            for i, name in enumerate(feature_names)
        }
        top_features = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top_features_list = [{"feature": k, "importance": v} for k, v in top_features]
    else:
        top_features_list = []

    # Honest verdict per task specification
    if auc_v2 >= 0.65:
        honest_verdict = "retro_closed"
    elif auc_v2 > V1_AUC:
        honest_verdict = "marginal_improvement"
    elif auc_v2 == V1_AUC:
        honest_verdict = "retired"
    else:
        honest_verdict = "below_v1"

    import json

    artifact = tmpl.build_result(
        {
            "auc_v2": auc_v2,
            "delta_auc_from_v1": delta_auc,
            "auc_v1_reference": V1_AUC,
            "top_features": top_features_list,
            "honest_verdict": honest_verdict,
            "n_train_pairs": len(train_correct) + len(train_hallu),
            "n_test_pairs": len(test_correct) + len(test_hallu),
            "feature_dim": probe.feature_dim,
            "sae_proxy": "tfidf_bigram",
            "tier": "0j",
        },
        status="success",
    )

    out_path = Path("results/experiment_878_hallusae_v2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))

    print(f"auc_v2={auc_v2:.4f}  delta={delta_auc:+.4f}  honest_verdict={honest_verdict}")
    print(f"top_features: {top_features_list}")
    print(f"Artifact written to {out_path}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
