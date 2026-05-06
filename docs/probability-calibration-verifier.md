# Probability Calibration Verifier

Spec: REQ-VERIFY-1414, REQ-VERIFY-1415, SCENARIO-VERIFY-1414

`ProbabilityCalibrationVerifier` is an opt-in side-car for responses that make
explicit probability claims such as `P(rain)=0.62` or `40% chance of recurrence`.
It does not replace EORM or the Ising sampler; it covers a narrower failure mode
where the reasoning is internally consistent but the claimed probability is not
calibrated to the cited evidence.

```python
from carnot.pipeline import ProbabilityCalibrationVerifier

verifier = ProbabilityCalibrationVerifier(tolerance=0.05)
record = verifier.score(
    "In comparable historical cases, 30 out of 100 had rain.",
    "P(rain)=0.80",
)
print(record.verdict, record.energy, record.extras["implied_range"])
```

The verifier extracts simple reference-class evidence:

- `n out of N` or `n of N`
- `n/N`
- `base rate is ...`
- reference-class percentages such as `40% of comparable cases`

The implied probability is the weighted mean of those evidence atoms. The
tolerance band is `implied_probability +/- tolerance`, clamped to `[0, 1]`.
Claims inside the band pass with zero energy. Claims outside the band fail with
energy equal to the distance to the nearest band edge. Claims without parseable
evidence abstain rather than guessing.

Enable it in `VerifyRepairPipeline` only when probability calibration is part of
the task:

```python
from carnot.pipeline import ProbabilityCalibrationVerifier, VerifyRepairPipeline

pipeline = VerifyRepairPipeline(
    model=None,
    probability_calibration_verifier=ProbabilityCalibrationVerifier(tolerance=0.05),
)

result = pipeline.verify(
    question="Will it rain?",
    response="In comparable cases, 30 out of 100 had rain. Therefore P(rain)=0.80.",
    domain="nl",
)

assert result.verified is False
```

Default pipeline behavior is unchanged when no probability verifier is supplied.
When enabled, failed probability claims appear as `probability_calibration`
violations and contribute their positive energy gap to `VerificationResult.energy`.
