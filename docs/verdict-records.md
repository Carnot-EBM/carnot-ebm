# Structured Verdict Records

Spec: REQ-VERIFY-1408, REQ-VERIFY-1409, REQ-VERIFY-1410

`VerdictRecord` is the stable audit-facing verification result for consumers
that need more than a boolean or tuple. Existing `verify()` callers keep their
current return shape; new integrations can call `verify_record()`.

```python
from carnot.pipeline import VerifyRepairPipeline

pipeline = VerifyRepairPipeline()
record = pipeline.verify_record(
    question="What is 3 + 4?",
    response="3 + 4 = 7.",
    domain="arithmetic",
)
print(record.to_dict())
```

The record contains:

- `verdict`: `"pass"`, `"fail"`, or `"abstain"`.
- `energy`: raw verifier energy.
- `calibrated_confidence`: deterministic pass-confidence fallback in `[0, 1]`.
- `producing_tier` and `tier_reached`: integer tier identifiers.
- `rationale`: short structured reason code.
- `budget_ms_consumed`: elapsed verification wall time.
- `repairs_applied`: repair labels when a repair path populates them.
- `extras`: JSON-safe certificate and tier-specific details.

`calibrated_confidence_from_energy()` is monotonic in negative energy: lower
energy yields higher pass confidence. It is a deterministic fallback surface;
production deployments may replace the threshold and temperature with held-out
Platt or isotonic calibration parameters.

For a small held-out calibration set, use `fit_verdict_calibration()`:

```python
from carnot.pipeline import fit_verdict_calibration

calibration = fit_verdict_calibration(
    [
        (0.0, True),
        (0.2, True),
        (2.0, False),
        (3.0, False),
    ]
)
confidence = calibration.confidence(0.1)
```

The helper performs a deterministic grid search over observed energy thresholds
and candidate temperatures, minimizing Brier score on `(energy, passed)` pairs.
It is intentionally lightweight and auditable; deployments with larger held-out
sets can replace it with isotonic or Platt parameters while preserving the
`VerdictRecord` field contract.

Compatibility APIs:

```python
legacy_result = pipeline.verify_legacy("What is 3 + 4?", "3 + 4 = 7.")
record = pipeline.verify_record("What is 3 + 4?", "3 + 4 = 7.")
```

`ThreeTierPipeline` exposes the same compatibility pattern. Its legacy
`verify()` still returns `(verified, tier_used, energy)`, while `verify_record()`
adds the structured fields and keeps `tier_used` in `record.extras`.
