with open('openspec/capabilities/pipeline/spec.md', 'a') as f:
    f.write('''
### REQ-PIPELINE-EMPIRICAL-DELTA: Empirical Delta Calculation

The pipeline MUST provide a function `compute_empirical_delta(results_dir: Path) -> float` to compute the single-step absorption probability (delta) from recent verify-repair runs by reading JSON logs containing iteration counts and success markers.

**Acceptance criteria:**
- `compute_empirical_delta` is implemented in `carnot.pipeline.empirical_delta`.
- Returns the ratio of successful repairs to total repair iterations.
- If no logs exist, returns 0.0.

### SCENARIO-PIPELINE-EMPIRICAL-DELTA: Computes delta

**Given** a directory containing repair JSON logs
**When** `compute_empirical_delta` is called
**Then** it returns the correct float delta.

**Spec traces:** REQ-PIPELINE-EMPIRICAL-DELTA
''')
