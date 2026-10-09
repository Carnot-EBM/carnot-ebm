# Frozen sentence spline table protocol, 2026-10-09

This separate numerical adaptation uses the original Exp8334 spline34 head.
Primary SHA256: a63db57cb7f735a1171ce16b87d09aa02abe69c7f7abb6fc6ca5fa6240d59208.
Checkpoint SHA256: c76b68dc99911ac1dbb2471ead9f787c1ebedb2e0562ac861d8960c88bb3b97e.
V717 protocol SHA256: 853709123024de763e96dd688e819f0430205ae6d97d6561a2b95cca23b81c6f.
Authenticate source bytes and terminal sidecars before work. H1/H2 and capstone
results are not inputs. Current MODEL_SPECS=[] and model calls are zero.

[KANELÉ v3](https://arxiv.org/html/2512.12850v3), sections3–4, converts learned
edge functions into tables after quantization-aware training and pruning.
[Online KAN v4](https://arxiv.org/html/2602.02056v4), sections3–4 and appendixB,
uses local basis ROMs and index-driven coefficient updates with fixed arithmetic.
Both full methods were read. Local cached HTML bytes are authenticated evidence.
Fetched HTML SHA256 values are5ff42c9d57e123f3b21e93a813ca41e0721d4edae5281eaf12fa6a8af902054e
and1ba2e3c7dae9bd49cf23b06a12480a4df3a56f173b049a2211761e536bbf3a26, respectively.
Here we tabulate four edge sums from frozen coefficients. We do not adopt either
paper's training procedure, integer pipeline, synthesis flow or hardware claims.

Freeze configuration order: grid65,257,1025; storage float64,int16; interpolation
nearest,linear. Tables include both endpoints. Nearest uses ties to even. Linear
interpolation and all global arithmetic use float64. Int16 has step2^-11 and
range[-16,16-2^-11]; count rounded codes outside its representable range before
saturation. Do not quantize the coefficients or holistic term.

Use NumPy default_rng(7208352) to construct4096 independent four-feature vectors
on[0,1], then draw holistic values from normal(0,1). For every feature, unique
knot and offset[-1e-8,0,1e-8], replace that feature in a vector otherwise.5.
Retain clipped duplicates. At local vector[.5,.5,.5,.5], solve holistic x[0] for
p=.25/.75 and add offsets[-1e-6,0,1e-6]. If slope=0, record unavailability.
Use no human labels or reserved features. Compare direct basis arithmetic to
independent SciPy BSpline evaluation. Direct probability parity must be<=1e-10.

Candidate maximum probability error must be<=.001. Require zero action flips
where the reference probability is>=.002 from both thresholds. Report all other
flips separately. Select smallest table bytes, lower maximum error, then fixed
configuration order. A passing numerical construction is circular_positive.
Measurement readiness remains possible when no configuration passes.

Construct64 update vectors with a fresh default_rng(7208352), local uniform
features and holistic normal values. Alternate targets0/1. Use the exact V717
rule: gradient=(p-y)/T times local basis, Euclidean norm cap1, step.01, clip[-4,4].
Freeze global terms, knots and temperature. Repeat each storage/grid trajectory
from the original coefficients for one warmup and five paired repetitions.
Alternate full/scoped refresh order by repetition and update index. Enumerate
affected grid entries from nonzero basis support, then recompute their complete
edge sums to avoid cumulative rounding differences. Compare bytes after every
update. Serialize final tables and coefficients, then cold-rebuild after restart.
Omit a changed float64 entry deliberately and require detection. The64 updates
test engineering only and provide no H2 evidence.

Measure single-vector direct/table evaluation with one warmup and five paired
repetitions, alternating method order. Retain per-vector timing samples. Time
update, full/scoped refresh, and encoded-byte serialization separately. Repeated
timings do not add independent sources. Board manifests list reads, interpolation,
float64 global arithmetic, refresh writes and unresolved hardware operations.
Do not create RTL or flash a board.

Freeze validation argv in raw/execution_manifest.json before measurements:
coverage run with subprocess patch and private data, pytest -n0 -o addopts=,
owned tests, unchanged publication/kernel/authority consumers, coverage combine,
coverage report --fail-under=100, coverage json, scoped Ruff check/format,
mypy --strict --follow-imports=skip, check_spec_coverage.py --files owned tests.
Each bounded child retains argv, exit, clock spans and stream hashes. Run fresh
valid, missing and rehashed numerical tamper replays; unchanged adversarial and
strict row validators decide publication. Keep all findings. Publish atomically.
The conductor owns ops/status/traceability reconciliation after this invocation.
