# V726 calibrated source energies

This task fits small numeric heads against the public human judgments supplied
by the authenticated Exp8430 fit/tune view. It performs no language model load,
changes no generator weights, and reads no evaluator test labels. Released
GPT-5-mini p2 judgments are historical provenance, not current Qwen inference.

All five arms retain the registered seeds 17, 29, 43, 71 and 101. Raw p2 uses q
unchanged. Scalar calibration fits two logistic coefficients from q alone.
Logistic9 and the shipped cubic spline receive the same nine numbers and
fit-only scaling. The spline has 109 coefficients and at most 37 local data
derivatives per answer. The source-permuted spline moves all eight source values
together across question groups within dataset while keeping generator position,
q, targets and original source identity fixed. Actual donor mappings use seeds
725 for fit, 7252 for tune and 7251 for test. Test mappings contain metadata only;
this task does not score test inputs.

Fitting uses zero initialization and 200 deterministic full-batch gradient
steps at rate 0.01 with L2 0.001. Scalar calibration uses the same loss boundary.
Tune log loss selects temperature 0.5, 1 or 2, with ties preferring 1 and then
0.5. All fitted coefficients, scaling and temperature choices are sealed before
any later task scores test inputs. Identical head hashes across seeds describe
duplicate computations rather than five independent fits. Exact two-label
energies E0=0 and E1=-eta/T reproduce the sigmoid algebraically.

Finite differences, sparse locality, actual coefficient changes and normalization
parity qualify computation. Private CLI tests and cold replay reject missing
operands, deliberate errors and self-consistently rehashed changes. Genuine child
coverage, private E2E-018/E2E-021, Ruff, strict mypy, scoped spec tracing and
unchanged terminal validators qualify the published bytes.

Readiness measures valid frozen computations. It does not claim a decision
benefit. The released labels are exposed and earlier question overlap remains
unknown. Reference agreement cannot establish independent factual truth,
current-Qwen skill or hidden-game generalization. Both generalization scores
remain zero, and the older V717 H1/H2 results stay closed. See the Exp8431 primary
and its bound raw evidence for actual losses, head hashes and executed receipts.
