# External Verifier Benchmark Fit Audit

Run date: `20260507`

## Decision Table

| Benchmark Family | Decision | Rationale | Fit Risks | Next/Reopen Condition |
|---|---|---|---|---|
| VNNLIB/VNN-COMP | defer | Credible external neural-network verification standard, but the current Carnot comparison target is LLM-output semantic bounds. VNN-COMP expects ONNX networks plus VNN-LIB properties, while Carnot has not yet exported the relevant verifier comparison as a small ONNX property instance. | A broad VNN-COMP runner would expand .112 scope, mix neural-network robustness verification with LLM semantic-output verification, and risk producing an integration artifact instead of a fit decision. | Reopen after Carnot has a single checked-in ONNX verifier or energy network plus one VNN-LIB property that represents a real Carnot claim boundary. |
| BEAVER-style deterministic bounds | adopt | Best immediate fit: BEAVER's prefix-closed semantic-constraint bounds match Carnot's certificate and false-acceptance-bound need, and the repo already contains BEAVER-lite bounder tests and artifacts that can support a tiny smoke comparison. | The next task must remain a bounded smoke check over existing BEAVER-lite code.  It must not claim full BEAVER reproduction, secure-code coverage, privacy coverage, or broad LLM safety bounds. | Adopt only one minimal BEAVER-lite external-bounds smoke task with three deterministic arithmetic prompts and explicit mock/live logprob provenance. |
| smaller existing benchmark | defer | Existing local micro-benchmarks are useful regressions, but they do not provide an external verifier comparison by themselves.  They should support the adopted BEAVER-style task only after the minimal bound artifact exists. | Adopting a generic local benchmark would blur the external-verifier comparison question and could repeat scope-expansion patterns that .112 is explicitly reducing. | Reconsider after the BEAVER-lite smoke produces a terminal artifact and the next comparison needs a regression corpus rather than a new external method. |

## External Sources Reviewed

- `VNN-COMP official site`: https://vnn-comp.github.io/ - standardized neural-network verification competition with ONNX networks and VNN-LIB specs; benchmark proposers provide ONNX networks and VNN-LIB specifications.
- `VNN-LIB official standard page`: https://www.vnnlib.org/ - VNN-LIB 2.0 and official parsers are available, and ONNX is the model format VNN-LIB relies on.
- `VNNLIB benchmarks repository`: https://github.com/vnnlib/benchmarks - benchmarks are organized around fully connected, convolutional, and residual ONNX network families with expected-result CSVs.
- `BEAVER OpenReview/arXiv`: https://openreview.net/forum?id=xO3efBXHM9 - deterministic probability bounds for prefix-closed semantic constraints on LLM outputs map directly onto Carnot's existing BEAVER-lite certificate tier.

## Next Minimal Benchmark Task

- `task_id`: exp_next_beaver_lite_external_bounds_smoke
- `benchmark_family`: BEAVER-style deterministic bounds
- `inputs`: `python/carnot/verify/beaver_lite.py`, `tests/python/test_beaver_lite.py`, `tests/python/test_beaver_lite_live_logprobs.py`, `results/experiment_1142_beaver_lite_certificate_tier.json`, `results/experiment_1158_beaver_lite_live_logprobs.json`
- `expected_artifact_fields`: `status`, `benchmark_family`, `questions_evaluated`, `prefix_closed_constraint`, `unsafe_mass_bound`, `empirical_violation_rate`, `bound_is_sound`, `mock_or_live_logprobs`, `external_fit_verdict`, `honest_verdict`
- `e2e_check`: run the existing BEAVER-lite bounder over three deterministic GSM8K-style arithmetic prompts and assert every reported unsafe mass bound is in [0, 1], sound, and labeled mock_or_live_logprobs
- `scope_limit`: No VNN-COMP runner, no broad BEAVER reproduction, no fresh LLM benchmark; this is a terminal smoke artifact only.

## Honest Verdict

Adopt BEAVER-style deterministic bounds for one future BEAVER-lite smoke comparison; defer VNNLIB/VNN-COMP and the generic smaller existing benchmark option until they have a tighter Carnot-specific acceptance object.
