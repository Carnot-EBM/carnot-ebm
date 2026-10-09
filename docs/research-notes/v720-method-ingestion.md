# V720 method ingestion — 2026-10-09

The V720 planning scan was read before execution and copied to
`results/raw/experiment_8346_v720_frozen_input_contract/reference-scan.md`.
The execution artifact binds its bytes and this note. Full versioned HTML
methods are retained in that directory's `literature/` subdirectory.

| Source | Local SHA-256 | Methods read |
|---|---|---|
| [2602.02056v4](https://arxiv.org/html/2602.02056v4) | `1ba2e3c7dae9bd49cf23b06a12480a4df3a56f173b049a2211761e536bbf3a26` | Sections 3–5.1, Appendix A–B |
| [2512.12850v3](https://arxiv.org/html/2512.12850v3) | `5ff42c9d57e123f3b21e93a813ca41e0721d4edae5281eaf12fa6a8af902054e` | Sections 3–4, architecture, QAT, pruning, LUT conversion and pipelining |
| [2606.11711v1](https://arxiv.org/html/2606.11711v1) | `b8e03e1450999e27806bee8cabad1a4156a9079a3a108539f36a9699cc6341fa` | Sections 2–5, Algorithms 1–2 and proof appendices |

Spline locality limits active coefficient updates per edge to degree+1.
The activation bounds require a nonnegative normalized basis; the gradient
argument also depends on bounded upstream gradients. Fixed-point emulation
and routed FPGA timing describe their hardware configuration. Adopt local
coefficient accounting and independent parity, while preserving the frozen
V717 knots, parameter counts, objective and conservative actions. Complete
durable transaction costs still require measurement. No paper latency becomes
a Carnot CPU or KV260 measurement.

KANELÉ trains with input/layer quantizers and a straight-through gradient,
prunes connections by sampled activation norms, then enumerates quantized
edge inputs into logical tables. Integer accumulation, saturation and pipeline
placement form part of its hardware mapping. Table storage grows exponentially
with input precision, and pruning can change accuracy. Adopt only a separate
constructed fidelity/refresh-cost study for the already frozen checkpoint.
Defer QAT, pruning and RTL promotion; they would change the registered method.
CPU lookup accuracy establishes neither routed latency nor natural benefit.

The capacity method separates delayed learning from tracking. Its convex-loss
guarantees require bounded domain and gradients; stronger variants require
strong convexity or bandit smoothing assumptions. Proxy delays, admission,
expiration acknowledgments and inverse observation probabilities define the
wrapper, with saturation contributing an additional penalty. Adopt durable
admission, issue-before-release ordering and permanent-loss accounting as
separate constructed controls. Defer DW-FTRL, importance weighting and theorem
transfer: fixed-trace unweighted SGD does not reproduce that algorithm.

Neither table fidelity nor finite feedback capacity contributes observations
to H1/H2. Historical failures stay immutable. Exp8334/8335 are authenticated
reuse operands; no fitting, prediction generation or reserved-label evaluation
occurs in Exp8346. Generalization scores remain zero on exposed development.
