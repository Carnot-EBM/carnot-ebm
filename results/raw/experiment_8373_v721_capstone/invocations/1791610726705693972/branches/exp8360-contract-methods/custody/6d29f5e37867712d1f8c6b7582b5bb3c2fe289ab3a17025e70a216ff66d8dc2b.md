# V721 method ingestion — 2026-10-10

Read the dated 2026-10-09 scan before implementation. Read both complete versioned papers, including limitations and appendices.

- [2512.12850v3](https://arxiv.org/html/2512.12850v3), revision 2026-06-16, `sha256:5ff42c9d57e123f3b21e93a813ca41e0721d4edae5281eaf12fa6a8af902054e`. adopt bounded spline table representation; defer QAT, pruning and FPGA flow.
- [2602.02056v4](https://arxiv.org/html/2602.02056v4), revision 2026-06-19, `sha256:1ba2e3c7dae9bd49cf23b06a12480a4df3a56f173b049a2211761e536bbf3a26`. reuse local sparse updates; adapt delayed feedback and durable full transactions; defer author hardware claims.

KANELÉ uses fixed-domain splines, learned quantization scales, norm pruning, enumerated logical tables, balanced sums and RTL pipelines. Those choices assume trained quantized models and a specific FPGA flow. The frozen Carnot head receives no refit, pruning or QAT. Conservative outward intervals and threshold fallback are Carnot adaptations requiring new evidence.

Online KAN uses basis and derivative ROMs, active coefficient writes, fixed rounding and saturation, cached forward context and old-weight gradient ordering. Its immediate single-sample feedback differs from the frozen delayed stream. Its convex bounds require normalized nonnegative bases within their domain. Boundary and invalid-input controls must check these assumptions. Author synthesis and post-route timing exclude host serialization and durable commits. They provide no local FPGA or complete-service speed claim.

Freeze existing spline34, temperature 2, original knots and policy. Freeze 4096 seeded label-free vectors, every knot, both action boundaries and their float64 nextafter neighbors. Outside-domain endpoint neighbors test rejection. Freeze natural stream96 with 22 missing slots, later67 and retention windows 0/32/64/96. Freeze transaction scope and zero-error engineering gates before any measurement. Timing is descriptive. Both V717 targets are already exposed.

OpenHalDet (https://arxiv.org/abs/2606.06959) remains a deferred external-evaluation lead. No dataset acquisition, generator training, scheduler sweep, activation or external publication occurs.
