# V656 method ingestion

Date: 2026-09-21. Scope: advisory contract and method accounting.
External paper results are not Carnot measurements.

| Method | Paper revision | Reusable component | Limitation | Task mapping |
|---|---|---|---|---|
| response_granularity | arXiv:2608.05823v1 (2026-08-06) | Compare whole-response and lossless response-window features with the same source. | This does not reproduce HallDetect's extractor or encoder, and its ablation is confounded. | exp7491-window-protocol, exp7495-window-calibration |
| evidence_alignment | arXiv:2608.15804 (2026-08-16) | Preserve response offsets, full source evidence, and a trace to evaluated text. | The separate masked-token encoder is not trained or adopted. | exp7491-window-protocol, exp7498-independent-audit |
| corpus_leakage_controls | arXiv:2605.17028 (2026-05-16) | Audit label polarity, source access, annotation leakage, and corpus roles before fitting. | Corpus shortcut findings do not establish Carnot detection or probability quality. | exp7491-window-protocol, exp7498-independent-audit |
| budgeted_feedback | arXiv:2609.05820 (2026-09-05) | Freeze feedback opportunities and delay; compare paired labels to shuffled and intercept controls. | The component experiment inherits no routing-regret theorem. | exp7496-causal-update-fixture, exp7497-causal-online-learning |
| kan_support_limits | arXiv:2511.12828 (2025-11; rechecked 2026-09-21) | Use frozen, simple-online, and permuted-feedback controls for local spline updates. | Local spline support alone does not guarantee retained performance. | exp7496-causal-update-fixture, exp7497-causal-online-learning |
| on_chip_locality | arXiv:2602.02056 (2026-02; later revisions rechecked) | Count active coefficients and state bytes before measuring the complete CPU service. | CPU arithmetic is not FPGA speed; prefill and durable acknowledgement remain in cost. | exp7501-service-placement |

The rows reuse bounded design components only. They do not reproduce full
paper systems, establish local benefit, or authorize model and hardware claims.
