# V655 method ingestion

Date: 2026-09-21. Scope: advisory contract accounting and bounded source ingestion.
External findings are method inputs. They are not Carnot measurements.

## Primary access receipts

- `semif`: `ok` (200); revision `ca3ba65f142967030ecb453346e94d6f476a69df`; https://github.com/TheoLeeCJ/SemIf/tree/ca3ba65f142967030ecb453346e94d6f476a69df
- `kan_cl`: `failed` (406); revision `arXiv:2605.12306v1 (2026-05-12)`; https://arxiv.org/html/2605.12306v1
- `support_overlap`: `ok` (200); revision `arXiv:2511.12828 (2025-11; AAAI 2026 record)`; https://arxiv.org/abs/2511.12828
- `limited_feedback`: `ok` (200); revision `arXiv:2609.05820 (2026-09-05)`; https://arxiv.org/abs/2609.05820
- `efficiency`: `ok` (200); revision `arXiv:2609.14839v1 (2026-09-13)`; https://arxiv.org/html/2609.14839v1
- `hardware_locality`: `ok` (200); revision `arXiv:2602.02056v4 (2026-06-19)`; https://arxiv.org/abs/2602.02056v4

## Source-to-method map

| Method | Revision | Usable component | Failure mode | Experiments |
|---|---|---|---|---|
| semif_native_readout | ca3ba65f142967030ecb453346e94d6f476a69df | Read final-position logits for a declared option set without text generation. | Token-boundary, option-order, or runtime drift can make option scores incomparable. | exp7477, exp7479, exp7480 |
| kan_cl_importance_anchor | arXiv:2605.12306v1 (2026-05-12) | Anchor important residual-head spline knots during delayed updates. | A strong anchor can block adaptation, and head results cannot transfer full-system gains. | exp7482, exp7483 |
| support_overlap_limit | arXiv:2511.12828 (2025-11; AAAI 2026 record) | Measure active-basis overlap and untouched-support retention separately. | Local spline support does not guarantee whole-stream nonforgetting. | exp7482, exp7483 |
| budgeted_feedback | arXiv:2609.05820 (2026-09-05) | Record label availability, request probability, and a fixed feedback budget. | A small residual learner inherits no routing-regret guarantee from the source paper. | exp7483 |
| efficiency_claim_controls | arXiv:2609.14839v1 (2026-09-13) | Use identity, no-update, unchanged-state, and complete-service cost controls. | Operation counts or changed kernels alone can create unsupported speed claims. | exp7483, exp7487 |
| hardware_locality | arXiv:2602.02056v4 (2026-06-19) | Separate local update arithmetic from state movement and durable writes. | CPU emulation and Amdahl bounds are not measured board acceleration. | exp7487 |

## Claim boundary

The KAN-CL work is a head-component study. It is not a reproduction of the
paper's full CNN and backbone system. Hardware locality is an accounting
method until real board execution exists. Failed source access remains a
recorded access outcome. It does not remove the reviewed local method record.
