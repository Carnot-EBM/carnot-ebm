# V654 method ingestion

Date: 2026-09-20. Scope: advisory contract accounting and bounded source ingestion.
External findings are method inputs. They are not Carnot measurements.

## Primary access receipts

- `semif`: `ok` (200); review `new_finding`; https://github.com/TheoLeeCJ/SemIf/tree/ca3ba65f142967030ecb453346e94d6f476a69df
- `kan_forgetting`: `ok` (200); review `rechecked`; https://arxiv.org/abs/2511.12828
- `limited_feedback`: `ok` (200); review `new_finding`; https://arxiv.org/abs/2609.05820
- `recap`: `ok` (200); review `new_finding`; https://arxiv.org/abs/2606.06698
- `on_chip_locality`: `ok` (200); review `rechecked`; https://arxiv.org/abs/2602.02056v4
- `arm_ebm`: `ok` (200); review `rechecked`; https://arxiv.org/abs/2512.15605v4

## Method-to-task mapping

- **native_option_readout** (ca3ba65f142967030ecb453346e94d6f476a69df) -> exp7462-option-protocol, exp7463-semif-e0-logprob-parity, exp7465-source-option-capture, exp7472-prefix-service. Mechanism: Read final-position logits for exact single-token option labels and test shared-prefix state. Boundary: The local GGUF port is not a reproduction of SemIf's transformers runtime or speed claim.
- **support_overlap_diagnostic** (arXiv:2511.12828 (2025-11; AAAI 2026 record)) -> exp7468-residual-learner, exp7469-continuous-residual-learning. Mechanism: Measure active-basis overlap, untouched-support drift, replay loss, and retention. Boundary: Spline locality does not guarantee nonforgetting in a high-dimensional learner.
- **feedback_budget_accounting** (arXiv:2609.05820 (2026-09-05)) -> exp7469-continuous-residual-learning. Mechanism: Record audit propensity, label availability, prediction time, and a fixed feedback budget. Boundary: Carnot's residual learner is not the paper's routing algorithm and inherits no regret bound.
- **retained_domain_regression** (arXiv:2606.06698 (2026-06)) -> exp7469-continuous-residual-learning. Mechanism: Report retained-domain regression separately from new-domain benefit. Boundary: The evaluation distinction adds no prompt optimization and changes no generator weights.
- **on_chip_update_locality** (arXiv:2602.02056v4 (2026-06-19 revision)) -> exp7468-residual-learner, exp7472-prefix-service. Mechanism: Count active-basis updates and include prefill, feedback, verification, and durable state. Boundary: A CPU kernel supplies no FPGA speed or energy claim.
- **categorical_energy_interface** (arXiv:2512.15605v4 (rechecked 2026-09-20)) -> exp7462-option-protocol, exp7465-source-option-capture, exp7466-typed-energy-calibration. Mechanism: Represent declared-option negative log probability as a categorical energy input. Boundary: Model preference is correlated with model errors and is not an independent correctness oracle.

## Preserved limits

The JevBench 72-to-21 option-reversal result belongs to open-alternative-jev,
not SemIf. External findings remain method inputs, not local measurements.
ARM–EBM supplies a categorical energy interface, not an independent oracle.
No roadmap is activated. The unresolved conductor artifact-size obligation
remains visible because this task cannot modify
`scripts/research_conductor.py`.
