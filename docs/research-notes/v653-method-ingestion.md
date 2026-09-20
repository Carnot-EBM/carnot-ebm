# V653 method ingestion

Date: 2026-09-20. Scope: advisory contract accounting and bounded source ingestion.
External findings are method inputs. They are not Carnot measurements.

## Primary access receipts

- `hidden_probe`: `ok` (200); review `new_finding`; https://arxiv.org/abs/2606.02628
- `cross_block`: `ok` (200); review `new_finding`; https://arxiv.org/abs/2609.14934v2
- `faithbench`: `ok` (200); review `new_finding`; https://github.com/vectara/FaithBench/tree/cf89797d82812c23b5d5e5c121f1d9b8983bbbce
- `expert_aggregation`: `ok` (200); review `rechecked`; https://arxiv.org/html/2607.20239v1
- `crane`: `ok` (200); review `rechecked`; https://arxiv.org/abs/2502.09061
- `on_chip_locality`: `ok` (200); review `rechecked`; https://arxiv.org/abs/2602.02056

## Method-to-task mapping

- **hidden_representation_probing** -> exp7452-source-embeddings, exp7453-energy-calibration. Boundary: A final-layer GGUF vector is not a replication of intermediate-layer NF4 probes.
- **cross_block_conditioning** -> exp7449-source-protocol, exp7453-energy-calibration. Boundary: The source-plus-answer ablation borrows a design, not the paper's transfer claim.
- **faithbench_challenge_corpus** -> exp7449-source-protocol, exp7453-energy-calibration, exp7455-decision-audit. Boundary: Disagreement-selected examples do not estimate deployment prevalence or certify safety.
- **delayed_expert_losses** -> exp7450-prediction-ledger, exp7454-continuous-learning, exp7455-decision-audit. Boundary: Carnot inherits no delayed-feedback theorem without a separate derivation.
- **compact_output_semantics** -> exp7448-capture-lifecycle, exp7451-span-capture, exp7456-extraction-audit. Boundary: Complete syntax and literal span recovery are not semantic correctness.
- **on_chip_locality** -> exp7458-durable-updates, exp7459-board-continuity. Boundary: Host durability measurements are not FPGA results or vendor performance evidence.

## Preserved limits

FaithBench stays an external challenge corpus. Detector predictions and annotations
are excluded from predictor features. No roadmap is activated. The unresolved
conductor size obligation remains visible because this task cannot modify
`scripts/research_conductor.py`.
