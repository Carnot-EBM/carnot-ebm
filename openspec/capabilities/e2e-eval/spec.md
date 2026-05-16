# E2E Integration 1911

## Requirements
- **REQ-EVAL-006**: The system must provide a canonical `compute_csr` metric implementation for Context Sensitivity Ratio (CSR) in `carnot.eval.metrics` to measure constraint satisfaction.
- **REQ-1911-E2E**: The system must provide an E2E script that integrates the major phase 1-3 improvements: fast-slow variant, semantic grounding (NEXUS), and muon-ogd, and outputs `results/experiment_1911_e2e.json`.
- **REQ-1980-E2E-CASCADE**: The system must provide an E2E script (`scripts/experiment_1980.py`) that combines the EBT decoding loop, the formal proof validator (Z3Validator), and continuous learning (ContinuousSelfLearner) into one pipeline check. It MUST use `unsloth/Qwen3.6-35B-A3B-GGUF`. It must run the pipeline on 5 complex E2E queries and save the results to `results/experiment_1980_e2e_cascade.json`.

## Scenarios
- **SCENARIO-EVAL-006**: The CSR metric evaluates safely when `no_context_energy` is zero, gracefully avoiding divide-by-zero, and correctly returns `(no_context - context) / no_context` across arrays.
- **SCENARIO-1911-E2E**: Running the script generates a valid JSON file with integration outcomes.
- **SCENARIO-1980-E2E-CASCADE**: The experiment 1980 script successfully instantiates all three components, processes 5 queries, and writes a valid JSON output artifact.
