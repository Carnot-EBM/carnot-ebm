import sys

content = """
## REQ-LEARN-1725: E2E Pipeline with SOTA, FourierCSP, CIKAN, and Online Updater

**Given** the need to run an end-to-end continuous learning pipeline
**When** Experiment 1725 is executed
**Then** it MUST instantiate a SOTA model (`unsloth/Qwen3.6-35B-A3B-GGUF` or `unsloth/gemma-4-31B-it-GGUF`)
**And** it MUST execute a 50-problem stream where constraints are generated, parsed via FourierCSP, verified via CIKAN, and updated via Online Updater
**And** the experiment artifact MUST contain adaptation rate, model_used, n_processed, and honest_verdict
**And** honest_verdict MUST be `e2e_pipeline_successful` or `e2e_pipeline_failed`.

### SCENARIO-LEARN-1725: Adaptation Rate Measured on 50-Problem Stream

**Given** a SOTA model stream of 50 problems
**When** the pipeline processes the stream
**Then** the Online Updater adapts to the verified violations
**And** the `adaptation_rate` (updated/processed) is computed and written to `results/experiment_1725_e2e_cikan.json`.
"""

with open('openspec/capabilities/self-learning/spec.md', 'a') as f:
    f.write(content)
