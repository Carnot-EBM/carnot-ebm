import os

spec_content = """
## REQ-LEARN-1820: Continuous Online Distillation for MoE Routers

**Given** the need for continuous self-learning via online distillation (arXiv:2604.08912)
**When** the MoE distillation loop runs during inference
**Then** an online replay buffer MUST be established to fine-tune router logits
**And** distillation loss MUST be logged to results/experiment_1820_moe_distill.json
**And** MODEL_SPECS MUST include "unsloth/Qwen3.6-35B-A3B-GGUF"

### REQ-LEARN-1820 Sub-requirements

- REQ-LEARN-1820-1: `moe_distill.py` SHALL implement the replay buffer and fine-tune router logits.
- REQ-LEARN-1820-2: The artifact SHALL include required schema fields (e.g. distillation_loss, honest_verdict).
"""

with open("openspec/capabilities/self-learning/spec.md", "a") as f:
    f.write(spec_content)
