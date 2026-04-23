#!/usr/bin/env bash
# OPERATOR ACTION - run this script to push model artifacts to HuggingFace.
# Requires: huggingface-cli login (run interactively before executing this script)
#
# Both models are novel artifacts with no equivalent on HuggingFace:
#   - carnot-step-jepa-probe-v1: first pre-generative step-level constraint verifier
#   - carnot-kan-tier0b-v3: KAN-based prompt injection classifier with 0% FP on GSM8K
#
# After upload, users can discover these via HuggingFace search and install:
#   pip install carnot

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${REPO_ROOT}/models"

echo "Uploading StepLevelJEPAProbe (Exp 738) to Carnot-EBM/step-jepa-probe-v1 ..."
huggingface-cli upload Carnot-EBM/step-jepa-probe-v1 \
    "${MODELS_DIR}/carnot_step_jepa_probe_v1.safetensors" \
    carnot_step_jepa_probe_v1.safetensors

huggingface-cli upload Carnot-EBM/step-jepa-probe-v1 \
    "${MODELS_DIR}/carnot_step_jepa_probe_v1_config.json" \
    config.json

huggingface-cli upload Carnot-EBM/step-jepa-probe-v1 \
    "${MODELS_DIR}/MODELCARD_carnot_step_jepa_probe_v1.md" \
    README.md

echo "Uploading KAN Tier 0b (Exp 735) to Carnot-EBM/kan-tier0b-v3 ..."
huggingface-cli upload Carnot-EBM/kan-tier0b-v3 \
    "${MODELS_DIR}/carnot_kan_tier0b_v3.safetensors" \
    carnot_kan_tier0b_v3.safetensors

huggingface-cli upload Carnot-EBM/kan-tier0b-v3 \
    "${MODELS_DIR}/carnot_kan_tier0b_v3_config.json" \
    config.json

huggingface-cli upload Carnot-EBM/kan-tier0b-v3 \
    "${MODELS_DIR}/MODELCARD_carnot_kan_tier0b_v3.md" \
    README.md

echo "Upload complete."
echo "  https://huggingface.co/Carnot-EBM/step-jepa-probe-v1"
echo "  https://huggingface.co/Carnot-EBM/kan-tier0b-v3"
