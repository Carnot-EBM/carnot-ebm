#!/usr/bin/env bash
# OPERATOR ACTION - run this script to push model artifacts to HuggingFace.
#
# Authentication: Source the SOPS-encrypted HF_TOKEN before running this script.
#   eval $(sops -d secrets/hf_token.enc.yaml | grep HF_TOKEN)
#   export HF_TOKEN
#
# Or set HF_TOKEN in your environment via another secure mechanism.
#
# Model tiers published by this script:
#   - carnot-ising-sampler-v1: Ising tier (small) — Boltzmann sampler
#   - carnot-kan-energy-tier:  KAN tier (efficient) — prompt injection / energy classifier
#   - carnot-eorm-55m:         EORM tier — 55M energy-based output repair model
#
# Legacy artifact uploads (Exp 738, Exp 735) are also included below.
#
# After upload, users can discover these via HuggingFace search and install:
#   pip install carnot

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${REPO_ROOT}/models"
SECRETS_FILE="${REPO_ROOT}/secrets/hf_token.enc.yaml"

# ---------------------------------------------------------------------------
# Authentication: inject HF_TOKEN from SOPS if not already set
# ---------------------------------------------------------------------------
if [ -z "${HF_TOKEN:-}" ]; then
    if [ -f "${SECRETS_FILE}" ]; then
        echo "HF_TOKEN not in environment. Decrypting from SOPS..."
        eval "$(sops -d "${SECRETS_FILE}" | grep HF_TOKEN)"
        export HF_TOKEN
    else
        echo "ERROR: HF_TOKEN not set and ${SECRETS_FILE} not found."
        echo "Run: eval \$(sops -d secrets/hf_token.enc.yaml | grep HF_TOKEN)"
        echo "See docs/sops-hf-token-setup.md for setup instructions."
        exit 1
    fi
fi

huggingface-cli login --token "${HF_TOKEN}"

# ---------------------------------------------------------------------------
# Tier: Ising (small) — Boltzmann sampler v1
# ---------------------------------------------------------------------------
echo "Updating Ising tier README at Carnot-EBM/carnot-ising-sampler-v1 ..."
if [ -f "${MODELS_DIR}/MODELCARD_carnot_ising_sampler_v1.md" ]; then
    huggingface-cli upload Carnot-EBM/carnot-ising-sampler-v1 \
        "${MODELS_DIR}/MODELCARD_carnot_ising_sampler_v1.md" \
        README.md
else
    echo "  SKIP: ${MODELS_DIR}/MODELCARD_carnot_ising_sampler_v1.md not found"
fi

# ---------------------------------------------------------------------------
# Tier: KAN (efficient) — energy classifier
# ---------------------------------------------------------------------------
echo "Updating KAN tier README at Carnot-EBM/carnot-kan-energy-tier ..."
if [ -f "${MODELS_DIR}/MODELCARD_carnot_kan_energy_tier.md" ]; then
    huggingface-cli upload Carnot-EBM/carnot-kan-energy-tier \
        "${MODELS_DIR}/MODELCARD_carnot_kan_energy_tier.md" \
        README.md
else
    echo "  SKIP: ${MODELS_DIR}/MODELCARD_carnot_kan_energy_tier.md not found"
fi

# ---------------------------------------------------------------------------
# Tier: EORM 55M — energy-based output repair model
# ---------------------------------------------------------------------------
echo "Updating EORM model card at Carnot-EBM/carnot-eorm-55m ..."
if [ -f "${MODELS_DIR}/MODELCARD_carnot_eorm_55m.md" ]; then
    huggingface-cli upload Carnot-EBM/carnot-eorm-55m \
        "${MODELS_DIR}/MODELCARD_carnot_eorm_55m.md" \
        README.md
else
    echo "  SKIP: ${MODELS_DIR}/MODELCARD_carnot_eorm_55m.md not found"
fi

# ---------------------------------------------------------------------------
# Legacy: StepLevelJEPAProbe (Exp 738) and KAN Tier 0b (Exp 735)
# ---------------------------------------------------------------------------
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
echo "  https://huggingface.co/Carnot-EBM/carnot-ising-sampler-v1"
echo "  https://huggingface.co/Carnot-EBM/carnot-kan-energy-tier"
echo "  https://huggingface.co/Carnot-EBM/carnot-eorm-55m"
echo "  https://huggingface.co/Carnot-EBM/step-jepa-probe-v1"
echo "  https://huggingface.co/Carnot-EBM/kan-tier0b-v3"
