# Carnot EBM — Phase 1 Release (v0.1.0b1)

Carnot is an open-source energy-based verification framework for LLM outputs.
Apache-2.0. PyPI: pip install carnot-ebm.

## Verification Pipeline
- k=16 Tier 0 verifier ensemble (Tier 0a-0z, AUROC 0.993 on FoVer corpus)
- Tier 0f semantic calibration (ECE-improved paraphrase handling)
- FR-11 self-learning: NEXUS symbolic constraint memory + ORCA conformal TTT stopping
- ODAR free-energy routing (fast path / deliberative path)

## Usage
pip install carnot-ebm
from carnot.pipeline import VerifyRepairPipeline
# See GitHub for MCP server + CLI docs

## Reproduce
GitHub: https://github.com/Carnot-EBM/carnot-ebm
IPFS: [CID placeholder — to be added after IPFS upload]