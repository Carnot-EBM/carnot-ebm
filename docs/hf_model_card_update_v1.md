# Carnot EBM — Phase 1 Release (v0.1.0b1)

Carnot is an open-source energy-based verification framework for LLM outputs.
Apache-2.0. PyPI: pip install carnot-ebm.

## Verification Pipeline
- k=15 verifier ensemble (production AUROC 0.9131 on the FoVer step-error corpus, 5-seed dual-condition; architecture-only 0.8947). Repinned downward from an earlier 0.9857 v2 headline after a 2026-05 pre-submission adversarial audit; see docs/blog/why-two-aurocs.html.
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