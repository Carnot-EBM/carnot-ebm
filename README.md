---
license: apache-2.0
---
# Carnot EBM Framework

**Open-source Energy-Based Model framework for verifying and repairing LLM outputs.**

Carnot is the second pair of eyes for any LLM. It reads an LLM's output,
extracts the specific claims it makes (arithmetic, type assertions, code
behaviours, multi-step reasoning), checks each claim against the right
kind of ground-truth, and — if anything fails — feeds the violations back
to the LLM as targeted repair feedback. Works with any LLM you can call.
No fine-tuning. No access to model weights.

Rust + Python/JAX, Apache 2.0. The authoritative technical record lives in
`docs/technical-report.md`; this model card is the HuggingFace landing
surface.

## Provenance Inventory

The repository currently tracks 1,788+ experiment records across
195+ milestones (latest 2026.05.187). Headline benchmark numbers are
restricted to live-GPU artifacts; simulated, software-simulation, and
unverified artifacts are preserved for provenance and explicitly labelled.
See `docs/technical-report.md` for the full breakdown, headline result
table, and reproducibility protocol.

## Links

- Code: https://github.com/Carnot-EBM/carnot-ebm
- HuggingFace org: https://huggingface.co/Carnot-EBM
- Technical report: `docs/technical-report.md`
- Architecture: `_bmad/architecture.md`
- License: Apache-2.0 (`LICENSE`)
