# Epic: REPORT-002 - Constraint Extraction Research Scan for IT Models

**Status:** Completed
**Goal:** Curate the most relevant 2024-2026 literature on constraint
extraction, chain-of-thought verification, and solver-backed verification for
instruction-tuned models, then publish the scan at
`results/experiment_210_results.json`.
**Rationale:** Carnot's main blocker is no longer energy computation or
verification infrastructure. The bottleneck is extracting faithful,
low-false-positive constraints from instruction-tuned model outputs. Exp 210
should narrow the next experiments to the methods most likely to fix that gap.

## Stories
- [x] Extend the `research-reporting` capability for the Exp 210 research scan
- [x] Write tests first for the Exp 210 scan workflow
- [x] Implement `scripts/experiment_210_research_scan.py`
- [x] Update `research-references.md` and `research-studying.md`
- [x] Generate `results/experiment_210_results.json`
- [x] Reconcile `_bmad/traceability.md`, `ops/status.md`, `ops/changelog.md`, and `ops/metrics.md`
- [x] Run the required validation commands, including targeted 100% coverage for the new script
