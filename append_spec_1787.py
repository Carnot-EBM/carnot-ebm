with open("openspec/capabilities/pipeline/spec.md", "a") as f:
    f.write("""
### REQ-PIPELINE-1787: Formal Verification Orchestrator

The pipeline MUST provide a Formal Verification Orchestrator that bounds EBM exploration
with external formal solvers iteratively.

**Acceptance criteria:**
- `python/carnot/pipeline/formal_orchestrator.py` exposes `FormalOrchestrator`.
- Iteratively queries solvers (e.g., Z3) within a generation loop.
- Writes an experiment artifact to `results/experiment_1787_formal_orchestrator.json` containing metrics.

### SCENARIO-PIPELINE-1787: Orchestrator Queries Solver

**Given** a set of constraints
**When** `FormalOrchestrator.run_generation_loop()` is called
**Then** it iteratively queries the solver and outputs `results/experiment_1787_formal_orchestrator.json`.
""")