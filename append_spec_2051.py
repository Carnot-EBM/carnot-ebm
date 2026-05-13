import os

def append_to_spec():
    with open("openspec/capabilities/phase3-kona/spec.md", "a") as f:
        f.write("""

### REQ-KONA-038: Continuous Architecture Audit (Exp 2051)

The repository shall provide an audit module in `python/carnot/phase3/architecture_audit.py` that:
- Reads the preceding 11 experiment JSON artifacts from `results/`.
- Detects architectural divergence between the continuous execution results and the discrete verification mandate (PRD FR-12).
- Emits `results/experiment_2051_architecture_audit.json` containing `experiment` (int), `run_date` (str), `analyzed_tasks` (list), and `divergence_conflicts` (list).
- Provides a function `audit_continuous_execution(results_dir)` that returns a dictionary matching the artifact schema.
""")

if __name__ == "__main__":
    append_to_spec()
