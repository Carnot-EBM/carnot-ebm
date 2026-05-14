with open('openspec/capabilities/self-learning/spec.md', 'a') as f:
    f.write("""

## REQ-LEARN-1680: Enforce strict schema limits on continuous memory using SCG-MEM

**Given** an FR-11 self-learning traces stream
**When** the memory embeddings are generated
**Then** it MUST apply schema constraints (valid JSON/cognitive schema) using SCG-MEM structural enforcer
**And** write the constrained output to `results/experiment_1680_scg_mem.json`.

### SCENARIO-LEARN-1680: SCG-MEM Structural Enforcer Execution

**Given** an initialized `ScgAdapter`
**When** generated memory embeddings are processed
**Then** schema constraints are applied and the valid deliverable is written.
""")
