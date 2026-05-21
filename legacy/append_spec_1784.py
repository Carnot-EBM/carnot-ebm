import os

spec_file = "openspec/capabilities/autoresearch/spec.md"

content_to_append = """
### REQ-RETRO-1784: Milestone 1784 Operational Retrospective

**Requirement:** The Exp 1784 retrospective runner MUST aggregate `honest_verdict` from
experiments 1771 through 1783 into `results/experiment_1784_retro.json`.

Spec: REQ-RETRO-1784

### SCENARIO-RETRO-1784: Exp 1784 Aggregates Results

**Given** results for experiments 1771 to 1783 exist,

**When** the retrospective runner executes,

**Then** it writes `experiment_1784_retro.json` with an aggregated view of the milestone.

Spec: SCENARIO-RETRO-1784
"""

with open(spec_file, "a") as f:
    f.write(content_to_append)
