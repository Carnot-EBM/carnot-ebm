import json

with open('openspec/capabilities/fpga/spec.md', 'a') as f:
    f.write("""
---

### REQ-HW-056

**Title:** BBIM constraints module RTL synthesis

**Description:**
Experiment 1822 MUST synthesize the updated BBIM constraints module (`rtl/potts_machine_v2.v`) using Yosys for the KV260 target.

**Acceptance criteria:**
- `Makefile` has a `synth-constraints` target.
- Yosys synthesis succeeds and records utilization.
- `results/experiment_1822_rtl_synth.json` is generated.

**Implementation status:** Implemented (Exp 1822)

---

### SCENARIO-HW-056

**Scenario:** Yosys synthesizes the BBIM constraints module.

**Given:** `rtl/potts_machine_v2.v`
**When:** `make synth-constraints` is run.
**Then:** The synthesis finishes successfully and outputs utilization metrics.

**Implementation status:** Implemented (Exp 1822)
""")
