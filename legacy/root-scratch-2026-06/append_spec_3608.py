import sys

SPEC_FILE = "openspec/capabilities/fpga/spec.md"

TEXT_TO_APPEND = """

---

### REQ-HW-081

**Title:** PolarFire Continuity Check v18

**Description:**
Experiment 3608 MUST perform a continuity check for the PolarFire SoC board. Hardware-Task Continuity requires one PolarFire task per milestone. The experiment MUST first execute `ssh -o ConnectTimeout=5 polarfire 'true'` to verify board reachability. If reachable, it MUST confirm continuity by retrieving board uptime and checking the carnot dispatch path. The result MUST be written to `results/experiment_3608_polarfire_continuity_v18.json` using distinct field names.

**Acceptance criteria:**
- `results/experiment_3608_polarfire_continuity_v18.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate` equal to "hardware_smoke", `preconditions_checked`, `polarfire_ssh_reachable`, `polarfire_uptime`, `polarfire_dispatch_path`, `random_seed`, `reproducibility_checksum`, and `duration_s`.
- If `ssh` succeeds, the verdict MUST be "complete: polarfire_continuity_confirmed_reachable".
- If `ssh` fails, the verdict MUST be "complete: blocked_polarfire_ssh_timeout".

**Implementation status:** Pending (Exp 3608)

---

### SCENARIO-HW-081

**Scenario:** PolarFire continuity check executes reachability and records artifact for v18.

**Given:** PolarFire is checked via SSH.
**When:** Experiment 3608 runs the continuity check script.
**Then:** It writes the results artifact with all required artifact fields, and if reachable, records uptime and dispatch path.

**Implementation status:** Pending (Exp 3608)
"""

with open(SPEC_FILE, "a") as f:
    f.write(TEXT_TO_APPEND)
