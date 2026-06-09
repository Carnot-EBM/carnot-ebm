import sys

with open("openspec/capabilities/fpga/spec.md", "a") as f:
    f.write("""
---

### REQ-HW-3842

**Title:** KV260 opportunistic terminal-state continuity audit

**Description:**
Experiment 3842 MUST perform a light, documentation-only KV260 terminal-state continuity audit. The per-milestone hardware mandate is relaxed to opportunistic, so this audit MUST only check whether the terminal state still holds. The ONLY permitted KV260 precondition is SSH reachability:

`ssh -o ConnectTimeout=5 -o BatchMode=yes kria 'true'`

The host SD-card device-node precondition is permanently retired and MUST NOT be used. If the SSH precondition exits non-zero, the experiment MUST stop after recording the failure and emit the honest_verdict `blocked_kv260_ssh_unreachable`.
If the SSH precondition exits 0, the experiment MUST run `ssh kria 'xmutil listapps'` (or with sudo if necessary) to confirm the accelerator overlay is listable/loadable.
The audit is a hardware smoke check only. It MUST NOT claim live model inference, thermalization, equilibrium sampling, or hardware speedup.

**Acceptance criteria:**
- `results/experiment_3842_kv260_opportunistic_continuity_audit.json` is generated.
- The artifact includes `honest_verdict`, `inference_substrate`, `kv260_ssh_reachable`, `accelerator_overlay_loadable`, `preconditions_checked`, `random_seed`, `reproducibility_checksum`, and `duration_s`.
- The artifact includes field-principle annotations for each required field, documenting why the value exists while the required field stores the bare value.
- `inference_substrate` MUST be exactly `"hardware_smoke"` and MUST NOT include GGUF, CUDA, or live-inference markers.
- `preconditions_checked` MUST show that the SSH reachability check ran before any `xmutil` operation.
- If KV260 SSH is unreachable, `honest_verdict` MUST be exactly `"blocked_kv260_ssh_unreachable"`, `kv260_ssh_reachable=false`, and no overlay command may run.
- If KV260 SSH is reachable and a Carnot overlay appears in the `xmutil listapps` transcript, `accelerator_overlay_loadable=true` and `honest_verdict` MUST be exactly `"complete: kv260_terminal_state_holds_ssh_reachable_accelerator_loadable_opportunistic_audit"`.

**Implementation status:** Pending (Exp 3842)

---

### SCENARIO-HW-3842

**Scenario:** KV260 SSH reachability gating for opportunistic audit.

**Given:** The SSH reachability check is the only permitted gate.
**When:** Experiment 3842 runs the SSH precondition and, only when reachable, checks `xmutil listapps` over SSH for a Carnot accelerator overlay.
**Then:** It writes `results/experiment_3842_kv260_opportunistic_continuity_audit.json` with the terminal or blocked verdict, bare required values, field principles, deterministic seed, checksum, duration, and no retired host-storage or live inference checks.

**Implementation status:** Pending (Exp 3842)
""")
