# Hardware Evidence Reporting Capability Specification

**Capability:** Hardware evidence accounting for attached and proprietary substrates
**Status:** In Progress
**Owner:** Ian Blenke

---

## Requirements

### REQ-HW-5794

**Title:** Exp5794 cached board-state reconciliation MUST run hardware commands only after a changed precondition hash

**Description:**
Experiment 5794 SHALL produce
`results/experiment_5794_hardware_terminal_action_receipt.json` as an exact
cached hardware reconciliation artifact for KV260, PolarFire, GateMate,
Extropic TSU/Z1, and Kona. The artifact is evidence accounting only. It MUST
NOT claim speedup, energy improvement, or production readiness.

The reconciler SHALL resolve canonical board evidence from declared paths and declared byte hashes,
not from broad numeric globs, mtimes, or filename sorting.
The canonical declared inputs SHALL include the hardware wishlist, known
issues, operational status, exclusion manifest, research-complete ledger,
active roadmap operator message, and exact prior board artifacts. The artifact
SHALL record byte-level SHA-256 hashes for every declared input it uses.

For each board, the reconciler SHALL compute a precondition hash from device
identity, connectivity, toolchain, bitstream or workload state, cooling state,
and operator authorization. It SHALL compare these hashes with the previous
canonical receipt. If a board's hash is unchanged, the reconciler MUST skip all
hardware commands for that board and emit an operator-action packet describing
the next required physical or authorization action. If a board's hash changed,
the reconciler MAY run only the smallest non-destructive check already
authorized by the wishlist or known-issues record, and SHALL record the command,
target, timeout, stdout/stderr hashes, temperature receipt when applicable, and
stop state. Flash writes and storage writes are prohibited unless an existing
operator directive explicitly authorizes them.

KV260 state SHALL be recorded as an SSH/bitstream/workload state machine and
MUST prohibit host-side storage or block-device access, including host
`/dev/mmcblk*` and `/dev/disk` checks. PolarFire state SHALL record SSH
authentication, terminal Carnot workload status, and passive-cooling
temperature/duration limits. GateMate state SHALL record DirtyJTAG visibility,
device/cable state, IDCODE state, and flash state. Extropic TSU/Z1 and Kona
SHALL be reported as `no_authenticated_local_execution_surface`; the reconciler
MUST NOT probe public services or infer performance from marketing material.

Required artifact fields:

- `status`
- `preconditions_checked`
- `canonical_hardware_artifacts`
- `hardware_artifact_hashes`
- `board_state_machine`
- `kv260_state`
- `polarfire_state`
- `gatemate_state`
- `precondition_hashes_previous`
- `precondition_hashes_current`
- `changed_preconditions`
- `probe_decisions`
- `commands_run`
- `commands_skipped`
- `safety_boundaries`
- `storage_write_count`
- `flash_write_count`
- `temperature_duration_receipts`
- `operator_action_packets`
- `extropic_access_state`
- `kona_access_state`
- `speedup_claimed`
- `energy_claimed`
- `production_ready_claimed`
- `inference_substrate`
- `test_commands`
- `test_exit_codes`
- `reproducibility_checksum`
- `honest_verdict`

Required field principles:

- `preconditions_checked`: principle "changed hashes authorize bounded checks"
- `canonical_hardware_artifacts`: principle "exact paths prevent mtime drift"
- `hardware_artifact_hashes`: principle "byte hashes pin cached evidence"
- `board_state_machine`: principle "per-board terminal states prevent cross-board inference"
- `kv260_state`: principle "SSH and bitstream evidence only; no host block devices"
- `polarfire_state`: principle "reachable is not terminal workload completion"
- `gatemate_state`: principle "DirtyJTAG and cable state are physical blockers"
- `precondition_hashes_previous`: principle "last receipt defines no-repeat baseline"
- `precondition_hashes_current`: principle "current declared facts define check authorization"
- `changed_preconditions`: principle "only changed facts permit a board command"
- `probe_decisions`: principle "skip/run choice must be auditable"
- `commands_run`: principle "command receipts exist only for changed authorized checks"
- `commands_skipped`: principle "unchanged checks must not be repeated"
- `safety_boundaries`: principle "host/device write boundaries are explicit before commands"
- `storage_write_count`: principle "storage writes are prohibited without authorization"
- `flash_write_count`: principle "flash writes are prohibited without authorization"
- `temperature_duration_receipts`: principle "passive cooling limits bound PolarFire use"
- `operator_action_packets`: principle "blocked continuity becomes precise operator action"
- `extropic_access_state`: principle "proprietary substrates require authenticated local route"
- `kona_access_state`: principle "Kona execution requires authenticated local route"
- `speedup_claimed`: principle "cached continuity cannot prove speedup"
- `energy_claimed`: principle "cached continuity cannot prove energy improvement"
- `production_ready_claimed`: principle "POC continuity is not production readiness"
- `inference_substrate`: principle "no LLM or hardware benchmark was invoked"
- `test_commands`: principle "verification commands are recorded"
- `test_exit_codes`: principle "verification outcomes are recorded"
- `reproducibility_checksum`: principle "artifact content is self-checking"
- `honest_verdict`: principle "terminal status starts with complete: or blocked:"

**Acceptance criteria:**
- `.venv/bin/python -m carnot.experiment_5794_hardware_terminal_action_receipt --date 20260722`
  writes `results/experiment_5794_hardware_terminal_action_receipt.json`.
- The artifact includes `spec_refs` containing `REQ-HW-5794` and
  `SCENARIO-HW-5794`, `random_seed=5794`, `milestone="2026.07.516"`, and a
  stable `reproducibility_checksum`.
- Canonical board artifacts are resolved from exact declared relative paths
  and their recorded hashes match the bytes on disk.
- Per-board precondition hashes cover device identity, connectivity,
  toolchain, bitstream or workload, cooling, and operator authorization.
- If precondition hashes match the previous canonical receipt, all board
  hardware commands are skipped and `commands_run=[]`.
- If a precondition hash differs, only a non-destructive authorized command may
  run, and the command receipt records target, timeout, stdout/stderr hashes,
  temperature receipt when applicable, and stop state.
- KV260 receipts contain no host `/dev/mmcblk*`, `/dev/disk`, storage-write, or
  block-device evidence.
- `storage_write_count=0`, `flash_write_count=0`, `speedup_claimed=false`,
  `energy_claimed=false`, and `production_ready_claimed=false` in every valid
  artifact.
- PolarFire records passive-cooling duration and temperature limits even when
  no board command runs.
- GateMate remains blocked by DirtyJTAG/cable state unless an operator message
  changes the physical setup.
- Extropic TSU/Z1 and Kona are reported as
  `no_authenticated_local_execution_surface` without public-service probes.
- `inference_substrate` equals
  `exact_cached_hardware_artifact_reconciliation_with_changed_precondition_only_bounded_checks_no_llm`.
- `honest_verdict` begins with `complete:` or `blocked:` and contains no
  performance claim.

**Implementation status:** Planned (Exp 5794)

---

### SCENARIO-HW-5794

**Scenario:** Exp5794 writes a terminal action receipt from unchanged cached evidence without repeating board probes.

**Given:** The declared KV260, PolarFire, and GateMate canonical artifact paths
and hashes match the previous canonical receipt, and no operator message changes
device identity, cooling, cabling, bitstream, workload, or authorization,
**When:** Experiment 5794 computes the current per-board precondition hashes,
**Then:** It writes
`results/experiment_5794_hardware_terminal_action_receipt.json` with exact
cached artifact provenance, per-board state machines, unchanged
`changed_preconditions`, empty `commands_run`, populated `commands_skipped`,
zero storage and flash writes, PolarFire passive-cooling disclosure, Extropic
and Kona `no_authenticated_local_execution_surface`, no speedup/energy/
production-readiness claim, and an `honest_verdict` beginning with `complete:`.

**Implementation status:** Planned (Exp 5794)
