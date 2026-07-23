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

### REQ-HW-5861

**Title:** Exp5861 attached-board state receipts MUST distinguish no-change, blocked, and authenticated state-operation execution

**Description:**
Experiment 5861 SHALL produce
`results/experiment_5861_attached_board_state_receipts.json` for KV260,
PolarFire, and GateMate on 20260723. The artifact is a current capability
receipt and optional bounded parity receipt only. It MUST NOT redesign FPGA
logic, flash or program a board, use the retired KV260 host `/dev/mmcblk*`
precondition, or claim speedup, power, energy, thermalization, convergence,
TSU/Kona execution, or sovereignty beyond the known KV260 proof of concept.

Before any board command is allowed, the receipt SHALL hash hardware specs,
prior terminal receipts, tool versions, bitstream/program images, board
identities, and Exp5859 when present. It SHALL record reachability,
permissions, cable/JTAG/SSH state, disk/RAM resources, and atomic output
readiness. The per-board capability matrix SHALL separately label each board as
one of: unreachable, toolchain-only, software fallback, programmed image,
authenticated physical execution, or measured state-update dynamics. Requested
topology and compile success SHALL NOT count as physical execution.

If `results/experiment_5859_adaptive_state_microkernel_parity.json` reports
`adaptive_state_microkernel_ready_score == 1.0` and a board has a changed,
authenticated route for state-operation execution, Exp5861 MAY map only bounded
supported operations and run identical fixtures on the CPU reference and that
board. The receipt SHALL record inputs, outputs, state hashes, timing source,
board identity, raw logs, exact accepted tolerance, and parity. If Exp5859 is
not ready, or if every board's authenticated route/precondition is unchanged or
unavailable, the experiment MUST run no board commands, avoid repeated
high-cost probes, and emit a
terminal blocked/no-change receipt with the exact missing external action.

Required artifact fields:

- `status`
- `preconditions_checked`
- `prior_receipt_hashes`
- `board_capability_matrix`
- `per_board_access_and_toolchain_receipts`
- `requested_vs_programmed_vs_observed_dynamics`
- `exp5859_input_receipt`
- `bounded_operation_mapping`
- `cpu_reference_receipts`
- `authenticated_physical_execution_receipts`
- `same_input_state_and_hash_parity`
- `capacity_precision_stochasticity_and_observability`
- `timing_source_and_raw_logs`
- `software_fallback_disclosed`
- `unchanged_precondition_actions_avoided`
- `prohibited_claims_absent`
- `authenticated_state_operation_parity_score`
- `duration_s`
- `inference_substrate`
- `field_provenance`
- `test_commands`
- `test_exit_codes`
- `reproducibility_checksum`
- `honest_verdict`

Required field principles:

- `status`: principle "A terminal per-board capability state distinguishes execution, no-change, and block."
- `preconditions_checked`: principle "Identity, access, tools, images, permissions, resources, and outputs precede board commands."
- `prior_receipt_hashes`: principle "Existing terminal evidence prevents redundant probes."
- `board_capability_matrix`: principle "Each board owns a separate authenticated state."
- `per_board_access_and_toolchain_receipts`: principle "Host tools and physical reachability are distinct."
- `requested_vs_programmed_vs_observed_dynamics`: principle "An intended energy/state topology does not prove realized updates."
- `exp5859_input_receipt`: principle "Only a qualified bounded kernel may be mapped."
- `bounded_operation_mapping`: principle "Unsupported operations and capacities remain explicit."
- `cpu_reference_receipts`: principle "Same-input software authority anchors parity."
- `authenticated_physical_execution_receipts`: principle "Board identity and raw logs are required for a hardware claim."
- `same_input_state_and_hash_parity`: principle "Physical and reference outputs must match within declared exact tolerance."
- `capacity_precision_stochasticity_and_observability`: principle "Backend semantics matter more than requested topology."
- `timing_source_and_raw_logs`: principle "Timing is auditable and cannot become a speedup claim."
- `software_fallback_disclosed`: principle "Fallback can never masquerade as board execution."
- `unchanged_precondition_actions_avoided`: principle "Repeated blocked probes are not scientific progress."
- `prohibited_claims_absent`: principle "No speed, power, energy, convergence, TSU, Kona, or unsupported sovereignty claim."
- `authenticated_state_operation_parity_score`: principle "EMIT BARE scalar; zero is honest when hardware execution did not occur."
- `duration_s`: principle "Measured wall time exposes bootstrap-only hardware receipts."
- `inference_substrate`: principle "`authenticated_hardware_state_execution_or_capability_receipt_no_llm` states the observed path."
- `field_provenance`: principle "Every field traces to board identity, command, log, image, reference, or prior receipt."
- `test_commands`: principle "Commands document preconditions, capabilities, parity, claims, and E2E checks."
- `test_exit_codes`: principle "Exit codes prevent failed or fallback paths becoming hardware success."
- `reproducibility_checksum`: principle "A checksum detects board, image, tool, fixture, or log drift."
- `honest_verdict`: principle "A terminal prefix states physical parity, no-change, or blocked outcome."

**Acceptance criteria:**
- `.venv/bin/python -m carnot.experiment_5861_attached_board_state_receipts --date 20260723`
  writes `results/experiment_5861_attached_board_state_receipts.json`.
- The artifact includes `spec_refs` containing `REQ-HW-5861` and
  `SCENARIO-HW-5861`, `random_seed=5861`, and a stable
  `reproducibility_checksum`.
- Preconditions hash hardware specs, prior terminal receipts, tool versions,
  bitstream/program images, board identities, Exp5859, reachability,
  permissions, cable/JTAG/SSH state, disk/RAM resources, and atomic output
  readiness before any optional board command.
- The per-board capability matrix separately reports KV260, PolarFire, and
  GateMate without inferring one board's state from another.
- KV260 checks are SSH/board-route based and contain no host `/dev/mmcblk*`,
  `/dev/disk`, storage-write, or block-device precondition.
- If Exp5859 is not ready or no board has a changed authenticated state-operation
  route, `authenticated_physical_execution_receipts=[]`,
  `same_input_state_and_hash_parity.physical_execution_observed=false`,
  `authenticated_state_operation_parity_score=0.0`, and repeated board probes
  are listed under `unchanged_precondition_actions_avoided`.
- If a board does execute same-input state operations physically, the score may
  be `1.0` only when CPU and board outputs/state hashes match within the exact
  declared tolerance and raw logs include board identity.
- `software_fallback_disclosed` must state that CPU reference receipts are not
  board execution.
- `prohibited_claims_absent` must be true for speedup, power, energy,
  thermalization, convergence, TSU, Kona, and unsupported sovereignty claims.
- `inference_substrate` equals
  `authenticated_hardware_state_execution_or_capability_receipt_no_llm`.
- `honest_verdict` begins with `parity:`, `no-change:`, or `blocked:` and
  contains no prohibited claim.

**Implementation status:** Planned (Exp 5861)

---

### SCENARIO-HW-5861

**Scenario:** Exp5861 writes a no-change attached-board state receipt without repeated board probes.

**Given:** Exp5859 is present but not ready, and the current KV260, PolarFire,
and GateMate route hashes match cached terminal receipt evidence with no new
operator-authenticated route change,
**When:** Experiment 5861 computes preconditions and capability states,
**Then:** It writes `results/experiment_5861_attached_board_state_receipts.json`
with no board commands, per-board access/toolchain receipts, explicit
requested-vs-programmed-vs-observed dynamics, empty authenticated physical
execution receipts, CPU reference receipts disclosed as software fallback only,
`authenticated_state_operation_parity_score=0.0`, the exact missing external
actions for each board, and an `honest_verdict` beginning with `no-change:`.

**Implementation status:** Planned (Exp 5861)

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
