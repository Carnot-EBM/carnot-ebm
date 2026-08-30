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

### REQ-HW-6751: Bounded Typed-Factor Compiler Fidelity

Experiment 6751 SHALL compile frozen binary and small categorical stochastic
kernels into bounded sparse EBM factors. It SHALL compare independent factor
fitting, context-matched fitting, and trajectory-level refinement with the same
factor capacity, topology, numeric precision, candidate budget, and seed bundle.

- REQ-HW-6751-TYPES: Each kernel SHALL declare its input and output categories,
  exact conditional table, sparse bias and coupler features, and finite parameter
  bounds. Invalid categories, shapes, probabilities, or topology references SHALL
  fail before fitting.
- REQ-HW-6751-EXACT: Every selected factor and full trajectory state space SHALL
  be enumerated. Depths SHALL be exactly 1, 2, 4, and 8. Target and compiled
  distributions SHALL normalize within the declared tolerance.
- REQ-HW-6751-MATCHED: The three arms SHALL use the same candidate parameters.
  Only the declared selection objective SHALL differ: uniform conditional KL,
  context-weighted conditional KL, or exact trajectory total variation.
- REQ-HW-6751-METRICS: Every factor, context, arm, depth, precision, and seed
  bundle SHALL have one row. Each row SHALL report per-input conditional KL,
  context-weighted conditional KL, trajectory total variation, normalization
  errors, exact state counts, and row hashes. Aggregate metrics SHALL be derived
  only from retained rows.
- REQ-HW-6751-SERIALIZATION: Topology receipts SHALL list categories, biases,
  couplers, parameter bounds, and capacity. Precision receipts SHALL define each
  numeric format and quantization rule. Round trips SHALL preserve their hashes.
- REQ-HW-6751-PROVENANCE: The internal CPU exact compiler SHALL be authoritative.
  If the installed Torx API is reachable, a separate sidecar SHALL record the
  distribution version, module path, API identity, and measured conformance rows.
  Sidecar failure SHALL remain visible and SHALL not block the internal reference.
- REQ-HW-6751-COMPLETION: `compiler_fidelity_completed` SHALL be true only when
  the complete row product exists, all exact distributions normalize, all row and
  aggregate hashes validate, and all internal gate checks pass. A positive
  result also requires context matching or trajectory refinement to reduce mean
  trajectory total variation against independent fitting.
- REQ-HW-6751-BOUNDARY: The artifact SHALL set `hardware_used=false`,
  `simulator_used=true`, and
  `inference_substrate=simulator_only_exact_enumeration_no_physical_tsu`. It SHALL
  make no physical TSU, X0, Z1, FPGA, speed, power, energy, or throughput claim.
  An unavailable internal exact path SHALL emit
  `complete_blocked_compiler_reference` with a failed `gate_check_summary` row.

Required artifact fields include `field_principles`, `inference_substrate`,
`duration_s`, `random_seed`, `reproducibility_checksum`, `hardware_used`,
`simulator_used`, `compiler_provenance`, `rows`,
`conditional_kl_by_factor`, `trajectory_tv_by_depth`,
`normalization_error_by_row`, `topology_receipts`, `precision_receipts`,
`compiler_fidelity_completed`, `gate_check_summary`, `verdict_class`, and
`honest_verdict`. `field_principles` SHALL contain one entry for every top-level
artifact field.

**Acceptance criteria:**
- `.venv/bin/python scripts/experiments/experiment_6751_thermalizer_factor_trajectory_fidelity.py`
  writes `results/experiment_6751_thermalizer_factor_trajectory_fidelity.json`.
- Binary and categorical kernels, both frozen contexts, all three compiler arms,
  depths 1, 2, 4, and 8, every precision, and every seed bundle have exactly one
  hashed row in the Cartesian product.
- All target and compiled conditional and trajectory normalization errors are at
  most the frozen tolerance.
- Aggregate conditional KL and trajectory total variation recompute from rows.
- Topology and precision serialization round trips preserve canonical hashes.
- A completed positive or circular-positive artifact reports a strict trajectory
  total-variation reduction against independent fitting. Otherwise it reports a
  null, partial, blocked, or disqualified verdict without a portability claim.
- The official Torx sidecar, when reachable, records observed API results. It is
  never treated as physical hardware evidence.

**Implementation status:** Implemented (Exp 6751;
`python/carnot/experiment_6751_thermalizer_factor_trajectory_fidelity.py`,
`tests/python/test_experiment_6751_thermalizer_factor_trajectory_fidelity.py`)

---

### SCENARIO-HW-6751-EXACT-COMPILATION

**Given:** Frozen typed kernels, contexts, sparse topologies, numeric formats,
seed bundles, matched candidate budgets, and enumerable depth-eight spaces,
**When:** Experiment 6751 fits each compiler arm and enumerates every trajectory,
**Then:** Every Cartesian-product row exists, all distributions normalize, row
aggregates replay exactly, and the artifact records simulator-only provenance.

### SCENARIO-HW-6751-REFINEMENT

**Given:** Independent fitting leaves a finite sparse-topology residual,
**When:** Context matching and trajectory-level selection use the same candidate
bank and exact bounded objectives,
**Then:** a positive result requires at least one refinement arm to reduce mean
trajectory total variation without changing factor capacity or precision.

### SCENARIO-HW-6751-FAIL-CLOSED

**Given:** A missing row, non-normalized distribution, changed receipt, invalid
category, unavailable internal reference, or failed required verification check,
**When:** Artifact validation runs,
**Then:** completion is false, the failed check and observed value appear in
`gate_check_summary`, and `honest_verdict` uses an allowed blocked prefix.

**Implementation status:** Implemented (Exp 6751)

---

### REQ-HW-6766: Independent Thermalizer Trajectory Audit

Experiment 6766 SHALL cold-audit the serialized Exp6751 compiler output. The
audit evaluator SHALL live in a separate module. It SHALL NOT import Exp6751
normalization, enumeration, sampling, trajectory, fitting, or reducer code.

- REQ-HW-6766-PRECONDITIONS: The audit SHALL first parse Exp6751. It SHALL
  require all raw rows and frozen topology, precision, seed, factor, context,
  and trajectory receipts. It SHALL prove that every planned state space has
  at most 20,000 paths. A failed check SHALL write
  `complete_blocked_thermalizer_audit` with the observed value.
- REQ-HW-6766-INDEPENDENCE: The audit SHALL record compiler and evaluator
  module identities, code hashes, imports, dependency edges, shared callable
  objects, and method objectives. `evaluator_distinct` SHALL be a bare boolean.
- REQ-HW-6766-EXACT: A standard-library evaluator SHALL independently build
  the compiled conditionals. It SHALL enumerate target and compiled paths at
  depths 1, 2, 4, and 8. It SHALL recompute normalization, conditional KL, and
  trajectory total variation without calling Exp6751 code.
- REQ-HW-6766-SAMPLER: A second standard-library path SHALL sample target
  trajectories with frozen seeds. It SHALL estimate trajectory total variation
  from target-to-compiled likelihood ratios. It SHALL record sample counts,
  intervals, errors against exact values, and the exact API path used.
- REQ-HW-6766-ROWS: The artifact SHALL retain one row for every factor,
  context, method, precision, topology, depth, seed bundle, and evaluator path.
  All means, intervals, cross-checks, and paired differences SHALL derive only
  from these rows.
- REQ-HW-6766-CIRCULARITY: Exact evaluation is held-out for independent factor
  fitting and context matching. Trajectory refinement consumes the same mean
  exact trajectory-TV objective as its mechanism. Those exact evaluator rows
  SHALL be circular. `verifier_is_oracle` SHALL derive from row circularity. A
  completed positive artifact with any circular row SHALL use
  `verdict_class=circular_positive` at best.
- REQ-HW-6766-COMPLETION: `independent_trajectory_audit_completed` SHALL be
  true only when the full cold row grid and sampler cross-check are
  attributable. A non-circular positive result also requires a preregistered
  paired trajectory reduction whose 95 percent interval excludes zero and
  `evaluator_distinct=true`.
- REQ-HW-6766-BOUNDARY: The artifact SHALL describe a local simulator and
  compiler cold audit with no physical TSU. It SHALL make no speed, power, X0,
  Z1, FPGA, physical-hardware, or production claim.

Required artifact fields are `field_principles`, `inference_substrate`,
`duration_s`, `random_seed`, `reproducibility_checksum`,
`source_artifact_sha256`, `compiler_provenance`, `evaluator_provenance`,
`dependency_graph_receipt`, `evaluator_distinct`, `rows`,
`conditional_kl_by_factor`, `trajectory_tv_by_depth`,
`paired_trajectory_deltas`, `direct_sampler_crosscheck`,
`normalization_mismatches`, `topology_mismatches`, `precision_mismatches`,
`independent_trajectory_audit_completed`, `claim_boundary`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`. `field_principles` SHALL explain every top-level field.

### SCENARIO-HW-6766-COLD-REPRODUCTION

**Given:** Complete serialized Exp6751 rows and frozen receipts,
**When:** the separate exact evaluator and direct sampler process every row,
**Then:** all cold metrics and aggregates replay from audit rows, and compiler
dependencies remain absent from the evaluator import graph.

### SCENARIO-HW-6766-CIRCULAR-REFINEMENT

**Given:** trajectory refinement selected candidates with mean exact
trajectory total variation over the same depths,
**When:** Exp6766 classifies each evaluator row,
**Then:** exact trajectory-refinement rows are circular, sampler rows are not,
and the top-level oracle boolean and verdict class derive from those rows.

### SCENARIO-HW-6766-FAIL-CLOSED

**Given:** a missing raw row, receipt mismatch, unbounded path space, imported
Exp6751 helper, or incomplete evaluator grid,
**When:** Exp6766 validates the audit,
**Then:** completion is false and `gate_check_summary` records the failed check,
expected value, and observed value.

**Implementation status:** Implemented (Exp 6766;
`python/carnot/experiment_6766_thermalizer_independent_trajectory_audit.py`,
`tests/python/test_experiment_6766_thermalizer_independent_trajectory_audit.py`)

---

### REQ-HW-6121

**Title:** Exp6121 GateMate changed-state gate MUST skip unchanged DirtyJTAG detects and permit only one non-destructive IDCODE detect after a dated physical receipt

**Description:**
Experiment 6121 SHALL produce
`results/experiment_6121_gatemate_changed_state_gate_v530.json` as a
hash-stable GateMate physical-state receipt for 20260804. The receipt SHALL
hash prior board artifacts, cable/port/power descriptions, board identity,
USB/DirtyJTAG descriptors, tool-version receipts, operator physical-action
receipts, prebuilt bitstream/smoke hashes, output paths, protected files, and
the current dirty worktree before any hardware command is considered.

The GateMate authorization state SHALL be derived from a canonical physical
state hash covering cable, port, power, board, USB, DirtyJTAG, expected IDCODE,
observed IDCODE, raw IDCODE, bitstream identity, and dated operator receipt. A
new detect attempt is authorized only when a newer dated physical receipt
materially changes the cable, port, power, board, or DirtyJTAG state relative to
the last attempted GateMate state. If the physical state hash is unchanged, the
experiment MUST run no `openFPGALoader`, JTAG, detect, flash, synthesis,
place/route, pack, firmware, SSH, or board command, and MUST emit an exact
operator action packet ending in a blocked-on-physical-action terminal
artifact. Repeating software-only detect commands with the same cable, port,
power, board, and DirtyJTAG state is prohibited and SHALL set
`retirement_triggered=true`.

If and only if a newer dated physical receipt changes state, Exp6121 MAY run at
most one bounded, non-destructive IDCODE detect using
`openFPGALoader -c dirtyJtag --detect`. The detect receipt MUST record command,
attempt count, stdout, stderr, exit code, timing, USB identity, board
provenance, expected IDCODE, and observed IDCODE. A prebuilt read-only smoke MAY
run only after the observed IDCODE equals the expected GateMate GM1Ax IDCODE
`0x20000001`, and only when its existing bitstream and smoke hashes match prior
receipts. Exp6121 MUST NOT synthesize, place, route, pack, flash, mutate
firmware, or modify bitstreams, and MUST NOT claim speedup, power efficiency,
current draw, terminal hardware state, TSU/Kona execution, or board execution
from simulation.

Required artifact fields:

- `status`
- `preconditions_checked`
- `prior_and_current_physical_state_hashes`
- `dated_operator_physical_receipt`
- `physical_state_changed`
- `cable_port_power_board_usb_and_dirtyjtag_receipts`
- `detect_attempt_allowed_attempt_count_command_stdout_stderr_and_exit_code`
- `expected_and_observed_idcode`
- `prebuilt_bitstream_and_smoke_hashes`
- `flash_synthesis_place_route_pack_and_firmware_mutation_counts`
- `operator_action_packet`
- `hardware_execution_authenticated`
- `speed_power_and_terminal_claim_counts`
- `retirement_triggered`
- `protected_files_unchanged`
- `duration_s`
- `inference_substrate`
- `verifier_is_oracle`
- `missing_verifier_gaps`
- `field_provenance`
- `test_commands`
- `test_exit_codes`
- `reproducibility_checksum`
- `honest_verdict`

Required field principles:

- `prior_and_current_physical_state_hashes`: principle "Physical change, not another software loop, authorizes one attempt."
- `dated_operator_physical_receipt`: principle "A dated receipt is the only operator authorization for a new physical state."
- `physical_state_changed`: principle "Bare bool gates every JTAG command."
- `cable_port_power_board_usb_and_dirtyjtag_receipts`: principle "Every physical and transport assumption is explicit."
- `detect_attempt_allowed_attempt_count_command_stdout_stderr_and_exit_code`: principle "Unchanged state yields zero commands; changed state permits one auditable non-destructive command."
- `expected_and_observed_idcode`: principle "Smoke execution requires authenticated device identity."
- `prebuilt_bitstream_and_smoke_hashes`: principle "Existing artifacts must remain immutable before any read-only smoke."
- `flash_synthesis_place_route_pack_and_firmware_mutation_counts`: principle "All mutation counts remain zero without explicit operator authorization."
- `operator_action_packet`: principle "An unchanged-state block ends with one actionable physical next step."
- `hardware_execution_authenticated`: principle "No execution claim survives without raw hardware evidence."
- `speed_power_and_terminal_claim_counts`: principle "No speed, power, current-draw, or terminal-hardware claim is permitted."
- `retirement_triggered`: principle "Repeating the same physical block retires this changed-state task shape."
- `protected_files_unchanged`: principle "Conductor and operator-reconciled files remain byte-identical."
- `duration_s`: principle "Use measured `hardware_state_gate_with_optional_non_destructive_detect` wall time."
- `inference_substrate`: principle "Use `hardware_state_gate_with_optional_non_destructive_detect`."
- `verifier_is_oracle`: principle "Raw IDCODE/host-I/O evidence is authoritative; simulation is not board execution."
- `missing_verifier_gaps`: principle "Record missing raw IDCODE or host-I/O evidence instead of inferring."
- `field_provenance`: principle "Every field traces to receipts, hashes, command output, or tests."
- `test_commands`: principle "Verification commands are recorded."
- `test_exit_codes`: principle "Exit codes prevent failed checks becoming success."
- `reproducibility_checksum`: principle "Checksum detects physical-state, artifact, or receipt drift."
- `honest_verdict`: principle "Use `complete_changed_state:`, `blocked_physical_action:`, `retired:`, or `blocked:`."

**Acceptance criteria:**
- `.venv/bin/python -m carnot.experiment_6121_gatemate_changed_state_gate_v530 --date 20260804`
  writes `results/experiment_6121_gatemate_changed_state_gate_v530.json`.
- The artifact includes `spec_refs` containing `REQ-HW-6121` and
  `SCENARIO-HW-6121`, `random_seed=6121`, and stable
  `reproducibility_checksum`.
- Preconditions hash the declared prior GateMate artifacts, cable/port/power
  descriptions, USB/DirtyJTAG descriptors, tool-version receipts, operator
  physical receipts, bitstream/smoke hashes, output path, protected files, and
  dirty worktree before any optional hardware command.
- If no newer dated physical receipt changes cable, port, power, board, or
  DirtyJTAG state, `physical_state_changed=false`,
  `detect_attempt_allowed_attempt_count_command_stdout_stderr_and_exit_code.attempt_count=0`,
  no command stdout/stderr is present from a new detect, mutation counts are all
  zero, an exact operator action packet is populated,
  `hardware_execution_authenticated=false`, `retirement_triggered=true`, and
  `honest_verdict` begins with `blocked_physical_action:`.
- If a newer dated physical receipt changes state, exactly one non-destructive
  detect command may run, its command must equal
  `openFPGALoader -c dirtyJtag --detect`, and any prebuilt smoke is allowed only
  when `observed_idcode == expected_idcode == "0x20000001"` and prior bitstream
  and smoke hashes match.
- `flash_synthesis_place_route_pack_and_firmware_mutation_counts` reports zero
  for flash, synthesis, place, route, pack, and firmware mutation in every valid
  artifact.
- `speed_power_and_terminal_claim_counts` reports zero speedup, power,
  current-draw, terminal-hardware, TSU, and Kona claims.
- `inference_substrate` equals
  `hardware_state_gate_with_optional_non_destructive_detect`.
- `verifier_is_oracle=true` only for raw IDCODE/host-I/O evidence and hash
  equality; simulation never authenticates board execution.
- Protected files and `scripts/research_conductor.py` remain unchanged.

**Implementation status:** Planned (Exp 6121)

---

### SCENARIO-HW-6121

**Scenario:** Exp6121 blocks an unchanged GateMate physical state without running another DirtyJTAG detect.

**Given:** Exp5217 and Exp5861 record GateMate as blocked at DirtyJTAG/IDCODE
with raw IDCODE `0xffffffff`, the hardware wishlist says GateMate preserves the
v477 physical/JTAG block until cable, port, or board power changes, and no newer
dated operator receipt changes cable, port, power, board, or DirtyJTAG state,
**When:** Experiment 6121 hashes the prior and current physical-state receipts
and compares them with the last attempted state,
**Then:** It writes
`results/experiment_6121_gatemate_changed_state_gate_v530.json` with unchanged
physical-state hashes, zero JTAG/detect attempts, zero flash/synthesis/place/
route/pack/firmware mutation counts, a precise operator action packet, no
hardware execution or speed/power/terminal claim, `retirement_triggered=true`,
`inference_substrate=hardware_state_gate_with_optional_non_destructive_detect`,
and an `honest_verdict` beginning with `blocked_physical_action:`.

**Implementation status:** Planned (Exp 6121)

---

### REQ-HW-6199

**Title:** Exp6199 GateMate terminal-action audit MUST use cached receipts unless a newer material physical receipt authorizes one detect

**Description:**
Experiment 6199 SHALL produce
`results/experiment_6199_gatemate_terminal_action_audit_v537.json` as a
cached GateMate terminal-action audit for 20260807. The audit SHALL explicitly
inherit the REQ-HW-6121 authorization boundary: the canonical prior physical
state, as recorded by Exp6121, is the baseline; only a newer dated operator
receipt that materially changes cable, port, power, board, USB, or DirtyJTAG
state may authorize one bounded non-destructive detect. Visibility work that
does not change physical state MUST be a cached audit, not another terminal
probe.

Before any hardware command is considered, Exp6199 SHALL hash Exp6121, the
adversarial-flagged Exp3866 historical flash artifact, hardware bring-up and
wishlist/known-issues documents, cached tool identity receipts, protected
operator/conductor files, and any supplied dated operator physical receipt.
Exp3866 SHALL remain historical context only: its terminal flash transcript is
excluded from clean GateMate graduation evidence whenever it carries an
adversarial flag or pending corrigendum.

If the current physical state hash equals the canonical Exp6121 physical-state
hash, or if the supplied receipt is missing or stale, Exp6199 MUST run zero
`openFPGALoader`, JTAG, detect, synthesis, place, route, pack, flash, firmware,
SSH, timing, current, or power commands. It MUST set every hardware command and
mutation count to bare zero, set `hardware_command_authorized=false`, and emit
the exact operator action packet that names the required cable/port/power/board
or DirtyJTAG physical receipt.

If and only if a newer dated operator receipt materially changes cable, port,
power, board, USB, or DirtyJTAG state relative to Exp6121, Exp6199 MAY run at
most one command, exactly `openFPGALoader -c dirtyJtag --detect`, with bounded
timeout and exact stdout, stderr, and exit code captured. Exp6199 MUST NOT
synthesize, place, route, pack, flash, mutate firmware, run SSH, read timing,
read current, read power, or claim speed, power, energy, terminal hardware,
TSU, Kona, or sustained-performance results.

Required artifact fields:

- `status`
- `prior_receipt_paths_and_hashes`
- `current_dated_operator_receipt`
- `prior_and_current_physical_state_hashes`
- `physical_state_changed`
- `hardware_command_authorized`
- `detect_attempt_count_command_stdout_stderr_exit_code`
- `expected_and_observed_idcode`
- `mutation_command_counts`
- `historical_flagged_terminal_evidence_excluded`
- `operator_action_packet`
- `hardware_execution_authenticated`
- `speed_power_energy_terminal_tsu_kona_claim_counts`
- `passive_cooling_note`
- `protected_files_unchanged`
- `inference_substrate`
- `verifier_is_oracle`
- `missing_verifier_gaps`
- `field_provenance`
- `field_principles`
- `test_commands`
- `test_exit_codes`
- `duration_s`
- `reproducibility_checksum`
- `honest_verdict`

Required field principles:

- `prior_receipt_paths_and_hashes`: principle "Hashes precede authorization and prevent stale terminal evidence from graduating."
- `current_dated_operator_receipt`: principle "Only a newer dated physical receipt can move the GateMate state."
- `prior_and_current_physical_state_hashes`: principle "Exp6121 is the canonical no-repeat baseline."
- `physical_state_changed`: principle "A bare bool gates every hardware command."
- `hardware_command_authorized`: principle "Visibility checks require a material physical delta."
- `detect_attempt_count_command_stdout_stderr_exit_code`: principle "Zero on cached state; one exact non-destructive detect on changed state."
- `expected_and_observed_idcode`: principle "IDCODE is visibility evidence, not performance evidence."
- `mutation_command_counts`: principle "No synthesis, place, route, pack, flash, firmware, SSH, timing, current, or power command is allowed."
- `historical_flagged_terminal_evidence_excluded`: principle "Adversarial-flagged Exp3866 evidence stays quarantined."
- `operator_action_packet`: principle "A blocked audit ends with one concrete bench action."
- `hardware_execution_authenticated`: principle "Detect visibility is not workload execution."
- `speed_power_energy_terminal_tsu_kona_claim_counts`: principle "No speed, power, energy, terminal, TSU, or Kona claim is permitted."
- `passive_cooling_note`: principle "GateMate is passively cooled and no sustained-load inference is made."
- `protected_files_unchanged`: principle "Conductor and reconciler-owned docs remain byte-identical."
- `inference_substrate`: principle "Use cached receipt audit plus optional non-destructive detect, not LLM inference."
- `verifier_is_oracle`: principle "Raw hashes and IDCODE text are authoritative for this audit only."
- `missing_verifier_gaps`: principle "Missing IDCODE or workload evidence is recorded instead of inferred."
- `field_provenance`: principle "Every field traces to receipts, hashes, command output, or tests."
- `field_principles`: principle "Each required field carries its reason for existence."
- `test_commands`: principle "Verification commands are recorded."
- `test_exit_codes`: principle "Exit codes prevent failed checks becoming success."
- `duration_s`: principle "Measured wall time is reported without padding."
- `reproducibility_checksum`: principle "Checksum detects physical-state, receipt, or artifact drift."
- `honest_verdict`: principle "Use `blocked_no_change:`, `blocked_missing_receipt:`, `blocked_stale_receipt:`, `blocked_idcode:`, or `complete_visible:`."

**Acceptance criteria:**
- `.venv/bin/python -m carnot.experiment_6199_gatemate_terminal_action_audit_v537 --date 20260807`
  writes `results/experiment_6199_gatemate_terminal_action_audit_v537.json`.
- The artifact includes `spec_refs` containing `REQ-HW-6199` and the
  applicable `SCENARIO-HW-6199-*`, `random_seed=6199`, and a stable
  `reproducibility_checksum`.
- `inference_substrate` equals
  `cached_gatemate_terminal_action_audit_with_optional_single_detect`.
- Exp6121, Exp3866, hardware bring-up prep, hardware wishlist, known issues,
  cached tool identities, protected files, and any dated operator receipt are
  hashed before command authorization is evaluated.
- Missing, stale, or unchanged physical receipts produce
  `physical_state_changed=false`, `hardware_command_authorized=false`,
  zero detect attempts, zero mutation command counts, the exact operator action
  packet, and no stdout/stderr from a new hardware command.
- A newer material physical receipt permits exactly one
  `openFPGALoader -c dirtyJtag --detect` command and no other hardware,
  synthesis, packing, flashing, SSH, timing, current, or power command.
- A changed receipt with a wrong or missing IDCODE remains blocked and does not
  authenticate hardware execution.
- A changed receipt with the expected GateMate GM1Ax IDCODE
  `0x20000001` records board visibility only; it still makes no speed, power,
  energy, terminal-hardware, TSU, Kona, or sustained-load claim.
- Exp3866 remains unedited and excluded from clean terminal evidence when
  adversarial-flagged.
- Protected files and `scripts/research_conductor.py` remain unchanged.

**Implementation status:** Planned (Exp 6199)

---

### SCENARIO-HW-6199-1

**Scenario:** Exp6199 blocks unchanged GateMate state without executing hardware or tool commands.

**Given:** Exp6121 records an unchanged GateMate physical-state hash and no
newer dated operator receipt changes cable, port, power, board, USB, or
DirtyJTAG state,
**When:** Exp6199 hashes prior receipts and compares the current state to the
canonical Exp6121 baseline,
**Then:** It writes the terminal-action audit with
`physical_state_changed=false`, `hardware_command_authorized=false`, detect
attempt count zero, all mutation command counts zero, the exact operator action
packet, Exp3866 excluded as historical flagged evidence, and an
`honest_verdict` beginning with `blocked_no_change:`.

**Implementation status:** Planned (Exp 6199)

---

### SCENARIO-HW-6199-2

**Scenario:** Exp6199 rejects missing and stale operator receipts.

**Given:** No dated physical receipt exists, or a receipt date is not newer than
the Exp6121 baseline date,
**When:** Exp6199 computes the physical-state hash,
**Then:** It treats the state as unchanged for command authorization, runs zero
hardware/tool commands, and emits `blocked_missing_receipt:` or
`blocked_stale_receipt:` without lifting Exp3866 quarantine.

**Implementation status:** Planned (Exp 6199)

---

### SCENARIO-HW-6199-3

**Scenario:** Exp6199 permits one detect for a newer material receipt but blocks a wrong IDCODE.

**Given:** A dated operator receipt newer than Exp6121 materially changes
cable, port, power, board, USB, or DirtyJTAG state,
**When:** The single permitted `openFPGALoader -c dirtyJtag --detect` receipt
does not contain expected IDCODE `0x20000001`,
**Then:** Exp6199 records exactly one detect command, all mutation command
counts remain zero, hardware execution is not authenticated, and
`honest_verdict` begins with `blocked_idcode:`.

**Implementation status:** Planned (Exp 6199)

---

### SCENARIO-HW-6199-4

**Scenario:** Exp6199 records changed-state GateMate visibility without performance or terminal claims.

**Given:** A newer material operator receipt authorizes one detect,
**When:** The detect receipt contains the expected GateMate GM1Ax IDCODE
`0x20000001`,
**Then:** Exp6199 records exactly one non-destructive detect and
IDCODE visibility, keeps hardware execution unauthenticated, keeps all
speed/power/energy/terminal/TSU/Kona counts at zero, preserves the passive
cooling note, and emits an `honest_verdict` beginning with `complete_visible:`.

**Implementation status:** Planned (Exp 6199)

---

### SCENARIO-HW-6199-5

**Scenario:** Exp6199 fails closed on command-budget or allowlist violations.

**Given:** An artifact records more than one detect, records a detect command
without authorization, or records a command other than
`openFPGALoader -c dirtyJtag --detect`,
**When:** Exp6199 validates the audit,
**Then:** validation fails before the artifact can graduate.

**Implementation status:** Planned (Exp 6199)

---

### REQ-HW-6325

**Title:** Exp6325 GateMate dated receipt single-detect MUST run exactly one read-only DirtyJTAG detect and then stop

**Description:**
Experiment 6325 SHALL produce
`results/experiment_6325_gatemate_dated_receipt_single_detect.json` as the
single authorized GateMate detect receipt for the 2026-08-11 dated physical
power-cycle record. It SHALL validate the receipt in `ops/known-issues.md`,
prove that its `20260811` date is newer than the prior Exp6121 and Exp6199
failed GateMate attempts, and prove that the target is exactly the Cologne Chip
GateMate A1-EVB-2M with DirtyJTAG cable `1209:c0ca`.

Before any board-addressing command, Exp6325 SHALL hash the protected files,
hash the declared GateMate prior receipts, record read-only USB state, record
disk and permission receipts, resolve the exact `openFPGALoader` binary path,
record `openFPGALoader --version` without addressing a board, and record the
one-command budget. If the dated receipt is missing, stale, malformed, or
targets a non-GateMate board, the artifact MUST be blocked and run zero
hardware commands.

When the preconditions pass, Exp6325 SHALL run exactly one bounded command:
`openFPGALoader -c dirtyJtag --detect`. It SHALL capture UTC start and finish
times, stdout, stderr, exit code, timeout state, and any detected chain/device
IDs. It MUST stop after this attempt for success, empty chain, failure, or
timeout. It MUST NOT run flash, erase, reset, synthesis, place, route, timing,
KV260, or PolarFire commands.

Required artifact fields:

- `status`
- `dated_physical_receipt_path_hash_date_and_text`
- `receipt_newer_than_prior_attempts`
- `board_and_cable_target`
- `pre_command_usb_receipt`
- `openfpgaloader_version_receipt`
- `exact_authorized_command`
- `detect_command_count`
- `detect_started_utc`
- `detect_finished_utc`
- `detect_stdout`
- `detect_stderr`
- `detect_exit_code`
- `detect_timeout`
- `detected_chain_and_device_ids`
- `post_command_usb_receipt`
- `hardware_state_changed_from_prior_attempts`
- `flash_command_count`
- `erase_command_count`
- `reset_command_count`
- `synthesis_command_count`
- `place_route_command_count`
- `timing_command_count`
- `kv260_command_count`
- `polarfire_command_count`
- `stop_after_single_attempt_receipt`
- `protected_files_unchanged`
- `preconditions_checked`
- `inference_substrate`
- `verifier_is_oracle`
- `field_provenance`
- `field_principles`
- `test_commands`
- `test_exit_codes`
- `duration_s`
- `reproducibility_checksum`
- `honest_verdict`

Required field principles:

- `status`: principle "Terminal state separates blocked preconditions from a completed single detect."
- `dated_physical_receipt_path_hash_date_and_text`: principle "The operator receipt is the only physical-change authority."
- `receipt_newer_than_prior_attempts`: principle "Receipt date must be newer than prior failed attempts."
- `board_and_cable_target`: principle "The one command is scoped to GateMate plus DirtyJTAG only."
- `pre_command_usb_receipt`: principle "Read-only USB state is captured before the board command."
- `openfpgaloader_version_receipt`: principle "Tool version is recorded without addressing a board."
- `exact_authorized_command`: principle "Only one exact non-destructive command is allowed."
- `detect_command_count`: principle "A bare integer enforces the single-attempt budget."
- `detect_started_utc`: principle "UTC start time orders the hardware receipt."
- `detect_finished_utc`: principle "UTC finish time bounds the attempt."
- `detect_stdout`: principle "Raw stdout preserves the detection result."
- `detect_stderr`: principle "Raw stderr preserves tool failures."
- `detect_exit_code`: principle "Exit code prevents failed detects becoming success."
- `detect_timeout`: principle "Timeout is a terminal outcome, not a retry trigger."
- `detected_chain_and_device_ids`: principle "Parsed chain data is derived only from raw detect output."
- `post_command_usb_receipt`: principle "Read-only USB state is captured after the attempt."
- `hardware_state_changed_from_prior_attempts`: principle "Changed power state, not software repetition, authorizes this attempt."
- `flash_command_count`: principle "Flash commands are forbidden."
- `erase_command_count`: principle "Erase commands are forbidden."
- `reset_command_count`: principle "Reset commands are forbidden."
- `synthesis_command_count`: principle "Synthesis commands are forbidden."
- `place_route_command_count`: principle "Place and route commands are forbidden."
- `timing_command_count`: principle "Timing commands are forbidden."
- `kv260_command_count`: principle "KV260 commands are forbidden in this GateMate task."
- `polarfire_command_count`: principle "PolarFire commands are forbidden in this GateMate task."
- `stop_after_single_attempt_receipt`: principle "All outcomes stop after one attempt."
- `protected_files_unchanged`: principle "Operator and conductor files remain byte-identical."
- `preconditions_checked`: principle "Hashes, USB, tool path, disk, permissions, timeout, and budget are checked first."
- `inference_substrate`: principle "Use read-only host receipts plus one DirtyJTAG detect."
- `verifier_is_oracle`: principle "Raw receipts and command output are authoritative only for visibility."
- `field_provenance`: principle "Every field traces to a receipt, command output, parser, or test."
- `field_principles`: principle "Every required field declares why it exists."
- `test_commands`: principle "Verification commands are recorded."
- `test_exit_codes`: principle "Verification exit codes are recorded."
- `duration_s`: principle "Measured wall time is reported without padding."
- `reproducibility_checksum`: principle "Checksum detects receipt or artifact drift."
- `honest_verdict`: principle "Verdict names the raw outcome without inferring execution."

**Acceptance criteria:**
- `.venv/bin/python -m carnot.experiment_6325_gatemate_dated_receipt_single_detect --date 20260812`
  writes `results/experiment_6325_gatemate_dated_receipt_single_detect.json`.
- The artifact includes `spec_refs` containing `REQ-HW-6325` and the applicable
  `SCENARIO-HW-6325-*`, `random_seed=6325`, and a stable checksum.
- Missing, stale, malformed, or wrong-target receipts run zero hardware
  commands and write a blocked artifact.
- A valid 2026-08-11 GateMate power-cycle receipt permits exactly one
  `openFPGALoader -c dirtyJtag --detect` with a bounded timeout.
- Success, empty-chain, tool failure, and timeout all stop after that single
  attempt and preserve raw stdout/stderr.
- `detect_command_count` is bare `1` after the command.
- Flash, erase, reset, synthesis, place/route, timing, KV260, and PolarFire
  command counts are bare `0` in every valid artifact.

**Implementation status:** Planned (Exp 6325)

---

### SCENARIO-HW-6325-1

**Scenario:** Exp6325 records a matching GateMate detect once and makes no execution claim.

**Given:** The 2026-08-11 GateMate power-cycle receipt is present and newer
than Exp6121 and Exp6199,
**When:** the single detect output contains IDCODE `0x20000001`,
**Then:** the artifact records one detect, parsed GateMate chain visibility,
all forbidden command counts remain zero, and `honest_verdict` begins with
`complete_visible:`.

**Implementation status:** Planned (Exp 6325)

---

### SCENARIO-HW-6325-2

**Scenario:** Exp6325 records an empty chain once and stops.

**Given:** The dated GateMate receipt authorizes one detect,
**When:** the detect output has no device ID,
**Then:** the artifact records one detect, an empty parsed chain, zero
forbidden command counts, and `honest_verdict` begins with
`blocked_empty_chain:`.

**Implementation status:** Planned (Exp 6325)

---

### SCENARIO-HW-6325-3

**Scenario:** Exp6325 records a timeout once and stops.

**Given:** The dated GateMate receipt authorizes one detect,
**When:** the detect command times out,
**Then:** the artifact records one detect, `detect_timeout=true`, preserved
stderr, zero retry commands, and `honest_verdict` begins with
`blocked_timeout:`.

**Implementation status:** Planned (Exp 6325)

---

### SCENARIO-HW-6325-4

**Scenario:** Exp6325 blocks stale or missing receipts before hardware access.

**Given:** The physical receipt is missing or not newer than the prior failed
GateMate attempts,
**When:** Exp6325 checks preconditions,
**Then:** it writes a blocked artifact with `detect_command_count=0` and runs
zero hardware commands.

**Implementation status:** Planned (Exp 6325)

---

### SCENARIO-HW-6325-5

**Scenario:** Exp6325 blocks wrong-target receipts before hardware access.

**Given:** A dated receipt names any board or cable other than GateMate
A1-EVB-2M with DirtyJTAG `1209:c0ca`,
**When:** Exp6325 checks preconditions,
**Then:** it writes a blocked artifact with zero hardware commands and
`honest_verdict` beginning with `blocked_wrong_target:`.

**Implementation status:** Planned (Exp 6325)

---

### SCENARIO-HW-6325-6

**Scenario:** Exp6325 validation refuses a second detect or forbidden command count.

**Given:** An artifact records more than one detect, a different command, or a
non-zero flash, erase, reset, synthesis, place/route, timing, KV260, or
PolarFire count,
**When:** Exp6325 validates the artifact,
**Then:** validation fails before the artifact can graduate.

**Implementation status:** Planned (Exp 6325)

---

### REQ-HW-6525

**Title:** Exp6525 GateMate changed-state continuity MUST run zero commands without a post-Exp6325 dated physical receipt

**Description:**
Experiment 6525 SHALL produce
`results/experiment_6525_gatemate_changed_state_continuity.json` as the V564
GateMate continuity record for planning date 20260823. The experiment SHALL use
Exp6325 as the last hardware-action baseline. It MUST search only
operator-authored known issues and approved receipt locations for a dated
physical-state receipt newer than Exp6325 that names a concrete cable, port,
power, board, or DirtyJTAG change. Planner-created, undated, stale, malformed,
and USB-only evidence SHALL NOT authorize hardware access.

Before authorization, Exp6525 SHALL record git status, current time, historical
artifact paths and hashes, the Exp3866 exclusion state, protected-file hashes,
the exact receipt search locations, and the last known GateMate state. If no
valid post-Exp6325 physical receipt exists, Exp6525 MUST run zero `lsusb`,
`openFPGALoader`, yosys, nextpnr, gmpack, JTAG, flash, reset, SSH, timing,
current, or power commands, and SHALL emit `blocked_missing_new_physical_receipt`
with `hardware_command_count=0`.

If and only if a valid newer receipt exists and passes safe target validation,
Exp6525 MAY run exactly one predeclared bounded GateMate action: either the
read-only detect `openFPGALoader -c dirtyJtag --detect` or a flash step whose
board and bitstream identities are authenticated before execution. The
experiment SHALL stop at the first terminal result: success, failure, timeout,
or ambiguous target. It MUST NOT retry, MUST preserve Exp3866 exclusion and all
historical verdicts, MUST NOT infer physical change from USB enumeration alone,
and MUST NOT claim flash, smoke, latency, energy, speedup, availability, or
terminal state without same-run authenticated evidence.

Required artifact fields:

- `status`
- `honest_verdict`
- `verdict_class`
- `prior_failure_receipts`
- `historical_state_receipts`
- `dated_receipt_search_rows`
- `changed_state_receipt`
- `authorization_decision`
- `hardware_command_count`
- `command_rows`
- `terminal_disposition`
- `gatemate_continuity_slot_complete_score`
- `gatemate_bitstream_flashed`
- `hardware_speedup_claim`
- `gate_check_summary`
- `per_unit_rows`
- `aggregate_row_recomputation`
- `preconditions_checked`
- `protected_files_unchanged`
- `inference_substrate`
- `verifier_is_oracle`
- `field_principles`
- `field_provenance`
- `random_seed`
- `duration_s`
- `tests_run`
- `reproducibility_checksum`

Required field principles:

- `dated_receipt_search_rows`: principle "One row per candidate proves why it did or did not authorize hardware."
- `authorization_decision`: principle "Only a post-Exp6325 material physical receipt can spend the single action budget."
- `hardware_command_count`: principle "Bare zero or one enforces no unchanged reruns."
- `command_rows`: principle "If a command runs, argv, timing, exit, hashes, device identity, and terminal disposition are recorded."
- `terminal_disposition`: principle "The first terminal result stops the task."
- `gatemate_continuity_slot_complete_score`: principle "An honest closed block or one-action record completes the continuity slot."
- `gatemate_bitstream_flashed`: principle "True only for same-run authenticated flash evidence."
- `hardware_speedup_claim`: principle "This continuity task makes no performance claim."
- `protected_files_unchanged`: principle "Conductor and reconciler-owned files remain byte-identical."
- `inference_substrate`: principle "Use no-command dated receipt audit unless a command actually runs."
- `verifier_is_oracle`: principle "Only device and bitstream identity checks may be authoritative; positive claims are never oracle-backed."

**Acceptance criteria:**
- `.venv/bin/python -m carnot.experiment_6525_gatemate_changed_state_continuity --date 20260823`
  writes `results/experiment_6525_gatemate_changed_state_continuity.json`.
- The artifact includes `spec_refs` containing `REQ-HW-6525` and the
  applicable `SCENARIO-HW-6525-*`, `random_seed=6525`, and a stable checksum.
- Missing, stale, planner-created, undated, malformed, or USB-only receipt
  candidates produce `status=blocked_missing_new_physical_receipt`,
  `verdict_class=blocked`, `hardware_command_count=0`, `command_rows=[]`,
  `inference_substrate=dated_hardware_receipt_audit_no_command_no_llm`, and
  `gatemate_continuity_slot_complete_score=1.0`.
- A valid post-Exp6325 material physical receipt permits exactly one safe
  predeclared GateMate action and no retry. The command row records argv,
  start/end UTC, exit code, stdout/stderr hashes, device identity, and terminal
  disposition.
- `gatemate_bitstream_flashed=true` only when the same run records an
  authenticated flash command with matching GateMate target and bitstream
  identity.
- `hardware_speedup_claim=false` in every valid artifact.
- Exp3866 stays excluded from clean terminal evidence, and historical verdicts
  are copied only as historical state receipts.
- Protected files and `scripts/research_conductor.py` remain unchanged.

**Implementation status:** Planned (Exp 6525)

---

### SCENARIO-HW-6525-1

**Scenario:** Exp6525 closes GateMate continuity with zero commands when no newer physical receipt exists.

**Given:** Exp6325 consumed the 2026-08-11 GateMate power-cycle receipt and
stopped on a failed single detect,
**When:** Exp6525 searches approved receipt locations and finds no
operator-authored dated physical receipt after Exp6325 naming cable, port,
power, board, or DirtyJTAG change,
**Then:** it writes the required artifact with
`status=blocked_missing_new_physical_receipt`, `hardware_command_count=0`,
`command_rows=[]`, `verdict_class=blocked`, no speedup or flash claim, Exp3866
still excluded, and the continuity slot score equal to `1.0`.

**Implementation status:** Planned (Exp 6525)

---

### SCENARIO-HW-6525-2

**Scenario:** Exp6525 rejects stale, planner-created, undated, malformed, and USB-only candidates.

**Given:** Receipt candidate text may mention GateMate, DirtyJTAG, USB
enumeration, or planner next steps,
**When:** the candidate is not operator-authored, is undated, is not newer than
Exp6325, lacks a concrete physical change, or mentions only USB enumeration,
**Then:** the candidate row is marked invalid, no hardware command is
authorized, and the artifact remains a blocked no-command continuity record.

**Implementation status:** Planned (Exp 6525)

---

### SCENARIO-HW-6525-3

**Scenario:** Exp6525 permits one bounded action for a valid newer physical receipt and stops.

**Given:** An operator-authored receipt dated after Exp6325 names a concrete
GateMate cable, port, power, board, or DirtyJTAG change and passes safe target
validation,
**When:** the predeclared action is executed,
**Then:** Exp6525 records exactly one command row with argv, start/end, exit
code, stdout/stderr hashes, device identity, terminal disposition, zero retry
count, and no performance claim.

**Implementation status:** Planned (Exp 6525)

---

### SCENARIO-HW-6525-4

**Scenario:** Exp6525 validation fails closed on command budget, target, or claim violations.

**Given:** An artifact records more than one hardware command, an unauthorized
command, non-empty command output without authorization, a positive flash bool
without same-run flash evidence, or a hardware speedup claim,
**When:** Exp6525 validates the artifact,
**Then:** validation fails before the artifact can graduate.

**Implementation status:** Planned (Exp 6525)

---

### REQ-HW-6559

**Title:** Exp6559 GateMate changed-state continuity MUST use Exp6525 as the no-repeat boundary and stop after zero or one hardware action

**Description:**
Experiment 6559 SHALL produce
`results/experiment_6559_gatemate_changed_state_continuity.json` as the V567
GateMate continuity record for planning date 20260823. The experiment SHALL use
Exp6525 as the prior failed attempt and SHALL require a dated operator-authored
GateMate physical-state receipt newer than Exp6525 before any board command can
run. A physical-state receipt MUST name a concrete cable, port, board power,
board, or DirtyJTAG change. Planner text, agent-written plans, stale repeated
USB enumeration, command transcripts with no new physical change, and ambiguous
board targets SHALL NOT authorize hardware access.

Before authorization, Exp6559 SHALL record git status, current UTC time, exact
receipt search roots, Exp6525 artifact path and hash, USB enumeration only from
existing durable receipts, tool and bitstream identities without touching
hardware, CPU/RAM/disk receipts, protected-file hashes, and the Exp3866
exclusion state. If no valid newer physical receipt exists, Exp6559 MUST run
zero `openFPGALoader`, JTAG, flash, reset, USB, board, synthesis, place, route,
pack, timing, current, power, or SSH commands. It SHALL emit a terminal blocked
artifact with the failed check, the latest receipt date it observed, and
mechanical zero-command proof.

If and only if a valid newer receipt exists and safe target validation closes,
Exp6559 MAY run exactly one authorized GateMate action: either bounded
DirtyJTAG GM1Ax detect or a validated flash. The artifact SHALL capture command,
monotonic timing, exit status, stdout and stderr hashes, detected IDCODE or
flash receipt, USB identity from the receipt trail, and the board target. The
experiment MUST stop after the first terminal result. It MUST NOT retry at
multiple clock rates, change RTL, synthesize a design, infer availability from
USB enumeration, reopen Exp3866, or claim latency, speed, energy, sampling
quality, or general availability.

Required artifact fields:

- `status`
- `honest_verdict`
- `verdict_class`
- `prior_failure_receipt`
- `operator_physical_state_receipt`
- `safe_target_validation_receipt`
- `hardware_action_rows`
- `terminal_command_receipt`
- `zero_command_block_receipt`
- `exp3866_exclusion_preserved`
- `claim_boundary`
- `attack_matrix`
- `gatemate_changed_state_slot_complete_score`
- `gatemate_hardware_advanced_score`
- `per_unit_rows`
- `aggregate_row_recomputation`
- `gate_check_summary`
- `preconditions_checked`
- `protected_files_unchanged`
- `inference_substrate`
- `verifier_is_oracle`
- `field_provenance`
- `duration_s`
- `tests_run`
- `reproducibility_checksum`

Required field principles:

- `prior_failure_receipt`: principle "The artifact must identify Exp6525 and the stricter newer-than boundary."
- `operator_physical_state_receipt`: principle "Only a dated operator-authored physical change can authorize a new board command."
- `safe_target_validation_receipt`: principle "Board, cable, tool, action, and bitstream identity must close before hardware access."
- `hardware_action_rows`: principle "Zero or one action rows make the bounded command budget mechanically recheckable."
- `terminal_command_receipt`: principle "A real detect or flash result needs command, timing, exit, stream hashes, and device identity."
- `zero_command_block_receipt`: principle "A missing physical receipt must prove that no hardware command ran."
- `claim_boundary`: principle "The artifact must disclaim latency, speed, energy, quality, and general availability."
- `aggregate_row_recomputation`: principle "Command count and advancement must derive from the emitted rows."
- `gate_check_summary`: principle "A blocked verdict must name the missing receipt check and observed latest date."
- `inference_substrate`: principle "Use dated_hardware_receipt_audit_no_command_no_llm when no valid change exists, or hardware_smoke for one authorized board action."
- `verifier_is_oracle`: principle "Always false; the transcript is evidence for one action only, not a model verifier."

**Acceptance criteria:**
- `.venv/bin/python -m carnot.experiment_6559_gatemate_changed_state_continuity --date 20260823`
  writes `results/experiment_6559_gatemate_changed_state_continuity.json`.
- The artifact includes `spec_refs` containing `REQ-HW-6559` and the
  applicable `SCENARIO-HW-6559-*`, `random_seed=6559`, and a final checksum.
- Missing, stale, planner-created, agent-created, undated, USB-only, or
  ambiguous receipt candidates produce `status=blocked_missing_new_physical_receipt`,
  `verdict_class=blocked`, `hardware_action_rows=[]`,
  `terminal_command_receipt=null`,
  `inference_substrate=dated_hardware_receipt_audit_no_command_no_llm`,
  `verifier_is_oracle=false`,
  `gatemate_changed_state_slot_complete_score=1.0`, and
  `gatemate_hardware_advanced_score=0.0`.
- A valid post-Exp6525 material physical receipt permits exactly one safe
  predeclared GateMate action and no retry. The command row records argv,
  monotonic start/end, exit status, stdout/stderr hashes, device identity, USB
  identity, board target, and terminal disposition.
- `verdict_class=null` only for a terminal one-action detect or flash receipt
  without a performance claim. `verdict_class=partial` marks incomplete
  authenticated output. `verdict_class=disqualified` marks unauthorized
  commands, target ambiguity, false provenance, or overclaim.
- Exp3866 stays excluded from clean terminal evidence, and historical verdicts
  are copied only as historical state receipts.
- Protected files and `scripts/research_conductor.py` remain unchanged.

**Implementation status:** Planned (Exp 6559)

---

### SCENARIO-HW-6559-1

**Scenario:** Exp6559 closes GateMate continuity with zero commands when no post-Exp6525 physical receipt exists.

**Given:** Exp6525 already closed with zero hardware commands because no
operator-authored dated GateMate physical-state receipt existed,
**When:** Exp6559 searches durable operator-authored receipt locations and finds
no GateMate cable, port, power, board, or DirtyJTAG change dated after Exp6525,
**Then:** it writes a terminal blocked artifact with zero hardware action rows,
the failed newer-than-Exp6525 receipt check, the latest observed receipt date,
Exp3866 preserved, no performance claim, and the changed-state slot score equal
to `1.0`.

**Implementation status:** Planned (Exp 6559)

---

### SCENARIO-HW-6559-2

**Scenario:** Exp6559 rejects stale, planner-created, agent-created, USB-only, and ambiguous receipt candidates.

**Given:** Candidate text may mention GateMate, DirtyJTAG, USB enumeration,
operator plans, or command output,
**When:** the candidate is not a durable dated operator-authored physical change
newer than Exp6525, or the board target is ambiguous,
**Then:** the candidate row is invalid, no hardware command is authorized, and
the artifact remains a blocked no-command continuity record.

**Implementation status:** Planned (Exp 6559)

---

### SCENARIO-HW-6559-3

**Scenario:** Exp6559 permits one bounded action for a valid newer physical receipt and stops at the first terminal result.

**Given:** An operator-authored receipt dated after Exp6525 names a concrete
GateMate cable, port, power, board, or DirtyJTAG change and safe target
validation confirms board, cable, tool, action, bitstream if applicable, and
expected GM1Ax identity,
**When:** the predeclared detect or flash action is executed,
**Then:** Exp6559 records exactly one hardware action row with command, timing,
exit, stream hashes, device identity, terminal disposition, zero retry count,
and no performance or availability claim.

**Implementation status:** Planned (Exp 6559)

---

### SCENARIO-HW-6559-4

**Scenario:** Exp6559 validation fails closed on command budget, target, output, provenance, or claim violations.

**Given:** An artifact records more than one hardware command, a command without
receipt authorization, a non-allowlisted command, missing terminal output hashes,
a false receipt provenance, target ambiguity, a reopened Exp3866 path, or a
latency, speed, energy, quality, or availability claim,
**When:** Exp6559 validates the artifact,
**Then:** validation fails before the artifact can graduate.

**Implementation status:** Planned (Exp 6559)

---

### REQ-HW-5930

**Title:** Exp5930 adaptive-state ABI v2 board mapping MUST produce static receipts and skip unchanged physical probes

**Description:**
Experiment 5930 SHALL translate the qualified Exp5926 adaptive-state ABI v2
operations into backend-neutral RTL/HLS interface semantics and SHALL write
`results/experiment_5930_adaptive_state_board_mapping.json`. The artifact is a
static mapping and reproducibility receipt only unless a fresh pre-command
authenticated route-state diff proves that an attached board has a materially
new adaptive-state operation path compared with Exp5861. Repeating Exp5861's
KV260 programmed-image proof of concept, PolarFire prior physical workload, or
GateMate DirtyJTAG/IDCODE probes is prohibited when the route diff is unchanged.

The mapper SHALL replay the Exp5926 gate, hash ABI schema/traces, Exp5861
evidence, board state, installed toolchain versions, source files, generated
outputs, and protected files. It SHALL define fixed-width request and response
records for every ABI v2 operation, valid/ready ordering, state-version checks,
validator receipt transport, status/error codes, atomic commit, rollback, and
crash recovery without embedding model semantics or model weights. The
simulator/reference harness SHALL replay Exp5926 conformance traces and
adversarial stale, replay, tamper, and crash sequences with state/status/error
parity.

The mapper SHALL run available local lint, simulation, synthesis, timing
estimate, and resource-report commands for installed toolchains, and each
receipt SHALL record tool, version, target, command, exit code, stdout/stderr
hashes, output hashes, and whether the result is an estimate rather than a
physical measurement. If no authenticated route changed, the artifact SHALL set
`physical_probe_executed=false`, preserve
`kv260=programmed_image_poc`, `polarfire=prior_physical_workload_only`, and
`gatemate=blocked_idcode`, and record the exact no-unchanged-probe decision.

Required artifact fields:

- `status`
- `gate_replay_receipt`
- `preconditions_checked`
- `abi_v2_schema_hash_and_operation_mapping`
- `fixed_width_request_response_and_error_contract`
- `ordering_backpressure_atomicity_rollback_and_recovery`
- `simulator_reference_trace_parity`
- `stale_replay_tamper_and_crash_matrix`
- `installed_toolchain_target_command_exit_and_hash_receipts`
- `static_synthesis_timing_estimate_and_resource_reports`
- `authenticated_route_state_diff`
- `physical_probe_executed`
- `bounded_physical_trace_and_teardown_if_any`
- `kv260_polarfire_and_gatemate_state_receipts`
- `no_unchanged_probe_receipt`
- `no_speedup_power_energy_thermalization_convergence_tsu_kona_or_sovereignty_claim`
- `protected_files_unchanged`
- `board_abi_mapping_ready_score`
- `duration_s`
- `inference_substrate`
- `verifier_is_oracle`
- `field_provenance`
- `test_commands`
- `test_exit_codes`
- `reproducibility_checksum`
- `honest_verdict`

Required field principles:

- `status`: principle "Terminal mapping state without physical-performance implication."
- `gate_replay_receipt`: principle "Exp5926 readiness and trace hashes authorize mapping."
- `preconditions_checked`: principle "Hash source, tools, routes, outputs, and protected files before any board command."
- `abi_v2_schema_hash_and_operation_mapping`: principle "Every supported ABI v2 operation has a finite board-neutral encoding."
- `fixed_width_request_response_and_error_contract`: principle "Fixed-width request and response records make RTL/HLS behavior finite."
- `ordering_backpressure_atomicity_rollback_and_recovery`: principle "Valid/ready ordering, backpressure, commit, rollback, and recovery are explicit."
- `simulator_reference_trace_parity`: principle "Simulation parity is ABI trace/state/status parity, not performance."
- `stale_replay_tamper_and_crash_matrix`: principle "Unsafe sequences fail closed without partial mutation."
- `installed_toolchain_target_command_exit_and_hash_receipts`: principle "Tool receipts pin the local static evidence path."
- `static_synthesis_timing_estimate_and_resource_reports`: principle "Synthesis/resource/timing receipts are estimates unless physical measurement exists."
- `authenticated_route_state_diff`: principle "Only a materially new authenticated route may permit a board command."
- `physical_probe_executed`: principle "Bare true only after a fresh changed authenticated route and recorded teardown."
- `bounded_physical_trace_and_teardown_if_any`: principle "Physical execution requires exact commands, identity, trace, rollback, and teardown."
- `kv260_polarfire_and_gatemate_state_receipts`: principle "Upstream board states stay separate and cannot imply new execution."
- `no_unchanged_probe_receipt`: principle "Retired probes are skipped when routes are unchanged."
- `no_speedup_power_energy_thermalization_convergence_tsu_kona_or_sovereignty_claim`: principle "Bare true unless fresh physical measurements authorize a narrower claim."
- `protected_files_unchanged`: principle "Conductor and ops reconciliation files remain byte-identical."
- `board_abi_mapping_ready_score`: principle "Bare 1.0 means ABI trace parity plus static tool receipts, not acceleration."
- `duration_s`: principle "Wall time exposes receipt generation scope."
- `inference_substrate`: principle "Use `rtl_hls_simulation_and_static_synthesis_no_llm`."
- `verifier_is_oracle`: principle "True only for ABI trace/state/status parity, hashes, and tool receipts."
- `field_provenance`: principle "Every field traces to specs, upstream artifacts, source, tools, traces, or route diff."
- `test_commands`: principle "Verification commands are recorded."
- `test_exit_codes`: principle "Exit codes prevent failed static checks from becoming readiness."
- `reproducibility_checksum`: principle "A checksum detects ABI, source, tool, trace, route, or artifact drift."
- `honest_verdict`: principle "Use `complete_static_mapping:`, `complete_physical_receipt:`, `no_change:`, `retired:`, or `blocked:`."

**Acceptance criteria:**
- `.venv/bin/python -m carnot.experiment_5930_adaptive_state_board_mapping --date 20260726`
  writes `results/experiment_5930_adaptive_state_board_mapping.json`.
- The artifact includes `spec_refs` containing `REQ-HW-5930`,
  `SCENARIO-HW-5930`, `REQ-FPGA-5930`, and `SCENARIO-FPGA-5930`, with a stable
  `reproducibility_checksum`.
- Exp5926 is replay-gated by artifact hash, ready score, ABI schema, operation
  set, trace hash, and test receipts before static mapping is marked ready.
- Fixed-width request/response records cover snapshot, lookup, propose, commit,
  validate, promote, quarantine, supersede, reject, rollback, and recover.
- Valid/ready semantics preserve in-order acceptance, backpressure stalls
  mutation, stale versions reject, validator receipts are transported as hashes,
  and commit/rollback/recovery are atomic.
- Simulator/reference replay reports exact state/status/error parity for the
  Exp5926 trace and for stale, replay, tamper, and crash sequences.
- Installed lint/simulation/synthesis/resource commands record tool, version,
  target, command, exit code, stdout/stderr hash, and output hash. Timing and
  resource reports are labelled as static estimates.
- `physical_probe_executed=true` is valid only when
  `authenticated_route_state_diff.materially_new_authenticated_route=true` and
  exact physical commands plus teardown are recorded.
- When the authenticated route diff is unchanged, `physical_probe_executed=false`,
  `bounded_physical_trace_and_teardown_if_any=[]`, board states preserve
  Exp5861's KV260/PolarFire/GateMate labels, and no SSH/JTAG/IDCODE/programming
  commands are repeated.
- The no-claim field is bare `true`, `inference_substrate` equals
  `rtl_hls_simulation_and_static_synthesis_no_llm`, `verifier_is_oracle=true`,
  `board_abi_mapping_ready_score=1.0`, and `honest_verdict` starts with an
  allowed terminal prefix without speed, power, energy, thermalization,
  convergence, TSU, Kona, or sovereignty claims.

**Implementation status:** Planned (Exp 5930)

---

### SCENARIO-HW-5930

**Scenario:** Exp5930 writes a complete static ABI v2 board-mapping receipt without repeating unchanged board probes.

**Given:** Exp5926 reports ABI v2 ready score 1.0, Exp5861 reports no changed
authenticated state-operation route, KV260 remains a programmed-image proof of
concept, PolarFire remains prior physical workload only, and GateMate remains
blocked by IDCODE,
**When:** Experiment 5930 hashes inputs, computes a read-only route-state diff,
maps ABI v2 fixed-width request/response records, replays the trace simulator,
and runs installed static lint/simulation/synthesis/resource flows,
**Then:** It writes
`results/experiment_5930_adaptive_state_board_mapping.json` with complete ABI
trace parity, static tool receipts, `physical_probe_executed=false`,
`board_abi_mapping_ready_score=1.0`, preserved upstream board states, no
unchanged probe receipt, bare no-claim true, and an `honest_verdict` beginning
with `complete_static_mapping:`.

**Implementation status:** Planned (Exp 5930)

---

### REQ-HW-5967

**Title:** Exp5967 delayed-commit memory fixture MUST emit backend-neutral fixed-width trace receipts without hardware execution claims

**Description:**
Experiment 5967 SHALL emit a fixed-width backend-neutral operation trace for
the delayed-commit memory lifecycle derived from ready Exp5920, Exp5924, and
Exp5926 receipts. The trace SHALL encode read snapshot, propose, validate,
commit, quarantine, supersede, rollback, crash replay, rejection, and control
operations with finite operation ids, state versions, event indices, capacity
bounds, payload hashes, validator receipt hashes, return codes, and resulting
state hashes suitable for CPU, GPU, FPGA, or future TSU mapping.

The trace is portability evidence only. Exp5967 MUST NOT claim board
execution, speedup, power, energy, thermalization, convergence, TSU execution,
Kona execution, or production readiness. Python/Rust/PyO3 parity receipts MAY
prove operation/state/status/error equality for the trace, but physical
execution requires a separate authenticated route and is out of scope.

Required artifact fields:

- `fixed_width_operation_trace_path_and_hash`
- `python_rust_pyo3_trace_parity`
- `inference_substrate`
- `honest_verdict`

Required field principles:

- `fixed_width_operation_trace_path_and_hash`: principle "Portability evidence is the immutable ABI trace, not a board claim."
- `python_rust_pyo3_trace_parity`: principle "Every operation, version, return value, and final hash agrees exactly."
- `inference_substrate`: principle "Use deterministic transactional replay with no LLM and no board execution."
- `honest_verdict`: principle "Use a terminal delayed-commit fixture prefix without performance or hardware-execution claims."

**Acceptance criteria:**
- `.venv/bin/python -m carnot.experiment_5967_delayed_commit_memory_fixture --date 20260803`
  writes `results/experiment_5967_delayed_commit_memory_fixture.json` and a
  hashed fixed-width operation trace.
- The trace covers delayed production writes and the matched same-event
  write-through control with identical capacity, retrieval policy, event
  order, and compute accounting.
- Python/Rust/PyO3 trace parity receipts match exactly for operation names,
  versions, return values, state hashes, final hashes, and error codes.
- The artifact states no hardware execution or performance claim and uses
  `inference_substrate="deterministic_delayed_commit_transactional_replay_no_llm"`.

**Implementation status:** Planned (Exp 5967)

---

### SCENARIO-HW-5967

**Scenario:** Exp5967 writes a portable delayed-commit trace without board claims.

**Given:** Exp5920, Exp5924, and Exp5926 are ready and no authenticated
hardware route is being exercised,
**When:** Exp5967 writes its delayed-commit fixed-width operation trace,
**Then:** the trace hash is recorded, Python/Rust/PyO3 trace parity is exact,
the same-event write-through arm is labelled as a control, and no board
execution, speedup, power, energy, thermalization, convergence, TSU, Kona, or
production-readiness claim is made.

**Implementation status:** Planned (Exp 5967)

---

### REQ-HW-5969

**Title:** Exp5969 attacked CSL ABI audit MUST report fixed-width portability without board or TSU execution claims

**Description:**
Experiment 5969 SHALL map the delayed-commit CSL attacked transaction trace to
the fixed-width ABI v2 operation vocabulary qualified by Exp5926 and Exp5967.
The mapping SHALL cover bounded poison, contradiction, drift, protected-prefix
replay, duplicate, capacity, crash/restart, ledger-tamper, and rollback attack
families with finite operation ids, state versions, event indices, payload
hashes, validator receipt hashes, return codes, rejection reasons, and resulting
state hashes.

The receipt is portability evidence only. Exp5969 MUST NOT claim attached-board
execution, TSU execution, speedup, power, energy, thermalization, convergence,
Kona execution, or production readiness. Python/Rust/PyO3 parity receipts MAY
prove operation/state/status/error equality for the attacked trace, but
physical execution requires a separate authenticated route and is out of scope.

Required artifact fields:

- `hardware_abi_mapping_receipt`
- `python_rust_pyo3_attacked_trace_parity`
- `inference_substrate`
- `honest_verdict`

Required field principles:

- `hardware_abi_mapping_receipt`: principle "Report fixed-width portability only; no attached-board or TSU execution/speed claim."
- `python_rust_pyo3_attacked_trace_parity`: principle "All operations, versions, reasons, and final hashes agree exactly."
- `inference_substrate`: principle "Use measured deterministic attacked state replay with no LLM."
- `honest_verdict`: principle "Use `complete_ready:`, `complete_partial:`, `retired:`, or `blocked:`."

**Acceptance criteria:**
- `.venv/bin/python -m carnot.experiment_5969_csl_poison_drift_abi_audit --date 20260803`
  writes `results/experiment_5969_csl_poison_drift_abi_audit.json`.
- The artifact maps every attacked trace operation to a bounded fixed-width ABI
  v2 request/response receipt.
- Python/Rust/PyO3 attacked trace parity receipts match exactly for operation
  names, versions, statuses, rejection reasons, state hashes, final hashes, and
  final energies.
- The artifact states no hardware execution or performance claim and uses
  `inference_substrate="deterministic_csl_poison_drift_abi_audit_no_llm"`.

**Implementation status:** Planned (Exp 5969)

---

### SCENARIO-HW-5969

**Scenario:** Exp5969 writes a portable attacked CSL ABI receipt without board claims.

**Given:** Exp5968 is prospectively ready and no authenticated hardware route
is being exercised,
**When:** Exp5969 writes the attacked CSL ABI mapping receipt,
**Then:** Python/Rust/PyO3 attacked-trace parity is exact, fixed-width fields
are recorded, unsupported operations fail closed, and no board execution,
speedup, power, energy, thermalization, convergence, TSU, Kona, or production
readiness claim is made.

**Implementation status:** Planned (Exp 5969)

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

**Implementation status:** Implemented (Exp 5861)

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

**Implementation status:** Implemented (Exp 5861)

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
