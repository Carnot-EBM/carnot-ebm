# GateMate physical reopening contract — 2026-10-07

GateMate A1-EVB-2M retains the historical JTAG observation `0xffffffff`.
Exp8216 authenticates the Exp6559 continuity source; neither artifact proves
current reachability or a qualified n16 device workload. Exp3866 remains
historical and cannot supply missing readback or timing acceptance.

An operator must record a material cable, USB port, power, board or DirtyJTAG
change. The structured receipt must identify the exact board, confirm presence,
describe power and USB/JTAG cable state, name the host path, list changed physical
fields, set `operator_authored: true`, and give a receipt date strictly after
2026-08-23 and no later than the audit date. Prose plans, USB enumeration,
software rebuilds and elapsed wall-clock time do not establish a physical change.
Receipt locations are `ops/known-issues.md`, `research-hardware-wishlist.md`,
`ops/hardware-bringup-prep.md` and `ops/operator-followup.md`. Exp8232 reads them
in dry-run mode and runs no JTAG command, even when a valid change is recorded.

A future device experiment must authenticate that receipt and preflight board
identity, onboard DirtyJTAG (`1209:c0ca`), safe power/cable/port setup, permissions,
tool identity and bounded process cleanup. The Olimex board uses its onboard
programmer over USB-C; the external-adapter wiring document is reference only.
The future detection transcript must contain the expected GM1Ax IDCODE
`0x20000001`, with actual argv, normal exit, monotonic clocks and complete
stdout/stderr SHA256 values. `0xffffffff` is a failed chain observation.

After that preflight, authenticate an existing n16 Ising tile's RTL, constraints,
bitstream and CPU reference input/output hashes. Record a successful bounded
flash to the identified board, then an on-device tile/hash smoke whose output
matches the frozen CPU reference for the same inputs. Preserve the full device
transcript and sample-level timing, including transfer and readout. A flash exit
alone does not establish execution, output parity, sampler quality or speedup.
Host file presence, synthesis and local hash checks remain host evidence. Missing
device readback is unavailable evidence and keeps terminal device acceptance false.

Audit readiness certifies this documentation and authenticated custody. Execution
readiness remains zero until future preflight and smoke evidence passes. This
audit reports no latency, energy, learning or independent generalization benefit.
