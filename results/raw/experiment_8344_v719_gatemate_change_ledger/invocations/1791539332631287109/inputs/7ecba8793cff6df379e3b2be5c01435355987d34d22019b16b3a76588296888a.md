# GateMate physical delta — 2026-10-08

Exp8274 is the evidence frontier, pinned to primary SHA256
`46e9174e6fe6949b1e88d991909a66e25477fa23225c44faa1c50e4bcfa0d241`.
The boundary is its recorded end clock, retaining all previously seen operator
receipt hashes. The original Exp6559 transcript remains byte-bound to SHA256
`59a76f8ab46fa24b1ebe9aa038dde2ccf35a32a348e02696409b03ff096c8e66`.
Its observed `0xffffffff` is a historical physical/JTAG block.

| Condition | Evidence | Consequence |
|---|---|---|
| Unchanged | No authenticated physical setup change after Exp8274 | One excluded obligation; complete_blocked_gatemate_physical_change |
| Changed | New dated operator cable, port, power, board or DirtyJTAG receipt after Exp8274 | Freeze the next probe contract; successful bring-up remains unproved |

Reopening requires **a documented setup change, authenticated GM1Ax IDCODE
`0x20000001`, then a flashed n16 tile with device sample/hash smoke evidence**.
Authenticate board presence, safe power/cable/port, permissions, onboard DirtyJTAG
`1209:c0ca` and required tools before any future probe. Its frozen detection
command is `openFPGALoader -c dirtyJtag --detect`, bounded at 30 seconds with
first-failure stop. An unchanged `0xffffffff` does not authorize another retry.

Authenticate the existing n16 RTL, constraints, bitstream and CPU smoke input and
output hashes before a bounded flash. Require device sample/hash smoke parity
against those same inputs/outputs, separate transfer/readout/sample clocks and
complete argv, exit, stdout and stderr evidence. A host bitstream or flash exit
does not establish board execution, sampler quality, acceleration or speedup.

Exp8288 compares only established operator receipt documents whose bytes changed
since the frontier. Previously seen receipts, plans, milestones and elapsed time
cannot qualify. Missing evidence is unavailable rather than a measured zero.
The artifact retains changed/unchanged/missing document rows and parser reasons.

This invocation loads no model, trains no head, retries no JTAG, flashes no board
and runs no device benchmark. Obligation readiness certifies evidence custody
and owned checks. Execution readiness, scientific benefit and both generalization
scores remain zero. Frozen private E2E-015/019, consumer, coverage, lint, type and
spec checks precede atomic publication; full-suite health remains separate.
The conductor reconciles ops/status.md, ops/changelog.md and traceability.
