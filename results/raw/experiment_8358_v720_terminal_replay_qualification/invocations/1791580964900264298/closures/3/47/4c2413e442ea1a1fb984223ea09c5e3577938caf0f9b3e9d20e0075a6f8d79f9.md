# GateMate physical delta — 2026-10-07

Exp8246 is the evidence frontier. Its primary SHA256 is
`f74888fb3cd4d6bcf585b7bbfe40d2977f8ecb9b8f2041383b9214466c351f4c`.
The frontier uses its recorded end clock, not a milestone or elapsed calendar
time. The original Exp6559 transcript SHA256 remains
`59a76f8ab46fa24b1ebe9aa038dde2ccf35a32a348e02696409b03ff096c8e66`.
Its `0xffffffff` observation establishes a historical physical/JTAG block.

| Condition | Evidence | Consequence |
|---|---|---|
| Unchanged | No authenticated operator setup change after Exp8246 | One excluded board obligation; complete_blocked_gatemate_physical_change |
| Changed | New operator cable, port, power, board or DirtyJTAG receipt after that frontier | Freeze a future probe contract; device bring-up remains unproved |

The four established operator receipt documents currently contain no new physical
change. Plans, passage of time, and host bitstreams establish no board execution.
The artifact compares each actual document hash with Exp8246's frozen copy and
preserves accepted/rejected candidates. Previously seen receipt hashes cannot
be counted again. Same-day evidence needs a timezone-bearing receipt_timestamp
strictly after Exp8246's actual end and no later than the current invocation.

Reopening requires a documented setup change, authenticated GM1Ax IDCODE
`0x20000001`, then a flashed n16 tile with sample/hash smoke evidence. Authenticate
board presence, safe power/cable/port, permissions, onboard DirtyJTAG `1209:c0ca`
and tools before any future probe. Its frozen command is
`openFPGALoader -c dirtyJtag --detect`, with a 30-second deadline. Stop on the
first failure. An unchanged `0xffffffff` never authorizes another retry.

Authenticate existing n16 RTL, constraints, bitstream and CPU sample input/output
hashes before a bounded flash. Require device sample/hash smoke parity with those
same inputs and outputs, and separate transfer/readout/sample clocks. A host
bitstream or flash exit alone proves no board execution, sampler quality or
speedup. Preserve argv, normal exits, clocks and complete stdout/stderr hashes.

Exp8260 runs no JTAG detection, flash, model load or benchmark. Audit readiness
certifies the obligation and evidence checks. Execution readiness, scientific
benefit and both generalization scores remain zero. Historical model provenance
creates no current LLM calls. Missing physical evidence is unavailable rather
than a newly measured zero. The conductor owns ops and traceability reconciliation.
