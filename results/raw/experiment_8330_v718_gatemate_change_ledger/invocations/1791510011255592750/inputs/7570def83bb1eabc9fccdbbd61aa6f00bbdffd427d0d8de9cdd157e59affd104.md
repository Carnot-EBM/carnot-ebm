# GateMate obligation — 2026-10-08

Exp8316 preserves the GateMate obligation using the exact declared Exp8302
primary at `results/experiment_8302_v716_gatemate_physical_delta.json`, SHA256
`fb847291bb18a922ed944e08f38b98d486a6dbc1dce29cf322155a7f61ee6f12`.
Its byte-bound publication sidecar is
`results/raw/experiment_8302_v716_gatemate_physical_delta/validators/fb847291bb18a922ed944e08f38b98d486a6dbc1dce29cf322155a7f61ee6f12.json`.
Its terminal/adversarial receipts are in
`results/raw/experiment_8302_v716_gatemate_physical_delta/invocations/1791469599712433075/terminal_validation.json`.
The original transcript is
`results/experiment_6559_gatemate_changed_state_continuity.json`, SHA256
`59a76f8ab46fa24b1ebe9aa038dde2ccf35a32a348e02696409b03ff096c8e66`.
The historical observed IDCODE is `0xffffffff`.

| Condition | Actual evidence paths | Consequence |
|---|---|---|
| Unchanged: no authenticated cable, port, power or board receipt after Exp8302's end-clock frontier | `ops/operator-followup.md`, `ops/hardware-bringup-prep.md`, `ops/known-issues.md`, `research-hardware-wishlist.md`; frozen document bytes, primitive parser rows and frontier under `results/raw/experiment_8316_v717_gatemate_obligation/` | One excluded hardware obligation; `complete_blocked_gatemate_physical_change`; successful audit readiness remains distinct from execution readiness zero |

Reopening requires a **documented physical cable, port, power or board change
with a dated operator receipt after the authenticated Exp8302 frontier,
authenticated GM1Ax IDCODE `0x20000001`, then n16 tile flash and device
sample/hash smoke**. A subsequent bring-up plan must name and hash the new
physical evidence. Authenticate the existing RTL, constraints, bitstream and CPU
smoke input/output hashes before a bounded flash; require device sample/hash
parity, transfer/readout/sample clocks, exact argv, normal exit and output hashes.
A host bitstream build or new planning date supplies no physical change.

This task makes zero JTAG calls, including when a new receipt is found. It loads
no model and executes no device workload. Missing evidence is unavailable;
external incompleteness is blocked. Obligation readiness certifies authenticated
history and owned checks, while execution readiness and both generalization
scores remain zero. Exposed cached development history provides no independent
generalization claim. Private E2E-018 authenticated-history cases, applicable
E2E-015/019, consumer tests, 100 percent new statement coverage, scoped lint,
types and spec references precede atomic publication. Repository-wide health
is not run in this bounded experiment. The conductor reconciles status,
changelog and traceability after completion.
