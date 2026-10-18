# V714 coverage producer and consumer contract

REQ-REPORT-8262 and REQ-VERIFY-8262 govern this administrative qualification.
The V713 science protocol remains byte-identical at SHA-256
f4c06e17a9f8eb72f1be80b265ab7e3507cbc5f6f3b918df6421fe49dae02018.
The execution binding names all fourteen current producer paths and their
current gates. Historical experiment numbers do not become current gates.

The frozen validation manifest names the actual coverage JSON output with
`coverage json ... -o <private-path>`. The runner reads that operand and
preserves its bytes, actual command receipt and owned source bytes under
`results/raw/experiment_8262_v714_coverage_custody/invocations/<invocation>/coverage/`.
The report is `coverage.json`; its receipt is `coverage_command_receipt.json`.
Their byte hashes bind the cold reader. The report must be generated during
the actual child interval, identify every owned file, contain nonzero measured
statements and show complete coverage with zero exclusions. The reader checks
line sets against saved source bytes and recomputes both file and overall totals.
It never infers coverage from console output.

Scratch is removed before fresh CLI readers run. A valid report succeeds;
missing evidence and a rehashed contradictory report fail. Private scripted
validation children exercise the normal runner branch for missing, stale,
foreign, partial, failed and tampered coverage. No fixture shortcut qualifies
readiness. The reusable hook is `coverage_custody_8262.preserve`; its consumer
is `coverage_custody_8262.replay`.

Every V713 disposition is retained by exact path and hash: Exp8248 qualified
methods; Exp8249 disqualified custody; Exp8250 conductor pre-gate; Exp8251–8256
six absent primaries; Exp8257 no new outcomes; Exp8258 blocked capture; Exp8259
qualified board-local CPU dispatch; Exp8260 blocked physical change; Exp8261
qualified accounting with blocked science. Missing primaries have no invented
producer verdict. H1 and H2 remain unmeasured. Historical scientific failure
does not prevent qualification of a current coverage reader.

Both current scores measure administrative readiness. Both generalization
scores remain zero. No model loads, LLM calls, weight updates or external
publication occur. Required validation failures disqualify current work.
Bounded repository health remains a separate receipt. An existing health
receipt may be imported through `CARNOT8262_HEALTH_RECEIPT`; the importer checks
the exact full-suite argv and stream hashes and copies its logs durably.
The conductor owns ops and traceability reconciliation.
