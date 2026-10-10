# V722 GateMate obligation delta

Exp8386 reads Exp8372 and only explicitly supplied receipts received after its
sealed terminal-check cutoff, Unix nanoseconds 1791608068368056749. The receipt
uses the existing `carnot.gatemate.supplied_evidence.v1` reader and additionally
binds `exp8372_primary_sha256` and `received_wall_ns`. The scan budget is five
minutes. No source reconstruction, JTAG detection, board flash or model load runs.

The two unavailable historical source hashes remain exact obligations:

- `python/carnot/reporting/gatemate_change_ledger_8344.py`:
  `4034850aa978b9fc3f63fe960ad2c586069a46eee826923cc9cfae94bca73f03`.
- `tests/python/test_gatemate_change_ledger_8344.py`:
  `72a713565f099e2885ecb73c72c44305270bf950e89a6079ae2c9a6acde206e6`.

Present source bytes cannot replace either historical operand. Exact supplied
bytes would queue source custody and future historical replay; this task does
not itself qualify history. Independent present physical evidence cannot repair
the source gap. Source recovery likewise cannot qualify physical execution.

The physical conditions remain separately testable and ordered:

1. A dated operator cable, port or power change receipt after the cutoff.
2. A new authenticated GM1Ax transcript with IDCODE `0x20000001`.
3. An authenticated n16 bitstream flash after valid device identification.
4. Device sample/hash smoke evidence after that flash.

The original `0xffffffff` observations remain preserved. An authenticated change
receipt queues the first condition without claiming any later device observation.
No new hardware runner is needed. This documentation fulfills CLAUDE.md's
Hardware-Task Continuity duty while retaining the unchanged physical block.

V717/V721 protocol bytes and closed utility procedures remain unchanged. Direct
V722 service measurements, numerical parity and semantic benefit remain separate.
Both generalization scores and current LLM/device command counts are zero.
Existing private E2E-018 authority/publication consumers and unchanged terminal
validators qualify the receipt. Owned failures disqualify it; missing external
evidence produces one complete_blocked terminal record. Global repository health
is recorded separately. The conductor owns ops and BMAD reconciliation.
