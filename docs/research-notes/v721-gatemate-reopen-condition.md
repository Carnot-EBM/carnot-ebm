# GateMate reopening condition — 2026-10-10

Exp8372 records an external obligation. It does no JTAG retry or flash.
History authentication remains false. Execution readiness stays zero.

The authenticated Exp8357 failure names these exact missing producer bytes:

| Producer path | Required SHA256 |
| --- | --- |
| python/carnot/reporting/gatemate_change_ledger_8344.py | 4034850aa978b9fc3f63fe960ad2c586069a46eee826923cc9cfae94bca73f03 |
| tests/python/test_gatemate_change_ledger_8344.py | 72a713565f099e2885ecb73c72c44305270bf950e89a6079ae2c9a6acde206e6 |

The Exp8344 producer primary hash is 48eb3ad7bccb2d869d561c3e021b50348fed3339bddbce2d493442ce7108f73b.
Its validator receipt hash is 0aaf24b1780d6580d9e71efccbaf3e9f3142bec140e183597dc0d4f42bb72e56.
Its terminal receipt hash is 5cbaee62cce2150a755e3ac78684389aef51d49ef0dbf1d59149455723ab077b.
The content-addressed request below the Exp8372 raw directory carries their original paths.
Planning found neither exact source version among reachable commits. No unchanged history search is repeated.
Regenerated equivalent text cannot satisfy these hashes.

An operator may explicitly supply a new receipt with `--supplied-receipt PATH`.
Authorized evidence locations are an operator-supplied snapshot, producer archive, or operator physical receipt.
The receipt must bind Exp8357 and a date after the 2026-10-09 frontier.
Source rows must identify the original path, supplied path and exact expected hash.
All supplied evidence remains a dependency for a future qualified history recovery.

The physical obligation remains independent of missing source history:

1. Record a dated operator cable, port or power change.
2. Preserve the original `0xffffffff` transcript, Exp6559 SHA256 59a76f8ab46fa24b1ebe9aa038dde2ccf35a32a348e02696409b03ff096c8e66.
3. In a future bounded hardware task, obtain GM1Ax IDCODE `0x20000001`.
4. Authenticate and flash the n16 bitstream.
5. Record device samples and hash smoke parity.

A new physical receipt is queued for the next bounded hardware task.
This ledger does not promote it to physical completion.
Original V720 failures and V717 protocol bytes remain unchanged.
