# V751 ordinary durable cost — 2026-11-03

This study measures the unchanged bounded workload after archived lifecycle
qualification. Exp8771 ordinary readiness is the admission gate. Its constructed
threshold score is separate. The full 122880-update study remains closed after
Exp8732 projected at least 71507.5252224 seconds. Exp8745's assumed dense middle
event was absent, and V750 cost execution was skipped after lifecycle qualification
failed. No dense event is injected into these ordinary windows.

The fixed design contains ten paired repeats of three 32-update windows at
offsets 0, 2032 and 4064, across Python, stateless native and resident native arms:
90 process cells, 120 child lifetimes with the prescribed restart after update
2048, and 2880 intended durable updates. Arm order rotates within each repeat.
The seed is 7518772. The same 10000 frozen repeat-block bootstrap draws resample
all windows and arms together. Repeats do not create independent scientific cases.

The worker reuses `resident_update_cost_8732.drive` and the shipped durable Store.
Two commits per update preserve prediction sealing before feedback and durable
acknowledgment after snapshot/pointer file and directory fsync. Successful fsync
calls are counted around the actual implementation. Prefix construction is timed
separately. Complete-window time includes setup, state import, child launch,
prescribed restart, durable updates, acknowledgment and shutdown. Arithmetic and
per-update latencies remain separate from complete-window costs.

Before measurement, the runner freezes source/input hashes, installed binary
identity, exact cost protocol, test node IDs, roster and bootstrap draws. The real
four-update admission probe runs on persistent storage. The complete projected
fixed work plus 25 percent reserve must fit 3000 seconds. A failed admission gate
retains all 90 windows and 2880 update slots with explicit missing metrics.

Bounded resident benefit requires the stateless-native comparison's complete
throughput lower95 above 1.05 and p95 regression no greater than five percent.
The Python comparison is reported separately. These bounded windows establish no
uninterrupted throughput, useful learning, independent generalization or physical
acceleration. No language model is loaded or called.

Owned validation retains exact argv, exits, original node outcomes, JUnit, child
coverage SQLite and log hashes. Coverage applies only to the two new modules and
thin CLI, including real children. Real native parity/serialization and private
E2E-018/021 checks retain their original assertions. One bounded invocation of
`.venv/bin/pytest tests/python -q` is separate repository health, without a
full-suite pass claim. Publication requires fresh-process cold replay and the
unchanged adversarial and strict row scanners on the terminal candidate bytes.

Ops status, changelog and BMAD reconciliation belong to the conductor for this task.

## Measured outcome

Invocation `1793723299871513279` completed all 90 windows and 2880 updates,
using 120 child lifetimes. Measurement ran from 16:28:19.871524 UTC to
17:02:54.952482 UTC for 2074.787650194 seconds. No cell failed or was censored.
Actual natural dense-event count was zero. The admission projection including
25 percent reserve was 2951.63135666875 seconds, below 3000.

| Arm | Mean complete 32-update window (s) | Mean durable update (ms) | Mean arithmetic step (ms) |
| --- | ---: | ---: | ---: |
| python | 22.736414 | 492.695177 | 1.068207 |
| json_native | 22.662667 | 495.564452 | 2.298435 |
| resident_native | 22.253392 | 492.483899 | 1.042468 |

Resident/stateless-native complete throughput was 1.043247x, with paired
95 percent interval 0.944142–1.138601. Its p95 ratio was 2.355170, a
135.517 percent regression. Both bounded-benefit gates fail. The p95 statistic
is the maximum, across windows, of the mean paired-repeat p95 ratios.

Independently, resident/Python complete throughput was 1.054651x, with paired
95 percent interval 0.993848–1.129328; its p95 ratio was 1.160394. This does
not establish a reliable Python-relative gain. Intervals use the same frozen
10000 repeat-block draws. Neither comparison supports a long-run claim.

The numerical steps above are real fixed-trajectory replay work. No new learner
is trained or promoted. Final execution acceptance and terminal publication
remain subject to the retained validation receipts.

## Terminal acceptance

Published `complete_null_ordinary_durable_cost` with ordinary cost readiness 1
and bounded durable benefit 0. All 26 new tests and the scoped consumers/E2Es
pass. Coverage is 360/360 new executable statements, including real CLI children.
Ruff, format, strict mypy and requirement tracing pass. Fresh-process cold replay,
unchanged adversarial verification and strict row validation accept the terminal
bytes. The separate full Python suite diagnostic timed out at 180 seconds
(exit -9); no full-suite pass is claimed, and it was not rerun.

Throughput estimates average paired cell ratios with equal weight for each
window and repeat. The frozen bootstrap resamples whole repeat blocks.

Primary: `results/experiment_8772_v751_ordinary_durable_cost.json`

Primary hash: `sha256:b8087c8d53324d46d419147bd31ac8797e05437123e6cf8ef6a9e85bf4911ccd`

Validation sidecar: `/home/ianblenke/github.com/ianblenke/carnot/results/raw/experiment_8772_v751_ordinary_durable_cost/invocations/1793723299871513279/terminal_validation.json`
