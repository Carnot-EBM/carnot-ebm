# V750 deadline observer control

Exp8750 uses no model. It compares the unchanged original deadline pytest nodes
in an unprofiled child and a child with repeated 60-second faulthandler dumps.
Both children use the same pytest boundary observer and coverage configuration.
Parent process/resource samples run every 30 seconds. Each arm stops at 420
seconds. There is no authentication callback, recursive frame inspection or
numerical-clock replacement beyond the original test's finite deadline clock.

The original Exp8724 observation is a 300-second timeout, exit -9, with empty
streams. The later Exp8738 observation is exit -11 after 148.406120203 seconds
under profiling. Its saved timer dumps show the old profile callback on the
stack. These are separate observations. They do not establish why the original
child exceeded its budget. The shipped supervisor supplies both checkout import
roots, including for drivers outside the checkout; this repair is reused.

Before each measured arm, the method binds original fixture operands, test node
IDs, source closure, configuration, environment and actual argv. Boundary events
label fixture setup and pytest case execution. Authentication, copy, publication
and replay durations remain unknown without independent boundary evidence.
Original 3599/3600/3601 assertions are retained. No real one-hour wait is used.

No correction or cache is authorized without measured repeatable waste. An
unresolved cause yields a terminal diagnostic null and deadline readiness zero.
It does not rehabilitate Exp8738 or authorize another blind timeout.

Validation covers new statements, including real CLI children, scoped clock
consumers, private E2E-018/021 and E2E-020. Repository health is recorded as a
separate bounded diagnostic. A stopped or failed global run is not a full-suite
pass and does not substitute for the required scoped checks. Publication uses
unchanged adversarial scanning, strict row lint and a fresh cold CLI replay.

The conductor owns ops/status, changelog and traceability reconciliation.

## Owned observer failure and one correction

The first unprofiled child exceeded the intended 420-second cap. The legacy
supervisor tried to parse this observer's boundary-event log as a historical
profile-event log and raised `KeyError: event` before cleanup. Its parent exited.
The child was subsequently killed by its owned process group. The retained
receipt records an actual lifetime of 748.50329205 seconds, 0.01-second process
start-clock resolution, cleanup, and an unknown wait status. It does not invent
an observed exit code from the signal sent. Its first numerical case completed
only after the budget, so that output remains ineligible. All three intended
unprofiled controls remain censored for qualification.

The failed invocation is retained under
`results/raw/experiment_8750_v750_deadline_observer_control/invocations/1793672403459757663/`.
Original observer and test source bytes were copied before the correction and
checked against the pre-measurement source hashes. The failed arm is authenticated
and reused as failure evidence; it is never run again to obtain a better result.

The single correction isolates the legacy supervisor's process-log directory
from the pytest boundary log. It changes no authentication callback, fixture,
numerical deadline, original assertion or shared validator. New regressions
reproduce the failure, require timeout cleanup, reject changed retained method
and source bytes, and prohibit another unprofiled launch. The first timer arm
uses the corrected parent observer. Retained failed supervision disqualifies
execution regardless of later test results. Original timeout causation remains
unresolved; neither historical producer is rehabilitated.

## Terminal result

Published `complete_disqualified_v750_deadline_observer_control`; readiness is zero. All sixteen new tests pass. Real unit/CLI/driver SQLite covers 273/273 module statements and 6/6 CLI statements, with no exclusions. Ruff, formatting, strict mypy, spec tracing and private E2E-018/021 pass. E2E-020 and clock consumers reach their 300-second validation bounds; clock failures are visible. The one repository-health run reaches its 180-second bound without a full-suite pass.

Final publication restores the actual independent child seal and rebuilds gate ordering from serialized canonical JSON. Measurement files, original assertions, coverage and source bytes stay unchanged. The original terminal helper overwrote its failed replay receipt when retried; the primitive records that retention failure and original report hash without claiming retained-byte authentication. Adversarial scanning, strict row lint, both production readers and fresh-process replay pass for the final disqualified bytes. No further deadline run is authorized.
