# ARC redirect effect audit, 2026-10-14

Exp8496 contains sixteen adapter-withheld episodes from the actual
`make_carnot_agent` and `E3AgentPolicy` path. The four games are
`tn36-ef4dde99`, `lf52-271a04aa`, `ft09-0d8bbf25` and `ar25-0c556536`.
Each game has seeds 11 and 22, with the supervisor off and on.
The original publication hash is
`sha256:5804a065a668c89df631687b9277a963643848b61b640e59b8efc1ca72a62ceb`.
Its date remains 20261013. Its original frozen V730 authority controls replay.
The registry records these public games as fully cleared. This audit adds no solve credit.

There are sixteen shadow recommendations and sixteen applied-mode selections.
All thirty-two outcomes remain pending at the 512-action episode limit.
Shadow recommendations describe controls. They are neither treatments nor failed treatments.
Right censoring gives no failure-rate estimate.

The primitive `supervisor_events` records contain a snapshot and a redirect.
They do not contain the policy's application disposition, old/new state hashes,
recorded eligibility predicates or direct induction-attempt receipts.
The audit retains those missing fields. It joins recorded actions and observations
by episode and action index. The initial reset explains the one-step offset
between supervisor action index 120 and SDK step 121.
It recomputes the fixed eligibility predicates from recorded snapshots.
These recomputed predicates are labeled separately from recorded predicates.

The eight applied reinduction selections have a true induction latch.
The eight shadow reinduction selections also have a true latch.
The proposed explanation that the live redirect resets an already-false latch
is therefore contradicted at selection time.
Later snapshots show induction-attempt count two in applied mode and one in shadow mode.
Those counters establish tier reentry. They do not establish generation or a useful model.
The original harness disables induction and rejects any unskipped attempt.
It reports zero LLM invocations. The direct per-attempt skip receipt is missing.

Eight bounded private fixtures exercise the unchanged supervisor and real policy
methods. Each mechanism runs in shadow and applied modes:

| Mechanism | Applied fixture result | Shadow fixture result |
| --- | --- | --- |
| True induction latch | State changes; induction tier enters; `disabled_by_env` skips generation | State remains unchanged |
| False induction latch | Direct mutation control has no state change; selector excludes this state | State remains unchanged |
| Diversity inactive | Real policy enables diversity and moves its counter past the threshold | State remains unchanged |
| Application exception | Real policy records `exception:ValueError` and no successful effect | Recommendation causes no exception or mutation |

These fixtures qualify the mechanism diagnosis. They are separate from live rows.
Their state hashes do not fill gaps in historical live receipts.
The audit loads no LLM or numeric head. It uses no game source, adapters,
stored routes, offline BFS or game environment.

Leave-one-game-out support remains separate for each mode and arm.
Each held-out game leaves three overlapping games, but every shared cell has
zero resolved outcomes. The unchanged requirement of five resolutions per
shared game/arm cell fails. Refinement remains unauthorized.
The scientific verdict is terminal null. Both generalization-benefit scores
and headline solve credit remain zero.

The narrowly specified future change is passive receipt capture: enable the
already shipped `EligibilityReceiptRecorder` in the panel observer and retain
episode-end `induction_attempts`. Record predicates, application disposition,
old/new state hashes and the skip reason. Change no arm, priority or live default.
The missing primitive fields justify this proposal. This audit does not repeat
the unchanged panel. Other V731 outputs remain future dependencies.

Run the thin runner with:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=python:. PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/experiments/experiment_8509_v731_arc_redirect_effect_audit.py --date 20261014
```

Validation receipts and genuine Coverage.py shards live below
`results/raw/experiment_8509_v731_arc_redirect_effect_audit/`.
The terminal sidecar binds the exact primary bytes and records the actual consumer.
Historical E2E-018 and E2E-021 tests use isolated fixtures and original dates.
The conductor owns ops and traceability reconciliation for this invocation.
