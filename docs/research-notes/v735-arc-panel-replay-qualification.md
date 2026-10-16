# V735 ARC panel replay qualification

Exp8561 qualifies execution receipts for stored ARC episodes. The terminal
[artifact](../../results/experiment_8561_v735_arc_panel_replay_qualification.json)
and its bound validation sidecar contain the measured verdict. Primitive rows,
command receipts and coverage data are under
`results/raw/experiment_8561_v735_arc_panel_replay_qualification/`.

V734's sealed primary records a successful 134.137591152-second terminal replay
under the 240-second deadline. Its combined data and child shards agree. The
coverage report has no missing lines, but the declared statement sets contain
four absent anchors: lines 78, 370 and 377 in `arc_panel_replay_8548.py`, and
line 384 in its runner. Exp8561 retains the original source bytes and recomputes
those differences from the original combined data and each authenticated shard.

The repair names each original predicate and puts its branch on one line. The
predicate expressions, rejection bodies, source checks, statement containment
and exact shard-union checks remain in force. Real sysmon children exercise both
truth values in the original multiline form and the explicit predicate form.
The same collector is used throughout. Historical shards are never rewritten.

The invocation reuses the original reducers and qualification fixtures. It runs
stored-byte cold replay, corruption and authority controls, capability controls,
private CLI publication, absent input and rehashed row rejection. Original test
IDs are collected before disjoint bounded children, including private E2E-018
and E2E-021. Only current-invocation coverage shards enter the new coverage union.
The actual Exp8562 gate reads exact candidate bytes before atomic publication.

`inference_substrate_class=no_model_load`, `MODEL_SPECS=[]`, and all current
model invocation counts are zero. Generator identities remain historical
provenance. No fresh episode, policy benefit, learning, native speed, independent
generalization or new solve is measured. Exp8537 and Exp8548 remain disqualified.
The successful replay does not clear their earlier failed checks. The separate
Exp8529 exposed source-information null remains unchanged.

Reproduce with:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=python:. PYTHONUNBUFFERED=1 \
  .venv/bin/python -u \
  scripts/experiments/experiment_8561_v735_arc_panel_replay_qualification.py \
  --date 20261016
```

Ops status, changelog and traceability reconciliation are conductor-owned for
this task, as directed by the final task instruction.
