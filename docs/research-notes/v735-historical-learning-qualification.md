# V735 historical learning qualification

Date: 20261015. Producer: Exp8554.

This study qualifies exposed historical validation mechanics. It makes zero
current model calls. Learning benefit, native speed and capability-mask benefit
remain unmeasured. Exp8529 remains a qualified exposed source-information null.
Exp8541, Exp8545 and Exp8548 retain their quarantines.

The fixed Exp8541 primary hash is
`sha256:9321fc009824b58b7345df2212ea7d5ebef4cea6df0db7acc0b67359edb89d75`.
Its original union contains 158 test nodes. Its family-wide timeout followed
152 passes. The remaining six nodes did not have recorded assertion failures.

Exp8554 collects the original nodes before sharding. Each suite uses the shipped
Exp8541 adapter and its own original sealed producer authority. The interrupted
retention node runs alone first. The 24-node capacity suite uses groups of at
most four nodes. Each process-exit test and retention fixture runs alone.
Each child has a 480-second maximum deadline and a 30-second heartbeat.
The union budget is 2400 seconds. Measurements stop by 3600 seconds, with at
least 900 seconds reserved for validation inside the 4800-second cap.

Each child retains its exact command, exit, duration, stdout and stderr hashes.
Pytest XML supplies per-node elapsed times. A killed child cannot qualify.
Cold replay rebuilds the result from primitive evidence and rejects changed
summaries, commands, receipts, sources and timings. The unchanged publication
validator and downstream accounting consumers check the actual result bytes.

The terminal result and its bound sidecar contain the measured outcome:
`results/experiment_8554_v735_historical_learning_qualification.json`.
Primitive evidence lives below the matching directory in `results/raw/`.
Later V735 producers are dependencies, not inputs to this qualification.

Run:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=python:. PYTHONUNBUFFERED=1 \
  .venv/bin/python -u \
  scripts/experiments/experiment_8554_v735_historical_learning_qualification.py \
  --date 20261015
```

The conductor reconciles changelog, status and traceability after this task.
