# FR-11 Trace2Skill Daily Eval 1497

Run date: 20260507

## Cadence

Run the bounded trace2skill manifest build once per day after the query-time
memory replay artifacts are available. The cadence replays only measured
Exp 1484/1485 rows and writes `results/fr11_trace2skill_daily_eval_manifest_1497.jsonl` plus the terminal
experiment artifact.

## Promotion Rules

Promote a trace-derived skill only when the source artifacts are present, the
verifier signal resolves, the Exp 1485 zero-soundness policy is allowed, no rot
criterion fires, and the memory-assisted outcome improves the paired baseline.

## Retirement Rules

Retire a skill when any rot criterion fires: missing source artifact,
unresolved verifier dependency, reduced task success, new soundness mistake, or
schema drift. Retirement is counted separately from promotion.

## Boundaries

This is a bounded daily evaluation cadence over replayed FR-11 trace-derived
skills. It does not claim broad autonomous learning, fresh LLM-generated skill
discovery, or production-default memory routing beyond the measured suite.
