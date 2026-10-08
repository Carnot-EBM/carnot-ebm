# V717 cached sentence cohort, 2026-10-08

Exp8305 reconstructs existing live observations. This cohort is exposed
development data. The original fit, tune and reserved source clusters remain
disjoint within this run, but the reserved rows have informed prior work.
Neither independent generalization nor generalized learning benefit is claimed;
both scores remain zero. No model is loaded or called in Exp8305.

The authenticated Exp8304 protocol preserves the V707 roster: fit128, tune64,
reserved128. Tune slots1..32 calibrate and slots33..64 select comparators.
Reserved slots1..96 form the stream; slots9..96 form its later segment;
slots97..128 form retention. Missing slots remain in their original positions.

| Role | Intended | Feature rows | Labeled feature rows |
|---|---:|---:|---:|
| Fit | 128 | 105 | 104 |
| Calibration | 32 | 25 | 25 |
| Comparator selection | 32 | 28 | 28 |
| Reserved | 128 | 97 | 97 |
| Stream | 96 | 74 | 74 |
| Later stream | 88 | 67 | 67 |
| Retention | 32 | 23 | 23 |

Exp8182 completed188 of192 sentence transport rows. These do not imply188
usable feature rows: only158 also have the twelve historical public signals,
and157 have a usable original binary human target. One fit feature row lacks
a usable human target. Labeled fit has54 supported and50 unsupported sources.
Calibration has10 supported and15 unsupported; selection has15 supported and13
unsupported. Later stream has32 supported and35 unsupported sources;
retention has9 supported and14 unsupported sources. The terminal artifact
records every role's measured class support and every unavailable source slot.

The adapter independently reparses bound holistic and source-span replies and
recomputes lexical signals from original source and answer bytes. It appends
sentence unsupported mean, unsupported maximum, contradicted fraction and
baseless fraction at indices12..15, all bounded by[0,1]. These sentence
relations are model predictions. The only target is original response-level
human unsupportedness: y=0 supported, y=1 unsupported, or missing. Three-way
sentence truth is not inferred from the model's own judgments.

Predictor files contain an explicit allowlist, sixteen finite signals and byte
hashes. They contain no y, nested paired controls or label-derived fields.
Identifiers provide custody only and cannot be used as numeric features.
Predictor bytes are hashed and made read-only before any downstream consumer.
Evaluator shards and copied primitive inputs live in protected invocation
directories; evaluator files use owner-only permissions. Gold-label joins are
forbidden on prediction paths. Holistic x[0] permits a static fitted slope;
that slope remains frozen during later online updates.

Cached-cohort readiness concerns exact reconstruction and authenticated custody.
Fit readiness separately requires at least96 labeled fit sources and12 of each
label, plus24 labeled sources and4 of each label in each tune half. Missingness
cannot authorize different rosters, replacement calls or relaxed thresholds.
Failed owned validation disqualifies and clears both readiness fields; absent
external evidence blocks. Historical Qwen model hashes and original GPU
receipts remain imported evidence, with zero current model invocation credit.

Validation uses private E2E-015/019 CLIs, scoped unit and consumer tests, new-code
statement coverage, Ruff, strict mypy, spec coverage, fresh-process replay and
unchanged publication auditors. Ops and traceability reconciliation is owned
by the conductor after this task exits.
