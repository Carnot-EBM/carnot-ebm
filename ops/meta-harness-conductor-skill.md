# Meta-Harness Conductor Skill

**Experiment:** 1281 meta-harness conductor search
**Status:** Complete

Use this guide when proposing or evaluating Carnot conductor-policy variants.
The goal is to improve harness behavior using full trace history, not only
scalar result artifacts.

## Inputs

- Charter: `ops/conductor-runtime-charter.md`
- Eval suite: `ops/conductor-harness-eval-suite.md`
- Search script: `scripts/meta_harness_conductor_search.py`
- Trace store: `meta_harness_runs/`
- Terminal artifact:
  `results/experiment_1281_meta_harness_conductor_search.json`

## Loop

1. Read the charter and eval-suite docs.
2. Inspect prior candidate directories under `meta_harness_runs/`.
3. Propose a policy variant with explicit capabilities and acceptance objects.
4. Evaluate it with `scripts/meta_harness_conductor_search.py`.
5. Preserve policy text, scores, trace files, verifier outputs, and final
   candidate artifacts.
6. Compare against the Pareto frontier in `meta_harness_runs/frontier.json`.
7. Recommend policy changes only when backed by candidate traces.

## Candidate Policy Dimensions

- terminal artifact rules
- gate-block behavior
- bootstrap artifact detection
- stale skeleton detection
- retry or retire policy
- evidence logging policy
- verifier mismatch handling
- pre-run environment bootstrap
- task packet format
- targeted-test policy
- result schema validation
- paper-claim audit behavior

## Leakage Rule

Candidate policies must not hard-code experiment ids or task-specific strings to
pass the eval suite. The search script audits policy text, candidate names,
capabilities, and recommendations for `expNNN`-style leakage before allowing a
clean terminal artifact.

## Command

```bash
.venv/bin/python scripts/meta_harness_conductor_search.py \
  --trace-store meta_harness_runs \
  --result results/experiment_1281_meta_harness_conductor_search.json
```
