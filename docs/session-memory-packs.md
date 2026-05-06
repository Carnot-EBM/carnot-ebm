# SessionMemory Portable Packs

Spec: REQ-LEARN-1405, REQ-LEARN-1406, REQ-LEARN-1407

Carnot stores learned `SessionMemory` locally as implementation state. Portable
packs wrap that state in `carnot.session_memory_pack.v1` JSON so learned cases,
constraint-template observations, and false-positive calibration can be shared
between installations or checked into a reproducibility bundle.

## Schema

The public schema lives at `python/carnot/schemas/session_memory_v1.json` and
uses JSON Schema draft-2020-12. A pack contains:

- Pack metadata: `schema_version`, `source`, `license`, `created_at`, and `carnot_version`.
- Per-model state: `model_id`, `safe_model_id`, `case_memory`, `template_library`,
  `constraint_templates`, `fp_tracker`, and `session_state`.
- Portable case summaries: `question_hash`, `canonical_form`, `observed_precision`,
  `n_observations`, and `last_seen`.

Version `1.x.y` is additive-only. Breaking changes require a new major version and
a migration path.

## CLI

```bash
carnot memory export --storage-dir .carnot_sessions --model-id qwen3 -o pack.json
carnot memory import pack.json --storage-dir .carnot_sessions --model-id qwen3 --merge
carnot memory import pack.json --storage-dir .carnot_sessions --model-id qwen3 --replace
carnot memory diff pack-a.json pack-b.json
```

`--merge` is the safe mode. Duplicate case entries are merged by adding support
counts and recomputing confidence as a support-weighted average. Template
observations and FP tracker counters are merged additively. `--replace` prints an
explicit reset warning before writing imported state.

## Starter Packs

Reference packs live under `examples/constraint_packs/`:

- `empty_v1.json`: valid blank pack for zero-state installs.
- `arithmetic_v1.json`: pre-warms carry, sign, and unit observations.
- `python_code_v1.json`: documents portable Python-code pattern observations for
  downstream custom template registration.
