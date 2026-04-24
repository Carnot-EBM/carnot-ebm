# Portable JSON schema for SessionMemory export/import

**Status:** Draft change proposal.
**Origin:** [GitHub issue #5](https://github.com/Carnot-EBM/carnot-ebm/issues/5) (2026-04-24).
**Target milestone:** 2026.04.63 or .64.
**Priority:** Medium. Unblocks community/enterprise deployment patterns
  (pre-warm packs, cross-installation sharing, reproducibility pinning).
**Depends on:** existing `SessionMemory` + `CaseMemory` + `ConstraintTemplateLibrary`
  (all live via Exps 345/344/456).

## Summary

`SessionMemory` persists learned constraints and case memory across
restarts within a single installation (FR-11 relay). The backing store is
implementation-internal — consumers have no documented way to ship a
pre-warmed constraint pack, share learned patterns between installations,
or feed external learning systems.

Define a JSON Schema (draft-2020-12 compatible) for SessionMemory + CaseMemory
+ constraint-template contents, with CLI and Python APIs for export/import.

See issue #5 for use-case scenarios (starter constraint packs,
cross-installation learning, reproducibility, auditability).

## Proposed experiments

### Exp A — Draft the JSON Schema + example packs

**Deliverable:** `python/carnot/schemas/session_memory_v1.json` (JSON Schema
draft-2020-12) + `examples/constraint_packs/arithmetic_v1.json` + 
`results/experiment_<N>_session_memory_schema.json`.

**Schema must cover:**

- Pack-level metadata (version, creation_date, source, license, carnot_version)
- Constraint templates (id, target_claim_type, regex/AST patterns, energy weight)
- CaseMemory entries (question_hash, canonical_form, observed_precision,
  n_observations, last_seen)
- SessionMemory state (rolling false-positive rates, constraint activation counts)
- Schema version field with backwards-compat policy

**Acceptance gates:**

1. Schema validates every entry currently in the dev installation's
   SessionMemory (round-trip: export → validate → import → diff = ∅).
2. An example "starter pack" (arithmetic_v1) loads on a fresh install and
   visibly raises the floor of extraction recall on a benchmark without
   additional training.
3. Schema-breaking changes require a major-version bump + migration script.

### Exp B — CLI `carnot memory export` / `carnot memory import`

**Deliverable:** CLI entry points under `python/carnot/cli/memory.py`.

**Acceptance gates:**

1. `carnot memory export --format json -o pack.json` produces a file that
   the schema validator accepts.
2. `carnot memory import pack.json --merge` merges without clobbering
   existing entries (observed_precision is re-computed, not replaced).
3. `--replace` flag available for full-reset imports; prints a
   loud-confirmation before proceeding.

### Exp C — Community starter packs

Ship 2-3 reference packs in `examples/constraint_packs/`:

- `arithmetic_v1.json` — numeric claim patterns (arithmetic, unit, bounds)
- `python_code_v1.json` — Python-specific extraction patterns (from Exp 764's
  AST verifier work)
- `empty_v1.json` — blank pack for users who want to start from zero

## Risks

- **Schema evolution.** Version 1 is going to be wrong in subtle ways.
  Mitigation: explicit `schema_version` field + migration script per
  breaking change; deprecate slowly.
- **Cross-installation trust.** Importing a pack from an untrusted source is
  an injection vector — the patterns themselves are regexes or AST specs
  that get executed. Mitigation: imports run inside the same sandbox as
  LLM-generated code (`CARNOT_USE_SANDBOX=1`), and the dogfood MCP guard
  (see `conductor-self-protection-safeguard.md`) screens pack contents
  before they're activated.
- **Packs drifting from production reality.** A starter pack that was great
  a year ago may be counterproductive on a newer model. Mitigation: track
  each pack's last-validated-against-benchmark date; surface stale packs
  in an audit command.
