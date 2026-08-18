---
type: Bundle Index
title: Carnot Knowledge Bundle
description: Curated claims and submission records for the ARC-AGI-3 track, in Open Knowledge Format.
tags: [okf, provenance, arc-agi-3]
generated: { by: outer-loop/claude-opus-5, at: 2026-08-18T05:25:00Z }
status: draft
---

# What this bundle is for

`results/**` holds machine-generated evidence and is read-only. `ops/changelog.md` holds a
chronological narrative. Neither answers the question that kept coming up on 2026-08-17:
**which claims are still standing, and what does each one rest on?**

Six stores currently hold pieces of that answer with no join key between them —
`ops/arc_flag_ledger.yaml`, `ops/exclusion_manifest.yaml`, the doomed-rerun ledger,
`research-complete.yaml`, `results/**`, and `ops/*.md`. Reconstructing "did the configuration
that scored 0.12 differ from the one that scored 0.08, and in which flags" is archaeology.
This bundle is the curated layer above those stores.

Format: [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md)
v0.2 — plain `.md`, YAML frontmatter, git-native, no central authority and no required tooling.
Chosen because three of its optional field families already match disciplines this project
enforces by hand: `sources` (provenance), `generated`/`verified` (the fabrication gate), and
`status`/`stale_after` (never-prune and the artifact-freshness lint).

## Why retractions are the seed content

A retracted claim is the hardest thing to reconstruct later and the most expensive to get
wrong, because the natural failure is that someone re-proposes a refuted lever. Four claims
were withdrawn on 2026-08-17 and each currently survives only in a commit body. `status:
deprecated` plus a link to what superseded it is exactly the shape that record wants.

**These documents are DEPRECATED ON PURPOSE.** They are kept, not deleted, per the project's
never-prune rule. Read the `superseded_by` field before acting on anything here.

## Claims

| Document | Status |
|---|---|
| [Marginal-band repair gains nothing](/okf/claims/marginal-band-null.md) | deprecated |
| [Cross-level engine carry is a null](/okf/claims/cross-level-carry-null.md) | deprecated |
| [Tool-assisted repair is worth +0.054](/okf/claims/repair-delta-0054.md) | deprecated |
| [A near-noise seed anchors induction](/okf/claims/seed-anchoring-mechanism.md) | deprecated |
| [Action budget can be pooled across games](/okf/claims/budget-pooling.md) | deprecated |
| [The goal predicate never fires on a real win](/okf/claims/goal-predicate-never-fires.md) | stable |
| [Seeded repair converts catastrophic cells](/okf/claims/seeded-repair-converts.md) | stable |
| [Carried engines do not transfer](/okf/claims/carried-engines-do-not-transfer.md) | stable |

## Submissions

| Document | Score |
|---|---|
| [2026-06-30 submission](/okf/submissions/2026-06-30.md) | 0.08 |
| [2026-07-15 submission, kernel v9](/okf/submissions/2026-07-15-kernel-v9.md) | 0.12 |

## Known gaps in this bundle

Stated plainly so nobody mistakes the seed for the system:

- OKF links are **untyped** — the spec puts relationship semantics in prose. `superseded_by`
  here is a local convention, not something the format validates.
- The spec is deliberately permissive and says consumers must tolerate broken cross-links.
  A lint that FAILS on a dangling link would be this project being stricter than OKF, which
  is allowed but is ours to build and does not exist yet.
- Only two submissions have ever been scored, so the configuration-to-score record this
  bundle wants to support has n=2 and cannot yet support attribution of any kind.
- No lint enforces that a claim document's numbers match the artifacts it cites.
