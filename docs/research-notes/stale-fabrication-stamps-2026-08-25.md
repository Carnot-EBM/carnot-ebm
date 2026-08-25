# Stale fabrication-gate determinations: measurement and correction proposal

Date: 2026-08-25. Scope: `results/**`, 6,924 readable artifacts.
Status: measurement complete; provenance shipped; **correction of history NOT executed**.

## The defect

`flagged_adversarial` is the fabrication gate's verdict on an artifact. 565 artifacts carry
it as `true`. Before this note, none recorded which gate version produced it or when.

That is not cosmetic. `scripts/conductor_gates.py` blocks a downstream task when its upstream
carries the flag ("UPSTREAM IS QUARANTINED"). A determination made under rules that have since
changed was indistinguishable, on disk, from one made by the gate now running.

The origin is commit `82d8219adf`. The conductor runs as one long-lived `--loop` process and
did a plain import of `adversarial_verify`, so it judged artifacts with the module copy cached
at first use. exp6593 was stamped CRITICAL `DURATION_TOO_SHORT` by a 14-hour-stale gate under
a rule its own commit had already fixed. That commit fixed the reload. It did not re-judge
history, and it did not make an old stamp recognisable.

## Method

Every readable artifact under `results/` was re-judged with the gate as of 2026-08-25.
Read-only: nothing under `results/` was written. Runtime 140s for 6,924 artifacts.
The `--backfill` column applies the same `_claims_live_model` precision guard the backfill
itself applies, so it is comparable to what a real run would do.

## Direction A: stamped today, would no longer draw a critical flag

| | count |
|---|---|
| stamped `flagged_adversarial: true` | 565 |
| would still draw a CRITICAL flag | 365 |
| **would draw NO critical flag** | **200** |

The prior estimate of ~200 is confirmed exactly.

Of the 200: 164 draw no flag at all, 36 still draw warnings. Their ORIGINAL flags were
`DURATION_TOO_SHORT` (135), `METHODOLOGY_MISSING` (101), `TAUTOLOGY` (88),
`IMPLAUSIBLE_PERFECT` (23). By declared verdict, 136 are terminal successes
(`complete_*` 124, `success_*` 12) and 51 are `blocked_*`. So roughly **136 honest results
are currently excluded from aggregation** by a determination the current gate would not make.
The 51 `blocked_*` artifacts are honest non-terminal states that were never headline-eligible
anyway, so clearing them changes nothing downstream.

**Downstream blocking is currently zero.** The live roadmaps contain exactly one `gated_on`
upstream reference (`exp6593-cfr-independent-row-reducer` gating on `exp6592`), and exp6592
carries no flag. The blocking mechanism is real and the brief is right that it matters, but
at this moment no task is blocked by a stale stamp. The cost today is exclusion from
aggregation, not cascade-blocking.

**Corrigendum trail at risk:** 191 of the 200 carry `corrigendum_pending` and 23 carry
`corrigendum_note`. Any correction must preserve both;
`scripts/determination_preservation_lint.py` exists because a corrigendum was once lost this
way, and its `MARKER_PATTERNS` match `^flagged_adversarial`.

## Direction B: unstamped, but the current gate calls them CRITICAL

This is the larger number and the more serious one.

| | count |
|---|---|
| unstamped artifacts | 6,359 |
| current gate returns a CRITICAL flag | **1,126** |
| backfill-eligible after the live-claim guard | 1,090 |
| of which `--high-precision-only` would stamp | **21** |

Decomposition of the 1,126:

| kind | count |
|---|---|
| `NONTERMINAL_DECLARED_ARTIFACT` | 995 |
| `TAUTOLOGY` | 133 |
| `DURATION_TOO_SHORT` | 119 |
| `MOAT_CLAIM_RIGOR` | 13 |
| other | 5 |

864 of the 1,126 carry `NONTERMINAL_DECLARED_ARTIFACT` and nothing else. That check landed
**2026-08-09** — 16 days ago — and has never run against history.

This is the generalisation of the incident. The origin commit noted that "any fabrication
check added in those 14h was equally inert." The measurement shows the same hole across the
whole history, not just a 14-hour window: **a check added at time T has never judged any
artifact stamped before T.** A stale stamp that should clear merely quarantines honest work.
A check that never ran means fabrication was never looked for.

## Correction of the brief

The brief proposed `--backfill --apply` as the sanctioned writer for correcting the 565.
It cannot do that. `backfill_stamps` begins:

```python
if not isinstance(d, dict) or d.get("flagged_adversarial"):
    continue
```

It **skips any artifact that already carries the flag**. It can only ADD determinations, never
revise or clear one. So `--backfill --apply` would do nothing at all to the 200, and would
instead stamp ~1,090 currently-clean artifacts. These are two different operations on two
disjoint sets, and only one of them has a tool. A regression test pins this
(`test_backfill_cannot_revise_an_existing_determination`).

## What shipped (REQ-VERIFY-6601)

`scripts/stamp_provenance.py`. Every gate write of `flagged_adversarial: true` now records:

```json
"flagged_adversarial_provenance": {
  "gate_version": "<sha256 of the gate's semantic fingerprint>",
  "gate_version_algo": "ast_normalized_docstring_stripped_sha256_v1",
  "stamped_at": "2026-08-25T16:04:11Z",
  "stamper": "research_conductor.completion_gate"
}
```

The version is a SEMANTIC fingerprint, not a content hash: the source is parsed, docstrings
stripped, the AST re-unparsed, then hashed. A raw file hash would move on every comment edit
and every artifact would read stale forever — the check-that-cries-wolf failure. A
comment-only or docstring-only edit does not move it; a change to executable logic does. Both
properties are under test.

Staleness is decided in O(1) per artifact with no gate run, so the whole corpus classifies in
under a second rather than the 140s a re-judge takes. `conductor_gates.py` now names the
staleness in its quarantine message and still blocks. It never silently re-verifies and never
lifts a quarantine.

Current corpus state under the new check: **565 unversioned, 0 current, 0 stale.**

## Proposal (operator decision — NOT executed)

Two decisions on two disjoint sets. They should not be taken together.

### Decision A — the 200 stale-positive determinations

**Recommendation: append-only correction, not a re-stamp. Batch it, do not do all 200.**

The correction convention already exists and is machine-checked by
`determination_preservation_lint.py`: keep the field, set it `false`, and add an accompanying
`flagged_adversarial_cleared_note`. A silent transition to absent/None is refused, because that
is indistinguishable from the accident the lint was written for.

Why append-only rather than a bare re-stamp:

1. **A re-stamp asserts something not measured.** "The current gate returns no critical flag"
   is not the same claim as "the original determination was wrong." Several of these were
   correct calls under rules later *relaxed* — the substrate allowlist has been widened
   repeatedly. An append-only note can state the measured fact without asserting the original
   reviewer erred.
2. **Never-prune.** 191 of the 200 carry a `corrigendum_pending` trail. A re-stamp that drops
   it repeats the 2026-07-27 incident by hand.
3. **The lint would refuse the silent form anyway**, so the append-only shape is the only one
   that can be committed.

There is no tool for this. It needs a small purpose-built writer, and it should run on a
reviewed subset first — the 136 terminal-success artifacts are where the value is; the 51
`blocked_*` ones change nothing and can be left alone.

The 7-artifact `flagged_adversarial_restoration_note` precedent is a *different* operation —
restoring fields destroyed by an overwrite, not clearing a determination — so it is a
precedent for the append-only STYLE, not for this action.

### Decision B — the 1,126 unstamped-critical artifacts

**Recommendation: do not run the default backfill. Run the high-precision form, and refer
`NONTERMINAL_DECLARED_ARTIFACT` to the operator separately.**

The exact commands, dry-run first in both cases:

```bash
# Conservative: 21 artifacts, DURATION_TOO_SHORT only, the guard's own trusted subset.
.venv/bin/python scripts/adversarial_verify.py --backfill --high-precision-only
.venv/bin/python scripts/adversarial_verify.py --backfill --high-precision-only --apply

# Default: 1,090 artifacts. NOT recommended without adjudicating the 995 below.
.venv/bin/python scripts/adversarial_verify.py --backfill
```

`HIGH_PRECISION_KINDS` exists precisely because retroactive stamping over-flags, and its own
comment says TAUTOLOGY is excluded for that reason. 21 artifacts is a reviewable blast radius.
1,090 is not.

The 995 `NONTERMINAL_DECLARED_ARTIFACT` artifacts are a policy question, not a mechanical one:
should a check added 2026-08-09 quarantine 16 days of prior history? It is the same shape as
the 102-artifact duration-floor gap that commit `82d8219adf` deliberately left open for the
operator. Recording it here rather than acting on it.

### Prevention (the part that outlasts both decisions)

Provenance shipped above closes the *detection* half permanently. The remaining hole is that a
newly added check still never runs against history, silently. The natural follow-on is a
milestone-close report of `stale` + `unversioned` counts, so the number is in front of someone
rather than needing a session like this one to discover. Not built here.

## Cross-references

- `scripts/stamp_provenance.py` — the mechanism
- `tests/python/test_stamp_provenance_stale_gate_6601.py` — 19 tests, 8 mutation proofs
- `openspec/capabilities/verification/spec.md` — REQ-VERIFY-6601 + 6 SCENARIOs
- commit `82d8219adf` — the reload fix; the origin incident
- `scripts/determination_preservation_lint.py` — the clearing convention Decision A must use
- CLAUDE.md "Test-Run Record Integrity Discipline" — why `results/` is evidence
