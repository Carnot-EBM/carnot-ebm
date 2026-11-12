# Exp9044 V772: original annotation feasibility (2026-11-12)

Repair note (2026-11-12): the earlier count-only annotation readiness claim is
superseded. Current code checks retained FAQ limits (EN/ZH 12, DE 4, others 3)
and blocks oversized rosters with `original_participation_authority`. ID-only
projection retains roster evidence even when span schema is invalid. Original
assignment/completion authority is required; empty spans cannot prove participation.
All 214 selected tests pass, with 226/226 owned statements and zero exclusions.
The publication and receipts linked below predate this repair and remain
historical evidence; no replacement source experiment was published.

Historical verdict: `complete_blocked_source_identity_ready_score` (`blocked`).
Historical annotation structure readiness: **1** (superseded).
Natural pilot readiness remains **0**.
This is a structural availability audit, not the400-group experiment.

The retained author release is revision
`3f99df059b001c174b4443e2c41840a5ff1320d7`, licensed CC BY4.0.
The reader authenticated the release files, license, author metadata, dataset
card and official FAQ instructions against retained hashes. The V771 preserved
design and queue have independent hard-coded hashes; the active roadmap was not
used as historical authority. Rejected Exp9015/9029 and the Exp9033 gate skip
remain unchanged. No new network request, model load or fit occurred.

| Official split | Records accessed | Original annotator-count histogram |
|---|---:|---|
| validation | 499 | {"3": 299, "4": 50, "53": 50, "6": 50, "95": 50} |
| test | 1902 | {"3": 896, "4": 550, "53": 150, "6": 152, "95": 154} |

## Per-record structural rows

[Every accessed record and its structural row](../../results/raw/experiment_9044_v772_annotation_feasibility/invocations/1794483052747942217/annotation_schema_rows.json) is retained as a
primitive JSON table (`sha256:5c9c787a40256cbc522fc365c1d8399b432a40e3b5f268208db49bd3924ba2d5`). Each row includes the original record
ID, official split, language, pseudonymous annotator IDs, total and distinct
nonempty counts, span-schema availability and explicit withheld outcome status.
Source groups remain unassigned. This table is the full per-record inventory,
not a sample or a new split. [Access receipts](../../results/raw/experiment_9044_v772_annotation_feasibility/invocations/1794483052747942217/access_receipts.json) enumerate the exact
projected fields and release shard hashes. The primary includes the same rows.

The only value columns read were `id`, `lang`, and
`annotations.list.element.annotator_id`. The span representation was inspected
from Arrow schema metadata: list of structs with a string annotator ID and a
list of integer `[start,end]` character spans. The reader never requested the
span leaf, soft labels, hard labels, text, logits or tokens. TEST annotation
values, span membership, disagreement and correctness were neither accessed
nor reduced. Actual span validity and class counts remain unknown.

## Original criterion and mapping

The retained dataset card describes original raw per-person annotations and
unique pseudonymous annotator IDs. The official FAQ states that people marked
minimal unsupported character spans relative to the supplied context/reference,
with conservative emphasis on content words. It also describes NO HALLUCINATION
and UNSURE comments. The released annotation struct has no comment, uncertainty
or adjudication fields. Their absence is unknown information, not a false flag.
The FAQ contains approximate instructions; a complete instruction document was
not retained. The audited structure score does not certify that missing material.

For a future authorized evaluator, let each original person vote1 if that
person marked any nonempty unsupported span, else0. `y_soft` is the fraction of
original people voting1; `y_hard = y_soft >=0.5`. Preserve all original IDs and
spans. Disjoint marks can produce unanimous response risk despite no common
character span. A character-majority union or soft-label array cannot reconstruct
these votes. Private Unicode/offset and disjoint-span tests exercise this rule
without opening TEST values.

The target is fallible human reference-support risk. It is not factual truth,
and missing reference support does not prove a claim false. The retained
[2610.08026 paper](../../results/raw/experiment_8388_v723_contract_methods/methods/2610.08026v1.html)
shows why criterion identity matters separately from human/automated source
identity; this audit transfers that concern, not the paper's factual labels.

## Exact requirements before any natural pilot

1. Close Exp9029's release identity and historical exposure joins. Its retained
   gate remains `source_identity_ready_score=0`, with2 release and213 historical
   unresolved identities. Source names alone cannot prove unseen eligibility.
2. Freeze source-disjoint unseen groups before label reduction, using unchanged
   salt `v770-human-v1`:160 FIT and40 CAL groups from official validation,200
   TEST groups from official test. No cohort was selected by this audit.
3. Authenticate original per-person span values and Unicode offsets only after
   sealing the cohort, using an evaluator boundary. Acquire the complete
   original instruction/comment/uncertainty records if uncertainty sensitivity
   requires them; do not fabricate those fields or substitute aggregate labels.
4. Establish at least5 examples per majority class in FIT and CAL and20 per
   class in TEST on that fixed cohort. This support is unknown here. Failure
   blocks the pilot without replacing records, changing targets or relaxing
   support. Keep all original V771 fitting and acceptance bounds unchanged.

The next planner can distinguish an annotation-schema limitation from source
closure or support failure. Structure availability does not qualify unseen
sources, independent generalization, a learned detector or the north-star win.

## Validation

The frozen scope passed owned tests, direct-consumer tests, private E2E-018/021,
Ruff check/format, strict mypy, explicit-path spec traceability, and100 percent
of222 newly owned statements with zero exclusions. Actual private children,
foreign-directory cold replay without PYTHONPATH, missing-input, altered-authority
and rehashed-tamper controls passed. Publication used `primary_publication` only
after these receipts passed; unchanged adversarial and strict row scanners passed.
Historical repository-wide health remains failed separately. The full repository
suite was not run, as required by this task's scoped-validation instruction.

All current model invocation counters are zero. No external publication or
activation occurred. Status/changelog/traceability reconciliation is delegated
to the conductor under the task's final instruction.
