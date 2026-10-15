# V735 HALLMARK release admissibility

Exp8553 audits public release evidence. It rejects these records as independent
Carnot sentence labels. Audit readiness is separate from label readiness.
No model, verifier, training, source refit or nine-input head ran.

The [October 11 paper](https://arxiv.org/html/2607.18360v2) describes pipeline
labels, database checks and author review. Valid inclusion requires database
confirmation. The Walters–Wilder supplement documents human coding, but its
bibliographic target does not establish Carnot sentence truth or disjointness.
The paper explicitly identifies tool/taxonomy co-design and excludes its own
reference tool from independent ranking. Source availability remains a separate
measurement condition. Every deployment flag requires human review.

The [pinned repository](https://github.com/rpatrik96/hallmark/tree/f774fa40675daa83eca6201637a94c4536b7bb3e)
contains v1.2.3 metadata. Public core bytes contain 1,119 dev, 831 test and 122
stress rows. The last includes one VALID contamination canary. Adding the
metadata-only 454 hidden rows gives 2,526; omitting the canary gives the stale
2,525 description. Hidden content was never fetched. Additional acquired
splits contain 500 cross-domain, 452 matched, 341 ChatGPT and 448 temporal rows.
The audit retains 3,813 split records, without treating repeats as independent.
MIT applies to the released dataset; the paper carries CC BY 4.0.

DOI and normalized-title hashes identify five dev/test base-work groups.
Restoring explicitly recorded original titles exposes two dev hallucinations
whose bases are test VALID records. The artifact retains every matching ID and
alias. These observable links prevent an independent-disjointness claim.
Fully corrupted bases can remain unresolved. They do not count as disjoint.

Two complete prediction snapshots are reduced without threshold changes.
GPT-5.1 supplement counts are TP143, FP49, TN123, FN26 across 341 records.
For the 831-record GPT-5.4 cascade, conservative counts are TP501, FP13, TN299,
FN18. Selective scoring excludes 13 abstentions: TP501, FP13, TN298, FN6
across 818. Aggressive scoring gives TP513, FP14, TN298, FN6 across 831.
These are release diagnostics. They are not Carnot detector accuracy.

The raw directory holds original bytes, Git blob checks, licenses, source hashes,
a checklist sealed before row inspection, all first-20 provenance samples,
a hand-checked rejection control, test nodes, logs and terminal receipts.
The runner uses unchanged publication and adversarial consumers. Cold replay
recomputes primitive rows and rejects rehashed substitutions. Existing V734
quarantines remain binding. Future use is limited to citation diagnostics after
base grouping, independent review and source-availability checks. Same-roadmap
outputs remain future dependencies.

Validation uses owned tests, statement coverage, strict typing, Ruff, spec tracing
and private E2E-018/021 consumers. Status, changelog and traceability reconciliation
are left to the conductor, as requested.
