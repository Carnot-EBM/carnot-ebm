# V751 evidence-set methods — Exp8768, 2026-11-03

This prototype qualifies a constructed relational test. It does not implement a
natural-language extractor, load an LLM, fit a numerical head, or demonstrate
independent semantic generalization. The terminal evidence is
[Exp8768](../../results/experiment_8768_v751_evidence_set_methods.json).
Qualification and comparative promotion are separate gates.

[SURE-RAG v2](https://arxiv.org/html/2605.03534v2) motivates treating sufficiency
as a property of the evidence set. [Constrained paraphrase consistency](https://arxiv.org/html/2606.08158v1)
motivates checking independent label preservation alongside prediction agreement.
These are narrow method adaptations to supplied signed records, not paper
reproductions or natural factuality measurements.

The scorer receives only `facts` and `query`. Facts are constructed signed
subject/relation/object records, with exact tab-separated document spans retained
separately. No answer label, condition, world ID, split, oracle result, or retrieval
metadata enters the scoring function. All predicates must share one binding for
repeated variables. Absence alone remains unknown. An explicit negative literal
can refute; any contradictory signed records yield insufficient evidence. The
feature trace lists required joins, missing premises, explicit refutations and
conflicts. This opt-in rule changes no shared verification default.

The local comparator calls the unchanged V750 per-premise evaluator. The strong
joint comparator calls the unchanged V749 joint evaluator. Their treatment of
contradictory records remains unchanged. Each local arm uses four development or
five evaluation primitive calls; each joint and new set-level arm uses one.
Rows retain actual monotonic wall time and primitive-call counts. Equal call caps
do not establish equal computational cost. Replay/audit work is validation, not
an additional scientific observation.

The corpus contains 60 development and 120 evaluation worlds. Each world has a
distinct directed graph up to constant-entity renaming, fixing the join variable.
Development queries have four edges; evaluation queries have five. Entity rosters,
relation vocabularies, rule templates and graph classes are disjoint across splits.
Six conditions are complete support, explicit refutation, missing required hop,
disconnected join, irrelevant distractor and explicit conflict. Each has an
original view and a bijective renaming with reversed record/query order. The
budget is exactly 2160 units and 6480 arm decisions; 4320 decisions belong to
120 evaluation clusters. Views and arms do not increase independent N.

The independent oracle enumerates finite Boolean completions of missing atoms.
It branches over unassigned atoms and prunes only a subcube whose query outcome
is already fixed. Both satisfiable outcomes mean unknown. No compatible models
under contradictory records means insufficient. Truth-bearing records and labels
are sealed in their own file. Separate input-only CLI children write decisions
before aggregation opens labels. No operating-system label sandbox is claimed;
strict parser controls and an oracle trap test check the implemented access path.

Implementation and validation inspect constructed generators and correctness
assertions. Initial correctness tests also instantiated draft evaluation cases.
This is not a blind evaluation, and the exposed draft is not a hidden test set.
The final graph roster and methods freeze before its comparative invocation.
Structural separation does not establish natural-data or hidden-game transfer.

The primary estimand is each evaluation world's class-balanced three-way error,
then the mean across 120 worlds. Paired world-cluster bootstrap uses 10000 draws.
The simultaneous family confidence is 97.5 percent across two comparators, using
Bonferroni individual two-sided 98.75 percent intervals. Percentile order indices
are 62 and 9937 (zero based). Promotion requires both lower improvement bounds
above zero, no added false acceptance, no extra rejection of valid alternatives,
and exact correct-control invariance. Constant accept/reject controls must fail.
A null is valid when the strong comparator is optimal. Any observed improvement
on these constructed semantics is `circular_positive`; independent generalization
remains zero. Degenerate intervals on fixed designed conditions do not estimate
natural population uncertainty.

Exp8755's closed schedule is not rerun or included in this N. Its zero gain and
zero surface-change result remains historical evidence. Exp8737's failed owned
validation remains failed history; it is not promoted by this prototype.

The first unit run failed at the missing module as intended. A structural-count
regression then rejected the draft's repeated graph shapes. A private prototype
also exposed tuple/list trace differences across JSON transport; traces now use
JSON-native lists. One private replay occurred while test source bytes were
changing and correctly rejected source drift. These attempts grant no readiness.

The requested full Python-suite command was launched once with its configured
coverage and four workers. It exceeded a real 120-second bound and was killed
and reaped. That receipt and both log hashes remain separate repository health;
there is no full-suite pass claim. Final owned tests, consumer checks, private
E2E-018/021, Ruff, strict mypy, explicit spec tracing, actual new-statement coverage,
unchanged adversarial/strict-row scans and fresh-process replay determine release.
Their exact argv, exits, JUnit, nodes and hashes are in the terminal artifact's
raw invocation references. The conductor owns ops and BMAD reconciliation.
