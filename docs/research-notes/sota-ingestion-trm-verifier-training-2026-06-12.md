# SOTA ingestion 2026-06-12: TRM baseline plus verifier-guided training

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_trm_verifier_training_mapped`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `TRM Sudoku baseline reproduction`, arxiv_id: `2510.04871`, url: `https://arxiv.org/abs/2510.04871`}
  - {name: `TTA-TRM full fine-tuning control`, arxiv_id: `2511.02886`, url: `https://arxiv.org/abs/2511.02886`}
  - {name: `V-STaR accepted/rejected trace selector`, arxiv_id: `2402.06457`, url: `https://arxiv.org/abs/2402.06457`}
  - {name: `STaR / ReST generate-filter-improve loop`, arxiv_id: `2203.14465`, url: `https://arxiv.org/abs/2203.14465`}
  - {name: `Verifier-guided adaptive Sudoku search`, arxiv_id: `2602.01070`, url: `https://arxiv.org/abs/2602.01070`}
  - principle: Each method MUST carry a real arXiv ID/URL; an ingestion note without verifiable citations is treated as fabrication.
- flagged_for_v381: `verifier_guided_adaptive_sudoku_search_before_training`
  - principle: Closes the discover->ingest->plan loop: names the strongest method for the next planner.

**Fresh-pass provenance**

Read the TRM and verifier-guided-training track in `research-studying.md` and
`research-references.md`, including the Exp 4102 `.379` ingestion that flagged
V-STaR for `.380`, the `.351` recursive-refiner notes, and the `.380` Exp 4108
and Exp 4109 result artifacts. Ran the required helpers:

- `.venv/bin/python scripts/sweep_clusters.py 0 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 1 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "TRM Tiny Recursive Models verifier Sudoku baseline test-time adaptation" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "verifier guided training V-STaR STaR ReST recursive reasoning verifier" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "Tiny Recursive Model TRM verifier guided self training process reward" --limit 8`

Semantic Scholar returned zero IDs for the V-STaR/STaR/ReST query and HTTP 429
for two TRM-focused queries. The arXiv cluster helpers emitted reliable
verifier/energy query URLs. Low-concurrency WebSearch/WebFetch then verified
the primary arXiv pages for `arXiv:2510.04871`, `arXiv:2511.02886`,
`arXiv:2402.06457`, `arXiv:2203.14465`, `arXiv:2308.08998`,
`arXiv:2602.01070`, `arXiv:2601.17223`, and `arXiv:2605.10325`. The
`/deep-research` loop was not invoked.

## Current .380 baseline-plus-verifier anchor

The `.380` headline should stay honest. Exp 4108 confirmed the native
nano-trm Sudoku Extreme trainer can produce and reload a checkpoint, but the
measured validation exact accuracy was 0.0232 and `matches_published_087=false`,
so it is a partial baseline rather than a reproduced published number. Exp 4109
then grafted the executable Sudoku verifier over that checkpoint's candidate
pools and found an honest null: verifier selection tied TRM vote with
`rerank_lift_vs_vote.delta=0.0`, and the bounded A-vs-cold comparison also
reported `delta=0.0`.

That changes the next SOTA question. Post-hoc verifier reranking is not enough
on the current checkpoint. The strongest follow-up should move the verifier
earlier in the loop, where it can shape candidate expansion and data admission
before any expensive full fine-tuning.

## TRM Sudoku baseline reproduction

**Method:** TRM, `arXiv:2510.04871`
(https://arxiv.org/abs/2510.04871), is the load-bearing substrate because it
reports a tiny recursive model that beats HRM-style baselines on Sudoku, maze,
and ARC-style puzzles with a 7M-parameter recursive network.

**Implementation over nano-trm + Carnot-verifier stack:** Treat reproduction
as a gate, not as a background detail. Re-run the native nano-trm Sudoku
Extreme baseline with a clean progress callback, stable dataset checksum, and
checkpoint reload proof. Only after the baseline approaches the published
target should the Carnot verifier graft be allowed to claim lift over vote.

**Pitfalls / where it fails:** Exp 4108 already showed the failure mode: a
checkpoint can exist and the trainer mechanism can be real while the accuracy
is far below the published target. A verifier experiment on that checkpoint can
still be useful as a mechanism probe, but it cannot support a reproduction
claim or a strong negative about verifier value on a faithful TRM.

## TTA-TRM full fine-tuning control

**Method:** Test-time Adaptation of Tiny Recursive Models,
`arXiv:2511.02886` (https://arxiv.org/abs/2511.02886), is the adaptation
control. It argues that bounded full fine-tuning can matter more than LoRA or
task-embedding updates for a tiny recursive model.

**Implementation over nano-trm + Carnot-verifier stack:** Keep three arms
separate: full fine-tuning without verifier labels, full fine-tuning admitted
by the executable Sudoku verifier, and post-hoc verifier reranking without
training. The comparison must report compute, optimizer steps, checkpoint
source, and data split so adaptation gain is not mislabeled as verifier gain.

**Pitfalls / where it fails:** TTA-TRM can win by spending adaptation compute or
memorizing public task structure. If the experiment does not isolate the
no-verifier full-fine-tune arm, every gain will be ambiguous.

## V-STaR accepted/rejected trace selector

**Method:** V-STaR, `arXiv:2402.06457`
(https://arxiv.org/abs/2402.06457), trains a verifier from both correct and
incorrect self-generated solutions rather than throwing away failures, then
uses that verifier to select among candidates.

**Implementation over nano-trm + Carnot-verifier stack:** Reuse the Exp 4109
candidate pools, but keep every sampled Sudoku completion: exact-valid,
near-valid, row/column/box-invalid, duplicate-vote, and timeout. Convert
within-puzzle pairs into selector data where the executable Sudoku verifier and
final exact-valid label agree. Use the selector first as a cheap reranker
against vote before letting it gate a second RFT corpus.

**Pitfalls / where it fails:** If the current TRM emits many near-duplicate
invalid completions, the selector learns shallow token regularities instead of
semantic validity. V-STaR is also downstream of verifier coverage; it cannot
invent correct completions absent from the sampled pool.

## STaR / ReST generate-filter-improve loop

**Method:** STaR, `arXiv:2203.14465`
(https://arxiv.org/abs/2203.14465), gives the minimal generate-filter-finetune
loop for self-generated reasoning traces. ReST, `arXiv:2308.08998`
(https://arxiv.org/abs/2308.08998), adds a reusable offline
generate/filter/improve cadence.

**Implementation over nano-trm + Carnot-verifier stack:** Treat Sudoku
candidate completions as rationale traces. Generate K candidates per puzzle,
filter exact-valid completions with the Carnot Sudoku verifier, fine-tune on
unique positives, regenerate from the updated checkpoint, and keep rejected
rows available for V-STaR-style selector training rather than discarding them.

**Pitfalls / where it fails:** STaR/ReST need support. If the TRM rarely samples
valid completions from the partial Exp 4108 checkpoint, filtering leaves too few
positives and the improve step becomes either unstable or a memorization pass.

## Verifier-guided adaptive Sudoku search

**Method:** Adaptive test-time compute allocation, `arXiv:2602.01070`
(https://arxiv.org/abs/2602.01070), is the search-side candidate: spend more
compute where verification says it can change the answer, not after a fixed
candidate pool has already been sampled. Verifiable process rewards,
`arXiv:2601.17223` (https://arxiv.org/abs/2601.17223) and `arXiv:2605.10325`
(https://arxiv.org/abs/2605.10325), give the adjacent dense-feedback pattern
when intermediate states are objectively checkable.

**Implementation over nano-trm + Carnot-verifier stack:** Move Sudoku row,
column, and box checks into candidate expansion. Instead of sampling K complete
boards and reranking, allocate extra recursive steps, resampling, or branch
budget to partial boards whose verifier state is recoverable and prune branches
that violate exact constraints irreparably. Keep final exact validity as the
only acceptance authority, and measure against the fixed-K vote and post-hoc
verifier rerank from Exp 4109.

**Pitfalls / where it fails:** Local validity is not final correctness. A board
can satisfy many local constraints and still be unrecoverable from the puzzle
givens. The verifier-guided arm must therefore report final exact accuracy,
oracle support, and prune-error rate, not just average verifier score.

## Flagged for the .381 roadmap

`verifier_guided_adaptive_sudoku_search_before_training` is the strongest
single `.381` candidate. Exp 4109 already tested post-hoc verifier reranking
and found no lift. Before spending on another full fine-tune or a V-STaR
selector, the next planner should test whether putting the executable Sudoku
verifier inside candidate expansion creates support that post-hoc reranking did
not have. If it does not beat fixed-K vote on pass@1 or oracle support, the
training routes should remain blocked.

