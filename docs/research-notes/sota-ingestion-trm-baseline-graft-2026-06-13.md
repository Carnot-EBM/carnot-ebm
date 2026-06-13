# SOTA ingestion 2026-06-13: TRM baseline graft with resumable verifier discipline

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_trm_baseline_graft_mapped`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `TRM resumable Sudoku baseline gate`, arxiv_id: `2510.04871`, url: `https://arxiv.org/abs/2510.04871`}
  - {name: `TTA-TRM full-fine-tune control`, arxiv_id: `2511.02886`, url: `https://arxiv.org/abs/2511.02886`}
  - {name: `Verifier-guided adaptive candidate expansion`, arxiv_id: `2602.01070`, url: `https://arxiv.org/abs/2602.01070`}
  - {name: `V-STaR accepted/rejected Sudoku selector`, arxiv_id: `2402.06457`, url: `https://arxiv.org/abs/2402.06457`}
  - {name: `ReST resumable generate-filter-improve curriculum`, arxiv_id: `2308.08998`, url: `https://arxiv.org/abs/2308.08998`}
  - principle: Each method MUST carry a real arXiv ID/URL; an ingestion note without verifiable citations is treated as fabrication.
- flagged_for_v382: `verifier_guided_adaptive_candidate_expansion_over_resumed_trm`
  - principle: Closes the discover->ingest->plan loop: names the strongest method for the next planner.

**Fresh-pass provenance**

Read the local TRM, verifier-guided-training, and long-horizon-training track in
`research-studying.md` and `research-references.md`, including the Exp 4102
`.379` V-STaR flag, the Exp 4111 `.380` in-loop verifier-guided search flag,
the `.351` recursive-refiner notes, and the long-horizon VPR notes. Ran the
required helpers:

- `.venv/bin/python scripts/sweep_clusters.py 0 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 1 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 3 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "TRM Tiny Recursive Models verifier Sudoku baseline test-time adaptation" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "verifier guided training V-STaR STaR ReST recursive reasoning verifier" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "long horizon training verifier process reward recursive reasoning TRM" --limit 8`

Semantic Scholar returned `arXiv:2603.15641` for the TRM baseline/adaptation
query and HTTP 429 for the two verifier-training queries; it did not displace
the operator-specified anchors. The arXiv cluster helpers emitted reliable
verifier, energy, and active-inference query URLs. Low-concurrency
WebSearch/WebFetch verified the primary arXiv pages for `arXiv:2510.04871`,
`arXiv:2511.02886`, `arXiv:2402.06457`, `arXiv:2203.14465`,
`arXiv:2308.08998`, `arXiv:2602.01070`, `arXiv:2601.17223`, and
`arXiv:2605.10325`. The `/deep-research` loop was not invoked.

## Current .381 resumable baseline-graft anchor

The `.381` headline should stay narrower than a general verifier-training
claim. Exp 4108 proved the native nano-trm Sudoku Extreme path can train,
checkpoint, and reload, but its measured validation exact accuracy was 0.0232
with `matches_published_087=false`. Exp 4109 then grafted the executable Sudoku
verifier onto candidate pools from that checkpoint and found no post-hoc lift
over vote. Exp 4111 therefore flagged moving verification into candidate
expansion before spending on another training loop.

For `.382`, the planner needs a method that preserves all three facts: the TRM
baseline must be resumable and reproduced before headline claims, the verifier
must act before a fixed weak pool is exhausted, and any training curriculum must
record rejected traces instead of throwing them away.

## TRM resumable Sudoku baseline gate

**Method:** TRM, `arXiv:2510.04871`
(https://arxiv.org/abs/2510.04871), is the substrate to reproduce. The paper
reports a tiny recursive model with a single small network and strong puzzle
generalization, including Sudoku Extreme and ARC-style tasks.

**Implementation over nano-trm + Carnot-verifier stack:** Treat the resumed
baseline as a gate. Continue from the saved nano-trm checkpoint only with a
stable dataset checksum, optimizer-state receipt, checkpoint reload proof, and
held-out Sudoku Extreme validation trace. The Carnot verifier graft should only
claim value after the TRM baseline approaches the published Sudoku target or is
explicitly labeled as a partial-baseline mechanism probe.

**Pitfalls / where it fails:** A resumable checkpoint is not a reproduced TRM.
The existing partial baseline can validate code paths while still producing a
candidate pool too weak for post-hoc verifier selection to matter.

## TTA-TRM full-fine-tune control

**Method:** Test-time Adaptation of Tiny Recursive Models,
`arXiv:2511.02886` (https://arxiv.org/abs/2511.02886), is the adaptation
control because it reports that full fine-tuning, not LoRA or task embeddings
alone, drives the tiny recursive model's competition-budget adaptation.

**Implementation over nano-trm + Carnot-verifier stack:** Keep three arms:
resumed baseline without extra training, no-verifier full fine-tuning, and
verifier-admitted full fine-tuning. Log optimizer steps, task splits, wall time,
checkpoint source, and verifier admission counts so adaptation gain is not
misreported as verifier gain.

**Pitfalls / where it fails:** Full fine-tuning can memorize public-task
structure or simply spend more compute. Without the no-verifier control, every
improvement is ambiguous.

## Verifier-guided adaptive candidate expansion

**Method:** Adaptive test-time compute allocation,
`arXiv:2602.01070` (https://arxiv.org/abs/2602.01070), is the strongest
follow-on because it uses verification during generation and expansion rather
than only for final reranking. Verifiable process reward work, `arXiv:2601.17223`
(https://arxiv.org/abs/2601.17223) and `arXiv:2605.10325`
(https://arxiv.org/abs/2605.10325), supplies the long-horizon dense-feedback
pattern when intermediate steps are objectively checkable.

**Implementation over nano-trm + Carnot-verifier stack:** Move row, column,
box, and given-cell checks into the recursive candidate expansion loop. Spend
extra samples or recursive steps on partial boards that remain recoverable,
prune irreparable branches early, and compare against fixed-K vote plus Exp
4109 post-hoc verifier reranking. Report pass@1, oracle support, final exact
validity, verifier-call count, and prune-error rate.

**Pitfalls / where it fails:** Sudoku has local constraints that can look good
while the board is globally unrecoverable. Final exact validity must remain the
acceptance authority, and the experiment must measure whether pruning removes
any candidate that could have become valid.

## V-STaR accepted/rejected Sudoku selector

**Method:** V-STaR, `arXiv:2402.06457`
(https://arxiv.org/abs/2402.06457), trains a verifier from both correct and
incorrect self-generated solutions and uses it to choose among candidates.

**Implementation over nano-trm + Carnot-verifier stack:** Keep all completions
from the resumed TRM: exact-valid, row/column/box invalid, duplicate vote,
timeout, and parse fail. Build within-puzzle preference pairs only where the
executable Sudoku verifier and final exact label agree. Use the selector first
as a reranker before allowing it to gate a second RFT corpus.

**Pitfalls / where it fails:** V-STaR needs diverse failures. If the resumed
checkpoint emits many near-duplicate wrong boards, the selector learns surface
regularities and still cannot create correct candidates absent from the pool.

## ReST resumable generate-filter-improve curriculum

**Method:** ReST, `arXiv:2308.08998`
(https://arxiv.org/abs/2308.08998), gives the reusable offline
generate-filter-improve cadence. STaR, `arXiv:2203.14465`
(https://arxiv.org/abs/2203.14465), supplies the older rationale
self-training loop that keeps only generated reasoning that reaches correct
answers.

**Implementation over nano-trm + Carnot-verifier stack:** Cache Sudoku
candidate batches, filter exact-valid completions with Carnot, train on unique
positives, and then resume generation from the updated checkpoint. Retain
rejected rows for the V-STaR selector rather than discarding them.

**Pitfalls / where it fails:** The loop only amplifies support already present
in the generator. If resumed TRM rarely samples valid boards, the curriculum
creates too few positives and can collapse into memorization.

## Flagged for the .382 roadmap

`verifier_guided_adaptive_candidate_expansion_over_resumed_trm` is the strongest
single `.382` candidate. It directly addresses the Exp 4109 null by moving the
Sudoku verifier before fixed-pool reranking, while preserving the Exp 4108
baseline-reproduction gate. The next planner should require pass@1 or oracle
support lift over fixed-K vote and post-hoc verifier rerank. If it fails,
selector/RFT work should stay blocked.

