# SOTA ingestion 2026-06-19: .411 library-learning map for .412

Reliable channel only: `scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
arXiv abs-page HTTP-200 checks, and low-concurrency WebSearch/WebFetch of the
top abstracts. `scripts/sweep_clusters.py --help` succeeded and arXiv was
reachable. Semantic Scholar returned HTTP 429 on four of five focused queries;
the one successful counterexample-guided query surfaced CodeARC plus adjacent
non-promoted IDs. `/deep-research` was not invoked. No leaderboard submission
was made. No live solve or training run was launched.

## Focused sweep result

- LILO documented library induction, arXiv:2310.19791, is still the strongest
  fit for `.412`: synthesize, compress, and document reusable abstractions over
  the ARC solver corpus.
- DreamCoder wake-sleep abstraction discovery, arXiv:2006.08381, supplies the
  older generalizable library-learning backbone.
- Stitch top-down synthesis, arXiv:2211.16605, supplies the scalable compressor
  that LILO builds on.
- HYSYNTH context-free LLM approximation, arXiv:2405.15880, maps to task-local
  symbolic search from LLM completions.
- CodeARC differential-query induction, arXiv:2503.23145, maps to
  counterexample-led refinement of open verifier and predicate gaps.
- Executable World Models for ARC-AGI-3, arXiv:2605.05138, remains the harness
  for generic interactive-game transfer without game-specific prompts.
- Loop-OWM object-centric world-model transfer, arXiv:2606.12316, is the
  freshest direct ARC transfer paper and supplies object-state transition tests.
- ARC-TGI task-family generators, arXiv:2603.05099, supplies held-out
  variation so libraries are tested on rule families rather than one trace.

## SOTA->experiment mapping

The `.412` planner should run LILO-style documented primitive induction over the
solved ARC corpus: compress existing predicates/world-model snippets, generate
human-readable names and docstrings, retrieve those primitives before first
contact, and score only held-out reproduction-gated improvements. Loop-OWM and
Executable World Models provide the transfer evaluation substrate; CodeARC and
HYSYNTH provide counterexample-guided repair when a documented primitive fails.

flagged_for_v412: LILO-style documented library induction over the ARC solver corpus (arXiv:2310.19791)
