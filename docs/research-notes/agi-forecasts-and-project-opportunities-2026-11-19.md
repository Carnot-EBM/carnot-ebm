# AGI forecasts and where Carnot can add most (2026-11-19)

Written after an operator question: what does AGI mean, where does this project fit,
and where is the largest opportunity? This note records the answer so it is not lost.
All claims below come from web search results. None was checked against a primary source
unless the line says so. Treat every number as secondary.

## Working definition

AGI here means a system that learns a new task with little guidance and does it about
as well as a competent person. It must hold across unrelated domains and over long
stretches without a human fixing its mistakes. The test is novelty, not performance on
seen tasks. AGI is a bundle of capabilities, not one threshold.

## Forecasts (search results, 2026-11-19)

| Source | Finding |
|---|---|
| Metaculus (strong AGI) | Median moved from July 2031 to November 2033 during 2025 |
| Metaculus (first announced general AI) | Median about November 2027. A different question, so do not compare dates |
| AI Impacts survey, 2023, n=2,778 | 50% chance machines beat humans at every task by 2047 |
| Forecasting Research Institute, 2026 | Experts median 2050. Superforecasters median 2047 |
| Superforecasters, 2022 | 25% chance by 2048 |
| METR time horizon, TH1.1, 2026-01-29 | Doubling time about 131 days since 2023. Earlier fit: about 7 months |
| ARC-AGI-3 at launch, March 2026 | All frontier models under 1%. Humans 100% |
| ARC-AGI-3 later (one aggregator) | Top model 62.7%. NOT confirmed against arcprize.org |

Short-run capability trends moved faster than the surveys. The surveys still point
to the 2030s through the 2050s. Metaculus users are a selected group, so that figure
may lean early.

## Where Carnot can add most (ranked)

1. **A calibrated, oracle-distinct verifier.** This is the north star since 2026-11-07.
   The field moves from deterministic verifiers (math, code) to model judges. Papers
   report judge exploitation, open calibration problems, and a benefit from
   high-precision verifiers. A learned energy verifier with a Brier score and a CI95
   against a matched control, where no cheap oracle exists, would count. We have not
   landed it. Our record so far is honest nulls, circular wins, and a calibrated-decision
   floor still being built. Self-consistency was near-optimal on our corpora.
2. **Trustworthy evaluation method.** Our fabrication gate, circularity declaration and
   positive-control rule address benchmark overstatement. A write-up or tool would help
   the field without a capability result. This is our most distinctive asset.
3. **Sample-efficient hidden-game discovery on small local models.** Frontier scores are
   rising, so scale will not win. Our 25 hand-built adapters do not transfer. The 27B
   local-model limit and efficiency scoring could set us apart. ARC is now opportunistic.
4. **Self-improvement against a hard, deterministic signal.** The Victris Arena challenge
   fits. See `victris-arena-rsi-challenge-2026-11-19.md`.

## Limits

A small team and one GPU (GPU 1 is detached). We cannot compete on training scale.
Our opportunities are in verification and measurement, not raw capability.

## Not checked

- No primary survey or paper was read in full.
- The post-October ARC-AGI-3 scores and model names are unverified.
- The METR figures of 14.5 hours and 49 days came from secondary pages only.
- I did not search whether others already fill opportunity 1, so its novelty is untested.
- The paper cited as arXiv 2609.33467 came from search results. Confirm it before citing.

## Sources

- https://80000hours.org/2025/03/when-do-experts-expect-agi-to-arrive/
- https://forecastingresearch.substack.com/p/leap-wave-8-ai-timelines
- https://metr.org/blog/2026-1-29-time-horizon-1-1
- https://www.lesswrong.com/posts/EYb2K9acKfyG2bome/metr-time-horizons-now-10x-year
- https://officechai.com/ai/arc-agi-3
- https://benchlm.ai/benchmarks/arcagi3
- https://aclanthology.org/2026.acl-long.178/
- https://arxiv.org/pdf/2609.33467
