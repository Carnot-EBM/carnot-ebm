# SOTA ingestion 2026-06-15: .394 forks map for .395

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_v395_mapped`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `ARC-TGI task-family generators for cross-game generalization`, arxiv_id_or_url: `2603.05099`, url: `https://arxiv.org/abs/2603.05099`}
  - {name: `Reliability Gap benchmark-auditing provenance discipline`, arxiv_id_or_url: `2606.03305`, url: `https://arxiv.org/abs/2606.03305`}
  - {name: `DPRM token-ordering guidance for diffusion language models`, arxiv_id_or_url: `2604.24357`, url: `https://arxiv.org/abs/2604.24357`}
  - {name: `Entropy-guided step selection and stepwise advantages for diffusion LLM RL`, arxiv_id_or_url: `2603.12554`, url: `https://arxiv.org/abs/2603.12554`}
  - {name: `L-VARC language-guided abstraction with inference-time visual backbone`, arxiv_id_or_url: `2606.12847`, url: `https://arxiv.org/abs/2606.12847`}
  - principle: Each method MUST carry a real arXiv ID/URL (no citation = fabrication per adversarial_verify discipline) + a one-line .395 experiment mapping.
- flagged_for_v395: `arc_tgi_family_generator_cross_game_generalization_v395`
  - principle: Closes discover->ingest->plan: names the strongest method for the .395 planner, conditioned on the .394 outcomes.
- random_seed: `4265`
  - principle: Determinism placeholder for the discovery query set (recorded for reproducibility of the sweep).

## Fresh-pass provenance

Read `CLAUDE.md` SOTA-Ingestion Cycle Discipline, `research-studying.md`,
`research-references.md`, `results/experiment_4251_sota_ingestion_set_encoder_offline_rft.json`,
the .394 fork artifacts Exp 4256 through Exp 4264, `scripts/sweep_clusters.py`,
and `scripts/sweep_semscholar.py`.

Reliable-channel helper pass, not `/deep-research`:
- `python3 scripts/sweep_clusters.py 0 --max-results 8`
- `python3 scripts/sweep_clusters.py 1 --max-results 8`
- `python3 scripts/sweep_semscholar.py "Compute as Teacher generative synthesis selection robustness best of n aggregation" --limit 8`
- `python3 scripts/sweep_semscholar.py "discrete diffusion classifier guidance verifier energy guided language model EDLM" --limit 8`
- `python3 scripts/sweep_semscholar.py "benchmark contamination detection data leakage auditing learned verifier membership inference" --limit 8`
- `python3 scripts/sweep_semscholar.py "ARC abstraction reasoning cross task transfer cross game generalization" --limit 8`

The sweep helpers imported successfully. The cluster helper emitted the
broadened arXiv API URLs for the reliable channel. Semantic Scholar returned
HTTP 429 for the four focused queries, so no Semantic-Scholar-only promotion is
claimed. WebSearch/WebFetch was reachable and verified arXiv:2603.05099,
arXiv:2606.03305, arXiv:2604.24357, arXiv:2603.12554, and arXiv:2606.12847.
The banned `/deep-research` channel was not invoked.

## Prior-covered methods not re-ingested

The .392/.393/.394 sweeps already covered Compute-as-Teacher (arXiv:2509.14234),
GSA / LLMs Can Generate a Better Answer by Aggregating Their Own Responses
(arXiv:2503.04104), GenSelect-BoN (arXiv:2602.02143), Reward-Guided Stitching
(arXiv:2602.22871), S3 (arXiv:2604.06260), EDLM (arXiv:2410.21357),
Unlocking Guidance for Discrete State-Space Diffusion and Flow Models
(arXiv:2406.01572), CoDeC (arXiv:2510.27055), ARC of Progress
(arXiv:2603.13372), ARCTraj (arXiv:2511.11079), and Compositional
Neuro-Symbolic Reasoning (arXiv:2604.02434). They remain context, but they are
not counted as fresh `methods_mapped` rows here.

## .394 fork outcome read

Exp 4256: `arc_provenance_blind_win_survives`,
`provenance_blind_delta=0.3846153846`, `win_survives_provenance_blind=true`.
The leak audit hardened the selector win, but the high origin probe means .395
must keep transparent provenance rather than trusting detector-only audits.

Exp 4257: `arc_oracle_distinct_win_replicates_multiseed`,
`mean_delta=0.4576923077`, `cross_seed_ci95=[0.4377176136, 0.4776670017]`,
and `oracle_distinct_win_replicates=true`. The within-pool selector win is now
robust enough to test transfer.

Exp 4258: `blocked_arc_game_ids_unrecoverable`, so cross-game transfer was not
measured. This is a data/partition blocker, not evidence of generalization or
collapse.

Exp 4259: `arc_synthesis_underperforms_selection`,
`synthesis_breaks_oracle_ceiling=false`, `synthesis_minus_oracle_delta=-0.2826086957`,
`synthesis_beats_selection=false`, and `exact_match_validated=true`. The
selection win should not be escalated into a generative synthesis headline.

Exp 4260: `blocked_diffusiongemma_gguf_loader_failed` and `preflight_go=false`.
DiffusionGemma remains a loader-repair path, not a .395 full-run bet.

Exp 4264: `code_oracle_distinct_replication_corpus_specific`,
`code_replication_beats_vote=false`, and `code_predictor_minus_vote_delta=-0.00625`.
The code read does not replicate the ARC moat; it supports keeping .395 focused
on ARC transfer and provenance.

## SOTA -> experiment mapping

## ARC-TGI: recover the blocked cross-game axis

**Method/source:** ARC-TGI, arXiv:2603.05099
(https://arxiv.org/abs/2603.05099), provides human-validated ARC task-family
generators with reasoning-chain templates and task-level constraints.

**Carnot stack mapping:** Persist generator or task-family IDs next to each ARC
candidate row, use them to create family-disjoint train/test splits, and score
the existing Set-Encoder against vote on held-out families.

**.394 conditioning:** Exp 4256 and Exp 4257 hardened the within-pool win, but
Exp 4258 blocked the real OOD test because game IDs were unrecoverable. ARC-TGI
directly fixes that missing split variable.

**Failure mode:** Generator data can overfit to generator artifacts. Keep
original ARC held-out tasks as a sanity read and separate generated-family
metrics from original-task metrics.

**Experiment mapping:** .395 builds an ARC-TGI-style family-disjoint candidate
pool and reruns Set-Encoder versus vote on held-out families.

## Reliability Gap: provenance before detector-only leak audits

**Method/source:** Reliability Gap in Benchmark Auditing, arXiv:2606.03305
(https://arxiv.org/abs/2606.03305), shows contamination detectors can fail
under distribution shift and small benchmark scale.

**Carnot stack mapping:** Make source-kind, generator-family, fold, and target
hashes first-class manifest columns before training or evaluation.

**.394 conditioning:** Exp 4256 survived provenance-blind scoring, but the
origin probe was high. The next milestone should rely on transparent row
provenance and family splits, not post-hoc detector confidence.

**Failure mode:** Statistical leak detectors can be underpowered. Treat them as
diagnostics, not acceptance gates.

**Experiment mapping:** .395 rejects any selector, synthesis, or transfer claim
whose rows cannot be traced to a source-kind and family manifest.

## DPRM: queue diffusion guidance behind loader repair

**Method/source:** DPRM, arXiv:2604.24357
(https://arxiv.org/abs/2604.24357), uses a Doob h-transform process-reward
module to guide token ordering in diffusion language models without changing the
host denoiser.

**Carnot stack mapping:** Once DiffusionGemma loads, map verifier rewards into
token or cell reveal ordering for a tiny guided denoising smoke.

**.394 conditioning:** Exp 4260 blocked before guidance could run, so DPRM is
not a full-run recommendation yet.

**Failure mode:** Ordering guidance may optimize confidence rather than exact
grid correctness. Exact ARC match and selector-only controls remain required.

**Experiment mapping:** .395 only after loader repair: run a tiny DPRM-style
guided reveal smoke, then decide whether a full DiffusionGemma run is warranted.

## Entropy-guided diffusion RL: stepwise rewards, not final-only guesses

**Method/source:** Reinforcement Learning for Diffusion LLMs with
Entropy-Guided Step Selection and Stepwise Advantages, arXiv:2603.12554
(https://arxiv.org/abs/2603.12554), derives stepwise policy-gradient updates
over denoising trajectories.

**Carnot stack mapping:** If DiffusionGemma loads, use verifier rewards as
intermediate denoising advantages rather than final-output-only reward.

**.394 conditioning:** Loader failure and synthesis underperformance put this
behind the cross-game/provenance work.

**Failure mode:** Sparse ARC exact-match reward may still be too coarse without
per-cell evidence.

**Experiment mapping:** .395 deferred path: bounded entropy-step smoke after
DiffusionGemma loader repair, compared to unguided diffusion.

## L-VARC: training-only semantic abstraction

**Method/source:** L-VARC, arXiv:2606.12847
(https://arxiv.org/abs/2606.12847), trains a lightweight ARC visual model with
a language-guided privileged-information branch that is discarded at inference.

**Carnot stack mapping:** Attach reasoning-template or family embeddings during
training, remove them at inference, and evaluate exact matches on held-out
families.

**.394 conditioning:** Exp 4259 says naive synthesis did not beat selection, so
semantic abstraction should be a transfer scaffold rather than a generation
headline.

**Failure mode:** Language descriptions can leak hidden rules. The branch must
be training-only and family-disjoint.

**Experiment mapping:** .395 ablates template-privileged training on ARC-TGI
families with privileged features removed at test time.

## Flagged for .395

`arc_tgi_family_generator_cross_game_generalization_v395` is the strongest next
method. The reason is conditional on the .394 outcomes: the ARC selector win
survived leak and multi-seed hardening, but synthesis did not break the
selection ceiling, cross-game transfer was blocked by missing game IDs,
DiffusionGemma preflight did not load, and code replication was corpus-specific.
Therefore .395 should repair the missing transfer substrate first: build a
transparent provenance manifest plus ARC-TGI-style family-disjoint candidate
pool, then rerun Set-Encoder versus vote on held-out task families. Keep
DiffusionGemma as loader repair, not a full-run .395 bet.

random_seed=4265
