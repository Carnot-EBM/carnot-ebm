# SOTA ingestion 2026-06-16: .395 forks map for .396

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_v396_mapped`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `Paying Less Generalization Tax cross-domain RL recipe`, arxiv_id_or_url: `2601.18217`, url: `https://arxiv.org/abs/2601.18217`}
  - {name: `ARC-GEN mimetic procedural ARC generator`, arxiv_id_or_url: `2511.00162`, url: `https://arxiv.org/abs/2511.00162`}
  - {name: `RFG reward-free guidance for diffusion LLM reasoning`, arxiv_id_or_url: `2509.25604`, url: `https://arxiv.org/abs/2509.25604`}
  - {name: `Self-Improving LLM Agents at Test-Time`, arxiv_id_or_url: `2510.07841`, url: `https://arxiv.org/abs/2510.07841`}
  - {name: `SEVerA verified synthesis for self-evolving agents`, arxiv_id_or_url: `2603.25111`, url: `https://arxiv.org/abs/2603.25111`}
  - principle: Each method MUST carry a real arXiv ID/URL (no citation = fabrication per adversarial_verify discipline) + a one-line .396 experiment mapping.
- flagged_for_v396: `rfg_diffusiongemma_full_run_plus_arcgen_transfer_stress_v396`
  - principle: Closes discover->ingest->plan: names the strongest method for the .396 planner, conditioned on whether cross-family generalized or collapsed.
- random_seed: `4276`
  - principle: Determinism placeholder for the discovery query set (recorded for reproducibility of the sweep).

## Fresh-pass provenance

Read `CLAUDE.md` SOTA-Ingestion Cycle Discipline, `research-studying.md`,
`research-references.md`, `results/experiment_4265_sota_ingestion_v395.json`,
`results/experiment_4271_arc_cross_family_transfer_existing_pool.json`,
`results/experiment_4272_arc_cross_family_transfer_fresh_tgi_pool.json`,
`scripts/sweep_clusters.py`, and `scripts/sweep_semscholar.py`.

Reliable-channel helper pass, not `/deep-research`:
- `python3 -c "import importlib; importlib.import_module('scripts.sweep_clusters'); importlib.import_module('scripts.sweep_semscholar')"`
- `python3 scripts/sweep_clusters.py 0 --max-results 8`
- `python3 scripts/sweep_clusters.py 1 --max-results 8`
- `python3 scripts/sweep_clusters.py all --max-results 8`
- `python3 scripts/sweep_semscholar.py "cross task verifier generalization learned selector out of domain test time adaptation" --limit 8`
- `python3 scripts/sweep_semscholar.py "diffusion language model process reward verifier guidance discrete diffusion DPRM" --limit 8`

The sweep helpers imported successfully and `sweep_clusters.py` emitted the
broadened arXiv API URLs for the reliable channel. The local Semantic Scholar
helper was reachable as code but degraded at fetch time due to TLS certificate
verification failure, so no Semantic-Scholar-only promotion is claimed.
WebSearch/WebFetch was reachable and verified arXiv:2601.18217,
arXiv:2511.00162, arXiv:2509.25604, arXiv:2510.07841, arXiv:2603.25111,
plus prior-covered context arXiv:2604.24357, arXiv:2602.05000, and
arXiv:2603.05099. The banned `/deep-research` channel was not invoked.

## Prior-covered methods not re-ingested

The .394/.395 sweeps already covered ARC-TGI (arXiv:2603.05099), Reliability
Gap (arXiv:2606.03305), DPRM (arXiv:2604.24357), entropy-guided diffusion RL
(arXiv:2603.12554), L-VARC (arXiv:2606.12847), TrajAD (arXiv:2602.06443),
RL^V / Putting the Value Back in RL (arXiv:2505.04842), EntRGi
(arXiv:2602.05000), and Self-Trained Verification (arXiv:2605.30290). They
remain context for .396, but they are not counted as fresh `methods_mapped`
rows here.

## .395 cross-family outcome read

Exp 4271: `cross_family_generalizes`, `cross_family_win_holds=true`,
`cross_family_delta=0.4038461538`, `cross_family_ci95=[0.25, 0.5576923077]`,
`held_out_family_n=52`, `held_out_task_n=52`, and `verifier_is_oracle=false`.
The hardened Set-Encoder selector survived the load-bearing OOD gate, so .396
can treat the selector as a general transfer signal rather than within-pool
memorization.

Exp 4272 was correctly blocked because Exp 4270 found
`family_split_feasible=true` for the existing pool; the fresh ARC-TGI fallback
was not needed for .395. This means .396 should use fresh procedural generators
as a stronger stress test, not as a repair for a failed .395 gate.

## SOTA -> experiment mapping

## Paying Less Generalization Tax: stress the transfer claim

**Method/source:** Paying Less Generalization Tax, arXiv:2601.18217
(https://arxiv.org/abs/2601.18217), studies cross-domain RL transfer and
identifies state-information richness, planning complexity, and step-by-step
thinking as stronger transfer drivers than surface domain similarity.

**Carnot stack mapping:** Add family-rich but label-irrelevant ARC meta-features
and randomized distractor channels to the held-out-family split, then check
whether the generalized Set-Encoder lift survives removal or randomization of
those channels.

**.395 conditioning:** Because Exp 4271 generalized, .396 should deepen the
generalization headline rather than repair a collapse.

**Failure mode:** Randomization can become a cosmetic robustness test if the
added feature does not touch the causal family rule. Keep original ARC,
ARC-TGI, and ARC-GEN metrics separate.

**Experiment mapping:** .396 adds a richer cross-family stress split and reports
selector lift separately by source family.

## ARC-GEN: independent procedural-family replication

**Method/source:** ARC-GEN, arXiv:2511.00162
(https://arxiv.org/abs/2511.00162), is a mimetic procedural benchmark generator
covering all 400 ARC-AGI-1 tasks.

**Carnot stack mapping:** Build a second family-disjoint candidate pool from
ARC-GEN, materialize generator IDs and target hashes, and rerun the same
Set-Encoder versus vote bootstrap gate.

**.395 conditioning:** Exp 4271 generalized on recovered original-task family
IDs. ARC-GEN checks whether the transfer survives a different procedural
family substrate rather than only the recovered-manifest split.

**Failure mode:** Mimetic generation can clone original task quirks. Report
original-task and generated-family metrics separately.

**Experiment mapping:** .396 runs ARC-GEN as the independent transfer stress
gate after the .395 generalization win.

## RFG: bounded DiffusionGemma full run

**Method/source:** RFG, arXiv:2509.25604
(https://arxiv.org/abs/2509.25604), guides diffusion LLM reasoning with
log-likelihood ratios between enhanced and reference diffusion models instead
of explicit process-reward labels.

**Carnot stack mapping:** Pair loader-fixed DiffusionGemma with a reference
unguided pass, apply reward-free guidance at denoising time, and use the
generalized selector as the final exact-grid arbiter.

**.395 conditioning:** The cross-family selector generalized, so .396 can spend
the diffusion scale-up budget. RFG is the lowest-label-debt full-run method
when ARC process rewards remain sparse.

**Failure mode:** RFG can amplify the enhanced model's biases and may not align
with ARC exact match. Compare to unguided diffusion, selector-only, DPRM-style,
and EntRGi-style controls.

**Experiment mapping:** .396 runs a bounded DiffusionGemma RFG arm with exact
grid validation and cost-normalized controls.

## Test-Time Self-Improvement: low-margin selector adaptation

**Method/source:** Self-Improving LLM Agents at Test-Time, arXiv:2510.07841
(https://arxiv.org/abs/2510.07841), identifies uncertain cases, generates
similar examples, and adapts at test time from those examples.

**Carnot stack mapping:** Trigger adaptation only on low-margin held-out-family
tasks, create synthetic same-rule variants from the procedural generator,
fine-tune a tiny selector head, and compare to frozen-selector and random-update
controls.

**.395 conditioning:** A static selector already generalized, so adaptation is
an optional lift test, not a rescue path.

**Failure mode:** Adaptation can leak target structure or overfit one family.
Keep target outputs hidden and cap the adaptation budget.

**Experiment mapping:** .396 adds a low-margin TT-SI selector-head adaptation
arm on held-out ARC-GEN families.

## SEVerA: verified fallback for self-improving branches

**Method/source:** SEVerA, arXiv:2603.25111
(https://arxiv.org/abs/2603.25111), combines formal output contracts, verified
fallbacks, and learning over soft objectives for self-evolving agents.

**Carnot stack mapping:** Wrap online selector/refiner updates in contracts that
check grid shape, palette, immutable train examples, and provenance hashes, then
fallback to the frozen generalized selector on contract failure.

**.395 conditioning:** The selector generalized, but any .396 online
self-improvement branch needs guardrails before it can affect reported ARC
outputs.

**Failure mode:** Contracts enforce syntax and provenance, not semantic ARC
correctness. They are a safety guard, not a replacement for exact-match gates.

**Experiment mapping:** .396 guards TT-SI and DiffusionGemma refinement arms
with verified fallback contracts before counting online-adaptation lift.

## Flagged for .396

`rfg_diffusiongemma_full_run_plus_arcgen_transfer_stress_v396` is the strongest
next method because the condition is the positive one: Exp 4271 reports
`cross_family_generalizes` with `cross_family_win_holds=true`,
`cross_family_delta=0.4038461538`, and CI95 `[0.25, 0.5576923077]`. That opens
the scale-up gate. The .396 planner should run a bounded RFG-style
DiffusionGemma full-run arm using the generalized selector as the exact-grid
arbiter, while ARC-GEN supplies the independent procedural-family stress test.
If the cross-family result had collapsed, the correct flag would have been
`generalizing_verifier_meta_feature_repair_v396`; it did not collapse.

random_seed=4276
