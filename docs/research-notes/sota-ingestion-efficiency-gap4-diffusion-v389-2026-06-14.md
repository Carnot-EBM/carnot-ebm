# SOTA ingestion 2026-06-14: efficiency, GAP-4, and diffusion map for .389

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_efficiency_gap4_diffusion_mapped_v389`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `Reward-Guided Stitching DiffusionGemma scale-up`, arxiv_id_or_url: `2602.22871`, url: `https://arxiv.org/abs/2602.22871`}
  - {name: `S^3 verifier-guided denoising search`, arxiv_id_or_url: `2604.06260`, url: `https://arxiv.org/abs/2604.06260`}
  - {name: `Self-Rewarding SMC particle guidance`, arxiv_id_or_url: `2602.01849`, url: `https://arxiv.org/abs/2602.01849`}
  - {name: `OpenReview cve4NOiyVp judge-cost tuning`, arxiv_id_or_url: `2501.17178`, url: `https://arxiv.org/abs/2501.17178`}
  - {name: `When To Solve/Verify compute-normalized verifier bar`, arxiv_id_or_url: `2504.01005`, url: `https://arxiv.org/abs/2504.01005`}
  - {name: `ThinkPRM process-verifier comparator`, arxiv_id_or_url: `2504.16828`, url: `https://arxiv.org/abs/2504.16828`}
  - {name: `CEM operator authorization flag`, arxiv_id_or_url: `2510.20607`, url: `https://arxiv.org/abs/2510.20607`}
  - principle: Each method MUST carry a real arXiv ID/URL; an ingestion note without verifiable citations is treated as fabrication (adversarial_verify discipline).
- cem_operator_authorization_flag:
  - principle: Explicitly records that CEM (2510.20607) needs operator authorization before activation (the retired trained-content-energy selector lineage) - closes the loop honestly instead of silently dropping or auto-running it.
  - source_id: `2510.20607`
  - operator_authorization_required: `true`
  - auto_activation_recommended: `false`
  - retirement_marker: `gap3_trained_content_energy_selector_retired_stage2v2_2026_06_09`
- flagged_for_v389: `s3_diffusiongemma_verifier_guided_search_scaleup_v389`
  - principle: Closes discover->ingest->plan: names the strongest method for the next planner.

## Fresh-pass provenance

Read `research-references.md` `.388 planning sweep`, `research-studying.md`,
and `ops/exclusion_manifest.yaml` for the GAP-3 trained-content-energy selector
retirement. The CEM entry is therefore surfaced to the operator instead of
being silently dropped, auto-activated, or treated as eligible for
auto-activation.

Reliable-channel helper pass, not `/deep-research`:
- `.venv/bin/python scripts/sweep_clusters.py 0 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 1 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 3 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "Diffusion Language Models reward-guided stitching stratified scaling search self-rewarding SMC" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "LLM judge cost normalization compute optimal verification ThinkPRM" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "compositional energy minimization ARC learned energy landscapes" --limit 8`

The cluster helper emitted broadened verifier, energy, and world-model arXiv
API URLs. Semantic Scholar returned arXiv:2602.22871 for the diffusion query and
HTTP 429 for the judge-cost and CEM focused queries, so no S2-only promotion is
claimed. Low-concurrency WebSearch/WebFetch verified arXiv:2602.22871,
arXiv:2604.06260, arXiv:2602.01849, OpenReview:cve4NOiyVp,
arXiv:2501.17178, arXiv:2504.01005, arXiv:2504.16828, and arXiv:2510.20607.

## SOTA -> experiment mapping

## Reward-Guided Stitching DiffusionGemma scale-up

**Method/source:** Test-Time Scaling with Diffusion Language Models via
Reward-Guided Stitching, arXiv:2602.22871
(https://arxiv.org/abs/2602.22871), turns diffusion-sampled partial reasoning
into a pool of step-level candidates and stitches high-scoring steps.

**Carnot stack mapping:** This maps to DiffusionGemma guidance scale-up: score
intermediate denoising or reasoning steps with Carnot's verifier, then reuse
good partials instead of only reranking completed samples.

**Implication:** Parallel diffusion rollouts can become reusable search
material for .389 rather than disposable final-answer samples.

**Failure mode:** The paper depends on PRM-style step scores and an AR solver
to repair stitched rationales; it does not prove Carnot's executable energy can
score intermediate ARC or code states.

**Experiment mapping:** Run it as an ablation beside final-output rerank and
S^3-style denoising search with matched verifier-call budgets.

## S^3 verifier-guided denoising search

**Method/source:** S^3: Stratified Scaling Search for Test-Time in Diffusion
Language Models, arXiv:2604.06260 (https://arxiv.org/abs/2604.06260), expands
and scores denoising-frontier candidates, then resamples promising trajectories
while preserving diversity.

**Carnot stack mapping:** This is the cleanest DiffusionGemma guidance scale-up
map: place the Carnot verifier inside the denoising frontier search rather than
after generation has already collapsed to final strings.

**Implication:** .389 can test whether executable verifier energy improves
masked-diffusion search under a fixed denoising and verifier-call budget.

**Failure mode:** S^3 uses a lightweight reference-free verifier on language
benchmarks; Carnot must prove partial-state scoring is valid for the executable
domains it cares about.

**Experiment mapping:** Flag `s3_diffusiongemma_verifier_guided_search_scaleup_v389`
as the next planner target with no-guidance, best-of-K, self-rewarding SMC, and
Carnot-verifier frontier-search arms.

## Self-Rewarding SMC particle guidance

**Method/source:** Self-Rewarding Sequential Monte Carlo for Masked Diffusion
Language Models, arXiv:2602.01849 (https://arxiv.org/abs/2602.01849), uses
trajectory confidence to weight and resample multiple masked-diffusion
particles.

**Carnot stack mapping:** This is the self-guided control for DiffusionGemma:
parallel particle search without external Carnot verifier calls.

**Implication:** The .389 scale-up can distinguish a true external-verifier
gain from ordinary benefits of particle search and resampling.

**Failure mode:** Model confidence is not executable correctness, so a
self-rewarding run can become more confident without becoming more valid.

**Experiment mapping:** Include as the SMC comparator against S^3/Carnot-guided
denoising under the same particle count and denoising steps.

## OpenReview cve4NOiyVp judge-cost tuning

**Method/source:** Tuning LLM Judge Design Decisions for 1/1000 of the Cost,
OpenReview:cve4NOiyVp (https://openreview.net/forum?id=cve4NOiyVp) and
arXiv:2501.17178 (https://arxiv.org/abs/2501.17178), tunes LLM-judge settings
with multi-objective, multi-fidelity search.

**Carnot stack mapping:** This maps to the efficiency-moat judge comparator:
compare Carnot's executable verifier to a tuned open-weight judge frontier, not
to a single expensive judge setting.

**Implication:** Carnot must report cost-normalized accuracy against a strong
judge baseline that trades accuracy, tokens, latency, and model choice.

**Failure mode:** A tuned LLM judge remains an opaque evaluator, not an
executable action or transition verifier.

**Experiment mapping:** Add tuned judge arms and cost-per-accepted-correct
normalization to the .389 efficiency-moat table.

## When To Solve/Verify compute-normalized verifier bar

**Method/source:** When To Solve, When To Verify, arXiv:2504.01005
(https://arxiv.org/abs/2504.01005), compares extra solution sampling against
generative verification under fixed inference budgets.

**Carnot stack mapping:** This maps to the efficiency-moat normalization:
every verifier result must be compared to spending the same budget on more
candidate generation and self-consistency.

**Implication:** A Carnot verifier result that improves accuracy but costs more
than scaled sampling is not a moat.

**Failure mode:** It studies generative reward-model verification rather than
Carnot executable energy, so it sets the bar rather than the implementation.

**Experiment mapping:** Keep vote@K, judge@K, Carnot-rerank@K, verifier calls,
wall-clock, and token-cost columns in the .389 comparator.

## ThinkPRM process-verifier comparator

**Method/source:** Process Reward Models That Think, arXiv:2504.16828
(https://arxiv.org/abs/2504.16828), uses long-CoT generative process
verification with far fewer process labels than discriminative PRMs.

**Carnot stack mapping:** This maps to the expensive quality comparator around
the efficiency moat.

**Implication:** If ThinkPRM wins quality, Carnot can still have a moat only if
it occupies a cheaper cost-normalized point with acceptable accuracy.

**Failure mode:** ThinkPRM's long generative judging is too expensive to be the
cheap executable verifier mechanism Carnot is trying to prove.

**Experiment mapping:** Report it as the process-verifier quality ceiling and
separate that from Carnot's cheap executable-verifier arm.

## CEM operator authorization flag

**Method/source:** Generalizable Reasoning through Compositional Energy
Minimization, arXiv:2510.20607 (https://arxiv.org/abs/2510.20607), learns
subproblem energy landscapes and composes them at inference time.

**Carnot stack mapping:** Conceptually, CEM maps to learned compositional ARC
energies. Operationally, the adjacent GAP-3 trained-content-energy selector
lineage is retired in `ops/exclusion_manifest.yaml`, so this ingestion cannot
activate it.

**Implication:** CEM should be surfaced to the operator as a possible reopened
line, with operator authorization required before activation.

**Failure mode:** The retired selector lineage already fit synthetic curricula
and still landed at random on real candidate selection; CEM does not remove the
gate-1R requirement.

**Experiment mapping:** Record `cem_operator_authorization_flag` with
`operator_authorization_required=true`, `auto_activation_recommended=false`,
and retirement marker `gap3_trained_content_energy_selector_retired_stage2v2_2026_06_09`.
Do not auto-activate or recommend CEM as the `.389` method.

## Flagged for .389

`s3_diffusiongemma_verifier_guided_search_scaleup_v389` is the strongest
follow-on. It directly tests verifier-guided search inside the DiffusionGemma
denoising loop, while Reward-Guided Stitching and Self-Rewarding SMC become
ablation/control arms. CEM remains operator-only until operator authorization
and gate-1R are satisfied.

