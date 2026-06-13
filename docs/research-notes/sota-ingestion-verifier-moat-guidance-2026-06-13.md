# SOTA ingestion 2026-06-13: verifier moat and DiffusionGemma guidance map

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_verifier_moat_guidance_mapped`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `ARBITER reasoning-basin verifier-moat anchor`, arxiv_id_or_url: `2605.26172`, url: `https://arxiv.org/abs/2605.26172`}
  - {name: `ThinkPRM data-efficient process verifier`, arxiv_id_or_url: `2504.16828`, url: `https://arxiv.org/abs/2504.16828`}
  - {name: `Optimal LLM+PRM aggregation`, arxiv_id_or_url: `2510.13918`, url: `https://arxiv.org/abs/2510.13918`}
  - {name: `RLV unified reasoner-verifier value head`, arxiv_id_or_url: `2505.04842`, url: `https://arxiv.org/abs/2505.04842`}
  - {name: `EntRGi entropy-aware reward guidance`, arxiv_id_or_url: `2602.05000`, url: `https://arxiv.org/abs/2602.05000`}
  - {name: `Executable World Models for ARC-AGI-3`, arxiv_id_or_url: `2605.05138`, url: `https://arxiv.org/abs/2605.05138`}
  - {name: `ARC-AGI-3 technical report`, arxiv_id_or_url: `2603.24621`, url: `https://arxiv.org/abs/2603.24621`}
  - principle: Each method/source MUST carry a real arXiv ID/URL (verified); an ingestion note without verifiable citations is treated as fabrication.
- flagged_for_v386: `entrgi_diffusiongemma_energy_guidance_after_positive_discrimination_gate_v386`
  - principle: Closes the discover->ingest->plan loop: names the strongest method for the next planner.

## Fresh-pass provenance

Read `research-studying.md` and `research-references.md` filtered to
verifier-vs-self-consistency, reward-guided-generation, and ARC-AGI-3, plus
`results/experiment_4152_sota_ingestion_recursive_reasoner_verifier.json` for
the DiffusionGemma gate provenance. The prior TRM/TTA-TRM/V-STaR/SEDD/CFG
milestone ingestion is treated as already banked and is not duplicated here.

Reliable-channel helper pass, not `/deep-research`:
- `python3 scripts/sweep_clusters.py 0 --max-results 8`
- `python3 scripts/sweep_clusters.py 1 --max-results 8`
- `python3 scripts/sweep_clusters.py 3 --max-results 8`
- `python3 scripts/sweep_semscholar.py "verifier self consistency process reward aggregation LLM PRM" --limit 8`
- `python3 scripts/sweep_semscholar.py "reward guided generation diffusion language model energy guidance" --limit 8`
- `python3 scripts/sweep_semscholar.py "ARC-AGI-3 executable world models tech report" --limit 8`

The cluster helper emitted the broadened verifier, energy, and world-model
arXiv API URLs. Semantic Scholar returned `2510.13918` among eight IDs for the
verifier/self-consistency query and HTTP 429 for the reward-guidance and
ARC-AGI-3 queries. Low-concurrency WebSearch/WebFetch verified all seven
requested arXiv anchors: arXiv:2605.26172, arXiv:2504.16828, arXiv:2510.13918,
arXiv:2505.04842, arXiv:2602.05000, arXiv:2605.05138, and arXiv:2603.24621.

## SOTA -> experiment mapping

## ARBITER reasoning-basin verifier-moat anchor

**Method/source:** ARBITER, arXiv:2605.26172
(https://arxiv.org/abs/2605.26172), shows sampled reasoning trajectories
cluster into basins and that majority vote can pick a stable wrong basin even
when the correct answer is present in the candidate pool.

**Carnot moat implication:** This is the .385 rerank-recovery design anchor:
the external Carnot verifier must recover correct minority candidates that
self-consistency misses. The verifier should aggregate with the vote, because
vote mass is still useful evidence; it should not replace the vote blindly.

**Efficiency implication:** The relevant metric is recovered oracle headroom per
unit cost. A cheap executable verifier earns its place only if it recovers
wrong-majority cases at lower cost than LLM-judge rescoring.

**DiffusionGemma guidance implication:** Do not launch guidance just because a
diffusion substrate exists. First require a positive discrimination gate showing
the verifier can separate correct minority basins from stable wrong basins.

## ThinkPRM data-efficient process verifier

**Method/source:** Process Reward Models That Think, arXiv:2504.16828
(https://arxiv.org/abs/2504.16828), trains generative process verifiers with far
fewer process labels than ordinary discriminative PRMs and reports wins against
LLM-as-judge and other verifier baselines.

**Carnot moat implication:** It is the positive existence proof that a verifier
can beat self-consistency when it checks the process instead of only final
answer agreement. Carnot should score verifier-plus-vote against vote-only,
LLM-judge, and post-hoc verifier-only baselines.

**Efficiency implication:** ThinkPRM makes the LLM-judge efficiency comparison
load-bearing: if a verifier is more expensive than judge rescoring, it is not
the Carnot moat.

**DiffusionGemma guidance implication:** Process-style partial scores are the
right shape for denoising-time guidance, where the sampler needs intermediate
signals before a final candidate exists.

## Optimal LLM+PRM aggregation

**Method/source:** Optimal Aggregation of LLM and PRM Signals for Efficient
Test-Time Scaling, arXiv:2510.13918 (https://arxiv.org/abs/2510.13918), argues
for calibrated weighted aggregation of LLM and PRM signals and reports better
test-time scaling efficiency than vanilla weighted vote.

**Carnot moat implication:** The verifier should be a calibrated term in the
selector, not an unconditional replacement for the vote. The .385 artifact
should therefore report vote-only, verifier-only, and calibrated
vote-plus-verifier arms.

**Efficiency implication:** Precomputed aggregation weights are the cheap path:
spend compute once to calibrate the selector rather than repeatedly increasing
candidate K or sending each candidate to a large judge.

**DiffusionGemma guidance implication:** The same calibration lesson applies to
guidance weights. Carnot energy should be swept and mixed with the base
denoising confidence, with no-guidance and base-guidance controls.

## RLV unified reasoner-verifier value head

**Method/source:** Putting the Value Back in RL, arXiv:2505.04842
(https://arxiv.org/abs/2505.04842), co-trains LLM reasoners with a generative
verifier/value capability and reports efficient parallel test-time scaling.

**Carnot moat implication:** RLV supports the reward-graft thesis: a value or
verifier head can make parallel samples more selectable. The Carnot claim still
needs the external-verifier-vs-self-consistency head-to-head and a
vote-plus-verifier aggregation arm.

**Efficiency implication:** The strongest .386 efficiency test is a cheap
verifier/value head versus LLM-judge rescoring under matched candidate pools
and matched parallel sampling.

**DiffusionGemma guidance implication:** A learned value head is a plausible
reward for token-level or trace-level guidance, but only after executable
labels confirm it tracks validity rather than model preference artifacts.

## EntRGi entropy-aware reward guidance

**Method/source:** EntRGi, arXiv:2602.05000
(https://arxiv.org/abs/2602.05000), studies reward guidance for discrete
diffusion language models by interpolating between continuous token relaxations
and hard tokens according to predictive entropy.

**Carnot moat implication:** EntRGi is not evidence that Carnot has a moat. It
is the implementation template to use after the moat gate is positive.

**Efficiency implication:** Guidance spends verifier/reward calls during
denoising rather than after generation. The next run must report cost per
reward call and compare it to post-hoc rerank and LLM-judge baselines.

**DiffusionGemma guidance implication:** This is the strongest .386 method:
apply Carnot verifier energy through entropy-aware soft/hard token
interpolation during DiffusionGemma denoising, gated on positive verifier
discrimination.

## Executable World Models for ARC-AGI-3

**Method/source:** Executable World Models for ARC-AGI-3, arXiv:2605.05138
(https://arxiv.org/abs/2605.05138), evaluates coding agents that maintain,
verify, refactor, and plan through executable Python world models for ARC-AGI-3.

**Carnot moat implication:** The moat can be transition verification and action
selection, not only answer reranking. Carnot should keep executable validation
as the external signal that self-consistency lacks.

**Efficiency implication:** Use action efficiency and RHAE-style metrics. A
verifier that reduces actions-to-progress is valuable even before it increases
the headline solve count.

**DiffusionGemma guidance implication:** Guided generation can target compact
world-model edits, transition hypotheses, and plans. The verifier energy should
favor executable hypotheses that survive held-out transition checks.

## ARC-AGI-3 technical report

**Method/source:** ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence,
arXiv:2603.24621 (https://arxiv.org/abs/2603.24621), defines an interactive
benchmark centered on exploration, goal inference, world-model building, and
human-action-normalized adaptive efficiency.

**Carnot moat implication:** The verifier moat must show up as better adaptive
progress under real rules, not only as a better static answer selector.

**Efficiency implication:** The official benchmark framing makes efficiency
load-bearing. Report actions, RHAE-style ratios, and solve progress rather than
letting more compute masquerade as better intelligence.

**DiffusionGemma guidance implication:** Diffusion guidance should be aimed at
executable hypotheses and action plans that reduce exploration cost, not just
more fluent reasoning text.

## Flagged for .386

`entrgi_diffusiongemma_energy_guidance_after_positive_discrimination_gate_v386`
is the strongest follow-on. It should run only after .385 verifier
discrimination is positive. If the verifier does not beat or complement
self-consistency under calibrated vote aggregation, the next planner should
choose the RLV-style energy-verifier-vs-LLM-judge efficiency head-to-head
instead of spending on DiffusionGemma guidance.

