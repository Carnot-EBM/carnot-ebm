# SOTA ingestion 2026-06-14: moat, GAP-3, and diffusion map for .388

**Artifact fields**
- honest_verdict: `complete: sota_ingestion_moat_gap3_diffusion_mapped_v388`
  - principle: Terminal-prefixed. Records ingestion completed with verifiable citations.
- methods_mapped:
  - {name: `Unsolvability Ceiling headroom sanitization`, arxiv_id_or_url: `2605.07395`, url: `https://arxiv.org/abs/2605.07395`}
  - {name: `When To Solve/Verify accuracy-cost moat`, arxiv_id_or_url: `2504.01005`, url: `https://arxiv.org/abs/2504.01005`}
  - {name: `ThinkPRM process-verifier cost control`, arxiv_id_or_url: `2504.16828`, url: `https://arxiv.org/abs/2504.16828`}
  - {name: `CEM compositional ARC energy`, arxiv_id_or_url: `2510.20607`, url: `https://arxiv.org/abs/2510.20607`}
  - {name: `Self-Rewarding SMC DiffusionGemma guidance`, arxiv_id_or_url: `2602.01849`, url: `https://arxiv.org/abs/2602.01849`}
  - {name: `TRM ARC headroom-vote decomposition`, arxiv_id_or_url: `2512.11847`, url: `https://arxiv.org/abs/2512.11847`}
  - principle: Each method MUST carry a real arXiv ID/URL; an ingestion note without verifiable citations is treated as fabrication (adversarial_verify discipline).
- flagged_for_v388: `cem_gap3_stage2_compositional_arc_energy_v388`
  - principle: Closes discover->ingest->plan: names the strongest method for the next planner.

## Fresh-pass provenance

Read `research-references.md` `.387 planning sweep` and the
`research-studying.md` / `research-references.md` verifier-as-reward,
headroom, and energy-guided diffusion entries.

Reliable-channel helper pass, not `/deep-research`:
- `.venv/bin/python scripts/sweep_clusters.py 0 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 1 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "verifier as reward headroom compute optimal verification process reward model" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "compositional energy minimization ARC masked diffusion self rewarding SMC" --limit 8`

The cluster helper emitted the broadened verifier and energy arXiv API URLs.
Semantic Scholar returned HTTP 429 for both focused queries during this run, so
no fresh S2-only promotion is claimed. Low-concurrency WebSearch/WebFetch
verified the requested mapped paper set: arXiv:2605.07395, arXiv:2504.01005,
arXiv:2504.16828, arXiv:2510.20607, arXiv:2602.01849, and arXiv:2512.11847.

## SOTA -> experiment mapping

## Unsolvability Ceiling headroom sanitization

**Method/source:** Unsolvability Ceiling, arXiv:2605.07395
(https://arxiv.org/abs/2605.07395), audits multi-LLM routing headroom and
shows that judge bias, truncation, and output-format mismatch can inflate the
apparent oracle gap.

**Carnot stack mapping:** This maps to the A1 headroom-gate sanitization
already applied: use executable or exact objective checks, retain oracle@K,
and reject unsanitized judge-only headroom.

**Implication:** A positive verifier result is only decision-grade if it
survives objective oracle checks and cost-sensitive routing controls.

**Failure mode:** This paper does not provide a verifier, ARC energy, or
DiffusionGemma mechanism. It only tells us how the measurement can lie.

**Experiment mapping:** Keep A1 as a mandatory precondition for A3 and GAP-3:
no accuracy-cost moat claim without executable headroom.

## When To Solve/Verify accuracy-cost moat

**Method/source:** When To Solve, When To Verify, arXiv:2504.01005
(https://arxiv.org/abs/2504.01005), compares self-consistency with generative
verification under fixed inference budgets.

**Carnot stack mapping:** This maps to A3's accuracy-and-cost moat framing:
report vote@K, oracle@K, Carnot rerank@K, verifier calls, and wall-clock cost.

**Implication:** Carnot must prove that a cheap executable energy beats or
sits on the Pareto frontier against simply sampling more solutions.

**Failure mode:** The paper studies GenRM-style generative verification, not
Carnot's executable energy. It sets a cost bar but is not a direct substrate.

**Experiment mapping:** Keep the A3 table cost-normalized and do not report an
accuracy-only moat result.

## ThinkPRM process-verifier cost control

**Method/source:** ThinkPRM, arXiv:2504.16828
(https://arxiv.org/abs/2504.16828), uses generative process verification with
small synthetic supervision to judge reasoning steps.

**Carnot stack mapping:** This also maps to A3: it is the high-quality PRM
comparator for an accuracy-and-cost moat, not the cheap-energy method itself.

**Implication:** Verifier advantage is real but budget-conditional; Carnot must
show both quality and cost separation from an expensive process verifier.

**Failure mode:** ThinkPRM's long generative judging can be too expensive for
Carnot's claimed moat, so it validates the target class without proving our
efficiency claim.

**Experiment mapping:** Use ThinkPRM as the expensive quality comparator while
Carnot energy must carry the cheap verifier arm.

## CEM compositional ARC energy

**Method/source:** Generalizable Reasoning through Compositional Energy
Minimization, arXiv:2510.20607 (https://arxiv.org/abs/2510.20607), learns
subproblem energy landscapes and composes them at test time.

**Carnot stack mapping:** This is the GAP-3 Stage-2 compositional ARC energy
map: factor rule/content energies, compose them on held-out tasks, and sample
with a PEM-like parallel minimization loop.

**Implication:** This is the strongest .388 method because it targets the next
unbuilt Carnot capability: a learned transition energy, not another post-hoc
reranker.

**Failure mode:** CEM is not an ARC-AGI-3 result and not a Carnot executable
transition verifier. Transfer must be measured on the GAP-3 harness.

**Experiment mapping:** Flag .388 for `cem_gap3_stage2_compositional_arc_energy_v388`:
train local transformation energies, compose them on held-out ARC-style tasks,
and compare PEM sampling against the current GAP-3 candidates.

## Self-Rewarding SMC DiffusionGemma guidance

**Method/source:** Self-Rewarding SMC for Masked Diffusion Language Models,
arXiv:2602.01849 (https://arxiv.org/abs/2602.01849), weights and resamples
parallel masked-diffusion particles using trajectory confidence.

**Carnot stack mapping:** This maps to the queued DiffusionGemma guidance plan:
use particle weighting and resampling as the training-free guidance template.

**Implication:** A future DiffusionGemma run can convert parallel denoising
capacity into better candidates without training a separate reward model.

**Failure mode:** The reward is model confidence, not external executable
correctness. It can improve confident sampling without improving task validity.

**Experiment mapping:** Keep SMC-style guidance queued behind A3: use it for
DiffusionGemma guidance once the energy gate is positive.

## TRM ARC headroom-vote decomposition

**Method/source:** Tiny Recursive Models on ARC-AGI-1, arXiv:2512.11847
(https://arxiv.org/abs/2512.11847), decomposes TRM performance into voting,
identity conditioning, and shallow recursion effects.

**Carnot stack mapping:** This maps to the TRM headroom/vote decomposition:
separate single-pass, vote@1000, oracle@K, and identity-ID ablation before
crediting a verifier.

**Implication:** TRM can remain the candidate generator, but only with a clean
headroom/vote receipt that shows where selection value could exist.

**Failure mode:** The ablation is a warning about substrate artifacts, not an
energy or guidance mechanism.

**Experiment mapping:** Keep TRM as a controlled generator and require identity
conditioning plus vote/headroom ablations before any rerank claim.

## Flagged for .388

`cem_gap3_stage2_compositional_arc_energy_v388` is the strongest follow-on.
A1 headroom sanitation and A3 accuracy-cost framing are necessary gates, and
Self-Rewarding SMC is a useful DiffusionGemma template after the verifier gate
turns positive. The method that most directly advances Carnot's stack is CEM:
it gives GAP-3 Stage-2 a concrete learned compositional energy experiment.

