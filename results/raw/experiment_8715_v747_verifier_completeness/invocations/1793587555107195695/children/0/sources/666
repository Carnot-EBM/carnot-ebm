"""Exp 4192 SOTA ingestion for efficiency, GAP-4, and diffusion planning.

Spec refs: REQ-REPORT-4192, SCENARIO-REPORT-4192.

This module writes a planning artifact, not a benchmark result. It closes the
discover->ingest->plan loop for the `.388 planning sweep`, while explicitly
keeping CEM behind operator authorization because the related GAP-3
trained-content-energy selector lineage is retired.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any


REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "honest_verdict",
        "methods_mapped",
        "cem_operator_authorization_flag",
        "flagged_for_v389",
        "field_principles",
    }
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "name",
        "arxiv_id_or_url",
        "url",
        "carnot_stack_mapping",
        "implication",
        "failure_mode",
        "experiment_mapping",
    }
)
REQUIRED_CEM_FLAG_FIELDS = frozenset(
    {
        "source_id",
        "url",
        "operator_authorization_required",
        "auto_activation_recommended",
        "retired_lineage",
        "retirement_marker",
        "required_gate",
        "reason",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = (
    "complete: sota_ingestion_efficiency_gap4_diffusion_mapped_v389"
)
DEFAULT_FLAGGED_FOR_V389 = (
    "s3_diffusiongemma_verifier_guided_search_scaleup_v389"
)
RETIREMENT_MARKER = "gap3_trained_content_energy_selector_retired_stage2v2_2026_06_09"

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records ingestion completed with verifiable citations.",
    "methods_mapped": (
        "Each method MUST carry a real arXiv ID/URL; an ingestion note without "
        "verifiable citations is treated as fabrication (adversarial_verify discipline)."
    ),
    "cem_operator_authorization_flag": (
        "Explicitly records that CEM (2510.20607) needs operator authorization "
        "before activation (the retired trained-content-energy selector lineage) "
        "- closes the loop honestly instead of silently dropping or auto-running it."
    ),
    "flagged_for_v389": (
        "Closes discover->ingest->plan: names the strongest method for the next planner."
    ),
}

VERIFIED_SOURCE_URLS = {
    "2602.22871": "https://arxiv.org/abs/2602.22871",
    "2604.06260": "https://arxiv.org/abs/2604.06260",
    "2602.01849": "https://arxiv.org/abs/2602.01849",
    "2501.17178": "https://arxiv.org/abs/2501.17178",
    "2504.01005": "https://arxiv.org/abs/2504.01005",
    "2504.16828": "https://arxiv.org/abs/2504.16828",
    "2510.20607": "https://arxiv.org/abs/2510.20607",
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    {f"arXiv:{source}" for source in VERIFIED_SOURCE_URLS}
    | {"OpenReview:cve4NOiyVp"}
)

DEFAULT_CEM_OPERATOR_AUTHORIZATION_FLAG = {
    "source_id": "2510.20607",
    "url": "https://arxiv.org/abs/2510.20607",
    "operator_authorization_required": True,
    "auto_activation_recommended": False,
    "retired_lineage": "GAP-3 trained-content-energy ARC candidate selector",
    "retirement_marker": RETIREMENT_MARKER,
    "required_gate": (
        "operator authorization plus gate-1R: gold-vs-REAL-mined-near-miss "
        "AUROC >= 0.70 on tasks held out from training and checkpoint selection"
    ),
    "reason": (
        "ops/exclusion_manifest.yaml retires the trained-content-energy lineage "
        "after Stage-2 v1/v2 selection landed statistically at random; CEM may "
        "be surfaced to the operator but must not be auto-activated."
    ),
}

DEFAULT_METHODS_MAPPED = [
    {
        "name": "Reward-Guided Stitching DiffusionGemma scale-up",
        "arxiv_id_or_url": "2602.22871",
        "url": "https://arxiv.org/abs/2602.22871",
        "carnot_stack_mapping": (
            "DiffusionGemma guidance scale-up: sample many low-cost masked "
            "diffusion trajectories, score intermediate steps with the Carnot "
            "verifier, then stitch reusable high-score steps before final solving."
        ),
        "implication": (
            "The next DiffusionGemma experiment can exploit parallel partial "
            "reasoning rather than only final-answer reranking, which matches "
            "Carnot's verifier-as-guidance thesis."
        ),
        "failure_mode": (
            "The paper relies on PRM-style step scoring and an autoregressive "
            "solver to repair stitched rationales; it does not prove Carnot's "
            "executable energy can score useful intermediate ARC states."
        ),
        "experiment_mapping": (
            "Use as a .389 ablation: compare final-output rerank, step-level "
            "stitching, and no-guidance DiffusionGemma under matched verifier "
            "calls and wall-clock budgets."
        ),
    },
    {
        "name": "S^3 verifier-guided denoising search",
        "arxiv_id_or_url": "2604.06260",
        "url": "https://arxiv.org/abs/2604.06260",
        "carnot_stack_mapping": (
            "DiffusionGemma guidance scale-up: expand multiple denoising "
            "frontier candidates, score them with a lightweight verifier, and "
            "resample promising trajectories while preserving diversity."
        ),
        "implication": (
            "This is the strongest .389 method because it puts verifier guidance "
            "inside the denoising loop, exactly where DiffusionGemma exposes "
            "test-time search budget."
        ),
        "failure_mode": (
            "S^3 uses a reference-free verifier on language benchmarks; Carnot "
            "must show that executable ARC or code validators can score partial "
            "denoising states without collapsing diversity."
        ),
        "experiment_mapping": (
            "Flag .389 for S^3-style verifier-guided denoising: no-guidance, "
            "best-of-K, self-rewarding SMC, and Carnot-verifier frontier search "
            "arms with normalized verifier-call and latency accounting."
        ),
    },
    {
        "name": "Self-Rewarding SMC particle guidance",
        "arxiv_id_or_url": "2602.01849",
        "url": "https://arxiv.org/abs/2602.01849",
        "carnot_stack_mapping": (
            "DiffusionGemma guidance scale-up: maintain interacting masked "
            "diffusion particles and use trajectory confidence for weighting "
            "and resampling when an external verifier is unavailable."
        ),
        "implication": (
            "This provides the no-external-verifier particle-search control for "
            "the .389 DiffusionGemma scale-up, keeping the Carnot-verifier gain "
            "from being confused with generic parallel search."
        ),
        "failure_mode": (
            "The reward is model confidence, not task correctness; it can improve "
            "fluent or globally confident samples while still missing executable "
            "validity."
        ),
        "experiment_mapping": (
            "Use as the self-guided SMC comparator against S^3/Carnot-guided "
            "denoising under identical particle counts and denoising steps."
        ),
    },
    {
        "name": "OpenReview cve4NOiyVp judge-cost tuning",
        "arxiv_id_or_url": "2501.17178",
        "url": "https://arxiv.org/abs/2501.17178",
        "carnot_stack_mapping": (
            "Efficiency-moat judge comparator: use multi-objective, multi-fidelity "
            "judge tuning to normalize LLM-judge accuracy against dollar, token, "
            "and latency cost."
        ),
        "implication": (
            "Carnot's efficiency claim must beat a tuned open-weight judge "
            "frontier, not a single expensive frontier-judge configuration."
        ),
        "failure_mode": (
            "It optimizes LLM judges for evaluation, not executable validators; "
            "a cheaper judge can still be an opaque comparator rather than an "
            "ARC/action verifier."
        ),
        "experiment_mapping": (
            "In the .389 efficiency table, include tuned open-weight judge arms "
            "from OpenReview:cve4NOiyVp / arXiv:2501.17178 and compare cost per "
            "accepted correct candidate."
        ),
    },
    {
        "name": "When To Solve/Verify compute-normalized verifier bar",
        "arxiv_id_or_url": "2504.01005",
        "url": "https://arxiv.org/abs/2504.01005",
        "carnot_stack_mapping": (
            "Efficiency-moat cost normalization: compare verifier compute "
            "against spending the same budget on more solution samples and "
            "self-consistency."
        ),
        "implication": (
            "Any Carnot verifier win must be reported as an accuracy-cost Pareto "
            "point; a raw accuracy lift is insufficient when extra sampling may "
            "be cheaper."
        ),
        "failure_mode": (
            "The paper studies generative reward-model verification, not "
            "Carnot's cheap executable energy, so it sets the bar rather than "
            "the implementation."
        ),
        "experiment_mapping": (
            "Carry fixed-budget vote@K, judge@K, Carnot-rerank@K, verifier-call "
            "count, wall-clock, and token-cost columns in the .389 comparator."
        ),
    },
    {
        "name": "ThinkPRM process-verifier comparator",
        "arxiv_id_or_url": "2504.16828",
        "url": "https://arxiv.org/abs/2504.16828",
        "carnot_stack_mapping": (
            "Efficiency-moat comparator: use ThinkPRM as the expensive, "
            "high-quality process verifier reference when evaluating whether "
            "Carnot's cheap verifier has a real moat."
        ),
        "implication": (
            "The moat should separate quality from cost: Carnot can lose quality "
            "to a strong PRM only if it clearly wins the cost-normalized frontier."
        ),
        "failure_mode": (
            "ThinkPRM is a long-CoT generative verifier with process supervision; "
            "it is too costly to be treated as Carnot's efficiency mechanism."
        ),
        "experiment_mapping": (
            "Report ThinkPRM-style judging as the quality ceiling and tuned "
            "open-weight judge as the cost frontier around the Carnot verifier."
        ),
    },
    {
        "name": "CEM operator authorization flag",
        "arxiv_id_or_url": "2510.20607",
        "url": "https://arxiv.org/abs/2510.20607",
        "carnot_stack_mapping": (
            "CEM maps conceptually to learned compositional ARC energies, but "
            "the adjacent GAP-3 trained-content-energy selector lineage is "
            "retired and cannot be activated by the ingestion workflow."
        ),
        "implication": (
            "The operator should see CEM as a possible future reopened line, "
            "not as a recommended automatic .389 experiment."
        ),
        "failure_mode": (
            "The retired lineage fit synthetic curricula yet landed at random "
            "on real candidate selection; CEM also does not by itself solve the "
            "real-mined near-miss gate that the manifest requires."
        ),
        "experiment_mapping": (
            "Record CEM only in cem_operator_authorization_flag: operator "
            "authorization and gate-1R are required before any activation, and "
            "auto-activation is explicitly not recommended."
        ),
    },
]

NOTE_MARKDOWN = """# SOTA ingestion 2026-06-14: efficiency, GAP-4, and diffusion map for .389

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
"""

STUDYING_SECTION = """## 2026-06-14 Exp 4192 - .388 planning sweep SOTA ingestion ingested

**Status:** INGESTED into `docs/research-notes/sota-ingestion-efficiency-gap4-diffusion-v389-2026-06-14.md`.

**Filtered track:** DiffusionGemma verifier-guided test-time scale-up,
efficiency-moat LLM-judge comparator and cost normalization, plus the CEM
operator-authorization closure for the retired GAP-3 trained-content-energy
selector lineage.

**Seed and fresh-pass candidates marked ingested:**
- Test-Time Scaling with Diffusion Language Models via Reward-Guided Stitching,
  arXiv:2602.22871 - mapped to step-level DiffusionGemma guidance and stitching
  ablations.
- S^3 Stratified Scaling Search, arXiv:2604.06260 - mapped to the strongest
  `.389` DiffusionGemma verifier-guided denoising-search target.
- Self-Rewarding SMC, arXiv:2602.01849 - mapped as the self-guided particle
  control for the DiffusionGemma scale-up.
- Tuning LLM Judge Design Decisions for 1/1000 of the Cost,
  OpenReview:cve4NOiyVp / arXiv:2501.17178 - mapped to tuned LLM-judge
  comparator and cost-normalized moat accounting.
- When To Solve/Verify, arXiv:2504.01005 - mapped to the fixed-budget
  solve-versus-verify normalization bar.
- ThinkPRM, arXiv:2504.16828 - mapped as the high-quality but expensive
  process-verifier comparator.
- CEM, arXiv:2510.20607 - re-flagged to the operator only:
  `operator_authorization_required=true`, `auto_activation_recommended=false`,
  retirement marker `gap3_trained_content_energy_selector_retired_stage2v2_2026_06_09`.

cem_operator_authorization_flag:
`source_id=2510.20607; operator_authorization_required=true; auto_activation_recommended=false; retirement_marker=gap3_trained_content_energy_selector_retired_stage2v2_2026_06_09`.

flagged_for_v389:
`s3_diffusiongemma_verifier_guided_search_scaleup_v389`.

Flagged for .389: `s3_diffusiongemma_verifier_guided_search_scaleup_v389`.

**Bottom line for the .389 roadmap:** run the S^3-style DiffusionGemma
verifier-guided denoising search first, with Reward-Guided Stitching and
Self-Rewarding SMC as ablation/control arms and judge-cost normalization around
the efficiency moat. Keep CEM on the operator surface only; do not activate it
until operator authorization is granted and gate-1R is passed.
"""

STUDYING_MARKER = "## 2026-06-14 Exp 4192 - .388 planning sweep SOTA ingestion ingested"


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]],
    cem_operator_authorization_flag: Mapping[str, object],
    flagged_for_v389: str,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-REPORT-4192 mapping artifact."""

    artifact: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped": [dict(method) for method in methods_mapped],
        "cem_operator_authorization_flag": dict(cem_operator_authorization_flag),
        "flagged_for_v389": flagged_for_v389,
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the JSON contract so uncited or unsafe method rows fail closed."""

    missing = REQUIRED_ARTIFACT_FIELDS.difference(artifact)
    extra = set(artifact).difference(REQUIRED_ARTIFACT_FIELDS)
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"artifact has unexpected fields: {sorted(extra)}")

    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")

    field_principles = artifact["field_principles"]
    if field_principles != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match the required annotations")

    methods_mapped = artifact["methods_mapped"]
    if not isinstance(methods_mapped, list) or len(methods_mapped) < 3:
        raise ValueError("methods_mapped must contain at least three methods")

    seen: set[str] = set()
    for method in methods_mapped:
        if not isinstance(method, dict) or set(method) != REQUIRED_METHOD_FIELDS:
            raise ValueError(
                "each method must contain exactly name, arxiv_id_or_url, url, "
                "carnot_stack_mapping, implication, failure_mode, and experiment_mapping"
            )
        source = method["arxiv_id_or_url"]
        if source not in VERIFIED_SOURCE_URLS:
            raise ValueError(f"method arxiv_id_or_url must be a verified source: {source}")
        if source in seen:
            raise ValueError(f"duplicate source: {source}")
        seen.add(source)
        expected_url = VERIFIED_SOURCE_URLS[source]
        if method["url"] != expected_url:
            raise ValueError(f"method url must be {expected_url!r}")
        for field in REQUIRED_METHOD_FIELDS - {"arxiv_id_or_url", "url"}:
            value = method[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"method {field} must be a non-empty string")

    if "2510.20607" not in seen:
        raise ValueError("methods_mapped must include the CEM operator flag source")

    _validate_cem_operator_authorization_flag(
        artifact["cem_operator_authorization_flag"]
    )

    flagged = artifact["flagged_for_v389"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v389 must be a non-empty string")
    if "cem" in flagged.lower():
        raise ValueError("flagged_for_v389 must not auto-select CEM")


def _validate_cem_operator_authorization_flag(flag: Any) -> None:
    if not isinstance(flag, dict) or set(flag) != REQUIRED_CEM_FLAG_FIELDS:
        raise ValueError("CEM flag must contain exactly the required fields")

    if flag["source_id"] != "2510.20607":
        raise ValueError("CEM source must be 2510.20607")
    if flag["url"] != "https://arxiv.org/abs/2510.20607":
        raise ValueError("CEM flag url must be the verified arXiv URL")
    if flag["operator_authorization_required"] is not True:
        raise ValueError("CEM operator authorization must be required")
    if flag["auto_activation_recommended"] is not False:
        raise ValueError("CEM auto-activation must be false")
    if flag["retirement_marker"] != RETIREMENT_MARKER:
        raise ValueError("CEM retirement marker must match the exclusion manifest")

    for field in REQUIRED_CEM_FLAG_FIELDS - {
        "operator_authorization_required",
        "auto_activation_recommended",
    }:
        value = flag[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"CEM flag {field} must be a non-empty string")


def validate_markdown_note(markdown: str) -> None:
    """Check that the paired note maps verified sources to required axes."""

    required_phrases = (
        "Fresh-pass provenance",
        "SOTA -> experiment mapping",
        "Reward-Guided Stitching DiffusionGemma scale-up",
        "S^3 verifier-guided denoising search",
        "Self-Rewarding SMC particle guidance",
        "OpenReview cve4NOiyVp judge-cost tuning",
        "When To Solve/Verify compute-normalized verifier bar",
        "ThinkPRM process-verifier comparator",
        "CEM operator authorization flag",
        "Carnot stack mapping",
        "Implication",
        "Failure mode",
        "Experiment mapping",
        "Flagged for .389",
    )
    missing_phrases = [phrase for phrase in required_phrases if phrase not in markdown]
    if missing_phrases:
        raise ValueError(f"markdown note missing required sections: {missing_phrases}")

    missing_sources = [
        source for source in NOTE_REQUIRED_SOURCE_CITATIONS if source not in markdown
    ]
    if missing_sources:
        raise ValueError(
            f"markdown note missing verified source citations: {missing_sources}"
        )
    if "auto-activation" not in markdown:
        raise ValueError("markdown note must reject CEM auto-activation")


def write_outputs(
    *,
    note_path: Path,
    artifact_path: Path,
    studying_path: Path,
) -> dict[str, object]:
    """Write the note, JSON artifact, and idempotent studying update."""

    artifact = build_artifact(
        methods_mapped=DEFAULT_METHODS_MAPPED,
        cem_operator_authorization_flag=DEFAULT_CEM_OPERATOR_AUTHORIZATION_FLAG,
        flagged_for_v389=DEFAULT_FLAGGED_FOR_V389,
    )
    validate_markdown_note(NOTE_MARKDOWN)

    note_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(NOTE_MARKDOWN + "\n", encoding="utf-8")
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    studying_path.write_text(
        _with_studying_section(studying_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    return artifact


def _replace_or_insert_section(
    existing: str,
    *,
    marker: str,
    section: str,
) -> str:
    if marker not in existing:
        if existing.startswith("## "):
            return section + "\n" + existing
        if "\n## " not in existing:
            return existing.rstrip() + "\n\n" + section
        return existing.replace("\n## ", "\n" + section + "\n## ", 1)

    before, after_marker = existing.split(marker, 1)
    next_section = after_marker.find("\n## ")
    if next_section == -1:
        return before + section.rstrip() + "\n"
    return before + section + after_marker[next_section + 1 :]


def _with_studying_section(existing: str) -> str:
    return _replace_or_insert_section(
        existing,
        marker=STUDYING_MARKER,
        section=STUDYING_SECTION,
    )


def main() -> int:
    """Write the default Exp 4192 deliverables under the repository root."""

    repo_root = Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        note_path=repo_root
        / "docs/research-notes/sota-ingestion-efficiency-gap4-diffusion-v389-2026-06-14.md",
        artifact_path=repo_root
        / "results/experiment_4192_sota_ingestion_efficiency_gap4_diffusion.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
