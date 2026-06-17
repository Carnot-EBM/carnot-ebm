"""Exp 4309 SOTA ingestion for the .398 fork outcomes feeding .399.

Spec refs: REQ-REPORT-4309, SCENARIO-REPORT-4309.

This module writes a planning artifact, not a benchmark result. It turns the
`.398` fork outcomes into a citation-gated SOTA-to-experiment map: the
energy verifier beat the strongest local judge at far lower cost, the
DiffusionGemma guidance arm did not clear its engaged-control CI gate, and the
cross-domain selector did not generalize decisively. The .399 flag therefore
moves to a budget-aware cascade-router deployment headline, with guided
generation and cross-domain router repair kept as secondary tracks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any


REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "honest_verdict",
        "methods_mapped",
        "flagged_for_v399",
        "random_seed",
        "field_principles",
    }
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "name",
        "arxiv_id_or_url",
        "url",
        "track",
        "source_read",
        "v398_outcome_conditioning",
        "carnot_stack_mapping",
        "failure_mode",
        "experiment_mapping",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_v399_mapped"
DEFAULT_FLAGGED_FOR_V399 = "budget_aware_discriminative_cascade_router_v399"
SMC_GUIDED_GENERATION_FLAGGED_FOR_V399 = "smc_guided_diffusiongemma_generation_v399"
FOURTH_DOMAIN_ROUTER_FLAGGED_FOR_V399 = "fourth_domain_router_generalization_v399"
ROUTER_REBUILD_FLAGGED_FOR_V399 = "domain_invariant_router_rebuild_v399"
DEFAULT_RANDOM_SEED = 4309

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records ingestion completed with verifiable citations.",
    "methods_mapped": (
        "Each method MUST carry a real arXiv ID/URL (no citation = fabrication "
        "per adversarial_verify discipline) + a one-line .399 experiment mapping."
    ),
    "flagged_for_v399": (
        "Closes discover->ingest->plan: names the strongest method for the .399 "
        "planner, conditioned on the .398 outcomes."
    ),
    "random_seed": (
        "Determinism placeholder for the discovery query set (recorded for "
        "reproducibility of the sweep)."
    ),
}

VERIFIED_SOURCE_URLS = {
    "2510.14913": "https://arxiv.org/abs/2510.14913",
    "2606.06098": "https://arxiv.org/abs/2606.06098",
    "2601.09692": "https://arxiv.org/abs/2601.09692",
    "2601.11443": "https://arxiv.org/abs/2601.11443",
    "2505.22524": "https://arxiv.org/abs/2505.22524",
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{source}" for source in VERIFIED_SOURCE_URLS
)

DEFAULT_V398_OUTCOMES = {
    "efficiency_pareto_holds": True,
    "diffusiongemma_guidance_moat": False,
    "controls_differentiated": True,
    "scorer_leak_recheck_passed": True,
    "cross_domain_selection_holds": False,
    "cross_domain_ci_excludes_zero": False,
    "label_ablation_robust": True,
}

DEFAULT_METHODS_MAPPED = [
    {
        "name": "Budget-aware discriminative verification",
        "arxiv_id_or_url": "2510.14913",
        "url": "https://arxiv.org/abs/2510.14913",
        "track": "efficiency-parity and cascade routing",
        "source_read": (
            "The paper shows that hybrid discriminative verification with "
            "self-consistency can outperform generative verification under fixed "
            "inference budgets, reporting up to a 15.3 percent AIME2025 gain."
        ),
        "v398_outcome_conditioning": (
            "Exp 4303 hardened the Pareto result: the energy verifier reached "
            "0.8 accuracy versus the best judge at 0.5 with cost_ratio=1.03e-08."
        ),
        "carnot_stack_mapping": (
            "Promote the Set-Encoder energy verifier into a budget-aware cascade "
            "policy that first runs cheap discriminative scoring, then escalates "
            "only low-margin cases to a judge or specialist router."
        ),
        "failure_mode": (
            "A cascade can hide regressions if escalation thresholds are tuned on "
            "the held-out tasks. The .399 gate needs frozen thresholds, iso-FLOPs "
            "curves, and per-domain failure rows."
        ),
        "experiment_mapping": (
            ".399: deploy a budget-aware discriminative cascade-router and compare "
            "accuracy/cost against always-judge and always-energy controls."
        ),
    },
    {
        "name": "IR3DE linear domain-expert router",
        "arxiv_id_or_url": "2606.06098",
        "url": "https://arxiv.org/abs/2606.06098",
        "track": "cross-domain router rebuild",
        "source_read": (
            "IR3DE uses ridge regression as a cheap router for domain experts, "
            "matching baselines in language-modeling settings and reaching 98.4 "
            "percent normalized performance in a reasoning setting."
        ),
        "v398_outcome_conditioning": (
            "Exp 4305 collapsed on broad cross-domain selection, so .399 should "
            "test a simpler domain-expert router before adding more domains."
        ),
        "carnot_stack_mapping": (
            "Replace the nearest-centroid router with a ridge-regression router "
            "over normalized selector features, trained leave-one-domain-out and "
            "evaluated with domain labels ablated from inference inputs."
        ),
        "failure_mode": (
            "Linear routing can still learn domain proxies from feature scale or "
            "candidate-count artifacts. It needs feature standardization, label "
            "ablation, and held-out-domain calibration."
        ),
        "experiment_mapping": (
            ".399: rebuild the failed cross-domain selector as an IR3DE-style "
            "linear router over ARC, ARC-GEN, and FoVer candidates."
        ),
    },
    {
        "name": "Routing with Generated Data / CASCAL",
        "arxiv_id_or_url": "2601.09692",
        "url": "https://arxiv.org/abs/2601.09692",
        "track": "cold-start router data and anti-leak",
        "source_read": (
            "Routing with Generated Data trains LLM routers from generated "
            "queries and answers; its CASCAL query-only router uses consensus "
            "voting and hierarchical clustering to estimate model skill niches."
        ),
        "v398_outcome_conditioning": (
            "Exp 4305's label ablation survived but the cross-domain CI included "
            "zero, pointing to insufficient domain-invariant router data rather "
            "than an obvious label leak."
        ),
        "carnot_stack_mapping": (
            "Generate train-only synthetic selector tasks from domain descriptions, "
            "filter generators whose outputs lack performance differentiation, "
            "and train query-only selector routers without target labels."
        ),
        "failure_mode": (
            "Generated tasks can match the generator rather than the target domain. "
            "The generator-quality filter and a frozen real held-out split are "
            "load-bearing."
        ),
        "experiment_mapping": (
            ".399: add a generated-data CASCAL router pretraining arm before "
            "retesting held-out-domain selector generalization."
        ),
    },
    {
        "name": "TTARAG retrieval-prediction adaptation",
        "arxiv_id_or_url": "2601.11443",
        "url": "https://arxiv.org/abs/2601.11443",
        "track": "powered retrieval-augmented selector adaptation",
        "source_read": (
            "TTARAG adapts retrieval-augmented generation at test time by making "
            "the model predict retrieved content, improving specialized-domain "
            "RAG under distribution shift across six domains."
        ),
        "v398_outcome_conditioning": (
            "Exp 4305 showed a positive FoVer held-out delta but an underpowered "
            "CI, so .399 needs adaptation that uses retrieved selector evidence "
            "without treating domain identity as the signal."
        ),
        "carnot_stack_mapping": (
            "Adapt only the selector context or a small retrieval adapter on "
            "train-side traces, then freeze it for held-out domains; no base "
            "model weight mutation is needed for the headline path."
        ),
        "failure_mode": (
            "Unbounded test-time parameter updates would violate Carnot's "
            "traceability constraints and can leak target outcomes. Keep the "
            "adaptation bounded and replayable."
        ),
        "experiment_mapping": (
            ".399: run retrieval-augmented selector adaptation with frozen target "
            "domains and compare against static retrieval-only context."
        ),
    },
    {
        "name": "SMC importance weighting for discrete diffusion",
        "arxiv_id_or_url": "2505.22524",
        "url": "https://arxiv.org/abs/2505.22524",
        "track": "reward-guided discrete diffusion controls",
        "source_read": (
            "The SMC discrete-diffusion paper derives tractable importance weights "
            "and practical optimal-proposal approximations for inference-time "
            "control across language modeling, biology, and text-to-image tasks."
        ),
        "v398_outcome_conditioning": (
            "Exp 4304 engaged EntRGi controls and kept the scorer leak-free, but "
            "the Carnot-minus-best-control CI still included zero."
        ),
        "carnot_stack_mapping": (
            "Wrap DiffusionGemma candidates in a particle/reweighting controller "
            "using the partial-state scorer as a reward tilt, with unguided, "
            "EntRGi, and SMC proposal controls all kept engaged."
        ),
        "failure_mode": (
            "Particle control can spend more compute without improving exact ARC "
            "grids, and a reward tilt can collapse diversity. Report cost, "
            "effective sample size, and no-op-control checks."
        ),
        "experiment_mapping": (
            ".399: keep SMC-guided DiffusionGemma as a secondary repair track "
            "after the cascade-router headline."
        ),
    },
]

STUDYING_SECTION = """## 2026-06-17 Exp 4309 - .398 fork SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4309_sota_ingestion_v399.json`.

**Reliable-channel provenance:** `scripts/sweep_clusters.py` and
`scripts/sweep_semscholar.py` imported successfully; `sweep_clusters.py`
emitted focused arXiv discovery URLs for verifier/energy clusters. Semantic
Scholar returned arXiv IDs for budget-aware discriminative verification,
domain routing, and discrete diffusion probes, and returned HTTP 429 for one
RAG adaptation probe and one rerouting-security probe. WebSearch/WebFetch
verified arXiv:2510.14913, arXiv:2606.06098, arXiv:2601.09692,
arXiv:2601.11443, arXiv:2505.22524, arXiv:2601.21380, arXiv:2602.09424, and
arXiv:2605.05007. The banned `/deep-research` channel was not invoked.

**Filtered track:** .398 outcomes after the hardened iso-FLOPs verifier-vs-judge
run, the DiffusionGemma engaged-control guidance run, and the cross-domain
selector generalization stress.

**.398 outcome conditioning:**
- Exp 4303: `efficiency_pareto_holds=true`,
  `accuracy_energy_verifier=0.8`, `accuracy_best_judge=0.5`,
  `accuracy_delta_ci95=[0.1, 0.5]`, and `cost_ratio=1.03e-08`; the efficiency
  axis hardened into a decision-grade Pareto win.
- Exp 4304: `diffusiongemma_guidance_moat=false`,
  `controls_differentiated=true`, `scorer_leak_recheck_passed=true`,
  `carnot_minus_best_control_delta=0.133334`, and
  `guidance_moat_ci95=[-0.066667, 0.366667]`; guided generation improved the
  point estimate but did not clear the engaged-control CI gate.
- Exp 4305: `cross_domain_selection_holds=false`,
  `cross_domain_delta=0.2307692308`,
  `cross_domain_ci95=[-0.1153846154, 0.5384615385]`, and
  `label_ablation_robust=true`; the FoVer slice was positive but underpowered,
  while ARC and ARC-GEN held-out reads collapsed.

**Fresh-pass candidates marked ingested:**
- Budget-aware discriminative verification, arXiv:2510.14913 - mapped to the
  .399 deployment/cascade-router headline after Exp 4303 hardened efficiency.
- IR3DE linear domain-expert router, arXiv:2606.06098 - mapped to a simpler
  domain-invariant router rebuild after Exp 4305's broad cross-domain collapse.
- Routing with Generated Data / CASCAL, arXiv:2601.09692 - mapped to
  generated-data router pretraining with query-only anti-leak controls.
- TTARAG retrieval-prediction adaptation, arXiv:2601.11443 - mapped to
  powered retrieval-augmented selector adaptation on train-side traces only.
- SMC importance weighting for discrete diffusion, arXiv:2505.22524 - mapped
  to a secondary DiffusionGemma repair track with engaged particle/reweighting
  controls.

**Screened but not mapped as strongest rows:** RerouteGuard (arXiv:2601.21380),
CSMC clean-sample Markov chains (arXiv:2602.09424), Uno-Orchestra
(arXiv:2605.05007), and TRouter (arXiv:2604.09377) were read as adjacent
routing-security, clean-sample diffusion, selective-delegation, and cold-start
routing evidence. They remain weaker for `.399` than the mapped rows because
RerouteGuard is attack-specific, CSMC is molecule/biology-centered,
Uno-Orchestra is a broader multi-agent policy, and TRouter overlaps the more
direct IR3DE/CASCAL router rebuild path.

Already-covered context not re-ingested as fresh method rows: EEVEE,
optimize_anything / GEPA, RefGRPO, SLMJury, ReMDM, ARC-TGI, ARC-GEN, RFG,
EDLM, EntRGi, Self-Improving LLM Agents at Test-Time, SEVerA, DPRM,
Reward-Guided Stitching, Manta-LM, masked-discrete-diffusion guidance dynamics,
INSPECTOR Representation-as-a-Judge, ABPR, Decocted Experience, and COVER.

flagged_for_v399:
`budget_aware_discriminative_cascade_router_v399`.

Flagged for .399: `budget_aware_discriminative_cascade_router_v399`.

random_seed=4309

**Bottom line for the .399 roadmap:** the efficiency axis is the only .398 fork
that hardened cleanly, so make the next headline a budget-aware discriminative
cascade-router. Rebuild the cross-domain router with IR3DE/CASCAL-style
domain-invariant training before claiming broader transfer, and keep
keep SMC-guided DiffusionGemma as the secondary repair track rather than the
.399 headline.
"""


def extract_v398_outcomes(
    *,
    efficiency: Mapping[str, Any],
    guidance: Mapping[str, Any],
    cross_domain: Mapping[str, Any],
) -> dict[str, bool]:
    """Extract the load-bearing .398 outcome booleans from source artifacts."""

    ci95 = cross_domain.get("cross_domain_ci95")
    ci_excludes_zero = False
    if isinstance(ci95, Sequence) and len(ci95) == 2:
        lower, upper = ci95
        if isinstance(lower, int | float) and isinstance(upper, int | float):
            ci_excludes_zero = lower > 0 or upper < 0

    return {
        "efficiency_pareto_holds": efficiency.get("efficiency_pareto_holds") is True,
        "diffusiongemma_guidance_moat": (
            guidance.get("diffusiongemma_guidance_moat") is True
        ),
        "controls_differentiated": guidance.get("controls_differentiated") is True,
        "scorer_leak_recheck_passed": (
            guidance.get("scorer_leak_recheck_passed") is True
        ),
        "cross_domain_selection_holds": (
            cross_domain.get("cross_domain_selection_holds") is True
        ),
        "cross_domain_ci_excludes_zero": ci_excludes_zero,
        "label_ablation_robust": cross_domain.get("label_ablation_robust") is True,
    }


def select_flagged_for_v399(outcomes: Mapping[str, bool]) -> str:
    """Choose the .399 flag from the .398 fork outcomes."""

    if outcomes.get("efficiency_pareto_holds"):
        return DEFAULT_FLAGGED_FOR_V399
    if (
        outcomes.get("diffusiongemma_guidance_moat")
        and outcomes.get("controls_differentiated")
        and outcomes.get("scorer_leak_recheck_passed")
    ):
        return SMC_GUIDED_GENERATION_FLAGGED_FOR_V399
    if outcomes.get("cross_domain_selection_holds") and outcomes.get(
        "label_ablation_robust"
    ):
        return FOURTH_DOMAIN_ROUTER_FLAGGED_FOR_V399
    return ROUTER_REBUILD_FLAGGED_FOR_V399


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]] | None = None,
    flagged_for_v399: str = DEFAULT_FLAGGED_FOR_V399,
    random_seed: int = DEFAULT_RANDOM_SEED,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build the deterministic Exp 4309 mapping artifact."""

    return {
        "honest_verdict": honest_verdict,
        "methods_mapped": [
            dict(method) for method in (methods_mapped or DEFAULT_METHODS_MAPPED)
        ],
        "flagged_for_v399": flagged_for_v399,
        "random_seed": random_seed,
        "field_principles": dict(FIELD_PRINCIPLES),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 4309 artifact before it can be written to disk."""

    fields = set(artifact)
    missing = REQUIRED_ARTIFACT_FIELDS - fields
    extra = fields - REQUIRED_ARTIFACT_FIELDS
    if missing:
        raise ValueError(f"artifact missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"artifact has unexpected fields: {sorted(extra)}")

    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")

    if artifact["field_principles"] != FIELD_PRINCIPLES:
        raise ValueError("field_principles must match REQ-REPORT-4309")

    random_seed = artifact["random_seed"]
    if not isinstance(random_seed, int):
        raise ValueError("random_seed must be an integer")

    methods = artifact["methods_mapped"]
    if not isinstance(methods, list) or not 3 <= len(methods) <= 5:
        raise ValueError("methods_mapped must contain three to five methods")

    seen_sources: set[str] = set()
    for method in methods:
        if not isinstance(method, Mapping) or set(method) != REQUIRED_METHOD_FIELDS:
            raise ValueError("each method must be a dict with exactly the required fields")

        for key, value in method.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"method field {key!r} must be a non-empty string")

        source = method["arxiv_id_or_url"]
        if source not in VERIFIED_SOURCE_URLS:
            raise ValueError(f"method source {source!r} is not a verified source")
        if method["url"] != VERIFIED_SOURCE_URLS[source]:
            raise ValueError(f"method url for {source!r} must match the verified url")
        if source in seen_sources:
            raise ValueError(f"duplicate source in methods_mapped: {source}")
        seen_sources.add(source)

    flagged = artifact["flagged_for_v399"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v399 must be non-empty")
    flagged_lower = flagged.lower()
    if not any(
        marker in flagged_lower
        for marker in (
            "budget_aware",
            "cascade_router",
            "guided_generation",
            "fourth_domain",
            "router_rebuild",
            "retrieval",
            "smc",
        )
    ):
        raise ValueError("flagged_for_v399 must be conditioned on the .398 outcomes")


def validate_studying_section(section: str) -> None:
    """Check that the research-studying entry keeps citations and outcome context."""

    required_phrases = [
        "flagged_for_v399",
        "sweep_clusters.py",
        "sweep_semscholar.py",
        "WebSearch/WebFetch",
        "/deep-research",
        "not invoked",
        "efficiency_pareto_holds=true",
        "accuracy_energy_verifier=0.8",
        "accuracy_best_judge=0.5",
        "cost_ratio=1.03e-08",
        "diffusiongemma_guidance_moat=false",
        "controls_differentiated=true",
        "scorer_leak_recheck_passed=true",
        "cross_domain_selection_holds=false",
        "cross_domain_delta=0.2307692308",
        "cross_domain_ci95=[-0.1153846154, 0.5384615385]",
        DEFAULT_FLAGGED_FOR_V399,
        f"random_seed={DEFAULT_RANDOM_SEED}",
    ]
    for phrase in required_phrases:
        if phrase not in section:
            raise ValueError(f"studying section missing required phrase: {phrase}")

    missing_sources = sorted(
        source for source in NOTE_REQUIRED_SOURCE_CITATIONS if source not in section
    )
    if missing_sources:
        raise ValueError(f"studying section missing verified source citations: {missing_sources}")


def _with_studying_section(existing: str) -> str:
    marker = "## 2026-06-17 Exp 4309"
    next_marker = "\n## "
    section = STUDYING_SECTION.rstrip() + "\n"

    if marker in existing:
        start = existing.index(marker)
        next_start = existing.find(next_marker, start + 1)
        if next_start == -1:
            return existing[:start] + section
        return existing[:start] + section + existing[next_start:]

    if existing.startswith("## "):
        return section + "\n" + existing

    first_section = existing.find(next_marker)
    if first_section == -1:
        return existing.rstrip() + "\n\n" + section
    return existing[: first_section + 1] + section + "\n" + existing[first_section + 1 :]


def write_outputs(
    *,
    artifact_path: Path,
    studying_path: Path,
) -> dict[str, object]:
    """Write the JSON artifact and idempotent research-studying entry."""

    flagged_for_v399 = select_flagged_for_v399(DEFAULT_V398_OUTCOMES)
    artifact = build_artifact(flagged_for_v399=flagged_for_v399)
    validate_artifact(artifact)
    validate_studying_section(STUDYING_SECTION)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    existing_studying = (
        studying_path.read_text(encoding="utf-8") if studying_path.exists() else ""
    )
    studying_path.write_text(_with_studying_section(existing_studying), encoding="utf-8")

    return artifact


def main() -> int:
    root_override = os.environ.get("CARNOT_EXP4309_ROOT")
    repo_root = Path(root_override) if root_override else Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        artifact_path=repo_root / "results/experiment_4309_sota_ingestion_v399.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
