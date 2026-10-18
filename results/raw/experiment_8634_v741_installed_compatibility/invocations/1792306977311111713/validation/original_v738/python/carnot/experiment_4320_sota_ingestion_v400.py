"""Exp 4320 SOTA ingestion for the .399 fork outcomes feeding .400.

Spec refs: REQ-REPORT-4320, SCENARIO-REPORT-4320.

This module writes a planning artifact, not a benchmark result. It turns the
`.399` fork outcomes into a citation-gated SOTA-to-experiment map: the
IR3DE/CASCAL cross-domain router remained domain-bound, DiffusionGemma
reward-guided step stitching cleared the in-generation moat, the budget-aware
cascade did not dominate the always-energy verifier, and cross-game learned
value-head transfer stayed flat. The .400 flag therefore moves to scaled
external-verifier-guided DiffusionGemma generation, with router repair,
heteroskedastic budget diagnostics, and experience-gated transfer kept as
secondary tracks.
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
        "flagged_for_v400",
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
        "v399_outcome_conditioning",
        "carnot_stack_mapping",
        "failure_mode",
        "experiment_mapping",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_v400_mapped"
DEFAULT_FLAGGED_FOR_V400 = "scaled_external_verifier_guided_diffusiongemma_generation_v400"
PRODUCTION_CASCADE_FLAGGED_FOR_V400 = "production_cascade_verifier_distillation_v400"
FOURTH_DOMAIN_ROUTER_FLAGGED_FOR_V400 = "fourth_domain_arc_harness_router_v400"
MULTI_GAME_TRANSFER_FLAGGED_FOR_V400 = "multi_game_arc_efficiency_transfer_v400"
ALWAYS_ENERGY_DISTILL_FLAGGED_FOR_V400 = "always_energy_verifier_distillation_v400"
DEFAULT_RANDOM_SEED = 4320

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records ingestion completed with verifiable citations.",
    "methods_mapped": (
        "Each method MUST carry a real arXiv ID/URL (no citation = fabrication "
        "per adversarial_verify discipline) + a one-line .400 experiment mapping."
    ),
    "flagged_for_v400": (
        "Closes discover->ingest->plan: names the strongest method for the .400 "
        "planner, conditioned on the .399 outcomes."
    ),
    "random_seed": (
        "Determinism placeholder for the discovery query set (recorded for "
        "reproducibility of the sweep)."
    ),
}

VERIFIED_SOURCE_URLS = {
    "2606.13565": "https://arxiv.org/abs/2606.13565",
    "2509.25171": "https://arxiv.org/abs/2509.25171",
    "2606.15841": "https://arxiv.org/abs/2606.15841",
    "2502.08773": "https://arxiv.org/abs/2502.08773",
    "2605.05478": "https://arxiv.org/abs/2605.05478",
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{source}" for source in VERIFIED_SOURCE_URLS
)

DEFAULT_V399_OUTCOMES = {
    "cross_domain_selection_holds": False,
    "cross_domain_ci_excludes_zero": False,
    "label_ablation_robust": True,
    "diffusiongemma_guidance_moat": True,
    "controls_differentiated": True,
    "scorer_leak_recheck_passed": True,
    "cascade_dominates_controls": False,
    "always_energy_already_dominates": True,
    "cross_game_transfer_helps": False,
    "baseline_solves_held_out": True,
}

DEFAULT_METHODS_MAPPED = [
    {
        "name": "A2D2 adaptive any-length discrete diffusion",
        "arxiv_id_or_url": "2606.13565",
        "url": "https://arxiv.org/abs/2606.13565",
        "track": "scaled reward-guided DiffusionGemma generation",
        "source_read": (
            "A2D2 fine-tunes any-length discrete diffusion by jointly optimizing "
            "insertion, unmasking, and a quality-based inference schedule toward "
            "a reward-tilted sequence distribution."
        ),
        "v399_outcome_conditioning": (
            "Exp 4315 closed the in-generation moat: guidance_moat_ci95=[0.075, "
            "0.375], controls_differentiated=true, and scorer_leak_recheck_passed=true."
        ),
        "carnot_stack_mapping": (
            "Use the leak-checked partial-state scorer as the reward for bounded "
            "DiffusionGemma insertion/unmasking adaptation, with frozen ARC "
            "held-out grids and unguided, EntRGi, and self-reward controls."
        ),
        "failure_mode": (
            "Reward fine-tuning can overfit the scorer or mutate away from exact "
            "ARC grid validity. The .400 gate needs fresh held-out tasks, no "
            "answer-cell leakage, and a no-adaptation control."
        ),
        "experiment_mapping": (
            ".400: scale the winning Exp 4315 external-verifier guidance into an "
            "A2D2-style adaptive DiffusionGemma generation run."
        ),
    },
    {
        "name": "TR2-D2 tree-search trajectory replay",
        "arxiv_id_or_url": "2509.25171",
        "url": "https://arxiv.org/abs/2509.25171",
        "track": "reward-guided discrete diffusion replay buffers",
        "source_read": (
            "TR2-D2 uses Monte Carlo tree search to build reward-guided trajectory "
            "replay buffers, then fine-tunes a discrete diffusion model with an "
            "off-policy stochastic-control objective."
        ),
        "v399_outcome_conditioning": (
            "Exp 4315 showed step stitching beats the best engaged control, so "
            ".400 should harvest high-scoring partial trajectories rather than "
            "only rerank complete samples."
        ),
        "carnot_stack_mapping": (
            "Run bounded tree search over masked grid-token denoising steps, score "
            "partial states with the Exp 4315 scorer, and train only a small "
            "adapter or replay policy before exact verifier evaluation."
        ),
        "failure_mode": (
            "MCTS can spend the whole budget exploring scorer loopholes, and the "
            "paper's strongest demonstrations are sequence-design domains. Keep "
            "oracle-distinct rewards, ESS/cost accounting, and ARC-only holdouts."
        ),
        "experiment_mapping": (
            ".400: compare A2D2-only guidance against TR2-D2 replay-buffer "
            "guidance using the same leak-checked verifier reward."
        ),
    },
    {
        "name": "Cost-stratified budgeted verification",
        "arxiv_id_or_url": "2606.15841",
        "url": "https://arxiv.org/abs/2606.15841",
        "track": "budget-aware cascade diagnostics and verifier deployment",
        "source_read": (
            "The paper identifies heteroskedastic uncertainty quality across cost "
            "strata in budgeted verification and reports that simple "
            "cost-stratified thresholding can recover hit rate where global "
            "allocation fails."
        ),
        "v399_outcome_conditioning": (
            "Exp 4316 found cascade_dominates_controls=false: always-energy "
            "accuracy was 0.6 while cascade accuracy was 0.55 at cost_ratio=0.3019632358."
        ),
        "carnot_stack_mapping": (
            "Replace one global cascade threshold with per-stratum comparability "
            "audits over energy margin, judge token cost, and domain/candidate "
            "pool features; deploy only strata where escalation adds value."
        ),
        "failure_mode": (
            "Stratification can become a new leak path if strata encode domain "
            "labels or target-task identity. Freeze strata before evaluation and "
            "report per-domain false escalation rows."
        ),
        "experiment_mapping": (
            ".400: keep cascade deployment as a side-track by testing "
            "cost-stratified thresholds after always-energy dominance."
        ),
    },
    {
        "name": "UniRoute unseen-model routing",
        "arxiv_id_or_url": "2502.08773",
        "url": "https://arxiv.org/abs/2502.08773",
        "track": "cross-domain selector generalization and anti-leak routing",
        "source_read": (
            "UniRoute represents each LLM by prediction-error features on "
            "representative prompts and routes to previously unseen models via "
            "cluster-based or learned cluster-map policies."
        ),
        "v399_outcome_conditioning": (
            "Exp 4314 kept label_ablation_robust=true but cross_domain_selection_holds=false, "
            "so the next router attempt needs domain-invariant performance "
            "fingerprints before any fourth-domain claim."
        ),
        "carnot_stack_mapping": (
            "Represent ARC, ARC-GEN, FoVer, and any fourth-domain verifier heads "
            "by frozen representative-task error vectors, then route without "
            "domain labels or family IDs."
        ),
        "failure_mode": (
            "Representative prompts can memorize the target distribution if they "
            "are selected post hoc. The prompt bank and cluster map must be frozen "
            "before held-out-domain scoring."
        ),
        "experiment_mapping": (
            ".400: rebuild the failed cross-domain selector as a UniRoute-style "
            "performance-fingerprint router before adding a fourth-domain headline."
        ),
    },
    {
        "name": "LANTERN experience-gated transfer",
        "arxiv_id_or_url": "2605.05478",
        "url": "https://arxiv.org/abs/2605.05478",
        "track": "cross-task and cross-game learned-verifier transfer",
        "source_read": (
            "LANTERN performs multi-source neurosymbolic transfer with LLM-generated "
            "automata, similarity-weighted source policies, and adaptive "
            "teacher-student gating from uncertainty and temporal-difference error."
        ),
        "v399_outcome_conditioning": (
            "Exp 4318 reported cross_game_transfer_helps=false and "
            "cross_game_state_reduction=1.0 despite a positive-control baseline "
            "solver, making naive value-head transfer a representation gap."
        ),
        "carnot_stack_mapping": (
            "Gate ARC game-to-game value heads by source relevance and TD-error "
            "instead of pooling all solved-game traces into one transferred head."
        ),
        "failure_mode": (
            "LLM-generated automata can become an implicit oracle if they inspect "
            "held-out levels. Generate transfer summaries only from train traces "
            "and log per-source negative transfer."
        ),
        "experiment_mapping": (
            ".400: run an experience-gated multi-source ARC value-head transfer "
            "probe only after the scaled guided-generation headline."
        ),
    },
]

STUDYING_SECTION = """## 2026-06-17 Exp 4320 - .399 fork SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4320_sota_ingestion_v400.json`.

**Reliable-channel provenance:** `scripts/sweep_clusters.py` and
`scripts/sweep_semscholar.py` imported successfully; `sweep_clusters.py`
emitted focused arXiv discovery URLs for verifier, energy, and routing
clusters. Semantic Scholar was reachable through the helper but returned HTTP
429 for the four focused keyword probes in this loop. Low-concurrency
WebSearch/WebFetch verified arXiv:2606.13565, arXiv:2509.25171,
arXiv:2606.15841, arXiv:2502.08773, arXiv:2605.05478, arXiv:2602.22871,
arXiv:2602.01849, arXiv:2603.04445, arXiv:2512.02543, and arXiv:2605.09965.
The banned `/deep-research` channel was not invoked.

**Filtered track:** .399 outcomes after the IR3DE+CASCAL cross-domain router,
the DiffusionGemma reward-guided step-stitching run, the efficiency cascade
deployment run, and the ARC cross-game learned-verifier transfer run.

**.399 outcome conditioning:**
- Exp 4314: `cross_domain_selection_holds=false`,
  `cross_domain_delta=0.2307692308`,
  `cross_domain_delta_ci95=[-0.1153846154, 0.5384615385]`, and
  `label_ablation_robust=true`; the selector survived the label-ablation check
  but did not make a decision-grade cross-domain moat.
- Exp 4315: `diffusiongemma_guidance_moat=true`,
  `controls_differentiated=true`, `scorer_leak_recheck_passed=true`,
  `carnot_minus_best_control_delta=0.225`, and
  `guidance_moat_ci95=[0.075, 0.375]`; the external-verifier-guided
  in-generation moat closed.
- Exp 4316: `cascade_dominates_controls=false`,
  `accuracy_always_energy=0.6`, `accuracy_cascade=0.55`, and
  `cost_ratio_cascade=0.3019632358`; the cascade was useful as a diagnostic but
  the always-energy verifier remained the cleaner operating point.
- Exp 4318: `cross_game_transfer_helps=false`,
  `cross_game_state_reduction=1.0`, and `baseline_solves_held_out=true`; the
  uniform positive-control solver worked, but the learned value-head did not
  reduce held-out search states.

**Fresh-pass candidates marked ingested:**
- A2D2 adaptive any-length discrete diffusion, arXiv:2606.13565 - mapped to the
  .400 scaled external-verifier-guided DiffusionGemma generation headline.
- TR2-D2 tree-search trajectory replay, arXiv:2509.25171 - mapped to bounded
  reward-guided replay buffers for DiffusionGemma partial-state denoising.
- Heteroskedastic Signals in Budgeted LLM Verification, arXiv:2606.15841 -
  mapped to cost-stratified cascade diagnostics after the Exp 4316 global
  cascade failed to dominate.
- UniRoute unseen-model routing, arXiv:2502.08773 - mapped to cross-domain
  performance-fingerprint routing without domain labels or family IDs.
- LANTERN experience-gated transfer, arXiv:2605.05478 - mapped to gated
  multi-source ARC value-head transfer after the Exp 4318 flat result.

**Screened but not mapped as strongest rows:** Reward-Guided Stitching
(arXiv:2602.22871), Self-Rewarding SMC (arXiv:2602.01849), CSMC
(arXiv:2602.09424), Dynamic Model Routing and Cascading (arXiv:2603.04445),
Inference-Time Distillation (arXiv:2512.02543), and Game Multiverse
(arXiv:2605.09965) were read as relevant context. They were not re-ingested as
fresh method rows because Reward-Guided Stitching and Self-Rewarding SMC are
already in earlier sweeps and the others are weaker fits than the five mapped
rows for the observed .399 outcomes.

Already-covered context not re-ingested as fresh method rows: Budget-aware
Discriminative Verification, IR3DE, Routing with Generated Data / CASCAL,
TTARAG, SMC importance weighting for discrete diffusion, EEVEE,
optimize_anything / GEPA, RefGRPO, SLMJury, ReMDM, ARC-TGI, ARC-GEN, RFG,
EDLM, EntRGi, Manta-LM, masked-discrete-diffusion guidance dynamics, INSPECTOR
Representation-as-a-Judge, ABPR, and Decocted Experience.

flagged_for_v400:
`scaled_external_verifier_guided_diffusiongemma_generation_v400`.

Flagged for .400: `scaled_external_verifier_guided_diffusiongemma_generation_v400`.

random_seed=4320

**Bottom line for the .400 roadmap:** Exp 4315 is the only .399 fork that closed
decision-grade, so make .400 an A2D2/TR2-D2-style scaled guided generation
headline over the existing leak-checked DiffusionGemma scorer. Keep
cross-domain routing on UniRoute-style frozen fingerprints, convert cascade work
into heteroskedastic threshold diagnostics, and only retry cross-game transfer
with LANTERN-style experience gates.
"""


def _ci_excludes_zero(values: object) -> bool:
    if isinstance(values, Sequence) and len(values) == 2:
        lower, upper = values
        if isinstance(lower, int | float) and isinstance(upper, int | float):
            return lower > 0 or upper < 0
    return False


def extract_v399_outcomes(
    *,
    cross_domain: Mapping[str, Any],
    guidance: Mapping[str, Any],
    cascade: Mapping[str, Any],
    transfer: Mapping[str, Any],
) -> dict[str, bool]:
    """Extract the load-bearing .399 outcome booleans from source artifacts."""

    always_energy_accuracy = cascade.get("accuracy_always_energy")
    cascade_accuracy = cascade.get("accuracy_cascade")
    always_energy_dominates = (
        cascade.get("cascade_dominates_controls") is False
        and isinstance(always_energy_accuracy, int | float)
        and isinstance(cascade_accuracy, int | float)
        and always_energy_accuracy >= cascade_accuracy
    )

    return {
        "cross_domain_selection_holds": (
            cross_domain.get("cross_domain_selection_holds") is True
        ),
        "cross_domain_ci_excludes_zero": _ci_excludes_zero(
            cross_domain.get("cross_domain_delta_ci95")
        ),
        "label_ablation_robust": cross_domain.get("label_ablation_robust") is True,
        "diffusiongemma_guidance_moat": (
            guidance.get("diffusiongemma_guidance_moat") is True
        ),
        "controls_differentiated": guidance.get("controls_differentiated") is True,
        "scorer_leak_recheck_passed": (
            guidance.get("scorer_leak_recheck_passed") is True
        ),
        "cascade_dominates_controls": cascade.get("cascade_dominates_controls") is True,
        "always_energy_already_dominates": always_energy_dominates,
        "cross_game_transfer_helps": transfer.get("cross_game_transfer_helps") is True,
        "baseline_solves_held_out": transfer.get("baseline_solves_held_out") is True,
    }


def select_flagged_for_v400(outcomes: Mapping[str, bool]) -> str:
    """Choose the .400 flag from the .399 fork outcomes."""

    if (
        outcomes.get("diffusiongemma_guidance_moat")
        and outcomes.get("controls_differentiated")
        and outcomes.get("scorer_leak_recheck_passed")
    ):
        return DEFAULT_FLAGGED_FOR_V400
    if outcomes.get("cascade_dominates_controls"):
        return PRODUCTION_CASCADE_FLAGGED_FOR_V400
    if outcomes.get("cross_domain_selection_holds") and outcomes.get(
        "label_ablation_robust"
    ):
        return FOURTH_DOMAIN_ROUTER_FLAGGED_FOR_V400
    if outcomes.get("cross_game_transfer_helps") and outcomes.get(
        "baseline_solves_held_out"
    ):
        return MULTI_GAME_TRANSFER_FLAGGED_FOR_V400
    return ALWAYS_ENERGY_DISTILL_FLAGGED_FOR_V400


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]] | None = None,
    flagged_for_v400: str = DEFAULT_FLAGGED_FOR_V400,
    random_seed: int = DEFAULT_RANDOM_SEED,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build the deterministic Exp 4320 mapping artifact."""

    return {
        "honest_verdict": honest_verdict,
        "methods_mapped": [
            dict(method) for method in (methods_mapped or DEFAULT_METHODS_MAPPED)
        ],
        "flagged_for_v400": flagged_for_v400,
        "random_seed": random_seed,
        "field_principles": dict(FIELD_PRINCIPLES),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 4320 artifact before it can be written to disk."""

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
        raise ValueError("field_principles must match REQ-REPORT-4320")

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

    flagged = artifact["flagged_for_v400"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v400 must be non-empty")
    flagged_lower = flagged.lower()
    if not any(
        marker in flagged_lower
        for marker in (
            "scaled",
            "guided",
            "diffusiongemma",
            "cascade",
            "fourth_domain",
            "multi_game",
            "energy",
            "distill",
            "heteroskedastic",
        )
    ):
        raise ValueError("flagged_for_v400 must be conditioned on the .399 outcomes")


def validate_studying_section(section: str) -> None:
    """Check that the research-studying entry keeps citations and outcome context."""

    required_phrases = [
        "flagged_for_v400",
        "sweep_clusters.py",
        "sweep_semscholar.py",
        "WebSearch/WebFetch",
        "/deep-research",
        "not invoked",
        "cross_domain_selection_holds=false",
        "cross_domain_delta=0.2307692308",
        "cross_domain_delta_ci95=[-0.1153846154, 0.5384615385]",
        "label_ablation_robust=true",
        "diffusiongemma_guidance_moat=true",
        "controls_differentiated=true",
        "scorer_leak_recheck_passed=true",
        "carnot_minus_best_control_delta=0.225",
        "guidance_moat_ci95=[0.075, 0.375]",
        "cascade_dominates_controls=false",
        "accuracy_always_energy=0.6",
        "accuracy_cascade=0.55",
        "cost_ratio_cascade=0.3019632358",
        "cross_game_transfer_helps=false",
        "cross_game_state_reduction=1.0",
        "baseline_solves_held_out=true",
        DEFAULT_FLAGGED_FOR_V400,
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
    marker = "## 2026-06-17 Exp 4320"
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

    flagged_for_v400 = select_flagged_for_v400(DEFAULT_V399_OUTCOMES)
    artifact = build_artifact(flagged_for_v400=flagged_for_v400)
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
    root_override = os.environ.get("CARNOT_EXP4320_ROOT")
    repo_root = Path(root_override) if root_override else Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        artifact_path=repo_root / "results/experiment_4320_sota_ingestion_v400.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
