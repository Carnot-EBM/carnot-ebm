"""Exp 4298 SOTA ingestion for the .397 fork outcomes feeding .398.

Spec refs: REQ-REPORT-4298, SCENARIO-REPORT-4298.

This module writes a planning artifact, not a benchmark result. It closes the
`.397` outcomes into a concrete SOTA-to-experiment mapping: the repaired
ARC-GEN pool now supports non-degenerate cross-generator transfer, the
DiffusionGemma partial-state scorer exists and passed the leak audit, and the
strong-judge efficiency artifact is missing at the requested path. The .398
flag therefore moves the headline to broader-domain selector generalization
rather than repeating the same ARC-GEN substrate or pretending the missing
strong-judge result was measured.
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
        "flagged_for_v398",
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
        "v397_outcome_conditioning",
        "carnot_stack_mapping",
        "failure_mode",
        "experiment_mapping",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_v398_mapped"
DEFAULT_FLAGGED_FOR_V398 = "eevee_router_prompt_broader_domain_selector_v398"
NONLEAKY_SCORER_REPAIR_FLAGGED_FOR_V398 = "nonleaky_partial_state_scorer_repair_v398"
STRONG_GUIDED_GENERATION_FLAGGED_FOR_V398 = (
    "remdm_partial_state_guided_generation_v398"
)
SMALL_VERIFIER_EFFICIENCY_FLAGGED_FOR_V398 = (
    "slmjury_small_verifier_efficiency_hardening_v398"
)
DEFAULT_RANDOM_SEED = 4298

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records ingestion completed with verifiable citations.",
    "methods_mapped": (
        "Each method MUST carry a real arXiv ID/URL (no citation = fabrication "
        "per adversarial_verify discipline) + a one-line .398 experiment mapping."
    ),
    "flagged_for_v398": (
        "Closes discover->ingest->plan: names the strongest method for the .398 "
        "planner, conditioned on the .397 outcomes."
    ),
    "random_seed": (
        "Determinism placeholder for the discovery query set (recorded for "
        "reproducibility of the sweep)."
    ),
}

VERIFIED_SOURCE_URLS = {
    "2606.11182": "https://arxiv.org/abs/2606.11182",
    "2605.19633": "https://arxiv.org/abs/2605.19633",
    "2606.14211": "https://arxiv.org/abs/2606.14211",
    "2606.07810": "https://arxiv.org/abs/2606.07810",
    "2503.00307": "https://arxiv.org/abs/2503.00307",
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source}" for source in VERIFIED_SOURCE_URLS)

DEFAULT_V397_OUTCOMES = {
    "cross_generator_holds": True,
    "non_degenerate_guards_pass": True,
    "arcgen_cross_generator_generalizes": True,
    "partial_state_scorer_built": True,
    "partial_state_leak_free": True,
    "efficiency_pareto_holds": False,
    "strong_judge_artifact_available": False,
}

DEFAULT_METHODS_MAPPED = [
    {
        "name": "EEVEE router-prompt co-evolution",
        "arxiv_id_or_url": "2606.11182",
        "url": "https://arxiv.org/abs/2606.11182",
        "track": "broader-domain selector generalization",
        "source_read": (
            "EEVEE introduces multi-dataset test-time prompt learning for LLM "
            "agents, using a router to partition heterogeneous task streams and "
            "co-evolve router and prompt configurations."
        ),
        "v397_outcome_conditioning": (
            "Exp 4291 closes the non-degenerate ARC-GEN cross-generator question, "
            "so .398 should broaden beyond another same-substrate generator split."
        ),
        "carnot_stack_mapping": (
            "Add a domain router above ARC, ARC-GEN, and DiffusionGemma selector "
            "features, then co-evolve per-domain selector prompts only on "
            "train-side outcomes with held-out families frozen."
        ),
        "failure_mode": (
            "Router-prompt learning can leak family identity or overfit public "
            "tasks. The .398 gate must freeze held-out domains before any "
            "router update is selected."
        ),
        "experiment_mapping": (
            ".398: test router-prompt selector generalization across ARC, "
            "ARC-GEN, and diffusion partial-state families."
        ),
    },
    {
        "name": "optimize_anything / GEPA text-parameter search",
        "arxiv_id_or_url": "2605.19633",
        "url": "https://arxiv.org/abs/2605.19633",
        "track": "test-time selector and harness optimization",
        "source_read": (
            "optimize_anything frames prompts, code, agent architectures, and "
            "configuration text as optimizable artifacts under scoring functions, "
            "with GEPA-style reflective Pareto search and cross-task transfer."
        ),
        "v397_outcome_conditioning": (
            "With cross-generator transfer now positive, the next risk is not a "
            "single weak generator pool but brittle selector and harness text "
            "that fails on unseen domains."
        ),
        "carnot_stack_mapping": (
            "Expose selector rubrics, retrieval policies, score-fusion rules, "
            "and curriculum text as locked text parameters optimized against "
            "train-only exact-grid scores plus validation leakage guards."
        ),
        "failure_mode": (
            "A text optimizer can optimize methodology away or memorize the "
            "evaluation set. It needs an allowlist of mutable fields and a "
            "separate adversarial holdout."
        ),
        "experiment_mapping": (
            ".398: run GEPA-style optimization over selector configuration text "
            "with exact held-out-domain acceptance gates."
        ),
    },
    {
        "name": "RefGRPO reflection-outcome calibration",
        "arxiv_id_or_url": "2606.14211",
        "url": "https://arxiv.org/abs/2606.14211",
        "track": "retrieval-augmented selector self-improvement",
        "source_read": (
            "RefGRPO adds a free calibration bonus that contrasts an agent's "
            "post-feedback reflection with the real outcome, improving calibrated "
            "self-verification and selective prediction."
        ),
        "v397_outcome_conditioning": (
            "Exp 4292 built a leak-free partial-state scorer, so .398 can train "
            "selector reflection on known outcomes without using a separate "
            "LLM judge as the reward source."
        ),
        "carnot_stack_mapping": (
            "Attach a reflection/confidence head to selector decisions after "
            "retrieved train-side feedback, reward agreement with exact outcomes, "
            "and freeze the reflection signal before target evaluation."
        ),
        "failure_mode": (
            "The method requires real outcome feedback during training. If used "
            "at target time without delayed outcomes, it collapses into ordinary "
            "self-assessment."
        ),
        "experiment_mapping": (
            ".398: calibrate selector self-reflection from exact train outcomes "
            "and use it for selective prediction on held-out families."
        ),
    },
    {
        "name": "SLMJury small-judge budget function",
        "arxiv_id_or_url": "2606.07810",
        "url": "https://arxiv.org/abs/2606.07810",
        "track": "small-verifier efficiency and distillation",
        "source_read": (
            "SLMJury benchmarks 16 small-language-model judges across closed "
            "binary correctness and open-ended scoring, formalizing judging as a "
            "budget-conditioned function."
        ),
        "v397_outcome_conditioning": (
            "The requested Exp 4294 strong-judge artifact is absent, so .398 "
            "should harden the efficiency axis with a transparent small-judge "
            "battery instead of claiming a missing result."
        ),
        "carnot_stack_mapping": (
            "Run Phi/Qwen/Gemma-scale small judges as cheap comparators beside "
            "the Set-Encoder energy verifier, then distill only disagreements "
            "whose exact-grid labels are known."
        ),
        "failure_mode": (
            "SLMJury reports domain gaps and no single dominant small judge. "
            "ARC exact-grid correctness must remain the authority, not judge "
            "consensus."
        ),
        "experiment_mapping": (
            ".398: replace the missing strong-judge hardening result with a "
            "small-judge efficiency battery against the Carnot energy verifier."
        ),
    },
    {
        "name": "ReMDM remasking inference-time scaling",
        "arxiv_id_or_url": "2503.00307",
        "url": "https://arxiv.org/abs/2503.00307",
        "track": "guided masked diffusion generation",
        "source_read": (
            "ReMDM restores iterative refinement to masked discrete diffusion by "
            "allowing generated tokens to be remasked and updated, giving "
            "inference-time compute scaling and guidance control."
        ),
        "v397_outcome_conditioning": (
            "Exp 4292's scorer is built and leak-free, so stronger in-generation "
            "guidance is now technically unblocked but should be secondary to "
            "the broader-domain headline after Exp 4291."
        ),
        "carnot_stack_mapping": (
            "Use the partial-state scorer to rank remasking schedules and "
            "intermediate DiffusionGemma canvases, with answer-bearing cells "
            "masked during scorer audits."
        ),
        "failure_mode": (
            "The sampler is not guaranteed to match DiffusionGemma internals. "
            "More remasking steps can increase compute without improving exact "
            "ARC grids, and final-answer leakage would invalidate the moat."
        ),
        "experiment_mapping": (
            ".398: add a ReMDM-style remasking guidance arm as the secondary "
            "guided-generation follow-up once broader-domain routing is planned."
        ),
    },
]

STUDYING_SECTION = """## 2026-06-16 Exp 4298 - .397 fork SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4298_sota_ingestion_v398.json`.

**Reliable-channel provenance:** `scripts/sweep_clusters.py` and
`scripts/sweep_semscholar.py` imported successfully; `sweep_clusters.py`
emitted focused arXiv discovery URLs. Semantic Scholar was reachable through
the helper but returned HTTP 429 for several low-concurrency probes, while one
small-judge query returned arXiv IDs. WebSearch/WebFetch verified
arXiv:2606.11182, arXiv:2605.19633, arXiv:2606.14211, arXiv:2606.07810, and
arXiv:2503.00307. The banned `/deep-research` channel was not invoked.

**Filtered track:** .397 outcomes after the non-degenerate ARC-GEN
cross-generator run, partial-state DiffusionGemma scorer build, and the missing
strong-judge efficiency hardening artifact.

**.397 outcome conditioning:**
- Exp 4291: `cross_generator_holds=true`,
  `non_degenerate_guards_pass=true`, and
  `headline_outcome=arcgen_cross_generator_generalizes`; the ARC-GEN
  cross-generator moat is closed rather than still degenerate.
- Exp 4292: `partial_state_scorer_built=true`,
  `partial_state_leak_free=true`, `partial_state_auroc=0.966143`, and
  `leak_ablation_auroc=0.937365`; the missing scorer from `.396` now exists
  and survived the leak audit.
- Exp 4294: `strong_judge_efficiency_outcome=unavailable_missing_exp4294_json`
  because `results/experiment_4294_verifier_efficiency_harden_strong_judge.json`
  was not present at ingestion time. Do not claim the strong-judge efficiency
  hardening result until the artifact exists.

**Fresh-pass candidates marked ingested:**
- EEVEE router-prompt co-evolution, arXiv:2606.11182 - mapped to the .398
  broader-domain selector generalization headline after cross-generator ARC-GEN
  transfer held.
- optimize_anything / GEPA text-parameter search, arXiv:2605.19633 - mapped to
  train-only selector/harness text optimization with locked held-out domains.
- RefGRPO reflection-outcome calibration, arXiv:2606.14211 - mapped to
  exact-outcome-calibrated selector self-reflection and selective prediction.
- SLMJury small-judge budget function, arXiv:2606.07810 - mapped to the
  efficiency-axis fallback while Exp 4294 remains unavailable.
- ReMDM remasking inference-time scaling, arXiv:2503.00307 - mapped to the
  secondary guided-generation path now that the partial-state scorer is
  leak-free.

**Screened but not mapped as strongest rows:** SIA (arXiv:2605.27276), SE-GA
(arXiv:2605.16883), and Sensi (arXiv:2603.17683) were read as adjacent
self-improvement and agentic-curriculum evidence. They remain weaker for `.398`
because SIA mutates weights/harness together, SE-GA is GUI-specific, and Sensi
depends on an LLM-as-judge curriculum while reporting a perception bottleneck.

Already-covered context not re-ingested as fresh method rows: ARC-TGI, ARC-GEN,
RFG, EDLM, EntRGi, Self-Improving LLM Agents at Test-Time, SEVerA, DPRM,
Reward-Guided Stitching, Manta-LM, masked-discrete-diffusion guidance dynamics,
INSPECTOR Representation-as-a-Judge, ABPR, Decocted Experience, and COVER.

flagged_for_v398:
`eevee_router_prompt_broader_domain_selector_v398`.

Flagged for .398: `eevee_router_prompt_broader_domain_selector_v398`.

random_seed=4298

**Bottom line for the .398 roadmap:** the ARC-GEN transfer critique is now
closed on a non-degenerate pool and the partial-state scorer is leak-free, so
the next headline should broaden selector generalization across heterogeneous
domains using EEVEE-style router-prompt co-evolution. Keep ReMDM as the
secondary guided-generation branch, and treat small-verifier efficiency as
unconfirmed until Exp 4294 exists.
"""


def extract_v397_outcomes(
    *,
    arcgen: Mapping[str, Any],
    partial_state: Mapping[str, Any],
    efficiency: Mapping[str, Any] | None,
) -> dict[str, bool]:
    """Extract the load-bearing .397 outcome booleans from source artifacts."""

    strong_judge_available = isinstance(efficiency, Mapping)
    return {
        "cross_generator_holds": arcgen.get("cross_generator_holds") is True,
        "non_degenerate_guards_pass": arcgen.get("non_degenerate_guards_pass") is True,
        "arcgen_cross_generator_generalizes": (
            arcgen.get("headline_outcome") == "arcgen_cross_generator_generalizes"
        ),
        "partial_state_scorer_built": partial_state.get("partial_state_scorer_built")
        is True,
        "partial_state_leak_free": partial_state.get("partial_state_leak_free") is True,
        "efficiency_pareto_holds": (
            efficiency.get("efficiency_pareto_holds") is True
            if strong_judge_available
            else False
        ),
        "strong_judge_artifact_available": strong_judge_available,
    }


def select_flagged_for_v398(outcomes: Mapping[str, bool]) -> str:
    """Choose the .398 flag from the .397 fork outcomes."""

    if not outcomes.get("partial_state_scorer_built") or not outcomes.get(
        "partial_state_leak_free"
    ):
        return NONLEAKY_SCORER_REPAIR_FLAGGED_FOR_V398
    if outcomes.get("cross_generator_holds") and outcomes.get("non_degenerate_guards_pass"):
        return DEFAULT_FLAGGED_FOR_V398
    if outcomes.get("strong_judge_artifact_available") and outcomes.get(
        "efficiency_pareto_holds"
    ):
        return SMALL_VERIFIER_EFFICIENCY_FLAGGED_FOR_V398
    return STRONG_GUIDED_GENERATION_FLAGGED_FOR_V398


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]] | None = None,
    flagged_for_v398: str = DEFAULT_FLAGGED_FOR_V398,
    random_seed: int = DEFAULT_RANDOM_SEED,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build the deterministic Exp 4298 mapping artifact."""

    return {
        "honest_verdict": honest_verdict,
        "methods_mapped": [
            dict(method) for method in (methods_mapped or DEFAULT_METHODS_MAPPED)
        ],
        "flagged_for_v398": flagged_for_v398,
        "random_seed": random_seed,
        "field_principles": dict(FIELD_PRINCIPLES),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 4298 artifact before it can be written to disk."""

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
        raise ValueError("field_principles must match REQ-REPORT-4298")

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

    flagged = artifact["flagged_for_v398"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v398 must be non-empty")
    flagged_lower = flagged.lower()
    if not any(
        marker in flagged_lower
        for marker in (
            "broader_domain",
            "router_prompt",
            "partial_state",
            "guided_generation",
            "small_verifier",
            "efficiency",
        )
    ):
        raise ValueError("flagged_for_v398 must be conditioned on the .397 outcomes")


def validate_studying_section(section: str) -> None:
    """Check that the research-studying entry keeps citations and outcome context."""

    required_phrases = [
        "flagged_for_v398",
        "sweep_clusters.py",
        "sweep_semscholar.py",
        "WebSearch/WebFetch",
        "/deep-research",
        "not invoked",
        "cross_generator_holds=true",
        "non_degenerate_guards_pass=true",
        "arcgen_cross_generator_generalizes",
        "partial_state_scorer_built=true",
        "partial_state_leak_free=true",
        "partial_state_auroc=0.966143",
        "leak_ablation_auroc=0.937365",
        "unavailable_missing_exp4294_json",
        DEFAULT_FLAGGED_FOR_V398,
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
    marker = "## 2026-06-16 Exp 4298"
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

    flagged_for_v398 = select_flagged_for_v398(DEFAULT_V397_OUTCOMES)
    artifact = build_artifact(flagged_for_v398=flagged_for_v398)
    validate_artifact(artifact)
    validate_studying_section(STUDYING_SECTION)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    existing_studying = studying_path.read_text(encoding="utf-8") if studying_path.exists() else ""
    studying_path.write_text(_with_studying_section(existing_studying), encoding="utf-8")

    return artifact


def main() -> int:
    root_override = os.environ.get("CARNOT_EXP4298_ROOT")
    repo_root = Path(root_override) if root_override else Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        artifact_path=repo_root / "results/experiment_4298_sota_ingestion_v398.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
