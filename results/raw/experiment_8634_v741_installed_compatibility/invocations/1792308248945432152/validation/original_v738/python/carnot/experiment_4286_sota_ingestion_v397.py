"""Exp 4286 SOTA ingestion for the .396 fork outcomes feeding .397.

Spec refs: REQ-REPORT-4286, SCENARIO-REPORT-4286.

This module writes a planning artifact, not a benchmark result. It closes the
`.396` outcomes into a concrete SOTA-to-experiment mapping: DiffusionGemma did
not establish an external verifier guidance moat because the learned verifier
cannot score partial masked token states; ARC-GEN is not headline-clean after
the degenerate-separation correction; and the cheap energy verifier beat the
LLM judge at much lower cost. The .397 flag therefore prioritizes a learned
partial-state scorer/controller before another guided-generation headline.
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
        "flagged_for_v397",
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
        "v396_outcome_conditioning",
        "carnot_stack_mapping",
        "failure_mode",
        "experiment_mapping",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_v397_mapped"
DEFAULT_FLAGGED_FOR_V397 = "manta_partial_state_scorer_diffusiongemma_v397"
STRONG_GUIDED_GENERATION_FLAGGED_FOR_V397 = (
    "stronger_guided_generation_diffusiongemma_headline_v397"
)
BROADER_DOMAIN_GENERALIZATION_FLAGGED_FOR_V397 = (
    "abpr_broader_domain_generalization_stress_v397"
)
SMALL_VERIFIER_DISTILLATION_FLAGGED_FOR_V397 = (
    "representation_judge_small_verifier_distillation_v397"
)
DEFAULT_RANDOM_SEED = 4286

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. Records ingestion completed with verifiable citations.",
    "methods_mapped": (
        "Each method MUST carry a real arXiv ID/URL (no citation = fabrication "
        "per adversarial_verify discipline) + a one-line .397 experiment mapping."
    ),
    "flagged_for_v397": (
        "Closes discover->ingest->plan: names the strongest method for the .397 "
        "planner, conditioned on the .396 outcomes."
    ),
    "random_seed": (
        "Determinism placeholder for the discovery query set (recorded for "
        "reproducibility of the sweep)."
    ),
}

VERIFIED_SOURCE_URLS = {
    "2605.14531": "https://arxiv.org/abs/2605.14531",
    "2506.10971": "https://arxiv.org/abs/2506.10971",
    "2601.22588": "https://arxiv.org/abs/2601.22588",
    "2603.20334": "https://arxiv.org/abs/2603.20334",
    "2604.04373": "https://arxiv.org/abs/2604.04373",
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(f"arXiv:{source}" for source in VERIFIED_SOURCE_URLS)

DEFAULT_V396_OUTCOMES = {
    "diffusiongemma_guidance_moat": False,
    "partial_state_blocked": True,
    "arcgen_clean_generalization": False,
    "arcgen_degenerate_corrected": True,
    "efficiency_parity_at_lower_cost": True,
}

DEFAULT_METHODS_MAPPED = [
    {
        "name": "Manta-LM closed-loop diffusion control",
        "arxiv_id_or_url": "2605.14531",
        "url": "https://arxiv.org/abs/2605.14531",
        "track": "learned partial-state diffusion scorer",
        "source_read": (
            "Language Generation as Optimal Control frames diffusion generation "
            "as a stochastic control problem and approximates a closed-loop "
            "policy in latent control space."
        ),
        "v396_outcome_conditioning": (
            "Exp 4281 reports diffusiongemma_guidance_moat=false because no "
            "learned Carnot verifier can score partial DiffusionGemma token "
            "canvases. .397 needs the missing partial-state scorer first."
        ),
        "carnot_stack_mapping": (
            "Train a small masked-canvas value head over DiffusionGemma partial "
            "states using final exact-grid outcomes and energy-verifier labels, "
            "then expose score_partial_state before rerunning verifier guidance."
        ),
        "failure_mode": (
            "The paper's latent controller is not a drop-in verifier. If partial "
            "targets leak final answers, the rerun becomes circular rather than "
            "a learned external guidance moat."
        ),
        "experiment_mapping": (
            ".397: build and gate a learned partial-state scorer for "
            "DiffusionGemma masked canvases before any new guided-generation "
            "headline."
        ),
    },
    {
        "name": "Masked discrete diffusion guidance dynamics",
        "arxiv_id_or_url": "2506.10971",
        "url": "https://arxiv.org/abs/2506.10971",
        "track": "guidance-strength diagnostic for masked diffusion",
        "source_read": (
            "What Exactly Does Guidance Do in Masked Discrete Diffusion Models "
            "derives how classifier-free guidance shapes reverse masked "
            "diffusion dynamics and convergence behavior."
        ),
        "v396_outcome_conditioning": (
            "Exp 4281 showed guidance changes selection but the learned arm "
            "could not score partial states. .397 should add trajectory-level "
            "diagnostics before treating stronger guidance as progress."
        ),
        "carnot_stack_mapping": (
            "Log guidance strength, mask entropy, token-change covariance, and "
            "trajectory stability for unguided, RFG, EntRGi, and Carnot-scored "
            "partial-state arms."
        ),
        "failure_mode": (
            "The theory assumes clean model scores and simplified mixtures; ARC "
            "exact-match grids may violate those assumptions, so it is a "
            "diagnostic gate rather than a success metric."
        ),
        "experiment_mapping": (
            ".397: add a masked-diffusion guidance dynamics audit that rejects "
            "over-guided partial-state scorers before full-run promotion."
        ),
    },
    {
        "name": "INSPECTOR Representation-as-a-Judge",
        "arxiv_id_or_url": "2601.22588",
        "url": "https://arxiv.org/abs/2601.22588",
        "track": "small-verifier efficiency and distillation",
        "source_read": (
            "Representation-as-a-Judge replaces prompted LLM judging with "
            "lightweight probes over small-model hidden states, arguing that "
            "evaluation needs less semantic capacity than generation."
        ),
        "v396_outcome_conditioning": (
            "Exp 4284 already shows the Carnot energy verifier beats the LLM "
            "judge by +0.4423 accuracy at cost_ratio=1.95e-08. .397 should "
            "broaden cheap judging without returning to generative judges."
        ),
        "carnot_stack_mapping": (
            "Distill held-out selector outcomes and Qwen judge disagreements "
            "into a decoding-free representation probe, then compare it against "
            "the existing Set-Encoder energy verifier."
        ),
        "failure_mode": (
            "Representation probes can inherit teacher-judge bias and may miss "
            "grid-specific exactness. Keep exact target hashes as the final "
            "acceptance object."
        ),
        "experiment_mapping": (
            ".397: train a small representation judge as an efficiency-axis "
            "replication of Exp 4284, with exact-grid calibration gates."
        ),
    },
    {
        "name": "ABPR trace-guided procedural refinement",
        "arxiv_id_or_url": "2603.20334",
        "url": "https://arxiv.org/abs/2603.20334",
        "track": "cross-substrate verifier generalization",
        "source_read": (
            "Abduction-Based Procedural Refinement couples LLM hypotheses with "
            "a Prolog meta-interpreter and proof-tree traces, with reported "
            "extensions beyond ARC into RAVEN-style relational abstractions."
        ),
        "v396_outcome_conditioning": (
            "Exp 4282's raw ARC-GEN win is adversarial-corrected as degenerate "
            "separation, so .397 needs a non-degenerate relational substrate "
            "rather than another same-shaped generator pool."
        ),
        "carnot_stack_mapping": (
            "Require candidate selectors to emit or consume proof-tree features "
            "on ARC-AGI-2 and RAVEN-style tasks, then test whether the cheap "
            "energy verifier transfers across grid and relational substrates."
        ),
        "failure_mode": (
            "ABPR depends on executable symbolic hypotheses. Pure grid outputs "
            "without stable traces cannot be scored this way without adding a "
            "trace extractor."
        ),
        "experiment_mapping": (
            ".397: replace the degenerate ARC-GEN headline with a proof-trace "
            "cross-substrate generalization stress gate."
        ),
    },
    {
        "name": "Decocted experience for test-time inference",
        "arxiv_id_or_url": "2604.04373",
        "url": "https://arxiv.org/abs/2604.04373",
        "track": "online selector improvement without weight mutation",
        "source_read": (
            "Decocted Experience studies context construction from accumulated "
            "experience as a test-time scaling axis for reasoning and agentic "
            "tasks without updating model parameters."
        ),
        "v396_outcome_conditioning": (
            "Exp 4283's online adaptation result is adversarial-flagged for a "
            "tautology-style metric match, so .397 should prefer retrieval-only "
            "experience context before any selector-head update."
        ),
        "carnot_stack_mapping": (
            "Build a provenance-safe memory of failed selector cases, retrieve "
            "only train-side rule summaries, and feed those summaries as frozen "
            "context features to the selector."
        ),
        "failure_mode": (
            "Experience retrieval can leak target-family structure or stale "
            "incorrect rules. Keep target outputs hidden and report frozen "
            "context versus random-context controls."
        ),
        "experiment_mapping": (
            ".397: test retrieval-only selector context on held-out families "
            "before reviving online weight updates."
        ),
    },
]

STUDYING_SECTION = """## 2026-06-16 Exp 4286 - .396 fork SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4286_sota_ingestion_v397.json`.

**Reliable-channel provenance:** `scripts/sweep_clusters.py` and
`scripts/sweep_semscholar.py` imported successfully; `sweep_clusters.py`
emitted the focused arXiv discovery URLs; Semantic Scholar returned HTTP 429
for the two low-concurrency keyword probes, so it was reachable as code but did
not promote sources. WebSearch/WebFetch verified arXiv:2605.14531,
arXiv:2506.10971, arXiv:2601.22588, arXiv:2603.20334, and arXiv:2604.04373.
The banned `/deep-research` channel was not invoked.

**Filtered track:** .396 outcomes after the DiffusionGemma full run,
ARC-GEN cross-family stress, self-learning repower, and verifier-efficiency
head-to-head.

**.396 outcome conditioning:**
- Exp 4281: `diffusiongemma_guidance_moat=false` and
  `blocked_partial_state_verifier`; the learned verifier cannot score partial
  DiffusionGemma token canvases.
- Exp 4282: raw `arcgen_cross_family_holds=true`, but the outer-loop correction
  records `arcgen_cross_family_holds_outerloop_corrected=false` with
  `DEGENERATE_SEPARATION`, so ARC-GEN is not headline-clean generalization.
- Exp 4284: `efficiency_parity_at_lower_cost=true`,
  `accuracy_delta=0.4423076923`, CI95 `[0.3076923077, 0.5769230769]`, and
  `cost_ratio=1.95e-08`; the cheap energy verifier remains the efficient
  judging path.

**Fresh-pass candidates marked ingested:**
- Manta-LM closed-loop diffusion control, arXiv:2605.14531 - mapped to the
  missing partial-state scorer/controller required before another
  DiffusionGemma guidance headline.
- Masked discrete diffusion guidance dynamics, arXiv:2506.10971 - mapped to a
  guidance-strength and trajectory-stability audit for masked denoising.
- INSPECTOR Representation-as-a-Judge, arXiv:2601.22588 - mapped to small
  representation-probe verifier distillation after Exp 4284's cost win.
- ABPR trace-guided procedural refinement, arXiv:2603.20334 - mapped to a
  non-degenerate proof-trace cross-substrate generalization stress gate.
- Decocted experience for test-time inference, arXiv:2604.04373 - mapped to
  retrieval-only selector context while the online-weight-update result remains
  under tautology correction.

Already-covered context not re-ingested as fresh method rows: RFG, EDLM,
EntRGi, ARC-GEN, Paying Less Generalization Tax, S3, Self-Improving LLM Agents
at Test-Time, SEVerA, ARC-TGI, DPRM, Reward-Guided Stitching, and COVER.

flagged_for_v397:
`manta_partial_state_scorer_diffusiongemma_v397`.

Flagged for .397: `manta_partial_state_scorer_diffusiongemma_v397`.

random_seed=4286

**Bottom line for the .397 roadmap:** the DiffusionGemma moat FAILED for a
specific engineering reason, not because a learned external verifier lost to
RFG. Build the learned partial-state scorer first, keep ARC-GEN out of the
headline until the degenerate pool is repaired with proof-trace or relational
substrates, preserve the cheap-verifier efficiency path, and keep online updates
retrieval-only until the tautology audit is fixed.
"""


def extract_v396_outcomes(
    *,
    diffusiongemma: Mapping[str, Any],
    arcgen: Mapping[str, Any],
    efficiency: Mapping[str, Any],
) -> dict[str, bool]:
    """Extract the load-bearing .396 outcome booleans from source artifacts."""

    headline_arm = diffusiongemma.get("headline_arm")
    headline_status = headline_arm.get("status") if isinstance(headline_arm, Mapping) else None
    corrigendum = arcgen.get("corrigendum_pending")
    has_degenerate_correction = any(
        isinstance(item, Mapping) and item.get("kind") == "DEGENERATE_SEPARATION"
        for item in (corrigendum if isinstance(corrigendum, list) else [])
    )
    arcgen_clean = (
        arcgen.get("arcgen_cross_family_holds") is True
        and arcgen.get("flagged_adversarial") is not True
        and arcgen.get("arcgen_cross_family_holds_outerloop_corrected") is not False
    )

    return {
        "diffusiongemma_guidance_moat": diffusiongemma.get("diffusiongemma_guidance_moat")
        is True,
        "partial_state_blocked": headline_status == "blocked_partial_state_verifier",
        "arcgen_clean_generalization": arcgen_clean,
        "arcgen_degenerate_corrected": has_degenerate_correction
        or arcgen.get("arcgen_cross_family_holds_outerloop_corrected") is False,
        "efficiency_parity_at_lower_cost": efficiency.get("efficiency_parity_at_lower_cost")
        is True,
    }


def select_flagged_for_v397(outcomes: Mapping[str, bool]) -> str:
    """Choose the .397 flag from the .396 fork outcomes."""

    if outcomes.get("diffusiongemma_guidance_moat"):
        return STRONG_GUIDED_GENERATION_FLAGGED_FOR_V397
    if outcomes.get("partial_state_blocked"):
        return DEFAULT_FLAGGED_FOR_V397
    if outcomes.get("arcgen_clean_generalization"):
        return BROADER_DOMAIN_GENERALIZATION_FLAGGED_FOR_V397
    if outcomes.get("efficiency_parity_at_lower_cost"):
        return SMALL_VERIFIER_DISTILLATION_FLAGGED_FOR_V397
    return DEFAULT_FLAGGED_FOR_V397


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]] | None = None,
    flagged_for_v397: str = DEFAULT_FLAGGED_FOR_V397,
    random_seed: int = DEFAULT_RANDOM_SEED,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build the deterministic Exp 4286 mapping artifact."""

    return {
        "honest_verdict": honest_verdict,
        "methods_mapped": [
            dict(method) for method in (methods_mapped or DEFAULT_METHODS_MAPPED)
        ],
        "flagged_for_v397": flagged_for_v397,
        "random_seed": random_seed,
        "field_principles": dict(FIELD_PRINCIPLES),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 4286 artifact before it can be written to disk."""

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
        raise ValueError("field_principles must match REQ-REPORT-4286")

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

    flagged = artifact["flagged_for_v397"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v397 must be non-empty")
    flagged_lower = flagged.lower()
    if not any(
        marker in flagged_lower
        for marker in ("partial_state", "guided_generation", "generalization", "distillation")
    ):
        raise ValueError("flagged_for_v397 must be conditioned on the .396 outcomes")


def validate_studying_section(section: str) -> None:
    """Check that the research-studying entry keeps citations and outcome context."""

    required_phrases = [
        "flagged_for_v397",
        "sweep_clusters.py",
        "sweep_semscholar.py",
        "WebSearch/WebFetch",
        "/deep-research",
        "not invoked",
        "diffusiongemma_guidance_moat=false",
        "blocked_partial_state_verifier",
        "arcgen_cross_family_holds_outerloop_corrected=false",
        "DEGENERATE_SEPARATION",
        "efficiency_parity_at_lower_cost=true",
        "accuracy_delta=0.4423076923",
        "cost_ratio=1.95e-08",
        DEFAULT_FLAGGED_FOR_V397,
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
    marker = "## 2026-06-16 Exp 4286"
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

    flagged_for_v397 = select_flagged_for_v397(DEFAULT_V396_OUTCOMES)
    artifact = build_artifact(flagged_for_v397=flagged_for_v397)
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
    root_override = os.environ.get("CARNOT_EXP4286_ROOT")
    repo_root = Path(root_override) if root_override else Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        artifact_path=repo_root / "results/experiment_4286_sota_ingestion_v397.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
