"""Exp 4343 SOTA ingestion for the .401 outcomes feeding .402.

Spec refs: REQ-REPORT-4343, SCENARIO-REPORT-4343.

This module writes a planning artifact, not a benchmark result. It turns the
`.401` fork outcomes into a citation-gated SOTA-to-experiment map: the
leak-robust in-generation moat replicated, E3 reproduced ar25 and sc25 L1, and
the action-role cross-game value encoder was a powered null. The .402 flag
therefore moves to verifier-guided denoising-trajectory scale-up for the
replicated guided-generation moat, while keeping E3 and cross-game-transfer
follow-ups explicitly conditioned on their observed outcomes.
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
        "flagged_for_v402",
        "random_seed",
        "field_principles",
    }
)
REQUIRED_METHOD_FIELDS = frozenset(
    {
        "name",
        "arxiv_id_or_url",
        "url",
        "source_verification",
        "track",
        "v401_outcome_conditioning",
        "carnot_stack_mapping",
        "failure_mode",
        "experiment_mapping",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_v402_mapped"
DEFAULT_FLAGGED_FOR_V402 = "s3_stratified_scaling_search_guided_generation_v402"
CONSEQUENCE_ORACLE_FREE_FLAGGED_FOR_V402 = (
    "consequence_based_oracle_free_verifier_v402"
)
MULTI_GAME_E3_FLAGGED_FOR_V402 = "multi_game_e3_world_model_sweep_v402"
WORLD_MODEL_INTERACTION_FLAGGED_FOR_V402 = (
    "world_model_interaction_representation_or_retire_transfer_v402"
)
ALLOWED_FLAGGED_FOR_V402 = frozenset(
    {
        DEFAULT_FLAGGED_FOR_V402,
        CONSEQUENCE_ORACLE_FREE_FLAGGED_FOR_V402,
        MULTI_GAME_E3_FLAGGED_FOR_V402,
        WORLD_MODEL_INTERACTION_FLAGGED_FOR_V402,
    }
)
DEFAULT_RANDOM_SEED = 4343

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. Records ingestion completed with verifiable "
        "citations (or blocked_network_unavailable)."
    ),
    "methods_mapped": (
        "Each method MUST carry a real, VERIFIED arXiv ID/URL (no citation = "
        "fabrication) + a one-line .402 experiment mapping + the failure mode "
        "+ the .401-outcome conditioning."
    ),
    "flagged_for_v402": (
        "Closes discover->ingest->plan: names the single strongest method for "
        "the .402 planner, conditioned on the .401 in-generation-moat-settle + "
        "E3 + self-learning outcomes."
    ),
    "random_seed": (
        "Determinism placeholder for the discovery query set (reproducibility "
        "of the sweep)."
    ),
}

VERIFIED_SOURCE_URLS = {
    "2604.06260": "https://arxiv.org/abs/2604.06260",
    "2606.13565": "https://arxiv.org/abs/2606.13565",
    "2606.08501": "https://arxiv.org/abs/2606.08501",
    "2605.05138": "https://arxiv.org/abs/2605.05138",
    "2605.15256": "https://arxiv.org/abs/2605.15256",
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{source}" for source in VERIFIED_SOURCE_URLS
)

DEFAULT_V401_OUTCOMES = {
    "in_generation_moat_replicates": True,
    "scorer_leak_recheck_passed": True,
    "controls_differentiated": True,
    "benchmark_powered": True,
    "replication_ci_excludes_zero": True,
    "e3_ar25_reproduced": True,
    "e3_sc25_reproduced": True,
    "e3_reproduced_levels_positive": True,
    "learned_encoder_transfer_helps": False,
    "positive_control_passed": True,
    "cross_game_ci_lower_exceeds_one": False,
}

DEFAULT_METHODS_MAPPED = [
    {
        "name": "S3 Stratified Scaling Search for diffusion language models",
        "arxiv_id_or_url": "2604.06260",
        "url": "https://arxiv.org/abs/2604.06260",
        "source_verification": (
            "Verified by arXiv API and low-concurrency WebFetch on 2026-06-17; "
            "the first focused Semantic Scholar query surfaced arXiv:2604.06260."
        ),
        "track": "adaptive guided-generation scale-up after a replicated moat",
        "v401_outcome_conditioning": (
            "Exp 4338 reports in_generation_moat_replicates=true, "
            "scorer_leak_recheck_passed=true, controls_differentiated=true, "
            "benchmark_n=240, and replication_ci95=[0.283333, 0.4375]."
        ),
        "carnot_stack_mapping": (
            "Use the leak-robust partial-state scorer as the lightweight verifier "
            "inside denoising-trajectory search: expand candidate partial states, "
            "score them, resample promising trajectories, and preserve diversity "
            "against unguided, best-of-K, and intrinsic self-reward controls."
        ),
        "failure_mode": (
            "Verifier-guided search can over-optimize the scorer, collapse frontier "
            "diversity, or spend the compute budget on redundant denoising branches; "
            "the .402 run needs fixed-NFE controls and held-out leak audits."
        ),
        "experiment_mapping": (
            ".402: run S3-style stratified verifier-guided denoising search as the "
            "headline scale-up of the replicated in-generation moat."
        ),
    },
    {
        "name": "A2D2 reward-guided any-length discrete diffusion",
        "arxiv_id_or_url": "2606.13565",
        "url": "https://arxiv.org/abs/2606.13565",
        "source_verification": (
            "Verified by arXiv API and low-concurrency WebFetch on 2026-06-17."
        ),
        "track": "reward-guided fine-tuning of insertion and unmasking policies",
        "v401_outcome_conditioning": (
            "Exp 4338 makes scaling legitimate, but Exp 4326 showed schedule-only "
            "adaptive guidance was insufficient; .402 needs real reward-tilted "
            "path optimization as a secondary arm."
        ),
        "carnot_stack_mapping": (
            "Translate the leak-robust scorer into a reward for insertion/unmasking "
            "policy updates, with the A2D2 adaptive joint decoding loss compared "
            "against the non-training S3 arm."
        ),
        "failure_mode": (
            "Fine-tuning can optimize verifier artifacts, drift away from the base "
            "DiffusionGemma prior, and blur the oracle-distinct claim; keep this "
            "behind scorer-leak checks and no-weight-update controls."
        ),
        "experiment_mapping": (
            ".402: add a bounded A2D2 adapter arm only after the S3 no-training "
            "scale-up establishes the fixed-model gain."
        ),
    },
    {
        "name": "PAPO reward-state alignment for diffusion LLM reasoning",
        "arxiv_id_or_url": "2606.08501",
        "url": "https://arxiv.org/abs/2606.08501",
        "source_verification": (
            "Verified by arXiv API and low-concurrency WebFetch on 2026-06-17."
        ),
        "track": "step-aware process rewards and entropy-guided replay",
        "v401_outcome_conditioning": (
            "Exp 4338 replicated the moat, so the next risk is not whether the "
            "scorer works but whether rewards remain aligned to authentic "
            "denoising states during scale-up."
        ),
        "carnot_stack_mapping": (
            "Record real denoising trajectories, evaluate one-step partial-state "
            "predictions with the leak-robust scorer, and focus any policy update "
            "or replay budget on high-entropy historical states."
        ),
        "failure_mode": (
            "Process rewards can become dense but wrong if intermediate states "
            "leak final answers or if replay reconstructs artificial contexts; "
            "the experiment must distinguish authentic trajectory states from "
            "random remasking."
        ),
        "experiment_mapping": (
            ".402: use PAPO-style step-aware rewards as the diagnostic ablation "
            "for reward-state alignment in guided denoising."
        ),
    },
    {
        "name": "Executable World Models for ARC-AGI-3",
        "arxiv_id_or_url": "2605.05138",
        "url": "https://arxiv.org/abs/2605.05138",
        "source_verification": (
            "Verified by arXiv API and low-concurrency WebFetch on 2026-06-17."
        ),
        "track": "E3 verified executable world-model sweep after reproduced levels",
        "v401_outcome_conditioning": (
            "Exp 4339 reproduced ar25 L1 and Exp 4341 reproduced sc25 L1, so E3 "
            "has moved from partial model repair to multi-game and deeper-level "
            "verified world-model progression."
        ),
        "carnot_stack_mapping": (
            "Keep the coding agent as proposer, verifier programs as the moat, "
            "and per-game clean workspaces as the leakage control; target the next "
            "deep-tail levels rather than re-running ar25/sc25 L1."
        ),
        "failure_mode": (
            "Public-game overfitting and hidden information channels can make E3 "
            "progress look stronger than it is; the .402 sweep needs fresh "
            "workspace audits and offline reproduction receipts per level."
        ),
        "experiment_mapping": (
            ".402: run a multi-game E3 sweep that extends ar25/sc25 beyond L1 and "
            "targets the remaining deep-tail games with verifier-gated plans."
        ),
    },
    {
        "name": "ReactiveGWM game-agnostic interaction representation",
        "arxiv_id_or_url": "2605.15256",
        "url": "https://arxiv.org/abs/2605.15256",
        "source_verification": (
            "Verified by arXiv API and low-concurrency WebFetch on 2026-06-17."
        ),
        "track": "cross-game transfer after the action-role encoder null",
        "v401_outcome_conditioning": (
            "Exp 4342 reports learned_encoder_transfer_helps=false, "
            "cross_game_state_reduction_ci95=[1.0, 1.0168354897287482], and "
            "positive_control_passed=true; the action-role value head is a real "
            "null rather than a degenerate test."
        ),
        "carnot_stack_mapping": (
            "If cross-game transfer is kept alive, move from a shallow value head "
            "to a world-model interaction module that explicitly separates player "
            "actions, object responses, and transferable interaction roles."
        ),
        "failure_mode": (
            "ReactiveGWM is video/NPC-centered, not symbolic ARC-grid-centered, "
            "and Exp 4342 already nulls the cheap action-role version; a .402 "
            "retry must be full interaction-model transfer or retire the line."
        ),
        "experiment_mapping": (
            ".402: either build a full interaction-world-model transfer arm or "
            "retire cross-game value transfer after the third powered null."
        ),
    },
]

STUDYING_SECTION = """## 2026-06-17 Exp 4343 - .401 outcome SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4343_sota_ingestion_v402.json`.

**Preconditions:** network precondition passed via Hugging Face reachability and
arXiv API verification. If that check had failed, the only honest artifact would
have been `honest_verdict=blocked_network_unavailable`.

**Reliable-channel provenance:** `scripts/sweep_clusters.py` emitted focused
arXiv discovery URLs for energy/reward and world-model clusters. The first
`scripts/sweep_semscholar.py` query returned arXiv:2604.06260 and
arXiv:2602.23997; subsequent focused Semantic Scholar probes returned HTTP 429.
Low-concurrency WebSearch/WebFetch plus the arXiv API verified arXiv:2604.06260,
arXiv:2606.13565, arXiv:2606.08501, arXiv:2606.10829, arXiv:2603.12554,
arXiv:2509.25420, arXiv:2605.05138, arXiv:2605.15256, arXiv:2602.06291, and
arXiv:2602.23997. The banned `/deep-research` channel was not invoked.

**Filtered track:** .401 outcomes after leak-robust in-generation moat
replication, E3 explore-verify-plan reproduction on ar25 and sc25, and
action-role cross-game self-learning.

**.401 outcome conditioning:**
- Exp 4338: `honest_verdict=complete: in_generation_moat_replicates`,
  `in_generation_moat_replicates=true`, `scorer_leak_recheck_passed=true`,
  `controls_differentiated=true`, `benchmark_n=240`,
  `carnot_minus_best_control_delta=0.358333`, and
  `replication_ci95=[0.283333, 0.4375]`; the leak-robust in-generation moat
  replicated and the .402 headline should scale it rather than pivot away.
- Exp 4339: `game=ar25`, `offline_reproduced=true`, `plan_executed=true`,
  `reproduced_levels=1`, and `explore_lemmas_collected=7`; E3 has a reproduced
  ar25 level and should move to deeper/multi-game progression.
- Exp 4341: `game=sc25`, `offline_reproduced=true`, `plan_executed=true`,
  `reproduced_levels=1`, and `explore_lemmas_collected=6`; sc25 reproduction
  opens the path to converting the live-recorded levels, not another L1 replay.
- Exp 4342: `learned_encoder_transfer_helps=false`,
  `cross_game_state_reduction=1.00635593220339`,
  `cross_game_state_reduction_ci95=[1.0, 1.0168354897287482]`, and
  `positive_control_passed=true`; action-role cross-game value transfer is a
  powered null and needs a full interaction-world-model transfer arm or
  retirement.

**Fresh-pass candidates marked ingested:**
- S3 Stratified Scaling Search, arXiv:2604.06260 - mapped to the .402 headline:
  verifier-guided denoising-trajectory search over the leak-robust scorer.
- A2D2 adaptive any-length discrete diffusion, arXiv:2606.13565 - mapped to a
  secondary reward-guided fine-tuning arm if the fixed-model S3 scale-up holds.
- PAPO reward-state alignment, arXiv:2606.08501 - mapped to step-aware process
  rewards and entropy-guided replay diagnostics for authentic denoising states.
- Executable World Models for ARC-AGI-3, arXiv:2605.05138 - mapped to a
  multi-game/deeper-level E3 sweep after ar25 and sc25 L1 reproduced.
- ReactiveGWM, arXiv:2605.15256 - mapped to the only remaining cross-game path:
  full interaction-world-model transfer, otherwise retire the transfer line.

**Screened but not mapped as strongest rows:** ADAS (arXiv:2606.10829),
Entropy-Guided Step Selection (arXiv:2603.12554), Reward-Guided Dual-Phase
Search (arXiv:2509.25420), Foundation World Models (arXiv:2602.23997), and
Consequence-Based Utility (arXiv:2602.06291) were verified and read as context.
Consequence-Based Utility remains the correct lead if a future leak-robust
moat recheck retires in-generation guidance, but Exp 4338 makes the active
.402 branch a guided-generation scale-up instead.

flagged_for_v402:
`s3_stratified_scaling_search_guided_generation_v402`.

Flagged for .402: `s3_stratified_scaling_search_guided_generation_v402`.

random_seed=4343

**Bottom line for the .402 roadmap:** the in-generation moat settled positive in
Exp 4338, so do not pivot to consequence-based oracle-free ranking as the lead.
Scale the moat with S3-style verifier-guided denoising-trajectory search under
fixed-compute controls, keep A2D2/PAPO as training and reward-state ablations,
turn E3 into a multi-game/deeper-level reproduced-world-model sweep, and either
upgrade cross-game transfer to a full interaction-world-model representation or
retire it after the powered action-role null.
"""


def _ci_excludes_zero(values: object) -> bool:
    if isinstance(values, Sequence) and len(values) == 2:
        lower, upper = values
        if isinstance(lower, int | float) and isinstance(upper, int | float):
            return lower > 0 or upper < 0
    return False


def _ci_lower_exceeds_one(values: object) -> bool:
    if isinstance(values, Sequence) and len(values) == 2:
        lower = values[0]
        return isinstance(lower, int | float) and lower > 1.0
    return False


def _reproduced_level(artifact: Mapping[str, Any], game: str) -> bool:
    reproduced_levels = artifact.get("reproduced_levels")
    return (
        artifact.get("game") == game
        and artifact.get("offline_reproduced") is True
        and artifact.get("plan_executed") is True
        and isinstance(reproduced_levels, int)
        and reproduced_levels > 0
    )


def extract_v401_outcomes(
    *,
    moat: Mapping[str, Any],
    e3_ar25: Mapping[str, Any],
    e3_sc25: Mapping[str, Any],
    transfer: Mapping[str, Any],
) -> dict[str, bool]:
    """Extract the load-bearing .401 outcome booleans from source artifacts."""

    benchmark_n = moat.get("benchmark_n")
    return {
        "in_generation_moat_replicates": (
            moat.get("in_generation_moat_replicates") is True
        ),
        "scorer_leak_recheck_passed": moat.get("scorer_leak_recheck_passed") is True,
        "controls_differentiated": moat.get("controls_differentiated") is True,
        "benchmark_powered": isinstance(benchmark_n, int) and benchmark_n >= 200,
        "replication_ci_excludes_zero": _ci_excludes_zero(
            moat.get("replication_ci95")
        ),
        "e3_ar25_reproduced": _reproduced_level(e3_ar25, "ar25"),
        "e3_sc25_reproduced": _reproduced_level(e3_sc25, "sc25"),
        "e3_reproduced_levels_positive": _reproduced_level(
            e3_ar25, "ar25"
        )
        or _reproduced_level(e3_sc25, "sc25"),
        "learned_encoder_transfer_helps": (
            transfer.get("learned_encoder_transfer_helps") is True
        ),
        "positive_control_passed": transfer.get("positive_control_passed") is True,
        "cross_game_ci_lower_exceeds_one": _ci_lower_exceeds_one(
            transfer.get("cross_game_state_reduction_ci95")
        ),
    }


def select_flagged_for_v402(outcomes: Mapping[str, bool]) -> str:
    """Choose the .402 flag from the .401 fork outcomes."""

    if not outcomes.get("in_generation_moat_replicates") or not outcomes.get(
        "scorer_leak_recheck_passed"
    ):
        return CONSEQUENCE_ORACLE_FREE_FLAGGED_FOR_V402
    if outcomes.get("replication_ci_excludes_zero"):
        return DEFAULT_FLAGGED_FOR_V402
    if outcomes.get("e3_ar25_reproduced") and outcomes.get("e3_sc25_reproduced"):
        return MULTI_GAME_E3_FLAGGED_FOR_V402
    if outcomes.get("learned_encoder_transfer_helps"):
        return WORLD_MODEL_INTERACTION_FLAGGED_FOR_V402
    return CONSEQUENCE_ORACLE_FREE_FLAGGED_FOR_V402


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]] | None = None,
    flagged_for_v402: str = DEFAULT_FLAGGED_FOR_V402,
    random_seed: int = DEFAULT_RANDOM_SEED,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build the deterministic Exp 4343 mapping artifact."""

    return {
        "honest_verdict": honest_verdict,
        "methods_mapped": [
            dict(method) for method in (methods_mapped or DEFAULT_METHODS_MAPPED)
        ],
        "flagged_for_v402": flagged_for_v402,
        "random_seed": random_seed,
        "field_principles": dict(FIELD_PRINCIPLES),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 4343 artifact before it can be written to disk."""

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
        raise ValueError("field_principles must match REQ-REPORT-4343")

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

    flagged = artifact["flagged_for_v402"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v402 must be non-empty")
    if flagged not in ALLOWED_FLAGGED_FOR_V402:
        raise ValueError("flagged_for_v402 must be conditioned on the .401 outcomes")


def validate_studying_section(section: str) -> None:
    """Check that the research-studying entry keeps citations and outcome context."""

    required_phrases = [
        "flagged_for_v402",
        "network precondition passed",
        "sweep_clusters.py",
        "sweep_semscholar.py",
        "WebSearch/WebFetch",
        "/deep-research",
        "not invoked",
        "in_generation_moat_replicates=true",
        "scorer_leak_recheck_passed=true",
        "controls_differentiated=true",
        "benchmark_n=240",
        "carnot_minus_best_control_delta=0.358333",
        "replication_ci95=[0.283333, 0.4375]",
        "game=ar25",
        "offline_reproduced=true",
        "plan_executed=true",
        "reproduced_levels=1",
        "explore_lemmas_collected=7",
        "game=sc25",
        "explore_lemmas_collected=6",
        "learned_encoder_transfer_helps=false",
        "cross_game_state_reduction=1.00635593220339",
        "cross_game_state_reduction_ci95=[1.0, 1.0168354897287482]",
        "positive_control_passed=true",
        DEFAULT_FLAGGED_FOR_V402,
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
    marker = "## 2026-06-17 Exp 4343"
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

    flagged_for_v402 = select_flagged_for_v402(DEFAULT_V401_OUTCOMES)
    artifact = build_artifact(flagged_for_v402=flagged_for_v402)
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
    root_override = os.environ.get("CARNOT_EXP4343_ROOT")
    repo_root = Path(root_override) if root_override else Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        artifact_path=repo_root / "results/experiment_4343_sota_ingestion_v402.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
