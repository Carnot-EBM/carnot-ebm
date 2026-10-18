"""Exp 4376 SOTA ingestion for the .404 outcomes feeding .405.

Spec refs: REQ-REPORT-4376, SCENARIO-REPORT-4376.

This module writes a planning artifact, not a benchmark result. It maps the
`.404` fork outcomes onto cited SOTA methods for `.405`: LLM-generated action
heuristics were a clean powered null, E3 advanced one more oracle-grounded ARC
level, DiffusionGemma in-generation conversion retired as unmeasurable, and the
verifier-as-detector probe produced a strong non-oracle step-error signal at
zero selection headroom. The single strongest .405 flag therefore moves to a
detector-first step-error localization plan. A2D2 and SEPO remain out of band
because they train the generator with verifier rewards.
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
        "flagged_for_v405",
        "out_of_band_flagged",
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
        "v404_outcome_conditioning",
        "carnot_stack_mapping",
        "failure_mode",
        "experiment_mapping",
    }
)
REQUIRED_OUT_OF_BAND_FIELDS = frozenset(
    {"name", "arxiv_id_or_url", "url", "reason", "owner_boundary"}
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_v405_mapped"
DEFAULT_FLAGGED_FOR_V405 = "biprm_processbench_detector_localization_v405"
MIND_STUDIO_E3_FLAGGED_FOR_V405 = "mind_studio_e3_lookahead_fidelity_v405"
LLM_HEURISTIC_FLAGGED_FOR_V405 = "llm_generated_action_heuristics_v405"
DIFFUSION_RETIRED_FLAGGED_FOR_V405 = "diffusiongemma_retired_no_inloop_v405"
ALLOWED_FLAGGED_FOR_V405 = frozenset(
    {
        DEFAULT_FLAGGED_FOR_V405,
        MIND_STUDIO_E3_FLAGGED_FOR_V405,
        LLM_HEURISTIC_FLAGGED_FOR_V405,
        DIFFUSION_RETIRED_FLAGGED_FOR_V405,
    }
)
DEFAULT_RANDOM_SEED = 4376

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. Records ingestion completed with verifiable "
        "citations (or blocked_network_unavailable)."
    ),
    "methods_mapped": (
        "Each method MUST carry a real, VERIFIED arXiv ID/URL (no citation = "
        "fabrication) + a one-line .405 experiment mapping + the failure mode "
        "+ the .404-outcome conditioning."
    ),
    "flagged_for_v405": (
        "Closes discover->ingest->plan: names the single strongest method for "
        "the .405 planner, conditioned on the .404 outcomes."
    ),
    "out_of_band_flagged": (
        "Records A2D2/SEPO (verifier-as-reward generator training) as "
        "operator-owned, NOT auto-run in-loop -- the standing HARD RULE."
    ),
    "random_seed": (
        "Determinism placeholder for the discovery query set (reproducibility "
        "of the sweep)."
    ),
}

VERIFIED_SOURCE_URLS = {
    "2508.01682": "https://arxiv.org/abs/2508.01682",
    "2606.16070": "https://arxiv.org/abs/2606.16070",
    "2605.05138": "https://arxiv.org/abs/2605.05138",
    "2503.18809": "https://arxiv.org/abs/2503.18809",
    "2603.20216": "https://arxiv.org/abs/2603.20216",
}
OUT_OF_BAND_SOURCE_URLS = {
    "2606.13565": "https://arxiv.org/abs/2606.13565",
    "2502.01384": "https://arxiv.org/abs/2502.01384",
}
NOTE_REQUIRED_SOURCE_CITATIONS = frozenset(
    f"arXiv:{source}" for source in VERIFIED_SOURCE_URLS
)
NOTE_REQUIRED_OUT_OF_BAND_CITATIONS = frozenset(
    f"arXiv:{source}" for source in OUT_OF_BAND_SOURCE_URLS
)

DEFAULT_V404_OUTCOMES = {
    "llm_acceptance_gate_passed": True,
    "llm_heuristic_beats_linear": False,
    "llm_equal_linear_totals": True,
    "llm_verifier_non_oracle": True,
    "e3_lp85_reproduced": True,
    "e3_new_levels_positive": True,
    "e3_reproducible_total_ge_34": True,
    "e3_verifier_is_oracle": True,
    "diffusion_retired_unmeasurable": True,
    "diffusion_scorer_requalified_leak_clean": False,
    "diffusion_codila_control_differentiates": False,
    "diffusion_benchmark_zero": True,
    "diffusion_guided_beats_control": False,
    "detector_beats_chance": True,
    "detector_ci_lower_gt_chance": True,
    "detector_zero_selection_headroom": True,
    "detector_n_ge_1000": True,
    "detector_verifier_non_oracle": True,
}

DEFAULT_METHODS_MAPPED = [
    {
        "name": "Bidirectional Process Reward Model for step-error detection",
        "arxiv_id_or_url": "2508.01682",
        "url": "https://arxiv.org/abs/2508.01682",
        "source_verification": (
            "Verified by low-concurrency WebSearch/WebFetch and arXiv page on "
            "2026-06-18; found in the focused verifier-as-detector / process "
            "reward model pass after Semantic Scholar returned HTTP 429."
        ),
        "track": "verifier-as-detector / step-error localization",
        "v404_outcome_conditioning": (
            "Exp 4375 reports detector_auroc=0.918304, CI95 lower=0.909296, "
            "detector_beats_chance=true, n_candidates=8829, "
            "selection_headroom.headroom=0.0, and verifier_is_oracle=false."
        ),
        "carnot_stack_mapping": (
            "Run a bidirectional L2R/R2L detector pass over cached FoVer rows "
            "and ARC/E3 trace prefixes, fuse the two scores, and report "
            "earliest-error localization plus risk-coverage without training "
            "a generator."
        ),
        "failure_mode": (
            "Math PRM gains can fail on FoVer or ARC traces; R2L evaluation can "
            "use future context unavailable to an online actor, so .405 must "
            "separate offline detection from in-loop action selection."
        ),
        "experiment_mapping": (
            ".405: make the positive Exp 4375 detector signal actionable with "
            "BiPRM-style bidirectional step-error localization and abstention."
        ),
    },
    {
        "name": "Mind-Studio executable world models with lookahead evaluation",
        "arxiv_id_or_url": "2606.16070",
        "url": "https://arxiv.org/abs/2606.16070",
        "source_verification": (
            "Fresh June 2026 cluster hit verified by arXiv API title lookup and "
            "low-concurrency WebSearch/WebFetch on 2026-06-18."
        ),
        "track": "ARC E3 executable-world-model continuation",
        "v404_outcome_conditioning": (
            "Exp 4372 reports success_e3_deeper_lp85_reproduced, "
            "new_levels_reproduced=1, reproducible_total_levels=34, and "
            "verifier_is_oracle=true."
        ),
        "carnot_stack_mapping": (
            "Add entropy-selected traces, a lightweight skill file, and K-step "
            "lookahead fidelity checks to the ARC E3 induction loop before "
            "planning on tn36/tr87/tu93/sc25/lp85 residual levels."
        ),
        "failure_mode": (
            "The paper targets pygame/Real-ALE-style games, not ARC-AGI-3; "
            "skill files can encode leaked mechanics, and executable rollout "
            "checks remain oracle-grounded rather than a verifier-moat result."
        ),
        "experiment_mapping": (
            ".405: use Mind-Studio-style lookahead fidelity as the next E3 "
            "north-star method after the lp85 level advance."
        ),
    },
    {
        "name": "Executable World Models for ARC-AGI-3",
        "arxiv_id_or_url": "2605.05138",
        "url": "https://arxiv.org/abs/2605.05138",
        "source_verification": (
            "Reverified by arXiv page and low-concurrency WebSearch/WebFetch on "
            "2026-06-18; carried forward as the ARC-AGI-3 E3 baseline."
        ),
        "track": "ARC E3 executable-world-model baseline",
        "v404_outcome_conditioning": (
            "Exp 4372 advanced the E3 line from 33 to 34 reproducible ARC "
            "levels with lp85, while preserving verifier_is_oracle=true."
        ),
        "carnot_stack_mapping": (
            "Continue clean-workspace Python world-model induction with "
            "offline reproduce receipts, explicit model files, and per-target "
            "mechanic-gap labels."
        ),
        "failure_mode": (
            "Solves can improve ARC count while still relying on execution as "
            "an oracle; public-game leakage and repeated-play memory must stay "
            "out of the north-star accounting."
        ),
        "experiment_mapping": (
            ".405: keep E3 as the ARC accuracy continuation, now augmented by "
            "lookahead-fidelity checks rather than promoted as oracle-free."
        ),
    },
    {
        "name": "Classical planning with LLM-generated heuristic programs",
        "arxiv_id_or_url": "2503.18809",
        "url": "https://arxiv.org/abs/2503.18809",
        "source_verification": (
            "Reverified by arXiv page and low-concurrency WebSearch/WebFetch on "
            "2026-06-18; this was the .404 headline method."
        ),
        "track": "LLM-generated / code heuristics for planning",
        "v404_outcome_conditioning": (
            "Exp 4370 reports complete: clean_powered_null_linear_not_beaten, "
            "llm_heuristic_beats_linear=false, and equal held-out totals for "
            "linear, llm_generated, and bfs_baseline at 646 actions."
        ),
        "carnot_stack_mapping": (
            "Retain the static-leakage and reproduce gates as a negative "
            "control, but do not spend the .405 headline on another generated "
            "heuristic sweep unless new held-out games or features appear."
        ),
        "failure_mode": (
            "The stronger function class did not beat the deployed linear "
            "action-cost baseline; rerunning it unchanged risks overfitting the "
            "same reproduced game set."
        ),
        "experiment_mapping": (
            ".405: mark the LLM-generated action-cost arm as settled/null for "
            "this corpus and use it only as a control."
        ),
    },
    {
        "name": "CoDiLA local coherence control for diffusion LMs",
        "arxiv_id_or_url": "2603.20216",
        "url": "https://arxiv.org/abs/2603.20216",
        "source_verification": (
            "Reverified by arXiv page and low-concurrency WebSearch/WebFetch on "
            "2026-06-18; carried forward from the .404 repair-or-retire fork."
        ),
        "track": "verifier-guided diffusion-LM search / local-coherence controls",
        "v404_outcome_conditioning": (
            "Exp 4374 reports retired_in_generation_conversion_unmeasurable, "
            "scorer_requalified_leak_clean=false, "
            "codila_control_differentiates=false, benchmark_n=0, and "
            "s3_guided_beats_control=false."
        ),
        "carnot_stack_mapping": (
            "Keep CoDiLA as a diagnostic control for any future operator repair "
            "of DiffusionGemma, but do not auto-run in-loop generation search "
            "while the scorer and local-control gates are both failed."
        ),
        "failure_mode": (
            "Local coherence scores tied across arms in Exp 4374 and did not "
            "produce a measurable benchmark; continuing the same harness would "
            "fabricate progress from a retired conversion path."
        ),
        "experiment_mapping": (
            ".405: retire DiffusionGemma in-generation conversion from the "
            "autonomous loop until a new scorer/control precondition is met."
        ),
    },
]

DEFAULT_OUT_OF_BAND_FLAGGED = [
    {
        "name": "A2D2 reward-guided any-length discrete diffusion",
        "arxiv_id_or_url": "2606.13565",
        "url": "https://arxiv.org/abs/2606.13565",
        "reason": "verifier-as-reward generator training",
        "owner_boundary": "operator-owned; NOT auto-run in-loop",
    },
    {
        "name": "SEPO score-entropy policy optimization",
        "arxiv_id_or_url": "2502.01384",
        "url": "https://arxiv.org/abs/2502.01384",
        "reason": "policy-gradient fine-tuning over non-differentiable rewards",
        "owner_boundary": "operator-owned; NOT auto-run in-loop",
    },
]

STUDYING_SECTION = """## 2026-06-18 Exp 4376 - .404 fork SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4376_sota_ingestion_v405.json`.

**Preconditions:** network precondition passed via Hugging Face reachability and
arXiv/WebFetch verification. If that check had failed, the only honest artifact
would have been `honest_verdict=blocked_network_unavailable`.

**Reliable-channel provenance:** `scripts/sweep_clusters.py` emitted focused
world-model and verifier/process-reward arXiv discovery URLs and the arXiv API
returned fresh June 2026 IDs. The relevant fresh cluster hit was
arXiv:2606.16070 (Mind-Studio). `scripts/sweep_semscholar.py` was run on focused
LLM-heuristic, diffusion-search, and step-error detector queries; Semantic
Scholar returned HTTP 429, so no S2-only result was promoted. Low-concurrency
WebSearch/WebFetch plus arXiv page checks verified arXiv:2508.01682,
arXiv:2606.16070, arXiv:2605.05138, arXiv:2503.18809, arXiv:2603.20216,
arXiv:2606.13565, and arXiv:2502.01384. Supporting detector benchmark context
was checked against arXiv:2412.06559 and ThinkPRM against arXiv:2504.16828. The
banned `/deep-research` channel was not invoked.

**Filtered track:** .404 outcomes after LLM-generated/code heuristics for
planning, E3 executable-world-model ARC progression, DiffusionGemma
repair-or-retire, and verifier-as-detector step-error measurement.

**.404 outcome conditioning:**
- Exp 4370: `complete: clean_powered_null_linear_not_beaten`,
  `acceptance_gate_passed=true`, `llm_heuristic_beats_linear=false`,
  `held_out_actions_equal=true`, and `verifier_is_oracle=false`. The stronger
  generated-heuristic function class is a clean null on the reproduced corpus,
  not the .405 headline.
- Exp 4372: `success_e3_deeper_lp85_reproduced`,
  `new_levels_reproduced=1`, `reproducible_total_levels=34`, and
  `verifier_is_oracle=true`. E3 remains the ARC north star, but still
  oracle-grounded.
- Exp 4374: `retired_in_generation_conversion_unmeasurable`,
  `scorer_requalified_leak_clean=false`, `codila_control_differentiates=false`,
  `benchmark_n=0`, and `s3_guided_beats_control=false`. DiffusionGemma
  in-generation conversion stays retired from the autonomous in-loop headline.
- Exp 4375: `complete: detector_beats_chance_zero_selection_headroom_fover`,
  `detector_auroc=0.918304`, `detector_beats_chance=true`,
  `selection_headroom.headroom=0.0`, `n_candidates=8829`, and
  `verifier_is_oracle=false`. This is the strongest non-oracle positive .404
  signal.

**Fresh-pass candidates marked ingested:**
- Bidirectional Process Reward Model, arXiv:2508.01682 - mapped to the .405
  detector-first follow-up: bidirectional step-error localization plus
  risk-coverage on cached FoVer and ARC/E3 traces.
- Mind-Studio executable world models with lookahead evaluation,
  arXiv:2606.16070 - mapped to the E3 continuation with entropy-selected traces,
  lightweight skill files, and K-step rollout-fidelity checks.
- Executable World Models for ARC-AGI-3, arXiv:2605.05138 - carried forward as
  the ARC E3 baseline after the lp85 level advance.
- Classical Planning with LLM-Generated Heuristics, arXiv:2503.18809 - marked as
  a clean-null control after Exp 4370 rather than a repeated .405 headline.
- CoDiLA locally coherent parallel decoding, arXiv:2603.20216 - retained only as
  a DiffusionGemma diagnostic/control once scorer and local-control preconditions
  are repaired.

out_of_band_flagged:
- A2D2 adaptive any-length discrete diffusion, arXiv:2606.13565 -
  operator-owned verifier-as-reward generator training, NOT auto-run in-loop.
- SEPO score-entropy policy optimization, arXiv:2502.01384 - operator-owned
  verifier-as-reward generator training, NOT auto-run in-loop.

flagged_for_v405: biprm_processbench_detector_localization_v405

Flagged for .405: `biprm_processbench_detector_localization_v405`

random_seed=4376

**Bottom line for the .405 roadmap:** do not re-run the LLM-generated heuristic
arm unchanged, and do not revive DiffusionGemma in-loop while both scorer and
CoDiLA gates failed. Continue E3 with Mind-Studio-style lookahead fidelity, but
put the single .405 flag on the detector-first BiPRM/ProcessBench-style
step-error localization and abstention path because Exp 4375 produced the clean
non-oracle positive signal. A2D2 and SEPO stay out of band for operator-owned
verifier-as-reward generator training.
"""


def _held_out_totals_equal(actions: object) -> bool:
    if not isinstance(actions, Mapping):
        return False
    linear = actions.get("linear")
    generated = actions.get("llm_generated")
    return isinstance(linear, int) and linear == generated


def _lp85_new_level_reproduced(artifact: Mapping[str, Any]) -> bool:
    rows = artifact.get("per_target_scorecard")
    if not isinstance(rows, Sequence) or isinstance(rows, str):
        return False
    for row in rows:
        if (
            isinstance(row, Mapping)
            and row.get("game") == "lp85"
            and row.get("offline_reproduced") is True
            and row.get("checkpoint_status") == "new_level_reproduced"
        ):
            return True
    return False


def _ci_lower_gt_chance(ci95: object) -> bool:
    if not isinstance(ci95, Sequence) or isinstance(ci95, str) or len(ci95) < 1:
        return False
    lower = ci95[0]
    return isinstance(lower, (int, float)) and lower > 0.5


def extract_v404_outcomes(
    *,
    llm_heuristic: Mapping[str, Any],
    e3: Mapping[str, Any],
    diffusion: Mapping[str, Any],
    detector: Mapping[str, Any],
) -> dict[str, bool]:
    """Extract the load-bearing .404 outcome booleans from source artifacts."""

    reproducible_total = e3.get("reproducible_total_levels")
    selection_headroom = detector.get("selection_headroom")
    n_candidates = detector.get("n_candidates")
    return {
        "llm_acceptance_gate_passed": (
            llm_heuristic.get("acceptance_gate_passed") is True
        ),
        "llm_heuristic_beats_linear": (
            llm_heuristic.get("llm_heuristic_beats_linear") is True
        ),
        "llm_equal_linear_totals": _held_out_totals_equal(
            llm_heuristic.get("held_out_actions_by_heuristic")
        ),
        "llm_verifier_non_oracle": llm_heuristic.get("verifier_is_oracle") is False,
        "e3_lp85_reproduced": _lp85_new_level_reproduced(e3),
        "e3_new_levels_positive": (
            isinstance(e3.get("new_levels_reproduced"), int)
            and e3.get("new_levels_reproduced", 0) > 0
        ),
        "e3_reproducible_total_ge_34": (
            isinstance(reproducible_total, int) and reproducible_total >= 34
        ),
        "e3_verifier_is_oracle": e3.get("verifier_is_oracle") is True,
        "diffusion_retired_unmeasurable": (
            diffusion.get("honest_verdict")
            == "retired_in_generation_conversion_unmeasurable"
        ),
        "diffusion_scorer_requalified_leak_clean": (
            diffusion.get("scorer_requalified_leak_clean") is True
        ),
        "diffusion_codila_control_differentiates": (
            diffusion.get("codila_control_differentiates") is True
        ),
        "diffusion_benchmark_zero": diffusion.get("benchmark_n") == 0,
        "diffusion_guided_beats_control": (
            diffusion.get("s3_guided_beats_control") is True
        ),
        "detector_beats_chance": detector.get("detector_beats_chance") is True,
        "detector_ci_lower_gt_chance": _ci_lower_gt_chance(
            detector.get("detector_auroc_ci95")
        ),
        "detector_zero_selection_headroom": (
            isinstance(selection_headroom, Mapping)
            and selection_headroom.get("headroom") == 0.0
        ),
        "detector_n_ge_1000": isinstance(n_candidates, int) and n_candidates >= 1000,
        "detector_verifier_non_oracle": detector.get("verifier_is_oracle") is False,
    }


def select_flagged_for_v405(outcomes: Mapping[str, bool]) -> str:
    """Choose the .405 flag from the .404 fork outcomes."""

    if (
        outcomes.get("detector_beats_chance")
        and outcomes.get("detector_ci_lower_gt_chance")
        and outcomes.get("detector_zero_selection_headroom")
        and outcomes.get("detector_n_ge_1000")
        and outcomes.get("detector_verifier_non_oracle")
    ):
        return DEFAULT_FLAGGED_FOR_V405
    if outcomes.get("e3_new_levels_positive"):
        return MIND_STUDIO_E3_FLAGGED_FOR_V405
    if outcomes.get("llm_heuristic_beats_linear") and outcomes.get(
        "llm_verifier_non_oracle"
    ):
        return LLM_HEURISTIC_FLAGGED_FOR_V405
    return DIFFUSION_RETIRED_FLAGGED_FOR_V405


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]] | None = None,
    flagged_for_v405: str = DEFAULT_FLAGGED_FOR_V405,
    out_of_band_flagged: Sequence[Mapping[str, str]] | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build the deterministic Exp 4376 mapping artifact."""

    return {
        "honest_verdict": honest_verdict,
        "methods_mapped": [
            dict(method) for method in (methods_mapped or DEFAULT_METHODS_MAPPED)
        ],
        "flagged_for_v405": flagged_for_v405,
        "out_of_band_flagged": [
            dict(method)
            for method in (out_of_band_flagged or DEFAULT_OUT_OF_BAND_FLAGGED)
        ],
        "random_seed": random_seed,
        "field_principles": dict(FIELD_PRINCIPLES),
    }


def _validate_out_of_band(rows: object) -> None:
    if not isinstance(rows, list) or len(rows) != len(OUT_OF_BAND_SOURCE_URLS):
        raise ValueError("out_of_band_flagged must list A2D2 and SEPO")

    seen_sources: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != REQUIRED_OUT_OF_BAND_FIELDS:
            raise ValueError(
                "each out_of_band_flagged row must have exactly the required fields"
            )
        for key, value in row.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"out_of_band_flagged field {key!r} must be a non-empty string"
                )
        source = row["arxiv_id_or_url"]
        if source not in OUT_OF_BAND_SOURCE_URLS:
            raise ValueError(f"out_of_band source {source!r} is not allowed")
        if row["url"] != OUT_OF_BAND_SOURCE_URLS[source]:
            raise ValueError(f"out_of_band url for {source!r} must match")
        if "operator-owned" not in row["owner_boundary"] or "NOT auto-run" not in row[
            "owner_boundary"
        ]:
            raise ValueError("out_of_band rows must record the operator boundary")
        seen_sources.add(source)

    if seen_sources != set(OUT_OF_BAND_SOURCE_URLS):
        raise ValueError("out_of_band_flagged must include A2D2 and SEPO")


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 4376 artifact before it can be written to disk."""

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
        raise ValueError("field_principles must match REQ-REPORT-4376")

    random_seed = artifact["random_seed"]
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("random_seed must be an integer")

    methods = artifact["methods_mapped"]
    if not isinstance(methods, list) or not 3 <= len(methods) <= 5:
        raise ValueError("methods_mapped must contain three to five methods")

    seen_sources: set[str] = set()
    for method in methods:
        if not isinstance(method, Mapping) or set(method) != REQUIRED_METHOD_FIELDS:
            raise ValueError(
                "each method must be a dict with exactly the required fields"
            )

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

    flagged = artifact["flagged_for_v405"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v405 must be non-empty")
    if flagged not in ALLOWED_FLAGGED_FOR_V405:
        raise ValueError("flagged_for_v405 must be conditioned on the .404 outcomes")

    _validate_out_of_band(artifact["out_of_band_flagged"])


def validate_studying_section(section: str) -> None:
    """Check that the research-studying entry keeps citations and outcome context."""

    required_phrases = [
        "flagged_for_v405",
        "out_of_band_flagged",
        "network precondition passed",
        "sweep_clusters.py",
        "sweep_semscholar.py",
        "WebSearch/WebFetch",
        "/deep-research",
        "not invoked",
        "complete: clean_powered_null_linear_not_beaten",
        "llm_heuristic_beats_linear=false",
        "held_out_actions_equal=true",
        "success_e3_deeper_lp85_reproduced",
        "new_levels_reproduced=1",
        "reproducible_total_levels=34",
        "verifier_is_oracle=true",
        "retired_in_generation_conversion_unmeasurable",
        "scorer_requalified_leak_clean=false",
        "codila_control_differentiates=false",
        "benchmark_n=0",
        "s3_guided_beats_control=false",
        "complete: detector_beats_chance_zero_selection_headroom_fover",
        "detector_auroc=0.918304",
        "detector_beats_chance=true",
        "selection_headroom.headroom=0.0",
        "operator-owned",
        "NOT auto-run",
        DEFAULT_FLAGGED_FOR_V405,
        f"random_seed={DEFAULT_RANDOM_SEED}",
    ]
    for phrase in required_phrases:
        if phrase not in section:
            raise ValueError(f"studying section missing required phrase: {phrase}")

    missing_sources = sorted(
        source for source in NOTE_REQUIRED_SOURCE_CITATIONS if source not in section
    )
    if missing_sources:
        raise ValueError(
            f"studying section missing verified source citations: {missing_sources}"
        )

    missing_oob = sorted(
        source
        for source in NOTE_REQUIRED_OUT_OF_BAND_CITATIONS
        if source not in section
    )
    if missing_oob:
        raise ValueError(
            f"studying section missing out-of-band citations: {missing_oob}"
        )


def _with_studying_section(existing: str) -> str:
    marker = "## 2026-06-18 Exp 4376"
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

    flagged_for_v405 = select_flagged_for_v405(DEFAULT_V404_OUTCOMES)
    artifact = build_artifact(flagged_for_v405=flagged_for_v405)
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
    root_override = os.environ.get("CARNOT_EXP4376_ROOT")
    repo_root = Path(root_override) if root_override else Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        artifact_path=repo_root / "results/experiment_4376_sota_ingestion_v405.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
