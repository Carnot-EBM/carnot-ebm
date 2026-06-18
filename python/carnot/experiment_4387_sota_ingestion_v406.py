"""Exp 4387 SOTA ingestion for the .405 outcomes feeding .406.

Spec refs: REQ-REPORT-4387, SCENARIO-REPORT-4387.

This module writes a planning artifact, not a benchmark result. It maps the
`.405` fork outcomes onto cited SOTA methods for `.406`: BiPRM-style
bidirectional localization was a clean null, the skeptic-proof phase stayed
blocked, detector self-learning compounded, cross-domain detection generalized
to GAP-4 ARC, and ARC E3 lookahead produced honest partials with zero new
reproduced levels. The single strongest .406 flag therefore moves to
verifiable process-data synthesis for cross-domain first-error localization.
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
        "flagged_for_v406",
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
        "v405_outcome_conditioning",
        "carnot_stack_mapping",
        "failure_mode",
        "experiment_mapping",
    }
)
REQUIRED_OUT_OF_BAND_FIELDS = frozenset(
    {"name", "arxiv_id_or_url", "url", "reason", "owner_boundary"}
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_v406_mapped"
DEFAULT_FLAGGED_FOR_V406 = "verifiable_process_data_cross_domain_localization_v406"
PVD_SELECTIVE_FLAGGED_FOR_V406 = "pvd_selective_abstention_v406"
MULTI_DOMAIN_CALIBRATION_FLAGGED_FOR_V406 = "multi_domain_calibrated_detector_v406"
MIND_STUDIO_GAP_REPAIR_FLAGGED_FOR_V406 = "mind_studio_mechanic_gap_repair_v406"
BIPRM_BASELINE_ONLY_FLAGGED_FOR_V406 = "biprm_baseline_only_no_reheadline_v406"
ALLOWED_FLAGGED_FOR_V406 = frozenset(
    {
        DEFAULT_FLAGGED_FOR_V406,
        PVD_SELECTIVE_FLAGGED_FOR_V406,
        MULTI_DOMAIN_CALIBRATION_FLAGGED_FOR_V406,
        MIND_STUDIO_GAP_REPAIR_FLAGGED_FOR_V406,
        BIPRM_BASELINE_ONLY_FLAGGED_FOR_V406,
    }
)
DEFAULT_RANDOM_SEED = 4387

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. Records ingestion completed with verifiable "
        "citations (or blocked_network_unavailable)."
    ),
    "methods_mapped": (
        "Each method MUST carry a real, VERIFIED arXiv ID/URL (no citation = "
        "fabrication) + a one-line .406 experiment mapping + the failure mode "
        "+ the .405-outcome conditioning."
    ),
    "flagged_for_v406": (
        "Closes discover->ingest->plan: names the single strongest method for "
        "the .406 planner, conditioned on the .405 outcomes."
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
    "2605.02395": "https://arxiv.org/abs/2605.02395",
    "2102.10395": "https://arxiv.org/abs/2102.10395",
    "2605.25133": "https://arxiv.org/abs/2605.25133",
    "2504.16828": "https://arxiv.org/abs/2504.16828",
    "2606.16070": "https://arxiv.org/abs/2606.16070",
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

DEFAULT_V405_OUTCOMES = {
    "biprm_localization_null": True,
    "detector_localization_actionable": False,
    "biprm_delta_zero": True,
    "biprm_abstention_no_useful_point": True,
    "biprm_verifier_non_oracle": True,
    "skeptic_proof_blocked": True,
    "detector_compounds": True,
    "compounding_ci_lower_gt_zero": True,
    "compounds_verifier_non_oracle": True,
    "fresh_headroom_points_to_cross_domain": True,
    "cross_domain_generalizes": True,
    "gap4_arc_ci_lower_gt_chance": True,
    "gap4_arc_selection_headroom_positive": True,
    "cross_domain_verifier_non_oracle": True,
    "e3_deeper_new_levels_positive": False,
    "e3_tails_new_levels_positive": False,
    "e3_total_levels_stable_34": True,
    "e3_verifier_is_oracle": True,
}

DEFAULT_METHODS_MAPPED = [
    {
        "name": "Controllable and Verifiable Process Data Synthesis for PRMs",
        "arxiv_id_or_url": "2605.02395",
        "url": "https://arxiv.org/abs/2605.02395",
        "source_verification": (
            "Verified by arXiv abs URL, focused Semantic Scholar query, and "
            "low-concurrency WebSearch/WebFetch on 2026-06-18."
        ),
        "track": "cross-domain first-error localization data",
        "v405_outcome_conditioning": (
            "Exp 4381 reports detector_localization_actionable=false with "
            "localization_delta_ci95=[0.0, 0.0] and no useful abstention point, "
            "while Exp 4385/4386 report detector_compounds=true and "
            "detector_generalizes_cross_domain=true."
        ),
        "carnot_stack_mapping": (
            "Generate prefix-invalid, trajectory-consistent first-error pairs "
            "for FoVer and GAP-4 ARC, verify the injected step against symbolic "
            "or executable prefixes, and fit the detector/localizer against a "
            "held-out cross-domain split."
        ),
        "failure_mode": (
            "Synthetic symbolic errors can miss ARC mechanics or leak template "
            "artifacts; .406 must keep source-domain labels, leave-domain-out "
            "validation, and executable prefix checks before claiming a localizer."
        ),
        "experiment_mapping": (
            ".406: turn the compounding/generalizing detector into a "
            "cross-domain first-error localizer by adding verifiable process "
            "data rather than retrying raw BiPRM fusion."
        ),
    },
    {
        "name": "Multi-domain calibration for OOD detector generalization",
        "arxiv_id_or_url": "2102.10395",
        "url": "https://arxiv.org/abs/2102.10395",
        "source_verification": (
            "Verified by arXiv abs URL and low-concurrency WebSearch/WebFetch "
            "on 2026-06-18 for the cross-domain/OOD detector track."
        ),
        "track": "cross-domain detector calibration",
        "v405_outcome_conditioning": (
            "Exp 4386 reports success: detector_generalizes_cross_domain_non_fover "
            "on GAP-4 ARC with detection_auroc=0.963317, "
            "auroc_ci95=[0.922285, 0.990662], selection_headroom=0.129, "
            "and verifier_is_oracle=false."
        ),
        "carnot_stack_mapping": (
            "Treat FoVer and GAP-4 ARC as separate calibration domains, add "
            "leave-one-domain-out calibration loss or temperature tuning, and "
            "report risk/coverage and localization-F1 by domain."
        ),
        "failure_mode": (
            "Two domains are too few for a broad OOD claim; calibration can "
            "improve confidence without improving first-error localization, and "
            "code/GSM remain unavailable until labeled verifier-score pools exist."
        ),
        "experiment_mapping": (
            ".406: scale the cross-domain detector win into a calibrated "
            "multi-domain detector contract with explicit unavailable-domain gaps."
        ),
    },
    {
        "name": "Prover-Verifier Deliberation for selective LLM prediction",
        "arxiv_id_or_url": "2605.25133",
        "url": "https://arxiv.org/abs/2605.25133",
        "source_verification": (
            "Verified by arXiv abs/html URL and low-concurrency WebSearch/WebFetch "
            "on 2026-06-18 in the selective-prediction/abstention pass."
        ),
        "track": "selective prediction / abstention for verifier outputs",
        "v405_outcome_conditioning": (
            "Exp 4381 reports useful_operating_point=null for threshold-only "
            "abstention, so .406 needs a structured report/abstain signal if "
            "selective prediction is retried."
        ),
        "carnot_stack_mapping": (
            "Wrap high-impact detector verdicts in a bounded prover/verifier "
            "challenge protocol over checkable subclaims, then accept only "
            "answers that survive without revision while logging abstentions."
        ),
        "failure_mode": (
            "PVD can collapse outside the verifier's effective region, costs "
            "extra calls, and does not itself repair first-error labels; it must "
            "remain an abstention layer over a validated detector/localizer."
        ),
        "experiment_mapping": (
            ".406: retry abstention only as structured verifier deliberation "
            "on cross-domain detector outputs, not as a raw score threshold."
        ),
    },
    {
        "name": "ThinkPRM generative step-wise verifier",
        "arxiv_id_or_url": "2504.16828",
        "url": "https://arxiv.org/abs/2504.16828",
        "source_verification": (
            "Reverified by arXiv abs URL and low-concurrency WebSearch/WebFetch "
            "on 2026-06-18; carried from prior verifier/PRM sweeps."
        ),
        "track": "generative verifier explanations for first-error gaps",
        "v405_outcome_conditioning": (
            "Exp 4381 logs missed_first_error_traces=103 under an untyped "
            "FoVer missing-verifier gap, while Exp 4385 shows detector learning "
            "can improve when better supervision exists."
        ),
        "carnot_stack_mapping": (
            "Use a small, bounded generative verifier pass to explain the first "
            "unsupported transition on missed traces and convert accepted "
            "explanations into typed verifier-gap labels for the detector."
        ),
        "failure_mode": (
            "Long-CoT PRMs are expensive and can hallucinate rationales; all "
            "explanations must be checked against executable or symbolic labels "
            "before they enter the training/evaluation set."
        ),
        "experiment_mapping": (
            ".406: use ThinkPRM-style explanations as a label generator for the "
            "untyped first-error gap, gated by verifiable prefix checks."
        ),
    },
    {
        "name": "Mind-Studio executable world models with lookahead evaluation",
        "arxiv_id_or_url": "2606.16070",
        "url": "https://arxiv.org/abs/2606.16070",
        "source_verification": (
            "Reverified by arXiv abs/html URL and low-concurrency WebSearch/WebFetch "
            "on 2026-06-18 for the ARC E3 lookahead track."
        ),
        "track": "ARC E3 executable-world-model mechanic-gap repair",
        "v405_outcome_conditioning": (
            "Exp 4383 and Exp 4384 both report honest partials with "
            "new_levels_reproduced=0, reproducible_total_levels=34, and "
            "verifier_is_oracle=true across high-headroom and blocked-tail games."
        ),
        "carnot_stack_mapping": (
            "Keep entropy-selected traces, skill files, and K-step lookahead "
            "fidelity, but aim .406 at named mechanic-gap repair for ar25, ka59, "
            "ft09, tn36, and tr87 before any replayable solve claim."
        ),
        "failure_mode": (
            "Lookahead fidelity stayed partial and oracle-grounded; skill files "
            "can encode leakage, and zero new levels in .405 means .406 must "
            "treat E3 as north-star repair work rather than the detector headline."
        ),
        "experiment_mapping": (
            ".406: preserve E3 as the ARC north star with mechanic-gap repair, "
            "but do not displace the cross-domain detector data plan."
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

STUDYING_SECTION = """## 2026-06-18 Exp 4387 - .405 fork SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4387_sota_ingestion_v406.json`.

**Preconditions:** network precondition passed via Hugging Face reachability and
arXiv/WebFetch verification. If that check had failed, the only honest artifact
would have been `honest_verdict=blocked_network_unavailable`.

**Reliable-channel provenance:** `scripts/sweep_clusters.py` emitted focused
verifier/process-reward and world-model arXiv discovery URLs. `scripts/sweep_semscholar.py`
was run on focused bidirectional-PRM, selective-prediction, cross-domain verifier,
and ARC E3 lookahead queries; it returned arXiv:2603.16253, arXiv:2605.02395,
arXiv:2601.18984, arXiv:2603.02119, arXiv:2506.11474, arXiv:2601.14209, and
arXiv:2603.25412 before Semantic Scholar rate-limited the remaining focused
queries with HTTP 429. Low-concurrency WebSearch/WebFetch plus arXiv abs/html
checks verified arXiv:2605.02395, arXiv:2102.10395, arXiv:2605.25133,
arXiv:2504.16828, arXiv:2606.16070, arXiv:2508.01682, arXiv:2605.05138,
arXiv:2606.13565, and arXiv:2502.01384. The banned `/deep-research` channel was
not invoked.

**Filtered track:** .405 outcomes after BiPRM detector localization/abstention,
skeptic-proof gating, detector self-learning compounding, cross-domain detector
generalization, and ARC E3 lookahead/mechanic-gap work.

**.405 outcome conditioning:**
- Exp 4381: `complete: clean_powered_null_bidirectional_not_actionable`,
  `detector_localization_actionable=false`, `localization_delta_ci95=[0.0, 0.0]`,
  `useful_operating_point=null`, and `verifier_is_oracle=false`. BiPRM-style
  bidirectional fusion is a baseline/null for this corpus, not the .406 headline.
- Exp 4382: `blocked_gate_check_failed` because
  `detector_localization_actionable=false`. The skeptic-proof phase remains
  gated off until localization becomes actionable.
- Exp 4385: `success: detector_compounds_heldout_localization_f1`,
  `detector_compounds=true`, `compounding_delta_ci95=[0.003396, 0.032772]`,
  and `verifier_is_oracle=false`. Detector self-learning is the clean .405
  oracle-distinct positive.
- Exp 4386: `success: detector_generalizes_cross_domain_non_fover`,
  `detector_generalizes_cross_domain=true`, GAP-4 ARC `detection_auroc=0.963317`,
  `auroc_ci95=[0.922285, 0.990662]`, `selection_headroom=0.129`,
  `n=28443`, and `verifier_is_oracle=false`. Detection generalizes beyond
  FoVer and has real selection headroom, but code/GSM remain unavailable-domain
  gaps.
- Exp 4383/4384: `complete_e3_deeper_partial` and
  `complete_e3_ar25_ka59_ft09_partial`, both with `new_levels_reproduced=0`,
  `reproducible_total_levels=34`, and `verifier_is_oracle=true`. ARC E3 remains
  the north star, but .405 yielded mechanic-gap repair work rather than new
  solves.

**Fresh-pass candidates marked ingested:**
- Controllable and Verifiable Process Data Synthesis for PRMs, arXiv:2605.02395
  - mapped to .406 verifiable first-error data for cross-domain localization.
- On Calibration and Out-of-domain Generalization, arXiv:2102.10395 - mapped to
  multi-domain detector calibration after the GAP-4 ARC generalization win.
- Trust but Verify: Prover-Verifier Deliberation for Selective LLM Prediction,
  arXiv:2605.25133 - mapped to a structured report/abstain layer if abstention
  is retried after the raw threshold null.
- ThinkPRM, arXiv:2504.16828 - mapped to bounded explanation labels for the
  untyped first-error gap, gated by executable or symbolic checks.
- Mind-Studio executable world models with lookahead evaluation,
  arXiv:2606.16070 - carried as E3 mechanic-gap repair after zero new .405
  reproduced levels.

Carried baseline context: BiPRM, arXiv:2508.01682, and Executable World Models
for ARC-AGI-3, arXiv:2605.05138, remain verified context, but the .405 outcomes
make them baseline/north-star supports rather than the single .406 flag.

out_of_band_flagged:
- A2D2 adaptive any-length discrete diffusion, arXiv:2606.13565 -
  operator-owned verifier-as-reward generator training, NOT auto-run in-loop.
- SEPO score-entropy policy optimization, arXiv:2502.01384 - operator-owned
  verifier-as-reward generator training, NOT auto-run in-loop.

flagged_for_v406: verifiable_process_data_cross_domain_localization_v406

Flagged for .406: `verifiable_process_data_cross_domain_localization_v406`

random_seed=4387

**Bottom line for the .406 roadmap:** do not re-headline BiPRM fusion after the
clean actionable-localization null, and do not unlock skeptic-proofing until a
localizer exists. The live .405 signal is detector self-learning plus GAP-4 ARC
cross-domain detection. The .406 flag should therefore add verifiable
process-supervision data for first-error localization across domains, then use
multi-domain calibration and structured abstention only after the localizer is
real. A2D2 and SEPO stay out of band for operator-owned verifier-as-reward
generator training.
"""


def _ci_lower_gt_zero(ci95: object) -> bool:
    if not isinstance(ci95, Sequence) or isinstance(ci95, str) or len(ci95) < 1:
        return False
    lower = ci95[0]
    return isinstance(lower, (int, float)) and lower > 0.0


def _ci_lower_gt_chance(ci95: object) -> bool:
    if not isinstance(ci95, Sequence) or isinstance(ci95, str) or len(ci95) < 1:
        return False
    lower = ci95[0]
    return isinstance(lower, (int, float)) and lower > 0.5


def _gap4_arc_domain_generalizes(artifact: Mapping[str, Any]) -> bool:
    rows = artifact.get("detection_by_domain")
    if not isinstance(rows, Sequence) or isinstance(rows, str):
        return False
    for row in rows:
        if not isinstance(row, Mapping) or row.get("domain") != "gap4_arc":
            continue
        headroom = row.get("selection_headroom")
        n = row.get("n")
        if (
            _ci_lower_gt_chance(row.get("auroc_ci95"))
            and isinstance(headroom, (int, float))
            and headroom > 0.0
            and isinstance(n, int)
            and n >= 1000
        ):
            return True
    return False


def extract_v405_outcomes(
    *,
    localization: Mapping[str, Any],
    skeptic: Mapping[str, Any],
    compounds: Mapping[str, Any],
    cross_domain: Mapping[str, Any],
    e3_deeper: Mapping[str, Any],
    e3_tails: Mapping[str, Any],
) -> dict[str, bool]:
    """Extract the load-bearing .405 outcome booleans from source artifacts."""

    deeper_total = e3_deeper.get("reproducible_total_levels")
    tails_total = e3_tails.get("reproducible_total_levels")
    abstention_curve = localization.get("abstention_curve")
    return {
        "biprm_localization_null": (
            localization.get("honest_verdict")
            == "complete: clean_powered_null_bidirectional_not_actionable"
        ),
        "detector_localization_actionable": (
            localization.get("detector_localization_actionable") is True
        ),
        "biprm_delta_zero": localization.get("localization_delta_ci95") == [0.0, 0.0],
        "biprm_abstention_no_useful_point": (
            isinstance(abstention_curve, Mapping)
            and abstention_curve.get("useful_operating_point") is None
        ),
        "biprm_verifier_non_oracle": localization.get("verifier_is_oracle") is False,
        "skeptic_proof_blocked": (
            skeptic.get("honest_verdict") == "blocked_gate_check_failed"
        ),
        "detector_compounds": compounds.get("detector_compounds") is True,
        "compounding_ci_lower_gt_zero": _ci_lower_gt_zero(
            compounds.get("compounding_delta_ci95")
        ),
        "compounds_verifier_non_oracle": compounds.get("verifier_is_oracle") is False,
        "fresh_headroom_points_to_cross_domain": (
            compounds.get("fresh_headroom_direction") == "cross_domain_detection_exp4386"
        ),
        "cross_domain_generalizes": (
            cross_domain.get("detector_generalizes_cross_domain") is True
        ),
        "gap4_arc_ci_lower_gt_chance": _gap4_arc_domain_generalizes(cross_domain),
        "gap4_arc_selection_headroom_positive": _gap4_arc_domain_generalizes(
            cross_domain
        ),
        "cross_domain_verifier_non_oracle": (
            cross_domain.get("verifier_is_oracle") is False
        ),
        "e3_deeper_new_levels_positive": (
            isinstance(e3_deeper.get("new_levels_reproduced"), int)
            and e3_deeper.get("new_levels_reproduced", 0) > 0
        ),
        "e3_tails_new_levels_positive": (
            isinstance(e3_tails.get("new_levels_reproduced"), int)
            and e3_tails.get("new_levels_reproduced", 0) > 0
        ),
        "e3_total_levels_stable_34": (
            isinstance(deeper_total, int)
            and isinstance(tails_total, int)
            and deeper_total >= 34
            and tails_total >= 34
        ),
        "e3_verifier_is_oracle": (
            e3_deeper.get("verifier_is_oracle") is True
            and e3_tails.get("verifier_is_oracle") is True
        ),
    }


def select_flagged_for_v406(outcomes: Mapping[str, bool]) -> str:
    """Choose the .406 flag from the .405 fork outcomes."""

    if outcomes.get("detector_localization_actionable") and not outcomes.get(
        "biprm_abstention_no_useful_point"
    ):
        return PVD_SELECTIVE_FLAGGED_FOR_V406
    if (
        outcomes.get("biprm_localization_null")
        and outcomes.get("detector_compounds")
        and outcomes.get("compounding_ci_lower_gt_zero")
        and outcomes.get("cross_domain_generalizes")
        and outcomes.get("gap4_arc_ci_lower_gt_chance")
        and outcomes.get("cross_domain_verifier_non_oracle")
    ):
        return DEFAULT_FLAGGED_FOR_V406
    if outcomes.get("cross_domain_generalizes") and outcomes.get(
        "gap4_arc_ci_lower_gt_chance"
    ):
        return MULTI_DOMAIN_CALIBRATION_FLAGGED_FOR_V406
    if outcomes.get("e3_deeper_new_levels_positive") or outcomes.get(
        "e3_tails_new_levels_positive"
    ):
        return MIND_STUDIO_GAP_REPAIR_FLAGGED_FOR_V406
    return BIPRM_BASELINE_ONLY_FLAGGED_FOR_V406


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]] | None = None,
    flagged_for_v406: str = DEFAULT_FLAGGED_FOR_V406,
    out_of_band_flagged: Sequence[Mapping[str, str]] | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build the deterministic Exp 4387 mapping artifact."""

    return {
        "honest_verdict": honest_verdict,
        "methods_mapped": [
            dict(method) for method in (methods_mapped or DEFAULT_METHODS_MAPPED)
        ],
        "flagged_for_v406": flagged_for_v406,
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
    """Validate the Exp 4387 artifact before it can be written to disk."""

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
        raise ValueError("field_principles must match REQ-REPORT-4387")

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

    flagged = artifact["flagged_for_v406"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v406 must be non-empty")
    if flagged not in ALLOWED_FLAGGED_FOR_V406:
        raise ValueError("flagged_for_v406 must be conditioned on the .405 outcomes")

    _validate_out_of_band(artifact["out_of_band_flagged"])


def validate_studying_section(section: str) -> None:
    """Check that the research-studying entry keeps citations and outcome context."""

    required_phrases = [
        "flagged_for_v406",
        "out_of_band_flagged",
        "network precondition passed",
        "sweep_clusters.py",
        "sweep_semscholar.py",
        "WebSearch/WebFetch",
        "/deep-research",
        "not invoked",
        "complete: clean_powered_null_bidirectional_not_actionable",
        "detector_localization_actionable=false",
        "localization_delta_ci95=[0.0, 0.0]",
        "useful_operating_point=null",
        "blocked_gate_check_failed",
        "success: detector_compounds_heldout_localization_f1",
        "detector_compounds=true",
        "compounding_delta_ci95=[0.003396, 0.032772]",
        "success: detector_generalizes_cross_domain_non_fover",
        "detector_generalizes_cross_domain=true",
        "detection_auroc=0.963317",
        "auroc_ci95=[0.922285, 0.990662]",
        "selection_headroom=0.129",
        "complete_e3_deeper_partial",
        "complete_e3_ar25_ka59_ft09_partial",
        "new_levels_reproduced=0",
        "reproducible_total_levels=34",
        "operator-owned",
        "NOT auto-run",
        DEFAULT_FLAGGED_FOR_V406,
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
    marker = "## 2026-06-18 Exp 4387"
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

    flagged_for_v406 = select_flagged_for_v406(DEFAULT_V405_OUTCOMES)
    artifact = build_artifact(flagged_for_v406=flagged_for_v406)
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
    root_override = os.environ.get("CARNOT_EXP4387_ROOT")
    repo_root = Path(root_override) if root_override else Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        artifact_path=repo_root / "results/experiment_4387_sota_ingestion_v406.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
