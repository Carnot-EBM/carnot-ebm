"""Exp 4398 SOTA ingestion for the .406 outcomes feeding .407.

Spec refs: REQ-REPORT-4398, SCENARIO-REPORT-4398.

This module writes a planning artifact, not a benchmark result. It maps the
`.406` fork outcomes onto cited SOTA methods for `.407`: process-data
localization won but was later quarantined as position/template-confounded,
simple localizer self-learning saturated, multi-domain calibration produced a
false deployable contract, and ARC E3 stayed at zero new reproduced levels.
The single strongest .407 flag therefore moves to real, intervention-style
first-error evidence that can deconfound the localizer before the planner treats
it as an actionable capability.
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
        "flagged_for_v407",
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
        "v406_outcome_conditioning",
        "carnot_stack_mapping",
        "failure_mode",
        "experiment_mapping",
    }
)
REQUIRED_OUT_OF_BAND_FIELDS = frozenset(
    {"name", "arxiv_id_or_url", "url", "reason", "owner_boundary"}
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = "complete: sota_ingestion_v407_mapped"
DEFAULT_FLAGGED_FOR_V407 = "intervention_active_real_first_error_deconfounding_v407"
ACTIVE_LEARNING_FLAGGED_FOR_V407 = "active_uncertainty_first_error_sampling_v407"
SEMANTIC_CALIBRATION_FLAGGED_FOR_V407 = "semantic_calibrated_detector_repair_v407"
MIND_STUDIO_FLAGGED_FOR_V407 = "mind_studio_mechanic_gap_tests_v407"
REASONING_MONITOR_FLAGGED_FOR_V407 = "reasoning_safety_monitor_baseline_v407"
ALLOWED_FLAGGED_FOR_V407 = frozenset(
    {
        DEFAULT_FLAGGED_FOR_V407,
        ACTIVE_LEARNING_FLAGGED_FOR_V407,
        SEMANTIC_CALIBRATION_FLAGGED_FOR_V407,
        MIND_STUDIO_FLAGGED_FOR_V407,
        REASONING_MONITOR_FLAGGED_FOR_V407,
    }
)
DEFAULT_RANDOM_SEED = 4398

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. Records ingestion completed with verifiable "
        "citations (or blocked_network_unavailable)."
    ),
    "methods_mapped": (
        "Each method MUST carry a real, VERIFIED arXiv ID/URL (no citation = "
        "fabrication) + a one-line .407 experiment mapping + the failure mode "
        "+ the .406-outcome conditioning."
    ),
    "flagged_for_v407": (
        "Closes discover->ingest->plan: names the single strongest method for "
        "the .407 planner, conditioned on the .406 outcomes."
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
    "2601.14209": "https://arxiv.org/abs/2601.14209",
    "2603.25412": "https://arxiv.org/abs/2603.25412",
    "2504.10559": "https://arxiv.org/abs/2504.10559",
    "2602.07842": "https://arxiv.org/abs/2602.07842",
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

DEFAULT_V406_OUTCOMES = {
    "process_localizer_win": True,
    "fover_localization_strong": True,
    "gap4_arc_localization_positive": True,
    "process_localizer_verifier_non_oracle": True,
    "localizer_skeptic_confounded": True,
    "localizer_win_genuine": False,
    "position_only_ties": True,
    "template_ablation_no_drop": True,
    "self_learning_saturated_null": True,
    "localizer_compounds": False,
    "compounding_delta_zero": True,
    "positive_control_passed": True,
    "self_learning_verifier_non_oracle": True,
    "multi_domain_contract_false": True,
    "nonfover_domains_above_chance": True,
    "calibration_transfer_false": True,
    "cross_domain_verifier_non_oracle": True,
    "e3_deeper_new_levels_positive": False,
    "e3_tails_new_levels_positive": False,
    "e3_total_levels_stable_34": True,
    "e3_verifier_is_oracle": True,
}

DEFAULT_METHODS_MAPPED = [
    {
        "name": "InT self-proposed interventions for first-error credit assignment",
        "arxiv_id_or_url": "2601.14209",
        "url": "https://arxiv.org/abs/2601.14209",
        "source_verification": (
            "Verified by arXiv abs URL, focused Semantic Scholar query, and "
            "low-concurrency WebSearch/WebFetch on 2026-06-18."
        ),
        "track": "real first-error intervention data for localizer deconfounding",
        "v406_outcome_conditioning": (
            "Exp 4392 reports localizer_beats_ensemble_baseline=true, but Exp "
            "4393 reports localizer_win_is_genuine=false, "
            "beats_position_only_baseline=false, and template_ablation_drop=0.0."
        ),
        "carnot_stack_mapping": (
            "Collect verifier-checked real traces where a single-step "
            "intervention redirects a failed trajectory, stratify by first-error "
            "position and template family, and train/evaluate the localizer on "
            "held-out intervention families."
        ),
        "failure_mode": (
            "Interventions can become post-hoc rationalizations or drift into "
            "generator SFT/RL; .407 must use them as offline labels only and "
            "require executable/reference checks plus position-diverse splits."
        ),
        "experiment_mapping": (
            ".407: replace the quarantined A1 headline with real intervention "
            "first-error data that tests whether localization survives "
            "position/template deconfounding."
        ),
    },
    {
        "name": "Reasoning Safety Monitor typed step-localization taxonomy",
        "arxiv_id_or_url": "2603.25412",
        "url": "https://arxiv.org/abs/2603.25412",
        "source_verification": (
            "Verified by arXiv abs URL and low-concurrency WebSearch/WebFetch "
            "on 2026-06-18 for step-level reasoning monitor localization."
        ),
        "track": "typed first-error discriminator and adversarial localizer audit",
        "v406_outcome_conditioning": (
            "Exp 4392 logs a GAP-4 ARC missing discriminator, and Exp 4393 "
            "quarantines the localizer because a position-only baseline ties A1."
        ),
        "carnot_stack_mapping": (
            "Add a typed reasoning-vulnerability taxonomy monitor as an "
            "independent label/audit channel, compare it with localizer onset "
            "scores, and require disagreement analysis before accepting an "
            "earliest-error claim."
        ),
        "failure_mode": (
            "Prompt monitors can overfit their taxonomy, miss ARC mechanics, or "
            "false-positive on correct but unusual reasoning; .407 must report "
            "per-type false positives and domain competence."
        ),
        "experiment_mapping": (
            ".407: use taxonomy-typed monitor labels to expose whether the "
            "localizer learned real first-error structure or only template and "
            "position artifacts."
        ),
    },
    {
        "name": "ActPRM active learning for process reward model training",
        "arxiv_id_or_url": "2504.10559",
        "url": "https://arxiv.org/abs/2504.10559",
        "source_verification": (
            "Verified by arXiv abs URL, Semantic Scholar focused pass, and "
            "low-concurrency WebSearch/WebFetch on 2026-06-18."
        ),
        "track": "active localizer self-learning after saturated simple growth",
        "v406_outcome_conditioning": (
            "Exp 4396 reports complete: clean_saturated_null_localizer with "
            "localizer_compounds=false, compounding_delta_ci95=[0.0, 0.0], and "
            "positive_control_passed=true."
        ),
        "carnot_stack_mapping": (
            "Replace size-only corpus growth with uncertainty, disagreement, "
            "and first-error-position diversity sampling, then measure held-out "
            "localization-F1 against a no-learning baseline."
        ),
        "failure_mode": (
            "Uncertainty sampling can select the same artifact family that "
            "fooled A1; .407 must keep template-family holdouts and a "
            "position-only control in the active-learning loop."
        ),
        "experiment_mapping": (
            ".407: retry localizer compounding only with active selection of "
            "uncertain and position-diverse real traces, not more of the same "
            "corpus stream."
        ),
    },
    {
        "name": "Semantic Confidence Aggregation for multi-answer calibration",
        "arxiv_id_or_url": "2602.07842",
        "url": "https://arxiv.org/abs/2602.07842",
        "source_verification": (
            "Verified by arXiv abs/html URL and low-concurrency WebSearch/WebFetch "
            "on 2026-06-18 for multi-answer confidence calibration."
        ),
        "track": "multi-domain detector calibration and selective prediction",
        "v406_outcome_conditioning": (
            "Exp 4397 reports detection_calibrated_multi_domain=false while "
            "non-FoVer domains remain above chance and code_humaneval is "
            "underpowered at n=100."
        ),
        "carnot_stack_mapping": (
            "Aggregate confidence across semantically equivalent valid answers "
            "or verifier-success modes before Platt/risk-coverage reporting, "
            "with per-domain base-rate and answer-cardinality metadata."
        ),
        "failure_mode": (
            "MACE/SCA is QA-calibration work, not a step-verifier theorem; "
            "semantic grouping can hide wrong verifier modes unless checked "
            "against executable labels and leave-domain-out calibration."
        ),
        "experiment_mapping": (
            ".407: repair the false calibrated multi-domain detector contract "
            "by separating calibration failure from multi-valid-output/base-rate "
            "effects before any deployable abstention claim."
        ),
    },
    {
        "name": "Mind-Studio executable world models with lookahead evaluation",
        "arxiv_id_or_url": "2606.16070",
        "url": "https://arxiv.org/abs/2606.16070",
        "source_verification": (
            "Reverified by arXiv abs/html URL and low-concurrency WebSearch/WebFetch "
            "on 2026-06-18 for the ARC E3 lookahead-fidelity track."
        ),
        "track": "ARC E3 executable-world-model mechanic-gap tests",
        "v406_outcome_conditioning": (
            "Exp 4394 and Exp 4395 both report new_levels_reproduced=0, "
            "reproducible_total_levels=34, and verifier_is_oracle=true."
        ),
        "carnot_stack_mapping": (
            "Keep entropy-selected traces and K-step lookahead fidelity, but "
            "turn ar25/ka59/ft09 mechanic gaps into explicit executable unit "
            "tests before planning or claiming new reproduced levels."
        ),
        "failure_mode": (
            "Mind-Studio targets partially observable games rather than ARC "
            "directly, and E3 remains oracle-grounded; .407 must report mechanic "
            "test pass/fail separately from any solve claim."
        ),
        "experiment_mapping": (
            ".407: preserve E3 as the ARC north star with mechanic-gap tests, "
            "but keep the single planner flag on deconfounding the real "
            "first-error localizer."
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

STUDYING_SECTION = """## 2026-06-18 Exp 4398 - .406 fork SOTA ingestion ingested

**Status:** INGESTED into `results/experiment_4398_sota_ingestion_v407.json`.

**Preconditions:** network precondition passed via Hugging Face reachability and
arXiv/WebFetch verification. If that check had failed, the only honest artifact
would have been `honest_verdict=blocked_network_unavailable`.

**Reliable-channel provenance:** `scripts/sweep_clusters.py` emitted focused
verifier/process-reward and world-model arXiv discovery URLs. `scripts/sweep_semscholar.py`
was run on focused verifiable-process-data, OOD verifier calibration, selective
prediction, and ARC E3 lookahead queries; Semantic Scholar returned arXiv:2504.00891,
arXiv:2502.11520, arXiv:2603.19310, arXiv:2409.13757, arXiv:2602.03412,
arXiv:2508.04748, arXiv:2407.05693, and arXiv:2606.08728 before HTTP 429 on
the remaining focused queries. Low-concurrency WebSearch/WebFetch plus arXiv
abs/html checks verified arXiv:2601.14209, arXiv:2603.25412, arXiv:2504.10559,
arXiv:2602.07842, arXiv:2606.16070, arXiv:2605.02395, arXiv:2605.25133,
arXiv:2606.13565, and arXiv:2502.01384. The banned `/deep-research` channel was
not invoked.

**Filtered track:** .406 outcomes after verifiable-process-data first-error
localization, localizer skeptic-proofing, localizer self-learning, cross-domain
detector calibration, and ARC E3 lookahead/mechanic-gap work.

**.406 outcome conditioning:**
- Exp 4392: `success: synthetic_process_localizer_beats_ensemble_baseline`,
  `localizer_beats_ensemble_baseline=true`, FoVer `synthetic_trained_localizer=1.0`,
  GAP-4 ARC `synthetic_trained_localizer=0.692308`, and `verifier_is_oracle=false`.
  The process-data localizer is the live vehicle, but not yet the trusted headline.
- Exp 4393: `complete: a1_win_quarantined_as_artifact_confounded`,
  `localizer_win_is_genuine=false`, `beats_position_only_baseline=false`, and
  `template_ablation_drop=0.0`. The A1 win is quarantined until real,
  position-diverse first-error evidence deconfounds it.
- Exp 4396: `complete: clean_saturated_null_localizer`,
  `localizer_compounds=false`, `compounding_delta_ci95=[0.0, 0.0]`, and
  `positive_control_passed=true`. Simple corpus growth saturated; .407 needs
  active/uncertainty selection rather than more of the same stream.
- Exp 4397: `complete: calibrated_multi_domain_contract_false`,
  `detection_calibrated_multi_domain=false`, `domains_at_chance=[]`, non-FoVer
  domains above chance (`gap4_arc`, `gsm8k`, `code_humaneval`), and
  `verifier_is_oracle=false`. Detection remains alive, but calibration/base-rate
  repair is required before a deployable detector contract.
- Exp 4394/4395: `complete_e3_deeper_partial` and
  `complete_e3_ar25_ka59_ft09_partial`, both with `new_levels_reproduced=0`,
  `reproducible_total_levels=34`, and `verifier_is_oracle=true`. ARC E3 remains
  the north star, but .406 produced mechanic-gap work rather than new solves.

**Fresh-pass candidates marked ingested:**
- InT Self-Proposed Interventions, arXiv:2601.14209 - mapped to real
  first-error intervention traces that deconfound the A1 localizer.
- Reasoning Safety Monitor, arXiv:2603.25412 - mapped to typed step-localizer
  audit labels and adversarial first-error taxonomy checks.
- ActPRM active learning, arXiv:2504.10559 - mapped to active uncertainty and
  first-error-position diversity sampling after the saturated self-learning null.
- Semantic Confidence Aggregation / MACE, arXiv:2602.07842 - mapped to
  multi-answer/base-rate calibration repair after the false calibrated
  multi-domain contract.
- Mind-Studio executable world models with lookahead evaluation,
  arXiv:2606.16070 - carried as E3 mechanic-gap tests after zero new .406
  reproduced levels.

Carried baseline context: Controllable and Verifiable Process Data Synthesis,
arXiv:2605.02395, and Prover-Verifier Deliberation, arXiv:2605.25133, remain
verified supports, but the .406 outcomes make real intervention/deconfounding
the single .407 flag.

out_of_band_flagged:
- A2D2 adaptive any-length discrete diffusion, arXiv:2606.13565 -
  operator-owned verifier-as-reward generator training, NOT auto-run in-loop.
- SEPO score-entropy policy optimization, arXiv:2502.01384 - operator-owned
  verifier-as-reward generator training, NOT auto-run in-loop.

flagged_for_v407: intervention_active_real_first_error_deconfounding_v407

Flagged for .407: `intervention_active_real_first_error_deconfounding_v407`

random_seed=4398

**Bottom line for the .407 roadmap:** build on the process-data localizer only
after deconfounding it. The single strongest .407 method is InT-style real
first-error intervention evidence, actively selected for position/template
diversity and audited by typed reasoning-monitor labels. Calibration repair and
E3 mechanic-gap tests stay live supporting tracks. A2D2 and SEPO stay out of
band for operator-owned verifier-as-reward generator training.
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


def _ci_equal_zero(ci95: object) -> bool:
    return isinstance(ci95, Sequence) and not isinstance(ci95, str) and ci95 == [0.0, 0.0]


def _domain_localization_beats_baseline(
    artifact: Mapping[str, Any], domain: str
) -> bool:
    rows = artifact.get("localization_f1_by_domain")
    if not isinstance(rows, Mapping):
        return False
    row = rows.get(domain)
    if not isinstance(row, Mapping):
        return False
    score = row.get("synthetic_trained_localizer")
    baseline = row.get("ensemble_baseline_0096")
    return (
        isinstance(score, (int, float))
        and isinstance(baseline, (int, float))
        and score > baseline
        and _ci_lower_gt_zero(row.get("delta_ci95"))
    )


def _nonfover_domains_above_chance(artifact: Mapping[str, Any]) -> bool:
    rows = artifact.get("detection_by_domain")
    if not isinstance(rows, Sequence) or isinstance(rows, str):
        return False
    count = 0
    for row in rows:
        if not isinstance(row, Mapping) or row.get("domain") == "fover":
            continue
        n = row.get("n")
        if _ci_lower_gt_chance(row.get("auroc_ci95")) and isinstance(n, int) and n >= 100:
            count += 1
    return count >= 2


def _calibration_transfer_false(artifact: Mapping[str, Any]) -> bool:
    rows = artifact.get("detection_by_domain")
    if not isinstance(rows, Sequence) or isinstance(rows, str):
        return False
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        uncalibrated = row.get("ece_uncalibrated")
        calibrated = row.get("ece_lodo_calibrated")
        if (
            isinstance(uncalibrated, (int, float))
            and isinstance(calibrated, (int, float))
            and calibrated > uncalibrated
        ):
            return True
    return False


def extract_v406_outcomes(
    *,
    localizer: Mapping[str, Any],
    skeptic: Mapping[str, Any],
    self_learning: Mapping[str, Any],
    calibration: Mapping[str, Any],
    e3_deeper: Mapping[str, Any],
    e3_tails: Mapping[str, Any],
) -> dict[str, bool]:
    """Extract the load-bearing .406 outcome booleans from source artifacts."""

    deeper_total = e3_deeper.get("reproducible_total_levels")
    tails_total = e3_tails.get("reproducible_total_levels")
    return {
        "process_localizer_win": (
            localizer.get("honest_verdict")
            == "success: synthetic_process_localizer_beats_ensemble_baseline"
            and localizer.get("localizer_beats_ensemble_baseline") is True
        ),
        "fover_localization_strong": _domain_localization_beats_baseline(
            localizer, "FoVer"
        ),
        "gap4_arc_localization_positive": _domain_localization_beats_baseline(
            localizer, "GAP-4 ARC"
        ),
        "process_localizer_verifier_non_oracle": (
            localizer.get("verifier_is_oracle") is False
        ),
        "localizer_skeptic_confounded": (
            skeptic.get("honest_verdict")
            == "complete: a1_win_quarantined_as_artifact_confounded"
        ),
        "localizer_win_genuine": skeptic.get("localizer_win_is_genuine") is True,
        "position_only_ties": skeptic.get("beats_position_only_baseline") is False,
        "template_ablation_no_drop": skeptic.get("template_ablation_drop") == 0.0,
        "self_learning_saturated_null": (
            self_learning.get("honest_verdict")
            == "complete: clean_saturated_null_localizer"
        ),
        "localizer_compounds": self_learning.get("localizer_compounds") is True,
        "compounding_delta_zero": _ci_equal_zero(
            self_learning.get("compounding_delta_ci95")
        ),
        "positive_control_passed": (
            self_learning.get("positive_control_passed") is True
        ),
        "self_learning_verifier_non_oracle": (
            self_learning.get("verifier_is_oracle") is False
        ),
        "multi_domain_contract_false": (
            calibration.get("honest_verdict")
            == "complete: calibrated_multi_domain_contract_false"
            and calibration.get("detection_calibrated_multi_domain") is False
        ),
        "nonfover_domains_above_chance": _nonfover_domains_above_chance(
            calibration
        ),
        "calibration_transfer_false": _calibration_transfer_false(calibration),
        "cross_domain_verifier_non_oracle": (
            calibration.get("verifier_is_oracle") is False
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


def select_flagged_for_v407(outcomes: Mapping[str, bool]) -> str:
    """Choose the .407 flag from the .406 fork outcomes."""

    if (
        outcomes.get("process_localizer_win")
        and outcomes.get("localizer_skeptic_confounded")
        and not outcomes.get("localizer_win_genuine")
    ):
        return DEFAULT_FLAGGED_FOR_V407
    if (
        not outcomes.get("localizer_compounds")
        and outcomes.get("positive_control_passed")
    ):
        return ACTIVE_LEARNING_FLAGGED_FOR_V407
    if outcomes.get("multi_domain_contract_false") and outcomes.get(
        "nonfover_domains_above_chance"
    ):
        return SEMANTIC_CALIBRATION_FLAGGED_FOR_V407
    if outcomes.get("e3_deeper_new_levels_positive") or outcomes.get(
        "e3_tails_new_levels_positive"
    ):
        return MIND_STUDIO_FLAGGED_FOR_V407
    return REASONING_MONITOR_FLAGGED_FOR_V407


def build_artifact(
    *,
    methods_mapped: Sequence[Mapping[str, str]] | None = None,
    flagged_for_v407: str = DEFAULT_FLAGGED_FOR_V407,
    out_of_band_flagged: Sequence[Mapping[str, str]] | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build the deterministic Exp 4398 mapping artifact."""

    return {
        "honest_verdict": honest_verdict,
        "methods_mapped": [
            dict(method) for method in (methods_mapped or DEFAULT_METHODS_MAPPED)
        ],
        "flagged_for_v407": flagged_for_v407,
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
    """Validate the Exp 4398 artifact before it can be written to disk."""

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
        raise ValueError("field_principles must match REQ-REPORT-4398")

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

    flagged = artifact["flagged_for_v407"]
    if not isinstance(flagged, str) or not flagged.strip():
        raise ValueError("flagged_for_v407 must be non-empty")
    if flagged not in ALLOWED_FLAGGED_FOR_V407:
        raise ValueError("flagged_for_v407 must be conditioned on the .406 outcomes")

    _validate_out_of_band(artifact["out_of_band_flagged"])


def validate_studying_section(section: str) -> None:
    """Check that the research-studying entry keeps citations and outcome context."""

    required_phrases = [
        "flagged_for_v407",
        "out_of_band_flagged",
        "network precondition passed",
        "sweep_clusters.py",
        "sweep_semscholar.py",
        "WebSearch/WebFetch",
        "/deep-research",
        "not invoked",
        "success: synthetic_process_localizer_beats_ensemble_baseline",
        "localizer_beats_ensemble_baseline=true",
        "synthetic_trained_localizer=1.0",
        "synthetic_trained_localizer=0.692308",
        "complete: a1_win_quarantined_as_artifact_confounded",
        "localizer_win_is_genuine=false",
        "beats_position_only_baseline=false",
        "template_ablation_drop=0.0",
        "complete: clean_saturated_null_localizer",
        "localizer_compounds=false",
        "compounding_delta_ci95=[0.0, 0.0]",
        "positive_control_passed=true",
        "complete: calibrated_multi_domain_contract_false",
        "detection_calibrated_multi_domain=false",
        "domains_at_chance=[]",
        "gap4_arc",
        "gsm8k",
        "code_humaneval",
        "complete_e3_deeper_partial",
        "complete_e3_ar25_ka59_ft09_partial",
        "new_levels_reproduced=0",
        "reproducible_total_levels=34",
        "operator-owned",
        "NOT auto-run",
        DEFAULT_FLAGGED_FOR_V407,
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
    marker = "## 2026-06-18 Exp 4398"
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

    flagged_for_v407 = select_flagged_for_v407(DEFAULT_V406_OUTCOMES)
    artifact = build_artifact(flagged_for_v407=flagged_for_v407)
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
    root_override = os.environ.get("CARNOT_EXP4398_ROOT")
    repo_root = Path(root_override) if root_override else Path(__file__).resolve().parents[2]
    artifact = write_outputs(
        artifact_path=repo_root / "results/experiment_4398_sota_ingestion_v407.json",
        studying_path=repo_root / "research-studying.md",
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
