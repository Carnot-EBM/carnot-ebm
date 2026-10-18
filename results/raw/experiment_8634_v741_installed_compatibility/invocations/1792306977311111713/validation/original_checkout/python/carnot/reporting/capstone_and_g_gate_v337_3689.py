"""Exp 3689 v337 capstone and G-gate synthesis.

Spec: REQ-PUBLISH-041, SCENARIO-PUBLISH-041.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3689_capstone_and_g_gate_v337.json")
RANDOM_SEED = 3689
FROZEN_FOVER_HEADLINE_AUROC = 0.9131
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts "
    "(principle: reads the gate script + artifacts; no live inference)."
)

UPSTREAM_ARTIFACTS: Mapping[str, Path] = {
    "exp3680": Path("results/experiment_3680_dependency_aware_dual_condition_integrity.json"),
    "exp3681": Path("results/experiment_3681_g2_reproducer_prep_operator_refreeze_package.json"),
    "exp3682": Path("results/experiment_3682_discrimination_vs_selection_gap.json"),
    "exp3683": Path("results/experiment_3683_detector_code_operating_point.json"),
    "exp3684": Path("results/experiment_3684_product_value_vs_self_certainty.json"),
    "exp3685": Path("results/experiment_3685_fr11_continuous_self_learning_v11.json"),
}

DEPENDENCY_STATUSES = {
    "g1_rigor_confirmed_package_ready",
    "g1_rigor_confirmed_package_blocked",
    "no_significant_gain_under_protocol",
    "flagged_still",
}
REFREEZE_STATUSES = {"ready_for_operator", "not_prepared_candidate_unconfirmed"}
SELECTION_VERDICTS = {
    "closed_by_per_question_calibration",
    "fundamental_decoupling",
    "not_measured",
}
DETECTOR_CODE_STATUSES = {"recovered_math_and_code", "math_only_earned", "not_measured"}
PRODUCT_VALUE_STATUSES = {
    "robust_beats_self_certainty",
    "narrowed_collapses_vs_self_certainty",
    "not_measured",
}
FR11_RESULTS = {
    "drift_aware_online_dependency_aware_recovers_no_collapse_quality_maintained",
    "collapse_or_quality_regression",
    "not_measured",
}
VERIFIER_SCOPES = {
    "math_plus_code_discrimination_facts_retired_selection_closed",
    "math_plus_code_discrimination_facts_retired_selection_earned_negative",
    "math_plus_code_discrimination_facts_retired_selection_not_measured",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "dependency_aware_g1_candidate_status",
    "refreeze_package_status",
    "selection_gap_verdict",
    "detector_code_operating_point",
    "product_value_vs_self_certainty",
    "fr11_v11_result",
    "verifier_value_scope",
    "g1",
    "g2",
    "g3",
    "g4",
    "paper_ready",
    "frozen_headline_unchanged",
    "unmet_gates",
    "p01_status",
    "facts_generalization_retired",
    "trained_judge_ood_retired",
    "paper_v6_safe_claims",
    "paper_v6_forbidden_claims",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix for reconciler classification.",
    "inference_substrate": (
        "aggregation_from_upstream_artifacts (principle: reads the gate script + "
        "artifacts; no live inference)."
    ),
    "dependency_aware_g1_candidate_status": (
        "g1_rigor_confirmed_package_ready / g1_rigor_confirmed_package_blocked / "
        "no_significant_gain_under_protocol / flagged_still -- the status of "
        "the .337 headline-advancement lead (exp3680/3681)."
    ),
    "refreeze_package_status": (
        "ready_for_operator / not_prepared_candidate_unconfirmed -- whether "
        "exp3681 produced an operator-ready re-freeze package; the frozen "
        "0.9131 stays frozen regardless."
    ),
    "selection_gap_verdict": (
        "closed_by_per_question_calibration / fundamental_decoupling / "
        "not_measured -- the discrimination-vs-selection diagnosis (exp3682)."
    ),
    "detector_code_operating_point": (
        "recovered_math_and_code / math_only_earned / not_measured -- did the "
        "code operating point harden (exp3683)?"
    ),
    "product_value_vs_self_certainty": (
        "robust_beats_self_certainty / narrowed_collapses_vs_self_certainty / "
        "not_measured -- does product value survive the stronger baseline (exp3684)?"
    ),
    "fr11_v11_result": (
        "Whether drift-aware online dependency-aware weighting recovered without "
        "collapse (exp3685)."
    ),
    "verifier_value_scope": (
        "The scoped product claim after .337: math+code DISCRIMINATION (frozen "
        "headline), facts RETIRED, selection earned-negative (or closed if "
        "exp3682 found a fix)."
    ),
    "g1": "Headline measured (FoVer 0.9131, 5-seed, CI, adversarial-clean).",
    "g2": "Independently reproduced (CI runner 26725185125).",
    "g3": "Prose narrowing-clean.",
    "g4": "Numbers trace to primary artifacts.",
    "paper_ready": "G1 and G2 and G3 and G4 -- must remain true; the milestone does not regress the gate.",
    "frozen_headline_unchanged": (
        "True iff the publication gate still reads 0.9131 -- a "
        "dependency-aware win is a candidate, never a silent swap."
    ),
    "unmet_gates": "Report which gates are unmet, not a count (publication_blocker_count is retired).",
    "p01_status": "P0.1 stays honest-negative; do not re-assert a positive.",
    "facts_generalization_retired": (
        "Records facts-generalization as RETIRED (exp3670 same-verdict on REAL RAGTruth)."
    ),
    "trained_judge_ood_retired": (
        "Records the trained-judge-OOD hypothesis as retired (exp3659 same-verdict)."
    ),
    "paper_v6_safe_claims": "Narrowing-clean claims.",
    "paper_v6_forbidden_claims": (
        "Overclaims to avoid (including: do NOT cite the dependency-aware win "
        "as the headline until re-frozen + re-reproduced)."
    ),
    "cited_upstream_artifacts": "sha256 provenance (G4).",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Drift detection.",
    "duration_s": "Plausibility floor.",
}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    gate_data: Mapping[str, Any] | None = None,
    summary_records: Sequence[Mapping[str, Any]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """Build the Exp 3689 terminal artifact from upstream result files."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    gate = dict(gate_data) if gate_data is not None else load_publication_gate(root_path)
    summaries = (
        [dict(record) for record in summary_records]
        if summary_records is not None
        else run_summarize_artifacts(root_path)
    )
    upstreams = {
        name: _read_optional_json_object(root_path / rel_path)
        for name, rel_path in UPSTREAM_ARTIFACTS.items()
    }
    flagged = {name: _is_flagged_adversarial(payload) for name, payload in upstreams.items()}

    dependency = _dependency_aware_status(
        upstreams["exp3680"],
        upstreams["exp3681"],
        exp3680_flagged=flagged["exp3680"],
        exp3681_flagged=flagged["exp3681"],
    )
    selection = _selection_gap_status(upstreams["exp3682"], flagged=flagged["exp3682"])
    detector = _detector_code_status(upstreams["exp3683"], flagged=flagged["exp3683"])
    product = _product_value_status(upstreams["exp3684"], flagged=flagged["exp3684"])
    fr11 = _fr11_v11_status(upstreams["exp3685"], flagged=flagged["exp3685"])
    scope = _verifier_scope(selection["verdict"])

    g1 = _gate_pass(gate, "G1")
    g2 = _gate_pass(gate, "G2")
    g3 = _gate_pass(gate, "G3")
    g4 = _gate_pass(gate, "G4")
    paper_ready = bool(gate.get("paper_ready") is True and g1 and g2 and g3 and g4)
    frozen_headline_unchanged = _frozen_headline_unchanged(gate)
    finished = time.perf_counter() if now_s is None else float(now_s)
    duration_s = round(max(0.0, finished - start), 6)

    artifact: JsonDict = {
        "honest_verdict": (
            "complete: capstone_v337_dependency_aware_"
            f"{dependency['status']}_selection_{selection['verdict']}_"
            f"detector_code_{detector['status']}_paper_ready_{str(paper_ready).lower()}_"
            f"{'frozen_headline_unchanged' if frozen_headline_unchanged else 'frozen_headline_changed'}"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "dependency_aware_g1_candidate_status": dependency["status"],
        "refreeze_package_status": dependency["refreeze_package_status"],
        "selection_gap_verdict": selection["verdict"],
        "detector_code_operating_point": detector["status"],
        "product_value_vs_self_certainty": product["status"],
        "fr11_v11_result": fr11["result"],
        "verifier_value_scope": scope,
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "g4": g4,
        "paper_ready": paper_ready,
        "frozen_headline_unchanged": frozen_headline_unchanged,
        "unmet_gates": list(gate.get("unmet_gates") or []),
        "p01_status": "honest-negative",
        "facts_generalization_retired": True,
        "trained_judge_ood_retired": True,
        "paper_v6_safe_claims": _safe_claims(
            dependency_status=dependency["status"],
            refreeze_status=dependency["refreeze_package_status"],
            selection_verdict=selection["verdict"],
            detector_status=detector["status"],
            product_status=product["status"],
            fr11_result=fr11["result"],
        ),
        "paper_v6_forbidden_claims": _forbidden_claims(
            flagged_upstreams=flagged,
            dependency=dependency,
            selection=selection,
        ),
        "cited_upstream_artifacts": _cited_upstreams(root_path, upstreams, flagged),
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "field_principles": dict(FIELD_PRINCIPLES),
        "frozen_fover_headline_auroc": FROZEN_FOVER_HEADLINE_AUROC,
        "dependency_aware_candidate": dependency,
        "refreeze_package": dependency["refreeze_package"],
        "selection_gap": selection,
        "detector_code": detector,
        "product_value": product,
        "fr11_v11": fr11,
        "publication_gate": _gate_details(gate),
        "summarized_upstream_artifacts": summaries,
        "flagged_upstream_artifacts_excluded": [
            str(UPSTREAM_ARTIFACTS[name]) for name, is_flagged in flagged.items() if is_flagged
        ],
        "source_artifacts": [
            str(path) for name, path in UPSTREAM_ARTIFACTS.items() if upstreams.get(name)
        ],
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
    }
    artifact["reproducibility_checksum"] = _payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    gate_data: Mapping[str, Any] | None = None,
    summary_records: Sequence[Mapping[str, Any]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3689 artifact."""

    root_path = Path(root)
    out_path = _repo_path(root_path, Path(output_path))
    artifact = build_artifact(
        root_path,
        gate_data=gate_data,
        summary_records=summary_records,
        started_s=started_s,
        now_s=now_s,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 3689 schema and publication-gate invariants."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be a mapping")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    if not str(artifact.get("honest_verdict") or "").startswith("complete:"):
        raise ValueError("honest_verdict must start with complete:")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate is not the Exp 3689 aggregation substrate")
    if artifact.get("dependency_aware_g1_candidate_status") not in DEPENDENCY_STATUSES:
        raise ValueError("dependency_aware_g1_candidate_status is unsupported")
    if artifact.get("refreeze_package_status") not in REFREEZE_STATUSES:
        raise ValueError("refreeze_package_status is unsupported")
    if artifact.get("selection_gap_verdict") not in SELECTION_VERDICTS:
        raise ValueError("selection_gap_verdict is unsupported")
    if artifact.get("detector_code_operating_point") not in DETECTOR_CODE_STATUSES:
        raise ValueError("detector_code_operating_point is unsupported")
    if artifact.get("product_value_vs_self_certainty") not in PRODUCT_VALUE_STATUSES:
        raise ValueError("product_value_vs_self_certainty is unsupported")
    if artifact.get("fr11_v11_result") not in FR11_RESULTS:
        raise ValueError("fr11_v11_result is unsupported")
    if artifact.get("verifier_value_scope") not in VERIFIER_SCOPES:
        raise ValueError("verifier_value_scope is outside the allowed scoped claim set")
    for gate in ("g1", "g2", "g3", "g4"):
        if artifact.get(gate) is not True:
            raise ValueError(f"{gate} must be true")
    if artifact.get("paper_ready") is not True:
        raise ValueError("paper_ready must remain true for this capstone")
    if artifact.get("frozen_headline_unchanged") is not True:
        raise ValueError("frozen_headline_unchanged must remain true")
    if not isinstance(artifact.get("unmet_gates"), list):
        raise ValueError("unmet_gates must be a list")
    if artifact.get("p01_status") != "honest-negative":
        raise ValueError("p01_status must remain honest-negative")
    if artifact.get("facts_generalization_retired") is not True:
        raise ValueError("facts_generalization_retired must remain true")
    if artifact.get("trained_judge_ood_retired") is not True:
        raise ValueError("trained_judge_ood_retired must remain true")
    if not isinstance(artifact.get("paper_v6_safe_claims"), list):
        raise ValueError("paper_v6_safe_claims must be a list")
    if not isinstance(artifact.get("paper_v6_forbidden_claims"), list):
        raise ValueError("paper_v6_forbidden_claims must be a list")
    duration = artifact.get("duration_s")
    if not isinstance(duration, int | float) or isinstance(duration, bool) or float(duration) < 0.0:
        raise ValueError("duration_s must be nonnegative numeric")
    cited = artifact.get("cited_upstream_artifacts")
    if not isinstance(cited, list):
        raise ValueError("cited_upstream_artifacts must be a list")
    for item in cited:
        if not isinstance(item, Mapping) or len(str(item.get("sha256") or "")) != 64:
            raise ValueError("cited_upstream_artifacts must include sha256 provenance")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("reproducibility_checksum must be a sha256 hex string")


def load_publication_gate(root: Path) -> JsonDict:  # pragma: no cover - subprocess boundary
    """Run publication_gate.py and return its parsed JSON result."""

    completed = subprocess.run(
        [sys.executable, "scripts/publication_gate.py", "--json"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def run_summarize_artifacts(root: Path) -> list[JsonDict]:  # pragma: no cover - subprocess boundary
    """Run summarize_artifact.py for Exp 3680 through Exp 3685."""

    records: list[JsonDict] = []
    for exp_id in range(3680, 3686):
        completed = subprocess.run(
            [sys.executable, "scripts/summarize_artifact.py", str(exp_id)],
            cwd=root,
            capture_output=True,
            text=True,
        )
        records.append(
            {
                "exp": exp_id,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-1000:],
            }
        )
    return records


def _dependency_aware_status(
    exp3680: Mapping[str, Any],
    exp3681: Mapping[str, Any],
    *,
    exp3680_flagged: bool,
    exp3681_flagged: bool,
) -> JsonDict:
    auroc = _point(exp3680.get("production_auroc_dependency_aware"))
    leak_free = exp3680.get("leak_free") is True
    leak_risk = bool(auroc is not None and auroc >= 0.99 and not leak_free)
    exp3680_clean = bool(
        exp3680
        and not exp3680_flagged
        and exp3680.get("adversarial_verify_clean") is True
        and _acceptance_pass(exp3680)
        and leak_free
        and not leak_risk
        and _seed_count(exp3680.get("n_seeds") or exp3680.get("random_seeds_used")) >= 5
    )
    exp3680_confirmed = bool(exp3680_clean and exp3680.get("dependency_aware_g1_rigor_confirmed") is True)
    refreeze = _refreeze_package_status(
        exp3681,
        exp3681_flagged=exp3681_flagged,
        exp3680_confirmed=exp3680_confirmed,
    )
    if exp3680 and not exp3680_clean:
        status = "flagged_still"
    elif not exp3680_confirmed:
        status = "no_significant_gain_under_protocol"
    elif refreeze["status"] == "ready_for_operator":
        status = "g1_rigor_confirmed_package_ready"
    else:
        status = "g1_rigor_confirmed_package_blocked"
    return {
        "status": status,
        "refreeze_package_status": refreeze["status"],
        "refreeze_package": refreeze,
        "exp3680_adversarial_verify_clean": exp3680.get("adversarial_verify_clean") is True,
        "exp3680_flagged_adversarial": exp3680_flagged,
        "exp3680_acceptance_gate_passed": _acceptance_pass(exp3680),
        "exp3680_leak_free": leak_free,
        "leak_risk": leak_risk,
        "n_seeds": _seed_count(exp3680.get("n_seeds") or exp3680.get("random_seeds_used")),
        "dependency_aware_g1_rigor_confirmed": exp3680_confirmed,
        "production_auroc_dependency_aware": auroc,
        "production_auroc_carnot_current": _point(exp3680.get("production_auroc_carnot_current")),
        "frozen_headline_auroc": _point(exp3680.get("frozen_headline_auroc")),
        "production_auroc_ci": exp3680.get("production_auroc_ci"),
        "learning_contribution_dependency_aware": _point(
            exp3680.get("learning_contribution_dependency_aware")
        ),
        "dependency_vs_carnot_delta_ci": exp3680.get("dependency_vs_carnot_delta_ci"),
        "claim_boundary": (
            "headline-advancement candidate with an operator-ready re-freeze "
            "package pending operator action + CI re-reproduction"
        ),
    }


def _refreeze_package_status(
    exp3681: Mapping[str, Any],
    *,
    exp3681_flagged: bool,
    exp3680_confirmed: bool,
) -> JsonDict:
    ready = bool(
        exp3680_confirmed
        and exp3681
        and not exp3681_flagged
        and exp3681.get("reproducer_extended") is True
        and exp3681.get("existing_0_9131_reproduction_still_green") is True
        and exp3681.get("candidate_reproduction_asserts_in_ci") is True
        and exp3681.get("north_star_unmodified_assert") is True
        and exp3681.get("ci_workflow_unmodified_assert") is True
        and exp3681.get("frozen_headline_unchanged_assert") is True
        and exp3681.get("github_actions_run_triggered") is False
        and exp3681.get("publication_gate_paper_ready_before") is True
        and exp3681.get("publication_gate_paper_ready_after") is True
    )
    return {
        "status": "ready_for_operator" if ready else "not_prepared_candidate_unconfirmed",
        "exp3681_flagged_adversarial": exp3681_flagged,
        "reproducer_extended": exp3681.get("reproducer_extended") is True,
        "existing_0_9131_reproduction_still_green": (
            exp3681.get("existing_0_9131_reproduction_still_green") is True
        ),
        "candidate_reproduction_asserts_in_ci": (
            exp3681.get("candidate_reproduction_asserts_in_ci") is True
        ),
        "frozen_headline_unchanged_assert": (
            exp3681.get("frozen_headline_unchanged_assert") is True
        ),
    }


def _selection_gap_status(exp3682: Mapping[str, Any], *, flagged: bool) -> JsonDict:
    measured = bool(
        exp3682
        and not flagged
        and _acceptance_pass(exp3682)
        and exp3682.get("positive_control_valid") is True
        and _point(exp3682.get("per_candidate_auroc")) is not None
        and _point(exp3682.get("within_question_rank_corr")) is not None
        and _point(exp3682.get("flip_count")) is not None
        and float(exp3682.get("flip_count")) > 0.0
    )
    outcome = str(exp3682.get("honest_outcome") or "")
    if not measured:
        verdict = "not_measured"
    elif exp3682.get("selection_gap_closed") is True or outcome == "closed_by_per_question_calibration":
        verdict = "closed_by_per_question_calibration"
    else:
        verdict = "fundamental_decoupling"
    return {
        "verdict": verdict,
        "flagged_adversarial": flagged,
        "positive_control_valid": exp3682.get("positive_control_valid") is True,
        "per_candidate_auroc": _point(exp3682.get("per_candidate_auroc")),
        "within_question_rank_corr": _point(exp3682.get("within_question_rank_corr")),
        "flip_count": _point(exp3682.get("flip_count")),
        "selection_gap_closed": exp3682.get("selection_gap_closed") is True,
        "best_fix_method": exp3682.get("best_fix_method"),
    }


def _detector_code_status(exp3683: Mapping[str, Any], *, flagged: bool) -> JsonDict:
    measured = bool(exp3683 and not flagged and _acceptance_pass(exp3683))
    if not measured:
        status = "not_measured"
    elif exp3683.get("code_operating_point_recovered") is True:
        status = "recovered_math_and_code"
    else:
        status = "math_only_earned"
    return {
        "status": status,
        "flagged_adversarial": flagged,
        "code_operating_point_recovered": exp3683.get("code_operating_point_recovered") is True,
        "code_auroc_recalibrated": _point(exp3683.get("code_auroc_recalibrated")),
        "code_auroc_dependency_aware": _point(exp3683.get("code_auroc_dependency_aware")),
        "e2e_test_passed": exp3683.get("e2e_test_passed") is True,
    }


def _product_value_status(exp3684: Mapping[str, Any], *, flagged: bool) -> JsonDict:
    measured = bool(exp3684 and not flagged and _acceptance_pass(exp3684))
    if not measured:
        status = "not_measured"
    elif exp3684.get("ensemble_adds_value_over_self_certainty") is True:
        status = "robust_beats_self_certainty"
    else:
        status = "narrowed_collapses_vs_self_certainty"
    return {
        "status": status,
        "flagged_adversarial": flagged,
        "ensemble_adds_value_over_self_certainty": (
            exp3684.get("ensemble_adds_value_over_self_certainty") is True
        ),
        "material_win_per_domain": exp3684.get("material_win_per_domain"),
        "ensemble_minus_self_certainty_delta_ci_per_domain": exp3684.get(
            "ensemble_minus_self_certainty_delta_ci_per_domain"
        ),
    }


def _fr11_v11_status(exp3685: Mapping[str, Any], *, flagged: bool) -> JsonDict:
    if flagged or not exp3685:
        result = "not_measured"
    elif (
        exp3685.get("drift_detected_deploy_arm") is True
        and exp3685.get("collapse_detected_deploy_arm") is False
        and exp3685.get("collapse_detected_control") is True
        and exp3685.get("quality_maintained") is True
        and exp3685.get("pass_rate_vs_true_accuracy_distinct_assert") is True
        and _acceptance_pass(exp3685)
    ):
        result = "drift_aware_online_dependency_aware_recovers_no_collapse_quality_maintained"
    else:
        result = "collapse_or_quality_regression"
    return {
        "result": result,
        "flagged_adversarial": flagged,
        "drift_detected_deploy_arm": exp3685.get("drift_detected_deploy_arm") is True,
        "collapse_detected_deploy_arm": exp3685.get("collapse_detected_deploy_arm") is True,
        "collapse_detected_control": exp3685.get("collapse_detected_control") is True,
        "quality_maintained": exp3685.get("quality_maintained") is True,
        "post_drift_auroc_gain_over_static_carnot": _point(
            exp3685.get("post_drift_auroc_gain_over_static_carnot")
        ),
        "post_drift_auroc_gain_over_v10": _point(exp3685.get("post_drift_auroc_gain_over_v10")),
    }


def _verifier_scope(selection_verdict: str) -> str:
    if selection_verdict == "closed_by_per_question_calibration":
        suffix = "closed"
    elif selection_verdict == "fundamental_decoupling":
        suffix = "earned_negative"
    else:
        suffix = "not_measured"
    return f"math_plus_code_discrimination_facts_retired_selection_{suffix}"


def _safe_claims(
    *,
    dependency_status: str,
    refreeze_status: str,
    selection_verdict: str,
    detector_status: str,
    product_status: str,
    fr11_result: str,
) -> list[str]:
    claims = [
        "FoVer headline remains frozen at 0.9131 AUROC with G1-G4 satisfied.",
        "P0.1 remains honest-negative; no positive is re-asserted.",
        "Facts-generalization and trained-judge-OOD are retired and not re-asserted.",
    ]
    if dependency_status == "g1_rigor_confirmed_package_ready":
        claims.append(
            "Dependency-aware weighting is a headline-advancement candidate with an "
            "operator-ready re-freeze package pending operator action + CI re-reproduction."
        )
    elif dependency_status == "g1_rigor_confirmed_package_blocked":
        claims.append(
            "Dependency-aware weighting has a clean G1-rigor candidate, but the "
            f"operator re-freeze package is {refreeze_status}."
        )
    elif dependency_status == "no_significant_gain_under_protocol":
        claims.append("Dependency-aware weighting has no significant gain under the protocol.")
    else:
        claims.append("Dependency-aware weighting remains unclaimable under the fabrication gate.")
    if selection_verdict == "closed_by_per_question_calibration":
        claims.append("The discrimination-vs-selection gap is closed by per-question calibration.")
    elif selection_verdict == "fundamental_decoupling":
        claims.append("The selection gap remains an earned negative distinct from discrimination.")
    else:
        claims.append("The selection gap is not measured because the gated artifact is missing or flagged.")
    claims.append(f"Detector code operating point: {detector_status}.")
    claims.append(f"Product value vs self-certainty: {product_status}.")
    claims.append(f"FR-11 v11 result: {fr11_result}.")
    return claims


def _forbidden_claims(
    *,
    flagged_upstreams: Mapping[str, bool],
    dependency: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> list[str]:
    claims = [
        "Do not cite the dependency-aware win as the headline until re-frozen + re-reproduced.",
        "Do not silently replace the frozen FoVer 0.9131 headline.",
        "Do not cite the dependency-aware number unless Exp 3680 is adversarial-clean and leak-free.",
        "Do not re-assert P0.1 as positive; it remains honest-negative.",
        "Do not re-assert facts-generalization or trained-judge-OOD; both are retired.",
        "Do not read missing gated-task fields as None and synthesize a result around them.",
        "Treat any AUROC >= 0.99 as a leak unless explicit leak-free evidence is present.",
    ]
    if dependency.get("refreeze_package_status") != "ready_for_operator":
        claims.append("Do not describe the operator re-freeze package as ready.")
    if selection.get("verdict") != "closed_by_per_question_calibration":
        claims.append("Do not claim that selection value is solved by calibration.")
    flagged_paths = [
        str(UPSTREAM_ARTIFACTS[name])
        for name, is_flagged in flagged_upstreams.items()
        if is_flagged
    ]
    if flagged_paths:
        claims.append(
            "Do not cite flagged_adversarial artifacts in paper-v6 claims: "
            + ", ".join(flagged_paths)
        )
    return claims


def _cited_upstreams(
    root: Path,
    upstreams: Mapping[str, Mapping[str, Any]],
    flagged: Mapping[str, bool],
) -> list[JsonDict]:
    cited: list[JsonDict] = []
    for name, rel_path in UPSTREAM_ARTIFACTS.items():
        payload = upstreams.get(name) or {}
        if not payload or flagged.get(name) is True:
            continue
        if name == "exp3680" and not _exp3680_citable(payload):
            continue
        cited.append(
            {
                "path": str(rel_path),
                "sha256": _sha256_file(root / rel_path),
                "honest_verdict": payload.get("honest_verdict"),
            }
        )
    return cited


def _exp3680_citable(payload: Mapping[str, Any]) -> bool:
    auroc = _point(payload.get("production_auroc_dependency_aware"))
    leak_free = payload.get("leak_free") is True
    return bool(
        payload
        and payload.get("adversarial_verify_clean") is True
        and payload.get("flagged_adversarial") is not True
        and _acceptance_pass(payload)
        and leak_free
        and not (auroc is not None and auroc >= 0.99 and not leak_free)
    )


def _acceptance_pass(payload: Mapping[str, Any]) -> bool:
    gate = payload.get("acceptance_gate")
    if not isinstance(gate, Mapping):
        return False
    if gate.get("passed") is True:
        return True
    return gate.get("required_fields_present") is True


def _gate_pass(gate_data: Mapping[str, Any], gate_name: str) -> bool:
    gates = gate_data.get("gates")
    if not isinstance(gates, Mapping):
        return False
    gate = gates.get(gate_name)
    return isinstance(gate, Mapping) and gate.get("pass") is True


def _frozen_headline_unchanged(gate_data: Mapping[str, Any]) -> bool:
    gates = gate_data.get("gates")
    if not isinstance(gates, Mapping):
        return False
    joined = json.dumps(gates, sort_keys=True)
    return "0.9131" in joined and "experiment_2850_fover_dual_condition_integrity_v4" in joined


def _gate_details(gate_data: Mapping[str, Any]) -> JsonDict:
    gates = gate_data.get("gates")
    return dict(gates) if isinstance(gates, Mapping) else {}


def _point(metric: Any) -> float | None:
    if isinstance(metric, Mapping):
        return _round_or_none(metric.get("point"))
    return _round_or_none(metric)


def _round_or_none(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return round(float(value), 6)
    return None


def _seed_count(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return len(value)
    return 0


def _is_flagged_adversarial(payload: Mapping[str, Any]) -> bool:
    return payload.get("flagged_adversarial") is True


def _read_optional_json_object(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _repo_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_checksum(payload: Mapping[str, Any]) -> str:
    filtered = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
