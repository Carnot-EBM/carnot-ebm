"""Exp 3701 v338 re-freeze capstone and publication-gate recheck.

Spec: REQ-PUBLISH-3701, SCENARIO-PUBLISH-3701.
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
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct import guard
    sys.path.insert(0, str(REPO_ROOT))

from scripts import adversarial_verify  # noqa: E402


OUTPUT_REL_PATH = Path("results/experiment_3701_capstone_and_g_gate_v338.json")
RANDOM_SEED = 3701
FROZEN_FOVER_HEADLINE_AUROC = 0.9131
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
REFREEZE_READY_VERDICT = (
    "complete: refreeze_package_reemitted_clean_for_operator_frozen_headline_unchanged"
)

UPSTREAM_ARTIFACTS: Mapping[str, Path] = {
    "exp3692": Path("results/experiment_3692_refreeze_package_clean_reemit.json"),
    "exp3693": Path("results/experiment_3693_external_comparator_dependency_vs_deentangled.json"),
    "exp3694": Path("results/experiment_3694_selection_gap_proper_rediagnosis.json"),
    "exp3695": Path("results/experiment_3695_code_native_verifier.json"),
    "exp3696": Path("results/experiment_3696_reship_detector_math_plus_code.json"),
    "exp3697": Path("results/experiment_3697_fr11_continuous_self_learning_v12.json"),
}

REFREEZE_STATUSES = {"reemitted_clean_for_operator", "still_flagged", "not_prepared"}
EXTERNAL_VERDICTS = {"yes", "ties_or_loses", "not_measured"}
SELECTION_VERDICTS = {"closed_new_method", "fundamental_decoupling", "not_measured"}
CODE_DETECTOR_STATUSES = {
    "code_native_recovered_reshipped",
    "code_remains_math_only_earned",
    "not_measured",
}
FR11_V12_RESULTS = {
    "drift_reset_and_cross_session_persistence_no_collapse_quality_maintained",
    "collapse_or_quality_regression",
    "not_measured",
}
VERIFIER_SCOPES = {
    "math_plus_code_discrimination_facts_retired_selection_closed",
    "math_plus_code_discrimination_facts_retired_selection_fundamental_decoupling",
    "math_plus_code_discrimination_facts_retired_selection_not_measured",
    "math_only_discrimination_facts_retired_selection_closed",
    "math_only_discrimination_facts_retired_selection_fundamental_decoupling",
    "math_only_discrimination_facts_retired_selection_not_measured",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "adversarial_verify_clean",
    "refreeze_package_status",
    "candidate_beats_external_comparator",
    "selection_gap_verdict",
    "code_detector_status",
    "fr11_v12_result",
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
        "aggregation_from_upstream_artifacts (principle: reads the gate script "
        "+ artifacts; no live inference; no compute-bound marker so it does not "
        "false-flag like exp3689)."
    ),
    "adversarial_verify_clean": (
        "True iff this capstone passes adversarial_verify with no "
        "DURATION_TOO_SHORT/critical flag -- the exp3689 fix; a flagged capstone "
        "does not count as a clean milestone close."
    ),
    "refreeze_package_status": (
        "reemitted_clean_for_operator / still_flagged / not_prepared -- whether "
        "exp3692 produced a clean operator-actionable package; the frozen 0.9131 "
        "stays frozen regardless."
    ),
    "candidate_beats_external_comparator": (
        "yes / ties_or_loses / not_measured -- does the dependency-aware "
        "candidate beat the published external baseline (exp3693)?"
    ),
    "selection_gap_verdict": (
        "closed_new_method / fundamental_decoupling / not_measured -- the PROPER "
        "discrimination-vs-selection diagnosis (exp3694), now non-degenerate."
    ),
    "code_detector_status": (
        "code_native_recovered_reshipped / code_remains_math_only_earned / "
        "not_measured -- did a code-native signal recover the detector (exp3695/3696)?"
    ),
    "fr11_v12_result": (
        "Whether the drift-reset + cross-session-persistence self-learning "
        "succeeded without collapse (exp3697)."
    ),
    "verifier_value_scope": (
        "The scoped product claim after .338: math DISCRIMINATION (frozen "
        "headline) + code (math+code if recovered, else honestly math-only), "
        "facts RETIRED, selection (closed or fundamental per exp3694)."
    ),
    "g1": "Headline measured (FoVer 0.9131, 5-seed, CI, adversarial-clean).",
    "g2": "Independently reproduced (CI runner 26725185125).",
    "g3": "Prose narrowing-clean.",
    "g4": "Numbers trace to primary artifacts.",
    "paper_ready": (
        "G1 and G2 and G3 and G4 -- must remain true; the milestone does not "
        "regress the gate."
    ),
    "frozen_headline_unchanged": (
        "True iff the publication gate still reads 0.9131 -- a dependency-aware "
        "win is a candidate, never a silent swap."
    ),
    "unmet_gates": (
        "Report which gates are unmet, not a count (publication_blocker_count is retired)."
    ),
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
    adversarial_reports: Mapping[str, Mapping[str, Any]] | None = None,
    capstone_adversarial_verify_clean: bool = True,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """Build the Exp 3701 terminal artifact from upstream result files."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    gate = dict(gate_data) if gate_data is not None else load_publication_gate(root_path)
    summaries = _compact_summaries(
        [dict(record) for record in summary_records]
        if summary_records is not None
        else run_summarize_artifacts(root_path)
    )
    upstreams = {
        name: _read_optional_json_object(root_path / rel_path)
        for name, rel_path in UPSTREAM_ARTIFACTS.items()
    }
    reports = (
        {name: dict(report) for name, report in adversarial_reports.items()}
        if adversarial_reports is not None
        else _verify_upstreams(root_path)
    )
    flagged = {
        name: _is_uncitable_upstream(payload, reports.get(name, {}))
        for name, payload in upstreams.items()
    }

    refreeze = _refreeze_status(upstreams["exp3692"], flagged=flagged["exp3692"])
    external = _external_status(upstreams["exp3693"], flagged=flagged["exp3693"])
    selection = _selection_status(upstreams["exp3694"], flagged=flagged["exp3694"])
    code = _code_detector_status(
        upstreams["exp3695"],
        upstreams["exp3696"],
        exp3695_flagged=flagged["exp3695"],
        exp3696_flagged=flagged["exp3696"],
    )
    fr11 = _fr11_v12_status(upstreams["exp3697"], flagged=flagged["exp3697"])
    scope = _verifier_scope(code["status"], selection["verdict"])

    g1 = _gate_pass(gate, "G1")
    g2 = _gate_pass(gate, "G2")
    g3 = _gate_pass(gate, "G3")
    g4 = _gate_pass(gate, "G4")
    paper_ready = bool(gate.get("paper_ready") is True and g1 and g2 and g3 and g4)
    frozen_headline_unchanged = _frozen_headline_unchanged(gate)
    duration_s = round(max(0.0001, (time.perf_counter() if now_s is None else float(now_s)) - start), 6)

    artifact: JsonDict = {
        "honest_verdict": (
            f"complete: capstone_v338_refreeze_{refreeze['status']}_"
            f"external_{external['verdict']}_selection_{selection['verdict']}_"
            f"detector_code_{code['status']}_paper_ready_{str(paper_ready).lower()}_"
            f"{'frozen_headline_unchanged' if frozen_headline_unchanged else 'frozen_headline_changed'}"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "adversarial_verify_clean": capstone_adversarial_verify_clean,
        "refreeze_package_status": refreeze["status"],
        "candidate_beats_external_comparator": external["verdict"],
        "selection_gap_verdict": selection["verdict"],
        "code_detector_status": code["status"],
        "fr11_v12_result": fr11["result"],
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
            refreeze_status=refreeze["status"],
            external_verdict=external["verdict"],
            selection_verdict=selection["verdict"],
            code_status=code["status"],
            fr11_result=fr11["result"],
        ),
        "paper_v6_forbidden_claims": _forbidden_claims(flagged, external["leak_risk"]),
        "cited_upstream_artifacts": _cited_upstreams(root_path, upstreams, reports, flagged),
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "field_principles": dict(FIELD_PRINCIPLES),
        "frozen_fover_headline_auroc": FROZEN_FOVER_HEADLINE_AUROC,
        "refreeze_package": refreeze,
        "external_comparator": external,
        "selection_gap": selection,
        "code_detector": code,
        "fr11_v12": fr11,
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
    """Build, persist, live-verify, and re-persist the Exp 3701 artifact."""

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
    report = adversarial_verify.verify_artifact(out_path)
    artifact["adversarial_verify_report"] = report
    artifact["adversarial_verify_clean"] = adversarial_report_is_clean(report)
    artifact["reproducibility_checksum"] = _payload_checksum(artifact)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_artifact(artifact)
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 3701 schema and publication-gate invariants."""

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
        raise ValueError("inference_substrate must be the bare aggregation substrate")
    if artifact.get("adversarial_verify_clean") is not True:
        raise ValueError("adversarial_verify_clean must be true")
    if artifact.get("refreeze_package_status") not in REFREEZE_STATUSES:
        raise ValueError("refreeze_package_status is unsupported")
    if artifact.get("candidate_beats_external_comparator") not in EXTERNAL_VERDICTS:
        raise ValueError("candidate_beats_external_comparator is unsupported")
    if artifact.get("selection_gap_verdict") not in SELECTION_VERDICTS:
        raise ValueError("selection_gap_verdict is unsupported")
    if artifact.get("code_detector_status") not in CODE_DETECTOR_STATUSES:
        raise ValueError("code_detector_status is unsupported")
    if artifact.get("fr11_v12_result") not in FR11_V12_RESULTS:
        raise ValueError("fr11_v12_result is unsupported")
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
    if "model_specs" in artifact:
        raise ValueError("model_specs must not be present on aggregation capstone")
    if "target_model" in artifact:
        raise ValueError("target_model must not be present on aggregation capstone")


def adversarial_report_is_clean(report: Mapping[str, Any]) -> bool:
    """Return true only when no critical or duration flag remains."""

    for flag in list(report.get("flags") or []):
        flag_dict = dict(flag)
        if str(flag_dict.get("kind") or "") == "DURATION_TOO_SHORT":
            return False
        if str(flag_dict.get("severity") or "").lower() == "critical":
            return False
    return True


def load_publication_gate(root: Path) -> JsonDict:  # pragma: no cover - subprocess boundary
    completed = subprocess.run(
        [sys.executable, "scripts/publication_gate.py", "--json"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def run_summarize_artifacts(root: Path) -> list[JsonDict]:  # pragma: no cover - subprocess boundary
    records: list[JsonDict] = []
    for exp_id in range(3692, 3698):
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
                "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
            }
        )
    return records


def _verify_upstreams(root: Path) -> dict[str, JsonDict]:
    reports: dict[str, JsonDict] = {}
    for name, rel_path in UPSTREAM_ARTIFACTS.items():
        path = root / rel_path
        reports[name] = adversarial_verify.verify_artifact(path) if path.exists() else {"flags": []}
    return reports


def _compact_summaries(records: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "exp": record.get("exp"),
            "returncode": record.get("returncode"),
            "stdout_sha256": record.get("stdout_sha256"),
            "stderr_sha256": record.get("stderr_sha256"),
        }
        for record in records
    ]


def _refreeze_status(exp3692: Mapping[str, Any], *, flagged: bool) -> JsonDict:
    ready = bool(
        exp3692
        and not flagged
        and _payload_declares_adversarial_clean(exp3692)
        and exp3692.get("honest_verdict") == REFREEZE_READY_VERDICT
        and exp3692.get("reproducer_extended") is True
        and exp3692.get("existing_0_9131_reproduction_still_green") is True
        and exp3692.get("candidate_reproduction_asserts_in_ci") is True
        and exp3692.get("north_star_unmodified_assert") is True
        and exp3692.get("ci_workflow_unmodified_assert") is True
        and exp3692.get("frozen_headline_unchanged_assert") is True
        and exp3692.get("github_actions_run_triggered") is False
        and exp3692.get("publication_gate_paper_ready_before") is True
        and exp3692.get("publication_gate_paper_ready_after") is True
    )
    if ready:
        status = "reemitted_clean_for_operator"
    elif exp3692 and (flagged or not _payload_declares_adversarial_clean(exp3692)):
        status = "still_flagged"
    else:
        status = "not_prepared"
    return {
        "status": status,
        "flagged_or_live_critical": flagged,
        "adversarial_verify_clean": _payload_declares_adversarial_clean(exp3692),
        "operator_package_ready": ready,
    }


def _external_status(exp3693: Mapping[str, Any], *, flagged: bool) -> JsonDict:
    leak_risk = _leak_risk(exp3693)
    measured = bool(
        exp3693
        and not flagged
        and not leak_risk
        and _payload_declares_adversarial_clean(exp3693)
        and _acceptance_pass(exp3693)
        and _point(exp3693.get("dependency_aware_auroc")) is not None
        and _point(exp3693.get("external_comparator_auroc")) is not None
        and isinstance(exp3693.get("dependency_vs_external_delta_ci"), Mapping)
    )
    delta = exp3693.get("dependency_vs_external_delta_ci")
    ci = delta.get("ci95") if isinstance(delta, Mapping) else None
    positive_ci = bool(
        isinstance(ci, Sequence)
        and not isinstance(ci, str | bytes)
        and len(ci) == 2
        and _point(ci[0]) is not None
        and _point(ci[1]) is not None
        and float(ci[0]) > 0.0
        and float(ci[1]) > 0.0
    )
    beats = bool(measured and exp3693.get("candidate_beats_external_comparator") is True and positive_ci)
    verdict = "yes" if beats else ("ties_or_loses" if measured else "not_measured")
    return {
        "verdict": verdict,
        "flagged_or_live_critical": flagged,
        "leak_risk": leak_risk,
        "dependency_aware_auroc": _point(exp3693.get("dependency_aware_auroc")),
        "external_comparator_auroc": _point(exp3693.get("external_comparator_auroc")),
        "dependency_vs_external_delta_ci": exp3693.get("dependency_vs_external_delta_ci"),
    }


def _selection_status(exp3694: Mapping[str, Any], *, flagged: bool) -> JsonDict:
    measured = bool(
        exp3694
        and not flagged
        and _payload_declares_adversarial_clean(exp3694)
        and _acceptance_pass(exp3694)
        and exp3694.get("positive_control_valid") is True
        and exp3694.get("non_degeneracy_assert") is True
        and _point(exp3694.get("per_candidate_auroc")) is not None
    )
    if not measured:
        verdict = "not_measured"
    elif exp3694.get("selection_gap_closed") is True:
        verdict = "closed_new_method"
    else:
        verdict = "fundamental_decoupling"
    return {
        "verdict": verdict,
        "flagged_or_live_critical": flagged,
        "positive_control_valid": exp3694.get("positive_control_valid") is True,
        "non_degeneracy_assert": exp3694.get("non_degeneracy_assert") is True,
        "per_candidate_auroc": _point(exp3694.get("per_candidate_auroc")),
        "within_question_rank_corr": _point(exp3694.get("within_question_rank_corr")),
    }


def _code_detector_status(
    exp3695: Mapping[str, Any],
    exp3696: Mapping[str, Any],
    *,
    exp3695_flagged: bool,
    exp3696_flagged: bool,
) -> JsonDict:
    exp3695_measured = bool(
        exp3695
        and not exp3695_flagged
        and _payload_declares_adversarial_clean(exp3695)
        and _acceptance_pass(exp3695)
    )
    code_signal_recovered = bool(exp3695_measured and exp3695.get("code_signal_recovered") is True)
    exp3696_reshipped = bool(
        code_signal_recovered
        and exp3696
        and not exp3696_flagged
        and _payload_declares_adversarial_clean(exp3696)
        and _acceptance_pass(exp3696)
        and exp3696.get("module_code_path_updated") is True
        and exp3696.get("math_operating_point_unchanged") is True
        and exp3696.get("e2e_test_passed") is True
    )
    if exp3696_reshipped:
        status = "code_native_recovered_reshipped"
    elif exp3695_measured and not code_signal_recovered:
        status = "code_remains_math_only_earned"
    else:
        status = "not_measured"
    return {
        "status": status,
        "exp3695_flagged_or_live_critical": exp3695_flagged,
        "exp3696_flagged_or_live_critical": exp3696_flagged,
        "code_signal_recovered": code_signal_recovered,
        "exp3696_reship_status": "reshipped" if exp3696_reshipped else "not_measured",
        "code_native_auroc": _point(exp3695.get("code_native_auroc")),
        "code_operating_point_auroc": _point(exp3696.get("code_operating_point_auroc")),
    }


def _fr11_v12_status(exp3697: Mapping[str, Any], *, flagged: bool) -> JsonDict:
    success = bool(
        exp3697
        and not flagged
        and _payload_declares_adversarial_clean(exp3697)
        and _acceptance_pass(exp3697)
        and exp3697.get("drift_detected_deploy_arm") is True
        and exp3697.get("reset_triggered_on_transient_drift") is True
        and exp3697.get("structure_persisted_and_restored") is True
        and exp3697.get("collapse_detected_deploy_arm") is False
        and exp3697.get("quality_maintained") is True
        and exp3697.get("pass_rate_vs_true_accuracy_distinct_assert") is True
    )
    if success:
        result = "drift_reset_and_cross_session_persistence_no_collapse_quality_maintained"
    elif exp3697 and not flagged:
        result = "collapse_or_quality_regression"
    else:
        result = "not_measured"
    return {
        "result": result,
        "flagged_or_live_critical": flagged,
        "drift_detected_deploy_arm": exp3697.get("drift_detected_deploy_arm") is True,
        "reset_triggered_on_transient_drift": exp3697.get("reset_triggered_on_transient_drift") is True,
        "structure_persisted_and_restored": exp3697.get("structure_persisted_and_restored") is True,
        "collapse_detected_deploy_arm": exp3697.get("collapse_detected_deploy_arm") is True,
        "quality_maintained": exp3697.get("quality_maintained") is True,
    }


def _verifier_scope(code_status: str, selection_verdict: str) -> str:
    code_prefix = "math_plus_code" if code_status == "code_native_recovered_reshipped" else "math_only"
    if selection_verdict == "closed_new_method":
        selection_suffix = "closed"
    elif selection_verdict == "fundamental_decoupling":
        selection_suffix = "fundamental_decoupling"
    else:
        selection_suffix = "not_measured"
    return f"{code_prefix}_discrimination_facts_retired_selection_{selection_suffix}"


def _safe_claims(
    *,
    refreeze_status: str,
    external_verdict: str,
    selection_verdict: str,
    code_status: str,
    fr11_result: str,
) -> list[str]:
    claims = [
        "FoVer headline remains frozen at 0.9131 AUROC with G1-G4 satisfied.",
        "P0.1 remains honest-negative; no positive is re-asserted.",
        "Facts-generalization and trained-judge-OOD are retired and not re-asserted.",
    ]
    if refreeze_status == "reemitted_clean_for_operator":
        claims.append(
            "Dependency-aware weighting has a clean operator re-freeze package, "
            "but remains only a headline-advancement candidate pending operator "
            "action + CI re-reproduction."
        )
    else:
        claims.append(f"Operator re-freeze package status: {refreeze_status}.")
    claims.append(f"External comparator verdict: {external_verdict}.")
    claims.append(f"Selection-gap verdict: {selection_verdict}.")
    claims.append(f"Code detector status: {code_status}.")
    claims.append(f"FR-11 v12 result: {fr11_result}.")
    return claims


def _forbidden_claims(flagged_upstreams: Mapping[str, bool], external_leak_risk: bool) -> list[str]:
    claims = [
        "Do not cite the dependency-aware win as the headline until re-frozen + re-reproduced.",
        "Do not silently replace the frozen FoVer 0.9131 headline.",
        "Do not describe a re-freeze package as operator-ready unless Exp 3692 is live-clean.",
        "Do not claim the dependency-aware candidate beats the external comparator unless Exp 3693 measured it cleanly.",
        "Do not re-assert P0.1 as positive; it remains honest-negative.",
        "Do not re-assert facts-generalization or trained-judge-OOD; both are retired.",
        "Do not read missing gated-task fields as None and synthesize a result around them.",
        "Treat any AUROC >= 0.99 on n>=1000 as a leak unless explicit leak-free evidence is present.",
    ]
    flagged_paths = [
        str(UPSTREAM_ARTIFACTS[name]) for name, is_flagged in flagged_upstreams.items() if is_flagged
    ]
    if flagged_paths:
        claims.append("Do not cite uncitable upstream artifacts: " + ", ".join(flagged_paths))
    if external_leak_risk:
        claims.append("Do not cite Exp 3693 external-comparator numbers because the leak guard tripped.")
    return claims


def _cited_upstreams(
    root: Path,
    upstreams: Mapping[str, Mapping[str, Any]],
    reports: Mapping[str, Mapping[str, Any]],
    flagged: Mapping[str, bool],
) -> list[JsonDict]:
    cited: list[JsonDict] = []
    for name, rel_path in UPSTREAM_ARTIFACTS.items():
        payload = upstreams.get(name) or {}
        report = reports.get(name, {})
        if not payload or flagged.get(name):
            continue
        cited.append(
            {
                "path": str(rel_path),
                "sha256": _sha256_file(root / rel_path),
                "honest_verdict": payload.get("honest_verdict"),
                "adversarial_verify": "clean" if adversarial_report_is_clean(report) else "flagged",
            }
        )
    return cited


def _is_uncitable_upstream(payload: Mapping[str, Any], report: Mapping[str, Any]) -> bool:
    return bool(payload.get("flagged_adversarial") is True or _report_has_critical(report) or _leak_risk(payload))


def _report_has_critical(report: Mapping[str, Any]) -> bool:
    return any(str(dict(flag).get("severity") or "").lower() == "critical" for flag in report.get("flags") or [])


def _payload_declares_adversarial_clean(payload: Mapping[str, Any]) -> bool:
    if payload.get("adversarial_verify_clean") is True:
        return True
    if payload.get("adversarial_verify") == "clean":
        return True
    report = payload.get("adversarial_verify_report")
    return isinstance(report, Mapping) and list(report.get("flags") or []) == []


def _leak_risk(payload: Mapping[str, Any]) -> bool:
    if payload.get("leak_free") is True:
        return False
    n = _sample_count(payload)
    if n < 1000:
        return False
    return any(
        "auroc" in key.lower() and (point := _point(value)) is not None and point >= 0.99
        for key, value in payload.items()
    )


def _sample_count(payload: Mapping[str, Any]) -> int:
    for key in ("n_examples", "n_pooled_examples", "n_examples_code", "n"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


def _acceptance_pass(payload: Mapping[str, Any]) -> bool:
    gate = payload.get("acceptance_gate")
    if not isinstance(gate, Mapping):
        return False
    return gate.get("passed") is True or gate.get("required_fields_present") is True


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


def _read_optional_json_object(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object at {path}")
    return data


def _payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable.pop("reproducibility_checksum", None)
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def _repo_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path
