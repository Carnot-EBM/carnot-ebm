"""Exp 3712 v339 re-freeze winner capstone and publication-gate recheck.

Spec: REQ-PUBLISH-3712, SCENARIO-PUBLISH-3712.
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


OUTPUT_REL_PATH = Path("results/experiment_3712_capstone_and_g_gate_v339.json")
RANDOM_SEED = 3712
FROZEN_FOVER_HEADLINE_AUROC = 0.9131
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

UPSTREAM_ARTIFACTS: Mapping[str, Path] = {
    "exp3704": Path("results/experiment_3704_refreeze_disambiguate_dependency_vs_external_vs_fusion.json"),
    "exp3705": Path("results/experiment_3705_code_native_leak_audit_heldout.json"),
    "exp3706": Path("results/experiment_3706_reconcile_shipped_detector_heldout.json"),
    "exp3707": Path("results/experiment_3707_selection_diagnosis_formal_closure.json"),
    "exp3708": Path("results/experiment_3708_fr11_continuous_self_learning_v13.json"),
    "exp3709": Path("results/experiment_3709_kv260_drive_to_terminal_latency_transcript.json"),
}

REFREEZE_CANDIDATES = {"dependency_aware", "external", "fusion", "none"}
REFREEZE_STATUSES = {
    "reemitted_clean_for_winner",
    "no_candidate_beats_frozen",
    "not_measured",
}
CODE_HELDOUT_VERDICTS = {
    "survives_heldout_real_signal",
    "one_point_zero_was_a_leak",
    "not_measured",
}
SHIPPED_DETECTOR_RECONCILIATIONS = {
    "code_recalibrated_to_heldout",
    "narrowed_to_math_only_abstain",
    "not_measured",
}
FR11_V13_RESULTS = {
    "multi_session_consolidation_transferred_no_collapse",
    "collapse_or_quality_regression",
    "not_measured",
}
KV260_TERMINAL_STATUSES = {
    "latency_transcript_captured_terminal_candidate",
    "blocked_unreachable",
    "not_measured",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "adversarial_verify_clean",
    "strongest_refreeze_candidate",
    "refreeze_package_status",
    "code_native_heldout_verdict",
    "shipped_detector_reconciliation",
    "selection_diagnosis_closed",
    "fr11_v13_result",
    "kv260_terminal_status",
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
        "+ artifacts; no live inference; NO compute-bound marker so it does not "
        "false-flag like exp3689)."
    ),
    "adversarial_verify_clean": (
        "True iff this capstone passes adversarial_verify with no "
        "DURATION_TOO_SHORT/critical flag -- a flagged capstone does not count "
        "as a clean milestone close."
    ),
    "strongest_refreeze_candidate": (
        "dependency_aware / external / fusion / none -- which single candidate "
        "is strongest (exp3704), and whether a clean package was re-emitted for it."
    ),
    "refreeze_package_status": (
        "reemitted_clean_for_winner / no_candidate_beats_frozen / not_measured "
        "-- the operator-actionable re-freeze state."
    ),
    "code_native_heldout_verdict": (
        "survives_heldout_real_signal / one_point_zero_was_a_leak / "
        "not_measured -- did the AUROC=1.0 survive the held-out leak-audit?"
    ),
    "shipped_detector_reconciliation": (
        "code_recalibrated_to_heldout / narrowed_to_math_only_abstain / "
        "not_measured -- was the shipped detector made honest?"
    ),
    "selection_diagnosis_closed": (
        "True iff the selection diagnosis was formally closed + retirement "
        "recommended (exp3707)."
    ),
    "fr11_v13_result": (
        "Whether multi-session Tier-2 consolidation transferred to a fresh "
        "session without collapse (exp3708)."
    ),
    "kv260_terminal_status": (
        "latency_transcript_captured_terminal_candidate / blocked_unreachable / "
        "not_measured -- did KV260 reach its north-star terminal anchor?"
    ),
    "verifier_value_scope": (
        "The scoped product claim after .339: math DISCRIMINATION (frozen "
        "headline + strongest re-freeze candidate), code (real held-out signal "
        "if survived, else math-only-with-abstain), facts RETIRED, selection CLOSED."
    ),
    "g1": "Headline measured (FoVer 0.9131, 5-seed, CI, adversarial-clean).",
    "g2": "Independently reproduced (CI runner 26725185125).",
    "g3": "Prose narrowing-clean.",
    "g4": "Numbers trace to primary artifacts.",
    "paper_ready": (
        "G1 and G2 and G3 and G4 -- must remain true; the milestone does not regress the gate."
    ),
    "frozen_headline_unchanged": (
        "True iff the publication gate still reads 0.9131 -- a winning candidate "
        "is a re-freeze candidate, never a silent swap."
    ),
    "unmet_gates": (
        "Report which gates are unmet, not a count (publication_blocker_count is retired)."
    ),
    "p01_status": "P0.1 stays honest-negative; do not re-assert a positive.",
    "facts_generalization_retired": "Records facts-generalization as RETIRED (exp3670).",
    "trained_judge_ood_retired": "Records the trained-judge-OOD hypothesis as retired (exp3659).",
    "paper_v6_safe_claims": "Narrowing-clean claims.",
    "paper_v6_forbidden_claims": (
        "Overclaims to avoid, including re-freeze headline swaps and unconfirmed code AUROC=1.0."
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
    """Build the Exp 3712 terminal artifact from upstream result files."""

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
    hard_flagged = {
        name: _is_live_critical_upstream(payload, reports.get(name, {}))
        for name, payload in upstreams.items()
    }
    uncitable = {
        name: hard_flagged[name] or _leak_risk(payload)
        for name, payload in upstreams.items()
    }

    refreeze = _refreeze_status(upstreams["exp3704"], hard_flagged=hard_flagged["exp3704"])
    code = _code_native_heldout_status(upstreams["exp3705"], hard_flagged=hard_flagged["exp3705"])
    shipped = _shipped_detector_status(upstreams["exp3706"], hard_flagged=hard_flagged["exp3706"])
    selection_closed = _selection_closed(upstreams["exp3707"], hard_flagged=hard_flagged["exp3707"])
    fr11 = _fr11_v13_status(upstreams["exp3708"], hard_flagged=hard_flagged["exp3708"])
    kv260 = _kv260_status(upstreams["exp3709"], hard_flagged=hard_flagged["exp3709"])
    scope = _verifier_scope(
        refreeze["candidate"],
        code["verdict"],
        shipped["status"],
    )

    g1 = _gate_pass(gate, "G1")
    g2 = _gate_pass(gate, "G2")
    g3 = _gate_pass(gate, "G3")
    g4 = _gate_pass(gate, "G4")
    paper_ready = bool(gate.get("paper_ready") is True and g1 and g2 and g3 and g4)
    frozen_headline_unchanged = _frozen_headline_unchanged(gate)
    duration_s = round(max(0.0001, (time.perf_counter() if now_s is None else float(now_s)) - start), 6)

    artifact: JsonDict = {
        "honest_verdict": (
            f"complete: capstone_v339_refreeze_winner_{refreeze['candidate']}_"
            f"code_native_{code['verdict']}_selection_closed_"
            f"kv260_{kv260['status']}_paper_ready_{str(paper_ready).lower()}_"
            f"{'frozen_headline_unchanged' if frozen_headline_unchanged else 'frozen_headline_changed'}"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "adversarial_verify_clean": capstone_adversarial_verify_clean,
        "strongest_refreeze_candidate": refreeze["candidate"],
        "refreeze_package_status": refreeze["status"],
        "code_native_heldout_verdict": code["verdict"],
        "shipped_detector_reconciliation": shipped["status"],
        "selection_diagnosis_closed": selection_closed,
        "fr11_v13_result": fr11["result"],
        "kv260_terminal_status": kv260["status"],
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
        "paper_v6_safe_claims": _safe_claims(refreeze, code, shipped, fr11, kv260),
        "paper_v6_forbidden_claims": _forbidden_claims(uncitable, code["leak_risk"]),
        "cited_upstream_artifacts": _cited_upstreams(root_path, upstreams, reports, uncitable),
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "field_principles": dict(FIELD_PRINCIPLES),
        "frozen_fover_headline_auroc": FROZEN_FOVER_HEADLINE_AUROC,
        "refreeze_candidate": refreeze,
        "code_native_heldout": code,
        "shipped_detector": shipped,
        "selection_diagnosis": {
            "closed": selection_closed,
            "honest_verdict": upstreams["exp3707"].get("honest_verdict"),
        },
        "fr11_v13": fr11,
        "kv260_terminal": kv260,
        "publication_gate": _gate_details(gate),
        "summarized_upstream_artifacts": summaries,
        "flagged_upstream_artifacts_excluded": [
            str(UPSTREAM_ARTIFACTS[name]) for name, is_flagged in uncitable.items() if is_flagged
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
    """Build, persist, live-verify, and re-persist the Exp 3712 artifact."""

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
    """Validate the required Exp 3712 schema and publication-gate invariants."""

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
    if artifact.get("strongest_refreeze_candidate") not in REFREEZE_CANDIDATES:
        raise ValueError("strongest_refreeze_candidate is unsupported")
    if artifact.get("refreeze_package_status") not in REFREEZE_STATUSES:
        raise ValueError("refreeze_package_status is unsupported")
    if artifact.get("code_native_heldout_verdict") not in CODE_HELDOUT_VERDICTS:
        raise ValueError("code_native_heldout_verdict is unsupported")
    if artifact.get("shipped_detector_reconciliation") not in SHIPPED_DETECTOR_RECONCILIATIONS:
        raise ValueError("shipped_detector_reconciliation is unsupported")
    if artifact.get("selection_diagnosis_closed") is not True:
        raise ValueError("selection_diagnosis_closed must be true")
    if artifact.get("fr11_v13_result") not in FR11_V13_RESULTS:
        raise ValueError("fr11_v13_result is unsupported")
    if artifact.get("kv260_terminal_status") not in KV260_TERMINAL_STATUSES:
        raise ValueError("kv260_terminal_status is unsupported")
    if not isinstance(artifact.get("verifier_value_scope"), str) or "facts_retired_selection_closed" not in artifact["verifier_value_scope"]:
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
    for exp_id in range(3704, 3710):
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


def _refreeze_status(exp3704: Mapping[str, Any], *, hard_flagged: bool) -> JsonDict:
    candidate = str(exp3704.get("strongest_candidate") or "none")
    if candidate not in {"dependency_aware", "external", "fusion"}:
        candidate = "none"
    package_ready = bool(
        exp3704
        and not hard_flagged
        and _payload_declares_adversarial_clean(exp3704)
        and _acceptance_pass(exp3704)
        and exp3704.get("strongest_candidate_beats_frozen") is True
        and exp3704.get("refreeze_package_reemitted_for_winner") is True
        and exp3704.get("frozen_headline_unchanged_assert") is True
        and candidate != "none"
    )
    no_candidate = bool(
        exp3704
        and not hard_flagged
        and _payload_declares_adversarial_clean(exp3704)
        and exp3704.get("strongest_candidate_beats_frozen") is False
    )
    if package_ready:
        status = "reemitted_clean_for_winner"
    elif no_candidate:
        status = "no_candidate_beats_frozen"
        candidate = "none"
    else:
        status = "not_measured"
        candidate = "none"
    return {
        "candidate": candidate,
        "status": status,
        "flagged_or_live_critical": hard_flagged,
        "adversarial_verify_clean": _payload_declares_adversarial_clean(exp3704),
        "refreeze_package_reemitted_for_winner": package_ready,
        "strongest_candidate_beats_frozen": exp3704.get("strongest_candidate_beats_frozen"),
    }


def _code_native_heldout_status(exp3705: Mapping[str, Any], *, hard_flagged: bool) -> JsonDict:
    leak_risk = _leak_risk(exp3705)
    measured = bool(
        exp3705
        and not hard_flagged
        and _payload_declares_adversarial_clean(exp3705)
        and _acceptance_pass(exp3705)
        and _point(exp3705.get("heldout_code_auroc")) is not None
    )
    leak_detected = bool(exp3705.get("leak_detected") is True or leak_risk)
    survives = bool(measured and exp3705.get("code_signal_survives_heldout") is True and not leak_detected)
    if survives:
        verdict = "survives_heldout_real_signal"
    elif measured and (leak_detected or exp3705.get("code_signal_survives_heldout") is False):
        verdict = "one_point_zero_was_a_leak"
    else:
        verdict = "not_measured"
    return {
        "verdict": verdict,
        "flagged_or_live_critical": hard_flagged,
        "leak_risk": leak_risk,
        "leak_detected": leak_detected,
        "heldout_code_auroc": _point(exp3705.get("heldout_code_auroc")),
        "n_examples_heldout": _sample_count(exp3705),
    }


def _shipped_detector_status(exp3706: Mapping[str, Any], *, hard_flagged: bool) -> JsonDict:
    measured = bool(
        exp3706
        and not hard_flagged
        and _payload_declares_adversarial_clean(exp3706)
        and _acceptance_pass(exp3706)
    )
    action = str(exp3706.get("reconciliation_action") or "")
    verdict = str(exp3706.get("honest_verdict") or "")
    if measured and (
        action == "recalibrated_to_heldout"
        or exp3706.get("detector_reconciled_to_heldout") is True
        or "code_recalibrated_to_heldout" in verdict
    ):
        status = "code_recalibrated_to_heldout"
    elif measured and (
        action == "narrowed_to_math_only_abstain"
        or exp3706.get("math_only_abstain") is True
        or exp3706.get("code_surface_abstains") is True
        or "narrowed_to_math_only_abstain" in verdict
    ):
        status = "narrowed_to_math_only_abstain"
    else:
        status = "not_measured"
    return {
        "status": status,
        "flagged_or_live_critical": hard_flagged,
        "reconciliation_action": action or None,
        "honest_verdict": exp3706.get("honest_verdict"),
    }


def _selection_closed(exp3707: Mapping[str, Any], *, hard_flagged: bool) -> bool:
    if not exp3707 or hard_flagged or not _payload_declares_adversarial_clean(exp3707):
        return False
    verdict = str(exp3707.get("honest_verdict") or "")
    return bool(
        exp3707.get("selection_diagnosis_closed") is True
        or (exp3707.get("question_closed") is True and "retirement_recommended" in verdict)
        or (exp3707.get("retirement_recommended") is True and "selection_diagnosis_formally_closed" in verdict)
    )


def _fr11_v13_status(exp3708: Mapping[str, Any], *, hard_flagged: bool) -> JsonDict:
    success = bool(
        exp3708
        and not hard_flagged
        and _payload_declares_adversarial_clean(exp3708)
        and _acceptance_pass(exp3708)
        and (
            exp3708.get("fresh_session_transfer_without_collapse") is True
            or exp3708.get("consolidated_template_transfer_gain_over_cold_start") is True
            or exp3708.get("multi_session_consolidation_transferred") is True
        )
        and (
            exp3708.get("collapse_detected") is False
            or exp3708.get("collapse_detected_deploy_arm") is False
        )
        and exp3708.get("quality_maintained") is True
    )
    if success:
        result = "multi_session_consolidation_transferred_no_collapse"
    elif exp3708 and not hard_flagged:
        result = "collapse_or_quality_regression"
    else:
        result = "not_measured"
    return {
        "result": result,
        "flagged_or_live_critical": hard_flagged,
        "honest_verdict": exp3708.get("honest_verdict"),
        "fresh_session_transfer_auroc_gain": _point(exp3708.get("fresh_session_transfer_auroc_gain")),
    }


def _kv260_status(exp3709: Mapping[str, Any], *, hard_flagged: bool) -> JsonDict:
    if not exp3709 or hard_flagged:
        status = "not_measured"
    elif exp3709.get("blocked_unreachable") is True or exp3709.get("kv260_ssh_reachable") is False:
        status = "blocked_unreachable"
    elif (
        exp3709.get("latency_transcript_captured") is True
        or exp3709.get("terminal_candidate") is True
        or exp3709.get("terminal_condition_met") is True
    ):
        status = "latency_transcript_captured_terminal_candidate"
    else:
        status = "blocked_unreachable"
    return {
        "status": status,
        "flagged_or_live_critical": hard_flagged,
        "honest_verdict": exp3709.get("honest_verdict"),
        "board_latency_median_ms": _point(exp3709.get("board_latency_median_ms")),
    }


def _verifier_scope(refreeze_candidate: str, code_verdict: str, shipped_status: str) -> str:
    refreeze_part = (
        f"plus_refreeze_candidate_{refreeze_candidate}"
        if refreeze_candidate in {"dependency_aware", "external", "fusion"}
        else "no_refreeze_candidate"
    )
    if code_verdict == "survives_heldout_real_signal" and shipped_status == "code_recalibrated_to_heldout":
        code_part = "code_heldout_real_signal"
    else:
        code_part = "code_math_only_with_abstain"
    return (
        f"math_discrimination_frozen_0_9131_{refreeze_part}_"
        f"{code_part}_facts_retired_selection_closed"
    )


def _safe_claims(
    refreeze: Mapping[str, Any],
    code: Mapping[str, Any],
    shipped: Mapping[str, Any],
    fr11: Mapping[str, Any],
    kv260: Mapping[str, Any],
) -> list[str]:
    candidate = refreeze.get("candidate")
    claims = [
        "FoVer headline remains frozen at 0.9131 AUROC with G1-G4 satisfied.",
        "P0.1 remains honest-negative; no positive is re-asserted.",
        "Facts-generalization and trained-judge-OOD are retired and not re-asserted.",
        "Selection diagnosis is formally closed with operator retirement recommended.",
    ]
    if refreeze.get("status") == "reemitted_clean_for_winner":
        claims.append(
            f"{candidate} is a headline-advancement candidate with an "
            "operator-ready re-freeze package pending operator action + CI re-reproduction."
        )
    else:
        claims.append(f"Re-freeze package status: {refreeze.get('status')}.")
    claims.append(f"Code-native held-out verdict: {code.get('verdict')}.")
    claims.append(f"Shipped detector reconciliation: {shipped.get('status')}.")
    claims.append(f"FR-11 v13 result: {fr11.get('result')}.")
    claims.append(f"KV260 terminal status: {kv260.get('status')}.")
    return claims


def _forbidden_claims(uncitable_upstreams: Mapping[str, bool], code_leak_risk: bool) -> list[str]:
    claims = [
        "Do not cite a re-freeze candidate as the headline until re-frozen + re-reproduced.",
        "Do not silently replace the frozen FoVer 0.9131 headline.",
        "Do not cite a code AUROC=1.0 unless held-out-confirmed leak-free.",
        "Do not re-assert P0.1 as positive; it remains honest-negative.",
        "Do not re-assert facts-generalization or trained-judge-OOD; both are retired.",
        "Do not read missing gated-task fields as None and synthesize a result around them.",
        "Treat any AUROC >= 0.99 on n>=1000 as a leak unless explicit leak-free evidence is present.",
    ]
    flagged_paths = [
        str(UPSTREAM_ARTIFACTS[name]) for name, is_flagged in uncitable_upstreams.items() if is_flagged
    ]
    if flagged_paths:
        claims.append("Do not cite uncitable upstream artifacts: " + ", ".join(flagged_paths))
    if code_leak_risk:
        claims.append("Do not cite the held-out code AUROC as a generalization success because the leak guard tripped.")
    return claims


def _cited_upstreams(
    root: Path,
    upstreams: Mapping[str, Mapping[str, Any]],
    reports: Mapping[str, Mapping[str, Any]],
    uncitable: Mapping[str, bool],
) -> list[JsonDict]:
    cited: list[JsonDict] = []
    for name, rel_path in UPSTREAM_ARTIFACTS.items():
        payload = upstreams.get(name) or {}
        report = reports.get(name, {})
        if not payload or uncitable.get(name):
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


def _is_live_critical_upstream(payload: Mapping[str, Any], report: Mapping[str, Any]) -> bool:
    return bool(payload.get("flagged_adversarial") is True or _report_has_critical(report))


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
    if payload.get("leak_free") is True or payload.get("heldout_leak_free") is True:
        return False
    if payload.get("leak_detected") is True:
        return True
    n = _sample_count(payload)
    if n < 1000:
        return False
    return any(
        "auroc" in key.lower() and (point := _point(value)) is not None and point >= 0.99
        for key, value in payload.items()
    )


def _sample_count(payload: Mapping[str, Any]) -> int:
    for key in ("n_examples_heldout", "n_examples", "n_pooled_examples", "n"):
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
