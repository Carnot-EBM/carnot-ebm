"""Exp 3723 v340 convergence capstone and hardened publication-gate recheck.

Spec: REQ-PUBLISH-3723, SCENARIO-PUBLISH-3723.
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


OUTPUT_REL_PATH = Path("results/experiment_3723_capstone_and_g_gate_v340.json")
RANDOM_SEED = 3723
FROZEN_FOVER_HEADLINE_AUROC = 0.9131
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SUMMARY_EXP_IDS = tuple(range(3715, 3723))
UPSTREAM_ARTIFACTS: Mapping[str, Path] = {
    "exp3659": Path("results/experiment_3659_trained_ebm_judge_ood_real_substrate_v3.json"),
    "exp3670": Path("results/experiment_3670_facts_row_real_benchmark.json"),
    "exp3707": Path("results/experiment_3707_selection_diagnosis_formal_closure.json"),
    "exp3715": Path("results/experiment_3715_refreeze_disambiguation_clean_corrigendum.json"),
    "exp3716": Path("results/experiment_3716_ship_paper_v6_narrowing_lint.json"),
    "exp3717": Path("results/experiment_3717_g4_full_provenance_audit.json"),
    "exp3718": Path("results/experiment_3718_risk_coverage_abstention_characterization.json"),
    "exp3719": Path("results/experiment_3719_headline_replication_fresh_corpus.json"),
    "exp3720": Path("results/experiment_3720_fr11_continuous_self_learning_v14.json"),
    "exp3721": Path("results/experiment_3721_hardware_kv260_terminal_confirm_and_continuity.json"),
    "exp3722": Path("results/experiment_3722_convergence_synthesis_operator_next_thesis.json"),
}

G4_PROVENANCE_RESULTS = {"fully_traced", "gap_found", "not_measured"}
ENERGY_ABSTENTION_VERDICTS = {
    "energy_better_than_entropy",
    "energy_ties_or_loses",
    "not_measured",
}
FRESH_CORPUS_RESULTS = {"generalizes", "fover_specific", "not_measured"}
FR11_V14_RESULTS = {
    "robust_under_shift_no_collapse",
    "falls_back_gracefully_under_shift_no_collapse",
    "hurts_under_distribution_shift",
    "not_measured",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "adversarial_verify_clean",
    "exp3704_corrigendum_clean",
    "g3_mechanically_enforced",
    "g4_provenance_audit_result",
    "energy_abstention_verdict",
    "fresh_corpus_generalization",
    "fr11_v14_result",
    "kv260_terminal_confirmed",
    "operator_next_thesis_recorded",
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
    "selection_diagnosis_closed",
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
        "Reads the gate script and upstream artifacts only; no live inference "
        "and no compute-bound marker belongs in this capstone."
    ),
    "adversarial_verify_clean": (
        "True iff this capstone passes adversarial_verify with no "
        "DURATION_TOO_SHORT or critical flag."
    ),
    "exp3704_corrigendum_clean": (
        "True iff the benign-flagged exp3704 was re-emitted clean as exp3715 "
        "and the headline stayed frozen."
    ),
    "g3_mechanically_enforced": (
        "True iff the paper_v6_narrowing_lint shipped and the current paper passes it."
    ),
    "g4_provenance_audit_result": (
        "fully_traced / gap_found / not_measured -- whether every headline "
        "number traces to a clean primary artifact."
    ),
    "energy_abstention_verdict": (
        "energy_better_than_entropy / energy_ties_or_loses / not_measured -- "
        "the deployable abstention-gate result."
    ),
    "fresh_corpus_generalization": (
        "generalizes / fover_specific / not_measured -- second-corpus "
        "headline-class discrimination result."
    ),
    "fr11_v14_result": (
        "Whether the consolidated template is robust, falls back gracefully, "
        "or hurts under distribution shift."
    ),
    "kv260_terminal_confirmed": (
        "True iff KV260 was confirmed terminal and the mandate-lift was recommended."
    ),
    "operator_next_thesis_recorded": (
        "True iff the converged-state synthesis and operator next-thesis request were recorded."
    ),
    "verifier_value_scope": (
        "Scoped product claim after .340: math discrimination, abstention when "
        "energy beats entropy, code math-only-with-abstain, facts retired, selection closed."
    ),
    "g1": "Headline measured (FoVer 0.9131, 5-seed, CI, adversarial-clean).",
    "g2": "Independently reproduced (CI runner 26725185125).",
    "g3": "Prose narrowing-clean, now mechanically enforced via exp3716.",
    "g4": "Numbers trace to primary artifacts, audited by exp3717.",
    "paper_ready": "G1 and G2 and G3 and G4 -- must remain true.",
    "frozen_headline_unchanged": "True iff the publication gate still reads 0.9131.",
    "unmet_gates": "Report which gates are unmet, not a retired blocker count.",
    "p01_status": "P0.1 stays honest-negative; do not re-assert a positive.",
    "facts_generalization_retired": "Records facts-generalization as retired.",
    "trained_judge_ood_retired": "Records the trained-judge-OOD hypothesis as retired.",
    "selection_diagnosis_closed": "Records the selection diagnosis as formally closed.",
    "paper_v6_safe_claims": "Narrowing-clean claims.",
    "paper_v6_forbidden_claims": "Overclaims to avoid under Paper-v6 Narrowing.",
    "cited_upstream_artifacts": "sha256 provenance for cited upstream artifacts.",
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
    """Build the Exp 3723 terminal artifact from checked-in upstream artifacts."""

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
    blocking = {
        name: _blocking_upstream(payload, reports.get(name, {}))
        for name, payload in upstreams.items()
    }
    leaks = {name: _leak_risk(payload) for name, payload in upstreams.items()}
    excluded = _excluded_upstreams(upstreams, blocking, leaks)

    exp3704_corrigendum_clean = _exp3704_corrigendum_clean(
        upstreams["exp3715"],
        blocked=blocking["exp3715"] or leaks["exp3715"],
    )
    g3_mechanically_enforced = _g3_mechanically_enforced(
        upstreams["exp3716"],
        blocked=blocking["exp3716"] or leaks["exp3716"],
    )
    g4_provenance_audit_result = _g4_result(
        upstreams["exp3717"],
        blocked=blocking["exp3717"] or leaks["exp3717"],
    )
    energy_abstention_verdict = _energy_verdict(
        upstreams["exp3718"],
        blocked=blocking["exp3718"] or leaks["exp3718"],
    )
    fresh_corpus_generalization = _fresh_corpus_result(
        upstreams["exp3719"],
        blocked=blocking["exp3719"] or leaks["exp3719"],
    )
    fr11_v14_result = _fr11_v14_result(
        upstreams["exp3720"],
        blocked=blocking["exp3720"] or leaks["exp3720"],
    )
    kv260_terminal_confirmed = _kv260_confirmed(
        upstreams["exp3721"],
        blocked=blocking["exp3721"] or leaks["exp3721"],
    )
    operator_next_thesis_recorded = _operator_next_thesis_recorded(
        upstreams["exp3722"],
        blocked=blocking["exp3722"] or leaks["exp3722"],
    )
    selection_diagnosis_closed = _selection_closed(
        upstreams["exp3707"],
        blocked=blocking["exp3707"] or leaks["exp3707"],
    )
    facts_generalization_retired = _facts_retired(
        upstreams["exp3670"],
        blocked=blocking["exp3670"] or leaks["exp3670"],
    )
    trained_judge_ood_retired = _judge_retired(
        upstreams["exp3659"],
        blocked=blocking["exp3659"] or leaks["exp3659"],
    )

    g1 = _gate_pass(gate, "G1")
    g2 = _gate_pass(gate, "G2")
    g3 = _gate_pass(gate, "G3")
    g4 = _gate_pass(gate, "G4")
    paper_ready = bool(gate.get("paper_ready") is True and g1 and g2 and g3 and g4)
    frozen_headline_unchanged = _frozen_headline_unchanged(gate)
    duration_s = round(max(0.0001, (time.perf_counter() if now_s is None else float(now_s)) - start), 6)
    scope = _verifier_scope(energy_abstention_verdict)

    artifact: JsonDict = {
        "schema": "carnot.capstone_and_g_gate_v340_3723.v1",
        "experiment_id": "exp3723",
        "honest_verdict": (
            "complete: capstone_v340_convergence_gates_hardened_g3_mechanical_g4_audited_"
            f"abstention_{energy_abstention_verdict}_"
            f"fresh_corpus_{fresh_corpus_generalization}_"
            f"kv260_{'terminal' if kv260_terminal_confirmed else 'not_terminal'}_"
            f"{'operator_thesis_requested' if operator_next_thesis_recorded else 'operator_thesis_not_measured'}_"
            f"paper_ready_{str(paper_ready).lower()}_"
            f"{'frozen_headline_unchanged' if frozen_headline_unchanged else 'frozen_headline_changed'}"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "adversarial_verify_clean": capstone_adversarial_verify_clean,
        "exp3704_corrigendum_clean": exp3704_corrigendum_clean,
        "g3_mechanically_enforced": g3_mechanically_enforced,
        "g4_provenance_audit_result": g4_provenance_audit_result,
        "energy_abstention_verdict": energy_abstention_verdict,
        "fresh_corpus_generalization": fresh_corpus_generalization,
        "fr11_v14_result": fr11_v14_result,
        "kv260_terminal_confirmed": kv260_terminal_confirmed,
        "operator_next_thesis_recorded": operator_next_thesis_recorded,
        "verifier_value_scope": scope,
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "g4": g4,
        "paper_ready": paper_ready,
        "frozen_headline_unchanged": frozen_headline_unchanged,
        "unmet_gates": list(gate.get("unmet_gates") or []),
        "p01_status": "honest-negative",
        "facts_generalization_retired": facts_generalization_retired,
        "trained_judge_ood_retired": trained_judge_ood_retired,
        "selection_diagnosis_closed": selection_diagnosis_closed,
        "paper_v6_safe_claims": _safe_claims(
            g3_mechanically_enforced=g3_mechanically_enforced,
            g4_provenance_audit_result=g4_provenance_audit_result,
            energy_abstention_verdict=energy_abstention_verdict,
            fresh_corpus_generalization=fresh_corpus_generalization,
            fr11_v14_result=fr11_v14_result,
            kv260_terminal_confirmed=kv260_terminal_confirmed,
        ),
        "paper_v6_forbidden_claims": _forbidden_claims(excluded),
        "cited_upstream_artifacts": _cited_upstreams(root_path, upstreams, reports, excluded),
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "field_principles": dict(FIELD_PRINCIPLES),
        "frozen_fover_headline_auroc": FROZEN_FOVER_HEADLINE_AUROC,
        "publication_gate": _gate_details(gate),
        "summarized_upstream_artifacts": summaries,
        "upstream_status": {
            "exp3715": {"clean": exp3704_corrigendum_clean},
            "exp3716": {"g3_mechanically_enforced": g3_mechanically_enforced},
            "exp3717": {"g4_provenance_audit_result": g4_provenance_audit_result},
            "exp3718": {"energy_abstention_verdict": energy_abstention_verdict},
            "exp3719": {"fresh_corpus_generalization": fresh_corpus_generalization},
            "exp3720": {"fr11_v14_result": fr11_v14_result},
            "exp3721": {"kv260_terminal_confirmed": kv260_terminal_confirmed},
            "exp3722": {"operator_next_thesis_recorded": operator_next_thesis_recorded},
        },
        "excluded_upstream_artifacts": [
            {"path": str(UPSTREAM_ARTIFACTS[name]), "reason": reason}
            for name, reason in excluded.items()
        ],
        "source_artifacts": [
            str(rel_path) for name, rel_path in UPSTREAM_ARTIFACTS.items() if upstreams.get(name)
        ],
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "acceptance_gate": {
            "passed": paper_ready and frozen_headline_unchanged,
            "condition": "paper_ready == true AND frozen_headline_unchanged == true",
        },
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
    adversarial_reports: Mapping[str, Mapping[str, Any]] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, persist, adversarial-verify, and re-persist the Exp 3723 artifact."""

    root_path = Path(root)
    out_path = _repo_path(root_path, Path(output_path))
    artifact = build_artifact(
        root_path,
        gate_data=gate_data,
        summary_records=summary_records,
        adversarial_reports=adversarial_reports,
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
    """Validate the required Exp 3723 schema and publication-gate invariants."""

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
    if not isinstance(artifact.get("exp3704_corrigendum_clean"), bool):
        raise ValueError("exp3704_corrigendum_clean must be a boolean")
    if not isinstance(artifact.get("g3_mechanically_enforced"), bool):
        raise ValueError("g3_mechanically_enforced must be a boolean")
    if artifact.get("g4_provenance_audit_result") not in G4_PROVENANCE_RESULTS:
        raise ValueError("g4_provenance_audit_result is unsupported")
    if artifact.get("energy_abstention_verdict") not in ENERGY_ABSTENTION_VERDICTS:
        raise ValueError("energy_abstention_verdict is unsupported")
    if artifact.get("fresh_corpus_generalization") not in FRESH_CORPUS_RESULTS:
        raise ValueError("fresh_corpus_generalization is unsupported")
    if artifact.get("fr11_v14_result") not in FR11_V14_RESULTS:
        raise ValueError("fr11_v14_result is unsupported")
    if not isinstance(artifact.get("kv260_terminal_confirmed"), bool):
        raise ValueError("kv260_terminal_confirmed must be a boolean")
    if not isinstance(artifact.get("operator_next_thesis_recorded"), bool):
        raise ValueError("operator_next_thesis_recorded must be a boolean")
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
    if artifact.get("selection_diagnosis_closed") is not True:
        raise ValueError("selection_diagnosis_closed must remain true")
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
    """Return true only when the capstone has no critical or duration flag."""

    return not _blocking_report(report)


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
    for exp_id in SUMMARY_EXP_IDS:
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


def _verify_upstreams(root: Path) -> dict[str, JsonDict]:  # pragma: no cover - subprocess boundary
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


def _exp3704_corrigendum_clean(payload: Mapping[str, Any], *, blocked: bool) -> bool:
    return bool(
        payload
        and not blocked
        and _payload_declares_clean(payload)
        and payload.get("no_candidate_beats_frozen") is True
        and payload.get("frozen_headline_unchanged_assert") is True
        and _point(payload.get("frozen_headline_auroc")) == FROZEN_FOVER_HEADLINE_AUROC
    )


def _g3_mechanically_enforced(payload: Mapping[str, Any], *, blocked: bool) -> bool:
    return bool(
        payload
        and not blocked
        and payload.get("g3_now_mechanically_enforced") is True
        and payload.get("current_paper_lint_clean") is True
        and payload.get("conductor_unmodified_assert") is True
    )


def _g4_result(payload: Mapping[str, Any], *, blocked: bool) -> str:
    if not payload or blocked:
        return "not_measured"
    if payload.get("all_numbers_trace_to_clean_artifacts") is True and payload.get("any_cited_source_flagged") is not True:
        return "fully_traced"
    return "gap_found"


def _energy_verdict(payload: Mapping[str, Any], *, blocked: bool) -> str:
    if not payload or blocked:
        return "not_measured"
    if payload.get("energy_beats_baseline_abstention") is True:
        return "energy_better_than_entropy"
    if payload.get("energy_beats_baseline_abstention") is False:
        return "energy_ties_or_loses"
    return "not_measured"


def _fresh_corpus_result(payload: Mapping[str, Any], *, blocked: bool) -> str:
    if not payload or blocked or "generalizes_beyond_fover" not in payload:
        return "not_measured"
    return "generalizes" if payload.get("generalizes_beyond_fover") is True else "fover_specific"


def _fr11_v14_result(payload: Mapping[str, Any], *, blocked: bool) -> str:
    if not payload or blocked:
        return "not_measured"
    if payload.get("template_robust_or_graceful_fallback") is True and payload.get("collapse_detected_deploy_arm") is False:
        if payload.get("conservative_fallback_triggered") is True or payload.get("fallback_effective_policy_no_worse_than_cold_start") is True:
            return "falls_back_gracefully_under_shift_no_collapse"
        return "robust_under_shift_no_collapse"
    return "hurts_under_distribution_shift"


def _kv260_confirmed(payload: Mapping[str, Any], *, blocked: bool) -> bool:
    return bool(
        payload
        and not blocked
        and payload.get("kv260_terminal_condition_confirmed") is True
        and "recommend" in str(payload.get("kv260_mandate_lift_recommendation") or "")
        and payload.get("speedup_claim_avoided_assert") is True
    )


def _operator_next_thesis_recorded(payload: Mapping[str, Any], *, blocked: bool) -> bool:
    verdict = str(payload.get("honest_verdict") or "")
    return bool(
        payload
        and not blocked
        and "operator_decision_requested" in verdict
        and bool(str(payload.get("operator_decision_request") or ""))
    )


def _selection_closed(payload: Mapping[str, Any], *, blocked: bool) -> bool:
    verdict = str(payload.get("honest_verdict") or "")
    return bool(
        payload
        and not blocked
        and (
            payload.get("selection_diagnosis_closed") is True
            or (
                payload.get("question_closed") is True
                and "selection_diagnosis_formally_closed" in verdict
            )
            or (
                payload.get("retirement_recommended") is True
                and "selection_diagnosis_formally_closed" in verdict
            )
        )
    )


def _facts_retired(payload: Mapping[str, Any], *, blocked: bool) -> bool:
    verdict = str(payload.get("honest_verdict") or "").lower()
    outcome = str(payload.get("honest_outcome") or "").lower()
    return bool(
        payload
        and not blocked
        and (
            payload.get("facts_generalization_retired") is True
            or payload.get("facts_generalize_or_adds_value_real") is False
            or "domain_bound_real" in outcome
            or "facts_domain_bound" in verdict
            or ("facts" in verdict and "retired" in verdict)
        )
    )


def _judge_retired(payload: Mapping[str, Any], *, blocked: bool) -> bool:
    verdict = str(payload.get("honest_verdict") or "").lower()
    return bool(
        payload
        and not blocked
        and (
            payload.get("trained_judge_ood_retired") is True
            or payload.get("trained_judge_transfers_ood") is False
            or "not_the_cross_domain_fix" in verdict
            or ("judge" in verdict and "retired" in verdict)
        )
    )


def _verifier_scope(energy_verdict: str) -> str:
    abstention = (
        "deployable_abstention_gate_if_energy_gt_entropy"
        if energy_verdict == "energy_better_than_entropy"
        else f"abstention_{energy_verdict}"
    )
    return (
        "math_discrimination_frozen_0_9131_second_corpus_datapoint_"
        f"{abstention}_code_math_only_with_abstain_facts_retired_selection_closed"
    )


def _safe_claims(
    *,
    g3_mechanically_enforced: bool,
    g4_provenance_audit_result: str,
    energy_abstention_verdict: str,
    fresh_corpus_generalization: str,
    fr11_v14_result: str,
    kv260_terminal_confirmed: bool,
) -> list[str]:
    return [
        "FoVer headline remains 0.9131 AUROC; no re-freeze candidate replaces it.",
        f"G3 mechanically enforced by Paper-v6 narrowing lint: {g3_mechanically_enforced}.",
        f"G4 provenance audit result: {g4_provenance_audit_result}.",
        f"Energy abstention verdict: {energy_abstention_verdict}.",
        f"Fresh-corpus headline discrimination verdict: {fresh_corpus_generalization}.",
        f"FR-11 v14 distribution-shift result: {fr11_v14_result}.",
        f"KV260 terminal confirmed: {kv260_terminal_confirmed}; no hardware speedup is claimed.",
        "P0.1 remains honest-negative; facts-generalization and trained-judge-OOD remain retired.",
        "Code scope remains math-only-with-abstain.",
    ]


def _forbidden_claims(excluded: Mapping[str, str]) -> list[str]:
    claims = [
        "Do not cite a re-freeze candidate as the headline; the frozen FoVer headline stays 0.9131.",
        "Do not cite a code AUROC>=0.99 as a valid code-generalization result.",
        "Do not claim hardware speedup; KV260 is a terminal POC latency anchor only.",
        "Do not revive facts-generalization or trained-judge-OOD claims; both are retired.",
        "Do not read missing gated-task fields as None and synthesize a positive result.",
        "Treat any AUROC >= 0.99 on n>=1000 as a leak unless explicit leak-free evidence is present.",
    ]
    if excluded:
        paths = ", ".join(str(UPSTREAM_ARTIFACTS[name]) for name in excluded)
        claims.append(f"Do not cite excluded upstream artifacts: {paths}.")
    return claims


def _cited_upstreams(
    root: Path,
    upstreams: Mapping[str, Mapping[str, Any]],
    reports: Mapping[str, Mapping[str, Any]],
    excluded: Mapping[str, str],
) -> list[JsonDict]:
    cited: list[JsonDict] = []
    for name, rel_path in UPSTREAM_ARTIFACTS.items():
        payload = upstreams.get(name) or {}
        if not payload or name in excluded:
            continue
        flags = [dict(flag) for flag in reports.get(name, {}).get("flags") or []]
        cited.append(
            {
                "path": str(rel_path),
                "sha256": _sha256_file(root / rel_path),
                "honest_verdict": payload.get("honest_verdict"),
                "adversarial_verify_status": "no_critical_or_duration_flags",
                "adversarial_flag_count": len(flags),
                "adversarial_warn_count": sum(
                    1 for flag in flags if str(flag.get("severity") or "").lower() == "warn"
                ),
            }
        )
    return cited


def _excluded_upstreams(
    upstreams: Mapping[str, Mapping[str, Any]],
    blocking: Mapping[str, bool],
    leaks: Mapping[str, bool],
) -> dict[str, str]:
    excluded: dict[str, str] = {}
    for name, payload in upstreams.items():
        if not payload:
            continue
        if blocking.get(name):
            excluded[name] = "adversarial_blocking_flag"
        elif leaks.get(name):
            excluded[name] = "leak_risk"
    return excluded


def _blocking_upstream(payload: Mapping[str, Any], report: Mapping[str, Any]) -> bool:
    return bool(payload.get("flagged_adversarial") is True or _blocking_report(report))


def _blocking_report(report: Mapping[str, Any]) -> bool:
    for flag in list(report.get("flags") or []):
        flag_dict = dict(flag)
        if str(flag_dict.get("kind") or "") == "DURATION_TOO_SHORT":
            return True
        if str(flag_dict.get("severity") or "").lower() == "critical":
            return True
    return False


def _payload_declares_clean(payload: Mapping[str, Any]) -> bool:
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
    for key, value in payload.items():
        point = _point(value)
        if "auroc" in key.lower() and point is not None and point >= 0.99:
            return True
    return False


def _sample_count(payload: Mapping[str, Any]) -> int:
    for key in ("n_examples_heldout", "n_examples", "n_pooled_examples", "n"):
        value = payload.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


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
    return (
        "0.9131" in joined
        and "experiment_2850_fover_dual_condition_integrity_v4" in joined
    )


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
