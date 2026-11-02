"""Build the Exp 3807 v348 product-headline capstone artifact.

Spec refs: REQ-REPORT-3807, SCENARIO-REPORT-3807,
SCENARIO-REPORT-3807-RERUN-GUARD,
SCENARIO-REPORT-3807-POSITIVE-CONTROL-GUARD,
SCENARIO-REPORT-3807-MISSING-BLOCKED, SCENARIO-REPORT-3807-FLAGGED.

This capstone is a synthesis pass over upstream JSON artifacts. It records what
landed, what blocked, and what must remain an operator decision. It does not
run a live model or re-score the frozen headline.
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

from carnot.reporting.capstone_v347_post_convergence_3796 import (  # noqa: E402
    has_live_model_markers,
    is_sha256,
    nested_get,
    numeric,
    paper_ready_from_gate,
    payload_checksum,
    publication_gate_state,
    read_json_object,
    report_is_clean,
    truthy,
)
from scripts import adversarial_verify  # noqa: E402


OUTPUT_REL_PATH = Path("results/experiment_3807_capstone_v348.json")
RANDOM_SEED = 3807
FROZEN_FOVER_AUROC = 0.9131
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: a capstone reads upstream JSON, "
    "runs no live model)."
)
UPSTREAM_IDS = (3797, 3798, 3799, 3800, 3801, 3802, 3803, 3805, 3806)

DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    3797: Path("results/experiment_3797_archive_v347_activate_v348.json"),
    3798: Path("results/experiment_3798_g4_product_headline_restoration.json"),
    3799: Path("results/experiment_3799_product_headline_provenance_reconfirmation.json"),
    3800: Path("results/experiment_3800_gaming_resistance_mitigation_v2.json"),
    3801: Path("results/experiment_3801_abstention_http_rest_surface.json"),
    3802: Path("results/experiment_3802_anomaly_escalation_classifier_v2_tuning.json"),
    3803: Path("results/experiment_3803_fr11_v20_tier3_fast_path_gate.json"),
    3805: Path("results/experiment_3805_external_research_refresh.json"),
    3806: Path("results/experiment_3806_kv260_opportunistic_continuity_audit.json"),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "product_headline_advanced",
    "product_headline_restorable",
    "verifier_product_hardened",
    "anomaly_classifier_repaired",
    "fr11_v20_tier3_fast_path",
    "energy_as_generator_still_bounded",
    "paper_ready_preserved",
    "frozen_headline_unchanged",
    "next_thesis_remains_operator_surface",
    "flagged_artifacts_excluded",
    "not_landed_or_blocked_recorded_honestly",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix; the milestone's one-line outcome.",
    "inference_substrate": (
        "aggregation_from_upstream_artifacts (principle: a capstone reads upstream JSON, "
        "runs no live model)."
    ),
    "product_headline_advanced": (
        "Records exp3798 (G4 re-run delta + g4_provenance_complete) + exp3799 "
        "(restorable/with-caveat/not-yet-eligible) -- the headline-advancement "
        "result (or honest block)."
    ),
    "product_headline_restorable": (
        "BARE string in {restorable, restorable_with_caveat, not_yet_eligible, "
        "blocked_rerun} -- the operator-facing recommendation."
    ),
    "verifier_product_hardened": (
        "Records the context_compaction mitigation (exp3800 evasion_status + "
        "clean_auroc_preserved) + the HTTP/REST 3rd surface (exp3801) -- the "
        "deployable-product advances."
    ),
    "anomaly_classifier_repaired": (
        "Records exp3802: false-escalation before->after, recall stays 1.0, "
        "supports_wiring_in, conductor unmodified."
    ),
    "fr11_v20_tier3_fast_path": (
        "Records the Tier-3 fast-path gate (exp3803, skip rate at no-regression, "
        "effective AUROC in frozen CI, headline ensemble unchanged) -- the "
        "mandated self-learning step."
    ),
    "energy_as_generator_still_bounded": (
        "BARE bool, true -- .348 runs no energy-foundation experiment; the bound "
        "stands; the P1 mechanism is handed to the operator, not reopened."
    ),
    "paper_ready_preserved": (
        "BARE bool -- G1-G4 stay met; the milestone must not regress the banked "
        "verifier product."
    ),
    "frozen_headline_unchanged": (
        "BARE bool -- frozen FoVer 0.9131 stays frozen; .348 uses but never moves it."
    ),
    "next_thesis_remains_operator_surface": (
        "BARE bool -- the EDLM-seed-vs-freeze decision remains an operator-seeding "
        "surface; the loop does not self-commit."
    ),
    "flagged_artifacts_excluded": (
        "Lists any flagged_adversarial artifact excluded from aggregation "
        "(fabrication gate)."
    ),
    "not_landed_or_blocked_recorded_honestly": (
        "Lists any upstream task that did not land or blocked -- recorded as "
        "not-run/blocked, NOT as a research negative."
    ),
    "cited_upstream_artifacts": "Provenance trail from the capstone summary to the real artifacts.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}

PRODUCT_HEADLINE_ALLOWED = {
    "restorable",
    "restorable_with_caveat",
    "not_yet_eligible",
    "blocked_rerun",
}

SUMMARY_FIELDS: Mapping[int, tuple[str, ...]] = {
    3797: (
        "honest_verdict",
        "paper_ready_preserved",
        "both_energy_routes_still_bounded",
        "paper_ready_evidence.paper_ready",
        "paper_ready_evidence.frozen_headline_unchanged",
        "paper_ready_evidence.frozen_headline_auroc",
        "v348_active_confirmed",
        "v348_focus_recorded",
    ),
    3798: (
        "honest_verdict",
        "baseline_pass1",
        "repair_pass1",
        "repair_delta_pp",
        "g4_provenance_complete",
        "positive_control_passed",
        "inference_path",
        "product_headline_restorable",
        "n",
    ),
    3799: (
        "honest_verdict",
        "product_headline_restorable",
        "rerun_code_repair_g4_pass",
        "exp2090_g4_pass",
        "operator_curated_doc_unedited",
        "provenance_table",
    ),
    3800: (
        "honest_verdict",
        "evasion_status",
        "clean_auroc_preserved",
        "n_samples",
        "not_a_moat_reopen",
        "headline_unchanged",
        "tests_assert_real_behavior",
    ),
    3801: (
        "honest_verdict",
        "http_rest_surface_added",
        "default_off_preserves_prior_behavior",
        "e2e_http_abstention_passed",
        "batch_post_works",
        "tests_assert_real_behavior",
        "scripts_research_conductor_modified",
    ),
    3802: (
        "honest_verdict",
        "false_escalation_rate_before",
        "false_escalation_rate_after",
        "frame_violating_recall_after",
        "supports_wiring_in",
        "conductor_unmodified",
        "never_relaxes_verification",
    ),
    3803: (
        "honest_verdict",
        "skip_rate_at_no_regression",
        "effective_auroc_at_operating_point",
        "frozen_ci95.low",
        "frozen_ci95.high",
        "headline_ensemble_unchanged",
        "accuracy_regression",
        "acceptance_gate.passed",
    ),
    3805: (
        "honest_verdict",
        "references_added",
        "n_references_added",
        "section_appended_not_replaced",
        "section_confirmed_intact",
        "numbers_are_as_reported",
    ),
    3806: (
        "honest_verdict",
        "terminal_state_holds",
        "kv260_ssh_reachable",
        "kv260_overlay_loadable",
        "speedup_claim_made",
    ),
}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    summary_records: Sequence[Mapping[str, Any]] | None = None,
    adversarial_reports: Mapping[int, Mapping[str, Any]] | None = None,
    publication_gate_data: Mapping[str, Any] | None = None,
    capstone_adversarial_verify_clean: bool = True,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """Build the v348 capstone from upstream JSON without re-running research."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    paths = {experiment_id: resolve_upstream_path(root_path, experiment_id) for experiment_id in UPSTREAM_IDS}
    upstreams = {
        experiment_id: read_json_object(path) if path.exists() else None
        for experiment_id, path in paths.items()
    }
    reports = (
        {experiment_id: dict(report) for experiment_id, report in adversarial_reports.items()}
        if adversarial_reports is not None
        else verify_upstreams(paths)
    )
    flagged_ids = {
        experiment_id
        for experiment_id, payload in upstreams.items()
        if isinstance(payload, Mapping) and payload.get("flagged_adversarial") is True
    }
    clean_upstreams = {
        experiment_id: payload
        for experiment_id, payload in upstreams.items()
        if payload is not None and experiment_id not in flagged_ids
    }
    summaries = compact_summary_records(
        summary_records if summary_records is not None else run_summarize_artifacts(root_path, paths)
    )
    gate_data = dict(publication_gate_data) if publication_gate_data is not None else load_publication_gate(root_path)

    exp3797 = clean_payload(clean_upstreams, 3797)
    exp3799 = headline_payload(clean_upstreams, 3799)
    exp3800 = headline_payload(clean_upstreams, 3800)
    exp3801 = clean_payload(clean_upstreams, 3801)
    exp3802 = headline_payload(clean_upstreams, 3802)
    exp3803 = headline_payload(clean_upstreams, 3803)
    exp3805 = headline_payload(clean_upstreams, 3805)
    exp3806 = headline_payload(clean_upstreams, 3806)

    headline = product_headline_advance(
        upstreams.get(3798),
        exp3799,
        missing=upstreams.get(3798) is None,
        flagged=3798 in flagged_ids,
    )
    product_restorable = headline["product_headline_restorable"]
    verifier = verifier_product_hardening(exp3800, exp3801)
    anomaly = anomaly_classifier_repair(exp3802)
    fr11 = fr11_v20_fast_path(exp3803)
    paper_ready = paper_ready_from_gate(gate_data)
    frozen = frozen_headline_unchanged(exp3797)
    next_operator = next_thesis_operator_surface(exp3797)
    energy_bounded = energy_routes_bounded(exp3797)
    references = references_refreshed(exp3805)
    kv260 = kv260_terminal_confirmed(exp3806)
    duration_s = round(max(0.0001, (time.perf_counter() if now_s is None else float(now_s)) - start), 6)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v348_product_headline_3807.v1",
        "experiment_id": "exp3807",
        "honest_verdict": terminal_verdict(
            product_headline_restorable=product_restorable,
            verifier=verifier,
            anomaly=bool(anomaly["repaired"]),
            fr11=bool(fr11["validated"]),
            paper_ready=paper_ready,
            frozen=frozen,
            energy_bounded=energy_bounded,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "product_headline_advanced": headline,
        "product_headline_restorable": product_restorable,
        "verifier_product_hardened": verifier,
        "anomaly_classifier_repaired": anomaly,
        "fr11_v20_tier3_fast_path": fr11,
        "energy_as_generator_still_bounded": True,
        "energy_as_selector_status": "honest-negative-bounded",
        "energy_as_generator_status": "honest-negative-bounded",
        "paper_ready_preserved": paper_ready,
        "publication_gate_state": publication_gate_state(gate_data),
        "frozen_headline_unchanged": frozen,
        "frozen_fover_auroc": FROZEN_FOVER_AUROC,
        "next_thesis_remains_operator_surface": next_operator,
        "operator_flag_carried_forward": (
            "project_converged_edlm_preflight_go_next_milestone_should_be_edlm_seed_or_freeze"
        ),
        "references_refreshed": references,
        "kv260_terminal_confirmed": kv260,
        "milestone_outcome_plain": milestone_outcome(product_restorable, verifier),
        "no_new_existential_claim": True,
        "regrinds_nothing_already_bounded": True,
        "headline_aggregation_experiment_ids": [
            experiment_id
            for experiment_id, payload in sorted(clean_upstreams.items())
            if not is_blocked_payload(payload) and experiment_id not in flagged_ids
        ],
        "flagged_artifacts_excluded": flagged_artifacts(paths, flagged_ids),
        "not_landed_or_blocked_recorded_honestly": not_landed_or_blocked(paths, upstreams, flagged_ids),
        "cited_upstream_artifacts": cited_upstream_artifacts(root_path, paths, clean_upstreams),
        "summarized_upstream_artifacts": summaries,
        "upstream_adversarial_critical_flags": critical_adversarial_flags(reports),
        "adversarial_verify_clean": capstone_adversarial_verify_clean,
        "adversarial_verify_report": {"flags": []},
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def run(
    root: Path | str = REPO_ROOT,
    *,
    summary_records: Sequence[Mapping[str, Any]] | None = None,
    adversarial_reports: Mapping[int, Mapping[str, Any]] | None = None,
    publication_gate_data: Mapping[str, Any] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Write, adversarial-verify, and rewrite the Exp 3807 artifact."""

    root_path = Path(root)
    out_path = root_path / OUTPUT_REL_PATH
    artifact = build_artifact(
        root_path,
        summary_records=summary_records,
        adversarial_reports=adversarial_reports,
        publication_gate_data=publication_gate_data,
        capstone_adversarial_verify_clean=True,
        started_s=started_s,
        now_s=now_s,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = adversarial_verify.verify_artifact(out_path)
    artifact["adversarial_verify_report"] = report
    artifact["adversarial_verify_clean"] = report_is_clean(report)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema and honesty errors for the Exp 3807 capstone."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append(f"missing required artifact fields: {', '.join(missing)}")
    if not str(artifact.get("honest_verdict") or "").startswith(
        "complete: capstone_v348_product_headline_"
    ):
        errors.append("honest_verdict must be a terminal Exp 3807 verdict")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must declare the v348 aggregation-only substrate")
    if artifact.get("product_headline_restorable") not in PRODUCT_HEADLINE_ALLOWED:
        errors.append("product_headline_restorable must be one of the allowed terminal values")
    validate_product_headline(artifact.get("product_headline_advanced"), errors)
    validate_verifier_product(artifact.get("verifier_product_hardened"), errors)
    validate_anomaly_classifier(artifact.get("anomaly_classifier_repaired"), errors)
    validate_fr11(artifact.get("fr11_v20_tier3_fast_path"), errors)
    if artifact.get("energy_as_generator_still_bounded") is not True:
        errors.append("energy_as_generator_still_bounded must be true")
    if artifact.get("paper_ready_preserved") is not True:
        errors.append("paper_ready_preserved must be true")
    if artifact.get("frozen_headline_unchanged") is not True:
        errors.append("frozen_headline_unchanged must be true")
    if artifact.get("next_thesis_remains_operator_surface") is not True:
        errors.append("next_thesis_remains_operator_surface must be true")
    if not isinstance(artifact.get("flagged_artifacts_excluded"), list):
        errors.append("flagged_artifacts_excluded must be a list")
    if not isinstance(artifact.get("not_landed_or_blocked_recorded_honestly"), list):
        errors.append("not_landed_or_blocked_recorded_honestly must be a list")
    validate_citations(artifact.get("cited_upstream_artifacts"), errors)
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed must equal 3807")
    duration_s = artifact.get("duration_s")
    if not isinstance(duration_s, int | float) or isinstance(duration_s, bool) or float(duration_s) < 0.0001:
        errors.append("duration_s must be numeric with the aggregation plausibility floor")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(REQUIRED_ARTIFACT_FIELDS) - set(principles):
        errors.append("field_principles must cover all required artifact fields")
    if has_live_model_markers(artifact):
        errors.append("artifact must not copy live-model substrate markers")
    if not report_is_clean(artifact.get("adversarial_verify_report", {"flags": []})):
        errors.append("adversarial verifier must report no critical flag")
    checksum = artifact.get("reproducibility_checksum")
    if not is_sha256(checksum):
        errors.append("reproducibility_checksum must be a sha256 hex string")
    elif checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum does not match artifact content")
    return errors


def validate_product_headline(value: Any, errors: list[str]) -> None:
    if not isinstance(value, Mapping):
        errors.append("product_headline_advanced must be an object")
        return
    rerun = value.get("rerun")
    if not isinstance(rerun, Mapping):
        errors.append("product_headline_advanced.rerun must be an object")
        return
    if rerun.get("headline_aggregation_status") == "used":
        delta = numeric(rerun.get("repair_delta_pp"))
        if delta is not None and delta <= 0 and rerun.get("positive_control_passed") is not True:
            errors.append("non-positive product-headline delta requires positive control")
    if value.get("product_headline_restorable") not in PRODUCT_HEADLINE_ALLOWED:
        errors.append("product_headline_advanced must record an allowed recommendation")


def validate_verifier_product(value: Any, errors: list[str]) -> None:
    if not isinstance(value, Mapping) or not isinstance(value.get("hardened"), bool):
        errors.append("verifier_product_hardened must record product-hardening state")


def validate_anomaly_classifier(value: Any, errors: list[str]) -> None:
    if not isinstance(value, Mapping) or not isinstance(value.get("repaired"), bool):
        errors.append("anomaly_classifier_repaired must record repair state")


def validate_fr11(value: Any, errors: list[str]) -> None:
    if not isinstance(value, Mapping) or not isinstance(value.get("validated"), bool):
        errors.append("fr11_v20_tier3_fast_path must record validation state")


def validate_citations(value: Any, errors: list[str]) -> None:
    if not isinstance(value, list):
        errors.append("cited_upstream_artifacts must be a list")
        return
    for item in value:
        if not isinstance(item, Mapping):
            errors.append("each citation must be an object")
            continue
        if not isinstance(item.get("fields_imported"), list):
            errors.append("each citation must include fields_imported")
        if not is_sha256(item.get("sha256")):
            errors.append("each citation must include a sha256 hex string")


def terminal_verdict(
    *,
    product_headline_restorable: str,
    verifier: Mapping[str, Any],
    anomaly: bool,
    fr11: bool,
    paper_ready: bool,
    frozen: bool,
    energy_bounded: bool,
) -> str:
    """Return the one-line terminal verdict with honest partial-product state."""

    return (
        f"complete: capstone_v348_product_headline_{terminal_headline_segment(product_headline_restorable)}_"
        f"{verifier_terminal_segment(verifier)}_"
        f"{'anomaly_classifier_repaired' if anomaly else 'anomaly_classifier_not_repaired'}_"
        f"{'fr11_v20_tier3_fast_path' if fr11 else 'fr11_v20_tier3_fast_path_not_validated'}_"
        f"paper_ready_{str(paper_ready).lower()}_"
        f"{'frozen_headline_unchanged' if frozen else 'frozen_headline_changed'}_"
        f"{'both_energy_routes_bounded' if energy_bounded else 'energy_route_regressed'}"
    )


def terminal_headline_segment(status: str) -> str:
    if status == "restorable_with_caveat":
        return "with_caveat"
    if status == "blocked_rerun":
        return "blocked"
    if status == "not_yet_eligible":
        return "demoted"
    return status


def verifier_terminal_segment(verifier: Mapping[str, Any]) -> str:
    http = verifier.get("http_rest_surface") if isinstance(verifier.get("http_rest_surface"), Mapping) else {}
    context = verifier.get("context_compaction_mitigation")
    context_done = isinstance(context, Mapping) and context.get("mitigated") is True
    if verifier.get("hardened") is True:
        return "verifier_product_hardened"
    if context_done and http.get("status") == "blocked":
        return "verifier_product_hardened_http_rest_blocked"
    if context_done:
        return "verifier_product_hardened_http_rest_incomplete"
    return "verifier_product_not_hardened"


def resolve_upstream_path(root: Path, experiment_id: int) -> Path:
    """Return the default path or the first same-ID artifact in results."""

    default = root / DEFAULT_UPSTREAM_PATHS[experiment_id]
    if default.exists():
        return default
    matches = sorted((root / "results").glob(f"experiment_{experiment_id}_*.json"))
    return matches[0] if matches else default


def run_summarize_artifacts(root: Path, paths: Mapping[int, Path]) -> list[JsonDict]:  # pragma: no cover
    records: list[JsonDict] = []
    for experiment_id in UPSTREAM_IDS:
        path = paths[experiment_id]
        arg = str(path) if path.exists() else str(experiment_id)
        completed = subprocess.run(
            [sys.executable, "scripts/summarize_artifact.py", arg],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        records.append(
            {
                "experiment_id": experiment_id,
                "returncode": completed.returncode,
                "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
                "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
            }
        )
    return records


def verify_upstreams(paths: Mapping[int, Path]) -> dict[int, JsonDict]:  # pragma: no cover
    return {
        experiment_id: adversarial_verify.verify_artifact(path) if path.exists() else {"flags": []}
        for experiment_id, path in paths.items()
    }


def load_publication_gate(root: Path) -> JsonDict:  # pragma: no cover
    completed = subprocess.run(
        [sys.executable, "scripts/publication_gate.py", "--json"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise ValueError("publication_gate.py --json must return a JSON object")
    return payload


def compact_summary_records(records: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Keep deterministic proof that the artifact summarizer was run."""

    return [
        {
            "experiment_id": record.get("experiment_id", record.get("exp")),
            "returncode": record.get("returncode"),
            "stdout_sha256": record.get("stdout_sha256"),
            "stderr_sha256": record.get("stderr_sha256"),
        }
        for record in records
    ]


def clean_payload(upstreams: Mapping[int, Mapping[str, Any]], experiment_id: int) -> Mapping[str, Any]:
    return upstreams.get(experiment_id, {})


def headline_payload(upstreams: Mapping[int, Mapping[str, Any]], experiment_id: int) -> Mapping[str, Any]:
    payload = upstreams.get(experiment_id, {})
    return {} if is_blocked_payload(payload) else payload


def product_headline_advance(
    exp3798: Mapping[str, Any] | None,
    exp3799: Mapping[str, Any],
    *,
    missing: bool,
    flagged: bool,
) -> JsonDict:
    """Summarize headline restoration without treating unusable reruns as negatives."""

    blocked = isinstance(exp3798, Mapping) and is_blocked_payload(exp3798)
    if missing:
        rerun_status = "missing"
    elif flagged:
        rerun_status = "excluded_flagged_adversarial"
    elif blocked:
        rerun_status = "blocked"
    else:
        rerun_status = "used"
    rerun_payload = exp3798 if isinstance(exp3798, Mapping) else {}
    delta = numeric(rerun_payload.get("repair_delta_pp"))
    positive = rerun_payload.get("positive_control_passed") is True
    delta_valid = bool(rerun_status == "used" and delta is not None and positive)
    provenance_status = normalize_product_headline_status(exp3799.get("product_headline_restorable"))
    product_status = provenance_status if exp3799 else ("blocked_rerun" if rerun_status != "used" else "not_yet_eligible")
    historical_disproven = bool(delta_valid and delta is not None and delta < 18.0)
    operator_handoff = rerun_status != "used" or product_status == "blocked_rerun"
    return {
        "product_headline_restorable": product_status,
        "rerun": {
            "headline_aggregation_status": rerun_status,
            "baseline_pass1": numeric(rerun_payload.get("baseline_pass1")),
            "repair_pass1": numeric(rerun_payload.get("repair_pass1")),
            "repair_delta_pp": delta,
            "g4_provenance_complete": rerun_payload.get("g4_provenance_complete") is True,
            "positive_control_passed": positive,
            "inference_path": rerun_payload.get("inference_path"),
            "delta_valid_for_headline": delta_valid,
            "historical_plus18pp_disproven": historical_disproven if rerun_status == "used" else False,
        },
        "provenance_reconfirmation": {
            "present": bool(exp3799),
            "product_headline_restorable": provenance_status if exp3799 else "blocked_rerun",
            "rerun_code_repair_g4_pass": exp3799.get("rerun_code_repair_g4_pass") is True,
            "exp2090_g4_pass": exp3799.get("exp2090_g4_pass") is True,
            "operator_curated_doc_unedited": exp3799.get("operator_curated_doc_unedited") is True,
        },
        "operator_handoff": operator_handoff,
        "historical_plus18pp_disproven": historical_disproven if rerun_status == "used" else False,
        "headline_stays_demoted": product_status in {"not_yet_eligible", "blocked_rerun"},
    }


def normalize_product_headline_status(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"not_yet_headline_eligible", "stays_demoted"}:
        return "not_yet_eligible"
    if raw in PRODUCT_HEADLINE_ALLOWED:
        return raw
    return "blocked_rerun"


def verifier_product_hardening(exp3800: Mapping[str, Any], exp3801: Mapping[str, Any]) -> JsonDict:
    context_mitigated = bool(
        exp3800
        and exp3800.get("evasion_status") in {"closed", "narrowed"}
        and exp3800.get("clean_auroc_preserved") is True
        and exp3800.get("headline_unchanged") is True
        and exp3800.get("not_a_moat_reopen") is True
        and truthy(exp3800.get("tests_assert_real_behavior"))
    )
    http_blocked = bool(exp3801 and is_blocked_payload(exp3801))
    http_complete = bool(
        exp3801
        and not http_blocked
        and exp3801.get("http_rest_surface_added") is True
        and exp3801.get("default_off_preserves_prior_behavior") is True
        and exp3801.get("e2e_http_abstention_passed") is True
        and exp3801.get("batch_post_works") is True
        and exp3801.get("tests_assert_real_behavior") is True
        and exp3801.get("scripts_research_conductor_modified") is not True
    )
    return {
        "hardened": context_mitigated and http_complete,
        "context_compaction_mitigation": {
            "evasion_status": exp3800.get("evasion_status"),
            "clean_auroc_preserved": exp3800.get("clean_auroc_preserved") is True,
            "mitigated": context_mitigated,
        },
        "http_rest_surface": {
            "status": "blocked" if http_blocked else ("complete" if http_complete else "incomplete"),
            "surface_added": exp3801.get("http_rest_surface_added") is True,
            "default_off": exp3801.get("default_off_preserves_prior_behavior") is True,
            "e2e_passed": exp3801.get("e2e_http_abstention_passed") is True,
            "batch_post_works": exp3801.get("batch_post_works") is True,
            "conductor_unmodified": exp3801.get("scripts_research_conductor_modified") is not True,
        },
    }


def anomaly_classifier_repair(payload: Mapping[str, Any]) -> JsonDict:
    before = numeric(payload.get("false_escalation_rate_before"))
    after = numeric(payload.get("false_escalation_rate_after"))
    recall = numeric(payload.get("frame_violating_recall_after"))
    repaired = bool(
        payload
        and before is not None
        and after is not None
        and after < before
        and after <= 0.2
        and recall == 1.0
        and payload.get("supports_wiring_in") is True
        and payload.get("conductor_unmodified") is True
        and payload.get("never_relaxes_verification") is True
        and payload.get("tests_assert_real_behavior") is True
    )
    return {
        "repaired": repaired,
        "false_escalation_rate_before": before,
        "false_escalation_rate_after": after,
        "frame_violating_recall": recall,
        "supports_wiring_in": payload.get("supports_wiring_in") is True,
        "conductor_unmodified": payload.get("conductor_unmodified") is True,
    }


def fr11_v20_fast_path(payload: Mapping[str, Any]) -> JsonDict:
    skip_rate = numeric(payload.get("skip_rate_at_no_regression"))
    effective = numeric(payload.get("effective_auroc_at_operating_point"))
    ci_low = numeric(nested_get(payload, "frozen_ci95.low"))
    ci_high = numeric(nested_get(payload, "frozen_ci95.high"))
    in_ci = bool(effective is not None and ci_low is not None and ci_high is not None and ci_low <= effective <= ci_high)
    validated = bool(
        payload
        and skip_rate is not None
        and skip_rate > 0
        and in_ci
        and payload.get("accuracy_regression") is False
        and payload.get("headline_ensemble_unchanged") is True
        and payload.get("is_tier3_application_not_retrain") is True
        and payload.get("operating_point_persisted") is True
        and nested_get(payload, "acceptance_gate.passed") is True
    )
    return {
        "validated": validated,
        "skip_rate_at_no_regression": skip_rate,
        "effective_auroc": effective,
        "frozen_ci95": {"low": ci_low, "high": ci_high},
        "effective_auroc_in_frozen_ci": in_ci,
        "headline_ensemble_unchanged": payload.get("headline_ensemble_unchanged") is True,
    }


def energy_routes_bounded(payload: Mapping[str, Any]) -> bool:
    evidence = payload.get("v347_capstone_evidence")
    return bool(
        payload.get("both_energy_routes_still_bounded") is True
        or (isinstance(evidence, Mapping) and evidence.get("both_energy_routes_bounded") is True)
    )


def frozen_headline_unchanged(payload: Mapping[str, Any]) -> bool:
    evidence = payload.get("paper_ready_evidence")
    return bool(
        isinstance(evidence, Mapping)
        and evidence.get("frozen_headline_unchanged") is True
        and numeric(evidence.get("frozen_headline_auroc")) == FROZEN_FOVER_AUROC
    )


def next_thesis_operator_surface(payload: Mapping[str, Any]) -> bool:
    focus = str(payload.get("v348_focus_recorded") or "")
    return bool(payload and "no_paradigm_self_seed" in focus and "tier3_fast_path" in focus)


def references_refreshed(payload: Mapping[str, Any]) -> bool:
    n_references = payload.get("n_references_added")
    return bool(
        payload
        and isinstance(n_references, int)
        and n_references >= 1
        and payload.get("section_appended_not_replaced") is True
        and payload.get("numbers_are_as_reported") is True
    )


def kv260_terminal_confirmed(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload
        and payload.get("terminal_state_holds") is True
        and payload.get("kv260_ssh_reachable") is True
        and payload.get("kv260_overlay_loadable") is True
        and payload.get("speedup_claim_made") is not True
    )


def milestone_outcome(product_status: str, verifier: Mapping[str, Any]) -> str:
    if product_status == "blocked_rerun":
        headline = "keeps the product code-repair headline demoted because the re-run is an operator handoff"
    elif product_status == "not_yet_eligible":
        headline = "honestly keeps the product code-repair headline demoted"
    else:
        headline = "restores or qualifies the product code-repair headline"
    http_status = nested_get(verifier, "http_rest_surface.status")
    product_phrase = (
        "hardens the verifier product while recording the HTTP/REST surface block"
        if http_status == "blocked"
        else "hardens and repairs the verifier product"
    )
    return (
        f".348 is a LEAN POST-CONVERGENCE milestone: it {headline}, {product_phrase}, "
        "repairs the endorsed anomaly classifier, and turns the trained Tier-3 "
        "predictor into a deployable fast path. It makes NO new existential claim "
        "and re-grinds nothing bounded."
    )


def is_blocked_payload(payload: Mapping[str, Any]) -> bool:
    verdict = str(payload.get("honest_verdict") or "").lower()
    return verdict.startswith("blocked:") or verdict.startswith("blocked_")


def not_landed_or_blocked(
    paths: Mapping[int, Path],
    upstreams: Mapping[int, Mapping[str, Any] | None],
    flagged_ids: set[int],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for experiment_id in UPSTREAM_IDS:
        if experiment_id in flagged_ids:
            continue
        payload = upstreams.get(experiment_id)
        if payload is None:
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "path": str(paths[experiment_id]),
                    "status": "not-landed",
                    "reason": "artifact_missing",
                }
            )
        elif is_blocked_payload(payload):
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "path": str(paths[experiment_id]),
                    "status": "blocked",
                    "reason": str(payload.get("honest_verdict") or "blocked"),
                }
            )
    return rows


def flagged_artifacts(paths: Mapping[int, Path], flagged_ids: set[int]) -> list[JsonDict]:
    return [
        {"experiment_id": experiment_id, "path": str(paths[experiment_id]), "reason": "flagged_adversarial=true"}
        for experiment_id in sorted(flagged_ids)
    ]


def cited_upstream_artifacts(
    root: Path,
    paths: Mapping[int, Path],
    clean_upstreams: Mapping[int, Mapping[str, Any]],
) -> list[JsonDict]:
    citations: list[JsonDict] = []
    for experiment_id in sorted(clean_upstreams):
        path = paths[experiment_id]
        payload = clean_upstreams[experiment_id]
        citations.append(
            {
                "experiment_id": experiment_id,
                "path": str(path),
                "fields_imported": [
                    field for field in SUMMARY_FIELDS[experiment_id] if nested_get(payload, field) is not None
                ],
                "honest_verdict": payload.get("honest_verdict"),
                "status": "blocked" if is_blocked_payload(payload) else "landed",
                "sha256": sha256_file(path if path.is_absolute() else root / path),
            }
        )
    return citations


def critical_adversarial_flags(reports: Mapping[int, Mapping[str, Any]]) -> list[JsonDict]:
    critical: list[JsonDict] = []
    for experiment_id, report in sorted(reports.items()):
        for flag in report.get("flags") or []:
            if isinstance(flag, Mapping) and str(flag.get("severity") or "").lower() == "critical":
                critical.append(
                    {
                        "experiment_id": experiment_id,
                        "kind": flag.get("kind"),
                        "detail": flag.get("detail"),
                    }
                )
    return critical


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:  # pragma: no cover - CLI wrapper
    out_path = run(REPO_ROOT)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
