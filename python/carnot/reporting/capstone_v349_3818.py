"""Build the Exp 3818 v349 lean-maintenance capstone artifact.

Spec refs: REQ-REPORT-3818, SCENARIO-REPORT-3818,
SCENARIO-REPORT-3818-BLOCKED-PARITY, SCENARIO-REPORT-3818-FLAGGED.

The capstone is a disciplined aggregation over already-written upstream
artifacts. It records what landed, what blocked, and which operator decision
remains open. It does not run a model or move the frozen FoVer headline.
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

from carnot.reporting.capstone_v347_post_convergence_3796 import (
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
from scripts import adversarial_verify


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3818_capstone_v349.json")
RANDOM_SEED = 3818
FROZEN_FOVER_AUROC = 0.9131
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: a capstone reads upstream JSON, "
    "runs no live model)."
)
TERMINAL_VERDICT = (
    "complete: capstone_v349_anomaly_advisory_wired_http_rest_repaired_"
    "parity_confirmed_product_headline_stays_demoted_fr11_v21_self_learning_"
    "publication_gate_confirmed_edlm_seed_staged_paper_ready_true_"
    "frozen_headline_unchanged_both_energy_routes_bounded"
)
UPSTREAM_IDS = tuple(range(3808, 3818))

DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    3808: Path("results/experiment_3808_archive_v348_activate_v349.json"),
    3809: Path("results/experiment_3809_anomaly_escalation_advisory_hook.json"),
    3810: Path("results/experiment_3810_abstention_http_rest_surface_v2.json"),
    3811: Path("results/experiment_3811_abstention_cross_surface_parity_smoke.json"),
    3812: Path("results/experiment_3812_product_headline_status_consolidation.json"),
    3813: Path("results/experiment_3813_fr11_v21_fast_path_robustness.json"),
    3814: Path("results/experiment_3814_publication_gate_regression_confirmation.json"),
    3815: Path("results/experiment_3815_edlm_operator_seed_staging_package.json"),
    3816: Path("results/experiment_3816_external_research_refresh_v349.json"),
    3817: Path("results/experiment_3817_kv260_opportunistic_continuity_audit.json"),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "anomaly_advisory_hook_wired",
    "verifier_product_repaired",
    "product_headline_status",
    "fr11_v21_self_learning",
    "publication_gate_confirmed",
    "edlm_seed_staged",
    "energy_routes_still_bounded",
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
    "inference_substrate": INFERENCE_SUBSTRATE,
    "anomaly_advisory_hook_wired": (
        "Records exp3809: recommend-only advisory wired, replay recall 1.0, "
        "never_relaxes_verification, conductor unmodified, integration proposal emitted."
    ),
    "verifier_product_repaired": (
        "Records exp3810 (HTTP/REST 3rd surface e2e_passed) + exp3811 "
        "(cross-surface parity, or honest block) -- the deployable-product advances."
    ),
    "product_headline_status": (
        "BARE string in {stays_demoted, restorable_via_operator_gpu_rerun} -- "
        "exp3812's honest recommendation; both candidate positives fail provenance, "
        "FoVer 0.9131 sole defensible."
    ),
    "fr11_v21_self_learning": (
        "Records exp3813: fast-path operating-point robustness on the second split "
        "(skip, effective AUROC, operating_point_generalizes, headline ensemble "
        "unchanged) -- the mandated self-learning step."
    ),
    "publication_gate_confirmed": (
        "Records exp3814: G1-G4 pass, paper_ready true, frozen 0.9131 unchanged, "
        "no gate redefined."
    ),
    "edlm_seed_staged": (
        "Records exp3815: one-command seed packaged + kill-gate design, "
        "loop_does_not_seed, operator-gated."
    ),
    "energy_routes_still_bounded": (
        "BARE bool, true -- .349 runs no energy-foundation experiment; both bounds "
        "stand; nothing reopened."
    ),
    "paper_ready_preserved": (
        "BARE bool -- G1-G4 stay met; the milestone must not regress the banked "
        "verifier product."
    ),
    "frozen_headline_unchanged": (
        "BARE bool -- frozen FoVer 0.9131 stays frozen; .349 uses but never moves it."
    ),
    "next_thesis_remains_operator_surface": (
        "BARE bool -- the EDLM-seed-vs-freeze decision remains an operator-seeding "
        "surface; the loop does not self-commit; the milestone after .349 should "
        "be the seed or an explicit freeze."
    ),
    "flagged_artifacts_excluded": (
        "Lists any flagged_adversarial artifact excluded from aggregation "
        "(fabrication gate)."
    ),
    "not_landed_or_blocked_recorded_honestly": (
        "Lists any upstream task that did not land or blocked (e.g. exp3811 if its "
        "gate did not fire) -- recorded as not-run/blocked, NOT as a research "
        "negative (the .344 capstone-confusion guard)."
    ),
    "cited_upstream_artifacts": "Provenance trail from the capstone summary to the real artifacts.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}

PRODUCT_HEADLINE_ALLOWED = {"stays_demoted", "restorable_via_operator_gpu_rerun"}

SUMMARY_FIELDS: Mapping[int, tuple[str, ...]] = {
    3808: (
        "honest_verdict",
        "paper_ready_preserved",
        "both_energy_routes_still_bounded",
        "edlm_remains_operator_seed_surface",
        "v349_active_confirmed",
    ),
    3809: (
        "honest_verdict",
        "advisory_module_added",
        "offline_replay_false_escalation_rate",
        "offline_replay_frame_violating_recall",
        "never_relaxes_verification",
        "conductor_unmodified",
        "integration_proposal_emitted",
    ),
    3810: (
        "honest_verdict",
        "http_rest_surface_added",
        "default_off_preserves_prior_behavior",
        "e2e_http_abstention_passed",
        "batch_post_works",
        "doc_proposal_emitted_not_curated_edit",
        "tests_assert_real_behavior",
    ),
    3811: (
        "honest_verdict",
        "all_surfaces_agree",
        "surfaces_compared",
        "n_candidates_compared",
        "mismatches",
        "tests_assert_real_behavior",
    ),
    3812: (
        "honest_verdict",
        "product_headline_recommendation",
        "code_repair_supports_headline",
        "crane_supports_headline",
        "sole_defensible_headline",
        "operator_restore_path",
        "operator_curated_doc_unedited",
    ),
    3813: (
        "honest_verdict",
        "continuous_self_learning_task",
        "skip_rate_second_split",
        "effective_auroc_second_split",
        "operating_point_generalizes",
        "headline_ensemble_unchanged",
        "acceptance_gate.passed",
    ),
    3814: (
        "honest_verdict",
        "g1_pass",
        "g2_pass",
        "g3_pass",
        "g4_pass",
        "paper_ready",
        "frozen_fover_auroc",
        "frozen_fover_auroc_unchanged",
        "gate_definitions_unchanged",
    ),
    3815: (
        "honest_verdict",
        "staging_note_written",
        "operator_seed_command",
        "kill_gate_design_documented",
        "loop_does_not_seed",
        "edlm_remains_operator_gated",
        "operator_curated_doc_unedited",
    ),
    3816: (
        "honest_verdict",
        "references_section_intact",
        "references_added",
        "n_references_added",
        "section_appended_not_replaced",
        "numbers_are_as_reported",
    ),
    3817: (
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
    """Build the v349 capstone from upstream JSON without re-running research."""

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

    exp3808 = usable_payload(clean_upstreams, 3808)
    exp3809 = usable_payload(clean_upstreams, 3809)
    exp3810 = clean_upstreams.get(3810, {})
    exp3811 = clean_upstreams.get(3811, {})
    exp3812 = usable_payload(clean_upstreams, 3812)
    exp3813 = usable_payload(clean_upstreams, 3813)
    exp3814 = usable_payload(clean_upstreams, 3814)
    exp3815 = usable_payload(clean_upstreams, 3815)
    exp3816 = usable_payload(clean_upstreams, 3816)
    exp3817 = usable_payload(clean_upstreams, 3817)

    advisory = anomaly_advisory_hook(exp3809)
    verifier = verifier_product(exp3810, exp3811, status_for(upstreams, flagged_ids, 3810), status_for(upstreams, flagged_ids, 3811))
    product_status = product_headline_status(exp3812)
    fr11 = fr11_v21_self_learning(exp3813)
    publication = publication_gate_confirmation(exp3814, gate_data)
    edlm = edlm_seed(exp3815)
    energy_bounded = energy_routes_bounded(exp3808)
    paper_ready = bool(publication["paper_ready"])
    frozen = bool(publication["frozen_headline_unchanged"])
    next_operator = next_thesis_operator_surface(exp3808, edlm)
    references = references_refreshed(exp3816)
    kv260 = kv260_terminal_confirmed(exp3817)
    duration_s = round(max(0.0001, (time.perf_counter() if now_s is None else float(now_s)) - start), 6)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v349_3818.v1",
        "experiment_id": "exp3818",
        "honest_verdict": terminal_verdict(
            advisory_wired=bool(advisory["wired"]),
            verifier=verifier,
            product_headline_status=product_status,
            fr11_continued=bool(fr11["continued"]),
            publication_confirmed=bool(publication["confirmed"]),
            edlm_staged=bool(edlm["staged"]),
            paper_ready=paper_ready,
            frozen=frozen,
            energy_bounded=energy_bounded,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "anomaly_advisory_hook_wired": advisory,
        "verifier_product_repaired": verifier,
        "product_headline_status": product_status,
        "product_headline_evidence": product_headline_evidence(exp3812),
        "fr11_v21_self_learning": fr11,
        "publication_gate_confirmed": publication,
        "publication_gate_state": publication_gate_state(gate_data),
        "edlm_seed_staged": edlm,
        "references_refreshed": references,
        "kv260_terminal_confirmed": kv260,
        "energy_routes_still_bounded": energy_bounded,
        "energy_as_selector_status": "honest-negative-bounded",
        "energy_as_generator_status": "honest-negative-bounded",
        "paper_ready_preserved": paper_ready,
        "frozen_headline_unchanged": frozen,
        "frozen_fover_auroc": FROZEN_FOVER_AUROC,
        "next_thesis_remains_operator_surface": next_operator,
        "operator_flag_carried_forward": (
            "project_converged_edlm_preflight_go_seed_staged_one_command_"
            "next_milestone_edlm_seed_or_explicit_freeze_not_a_fifth_deferral"
        ),
        "milestone_outcome_plain": milestone_outcome(verifier),
        "no_new_existential_claim": True,
        "regrinds_nothing_bounded": True,
        "headline_aggregation_experiment_ids": [
            experiment_id
            for experiment_id, payload in sorted(clean_upstreams.items())
            if not is_blocked_payload(payload)
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
    """Write, adversarial-verify, and rewrite the Exp 3818 artifact."""

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
    """Return schema and honesty errors for the Exp 3818 capstone."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append(f"missing required artifact fields: {', '.join(missing)}")
    if not str(artifact.get("honest_verdict") or "").startswith(
        "complete: capstone_v349_anomaly_advisory_"
    ):
        errors.append("honest_verdict must be a terminal Exp 3818 verdict")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must declare the v349 aggregation-only substrate")
    if artifact.get("product_headline_status") not in PRODUCT_HEADLINE_ALLOWED:
        errors.append("product_headline_status must be one of the allowed terminal values")
    validate_mapping_bool(artifact.get("anomaly_advisory_hook_wired"), "wired", "anomaly_advisory_hook_wired", errors)
    validate_mapping_bool(artifact.get("verifier_product_repaired"), "product_repaired", "verifier_product_repaired", errors)
    validate_mapping_bool(artifact.get("fr11_v21_self_learning"), "continued", "fr11_v21_self_learning", errors)
    validate_mapping_bool(artifact.get("publication_gate_confirmed"), "confirmed", "publication_gate_confirmed", errors)
    validate_mapping_bool(artifact.get("edlm_seed_staged"), "staged", "edlm_seed_staged", errors)
    for key in (
        "energy_routes_still_bounded",
        "paper_ready_preserved",
        "frozen_headline_unchanged",
        "next_thesis_remains_operator_surface",
    ):
        if artifact.get(key) is not True:
            errors.append(f"{key} must be true")
    if not isinstance(artifact.get("flagged_artifacts_excluded"), list):
        errors.append("flagged_artifacts_excluded must be a list")
    if not isinstance(artifact.get("not_landed_or_blocked_recorded_honestly"), list):
        errors.append("not_landed_or_blocked_recorded_honestly must be a list")
    validate_citations(artifact.get("cited_upstream_artifacts"), errors)
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed must equal 3818")
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


def validate_mapping_bool(value: Any, bool_key: str, name: str, errors: list[str]) -> None:
    if not isinstance(value, Mapping) or not isinstance(value.get(bool_key), bool):
        errors.append(f"{name} must record {bool_key} state")


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
    advisory_wired: bool,
    verifier: Mapping[str, Any],
    product_headline_status: str,
    fr11_continued: bool,
    publication_confirmed: bool,
    edlm_staged: bool,
    paper_ready: bool,
    frozen: bool,
    energy_bounded: bool,
) -> str:
    if (
        advisory_wired
        and verifier.get("product_repaired") is True
        and product_headline_status == "stays_demoted"
        and fr11_continued
        and publication_confirmed
        and edlm_staged
        and paper_ready
        and frozen
        and energy_bounded
    ):
        return TERMINAL_VERDICT
    verifier_segment = verifier_terminal_segment(verifier)
    headline_segment = (
        "stays_demoted"
        if product_headline_status == "stays_demoted"
        else "restorable_via_operator_gpu_rerun"
    )
    return (
        f"complete: capstone_v349_anomaly_advisory_{'wired' if advisory_wired else 'not_wired'}_"
        f"{verifier_segment}_product_headline_{headline_segment}_"
        f"{'fr11_v21_self_learning' if fr11_continued else 'fr11_v21_self_learning_not_continued'}_"
        f"{'publication_gate_confirmed' if publication_confirmed else 'publication_gate_not_confirmed'}_"
        f"{'edlm_seed_staged' if edlm_staged else 'edlm_seed_not_staged'}_"
        f"paper_ready_{str(paper_ready).lower()}_"
        f"{'frozen_headline_unchanged' if frozen else 'frozen_headline_changed'}_"
        f"{'both_energy_routes_bounded' if energy_bounded else 'energy_routes_not_confirmed'}"
    )


def verifier_terminal_segment(verifier: Mapping[str, Any]) -> str:
    http = verifier.get("http_rest_surface") if isinstance(verifier.get("http_rest_surface"), Mapping) else {}
    parity = (
        verifier.get("cross_surface_parity")
        if isinstance(verifier.get("cross_surface_parity"), Mapping)
        else {}
    )
    http_segment = "http_rest_repaired" if http.get("status") == "repaired" else "http_rest_blocked"
    parity_segment = "parity_confirmed" if parity.get("status") == "confirmed" else "parity_blocked"
    return f"{http_segment}_{parity_segment}"


def resolve_upstream_path(root: Path, experiment_id: int) -> Path:
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
    return [
        {
            "experiment_id": record.get("experiment_id", record.get("exp")),
            "returncode": record.get("returncode"),
            "stdout_sha256": record.get("stdout_sha256"),
            "stderr_sha256": record.get("stderr_sha256"),
        }
        for record in records
    ]


def status_for(
    upstreams: Mapping[int, Mapping[str, Any] | None],
    flagged_ids: set[int],
    experiment_id: int,
) -> str:
    if experiment_id in flagged_ids:
        return "flagged"
    payload = upstreams.get(experiment_id)
    if payload is None:
        return "not-landed"
    if is_blocked_payload(payload):
        return "blocked"
    return "landed"


def usable_payload(upstreams: Mapping[int, Mapping[str, Any]], experiment_id: int) -> Mapping[str, Any]:
    payload = upstreams.get(experiment_id, {})
    return {} if is_blocked_payload(payload) else payload


def anomaly_advisory_hook(payload: Mapping[str, Any]) -> JsonDict:
    recall = numeric(payload.get("offline_replay_frame_violating_recall"))
    false_escalation = numeric(payload.get("offline_replay_false_escalation_rate"))
    wired = bool(
        payload
        and payload.get("advisory_module_added") is True
        and false_escalation is not None
        and false_escalation <= 0.0
        and recall == 1.0
        and payload.get("never_relaxes_verification") is True
        and payload.get("conductor_unmodified") is True
        and payload.get("integration_proposal_emitted") is True
    )
    return {
        "wired": wired,
        "recommend_only": "recommend_only" in str(payload.get("honest_verdict") or ""),
        "offline_replay_false_escalation_rate": false_escalation,
        "offline_replay_frame_violating_recall": recall,
        "never_relaxes_verification": payload.get("never_relaxes_verification") is True,
        "conductor_unmodified": payload.get("conductor_unmodified") is True,
        "integration_proposal_emitted": payload.get("integration_proposal_emitted") is True,
    }


def verifier_product(
    exp3810: Mapping[str, Any],
    exp3811: Mapping[str, Any],
    exp3810_status: str,
    exp3811_status: str,
) -> JsonDict:
    http_repaired = bool(
        exp3810_status == "landed"
        and exp3810.get("http_rest_surface_added") is True
        and exp3810.get("default_off_preserves_prior_behavior") is True
        and exp3810.get("e2e_http_abstention_passed") is True
        and exp3810.get("batch_post_works") is True
        and exp3810.get("tests_assert_real_behavior") is True
        and exp3810.get("scripts_research_conductor_modified") is not True
    )
    parity_confirmed = bool(
        exp3811_status == "landed"
        and exp3811.get("all_surfaces_agree") is True
        and isinstance(exp3811.get("surfaces_compared"), list)
        and set(exp3811.get("surfaces_compared", [])) == {"verify_api", "cli", "http_rest"}
        and numeric(exp3811.get("n_candidates_compared")) is not None
        and numeric(exp3811.get("n_candidates_compared")) >= 1
        and exp3811.get("mismatches") == []
        and exp3811.get("tests_assert_real_behavior") is True
    )
    http_status = "repaired" if http_repaired else exp3810_status
    parity_status = "confirmed" if parity_confirmed else exp3811_status
    return {
        "product_repaired": http_repaired and parity_confirmed,
        "http_rest_surface": {
            "status": http_status,
            "third_surface_added": exp3810.get("http_rest_surface_added") is True,
            "default_off": exp3810.get("default_off_preserves_prior_behavior") is True,
            "e2e_passed": exp3810.get("e2e_http_abstention_passed") is True,
            "batch_post_works": exp3810.get("batch_post_works") is True,
            "conductor_unmodified": exp3810.get("scripts_research_conductor_modified") is not True,
        },
        "cross_surface_parity": {
            "status": parity_status,
            "all_surfaces_agree": exp3811.get("all_surfaces_agree") is True,
            "surfaces_compared": exp3811.get("surfaces_compared") if isinstance(exp3811.get("surfaces_compared"), list) else [],
            "n_candidates_compared": numeric(exp3811.get("n_candidates_compared")),
            "mismatches": exp3811.get("mismatches") if isinstance(exp3811.get("mismatches"), list) else None,
        },
        "not_a_research_negative": True,
    }


def product_headline_status(payload: Mapping[str, Any]) -> str:
    raw = str(payload.get("product_headline_recommendation") or "").strip().lower()
    if raw in PRODUCT_HEADLINE_ALLOWED:
        return raw
    return "stays_demoted"


def product_headline_evidence(payload: Mapping[str, Any]) -> JsonDict:
    restore_path = str(payload.get("operator_restore_path") or "")
    return {
        "code_repair_supports_headline": payload.get("code_repair_supports_headline") is True,
        "crane_supports_headline": payload.get("crane_supports_headline") is True,
        "sole_defensible_headline": payload.get("sole_defensible_headline"),
        "operator_gpu_rerun_handoff": bool("gpu" in restore_path.lower() or restore_path),
        "operator_curated_doc_unedited": payload.get("operator_curated_doc_unedited") is True,
    }


def fr11_v21_self_learning(payload: Mapping[str, Any]) -> JsonDict:
    skip = numeric(payload.get("skip_rate_second_split"))
    effective = numeric(payload.get("effective_auroc_second_split"))
    gate_passed = nested_get(payload, "acceptance_gate.passed") is True
    continued = bool(
        payload
        and payload.get("continuous_self_learning_task") is True
        and skip is not None
        and effective is not None
        and isinstance(payload.get("operating_point_generalizes"), bool)
        and payload.get("headline_ensemble_unchanged") is True
        and payload.get("is_measurement_not_retrain") is True
    )
    return {
        "continued": continued,
        "skip_rate_second_split": skip,
        "effective_auroc_second_split": effective,
        "operating_point_generalizes": payload.get("operating_point_generalizes"),
        "acceptance_gate_passed": gate_passed,
        "headline_ensemble_unchanged": payload.get("headline_ensemble_unchanged") is True,
        "measurement_not_retrain": payload.get("is_measurement_not_retrain") is True,
    }


def publication_gate_confirmation(payload: Mapping[str, Any], gate_data: Mapping[str, Any]) -> JsonDict:
    gate_ready = paper_ready_from_gate(gate_data)
    g1_g4 = all(payload.get(f"g{i}_pass") is True for i in range(1, 5))
    frozen = (
        numeric(payload.get("frozen_fover_auroc")) == FROZEN_FOVER_AUROC
        and payload.get("frozen_fover_auroc_unchanged") is True
    )
    no_redefine = payload.get("gate_definitions_unchanged") is True
    paper_ready = payload.get("paper_ready") is True and gate_ready
    return {
        "confirmed": bool(g1_g4 and paper_ready and frozen and no_redefine),
        "g1_g4_pass": g1_g4,
        "paper_ready": paper_ready,
        "frozen_fover_auroc": numeric(payload.get("frozen_fover_auroc")),
        "frozen_headline_unchanged": frozen,
        "gate_definitions_unchanged": no_redefine,
        "any_gate_regressed": payload.get("any_gate_regressed") is True,
    }


def edlm_seed(payload: Mapping[str, Any]) -> JsonDict:
    command = payload.get("operator_seed_command")
    staged = bool(
        payload
        and payload.get("staging_note_written") is True
        and isinstance(command, str)
        and bool(command.strip())
        and payload.get("kill_gate_design_documented") is True
        and payload.get("loop_does_not_seed") is True
        and payload.get("edlm_remains_operator_gated") is True
    )
    return {
        "staged": staged,
        "operator_seed_command": command,
        "kill_gate_design_documented": payload.get("kill_gate_design_documented") is True,
        "loop_does_not_seed": payload.get("loop_does_not_seed") is True,
        "operator_gated": payload.get("edlm_remains_operator_gated") is True,
        "operator_curated_doc_unedited": payload.get("operator_curated_doc_unedited") is True,
    }


def energy_routes_bounded(payload: Mapping[str, Any]) -> bool:
    return bool(payload.get("both_energy_routes_still_bounded") is True)


def next_thesis_operator_surface(exp3808: Mapping[str, Any], edlm: Mapping[str, Any]) -> bool:
    return bool(
        exp3808.get("edlm_remains_operator_seed_surface") is True
        and edlm.get("loop_does_not_seed") is True
        and edlm.get("operator_gated") is True
    )


def references_refreshed(payload: Mapping[str, Any]) -> bool:
    n_references = payload.get("n_references_added")
    return bool(
        payload
        and isinstance(n_references, int)
        and n_references >= 1
        and payload.get("references_section_intact") is True
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


def milestone_outcome(verifier: Mapping[str, Any]) -> str:
    surface = (
        "repairs the blocked product surface and confirms parity"
        if verifier.get("product_repaired") is True
        else "records blocked product-surface work honestly"
    )
    return (
        ".349 outcome: a LEAN POST-CONVERGENCE maintenance milestone -- it "
        "WIRES the endorsed advisory hook, "
        f"{surface}, RECORDS the product-headline status honestly, CONTINUES "
        "the mandated Tier-3 self-learning, CONFIRMS the convergence invariants, "
        "and STAGES the EDLM seed for the operator; it makes NO new existential "
        "claim and re-grinds nothing bounded."
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
