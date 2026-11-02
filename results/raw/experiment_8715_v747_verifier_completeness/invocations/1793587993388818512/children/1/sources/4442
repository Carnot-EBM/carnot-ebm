"""Build the Exp 3775 v345 recovery capstone artifact.

Spec refs: REQ-REPORT-3775, SCENARIO-REPORT-3775,
SCENARIO-REPORT-3775-MISSING, SCENARIO-REPORT-3775-FLAGGED.

This module is a capstone aggregator. It reads prior experiment artifacts,
quotes their provenance by hash, and writes a narrow milestone summary. It
does not rerun live models and it does not turn a missing upstream experiment
into a negative research result.
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


OUTPUT_REL_PATH = Path("results/experiment_3775_capstone_v345.json")
OPERATOR_SURFACE_REL_PATH = Path("results/experiment_3763_next_phase3_thesis_decision_menu.json")
RANDOM_SEED = 3775
FROZEN_FOVER_AUROC = 0.9131
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: a capstone reads upstream JSON, "
    "runs no live model)."
)
UPSTREAM_IDS = tuple(range(3765, 3775))

DEFAULT_UPSTREAM_PATHS: Mapping[int, Path] = {
    3765: Path("results/experiment_3765_archive_v344_activate_v345.json"),
    3766: Path("results/experiment_3766_thesis_a_definitive_reconcile.json"),
    3767: Path("results/experiment_3767_g2_mechanical_reproducer.json"),
    3768: Path("results/experiment_3768_g3_narrowing_lint.json"),
    3769: Path("results/experiment_3769_package_cli_mcp_e2e_smoke.json"),
    3770: Path("results/experiment_3770_distribution_mirror_publish_checklist.json"),
    3771: Path("results/experiment_3771_certified_abstention_operating_point.json"),
    3772: Path("results/experiment_3772_fr11_self_learning_v17_verifier_precision_tracker.json"),
    3773: Path("results/experiment_3773_verifier_product_prm_positioning.json"),
    3774: Path("results/experiment_3774_kv260_opportunistic_continuity_audit.json"),
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "v344_skip_cascade_recovered",
    "thesis_a_definitively_closed",
    "both_energy_routes_bounded",
    "gates_mechanized",
    "verifier_banked_for_ship",
    "certified_abstention_point_status",
    "verifier_positioned_vs_prm_sota",
    "paper_ready_preserved",
    "frozen_headline_unchanged",
    "next_thesis_remains_operator_surface",
    "flagged_artifacts_excluded",
    "not_landed_artifacts_recorded_honestly",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix; the milestone's one-line outcome.",
    "inference_substrate": (
        "A capstone reads upstream JSON, runs no live model, and must not add "
        "live-inference provenance to an aggregation artifact."
    ),
    "v344_skip_cascade_recovered": (
        "BARE bool -- records that the .344 un-landed agenda was re-executed "
        "this milestone."
    ),
    "thesis_a_definitively_closed": (
        "Records the milestone-defining fact restored to the record: "
        "energy-as-generator is bounded, discriminative-not-generative (exp3766)."
    ),
    "both_energy_routes_bounded": (
        "BARE bool -- both selection AND generation are bounded; the reason "
        ".345 is product-banking not re-grind."
    ),
    "gates_mechanized": (
        "Records G2 reproducer + G3 narrowing lint shipped -- the publication-gate "
        "hardening deliverable."
    ),
    "verifier_banked_for_ship": (
        "Records the Phase-1 software-ship evidence (E2E surfaces passed + mirror "
        "checklist) -- the product deliverable."
    ),
    "certified_abstention_point_status": (
        "The deployable abstention operating point outcome (shipped / gate-skipped "
        "if headline did not reproduce)."
    ),
    "verifier_positioned_vs_prm_sota": (
        "Records the honest PRM-positioning deliverable (exp3773)."
    ),
    "paper_ready_preserved": (
        "G1-G4 stay met; the milestone must not regress the banked verifier product."
    ),
    "frozen_headline_unchanged": (
        "Frozen FoVer 0.9131 stays frozen; .345 reproduces but never moves the headline."
    ),
    "next_thesis_remains_operator_surface": (
        "Records that the next-Phase-3 decision remains an operator-seeding surface "
        "(exp3763 menu) -- the loop does not self-commit."
    ),
    "flagged_artifacts_excluded": (
        "Lists any flagged_adversarial artifact excluded from aggregation "
        "(fabrication gate)."
    ),
    "not_landed_artifacts_recorded_honestly": (
        "Lists any upstream task that did not land -- recorded as not-run, NOT as "
        "a research negative."
    ),
    "cited_upstream_artifacts": "Provenance trail from the capstone numbers to the real artifacts.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}

SUMMARY_FIELDS: Mapping[int, tuple[str, ...]] = {
    3765: (
        "honest_verdict",
        "unlanded_v344_agenda_carried_to_v345",
        "v345_focus_recorded",
        "paper_ready_preserved",
        "paper_ready_evidence.paper_ready",
        "paper_ready_evidence.g1",
        "paper_ready_evidence.g2",
        "paper_ready_evidence.g3",
        "paper_ready_evidence.g4",
        "paper_ready_evidence.frozen_headline_unchanged",
        "paper_ready_evidence.frozen_headline_auroc",
    ),
    3766: (
        "honest_verdict",
        "thesis_a_part_a_outcome",
        "thesis_a_part_b_outcome",
        "ebt_discriminative_not_generative",
        "in_loop_chain_superseded",
        "thesis_menu_updated",
        "not_added_to_exclusion_manifest",
    ),
    3767: (
        "honest_verdict",
        "auroc_in_ci95",
        "frozen_headline_unchanged",
        "reproduced_auroc_mean",
        "source_headline.headline_matches_frozen_0_9131",
    ),
    3768: (
        "honest_verdict",
        "lint_extended_and_wired",
        "paper_v6_json_scan_extended",
        "precommit_hook_wired",
        "twelfth_retraction_added",
        "violations_found",
    ),
    3769: (
        "honest_verdict",
        "package_importable",
        "pipeline_e2e_passed",
        "cli_passed",
        "mcp_protocol_exchange_passed",
        "surfaces_passed",
        "is_wiring_smoke_not_accuracy_claim",
    ),
    3770: (
        "honest_verdict",
        "pypi_workflow_ready",
        "hf_mirror_documented",
        "ipfs_plan_documented",
        "operator_publish_checklist",
        "agent_published_nothing",
    ),
    3771: (
        "honest_verdict",
        "usable_operating_point_exists",
        "selected_threshold",
        "coverage_at_operating_point",
        "risk_target",
        "certified_risk_bound",
    ),
    3772: (
        "honest_verdict",
        "continuous_self_learning_task",
        "memory_contribution_preserved",
        "tracker_state_persisted",
        "pivoted_off_dead_ebt_lineage",
        "acceptance_gate.passed",
    ),
    3773: (
        "honest_verdict",
        "peer_numbers_are_as_reported_not_re_derived",
        "no_generalization_retest_run",
        "where_carnot_leads",
        "where_carnot_does_not_lead",
        "product_value_proposition",
    ),
    3774: (
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
    capstone_adversarial_verify_clean: bool = True,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """Build the v345 capstone from upstream JSON artifacts."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    paths = {
        experiment_id: resolve_upstream_path(root_path, experiment_id)
        for experiment_id in UPSTREAM_IDS
    }
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
    operator_surface = load_operator_surface(root_path)

    exp3765 = clean_upstreams.get(3765, {})
    exp3766 = clean_upstreams.get(3766, {})
    exp3767 = clean_upstreams.get(3767, {})
    exp3768 = clean_upstreams.get(3768, {})
    exp3769 = clean_upstreams.get(3769, {})
    exp3770 = clean_upstreams.get(3770, {})
    exp3771 = clean_upstreams.get(3771, {})
    exp3772 = clean_upstreams.get(3772, {})
    exp3773 = clean_upstreams.get(3773, {})
    exp3774 = clean_upstreams.get(3774, {})

    skip_recovered = skip_cascade_recovered(exp3765)
    thesis_closed = thesis_a_closed(exp3766)
    both_bounded = both_energy_routes_bounded(exp3766)
    g2_reproduced = g2_local_reproduced(exp3767)
    gates_mech = g2_reproduced and g3_narrowing_lint_shipped(exp3768)
    verifier_banked = package_cli_mcp_passed(exp3769) and distribution_ready(exp3770)
    abstention_status = certified_abstention_status(exp3767, exp3771)
    paper_ready = paper_ready_preserved(exp3765)
    frozen = frozen_headline_unchanged(exp3765, exp3767)
    fr11_v17 = fr11_v17_memory_preserved(exp3772)
    prm_positioned = verifier_positioned_vs_prm(exp3773)
    kv260_confirmed = kv260_terminal_confirmed(exp3774)
    next_operator_surface = next_thesis_operator_surface(operator_surface)
    duration_s = round(max(0.0001, (time.perf_counter() if now_s is None else float(now_s)) - start), 6)

    artifact: JsonDict = {
        "schema": "carnot.capstone_v345_recovery_3775.v1",
        "experiment_id": "exp3775",
        "honest_verdict": terminal_verdict(
            skip_recovered=skip_recovered,
            thesis_closed=thesis_closed,
            both_energy_bounded=both_bounded,
            gates_mechanized=gates_mech,
            verifier_banked=verifier_banked,
            abstention_status=abstention_status,
            fr11_v17=fr11_v17,
            prm_positioned=prm_positioned,
            paper_ready=paper_ready,
            frozen=frozen,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "v344_skip_cascade_recovered": skip_recovered,
        "thesis_a_definitively_closed": thesis_closed,
        "both_energy_routes_bounded": both_bounded,
        "gates_mechanized": gates_mech,
        "verifier_banked_for_ship": verifier_banked,
        "certified_abstention_point_status": abstention_status,
        "verifier_positioned_vs_prm_sota": prm_positioned,
        "paper_ready_preserved": paper_ready,
        "frozen_headline_unchanged": frozen,
        "next_thesis_remains_operator_surface": next_operator_surface,
        "milestone_outcome_plain": milestone_outcome(thesis_closed, both_bounded, verifier_banked),
        "thesis_a_status_note": thesis_status_note(thesis_closed, both_bounded),
        "energy_as_selector_status": "honest-negative-bounded",
        "energy_as_generator_status": (
            "honest-negative-bounded" if both_bounded else "not-landed: exp3766 missing or excluded"
        ),
        "no_new_existential_claim": True,
        "publication_gate_state": publication_gate_state(exp3765),
        "frozen_fover_auroc": FROZEN_FOVER_AUROC,
        "g2_local_reproducer_auroc_in_ci95": g2_reproduced,
        "g3_narrowing_lint_shipped": g3_narrowing_lint_shipped(exp3768),
        "phase1_ship_surfaces_passed": package_cli_mcp_passed(exp3769),
        "distribution_mirror_checklist_ready": distribution_ready(exp3770),
        "agent_published_nothing": truthy(exp3770.get("agent_published_nothing")),
        "fr11_v17_memory_contribution_preserved": fr11_v17,
        "prm_positioning_peer_numbers_as_reported": truthy(
            exp3773.get("peer_numbers_are_as_reported_not_re_derived")
        ),
        "kv260_terminal_confirmed": kv260_confirmed,
        "headline_aggregation_experiment_ids": sorted(clean_upstreams),
        "flagged_artifacts_excluded": flagged_artifacts(paths, flagged_ids),
        "not_landed_artifacts_recorded_honestly": not_landed_artifacts(paths, upstreams),
        "cited_upstream_artifacts": cited_upstream_artifacts(root_path, paths, clean_upstreams),
        "supporting_operator_surface_artifact": supporting_operator_surface(
            root_path, operator_surface
        ),
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
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Write, adversarial-verify, and rewrite the Exp 3775 artifact."""

    root_path = Path(root)
    out_path = root_path / OUTPUT_REL_PATH
    artifact = build_artifact(
        root_path,
        summary_records=summary_records,
        adversarial_reports=adversarial_reports,
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
    """Return schema and honesty errors for the Exp 3775 capstone."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append(f"missing required artifact fields: {', '.join(missing)}")
    if not str(artifact.get("honest_verdict") or "").startswith("complete: capstone_v345_"):
        errors.append("honest_verdict must be a terminal Exp 3775 verdict")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must declare the v345 aggregation-only substrate")
    for field in (
        "v344_skip_cascade_recovered",
        "thesis_a_definitively_closed",
        "both_energy_routes_bounded",
        "gates_mechanized",
        "verifier_banked_for_ship",
        "verifier_positioned_vs_prm_sota",
        "next_thesis_remains_operator_surface",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(f"{field} must be a bare bool")
    if artifact.get("certified_abstention_point_status") not in {"shipped", "skipped"}:
        errors.append("certified_abstention_point_status must be shipped or skipped")
    if artifact.get("paper_ready_preserved") is not True:
        errors.append("paper_ready_preserved must be true")
    if artifact.get("frozen_headline_unchanged") is not True:
        errors.append("frozen_headline_unchanged must be true")
    if not isinstance(artifact.get("flagged_artifacts_excluded"), list):
        errors.append("flagged_artifacts_excluded must be a list")
    if not isinstance(artifact.get("not_landed_artifacts_recorded_honestly"), list):
        errors.append("not_landed_artifacts_recorded_honestly must be a list")
    validate_citations(artifact.get("cited_upstream_artifacts"), errors)
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed must equal 3775")
    duration_s = artifact.get("duration_s")
    if (
        not isinstance(duration_s, int | float)
        or isinstance(duration_s, bool)
        or float(duration_s) < 0.0001
    ):
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


def terminal_verdict(
    *,
    skip_recovered: bool,
    thesis_closed: bool,
    both_energy_bounded: bool,
    gates_mechanized: bool,
    verifier_banked: bool,
    abstention_status: str,
    fr11_v17: bool,
    prm_positioned: bool,
    paper_ready: bool,
    frozen: bool,
) -> str:
    """Return the terminal verdict string using classified milestone states."""

    return (
        "complete: capstone_v345_"
        f"{'skip_cascade_recovered' if skip_recovered else 'skip_cascade_not_recovered'}_"
        f"{'thesis_a_closed' if thesis_closed else 'thesis_a_not_landed'}_"
        f"{'both_energy_routes_bounded' if both_energy_bounded else 'both_energy_routes_not_fully_landed'}_"
        f"{'gates_mechanized' if gates_mechanized else 'gates_not_mechanized'}_"
        f"{'verifier_banked' if verifier_banked else 'verifier_not_banked'}_"
        f"abstention_point_{abstention_status}_"
        f"{'fr11_v17' if fr11_v17 else 'fr11_v17_not_preserved'}_"
        f"{'prm_positioned' if prm_positioned else 'prm_not_positioned'}_"
        f"paper_ready_{str(paper_ready).lower()}_"
        f"{'frozen_headline_unchanged' if frozen else 'frozen_headline_changed'}"
    )


def resolve_upstream_path(root: Path, experiment_id: int) -> Path:
    """Return the default path or the first same-ID artifact in results."""

    default = root / DEFAULT_UPSTREAM_PATHS[experiment_id]
    if default.exists():
        return default
    matches = sorted((root / "results").glob(f"experiment_{experiment_id}_*.json"))
    return matches[0] if matches else default


def read_json_object(path: Path) -> JsonDict:
    """Read an upstream JSON object; array artifacts are invalid provenance."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def run_summarize_artifacts(
    root: Path,
    paths: Mapping[int, Path],
) -> list[JsonDict]:  # pragma: no cover - subprocess boundary
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


def skip_cascade_recovered(payload: Mapping[str, Any]) -> bool:
    verdict = str(payload.get("honest_verdict") or "").lower()
    agenda = payload.get("unlanded_v344_agenda_carried_to_v345")
    return bool(payload and "skip_cascade" in verdict and isinstance(agenda, list) and agenda)


def thesis_a_closed(payload: Mapping[str, Any]) -> bool:
    part_a = str(payload.get("thesis_a_part_a_outcome") or "").lower()
    part_b = str(payload.get("thesis_a_part_b_outcome") or "").lower()
    return bool(
        payload
        and "pass" in part_a
        and "bounded" in part_b
        and truthy(payload.get("ebt_discriminative_not_generative"))
        and truthy(payload.get("in_loop_chain_superseded"))
    )


def both_energy_routes_bounded(payload: Mapping[str, Any]) -> bool:
    return bool(payload and thesis_a_closed(payload))


def g2_local_reproduced(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload
        and truthy(payload.get("auroc_in_ci95"))
        and truthy(payload.get("frozen_headline_unchanged"))
        and truthy(nested_get(payload, "source_headline.headline_matches_frozen_0_9131"))
    )


def g3_narrowing_lint_shipped(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload
        and truthy(payload.get("lint_extended_and_wired"))
        and truthy(payload.get("paper_v6_json_scan_extended"))
        and truthy(payload.get("precommit_hook_wired"))
        and truthy(payload.get("twelfth_retraction_added"))
    )


def package_cli_mcp_passed(payload: Mapping[str, Any]) -> bool:
    surfaces = payload.get("surfaces_passed")
    expected = {"package_import", "pipeline", "mcp_protocol", "cli"}
    return bool(
        payload
        and truthy(payload.get("package_importable"))
        and truthy(payload.get("pipeline_e2e_passed"))
        and truthy(payload.get("cli_passed"))
        and truthy(payload.get("mcp_protocol_exchange_passed"))
        and isinstance(surfaces, list)
        and expected <= set(surfaces)
        and truthy(payload.get("is_wiring_smoke_not_accuracy_claim"))
    )


def distribution_ready(payload: Mapping[str, Any]) -> bool:
    checklist = payload.get("operator_publish_checklist")
    return bool(
        payload
        and truthy(payload.get("pypi_workflow_ready"))
        and truthy(payload.get("hf_mirror_documented"))
        and truthy(payload.get("ipfs_plan_documented"))
        and isinstance(checklist, list)
        and checklist
        and truthy(payload.get("agent_published_nothing"))
    )


def certified_abstention_status(exp3767: Mapping[str, Any], exp3771: Mapping[str, Any]) -> str:
    if not g2_local_reproduced(exp3767):
        return "skipped"
    if exp3771 and truthy(exp3771.get("usable_operating_point_exists")):
        return "shipped"
    return "skipped"


def paper_ready_preserved(payload: Mapping[str, Any]) -> bool:
    evidence = payload.get("paper_ready_evidence")
    return bool(
        payload.get("paper_ready_preserved") is True
        and isinstance(evidence, Mapping)
        and evidence.get("paper_ready") is True
        and all(evidence.get(name) is True for name in ("g1", "g2", "g3", "g4"))
    )


def frozen_headline_unchanged(exp3765: Mapping[str, Any], exp3767: Mapping[str, Any]) -> bool:
    evidence = exp3765.get("paper_ready_evidence")
    archive_frozen = (
        isinstance(evidence, Mapping)
        and evidence.get("frozen_headline_unchanged") is True
        and numeric(evidence.get("frozen_headline_auroc")) == FROZEN_FOVER_AUROC
    )
    reproduced_frozen = g2_local_reproduced(exp3767)
    return bool(archive_frozen or reproduced_frozen)


def fr11_v17_memory_preserved(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload
        and truthy(payload.get("continuous_self_learning_task"))
        and truthy(payload.get("memory_contribution_preserved"))
        and truthy(payload.get("tracker_state_persisted"))
        and truthy(payload.get("pivoted_off_dead_ebt_lineage"))
        and truthy(nested_get(payload, "acceptance_gate.passed"))
    )


def verifier_positioned_vs_prm(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload
        and truthy(payload.get("peer_numbers_are_as_reported_not_re_derived"))
        and truthy(payload.get("no_generalization_retest_run"))
        and payload.get("where_carnot_leads")
        and payload.get("where_carnot_does_not_lead")
        and payload.get("product_value_proposition")
    )


def kv260_terminal_confirmed(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload
        and truthy(payload.get("terminal_state_holds"))
        and truthy(payload.get("kv260_ssh_reachable"))
        and truthy(payload.get("kv260_overlay_loadable"))
        and payload.get("speedup_claim_made") is not True
    )


def load_operator_surface(root: Path) -> JsonDict | None:
    path = root / OPERATOR_SURFACE_REL_PATH
    return read_json_object(path) if path.exists() else None


def next_thesis_operator_surface(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping):
        return False
    verdict = str(payload.get("honest_verdict") or "")
    return bool(
        truthy(payload.get("loop_will_not_self_seed"))
        and truthy(payload.get("supersedes_340_menu"))
        and "operator_seeding" in verdict
    )


def publication_gate_state(exp3765: Mapping[str, Any]) -> JsonDict:
    evidence = exp3765.get("paper_ready_evidence")
    if not isinstance(evidence, Mapping):
        evidence = {}
    return {
        "paper_ready": evidence.get("paper_ready") is True,
        "g1": evidence.get("g1") is True,
        "g2": evidence.get("g2") is True,
        "g3": evidence.get("g3") is True,
        "g4": evidence.get("g4") is True,
    }


def milestone_outcome(thesis_closed: bool, both_bounded: bool, verifier_banked: bool) -> str:
    if thesis_closed and both_bounded and verifier_banked:
        return (
            ".345 is a recovery + convergence milestone: it re-banks the one "
            "positive verifier product result, restores the Thesis-A bounded "
            "closure to the record, and hands the next research decision to the "
            "operator. It makes no new existential claim."
        )
    return (
        ".345 capstone ran as aggregation, but one or more upstream artifacts "
        "needed for the clean milestone claim were missing or excluded; no absent "
        "result is inferred."
    )


def thesis_status_note(thesis_closed: bool, both_bounded: bool) -> str:
    if thesis_closed and both_bounded:
        return "landed: exp3766 restored PASS/discriminative plus BOUNDED/not-generative."
    return (
        "not-landed: exp3766 is missing or excluded, so the capstone records the "
        "upstream gap, not a research negative."
    )


def not_landed_artifacts(
    paths: Mapping[int, Path],
    upstreams: Mapping[int, Mapping[str, Any] | None],
) -> list[JsonDict]:
    return [
        {
            "experiment_id": experiment_id,
            "path": str(paths[experiment_id]),
            "status": "not-landed",
            "reason": "artifact_missing",
        }
        for experiment_id in UPSTREAM_IDS
        if upstreams.get(experiment_id) is None
    ]


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
                "sha256": sha256_file(path if path.is_absolute() else root / path),
            }
        )
    return citations


def supporting_operator_surface(root: Path, payload: Mapping[str, Any] | None) -> JsonDict:
    path = root / OPERATOR_SURFACE_REL_PATH
    if not path.exists() or not isinstance(payload, Mapping):
        return {
            "experiment_id": 3763,
            "path": str(path),
            "status": "not-landed",
            "reason": "supporting_operator_surface_missing",
        }
    return {
        "experiment_id": 3763,
        "path": str(path),
        "fields_imported": ["honest_verdict", "loop_will_not_self_seed", "supersedes_340_menu"],
        "honest_verdict": payload.get("honest_verdict"),
        "sha256": sha256_file(path),
    }


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


def has_live_model_markers(payload: Mapping[str, Any]) -> bool:
    blob = json.dumps(payload, sort_keys=True).lower()
    return any(
        marker in blob
        for marker in ("live_llm_inference", "target_model", "model_specs", "live_model_invoked")
    )


def report_is_clean(report: Any) -> bool:
    if not isinstance(report, Mapping):
        return True
    for flag in report.get("flags") or []:
        if not isinstance(flag, Mapping):
            continue
        if str(flag.get("severity") or "").lower() == "critical":
            return False
    return True


def nested_get(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def truthy(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.lower() in {"true", "pass", "passed", "shipped"})


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return round(float(value), 4)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    filtered = dict(payload)
    filtered["reproducibility_checksum"] = ""
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:  # pragma: no cover - CLI wrapper
    out_path = run(REPO_ROOT)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
