"""Archive milestone .348's terminal record and confirm .349 is active.

Spec refs: REQ-REPORT-3808, SCENARIO-REPORT-3808,
SCENARIO-REPORT-3808-PRODUCT-HEADLINE-GUARD.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import time
from typing import Any

from carnot.reporting.archive_v345_activate_v346_3776 import (
    JsonDict,
    compact_verify_report,
    duration_from,
    evaluate_publication_gate,
    extract_paper_ready_evidence,
    is_sha256,
    no_forbidden_markers,
    payload_checksum,
    read_active_milestone,
    read_json_object,
    report_is_clean,
    safe_point,
    sha256_path,
    write_payload,
    yaml_parses,
    yaml_string,
    _ensure,
)
from scripts import adversarial_verify


REPO_ROOT = Path(__file__).resolve().parents[3]
ARCHIVED_MILESTONE = "2026.06.348"
ACTIVATED_MILESTONE = "2026.06.349"
RANDOM_SEED = 3808
OUTPUT_REL_PATH = Path("results/experiment_3808_archive_v348_activate_v349.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
ROADMAP_DESIGN_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CAPSTONE_REL_PATH = Path("results/experiment_3807_capstone_v348.json")
CHANGELOG_REL_PATH = Path("ops/changelog.md")
NORTH_STAR_REL_PATH = Path("ops/north-star.md")

INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_VERDICT = (
    "complete: "
    "archived_v348_landed_all_product_headline_demoted_both_positives_fail_"
    "provenance_v349_lean_maintenance_active_paper_ready_true_both_energy_"
    "routes_bounded_frozen_headline_unchanged_edlm_operator_seed_surface"
)
V348_OUTCOME_RECORDED = (
    "landed_all_tasks_product_headline_demoted_both_positives_fail_provenance_"
    "gaming_closed_anomaly_wirable_http_rest_blocked_tier3_fast_path_"
    "paper_ready_true"
)
V349_FOCUS_RECORDED = (
    "advisory_hook_wiring_http_rest_repair_parity_product_headline_status_"
    "tier3_self_learning_publication_gate_confirm_edlm_seed_staging_"
    "no_bounded_regrind"
)
HTTP_REPAIR_TASK_ID = "exp3801-abstention-http-rest-surface"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "v348_outcome_recorded",
    "v349_focus_recorded",
    "product_headline_demoted_recorded",
    "research_complete_yaml_parses",
    "paper_ready_preserved",
    "both_energy_routes_still_bounded",
    "edlm_remains_operator_seed_surface",
    "n_tasks_archived",
    "adversarial_verify_clean",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal prefix lets the reconciler classify the transition complete "
        "without re-running it."
    ),
    "inference_substrate": (
        "JSON-read and format; 0.0001s floor; no compute-bound marker so it "
        "does not false-flag."
    ),
    "v348_outcome_recorded": (
        "Records .348's REAL state: landed all tasks, product headline demoted, "
        "gaming closed, anomaly classifier now wirable, HTTP/REST blocked, "
        "Tier-3 fast-path landed, paper_ready TRUE."
    ),
    "v349_focus_recorded": (
        "Records the .349 agenda: advisory-hook wiring, HTTP/REST repair and "
        "parity, product-headline status, Tier-3 self-learning, publication-gate "
        "confirm, EDLM seed staging, and no bounded re-grind."
    ),
    "product_headline_demoted_recorded": (
        "BARE bool -- records that both candidate product positives fail "
        "provenance and the FoVer methods headline is the sole defensible one."
    ),
    "research_complete_yaml_parses": (
        "BARE bool, MUST be true -- confirms research-complete.yaml safe-loads "
        "after the write."
    ),
    "paper_ready_preserved": (
        "BARE bool -- G1-G4 stay met; the transition must not silently regress "
        "paper_ready; frozen 0.9131 stays frozen."
    ),
    "both_energy_routes_still_bounded": (
        "BARE bool -- records that .349 does not reopen the bounded conclusion "
        "and runs no energy-foundation experiment."
    ),
    "edlm_remains_operator_seed_surface": (
        "BARE bool -- records that the EDLM seed-vs-freeze decision remains an "
        "operator surface and the loop does not self-seed."
    ),
    "n_tasks_archived": (
        "Sample-size hygiene -- confirms the full .348 milestone was archived, "
        "not a partial."
    ),
    "adversarial_verify_clean": (
        "True iff the artifact passes adversarial_verify with no critical flag."
    ),
    "random_seed": "Determinism precondition for reproducibility.",
    "reproducibility_checksum": "Content hash catches silent drift vs any replication.",
    "duration_s": "Wall-clock plausibility floor; missing duration is the fabrication signal.",
}

V348_TASKS = [
    {
        "id": "exp3797-archive-v347-activate-v348",
        "title": "Archive .347 and activate .348 post-convergence headline advancement",
        "deliverable": "results/experiment_3797_archive_v347_activate_v348.json",
        "result": (
            "COMPLETE: archived .347 with P1 blocked and handed to the operator, "
            "then activated the .348 headline-advancement and product-hardening "
            "agenda"
        ),
    },
    {
        "id": "exp3798-g4-product-headline-restoration",
        "title": "G4 product-headline restoration re-run",
        "deliverable": "results/experiment_3798_g4_product_headline_restoration.json",
        "result": (
            "COMPLETE_DEMOTED: exp3798 G4 re-run produced delta=0.0pp; the "
            "historical code-repair lift did not survive clean provenance and "
            "the product headline stays demoted"
        ),
    },
    {
        "id": "exp3799-product-headline-provenance-reconfirmation",
        "title": "Product headline provenance re-confirmation",
        "deliverable": "results/experiment_3799_product_headline_provenance_reconfirmation.json",
        "result": (
            "COMPLETE: product headline remained not_yet_eligible; the later "
            "planning re-check records exp2090 as failing provenance, so both "
            "candidate product positives fail provenance"
        ),
    },
    {
        "id": "exp3800-gaming-resistance-mitigation-v2",
        "title": "Gaming-resistance mitigation v2",
        "deliverable": "results/experiment_3800_gaming_resistance_mitigation_v2.json",
        "result": (
            "COMPLETE: context_compaction evasion closed at n=240 while clean "
            "AUROC stayed preserved"
        ),
    },
    {
        "id": HTTP_REPAIR_TASK_ID,
        "title": "HTTP/REST abstention surface",
        "deliverable": "results/experiment_3801_abstention_http_rest_surface.json",
        "result": (
            "BLOCKED_REPAIR_TARGET: blocked_http_abstention_e2e_failed; .349 "
            "carries HTTP/REST repair and cross-surface parity as product work"
        ),
    },
    {
        "id": "exp3802-anomaly-escalation-classifier-v2-tuning",
        "title": "Anomaly-Escalation classifier v2 tuning",
        "deliverable": "results/experiment_3802_anomaly_escalation_classifier_v2_tuning.json",
        "result": (
            "COMPLETE: false-escalation fell from 0.833333 to 0.0, frame-violating "
            "recall stayed 1.0, and supports_wiring_in=true"
        ),
    },
    {
        "id": "exp3803-fr11-self-learning-v20-tier3-fast-path-gate",
        "title": "FR-11 v20 Tier-3 fast-path gate",
        "deliverable": "results/experiment_3803_fr11_v20_tier3_fast_path_gate.json",
        "result": (
            "COMPLETE: Tier-3 fast-path gate landed with skip_rate=0.56 and no "
            "accuracy regression inside the frozen interval"
        ),
    },
    {
        "id": "exp3805-external-research-refresh",
        "title": "External research refresh .348",
        "deliverable": "results/experiment_3805_external_research_refresh.json",
        "result": "COMPLETE: .348 references filed append-only with peer numbers as reported",
    },
    {
        "id": "exp3806-kv260-opportunistic-continuity-audit",
        "title": "KV260 opportunistic continuity audit",
        "deliverable": "results/experiment_3806_kv260_opportunistic_continuity_audit.json",
        "result": "COMPLETE: KV260 terminal state held; SSH reachable and overlay loadable",
    },
    {
        "id": "exp3807-capstone-v348",
        "title": "Capstone .348",
        "deliverable": "results/experiment_3807_capstone_v348.json",
        "result": (
            "COMPLETE: capstone recorded product-headline demotion, HTTP/REST "
            "blocked as a repair target, anomaly classifier wirable, Tier-3 "
            "fast-path landed, paper_ready TRUE, frozen FoVer 0.9131 unchanged, "
            "and both energy routes bounded"
        ),
    },
]


def build_research_complete_block() -> str:
    """Return the single honest `research-complete.yaml` block for .348."""

    finding = (
        "LANDED TERMINAL MILESTONE: .348 completed its post-convergence product "
        "hardening milestone. The product headline stays demoted: exp3798's G4 "
        "re-run produced delta=0.0pp, and the .349 planning re-check records "
        "exp2090 CRANE as failing provenance, so both candidate product positives "
        "fail provenance. This is not a research negative; it records that the "
        "FoVer methods headline is the sole defensible headline. The "
        "context_compaction gaming evasion closed, the Anomaly-Escalation "
        "classifier is now wirable, the HTTP/REST abstention surface blocked as "
        "blocked_http_abstention_e2e_failed and becomes a .349 repair target, "
        "the Tier-3 fast-path gate landed at skip_rate=0.56, references were "
        "refreshed, KV260 continuity held, paper_ready TRUE, frozen FoVer 0.9131 "
        "unchanged, and both energy routes stayed bounded. .349 is active: wire "
        "the advisory hook, repair HTTP/REST plus parity, record product-headline "
        "status, continue Tier-3 self-learning, confirm publication-gate "
        "invariants, stage the EDLM seed for the operator, re-grind nothing "
        "bounded, and self-seed no paradigm."
    )
    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  title: {yaml_string('V348 lean product-hardening terminal milestone archived')}",
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-06-04'",
        f"  finding: {yaml_string(finding)}",
        "  tasks:",
    ]
    for task in V348_TASKS:
        lines.extend(
            [
                f"  - id: {task['id']}",
                f"    title: {yaml_string(task['title'])}",
                f"    deliverable: {task['deliverable']}",
                f"    result: {yaml_string(task['result'])}",
            ]
        )
    return "\n".join(lines) + "\n"


def rewrite_research_complete(text: str) -> str:
    """Replace or append the `.348` archive block without duplicating it."""

    block = build_research_complete_block()
    replacement = block.splitlines()
    if not text.strip():
        return "milestones:\n" + block

    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line == f"- id: {ARCHIVED_MILESTONE}"),
        None,
    )
    if start is None:
        prefix = text.rstrip()
        if any(line.strip() == "milestones:" for line in lines):
            return f"{prefix}\n{block}"
        return f"{prefix}\nmilestones:\n{block}"

    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index].startswith("- id: 2026.")
        ),
        len(lines),
    )
    return "\n".join([*lines[:start], *replacement, *lines[end:]]) + "\n"


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    research_complete_yaml_parses: bool,
    publication_gate_report: Mapping[str, Any] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    adversarial_report: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the Exp 3808 terminal artifact from checked-in evidence."""

    root_path = Path(root)
    active_milestone, active_roadmap_path = read_active_milestone(root_path)
    _ensure(active_milestone == ACTIVATED_MILESTONE, ".349 active milestone confirmation is required")

    active_roadmap = root_path / active_roadmap_path
    roadmap_text = active_roadmap.read_text(encoding="utf-8")
    design_text = (root_path / ROADMAP_DESIGN_REL_PATH).read_text(encoding="utf-8")
    roadmap_evidence = f"{roadmap_text}\n{design_text}".lower()
    _ensure(
        all(
            token in roadmap_evidence
            for token in (
                "2026.06.349",
                "advisory",
                "http/rest",
                "parity",
                "product",
                "tier-3",
                "publication-gate",
                "edlm",
            )
        ),
        ".349 roadmap evidence must record the lean maintenance agenda",
    )
    _ensure(
        "re-grind" in roadmap_evidence or "regrind" in roadmap_evidence,
        ".349 roadmap evidence must record no bounded re-grind",
    )

    changelog_path = root_path / CHANGELOG_REL_PATH
    changelog_checked = changelog_path.exists()
    if changelog_checked:
        changelog_path.read_text(encoding="utf-8")

    capstone = read_json_object(root_path / CAPSTONE_REL_PATH)
    v348_evidence = extract_v348_capstone_evidence(capstone, roadmap_evidence)
    _ensure(v348_evidence["record_honest"] is True, ".348 capstone evidence must be honest")

    gate_report = (
        dict(publication_gate_report)
        if publication_gate_report is not None
        else evaluate_publication_gate(root_path)
    )
    paper_ready_evidence = extract_paper_ready_evidence(capstone, gate_report)
    _ensure(paper_ready_evidence["paper_ready"] is True, "publication gate must confirm paper_ready")

    paper_ready_preserved = (
        paper_ready_evidence["paper_ready"] is True
        and paper_ready_evidence["capstone_paper_ready"] is True
        and paper_ready_evidence["frozen_headline_unchanged"] is True
        and paper_ready_evidence["frozen_headline_auroc"] == 0.9131
        and all(paper_ready_evidence[gate] is True for gate in ("g1", "g2", "g3", "g4"))
    )
    both_energy_routes_still_bounded = (
        capstone.get("energy_as_generator_still_bounded") is True
        and capstone.get("energy_as_selector_status") == "honest-negative-bounded"
        and capstone.get("energy_as_generator_status") == "honest-negative-bounded"
        and capstone.get("regrinds_nothing_already_bounded") is True
    )
    _ensure(paper_ready_preserved, "paper_ready and frozen headline must be preserved")
    _ensure(both_energy_routes_still_bounded, "both energy routes must stay bounded")

    report = compact_verify_report(adversarial_report or {"flags": [], "flag_count": 0, "max_severity": -1})
    duration_s = duration_from(started_s, now_s)
    payload: JsonDict = {
        "schema": "carnot.archive_activation.v348_to_v349_3808.v1",
        "experiment_id": "exp3808",
        "task_id": "exp3808-archive-v348-activate-v349",
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_roadmap_path": active_roadmap_path,
        "v349_active_confirmed": active_milestone == ACTIVATED_MILESTONE,
        "honest_verdict": TERMINAL_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "v348_outcome_recorded": V348_OUTCOME_RECORDED,
        "v349_focus_recorded": V349_FOCUS_RECORDED,
        "product_headline_demoted_recorded": True,
        "research_complete_yaml_parses": research_complete_yaml_parses,
        "paper_ready_preserved": paper_ready_preserved,
        "both_energy_routes_still_bounded": both_energy_routes_still_bounded,
        "edlm_remains_operator_seed_surface": True,
        "n_tasks_archived": len(V348_TASKS),
        "n_tasks_terminal": len(V348_TASKS),
        "http_rest_repair_task_ids": [HTTP_REPAIR_TASK_ID],
        "adversarial_verify_clean": report_is_clean(report),
        "adversarial_verify_report": report,
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "field_principles": dict(FIELD_PRINCIPLES),
        "paper_ready_evidence": paper_ready_evidence,
        "v348_capstone_evidence": v348_evidence,
        "v349_activation_evidence": {
            "active_milestone": active_milestone,
            "active_roadmap_path": active_roadmap_path,
            "first_task": "exp3808-archive-v348-activate-v349",
            "lean_maintenance_focus": V349_FOCUS_RECORDED,
            "edlm_operator_gated": True,
            "bounded_regrind_queued": False,
        },
        "source_artifact_checksums": [
            {"path": str(CAPSTONE_REL_PATH), "sha256": sha256_path(root_path / CAPSTONE_REL_PATH)}
        ],
        "source_document_checksums": [
            {"path": active_roadmap_path, "sha256": sha256_path(active_roadmap)},
            {
                "path": str(ROADMAP_DESIGN_REL_PATH),
                "sha256": sha256_path(root_path / ROADMAP_DESIGN_REL_PATH),
            },
            {"path": str(CHANGELOG_REL_PATH), "sha256": sha256_path(changelog_path)},
            {"path": str(NORTH_STAR_REL_PATH), "sha256": sha256_path(root_path / NORTH_STAR_REL_PATH)},
        ],
        "changelog_checked": changelog_checked,
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    validate_artifact(payload)
    return payload


def run(
    root: Path | str = REPO_ROOT,
    *,
    publication_gate_report: Mapping[str, Any] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Write the honest `.348` archive and terminal Exp 3808 artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    complete_path = root_path / RESEARCH_COMPLETE_REL_PATH
    current_complete = complete_path.read_text(encoding="utf-8") if complete_path.exists() else ""
    rewritten = rewrite_research_complete(current_complete)
    _ensure(yaml_parses(rewritten), "rewritten research-complete.yaml must safe-load")
    complete_path.write_text(rewritten, encoding="utf-8")
    research_complete_parses = yaml_parses(complete_path.read_text(encoding="utf-8"))

    out_path = root_path / OUTPUT_REL_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_artifact(
        root_path,
        research_complete_yaml_parses=research_complete_parses,
        publication_gate_report=publication_gate_report,
        started_s=start,
        now_s=now_s,
    )
    write_payload(out_path, payload)

    verify_report = adversarial_verify.verify_artifact(out_path)
    payload["adversarial_verify_report"] = compact_verify_report(verify_report)
    payload["adversarial_verify_clean"] = report_is_clean(verify_report)
    payload["reproducibility_checksum"] = payload_checksum(payload)
    validate_artifact(payload)
    write_payload(out_path, payload)
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the required Exp 3808 archive/activation contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _ensure(not missing, f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    _ensure(isinstance(principles, Mapping), "field_principles must be a mapping")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    _ensure(not missing_principles, f"missing field principles: {missing_principles}")
    _ensure("model_specs" not in artifact, "model_specs must not be present")
    _ensure("target_model" not in artifact, "target_model must not be present")
    _ensure(no_forbidden_markers(artifact), "artifact must not contain compute-bound markers")
    _ensure(artifact.get("honest_verdict") == TERMINAL_VERDICT, "terminal verdict mismatch")
    _ensure(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference substrate mismatch")
    _ensure(artifact.get("v348_outcome_recorded") == V348_OUTCOME_RECORDED, ".348 outcome mismatch")
    _ensure(artifact.get("v349_focus_recorded") == V349_FOCUS_RECORDED, ".349 focus mismatch")
    _ensure(
        artifact.get("product_headline_demoted_recorded") is True,
        "product headline demotion confirmation required",
    )
    _ensure(artifact.get("v349_active_confirmed") is True, ".349 active confirmation required")
    _ensure(artifact.get("research_complete_yaml_parses") is True, "safe-load confirmation required")
    _ensure(artifact.get("paper_ready_preserved") is True, "paper_ready must remain preserved")
    _ensure(
        artifact.get("both_energy_routes_still_bounded") is True,
        "both energy routes must remain bounded",
    )
    _ensure(
        artifact.get("edlm_remains_operator_seed_surface") is True,
        "EDLM must remain operator-gated",
    )
    _ensure(artifact.get("n_tasks_archived") == 10, "n_tasks_archived must equal 10")
    _ensure(artifact.get("n_tasks_terminal") == 10, "n_tasks_terminal must equal terminal task count")
    _ensure(
        artifact.get("http_rest_repair_task_ids") == [HTTP_REPAIR_TASK_ID],
        "exp3801 HTTP repair target must be recorded",
    )
    _ensure(artifact.get("adversarial_verify_clean") is True, "adversarial_verify_clean must be true")
    _ensure(artifact.get("random_seed") == RANDOM_SEED, "random_seed must equal 3808")
    duration_s = artifact.get("duration_s")
    _ensure(
        isinstance(duration_s, int | float)
        and not isinstance(duration_s, bool)
        and float(duration_s) >= 0.0001,
        "duration_s must be numeric with the 0.0001s floor",
    )
    checksum = artifact.get("reproducibility_checksum")
    _ensure(is_sha256(checksum), "reproducibility_checksum must be a sha256 hex string")
    _ensure(checksum == payload_checksum(artifact), "reproducibility_checksum does not match artifact content")
    _ensure(
        report_is_clean(artifact.get("adversarial_verify_report", {"flags": []})),
        "adversarial report has critical flag",
    )


def extract_v348_capstone_evidence(
    capstone: Mapping[str, Any],
    roadmap_evidence: str,
) -> JsonDict:
    """Extract the `.348` demotion and `.349` agenda facts from Exp 3807."""

    product = capstone.get("product_headline_advanced")
    rerun = product.get("rerun") if isinstance(product, Mapping) else None
    blocked = capstone.get("not_landed_or_blocked_recorded_honestly")
    blocked_http_record = next(
        (
            dict(item)
            for item in blocked
            if isinstance(blocked, list)
            and isinstance(item, Mapping)
            and item.get("experiment_id") == 3801
        ),
        None,
    )
    verifier_product = capstone.get("verifier_product_hardened")
    context_compaction = (
        verifier_product.get("context_compaction_mitigation")
        if isinstance(verifier_product, Mapping)
        else None
    )
    http_surface = (
        verifier_product.get("http_rest_surface") if isinstance(verifier_product, Mapping) else None
    )
    anomaly = capstone.get("anomaly_classifier_repaired")
    fr11 = capstone.get("fr11_v20_tier3_fast_path")

    product_headline_demoted = bool(
        isinstance(product, Mapping)
        and product.get("headline_stays_demoted") is True
        and product.get("product_headline_restorable") == "not_yet_eligible"
        and isinstance(rerun, Mapping)
        and safe_point(rerun.get("repair_delta_pp")) == 0.0
        and rerun.get("g4_provenance_complete") is True
        and rerun.get("positive_control_passed") is True
    )
    exp2090_fails_provenance = (
        "exp2090" in roadmap_evidence
        and "critical" in roadmap_evidence
        and "fail provenance" in roadmap_evidence
    )
    both_product_positives_fail_provenance = product_headline_demoted and exp2090_fails_provenance
    context_compaction_closed = bool(
        isinstance(context_compaction, Mapping)
        and context_compaction.get("evasion_status") == "closed"
        and context_compaction.get("clean_auroc_preserved") is True
    ) or bool(
        isinstance(verifier_product, Mapping)
        and verifier_product.get("context_compaction_closed") is True
        and verifier_product.get("clean_auroc_preserved") is True
    )
    http_rest_repair_target = bool(
        blocked_http_record
        and blocked_http_record.get("status") == "blocked"
        and blocked_http_record.get("reason") == "blocked_http_abstention_e2e_failed"
    ) or bool(
        isinstance(http_surface, Mapping)
        and http_surface.get("status") == "blocked"
        and http_surface.get("e2e_passed") is False
    ) or bool(isinstance(verifier_product, Mapping) and verifier_product.get("http_rest_blocked") is True)
    anomaly_wirable = bool(
        isinstance(anomaly, Mapping)
        and anomaly.get("supports_wiring_in") is True
        and safe_point(anomaly.get("false_escalation_rate_after")) == 0.0
        and safe_point(anomaly.get("frame_violating_recall")) == 1.0
    )
    tier3_fast_path_landed = bool(
        isinstance(fr11, Mapping)
        and fr11.get("validated") is True
        and safe_point(fr11.get("skip_rate_at_no_regression")) == 0.56
        and fr11.get("effective_auroc_in_frozen_ci") is True
        and fr11.get("headline_ensemble_unchanged") is True
    )
    edlm_operator_surface = capstone.get("next_thesis_remains_operator_surface") is True
    no_research_negative = bool(
        capstone.get("no_new_existential_claim") is True
        and "research negative" not in str(capstone.get("honest_verdict", "")).lower()
    )
    record_honest = bool(
        capstone.get("paper_ready_preserved") is True
        and capstone.get("energy_as_generator_still_bounded") is True
        and capstone.get("energy_as_selector_status") == "honest-negative-bounded"
        and capstone.get("energy_as_generator_status") == "honest-negative-bounded"
        and capstone.get("frozen_headline_unchanged") is True
        and safe_point(capstone.get("frozen_fover_auroc")) == 0.9131
        and product_headline_demoted
        and both_product_positives_fail_provenance
        and context_compaction_closed
        and http_rest_repair_target
        and anomaly_wirable
        and tier3_fast_path_landed
        and edlm_operator_surface
        and no_research_negative
    )
    return {
        "capstone_path": str(CAPSTONE_REL_PATH),
        "honest_verdict": capstone.get("honest_verdict"),
        "record_honest": record_honest,
        "product_headline_demoted": product_headline_demoted,
        "exp3798_delta_pp": 0.0 if product_headline_demoted else None,
        "exp2090_fails_provenance": exp2090_fails_provenance,
        "both_product_positives_fail_provenance": both_product_positives_fail_provenance,
        "product_headline_not_research_negative": no_research_negative,
        "context_compaction_closed": context_compaction_closed,
        "http_rest_repair_target": http_rest_repair_target,
        "http_rest_blocked_record": blocked_http_record,
        "anomaly_classifier_wirable": anomaly_wirable,
        "tier3_fast_path_landed": tier3_fast_path_landed,
        "edlm_operator_surface": edlm_operator_surface,
        "archived_task_count_recorded": len(V348_TASKS),
        "terminal_task_count_recorded": len(V348_TASKS),
        "both_energy_routes_bounded": capstone.get("energy_as_generator_still_bounded") is True,
        "frozen_headline_auroc": safe_point(capstone.get("frozen_fover_auroc")),
    }
