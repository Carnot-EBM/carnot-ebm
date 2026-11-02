"""Archive milestone .347's terminal record and confirm .348 is active.

Spec refs: REQ-REPORT-3797, SCENARIO-REPORT-3797,
SCENARIO-REPORT-3797-P1-HANDOFF-GUARD.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import time
from typing import Any

import yaml

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
ARCHIVED_MILESTONE = "2026.06.347"
ACTIVATED_MILESTONE = "2026.06.348"
RANDOM_SEED = 3797
OUTPUT_REL_PATH = Path("results/experiment_3797_archive_v347_activate_v348.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
ROADMAP_DESIGN_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CAPSTONE_REL_PATH = Path("results/experiment_3796_capstone_v347.json")
CHANGELOG_REL_PATH = Path("ops/changelog.md")
NORTH_STAR_REL_PATH = Path("ops/north-star.md")

INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_VERDICT = (
    "complete: "
    "archived_v347_landed_all_p1_blocked_no_free_gpu_handed_to_operator_"
    "v348_headline_advancement_active_paper_ready_true_both_energy_routes_bounded_"
    "frozen_headline_unchanged"
)
V347_OUTCOME_RECORDED = (
    "landed_all_tasks_p1_blocked_no_free_gpu_handed_to_operator_edlm_preflight_go_"
    "product_headline_partially_restorable_paper_ready_true"
)
V348_FOCUS_RECORDED = (
    "g4_headline_restoration_product_harden_repair_classifier_tuning_"
    "tier3_fast_path_no_bounded_regrind_no_paradigm_self_seed"
)
P1_HANDOFF_TASK_ID = "exp3787-p1-discrete-search-adjudication-v3-retry"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "v347_outcome_recorded",
    "v348_focus_recorded",
    "p1_handed_to_operator_recorded",
    "research_complete_yaml_parses",
    "paper_ready_preserved",
    "both_energy_routes_still_bounded",
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
    "v347_outcome_recorded": (
        "Records .347's REAL state: landed all tasks, P1 blocked and handed "
        "off, EDLM preflight GO, product headline partially restorable, "
        "paper_ready TRUE."
    ),
    "v348_focus_recorded": (
        "Records the .348 agenda: G4 headline restoration, product "
        "harden/repair, classifier tuning, Tier-3 self-learning -- re-grinding "
        "nothing bounded."
    ),
    "p1_handed_to_operator_recorded": (
        "BARE bool -- records that the P1 discrete-search adjudication is "
        "handed to the operator after a 2nd consecutive resource block and is "
        "not re-queued in .348."
    ),
    "research_complete_yaml_parses": (
        "BARE bool, MUST be true -- confirms research-complete.yaml safe_loads "
        "after the write."
    ),
    "paper_ready_preserved": (
        "BARE bool -- G1-G4 stay met; the transition must not silently regress "
        "paper_ready; frozen 0.9131 stays frozen."
    ),
    "both_energy_routes_still_bounded": (
        "BARE bool -- records that .348 does not reopen the bounded conclusion "
        "and runs no energy-foundation experiment."
    ),
    "n_tasks_archived": (
        "Sample-size hygiene -- confirms the full .347 milestone was archived, "
        "not a partial."
    ),
    "adversarial_verify_clean": (
        "True iff the artifact passes adversarial_verify with no critical flag."
    ),
    "random_seed": "Determinism precondition for reproducibility.",
    "reproducibility_checksum": "Content hash catches silent drift vs any replication.",
    "duration_s": "Wall-clock plausibility floor; missing duration is the fabrication signal.",
}

V347_TASKS = [
    {
        "id": "exp3786-archive-v346-activate-v347",
        "title": "Archive .346 and activate .347 post-convergence",
        "deliverable": "results/experiment_3786_archive_v346_activate_v347.json",
        "result": (
            "COMPLETE: archived .346 as 10 of 11 with exp3777 blocked on "
            "no-free-GPU, then activated .347 post-convergence"
        ),
    },
    {
        "id": P1_HANDOFF_TASK_ID,
        "title": "P1 discrete-search adjudication v3 retry",
        "deliverable": "results/experiment_3787_p1_discrete_search_adjudication_v3_retry.json",
        "result": (
            "BLOCKED_RESOURCE_HANDOFF: exp3787 blocked on no-free-GPU for the "
            "second consecutive milestone and set handoff_to_operator=true; "
            "the P1 adjudication is handed to the operator, not re-queued in "
            ".348, and not a research negative"
        ),
    },
    {
        "id": "exp3788-fr11-self-learning-v19-tier3-predictive",
        "title": "FR-11 v19 Tier-3 predictive verification",
        "deliverable": "results/experiment_3788_fr11_self_learning_v19_tier3_predictive.json",
        "result": (
            "COMPLETE: Tier-3 predictor trained with predictive AUROC 0.9715; "
            "headline ensemble unchanged and memory contribution preserved"
        ),
    },
    {
        "id": "exp3789-abstention-cli-batch-surface",
        "title": "Abstention CLI and batch surface",
        "deliverable": "results/experiment_3789_abstention_cli_batch_surface.json",
        "result": (
            "COMPLETE: abstention mode surfaced through CLI and batch scoring, "
            "default-OFF with E2E behavior confirmed"
        ),
    },
    {
        "id": "exp3790-verifier-gaming-resistance-characterization",
        "title": "Verifier gaming-resistance characterization",
        "deliverable": "results/experiment_3790_verifier_gaming_resistance_characterization.json",
        "result": (
            "COMPLETE: gaming-resistance curve characterized at n=240; "
            "context_compaction degradation recorded as product hardening work"
        ),
    },
    {
        "id": "exp3791-anomaly-escalation-classifier-validation",
        "title": "Anomaly-Escalation classifier validation",
        "deliverable": "results/experiment_3791_anomaly_escalation_classifier_validation.json",
        "result": (
            "COMPLETE: classifier validated recommend-only, but false-escalation "
            "rate 0.833333 means it needs tuning before wiring"
        ),
    },
    {
        "id": "exp3792-product-headline-provenance-confirmation-g4",
        "title": "Product headline G4 provenance confirmation",
        "deliverable": "results/experiment_3792_product_headline_provenance_confirmation_g4.json",
        "result": (
            "COMPLETE: product headline partially restorable; exp2090 passes G4 "
            "and exp1999 fails G4 because seed, checksum, n, and substrate are "
            "absent on disk"
        ),
    },
    {
        "id": "exp3793-edlm-no-train-preflight-readiness",
        "title": "EDLM no-train preflight readiness",
        "deliverable": "results/experiment_3793_edlm_no_train_preflight_readiness.json",
        "result": (
            "COMPLETE: EDLM preflight returned GO and emitted an operator seed "
            "command; the loop did not commit or self-seed"
        ),
    },
    {
        "id": "exp3794-external-research-refresh",
        "title": "External research refresh .347",
        "deliverable": "results/experiment_3794_external_research_refresh.json",
        "result": "COMPLETE: .347 references filed append-only with numbers treated as reported",
    },
    {
        "id": "exp3795-kv260-opportunistic-continuity-audit",
        "title": "KV260 opportunistic continuity audit",
        "deliverable": "results/experiment_3795_kv260_opportunistic_continuity_audit.json",
        "result": "COMPLETE: KV260 terminal state held; SSH reachable and overlay loadable",
    },
    {
        "id": "exp3796-capstone-v347",
        "title": "Capstone .347",
        "deliverable": "results/experiment_3796_capstone_v347.json",
        "result": (
            "COMPLETE: capstone recorded P1 as blocked and handed off, "
            "paper_ready TRUE, FoVer 0.9131 frozen, both energy routes bounded, "
            "and .348 headline-advancement focus established"
        ),
    },
]


def build_research_complete_block() -> str:
    """Return the single honest `research-complete.yaml` block for .347."""

    finding = (
        "LANDED TERMINAL MILESTONE: .347 completed its post-convergence archive. "
        "The P1 discrete-search v3 retry, exp3787, blocked on no-free-GPU for "
        "the second consecutive milestone and set handoff_to_operator=true; it "
        "is handed to the operator, not re-queued in .348, and not a research "
        "negative. EDLM preflight returned GO but remains operator-gated. The "
        "product headline is partially restorable: exp2090 passes G4 and "
        "exp1999 fails G4 because provenance is absent on disk. FR-11 Tier-3 "
        "predictive self-learning trained, abstention CLI/batch and "
        "gaming-resistance product work landed, the anomaly classifier was "
        "validated but over-fired and needs tuning, references refreshed, KV260 "
        "continuity held, paper_ready TRUE (G1-G4), frozen FoVer 0.9131 "
        "unchanged, and both energy routes stayed bounded. .348 is active: "
        "restore the demoted code-repair headline to G4 eligibility if the "
        "clean re-run supports it, harden/repair the banked product, tune the "
        "classifier, wire Tier-3 as a fast path, re-grind nothing bounded, and "
        "self-seed no paradigm."
    )
    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  title: {yaml_string('V347 post-convergence terminal handoff archived')}",
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-06-04'",
        f"  finding: {yaml_string(finding)}",
        "  tasks:",
    ]
    for task in V347_TASKS:
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
    """Replace or append the `.347` archive block without duplicating it."""

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
    """Build the Exp 3797 terminal artifact from checked-in evidence."""

    root_path = Path(root)
    active_milestone, active_roadmap_path = read_active_milestone(root_path)
    _ensure(active_milestone == ACTIVATED_MILESTONE, ".348 active milestone confirmation is required")

    active_roadmap = root_path / active_roadmap_path
    roadmap_text = active_roadmap.read_text(encoding="utf-8")
    design_text = (root_path / ROADMAP_DESIGN_REL_PATH).read_text(encoding="utf-8")
    roadmap_evidence = f"{roadmap_text}\n{design_text}".lower()
    _ensure(
        all(
            token in roadmap_evidence
            for token in (
                "2026.06.348",
                "headline",
                "harden",
                "repair",
                "classifier",
                "tier-3",
                "edlm",
                "not re-queue",
            )
        ),
        ".348 roadmap evidence must record the headline-advancement agenda",
    )
    _ensure(
        "re-grind" in roadmap_evidence or "regrind" in roadmap_evidence,
        ".348 roadmap evidence must record no bounded re-grind",
    )

    changelog_text = (root_path / CHANGELOG_REL_PATH).read_text(encoding="utf-8")
    changelog_evidence = changelog_text.lower()
    _ensure(
        "2026.06.348" in changelog_evidence
        and ".347" in changelog_evidence
        and "no-free-gpu" in changelog_evidence
        and "handoff_to_operator=true" in changelog_evidence
        and "paper_ready" in changelog_evidence,
        "changelog must confirm .347 closeout and .348 planning evidence",
    )

    capstone = read_json_object(root_path / CAPSTONE_REL_PATH)
    v347_evidence = extract_v347_capstone_evidence(capstone, roadmap_evidence)
    _ensure(v347_evidence["record_honest"] is True, ".347 capstone evidence must be honest")

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
    )
    _ensure(paper_ready_preserved, "paper_ready and frozen headline must be preserved")
    _ensure(both_energy_routes_still_bounded, "both energy routes must stay bounded")

    report = compact_verify_report(adversarial_report or {"flags": [], "flag_count": 0, "max_severity": -1})
    duration_s = duration_from(started_s, now_s)
    payload: JsonDict = {
        "schema": "carnot.archive_activation.v347_to_v348_3797.v1",
        "experiment_id": "exp3797",
        "task_id": "exp3797-archive-v347-activate-v348",
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_roadmap_path": active_roadmap_path,
        "v348_active_confirmed": active_milestone == ACTIVATED_MILESTONE,
        "honest_verdict": TERMINAL_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "v347_outcome_recorded": V347_OUTCOME_RECORDED,
        "v348_focus_recorded": V348_FOCUS_RECORDED,
        "p1_handed_to_operator_recorded": True,
        "research_complete_yaml_parses": research_complete_yaml_parses,
        "paper_ready_preserved": paper_ready_preserved,
        "both_energy_routes_still_bounded": both_energy_routes_still_bounded,
        "n_tasks_archived": len(V347_TASKS),
        "n_tasks_terminal": len(V347_TASKS),
        "blocked_handoff_task_ids": [P1_HANDOFF_TASK_ID],
        "adversarial_verify_clean": report_is_clean(report),
        "adversarial_verify_report": report,
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "field_principles": dict(FIELD_PRINCIPLES),
        "paper_ready_evidence": paper_ready_evidence,
        "v347_capstone_evidence": v347_evidence,
        "v348_activation_evidence": {
            "active_milestone": active_milestone,
            "active_roadmap_path": active_roadmap_path,
            "first_task": "exp3797-archive-v347-activate-v348",
            "headline_advancement_focus": V348_FOCUS_RECORDED,
            "p1_requeued": False,
            "edlm_operator_gated": True,
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
            {"path": str(CHANGELOG_REL_PATH), "sha256": sha256_path(root_path / CHANGELOG_REL_PATH)},
            {"path": str(NORTH_STAR_REL_PATH), "sha256": sha256_path(root_path / NORTH_STAR_REL_PATH)},
        ],
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
    """Write the honest `.347` archive and terminal Exp 3797 artifact."""

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
    """Validate the required Exp 3797 archive/activation contract."""

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
    _ensure(artifact.get("v347_outcome_recorded") == V347_OUTCOME_RECORDED, ".347 outcome mismatch")
    _ensure(artifact.get("v348_focus_recorded") == V348_FOCUS_RECORDED, ".348 focus mismatch")
    _ensure(artifact.get("p1_handed_to_operator_recorded") is True, "P1 handoff confirmation required")
    _ensure(artifact.get("v348_active_confirmed") is True, ".348 active confirmation required")
    _ensure(artifact.get("research_complete_yaml_parses") is True, "safe-load confirmation required")
    _ensure(artifact.get("paper_ready_preserved") is True, "paper_ready must remain preserved")
    _ensure(
        artifact.get("both_energy_routes_still_bounded") is True,
        "both energy routes must remain bounded",
    )
    _ensure(artifact.get("n_tasks_archived") == 11, "n_tasks_archived must equal 11")
    _ensure(artifact.get("n_tasks_terminal") == 11, "n_tasks_terminal must equal terminal task count")
    _ensure(
        artifact.get("blocked_handoff_task_ids") == [P1_HANDOFF_TASK_ID],
        "exp3787 blocked handoff task must be recorded",
    )
    _ensure(artifact.get("adversarial_verify_clean") is True, "adversarial_verify_clean must be true")
    _ensure(artifact.get("random_seed") == RANDOM_SEED, "random_seed must equal 3797")
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


def extract_v347_capstone_evidence(
    capstone: Mapping[str, Any],
    roadmap_evidence: str,
) -> JsonDict:
    """Extract the `.347` handoff and `.348` agenda facts from Exp 3796."""

    not_landed = capstone.get("not_landed_or_blocked_recorded_honestly")
    flagged = capstone.get("flagged_artifacts_excluded")
    cited = capstone.get("cited_upstream_artifacts")
    exp3787_record = next(
        (
            dict(item)
            for item in not_landed
            if isinstance(not_landed, list)
            and isinstance(item, Mapping)
            and item.get("experiment_id") == 3787
        ),
        None,
    )
    exp3787_blocked = bool(
        exp3787_record
        and exp3787_record.get("status") == "blocked"
        and exp3787_record.get("reason") == "blocked_no_free_gpu"
        and capstone.get("p1_adjudication") == "blocked_no_free_gpu"
    )
    p1_handoff = capstone.get("p1_handoff_to_operator") is True
    edlm = capstone.get("edlm_seed_preflighted")
    fr11 = capstone.get("fr11_v19_tier3_self_learning")
    product = capstone.get("verifier_product_hardened")
    anomaly_raw = capstone.get("anomaly_escalation_validated")
    anomaly = (
        anomaly_raw
        if isinstance(anomaly_raw, Mapping)
        else capstone.get("anomaly_escalation_validation")
    )
    n_upstream = len(cited) if isinstance(cited, list) else 0
    exp3787_not_requeued = "not re-queue" in roadmap_evidence and "exp3787" in roadmap_evidence
    edlm_preflight_go = bool(
        isinstance(edlm, Mapping)
        and edlm.get("preflighted") is True
        and edlm.get("readiness_verdict") == "go"
        and edlm.get("loop_does_not_commit") is True
    )
    fr11_trained = bool(
        isinstance(fr11, Mapping)
        and fr11.get("validated") is True
        and safe_point(fr11.get("predictive_auroc")) == 0.9715
        and fr11.get("headline_ensemble_unchanged") is True
    )
    product_hardened = bool(
        isinstance(product, Mapping)
        and product.get("hardened") is True
        and product.get("abstention_cli_batch_surface") is True
        and product.get("gaming_resistance_curve") is True
        and product.get("product_headline_provenance_confirmed") is True
    )
    anomaly_needs_tuning = bool(
        isinstance(anomaly, Mapping)
        and anomaly.get("validated") is True
        and anomaly.get("supports_wiring_in") is False
        and safe_point(anomaly.get("false_escalation_rate")) == 0.8333
    )
    record_honest = bool(
        capstone.get("paper_ready_preserved") is True
        and capstone.get("energy_as_generator_still_bounded") is True
        and capstone.get("energy_as_selector_status") == "honest-negative-bounded"
        and capstone.get("energy_as_generator_status") == "honest-negative-bounded"
        and capstone.get("frozen_headline_unchanged") is True
        and safe_point(capstone.get("frozen_fover_auroc")) == 0.9131
        and capstone.get("product_headline_restorable") == "not_yet_eligible"
        and exp3787_blocked
        and p1_handoff
        and exp3787_not_requeued
        and edlm_preflight_go
        and fr11_trained
        and product_hardened
        and anomaly_needs_tuning
        and isinstance(flagged, list)
        and not flagged
        and n_upstream == 10
    )
    return {
        "capstone_path": str(CAPSTONE_REL_PATH),
        "honest_verdict": capstone.get("honest_verdict"),
        "record_honest": record_honest,
        "p1_adjudication": capstone.get("p1_adjudication"),
        "p1_mechanism_status": capstone.get("p1_mechanism_status"),
        "p1_positive_control_passed": capstone.get("p1_positive_control_passed") is True,
        "p1_handoff_to_operator": p1_handoff,
        "exp3787_blocked_no_free_gpu": exp3787_blocked,
        "exp3787_not_requeued_in_v348": exp3787_not_requeued,
        "exp3787_record": exp3787_record,
        "edlm_preflight_go": edlm_preflight_go,
        "product_headline_restorable": capstone.get("product_headline_restorable"),
        "fr11_tier3_predictive_auroc": 0.9715 if fr11_trained else None,
        "verifier_product_hardened": product_hardened,
        "anomaly_classifier_needs_tuning": anomaly_needs_tuning,
        "archived_task_count_recorded": len(V347_TASKS),
        "terminal_task_count_recorded": len(V347_TASKS),
        "both_energy_routes_bounded": capstone.get("energy_as_generator_still_bounded") is True,
        "flagged_artifacts_excluded": list(flagged) if isinstance(flagged, list) else None,
    }
