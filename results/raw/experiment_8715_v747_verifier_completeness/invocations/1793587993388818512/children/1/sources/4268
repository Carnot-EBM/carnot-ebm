"""Archive milestone .346's 10-of-11 record and confirm .347 is active.

Spec refs: REQ-REPORT-3786, SCENARIO-REPORT-3786.
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
ARCHIVED_MILESTONE = "2026.06.346"
ACTIVATED_MILESTONE = "2026.06.347"
RANDOM_SEED = 3786
OUTPUT_REL_PATH = Path("results/experiment_3786_archive_v346_activate_v347.json")
RESEARCH_COMPLETE_REL_PATH = Path("research-complete.yaml")
ROADMAP_DESIGN_REL_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CAPSTONE_REL_PATH = Path("results/experiment_3785_capstone_v346.json")
CHANGELOG_REL_PATH = Path("ops/changelog.md")
NORTH_STAR_REL_PATH = Path("ops/north-star.md")

INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_VERDICT = (
    "complete: "
    "archived_v346_landed_10_of_11_exp3777_blocked_no_free_gpu_"
    "v347_post_convergence_active_paper_ready_true_both_energy_routes_bounded_"
    "frozen_headline_unchanged"
)
V346_OUTCOME_RECORDED = (
    "landed_10_of_11_verifier_product_banked_fr11_tier2_anomaly_escalation_"
    "edlm_seed_scaffolded_g4_correction_prepped_exp3777_blocked_no_free_gpu"
)
V347_FOCUS_RECORDED = (
    "retry_p1_v3_harden_banked_product_tier3_self_learning_validate_anomaly_"
    "edlm_preflight_regrind_nothing_bounded"
)
BLOCKED_TASK_ID = "exp3777-p1-discrete-search-adjudication-v3"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "v346_outcome_recorded",
    "v347_focus_recorded",
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
        "JSON-read + format; 0.0001s floor; no compute-bound marker so it does "
        "not false-flag."
    ),
    "v346_outcome_recorded": (
        "Records .346's REAL state: landed 10/11, verifier product banked, "
        "paper_ready TRUE, single un-landed exp3777 blocked on no-free-GPU "
        "(not a negative)."
    ),
    "v347_focus_recorded": (
        "Records the .347 agenda: retry P1 v3, harden the banked product, "
        "Tier-3 self-learning, anomaly-escalation validation, EDLM preflight "
        "-- re-grinding nothing bounded."
    ),
    "research_complete_yaml_parses": (
        "BARE bool, MUST be true -- confirms research-complete.yaml safe_loads "
        "after the write (anti-recurrence of the .344 poison)."
    ),
    "paper_ready_preserved": (
        "BARE bool -- G1-G4 stay met (confirmed via publication_gate.py); the "
        "transition must not silently regress paper_ready; frozen 0.9131 stays frozen."
    ),
    "both_energy_routes_still_bounded": (
        "BARE bool -- records that .347 does not reopen the bounded conclusion "
        "(P1 v3 only sharpens the mechanism)."
    ),
    "n_tasks_archived": (
        "Sample-size hygiene -- confirms the full .346 milestone was archived, "
        "not a partial."
    ),
    "adversarial_verify_clean": (
        "True iff the artifact passes adversarial_verify with no critical flag."
    ),
    "random_seed": "Determinism precondition for reproducibility.",
    "reproducibility_checksum": "Content hash catches silent drift vs any replication.",
    "duration_s": "Wall-clock plausibility floor; missing duration is the fabrication signal.",
}

V346_TASKS = [
    {
        "id": "exp3776-archive-v345-activate-v346",
        "title": "Archive .345 and activate .346 convergence",
        "deliverable": "results/experiment_3776_archive_v345_activate_v346.json",
        "result": (
            "COMPLETE: archived .345 fully landed and activated .346 "
            "post-bounded convergence"
        ),
    },
    {
        "id": BLOCKED_TASK_ID,
        "title": "P1 discrete-search adjudication v3",
        "deliverable": "results/experiment_3777_p1_discrete_search_adjudication_v3.json",
        "result": (
            "BLOCKED_RESOURCE: exp3777 blocked on no-free-GPU; the P1 mechanism "
            "question remains open and this is not a research negative"
        ),
    },
    {
        "id": "exp3778-fr11-self-learning-v18-tier2-constraint-memory",
        "title": "FR-11 v18 Tier-2 constraint memory",
        "deliverable": "results/experiment_3778_fr11_self_learning_v18_tier2_constraint_memory.json",
        "result": (
            "COMPLETE: Tier-2 constraint memory consolidated; AUROC stayed within "
            "the frozen CI and memory contribution was preserved"
        ),
    },
    {
        "id": "exp3779-abstention-operating-point-product-wiring",
        "title": "Abstention operating point product wiring",
        "deliverable": "results/experiment_3779_abstention_operating_point_product_wiring.json",
        "result": (
            "COMPLETE: abstention mode wired into the verify API, default-OFF, "
            "and E2E behavior confirmed"
        ),
    },
    {
        "id": "exp3779-mcp-abstention-surface-confirmation",
        "title": "MCP abstention surface confirmation",
        "deliverable": "results/experiment_3779_abstention_operating_point_product_wiring.json",
        "result": (
            "COMPLETE: banked verifier product was confirmed reachable through "
            "the MCP score_candidates surface"
        ),
    },
    {
        "id": "exp3780-anomaly-escalation-classifier-prototype",
        "title": "P3 Anomaly-Escalation classifier prototype",
        "deliverable": "results/experiment_3780_anomaly_escalation_classifier_prototype.json",
        "result": (
            "COMPLETE: recommend-only anomaly-escalation classifier prototyped "
            "and change proposal written"
        ),
    },
    {
        "id": "exp3781-edlm-next-thesis-feasibility-scoping",
        "title": "EDLM next-thesis feasibility scoping",
        "deliverable": "results/experiment_3781_edlm_next_thesis_feasibility_scoping.json",
        "result": (
            "COMPLETE: EDLM feasibility brief and minimal kill-gate scaffolded "
            "as an operator decision surface"
        ),
    },
    {
        "id": "exp3782-technical-report-g4-correction-prep",
        "title": "Technical-report G4 correction prep",
        "deliverable": "results/experiment_3782_technical_report_g4_correction_prep.json",
        "result": (
            "COMPLETE: G4 correction proposal prepared without editing the "
            "operator-curated report"
        ),
    },
    {
        "id": "exp3783-external-research-refresh",
        "title": "External research refresh .346",
        "deliverable": "results/experiment_3783_external_research_refresh.json",
        "result": (
            "COMPLETE: .346 references filed append-only with numbers treated "
            "as reported"
        ),
    },
    {
        "id": "exp3784-kv260-opportunistic-continuity-audit",
        "title": "KV260 opportunistic continuity audit",
        "deliverable": "results/experiment_3784_kv260_opportunistic_continuity_audit.json",
        "result": "COMPLETE: KV260 terminal state held; SSH reachable and overlay loadable",
    },
    {
        "id": "exp3785-capstone-v346",
        "title": "Capstone .346",
        "deliverable": "results/experiment_3785_capstone_v346.json",
        "result": (
            "COMPLETE: capstone recorded exp3777 as blocked, not a negative; "
            "paper_ready TRUE, FoVer 0.9131 frozen, both energy routes bounded"
        ),
    },
]


def build_research_complete_block() -> str:
    """Return the single honest `research-complete.yaml` block for .346."""

    finding = (
        "LANDED 10/11 MILESTONE: .346 banked the verifier product, advanced "
        "FR-11 to Tier-2, prototyped Anomaly-Escalation, scaffolded the EDLM "
        "seed, prepped the G4 correction, refreshed references, confirmed KV260 "
        "continuity, and completed the capstone. The single un-landed task was "
        "exp3777 P1 discrete-search v3, blocked on no-free-GPU; this is not a "
        "research negative. paper_ready TRUE (G1-G4), frozen FoVer 0.9131 "
        "unchanged, and both energy routes stayed bounded. .347 post-convergence "
        "is active: retry P1 v3, harden the banked product, continue Tier-3 "
        "self-learning, validate anomaly-escalation, scaffold EDLM preflight, "
        "and re-grind nothing bounded."
    )
    lines = [
        f"- id: {ARCHIVED_MILESTONE}",
        f"  title: {yaml_string('V346 post-bounded convergence landed 10 of 11')}",
        "  doc: openspec/change-proposals/research-roadmap-vNEXT.md",
        "  completed: '2026-06-04'",
        f"  finding: {yaml_string(finding)}",
        "  tasks:",
    ]
    for task in V346_TASKS:
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
    """Replace or append the `.346` archive block without duplicating it."""

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
    """Build the Exp 3786 terminal artifact from checked-in evidence."""

    root_path = Path(root)
    active_milestone, active_roadmap_path = read_active_milestone(root_path)
    _ensure(active_milestone == ACTIVATED_MILESTONE, ".347 active milestone confirmation is required")

    active_roadmap = root_path / active_roadmap_path
    roadmap_text = active_roadmap.read_text(encoding="utf-8")
    design_text = (root_path / ROADMAP_DESIGN_REL_PATH).read_text(encoding="utf-8")
    roadmap_evidence = f"{roadmap_text}\n{design_text}".lower()
    _ensure(
        all(
            token in roadmap_evidence
            for token in (
                "2026.06.347",
                "p1",
                "banked verifier",
                "tier-3",
                "anomaly-escalation",
                "edlm",
                "no-free-gpu",
            )
        ),
        ".347 roadmap evidence must record the post-convergence agenda",
    )
    _ensure(
        "re-grind" in roadmap_evidence or "regrind" in roadmap_evidence,
        ".347 roadmap evidence must record no bounded re-grind",
    )

    changelog_text = (root_path / CHANGELOG_REL_PATH).read_text(encoding="utf-8")
    changelog_evidence = changelog_text.lower()
    _ensure(
        "capstone .346" in changelog_evidence
        and "blocked_missing_upstream_artifact" in changelog_evidence
        and "paper_ready" in changelog_evidence,
        "changelog must confirm .346 capstone and blocked P1 evidence",
    )

    capstone = read_json_object(root_path / CAPSTONE_REL_PATH)
    v346_evidence = extract_v346_capstone_evidence(capstone, roadmap_evidence)
    _ensure(v346_evidence["record_honest"] is True, ".346 capstone evidence must be honest")

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
        "schema": "carnot.archive_activation.v346_to_v347_3786.v1",
        "experiment_id": "exp3786",
        "task_id": "exp3786-archive-v346-activate-v347",
        "archived_milestone": ARCHIVED_MILESTONE,
        "activated_milestone": ACTIVATED_MILESTONE,
        "active_roadmap_path": active_roadmap_path,
        "v347_active_confirmed": active_milestone == ACTIVATED_MILESTONE,
        "honest_verdict": TERMINAL_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "v346_outcome_recorded": V346_OUTCOME_RECORDED,
        "v347_focus_recorded": V347_FOCUS_RECORDED,
        "research_complete_yaml_parses": research_complete_yaml_parses,
        "paper_ready_preserved": paper_ready_preserved,
        "both_energy_routes_still_bounded": both_energy_routes_still_bounded,
        "n_tasks_archived": len(V346_TASKS),
        "n_tasks_landed": len(V346_TASKS) - 1,
        "blocked_task_ids": [BLOCKED_TASK_ID],
        "adversarial_verify_clean": report_is_clean(report),
        "adversarial_verify_report": report,
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "field_principles": dict(FIELD_PRINCIPLES),
        "paper_ready_evidence": paper_ready_evidence,
        "v346_capstone_evidence": v346_evidence,
        "v347_activation_evidence": {
            "active_milestone": active_milestone,
            "active_roadmap_path": active_roadmap_path,
            "first_task": "exp3786-archive-v346-activate-v347",
            "post_convergence_focus": V347_FOCUS_RECORDED,
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
    """Write the honest `.346` archive and terminal Exp 3786 artifact."""

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
    """Validate the required Exp 3786 archive/activation contract."""

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
    _ensure(artifact.get("v346_outcome_recorded") == V346_OUTCOME_RECORDED, ".346 outcome mismatch")
    _ensure(artifact.get("v347_focus_recorded") == V347_FOCUS_RECORDED, ".347 focus mismatch")
    _ensure(artifact.get("v347_active_confirmed") is True, ".347 active confirmation required")
    _ensure(artifact.get("research_complete_yaml_parses") is True, "safe-load confirmation required")
    _ensure(artifact.get("paper_ready_preserved") is True, "paper_ready must remain preserved")
    _ensure(
        artifact.get("both_energy_routes_still_bounded") is True,
        "both energy routes must remain bounded",
    )
    _ensure(artifact.get("n_tasks_archived") == 11, "n_tasks_archived must equal 11")
    _ensure(artifact.get("n_tasks_landed") == 10, "n_tasks_landed must equal 10")
    _ensure(artifact.get("blocked_task_ids") == [BLOCKED_TASK_ID], "exp3777 blocked task must be recorded")
    _ensure(artifact.get("adversarial_verify_clean") is True, "adversarial_verify_clean must be true")
    _ensure(artifact.get("random_seed") == RANDOM_SEED, "random_seed must equal 3786")
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
    _ensure(report_is_clean(artifact.get("adversarial_verify_report", {"flags": []})), "adversarial report has critical flag")


def extract_v346_capstone_evidence(
    capstone: Mapping[str, Any],
    roadmap_evidence: str,
) -> JsonDict:
    """Extract the `.346` 10-of-11 facts from Exp 3785 plus roadmap evidence."""

    not_landed = capstone.get("not_landed_or_blocked_recorded_honestly")
    flagged = capstone.get("flagged_artifacts_excluded")
    exp3777_record = next(
        (
            dict(item)
            for item in not_landed
            if isinstance(not_landed, list)
            and isinstance(item, Mapping)
            and item.get("experiment_id") == 3777
        ),
        None,
    )
    exp3777_blocked = bool(
        exp3777_record
        and exp3777_record.get("status") in {"not-landed", "blocked"}
        and capstone.get("p1_adjudication") == "blocked_missing_upstream_artifact"
    )
    exp3777_no_free_gpu = exp3777_blocked and "no-free-gpu" in roadmap_evidence
    record_honest = bool(
        capstone.get("paper_ready_preserved") is True
        and capstone.get("energy_as_generator_still_bounded") is True
        and capstone.get("energy_as_selector_status") == "honest-negative-bounded"
        and capstone.get("energy_as_generator_status") == "honest-negative-bounded"
        and capstone.get("verifier_product_banked") is True
        and capstone.get("anomaly_escalation_prototyped") is True
        and capstone.get("edlm_seed_scaffolded") is True
        and capstone.get("fr11_v18_self_learning") is True
        and capstone.get("g4_correction_prepped") is True
        and capstone.get("frozen_headline_unchanged") is True
        and safe_point(capstone.get("frozen_fover_auroc")) == 0.9131
        and exp3777_no_free_gpu
        and isinstance(flagged, list)
        and not flagged
    )
    return {
        "capstone_path": str(CAPSTONE_REL_PATH),
        "honest_verdict": capstone.get("honest_verdict"),
        "record_honest": record_honest,
        "p1_adjudication": capstone.get("p1_adjudication"),
        "p1_mechanism_status": capstone.get("p1_mechanism_status"),
        "p1_positive_control_passed": capstone.get("p1_positive_control_passed") is True,
        "exp3777_blocked_no_free_gpu": exp3777_no_free_gpu,
        "exp3777_record": exp3777_record,
        "landed_task_count_recorded": 10,
        "archived_task_count_recorded": len(V346_TASKS),
        "verifier_product_banked": capstone.get("verifier_product_banked") is True,
        "anomaly_escalation_prototyped": capstone.get("anomaly_escalation_prototyped") is True,
        "edlm_seed_scaffolded": capstone.get("edlm_seed_scaffolded") is True,
        "fr11_v18_self_learning": capstone.get("fr11_v18_self_learning") is True,
        "g4_correction_prepped": capstone.get("g4_correction_prepped") is True,
        "both_energy_routes_bounded": capstone.get("energy_as_generator_still_bounded") is True,
        "flagged_artifacts_excluded": list(flagged) if isinstance(flagged, list) else None,
    }
