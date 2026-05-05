"""Build the Exp 1350 milestone .104 carry-forward retrospective artifact.

Spec: REQ-REPORT-029, SCENARIO-REPORT-029.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT_FOR_METADATA = "/home/ianblenke/github.com/ianblenke/carnot"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUT_PATH = DEFAULT_RESULTS_DIR / "experiment_1350_milestone_104_retro_carryforward.json"

EXPERIMENT = "1350_milestone_104_retro_carryforward"
SCHEMA = "milestone_retro_104_carryforward_v1"
RUN_DATE = "20260505"
MILESTONE = "2026.04.104"

MET = "MET"
MISSING = "MISSING"
GATED = "GATED"
FAILED = "FAILED"

SOURCE_FILES = {
    1337: "experiment_1337_environment_gate_disk_pretest_stale_skeleton_audit.json",
    1338: "experiment_1338_exp1325_skeleton_and_gate_state_finalizer.json",
    1339: "experiment_1339_xgrammar2_tagdispatch_certificate_grammar_dryrun.json",
    1340: "experiment_1340_trigger_before_constrain_certificate_v6_sota.json",
    1341: "experiment_1341_halluguard_certificate_failure_split.json",
    1342: "experiment_1342_chopchop_nsvif_semantic_validator_gated.json",
    1343: "experiment_1343_margin_aware_beaver_cactus_scheduler.json",
    1344: "experiment_1344_continuous_self_learning_failure_type_memory_policy.json",
    1345: "experiment_1345_dvi_certificate_tail_v3_gated.json",
    1346: "experiment_1346_grpo_vprm_v13_gated_micro_audit.json",
    1347: "experiment_1347_thrml_compatibility_parity_audit.json",
    1348: "experiment_1348_pbit_update_dynamics_dual_bram_packet_v2.json",
    1349: "experiment_1349_ebt_citation_kona_parity_gap_audit.json",
}

CRITERIA: tuple[str, ...] = (
    "environment_gate_ready",
    "exp1325_stale_gate_state_closed",
    "dynamic_grammar_ready_or_terminal_blocker",
    "triggered_certificate_branch_recovered_or_retired",
    "halluguard_failure_split_no_universal_detector",
    "semantic_validator_executed_unknown_preserved",
    "margin_aware_scheduler_false_acceptance_risk_reported",
    "continuous_self_learning_accounted",
    "dvi_grpo_gate_discipline_preserved",
    "hardware_portability_evidence_without_unverified_claims",
    "external_ebt_kona_parity_mapped",
    "retro_104_complete",
)
CRITERION_NAMES = CRITERIA

REQUIRED_ARTIFACT_FIELDS = {
    "status",
    "criteria_total",
    "criteria_met",
    "experiment_statuses",
    "certificate_branch_verdict",
    "self_learning_verdict",
    "hardware_verdict",
    "publication_hold_state",
    "carry_forward_tasks",
    "prior_failure_hygiene_notes",
    "honest_verdict",
}


def _write_json(path: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def write_in_progress_artifact(out_path: Path | str = DEFAULT_OUT_PATH) -> dict[str, Any]:
    """REQ-REPORT-029: persist a bootstrap marker before reading source evidence."""

    return _write_json(
        Path(out_path),
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "run_date": RUN_DATE,
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "milestone": MILESTONE,
            "artifact_metadata": {
                "run_date": RUN_DATE,
                "project_root": PROJECT_ROOT_FOR_METADATA,
                "source_experiments": [f"exp{exp_id}" for exp_id in SOURCE_FILES],
            },
            "status": "in_progress",
            "criteria_total": len(CRITERIA),
            "criteria_met": 0,
            "experiment_statuses": {},
            "certificate_branch_verdict": {"headline_ready": False},
            "self_learning_verdict": {"headline_ready": False},
            "hardware_verdict": {"hardware_execution_claim_allowed": False},
            "publication_hold_state": {"hold_active": True, "hold_lift_evidence": "not_evaluated"},
            "carry_forward_tasks": [],
            "prior_failure_hygiene_notes": {},
            "honest_verdict": "milestone_104_in_progress",
        },
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_sources(results_dir: Path) -> tuple[dict[int, dict[str, Any]], set[int]]:
    sources: dict[int, dict[str, Any]] = {}
    missing: set[int] = set()
    for exp_id, filename in SOURCE_FILES.items():
        payload = _read_json(results_dir / filename)
        if payload is None:
            missing.add(exp_id)
        else:
            sources[exp_id] = payload
    return sources, missing


def _status(payload: Mapping[str, Any]) -> str:
    return str(payload.get("status", "")).lower()


def _honest_verdict(payload: Mapping[str, Any]) -> str:
    return str(payload.get("honest_verdict", ""))


def _is_complete(payload: Mapping[str, Any]) -> bool:
    return _status(payload) == "complete"


def _number_at_least(value: object, threshold: float) -> bool:
    return isinstance(value, int | float) and float(value) >= threshold


def _number_at_most(value: object, threshold: float) -> bool:
    return isinstance(value, int | float) and float(value) <= threshold


def _all_certificate_states_supported(payload: Mapping[str, Any]) -> bool:
    return {"REPAIR_HINT", "SAT", "UNKNOWN", "UNSAT"} <= set(payload.get("certificate_states_supported", []))


def _experiment_statuses(
    sources: Mapping[int, Mapping[str, Any]],
    missing_source_ids: set[int],
) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for exp_id in SOURCE_FILES:
        payload = sources.get(exp_id, {})
        status = "missing" if exp_id in missing_source_ids else _status(payload)
        statuses[f"exp{exp_id}"] = {
            "status": status,
            "artifact": SOURCE_FILES[exp_id],
            "honest_verdict": "" if exp_id in missing_source_ids else _honest_verdict(payload),
        }
        if status == "blocked":
            statuses[f"exp{exp_id}"]["blocked_at_layer"] = payload.get("blocked_at_layer", "")
            statuses[f"exp{exp_id}"]["gate_check_summary"] = payload.get("gate_check_summary", "")
    return statuses


def _criterion_results(
    sources: Mapping[int, Mapping[str, Any]],
    missing_source_ids: set[int],
) -> dict[str, str]:
    exp1337 = sources.get(1337, {})
    exp1338 = sources.get(1338, {})
    exp1339 = sources.get(1339, {})
    exp1341 = sources.get(1341, {})
    exp1343 = sources.get(1343, {})
    exp1344 = sources.get(1344, {})
    exp1347 = sources.get(1347, {})
    exp1348 = sources.get(1348, {})
    exp1349 = sources.get(1349, {})
    dvi_grpo_absent = 1345 in missing_source_ids and 1346 in missing_source_ids
    structured_gates_open = 1340 not in missing_source_ids and 1342 not in missing_source_ids

    return {
        "environment_gate_ready": (
            MET if _is_complete(exp1337) and exp1337.get("environment_ready") is True else FAILED
        ),
        "exp1325_stale_gate_state_closed": (
            MET
            if _is_complete(exp1338)
            and exp1338.get("exp1325_terminal_classification") == "stale_skeleton_environment_failure"
            and exp1338.get("certificate_recovery_ready") is True
            else FAILED
        ),
        "dynamic_grammar_ready_or_terminal_blocker": (
            MET
            if _is_complete(exp1339)
            and exp1339.get("dynamic_grammar_ready") is True
            and exp1339.get("unknown_state_supported") is True
            and _all_certificate_states_supported(exp1339)
            else FAILED
        ),
        "triggered_certificate_branch_recovered_or_retired": (
            MISSING
            if 1340 in missing_source_ids
            else (
                MET
                if _number_at_least(sources[1340].get("certificate_parse_rate"), 0.75)
                or sources[1340].get("certificate_branch_retired_with_evidence") is True
                else FAILED
            )
        ),
        "halluguard_failure_split_no_universal_detector": (
            MET
            if _is_complete(exp1341)
            and bool(exp1341.get("repair_policy_by_failure_type"))
            and exp1341.get("universal_detector_claim_allowed") is False
            else FAILED
        ),
        "semantic_validator_executed_unknown_preserved": (
            MISSING
            if 1342 in missing_source_ids
            else (
                MET
                if _is_complete(sources[1342])
                and _number_at_least(sources[1342].get("validator_execution_pass_rate"), 0.5)
                and sources[1342].get("unknown_state_preserved") is True
                else FAILED
            )
        ),
        "margin_aware_scheduler_false_acceptance_risk_reported": (
            GATED
            if _status(exp1343) == "blocked"
            else (
                MET
                if _is_complete(exp1343)
                and "false_acceptance_risk" in exp1343
                and exp1343.get("verifier_call_savings_claim_allowed") is not True
                else FAILED
            )
        ),
        "continuous_self_learning_accounted": (
            MET
            if _is_complete(exp1344)
            and _number_at_least(exp1344.get("nonforgetting_certificate_rate"), 0.9)
            and exp1344.get("memory_regression_count") == 0
            and _number_at_most(exp1344.get("accepted_violation_delta"), 0.0)
            else FAILED
        ),
        "dvi_grpo_gate_discipline_preserved": (
            MET if dvi_grpo_absent and not structured_gates_open else FAILED
        ),
        "hardware_portability_evidence_without_unverified_claims": (
            MET
            if _is_complete(exp1347)
            and _is_complete(exp1348)
            and exp1347.get("hardware_claim_allowed") is False
            and exp1348.get("hardware_claim_allowed") is False
            else FAILED
        ),
        "external_ebt_kona_parity_mapped": (
            MET
            if _is_complete(exp1349)
            and bool(exp1349.get("parity_gaps"))
            and exp1349.get("external_dependency_claim_allowed") is False
            else FAILED
        ),
        "retro_104_complete": MET,
    }


def _certificate_branch_verdict(
    sources: Mapping[int, Mapping[str, Any]],
    missing_source_ids: set[int],
) -> dict[str, Any]:
    return {
        "headline_ready": False,
        "environment_gate_ready": sources.get(1337, {}).get("environment_ready") is True,
        "dynamic_grammar_ready": sources.get(1339, {}).get("dynamic_grammar_ready") is True,
        "triggered_sota_artifact": "missing" if 1340 in missing_source_ids else _status(sources[1340]),
        "semantic_validator_artifact": "missing" if 1342 in missing_source_ids else _status(sources[1342]),
        "scheduler_artifact": _status(sources.get(1343, {})) or "missing",
        "claim_boundary": (
            "Dynamic grammar and failure taxonomy advanced, but the SOTA triggered "
            "certificate artifact is missing; semantic validation and scheduler claims remain gated."
        ),
    }


def _self_learning_verdict(
    sources: Mapping[int, Mapping[str, Any]],
    missing_source_ids: set[int],
) -> dict[str, Any]:
    exp1344 = sources.get(1344, {})
    return {
        "headline_ready": False,
        "replay_non_headline": exp1344.get("headline_result_allowed") is False,
        "dvi_ready_replay_only": exp1344.get("dvi_ready") is True,
        "dvi_artifact": "missing" if 1345 in missing_source_ids else _status(sources[1345]),
        "grpo_artifact": "missing" if 1346 in missing_source_ids else _status(sources[1346]),
        "claim_boundary": (
            "Failure-type memory policy preserved non-forgetting and reduced accepted "
            "violations in replay, but DVI and GRPO headline updates did not run."
        ),
    }


def _hardware_verdict(sources: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    exp1347 = sources.get(1347, {})
    exp1348 = sources.get(1348, {})
    exp1349 = sources.get(1349, {})
    return {
        "hardware_execution_claim_allowed": False,
        "thrml_status": _honest_verdict(exp1347),
        "pbit_status": _honest_verdict(exp1348),
        "external_parity_status": _honest_verdict(exp1349),
        "claim_boundary": (
            "Hardware work improved THRML mapping notes, p-bit packet accounting, and "
            "Kona/EBT parity obligations, but no FPGA, TSU, analog, or Kona execution was proven."
        ),
    }


def _publication_hold_state(sources: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    exp1349 = sources.get(1349, {})
    return {
        "hold_active": True,
        "hold_lift_evidence": "absent",
        "submission_ready": False,
        "source": "exp1349_publication_claim_changes_needed",
        "notes": exp1349.get("publication_claim_changes_needed", []),
    }


def _carry_forward_tasks() -> list[dict[str, Any]]:
    return [
        {
            "task": "produce_terminal_exp1340_or_retire_triggered_certificate_branch",
            "why": "Exp 1340 artifact is missing, so the parse gate and branch-retirement evidence are absent.",
            "prior_failures": ["exp1325_stale_skeleton_environment_failure", "exp1340_missing"],
        },
        {
            "task": "run_semantic_validator_only_after_parse_gate",
            "why": "Exp 1342 is missing because the SOTA certificate parse evidence did not exist.",
            "prior_failures": ["exp1342_missing_after_exp1340_missing"],
        },
        {
            "task": "rerun_margin_scheduler_after_semantic_validator_execution",
            "why": "Exp 1343 blocked on missing Exp 1342 validator_execution_pass_rate evidence.",
            "prior_failures": ["exp1343_blocked_gate_check_failed"],
        },
        {
            "task": "keep_dvi_and_grpo_closed_until_parse_semantic_and_nonforgetting_gates_pass",
            "why": "Exp 1344 is replay-positive but non-headline; Exp 1345 and Exp 1346 did not run.",
            "prior_failures": ["exp1345_missing_gated", "exp1346_missing_gated"],
        },
        {
            "task": "reconcile_specs_ops_and_docs_after_conductor_handoff",
            "why": "This run intentionally avoids ops/status, ops/changelog, research-complete, and traceability edits.",
            "prior_failures": ["docs_reconciliation_deferred_by_conductor_prompt"],
        },
        {
            "task": "keep_publication_hold_active_until_live_certificate_and_external_parity_claims_are_supported",
            "why": "No .104 artifact provides hold-lift evidence or valid submission readiness.",
            "prior_failures": ["publication_hold_active_no_lift_evidence"],
        },
    ]


def _prior_failure_hygiene_notes(sources: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    exp1337 = sources.get(1337, {})
    exp1338 = sources.get(1338, {})
    repeated = exp1337.get("repeated_pretest_signature", {})
    return {
        "disk_quota_closed_cleanly": exp1337.get("disk_quota_ok") is True,
        "focused_pretest_closed_cleanly": (
            exp1337.get("focused_pretest_status") == "passed"
            and not repeated.get("focused_pretest_signature_active", True)
        ),
        "stale_skeleton_closed_cleanly": (
            exp1337.get("stale_skeleton_count") == 1
            and exp1338.get("exp1325_terminal_classification") == "stale_skeleton_environment_failure"
            and exp1338.get("stale_artifacts_not_modified") is True
        ),
        "note": (
            ".103 stale skeleton and pre-test issues were classified as environment/scheduler "
            "facts; the scientific certificate branch still needs fresh Exp 1340 evidence."
        ),
    }


def build_artifact(
    sources: Mapping[int, Mapping[str, Any]],
    missing_source_ids: set[int],
    *,
    roadmap_next_present: bool,
    active_roadmap_present: bool,
) -> dict[str, Any]:
    """REQ-REPORT-029: build the terminal .104 retrospective from source fields."""

    criteria_results = _criterion_results(sources, missing_source_ids)
    criteria_met = sum(1 for status in criteria_results.values() if status == MET)
    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "project_root": PROJECT_ROOT_FOR_METADATA,
        "milestone": MILESTONE,
        "artifact_metadata": {
            "run_date": RUN_DATE,
            "project_root": PROJECT_ROOT_FOR_METADATA,
            "source_experiments": [f"exp{exp_id}" for exp_id in SOURCE_FILES],
        },
        "status": "complete",
        "criteria_total": len(CRITERIA),
        "criteria_met": criteria_met,
        "criteria_results": criteria_results,
        "experiment_statuses": _experiment_statuses(sources, missing_source_ids),
        "certificate_branch_verdict": _certificate_branch_verdict(sources, missing_source_ids),
        "self_learning_verdict": _self_learning_verdict(sources, missing_source_ids),
        "hardware_verdict": _hardware_verdict(sources),
        "publication_hold_state": _publication_hold_state(sources),
        "carry_forward_tasks": _carry_forward_tasks(),
        "prior_failure_hygiene_notes": _prior_failure_hygiene_notes(sources),
        "roadmap_inputs": {
            "research_roadmap_next_yaml_present": roadmap_next_present,
            "research_roadmap_yaml_present": active_roadmap_present,
            "change_proposal_present": True,
            "missing_requested_inputs": [] if roadmap_next_present else ["research-roadmap-next.yaml"],
        },
        "honest_verdict": f"milestone_104_{criteria_met}_of_{len(CRITERIA)}_criteria_met_carryforward_required",
    }


def run(
    root: Path | str = REPO_ROOT,
    out_path: Path | str = DEFAULT_OUT_PATH,
) -> dict[str, Any]:
    """REQ-REPORT-029: write bootstrap, read sources, and persist the final artifact."""

    root_path = Path(root)
    out = Path(out_path)
    write_in_progress_artifact(out)
    sources, missing_source_ids = _load_sources(root_path / "results")
    artifact = build_artifact(
        sources,
        missing_source_ids,
        roadmap_next_present=(root_path / "research-roadmap-next.yaml").exists(),
        active_roadmap_present=(root_path / "research-roadmap.yaml").exists(),
    )
    return _write_json(out, artifact)
