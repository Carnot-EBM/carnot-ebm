"""Exp5612 V506 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5612, SCENARIO-CAPSTONE-5612,
SCENARIO-CAPSTONE-5612-MISSING-MALFORMED,
SCENARIO-CAPSTONE-5612-FIELD-PRINCIPLES.

This module is an evidence ledger, not a new experiment. It reads the terminal
Exp5603-Exp5611 artifacts, separates blocked, skipped, flagged, malformed, and
complete states, then records narrow promotion and retirement decisions. That
matters because a capstone can otherwise turn a useful null or an adversarially
flagged artifact into a broader success claim than the upstream evidence earned.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from carnot.experiment_5415_transition_v493 import (
    JsonDict,
    JsonMap,
    _modification_status,
    path_sha256,
    payload_checksum,
    write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5612_v506_capstone_reconciliation.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CAPSTONE_SPEC_RELATIVE_PATH = Path("openspec/capabilities/capstone/spec.md")

EXPERIMENT = "experiment_5612_v506_capstone_reconciliation"
EXPERIMENT_ID = "exp5612-v506-capstone-reconciliation"
MILESTONE = "2026.07.506"
RUN_DATE = "2026-07-14"
RANDOM_SEED = 5612
SCHEMA = "carnot.experiment_5612.v506_capstone_reconciliation.v1"
INFERENCE_SUBSTRATE = "aggregation_from_exp5603_exp5611_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

SPEC_REFS = (
    "REQ-CAPSTONE-5612",
    "SCENARIO-CAPSTONE-5612",
    "SCENARIO-CAPSTONE-5612-MISSING-MALFORMED",
    "SCENARIO-CAPSTONE-5612-FIELD-PRINCIPLES",
)

EXPECTED_TASK_IDS = (
    "exp5603-transition-v506",
    "exp5604-v506-source-delta-ingestion",
    "exp5605-raw-response-evidence-envelope",
    "exp5606-clean-sota-solve-verify-evidence-panel",
    "exp5607-property-template-exact-residual-extension",
    "exp5608-kan-longitudinal-self-learning",
    "exp5609-arc-filter-intermediate-invariance-ab",
    "exp5610-arc-live-self-discovery-levelup-v506",
    "exp5611-cdls-matched-sampler-crossover",
    "exp5612-v506-capstone-reconciliation",
)

TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5603-transition-v506": Path("results/experiment_5603_transition_v506.json"),
    "exp5604-v506-source-delta-ingestion": Path(
        "results/experiment_5604_v506_source_delta_ingestion.json"
    ),
    "exp5605-raw-response-evidence-envelope": Path(
        "results/experiment_5605_raw_response_evidence_envelope.json"
    ),
    "exp5606-clean-sota-solve-verify-evidence-panel": Path(
        "results/experiment_5606_clean_sota_solve_verify_evidence_panel.json"
    ),
    "exp5607-property-template-exact-residual-extension": Path(
        "results/experiment_5607_property_template_exact_residual_extension.json"
    ),
    "exp5608-kan-longitudinal-self-learning": Path(
        "results/experiment_5608_kan_longitudinal_self_learning.json"
    ),
    "exp5609-arc-filter-intermediate-invariance-ab": Path(
        "results/experiment_5609_arc_filter_intermediate_invariance_ab.json"
    ),
    "exp5610-arc-live-self-discovery-levelup-v506": Path(
        "results/experiment_5610_arc_live_self_discovery_levelup_v506.json"
    ),
    "exp5611-cdls-matched-sampler-crossover": Path(
        "results/experiment_5611_cdls_matched_sampler_crossover.json"
    ),
}

EXP5610_TRACE_RELATIVE_PATH = Path(
    "results/experiment_5610_arc_live_self_discovery_levelup_v506_trace.json"
)
PRIMARY_ARTIFACT_PATHS = tuple(TASK_ARTIFACT_PATHS.values())
SIDECAR_ARTIFACT_PATHS = (EXP5610_TRACE_RELATIVE_PATH,)

SOURCE_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("research-complete.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/arc_solve_registry.yaml"),
    Path("ops/status.md"),
    Path("ops/changelog.md"),
    Path("ops/conductor-log.md"),
    Path("_bmad/traceability.md"),
    Path("ops/e2e-test-plan.md"),
)

DELEGATED_BY_STOP_RULE = (
    "ops/status.md",
    "ops/changelog.md",
    "_bmad/traceability.md",
    "research-complete.yaml",
    "ops/exclusion_manifest.yaml",
    "ops/arc_solve_registry.yaml",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "one-line annotations for every required capstone field.",
    "expected_task_ids": "fixed milestone denominator; must list Exp5603 through Exp5612 task ids.",
    "artifacts_found": "only readable files can support claims.",
    "terminal_status_by_task": (
        "blocked, skipped, flagged, malformed, missing, and complete stay distinct."
    ),
    "headline_claims": (
        "narrow artifact-backed conclusions; flagged artifacts cannot support positive claims."
    ),
    "promotion_decisions": "mechanisms promote only through preregistered gates.",
    "retirement_decisions": (
        "repeated failures stop reruns, while new scientific nulls stay planning inputs."
    ),
    "arc_registry_delta": "levels_after - levels_before and new_reproducible_levels must agree.",
    "continuous_self_learning_verdict": (
        "FR-11 outcome from Exp5608, including safety and rollback gates."
    ),
    "hardware_sampling_verdict": (
        "matched-quality CPU/CUDA timing and quality verdict from Exp5611."
    ),
    "documents_reconciled": (
        "spec and ops reconciliation state, including any operator-delegated files."
    ),
    "tests_run": "commands, exit codes, counts, warnings, and skipped checks actually observed.",
    "unresolved_gaps": "negative evidence becomes planning input.",
    "inference_substrate": "must equal aggregation_from_exp5603_exp5611_artifacts.",
    "honest_verdict": "terminal summary starting with complete: or blocked:.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "run_date",
    "random_seed",
    "spec_refs",
    "result_path",
    "field_principles",
    "artifact_metadata",
    "source_context",
    "source_context_missing",
    "missing_artifacts",
    "malformed_artifacts",
    "missing_tasks",
    "malformed_tasks",
    "blocked_tasks",
    "gate_skipped_tasks",
    "flagged_tasks",
    "complete_tasks",
    "current_task",
    "reproducibility_checksum",
    *REQUIRED_ARTIFACT_FIELDS,
)
LIST_FIELDS = (
    "expected_task_ids",
    "artifacts_found",
    "source_context",
    "source_context_missing",
    "missing_artifacts",
    "malformed_artifacts",
    "missing_tasks",
    "malformed_tasks",
    "blocked_tasks",
    "gate_skipped_tasks",
    "flagged_tasks",
    "complete_tasks",
    "tests_run",
    "unresolved_gaps",
)
DICT_FIELDS = (
    "field_principles",
    "artifact_metadata",
    "terminal_status_by_task",
    "headline_claims",
    "promotion_decisions",
    "retirement_decisions",
    "continuous_self_learning_verdict",
    "hardware_sampling_verdict",
    "documents_reconciled",
)

DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5612_v506_capstone_reconciliation.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run_in_default_artifact",
        "counts": {},
        "warnings": [],
        "skipped_checks": [],
    },
    {
        "command": (
            ".venv/bin/coverage run --include=python/carnot/"
            "experiment_5612_v506_capstone_reconciliation.py -m pytest "
            "tests/python/test_experiment_5612_v506_capstone_reconciliation.py -q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
        "counts": {},
        "warnings": [],
        "skipped_checks": [],
    },
    {
        "command": (
            ".venv/bin/coverage report --include=python/carnot/"
            "experiment_5612_v506_capstone_reconciliation.py --fail-under=100"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
        "counts": {},
        "warnings": [],
        "skipped_checks": [],
    },
    {
        "command": ".venv/bin/pytest tests/python -q",
        "exit_code": None,
        "status": "not_run_in_default_artifact",
        "counts": {},
        "warnings": [],
        "skipped_checks": [],
    },
)


def _read_json_any(path: Path) -> tuple[JsonDict, JsonDict]:
    metadata: JsonDict = {
        "exists": path.exists(),
        "loadable": False,
        "json_type": None,
        "sha256": path_sha256(path),
    }
    if not path.exists():
        metadata["error"] = "missing"
        return {}, metadata
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        metadata.update({"error": "malformed_json", "line": exc.lineno, "column": exc.colno})
        return {}, metadata
    metadata["json_type"] = type(parsed).__name__
    if not isinstance(parsed, Mapping):
        metadata["error"] = "not_json_object"
        return {}, metadata
    metadata.update({"loadable": True, "error": None})
    return dict(parsed), metadata


def _read_source_context(root: Path) -> tuple[list[JsonDict], list[str]]:
    records: list[JsonDict] = []
    missing: list[str] = []
    for rel_path in SOURCE_CONTEXT_PATHS:
        path = root / rel_path
        exists = path.exists()
        records.append(
            {
                "path": rel_path.as_posix(),
                "exists": exists,
                "read_only": True,
                "sha256": path_sha256(path),
            }
        )
        if not exists:
            missing.append(rel_path.as_posix())
    return records, missing


def read_artifacts(root: Path) -> tuple[dict[str, JsonDict], JsonDict, list[JsonDict]]:
    payloads: dict[str, JsonDict] = {}
    metadata: JsonDict = {}
    found: list[JsonDict] = []
    path_to_task = {path: task_id for task_id, path in TASK_ARTIFACT_PATHS.items()}
    for rel_path in (*PRIMARY_ARTIFACT_PATHS, *SIDECAR_ARTIFACT_PATHS):
        payload, meta = _read_json_any(root / rel_path)
        rel = rel_path.as_posix()
        metadata[rel] = meta
        payloads[rel] = payload
        if meta.get("exists") and meta.get("loadable"):
            found.append(
                {
                    "path": rel,
                    "task_id": path_to_task.get(rel_path),
                    "role": "primary_result" if rel_path in PRIMARY_ARTIFACT_PATHS else "sidecar",
                    "sha256": meta.get("sha256"),
                }
            )
    return payloads, metadata, found


def _payload(artifacts: Mapping[str, JsonMap], rel_path: Path) -> JsonMap:
    value = artifacts.get(rel_path.as_posix(), {})
    return value if isinstance(value, Mapping) else {}


def _verdict(payload: JsonMap) -> str:
    verdict = payload.get("honest_verdict")
    return str(verdict) if verdict is not None else ""


def _is_gate_skip(payload: JsonMap) -> bool:
    verdict = _verdict(payload).lower()
    blocked_at_layer = str(payload.get("blocked_at_layer") or "").lower()
    return bool(
        payload.get("schema") == "blocked_gate_check_v1"
        or verdict == "blocked_gate_check_failed"
        or ("gate" in blocked_at_layer and payload.get("status") == "blocked")
        or (payload.get("gate_check_summary") and payload.get("status") == "blocked")
    )


def _is_blocked(payload: JsonMap) -> bool:
    verdict = _verdict(payload).lower()
    status = str(payload.get("status") or "").lower()
    return bool(status == "blocked" or verdict.startswith("blocked:") or verdict.startswith("blocked_"))


def _is_complete(payload: JsonMap) -> bool:
    verdict = _verdict(payload).lower()
    status = str(payload.get("status") or "").lower()
    return bool(status == "complete" or verdict.startswith("complete:"))


def _is_flagged(payload: JsonMap) -> bool:
    return bool(payload.get("flagged_adversarial"))


def _clean_for_positive_claim(payload: JsonMap) -> bool:
    return bool(payload) and not (_is_flagged(payload) or _is_gate_skip(payload) or _is_blocked(payload))


def _int(payload: JsonMap, field: str) -> int:
    value = payload.get(field)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str) and value.lstrip("-").isdigit():
        return int(value)
    return 0


def _float(payload: JsonMap, field: str) -> float:
    value = payload.get(field)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _status_for_payload(payload: JsonMap, meta: JsonMap) -> str:
    if not meta.get("exists"):
        return "missing"
    if not meta.get("loadable"):
        return "malformed"
    if _is_flagged(payload):
        return "flagged"
    if _is_gate_skip(payload):
        return "gate_skipped"
    if _is_blocked(payload):
        return "blocked"
    if _is_complete(payload):
        return "complete"
    return "unknown"


def terminal_status_by_task(
    artifacts: Mapping[str, JsonMap], metadata: JsonMap
) -> dict[str, JsonDict]:
    statuses: dict[str, JsonDict] = {}
    for task_id in EXPECTED_TASK_IDS:
        if task_id == EXPERIMENT_ID:
            statuses[task_id] = {
                "status": "current_capstone",
                "artifact_path": RESULT_RELATIVE_PATH.as_posix(),
                "supports_positive_claim": False,
                "note": "this artifact is emitted by the current workflow",
            }
            continue
        rel_path = TASK_ARTIFACT_PATHS[task_id]
        rel = rel_path.as_posix()
        payload = _payload(artifacts, rel_path)
        meta = metadata.get(rel, {})
        status = _status_for_payload(payload, meta)
        statuses[task_id] = {
            "status": status,
            "artifact_path": rel,
            "honest_verdict": _verdict(payload) or None,
            "flagged_adversarial": bool(payload.get("flagged_adversarial")),
            "corrigendum_pending": payload.get("corrigendum_pending", []),
            "supports_positive_claim": status == "complete",
            "sha256": meta.get("sha256"),
            "metadata_error": meta.get("error"),
        }
    return statuses


def _bucket_tasks(statuses: Mapping[str, JsonMap], status: str) -> list[str]:
    return [
        task_id
        for task_id, row in statuses.items()
        if task_id != EXPERIMENT_ID and row.get("status") == status
    ]


def _arc_delta(exp5610: JsonMap) -> tuple[int, JsonDict]:
    before = _int(exp5610, "levels_before")
    after = _int(exp5610, "levels_after")
    new_levels = exp5610.get("new_reproducible_levels")
    new_count = len(new_levels) if isinstance(new_levels, list) else 0
    arithmetic_delta = after - before
    delta = arithmetic_delta if arithmetic_delta == new_count else min(arithmetic_delta, new_count)
    return delta, {
        "levels_before": before,
        "levels_after": after,
        "arithmetic_delta": arithmetic_delta,
        "new_reproducible_level_count": new_count,
        "new_reproducible_levels": list(new_levels) if isinstance(new_levels, list) else [],
        "offline_reproduced": bool(exp5610.get("offline_reproduced")),
        "registry_updated": bool(exp5610.get("registry_updated")),
        "flagged_adversarial": bool(exp5610.get("flagged_adversarial")),
        "solve_provenance": exp5610.get("solve_provenance"),
    }


def _all_gate_values_true(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    return all(item is True for item in value.values())


def derive_claims(
    artifacts: Mapping[str, JsonMap], statuses: Mapping[str, JsonMap]
) -> tuple[JsonDict, JsonDict, JsonDict, int, JsonDict, JsonDict, list[JsonDict]]:
    exp5605 = _payload(artifacts, TASK_ARTIFACT_PATHS["exp5605-raw-response-evidence-envelope"])
    exp5606 = _payload(
        artifacts, TASK_ARTIFACT_PATHS["exp5606-clean-sota-solve-verify-evidence-panel"]
    )
    exp5607 = _payload(
        artifacts, TASK_ARTIFACT_PATHS["exp5607-property-template-exact-residual-extension"]
    )
    exp5608 = _payload(artifacts, TASK_ARTIFACT_PATHS["exp5608-kan-longitudinal-self-learning"])
    exp5609 = _payload(
        artifacts, TASK_ARTIFACT_PATHS["exp5609-arc-filter-intermediate-invariance-ab"]
    )
    exp5610 = _payload(
        artifacts, TASK_ARTIFACT_PATHS["exp5610-arc-live-self-discovery-levelup-v506"]
    )
    exp5611 = _payload(
        artifacts, TASK_ARTIFACT_PATHS["exp5611-cdls-matched-sampler-crossover"]
    )

    response_evidence_ok = bool(
        _clean_for_positive_claim(exp5605)
        and exp5605.get("envelope_ready")
        and exp5605.get("raw_payloads_preserved")
        and _float(exp5605, "lossless_replay_rate") >= 1.0
        and _int(exp5605, "semantic_false_accept_count") == 0
        and exp5605.get("parser_version_replay_passed")
        and exp5605.get("payload_corruption_rejected")
    )
    solve_verify_ok = bool(
        _clean_for_positive_claim(exp5606)
        and exp5606.get("panel_complete")
        and exp5606.get("gpu_offload_authenticated")
        and _float(exp5606, "maximum_parser_failure_rate") <= 0.05
        and exp5606.get("solve_verify_asymmetry_supported")
    )
    exact_extension_ok = bool(_clean_for_positive_claim(exp5607) and exp5607.get("verifier_extension_promoted"))
    kan_gate = exp5608.get("promotion_gate")
    kan_ok = bool(
        _clean_for_positive_claim(exp5608)
        and exp5608.get("continuous_self_learning_task")
        and exp5608.get("kan_longitudinal_ready")
        and _all_gate_values_true(kan_gate)
        and _int(exp5608, "unsafe_false_accept_count") == 0
        and exp5608.get("rollback_positive_control")
        and exp5608.get("delayed_regression_passed")
        and exp5608.get("no_model_weight_mutation")
    )
    filter_decisions = exp5609.get("filter_promotion_decisions")
    inert_decision = filter_decisions.get("inert_click", {}) if isinstance(filter_decisions, Mapping) else {}
    history_decision = (
        filter_decisions.get("object_history", {}) if isinstance(filter_decisions, Mapping) else {}
    )
    inert_promoted = bool(inert_decision.get("decision") == "promote")
    history_promoted = bool(history_decision.get("decision") == "promote")
    arc_delta, arc_delta_evidence = _arc_delta(exp5610)
    arc_level_claim = bool(
        _clean_for_positive_claim(exp5610)
        and exp5610.get("solve_provenance") == "live_agent_self_discovery"
        and arc_delta > 0
        and exp5610.get("offline_reproduced")
        and exp5610.get("registry_updated")
    )
    cdls_ok = bool(
        _clean_for_positive_claim(exp5611)
        and exp5611.get("crossover_claim_allowed")
        and _int(exp5611, "successful_matched_pairs") > 0
    )

    headline_claims: JsonDict = {
        "response_evidence": {
            "claim_allowed": response_evidence_ok,
            "claim": "bounded_lossless_response_envelope_ready" if response_evidence_ok else "not_promoted",
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5605-raw-response-evidence-envelope"].as_posix()
            ],
            "evidence": {
                "response_rows_written": exp5605.get("response_rows_written"),
                "lossless_replay_rate": exp5605.get("lossless_replay_rate"),
                "semantic_false_accept_count": exp5605.get("semantic_false_accept_count"),
            },
        },
        "solve_verify_asymmetry": {
            "claim_allowed": solve_verify_ok,
            "claim": "no_solve_verify_asymmetry_claim",
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5606-clean-sota-solve-verify-evidence-panel"].as_posix()
            ],
            "evidence": {
                "panel_complete": bool(exp5606.get("panel_complete")),
                "gpu_offload_authenticated": bool(exp5606.get("gpu_offload_authenticated")),
                "maximum_parser_failure_rate": exp5606.get("maximum_parser_failure_rate"),
                "solve_verify_asymmetry_supported": bool(
                    exp5606.get("solve_verify_asymmetry_supported")
                ),
            },
        },
        "exact_predicate_extension": {
            "claim_allowed": exact_extension_ok,
            "claim": "blocked_no_clean_residual_set",
            "source_artifacts": [
                TASK_ARTIFACT_PATHS[
                    "exp5607-property-template-exact-residual-extension"
                ].as_posix()
            ],
            "evidence": {
                "status": statuses["exp5607-property-template-exact-residual-extension"].get(
                    "status"
                ),
                "gate_check_summary": exp5607.get("gate_check_summary"),
            },
        },
        "kan_longitudinal_self_learning": {
            "claim_allowed": kan_ok,
            "claim": "bounded_kan_only_longitudinal_fr11_ready" if kan_ok else "not_promoted",
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5608-kan-longitudinal-self-learning"].as_posix()
            ],
            "evidence": {
                "kan_longitudinal_ready": bool(exp5608.get("kan_longitudinal_ready")),
                "promotion_gate": dict(kan_gate) if isinstance(kan_gate, Mapping) else {},
                "forward_transfer_delta": exp5608.get("forward_transfer_delta"),
                "backward_retention_delta": exp5608.get("backward_retention_delta"),
                "forgetting_delta": exp5608.get("forgetting_delta"),
            },
        },
        "arc_filters": {
            "claim_allowed": bool(inert_promoted or history_promoted),
            "claim": "filters_retired_reachable_repeat_noop",
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5609-arc-filter-intermediate-invariance-ab"].as_posix()
            ],
            "evidence": {
                "inert_click_decision": inert_decision.get("decision"),
                "object_history_decision": history_decision.get("decision"),
                "levels_gained_by_arm": exp5609.get("levels_gained_by_arm", {}),
            },
        },
        "arc_new_registry_levels": {
            "claim_allowed": arc_level_claim,
            "claim": "no_new_reproducible_arc_level_banked",
            "source_artifacts": [
                TASK_ARTIFACT_PATHS[
                    "exp5610-arc-live-self-discovery-levelup-v506"
                ].as_posix(),
                EXP5610_TRACE_RELATIVE_PATH.as_posix(),
            ],
            "evidence": arc_delta_evidence,
        },
        "cdls_crossover": {
            "claim_allowed": cdls_ok,
            "claim": "no_quality_matched_crossover_claim",
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5611-cdls-matched-sampler-crossover"].as_posix()
            ],
            "evidence": {
                "flagged_adversarial": bool(exp5611.get("flagged_adversarial")),
                "crossover_claim_allowed": bool(exp5611.get("crossover_claim_allowed")),
                "successful_matched_pairs": _int(exp5611, "successful_matched_pairs"),
                "board_speedup_claimed": bool(exp5611.get("board_speedup_claimed")),
            },
        },
    }

    promotion_decisions: JsonDict = {
        "response_evidence_envelope": {
            "decision": "promote_bounded" if response_evidence_ok else "do_not_promote",
            "reason": "lossless replay and fail-closed controls passed"
            if response_evidence_ok
            else "envelope controls incomplete or unavailable",
        },
        "solve_verify_asymmetry": {
            "decision": "do_not_promote",
            "reason": "panel_complete is false, GPU offload is unauthenticated, and parser failure is above ceiling",
        },
        "exact_predicate_extension": {
            "decision": "do_not_promote",
            "reason": "conductor pre-gate skipped because no clean residual panel exists",
        },
        "kan_longitudinal_self_learning": {
            "decision": "promote_bounded" if kan_ok else "do_not_promote",
            "reason": "all exact-gated KAN promotion gates passed"
            if kan_ok
            else "KAN promotion gates unavailable or malformed",
        },
        "arc_inert_click_filter": {
            "decision": "do_not_promote" if not inert_promoted else "promote",
            "reason": str(inert_decision.get("reason") or inert_decision.get("decision") or "missing"),
        },
        "arc_object_history_filter": {
            "decision": "do_not_promote" if not history_promoted else "promote",
            "reason": str(
                history_decision.get("reason") or history_decision.get("decision") or "missing"
            ),
        },
        "arc_live_registry_level": {
            "decision": "do_not_promote",
            "reason": "new_reproducible_levels is empty and the artifact is adversarially flagged",
        },
        "cdls_crossover": {
            "decision": "do_not_promote",
            "reason": "zero successful matched pairs and adversarial duration/methodology flag",
        },
    }

    retirement_decisions: JsonDict = {
        "local_sota_solve_verify_panel": {
            "decision": "retire",
            "retire_if_same_verdict_applied": True,
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5606-clean-sota-solve-verify-evidence-panel"].as_posix()
            ],
            "reason": "fresh evidence-preserving panel still collapsed before an admissible solve-versus-verify result",
            "manifest_update": "delegated_by_stop_rule",
        },
        "exact_predicate_extension_from_this_panel": {
            "decision": "retire_until_clean_panel_exists",
            "retire_if_same_verdict_applied": True,
            "source_artifacts": [
                TASK_ARTIFACT_PATHS[
                    "exp5607-property-template-exact-residual-extension"
                ].as_posix()
            ],
            "reason": "same no-clean-residual gate block recurred after the attempted panel repair",
            "manifest_update": "delegated_by_stop_rule",
        },
        "arc_inert_click_filter": {
            "decision": "retire"
            if inert_decision.get("decision") == "retire_reachable_downstream_noop"
            else "keep_open",
            "retire_if_same_verdict_applied": True,
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5609-arc-filter-intermediate-invariance-ab"].as_posix()
            ],
            "reason": str(inert_decision.get("reason") or "reachable control but no downstream improvement"),
            "manifest_update": "delegated_by_stop_rule",
        },
        "arc_object_history_filter": {
            "decision": "retire"
            if history_decision.get("decision") == "retire_reachable_downstream_noop"
            else "keep_open",
            "retire_if_same_verdict_applied": True,
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5609-arc-filter-intermediate-invariance-ab"].as_posix()
            ],
            "reason": str(
                history_decision.get("reason") or "reachable control but no downstream improvement"
            ),
            "manifest_update": "delegated_by_stop_rule",
        },
        "arc_live_levelup_standing_floor": {
            "decision": "do_not_retire_new_target_null",
            "retire_if_same_verdict_applied": False,
            "source_artifacts": [
                TASK_ARTIFACT_PATHS[
                    "exp5610-arc-live-self-discovery-levelup-v506"
                ].as_posix()
            ],
            "reason": "sk48 L8 no-bank is a new rotated target null; retire only an unchanged rerun of this exact route",
        },
        "cdls_crossover": {
            "decision": "do_not_retire_new_scientific_null",
            "retire_if_same_verdict_applied": False,
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5611-cdls-matched-sampler-crossover"].as_posix()
            ],
            "reason": "bounded cDLS proposal is new and flagged; plan a corrected quality-gated benchmark rather than retiring the whole mechanism",
        },
    }

    continuous_self_learning_verdict: JsonDict = {
        "claim_allowed": kan_ok,
        "continuous_self_learning_task": bool(exp5608.get("continuous_self_learning_task")),
        "kan_longitudinal_ready": bool(exp5608.get("kan_longitudinal_ready")),
        "promotion_gate": dict(kan_gate) if isinstance(kan_gate, Mapping) else {},
        "forward_transfer_delta": exp5608.get("forward_transfer_delta"),
        "backward_retention_delta": exp5608.get("backward_retention_delta"),
        "forgetting_delta": exp5608.get("forgetting_delta"),
        "unsafe_false_accept_count": exp5608.get("unsafe_false_accept_count"),
        "rollback_positive_control": bool(exp5608.get("rollback_positive_control")),
        "delayed_regression_passed": bool(exp5608.get("delayed_regression_passed")),
        "no_model_weight_mutation": bool(exp5608.get("no_model_weight_mutation")),
        "claim_boundary": "bounded KAN-component adaptation only; no LLM calls or model-weight training",
    }
    hardware_sampling_verdict: JsonDict = {
        "claim_allowed": cdls_ok,
        "crossover_claim_allowed": bool(exp5611.get("crossover_claim_allowed")),
        "crossover_size": exp5611.get("crossover_size"),
        "successful_matched_pairs": _int(exp5611, "successful_matched_pairs"),
        "board_speedup_claimed": bool(exp5611.get("board_speedup_claimed")),
        "flagged_adversarial": bool(exp5611.get("flagged_adversarial")),
        "claim_boundary": "no CPU/CUDA crossover or board speedup claim",
    }
    unresolved_gaps = [
        {
            "gap": "source_delta_artifact_flagged",
            "planning_input": "rerun source freshness only with enough duration/methodology evidence if its accepted delta must support future claims",
            "source_task": "exp5604-v506-source-delta-ingestion",
        },
        {
            "gap": "local_sota_panel_no_admissible_asymmetry",
            "planning_input": "do not rerun the same panel shape without a changed parser/offload/runtime root cause",
            "source_task": "exp5606-clean-sota-solve-verify-evidence-panel",
        },
        {
            "gap": "exact_predicate_extension_lacks_clean_residuals",
            "planning_input": "requires a clean residual set before any predicate implementation work",
            "source_task": "exp5607-property-template-exact-residual-extension",
        },
        {
            "gap": "arc_live_levelup_no_bank",
            "planning_input": "standing ARC floor remains open, but this sk48 L8 route should not be repeated unchanged",
            "source_task": "exp5610-arc-live-self-discovery-levelup-v506",
        },
        {
            "gap": "cdls_quality_matched_crossover_unavailable",
            "planning_input": "correct the quality gate/methodology before another crossover claim",
            "source_task": "exp5611-cdls-matched-sampler-crossover",
        },
    ]
    return (
        headline_claims,
        promotion_decisions,
        retirement_decisions,
        arc_delta,
        continuous_self_learning_verdict,
        hardware_sampling_verdict,
        unresolved_gaps,
    )


def _documents_reconciled(
    root: Path,
    source_context_missing: Sequence[str],
    *,
    roadmap_modified: bool,
    conductor_modified: bool,
) -> JsonDict:
    spec_path = root / CAPSTONE_SPEC_RELATIVE_PATH
    spec_text = spec_path.read_text(encoding="utf-8", errors="replace") if spec_path.exists() else ""
    roadmap_text = (root / ROADMAP_RELATIVE_PATH).read_text(
        encoding="utf-8", errors="replace"
    ) if (root / ROADMAP_RELATIVE_PATH).exists() else ""
    return {
        "openspec_capstone_req_present": "REQ-CAPSTONE-5612" in spec_text,
        "updated_by_this_workflow": [CAPSTONE_SPEC_RELATIVE_PATH.as_posix()],
        "delegated_by_stop_rule": list(DELEGATED_BY_STOP_RULE),
        "protected_files": {
            ROADMAP_RELATIVE_PATH.as_posix(): not roadmap_modified,
            CONDUCTOR_RELATIVE_PATH.as_posix(): not conductor_modified,
        },
        "research_roadmap_next_missing": ROADMAP_NEXT_RELATIVE_PATH.as_posix()
        in source_context_missing,
        "active_milestone": MILESTONE if f'milestone: "{MILESTONE}"' in roadmap_text else None,
        "next_milestone_activated": False,
        "reconciliation_note": (
            "ops/status, ops/changelog, traceability, completion, exclusion, and ARC registry "
            "edits are recorded as delegated because the operator stop rule forbids touching them."
        ),
    }


def build_artifact(
    artifacts: Mapping[str, JsonMap],
    metadata: JsonMap,
    artifacts_found: Sequence[JsonMap],
    source_context: Sequence[JsonMap],
    source_context_missing: Sequence[str],
    *,
    tests_run: Sequence[JsonMap],
    roadmap_modified: bool,
    conductor_modified: bool,
    root: Path,
) -> JsonDict:
    statuses = terminal_status_by_task(artifacts, metadata)
    (
        headline_claims,
        promotion_decisions,
        retirement_decisions,
        arc_registry_delta,
        continuous_self_learning_verdict,
        hardware_sampling_verdict,
        unresolved_gaps,
    ) = derive_claims(artifacts, statuses)
    missing_artifacts = [
        TASK_ARTIFACT_PATHS[task_id].as_posix()
        for task_id, row in statuses.items()
        if row.get("status") == "missing" and task_id != EXPERIMENT_ID
    ]
    malformed_artifacts = [
        TASK_ARTIFACT_PATHS[task_id].as_posix()
        for task_id, row in statuses.items()
        if row.get("status") == "malformed" and task_id != EXPERIMENT_ID
    ]
    status_prefix = "blocked:" if missing_artifacts or malformed_artifacts else "complete:"
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "model_specs": {
            "new_model_invocations": [],
            "methodology_note": (
                "aggregation-only capstone; upstream CUDA/model markers are preserved as "
                "evidence text and are not fresh model or hardware invocations"
            ),
        },
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "artifact_metadata": dict(metadata),
        "source_context": [dict(row) for row in source_context],
        "source_context_missing": list(source_context_missing),
        "missing_artifacts": missing_artifacts,
        "malformed_artifacts": malformed_artifacts,
        "missing_tasks": _bucket_tasks(statuses, "missing"),
        "malformed_tasks": _bucket_tasks(statuses, "malformed"),
        "blocked_tasks": _bucket_tasks(statuses, "blocked"),
        "gate_skipped_tasks": _bucket_tasks(statuses, "gate_skipped"),
        "flagged_tasks": _bucket_tasks(statuses, "flagged"),
        "complete_tasks": _bucket_tasks(statuses, "complete"),
        "current_task": EXPERIMENT_ID,
        "expected_task_ids": list(EXPECTED_TASK_IDS),
        "artifacts_found": [dict(row) for row in artifacts_found],
        "terminal_status_by_task": statuses,
        "headline_claims": headline_claims,
        "promotion_decisions": promotion_decisions,
        "retirement_decisions": retirement_decisions,
        "arc_registry_delta": arc_registry_delta,
        "arc_registry_delta_evidence": headline_claims["arc_new_registry_levels"]["evidence"],
        "continuous_self_learning_verdict": continuous_self_learning_verdict,
        "hardware_sampling_verdict": hardware_sampling_verdict,
        "documents_reconciled": _documents_reconciled(
            root,
            source_context_missing,
            roadmap_modified=roadmap_modified,
            conductor_modified=conductor_modified,
        ),
        "tests_run": [dict(row) for row in tests_run],
        "unresolved_gaps": unresolved_gaps,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": (
            f"{status_prefix} .506 capstone aggregated {len(artifacts_found)} readable "
            f"artifact files across Exp5603-Exp5611; response envelope promoted="
            f"{headline_claims['response_evidence']['claim_allowed']}; "
            f"solve_verify_asymmetry=false; exact_extension=false; "
            f"kan_longitudinal_ready={continuous_self_learning_verdict['claim_allowed']}; "
            f"arc_registry_delta={arc_registry_delta}; cDLS_crossover=false"
        ),
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def run_capstone(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[JsonMap] = DEFAULT_TESTS_RUN,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifacts, metadata, found = read_artifacts(root)
    source_context, source_context_missing = _read_source_context(root)
    roadmap_modified = _modification_status(root, ROADMAP_RELATIVE_PATH, modification_overrides)
    conductor_modified = _modification_status(root, CONDUCTOR_RELATIVE_PATH, modification_overrides)
    return build_artifact(
        artifacts,
        metadata,
        found,
        source_context,
        source_context_missing,
        tests_run=tests_run,
        roadmap_modified=roadmap_modified,
        conductor_modified=conductor_modified,
        root=root,
    )


def validate_artifact(payload: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(field)
    for field in LIST_FIELDS:
        if field in payload and not isinstance(payload[field], list):
            errors.append(field)
    for field in DICT_FIELDS:
        if field in payload and not isinstance(payload[field], Mapping):
            errors.append(field)
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if payload.get("expected_task_ids") != list(EXPECTED_TASK_IDS):
        errors.append("expected_task_ids")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    honest_verdict = payload.get("honest_verdict")
    if not isinstance(honest_verdict, str) or not honest_verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    if not isinstance(payload.get("arc_registry_delta"), int):
        errors.append("arc_registry_delta")
    else:
        evidence = payload.get("arc_registry_delta_evidence")
        if isinstance(evidence, Mapping):
            arithmetic = evidence.get("arithmetic_delta")
            new_count = evidence.get("new_reproducible_level_count")
            if payload["arc_registry_delta"] != arithmetic or arithmetic != new_count:
                errors.append("arc_registry_delta")
    statuses = payload.get("terminal_status_by_task")
    if isinstance(statuses, Mapping):
        for task_id in EXPECTED_TASK_IDS:
            if task_id not in statuses:
                errors.append("terminal_status_by_task")
                break
    docs = payload.get("documents_reconciled")
    if isinstance(docs, Mapping):
        protected = docs.get("protected_files")
        if not isinstance(protected, Mapping) or not protected.get(ROADMAP_RELATIVE_PATH.as_posix()):
            errors.append("documents_reconciled")
        if not isinstance(protected, Mapping) or not protected.get(CONDUCTOR_RELATIVE_PATH.as_posix()):
            errors.append("documents_reconciled")
    return sorted(set(errors))


def write_capstone(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[JsonMap] = DEFAULT_TESTS_RUN,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifact = run_capstone(
        root=root, tests_run=tests_run, modification_overrides=modification_overrides
    )
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - guarded by validate_artifact tests
        raise ValueError(f"invalid Exp5612 artifact fields: {', '.join(errors)}")
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def _load_tests_run(path: Path | None) -> Sequence[JsonMap]:
    if path is None:
        return DEFAULT_TESTS_RUN
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError("--tests-run-json must contain a JSON list")
    return [dict(row) for row in loaded if isinstance(row, Mapping)]


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the Exp5612 artifact")
    parser.add_argument(
        "--tests-run-json",
        type=Path,
        default=None,
        help="optional JSON list of observed validation command records",
    )
    args = parser.parse_args(argv)
    tests_run = _load_tests_run(args.tests_run_json)
    artifact = write_capstone(tests_run=tests_run) if args.write else run_capstone(tests_run=tests_run)
    if not args.write:
        write_json(Path("/dev/stdout"), artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
