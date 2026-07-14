"""Exp5636 transition receipt from milestone .508 into .509.

Spec refs: REQ-REPORT-5636, SCENARIO-REPORT-5636,
SCENARIO-REPORT-5636-DEPENDENCY-MAP,
SCENARIO-REPORT-5636-FIELD-PRINCIPLES.

This module is a record-only evidence lock. It reads the terminal `.508`
artifacts and records which facts may safely seed `.509`. The main risk is
evidence laundering: an internally clean learner is not the same thing as an
independently promoted learner, a gate-schema failure is not a scientific
negative, and a flagged live ARC attempt must stay visible instead of being
rounded into progress.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
from typing import Any

from carnot.experiment_5415_transition_v493 import (
    JsonDict,
    JsonMap,
    _modification_status,
    extract_roadmap_tasks,
    path_sha256,
    payload_checksum,
    read_yaml_mapping,
    write_json,
)
from carnot.experiment_5625_transition_v508 import (
    _float,
    _int,
    _nested_total,
    _payload,
    _read_json_any,
    _status_for_payload,
    _task_range_from_text,
    _verdict,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5636_transition_v509.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")

EXPERIMENT = "experiment_5636_transition_v509"
EXPERIMENT_ID = "exp5636-transition-v509"
PREVIOUS_MILESTONE = "2026.07.508"
CURRENT_MILESTONE = "2026.07.509"
PREVIOUS_TASK_RANGE = "exp5625-exp5635"
CURRENT_TASK_RANGE = "exp5636-exp5647"
RUN_DATE = "2026-07-14"
RANDOM_SEED = 5636
SCHEMA = "carnot.experiment_5636.transition_v509.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

SPEC_REFS = (
    "REQ-REPORT-5636",
    "SCENARIO-REPORT-5636",
    "SCENARIO-REPORT-5636-DEPENDENCY-MAP",
    "SCENARIO-REPORT-5636-FIELD-PRINCIPLES",
)

EXP5625_TRANSITION_PATH = Path("results/experiment_5625_transition_v508.json")
EXP5626_SOURCE_PATH = Path("results/experiment_5626_v508_source_delta_ingestion.json")
EXP5627_QUALIFICATION_PATH = Path("results/experiment_5627_online_conformal_kan_qualification.json")
EXP5628_KAN_CSL_PATH = Path("results/experiment_5628_conformal_active_spline_kan_csl.json")
EXP5629_KAN_AUDIT_PATH = Path("results/experiment_5629_conformal_kan_independent_audit.json")
EXP5630_ARC_PROBE_PATH = Path("results/experiment_5630_arc_epistemic_object_probe_prototype.json")
EXP5631_ARC_AB_PATH = Path("results/experiment_5631_arc_epistemic_probe_live_ab.json")
EXP5632_ARC_LEVEL_PATH = Path("results/experiment_5632_arc_live_self_discovery_levelup_v508.json")
EXP5633_TEMPERATURE_EXACT_PATH = Path(
    "results/experiment_5633_temperature_exchange_cdls_exact_audit.json"
)
EXP5634_TEMPERATURE_QUALITY_PATH = Path(
    "results/experiment_5634_temperature_exchange_cdls_quality.json"
)
EXP5635_CAPSTONE_PATH = Path("results/experiment_5635_v508_capstone_reconciliation.json")

TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5625-transition-v508": EXP5625_TRANSITION_PATH,
    "exp5626-v508-source-delta-ingestion": EXP5626_SOURCE_PATH,
    "exp5627-online-conformal-kan-qualification": EXP5627_QUALIFICATION_PATH,
    "exp5628-conformal-active-spline-kan-csl": EXP5628_KAN_CSL_PATH,
    "exp5629-conformal-kan-independent-audit": EXP5629_KAN_AUDIT_PATH,
    "exp5630-arc-epistemic-object-probe-prototype": EXP5630_ARC_PROBE_PATH,
    "exp5631-arc-epistemic-probe-live-ab": EXP5631_ARC_AB_PATH,
    "exp5632-arc-live-self-discovery-levelup-v508": EXP5632_ARC_LEVEL_PATH,
    "exp5633-temperature-exchange-cdls-exact-audit": EXP5633_TEMPERATURE_EXACT_PATH,
    "exp5634-temperature-exchange-cdls-quality": EXP5634_TEMPERATURE_QUALITY_PATH,
    "exp5635-v508-capstone-reconciliation": EXP5635_CAPSTONE_PATH,
}
UPSTREAM_ARTIFACT_PATHS = tuple(TASK_ARTIFACT_PATHS.values())

SOURCE_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
)

EXPECTED_TASK_IDS = [
    "exp5636-transition-v509",
    "exp5637-v509-source-delta-ingestion",
    "exp5638-fr11-gate-schema-corrigendum",
    "exp5639-anytime-valid-csl-independent-audit",
    "exp5640-fr11-shadow-pipeline-integration",
    "exp5641-arc-counterexample-executable-model",
    "exp5642-arc-executable-model-live-ab",
    "exp5643-arc-live-self-discovery-levelup-v509",
    "exp5644-two-axis-parallel-tempering-exact-audit",
    "exp5645-two-axis-tempering-hard-constraint-quality",
    "exp5646-two-axis-tempering-rust-parity",
    "exp5647-v509-capstone-reconciliation",
]

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "One-line annotations for every required headline and gate field.",
    "artifacts_read": "claims trace to files",
    "terminal_findings": "only observed outcomes cross milestones",
    "promoted_substrates": "clean prerequisites are explicit",
    "promising_unpromoted_substrates": "Exp5628 is not laundered",
    "retired_scopes": "Exp5630 and prior closures remain closed",
    "adversarial_flags_preserved": "Exp5632 flags remain visible",
    "current_task_range": "IDs do not collide",
    "dependency_map": "gates are auditable",
    "inference_substrate": "no new inference occurred",
    "reproducibility_checksum": "transition content is stable",
    "honest_verdict": "terminal status starts complete: or blocked:",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "status",
    "run_date",
    "random_seed",
    "spec_refs",
    "result_path",
    "previous_milestone",
    "current_milestone",
    "previous_task_range",
    "current_task_range",
    "current_task_collision_check",
    "artifact_metadata",
    "source_context",
    "source_context_missing",
    "missing_artifacts",
    "malformed_artifacts",
    "roadmap_task_ids",
    "roadmap_task_count",
    "roadmap_doc_task_range",
    "protected_file_checks",
    "preconditions_checked",
    "failed_preconditions",
    "tests_run",
    "roadmap_yaml_unchanged",
    "conductor_unchanged",
    *REQUIRED_ARTIFACT_FIELDS,
)
LIST_FIELDS = (
    "artifacts_read",
    "promoted_substrates",
    "promising_unpromoted_substrates",
    "retired_scopes",
    "adversarial_flags_preserved",
    "source_context",
    "source_context_missing",
    "missing_artifacts",
    "malformed_artifacts",
    "roadmap_task_ids",
    "protected_file_checks",
    "failed_preconditions",
    "tests_run",
)
DICT_FIELDS = (
    "field_principles",
    "terminal_findings",
    "current_task_collision_check",
    "artifact_metadata",
    "dependency_map",
    "preconditions_checked",
)
BOOL_FIELDS = ("roadmap_yaml_unchanged", "conductor_unchanged")

DEFAULT_TESTS_RUN = (
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5636_transition_v509.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage run --include=python/carnot/"
            "experiment_5636_transition_v509.py -m pytest "
            "tests/python/test_experiment_5636_transition_v509.py -q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
    {
        "command": (
            ".venv/bin/coverage report --include=python/carnot/"
            "experiment_5636_transition_v509.py --fail-under=100"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
    },
)


def _read_text(root: Path, rel_path: Path) -> str:
    path = root / rel_path
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


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
    for rel_path in UPSTREAM_ARTIFACT_PATHS:
        payload, meta = _read_json_any(root / rel_path)
        rel = rel_path.as_posix()
        metadata[rel] = meta
        payloads[rel] = payload
        if meta.get("exists") and meta.get("loadable"):
            found.append(
                {
                    "path": rel,
                    "task_id": path_to_task[rel_path],
                    "sha256": meta.get("sha256"),
                    "status": _status_for_payload(payload, meta),
                    "honest_verdict": _verdict(payload) or None,
                }
            )
    return payloads, metadata, found


def _extract_exp_number(value: str) -> int | None:
    match = re.search(r"(?:exp|experiment_)(\d+)", value)
    return int(match.group(1)) if match else None


def _completed_task_ids(root: Path) -> list[str]:
    text = _read_text(root, RESEARCH_COMPLETE_RELATIVE_PATH)
    return sorted(set(re.findall(r"\bid:\s*(exp\d+(?:[-_][A-Za-z0-9_]+)*)", text)))


def _retired_task_ids(root: Path) -> list[str]:
    text = _read_text(root, CONDUCTOR_LOG_RELATIVE_PATH)
    rows = [line for line in text.splitlines() if "retired" in line.lower()]
    return sorted(set(re.findall(r"\bexp\d+(?:[-_][A-Za-z0-9_]+)*", "\n".join(rows))))


def _collision_check(root: Path, roadmap_task_ids: Sequence[str]) -> JsonDict:
    completed = _completed_task_ids(root)
    retired = _retired_task_ids(root)
    current_numbers = set(range(5636, 5648))
    completed_colliding = [
        task_id
        for task_id in completed
        if (number := _extract_exp_number(task_id)) in current_numbers
    ]
    retired_colliding = [
        task_id
        for task_id in retired
        if (number := _extract_exp_number(task_id)) in current_numbers
    ]
    highest = max((_extract_exp_number(task_id) or 0 for task_id in completed), default=0)
    colliding = sorted(set(completed_colliding + retired_colliding))
    return {
        "current_task_range": CURRENT_TASK_RANGE,
        "expected_task_ids": list(EXPECTED_TASK_IDS),
        "roadmap_task_ids": list(roadmap_task_ids),
        "completed_ids_checked_count": len(completed),
        "retired_ids_checked_count": len(retired),
        "highest_completed_id": f"exp{highest}" if highest else None,
        "completed_colliding_ids": completed_colliding,
        "retired_colliding_ids": retired_colliding,
        "colliding_ids": colliding,
        "collision_free": not colliding,
        "range_matches_roadmap": list(roadmap_task_ids) == list(EXPECTED_TASK_IDS),
    }


def terminal_findings(artifacts: Mapping[str, JsonMap], metadata: JsonMap) -> dict[str, JsonDict]:
    capstone = _payload(artifacts, EXP5635_CAPSTONE_PATH)
    complete = set(capstone.get("complete_tasks") or [])
    blocked = set(capstone.get("blocked_tasks") or [])
    gate_skipped = set(capstone.get("gate_skipped_tasks") or [])
    flagged = set(capstone.get("flagged_tasks") or [])
    promotion_ledger = capstone.get("promotion_ledger")
    promoted = {
        "exp5633-temperature-exchange-cdls-exact-audit"
        for key in ("replica_exchange_exact", "replica_exchange_quality")
        if isinstance(promotion_ledger, Mapping)
        and isinstance(promotion_ledger.get(key), Mapping)
        and promotion_ledger[key].get("promoted") is True
    }
    if "exp5634-temperature-exchange-cdls-quality" in complete:
        promoted.add("exp5634-temperature-exchange-cdls-quality")

    findings: dict[str, JsonDict] = {}
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        rel = rel_path.as_posix()
        payload = _payload(artifacts, rel_path)
        meta = metadata.get(rel, {})
        status = _status_for_payload(payload, meta)
        findings[task_id] = {
            "artifact_path": rel,
            "status": status,
            "capstone_classification": {
                "complete": task_id in complete,
                "blocked": task_id in blocked,
                "gate_skipped": task_id in gate_skipped,
                "flagged": task_id in flagged,
                "promoted": task_id in promoted,
            },
            "honest_verdict": _verdict(payload) or None,
            "flagged_adversarial": bool(payload.get("flagged_adversarial")),
            "corrigendum_pending": payload.get("corrigendum_pending", []),
            "sha256": meta.get("sha256"),
            "metadata_error": meta.get("error"),
            "supports_positive_claim": status == "complete" and task_id not in flagged,
        }
    return findings


def _unsafe_total(value: Any) -> int:
    if isinstance(value, Mapping) and "total" in value:
        return _nested_total(value.get("total"))
    return _nested_total(value)


def _headline_group_coverage(exp5627: JsonMap) -> float | None:
    groups = exp5627.get("worst_group_coverage")
    if not isinstance(groups, Mapping):
        return None
    headline = groups.get("group_conditional_online_conformal")
    if isinstance(headline, Mapping) and isinstance(headline.get("coverage"), int | float):
        return float(headline["coverage"])
    return None


def _one_axis_exact_clean(exp5633: JsonMap, status: str) -> bool:
    return bool(
        status == "complete"
        and _float(exp5633, "replica_exchange_kernel_ready_score") >= 1.0
        and _float(exp5633, "exact_distribution_tv_max") <= 1e-12
        and _float(exp5633, "swap_detailed_balance_residual_max") <= 1e-12
        and _float(exp5633, "transition_normalization_error_max") <= 1e-12
        and not exp5633.get("validity_regression_detected")
        and not exp5633.get("hardware_speedup_claimed")
        and not exp5633.get("timing_claimed")
    )


def _one_axis_quality_clean(exp5634: JsonMap, status: str) -> bool:
    wall_time = exp5634.get("wall_time_provenance_only")
    wall_time_clean = not isinstance(wall_time, Mapping) or not wall_time.get(
        "speedup_claim_allowed"
    )
    return bool(
        status == "complete"
        and exp5634.get("quality_mixing_ready") is True
        and exp5634.get("target_diagnostics_within_exp5633_bounds") is True
        and not exp5634.get("hardware_speedup_claimed")
        and not exp5634.get("timing_claimed")
        and wall_time_clean
    )


def promoted_substrates(
    artifacts: Mapping[str, JsonMap],
    findings: Mapping[str, JsonMap],
) -> list[JsonDict]:
    exp5633 = _payload(artifacts, EXP5633_TEMPERATURE_EXACT_PATH)
    exp5634 = _payload(artifacts, EXP5634_TEMPERATURE_QUALITY_PATH)
    rows: list[JsonDict] = []
    exact_status = str(findings["exp5633-temperature-exchange-cdls-exact-audit"]["status"])
    quality_status = str(findings["exp5634-temperature-exchange-cdls-quality"]["status"])
    if _one_axis_exact_clean(exp5633, exact_status):
        rows.append(
            {
                "key": "one_axis_temperature_exchange_exact",
                "source_artifacts": [EXP5633_TEMPERATURE_EXACT_PATH.as_posix()],
                "prerequisite_for": [
                    "exp5644-two-axis-parallel-tempering-exact-audit",
                    "exp5645-two-axis-tempering-hard-constraint-quality",
                    "exp5646-two-axis-tempering-rust-parity",
                ],
                "evidence": {
                    "replica_exchange_kernel_ready_score": exp5633.get(
                        "replica_exchange_kernel_ready_score"
                    ),
                    "exact_distribution_tv_max": exp5633.get("exact_distribution_tv_max"),
                    "swap_detailed_balance_residual_max": exp5633.get(
                        "swap_detailed_balance_residual_max"
                    ),
                    "transition_normalization_error_max": exp5633.get(
                        "transition_normalization_error_max"
                    ),
                    "broken_control_count": len(exp5633.get("broken_controls") or []),
                },
                "claim_boundary": (
                    "one-axis exact temperature-label exchange only; no timing, CPU/CUDA "
                    "crossover, board, SNN, TSU, or hardware-speedup claim"
                ),
            }
        )
    if _one_axis_quality_clean(exp5634, quality_status):
        rows.append(
            {
                "key": "one_axis_temperature_exchange_quality",
                "source_artifacts": [
                    EXP5633_TEMPERATURE_EXACT_PATH.as_posix(),
                    EXP5634_TEMPERATURE_QUALITY_PATH.as_posix(),
                ],
                "prerequisite_for": [
                    "exp5644-two-axis-parallel-tempering-exact-audit",
                    "exp5645-two-axis-tempering-hard-constraint-quality",
                    "exp5646-two-axis-tempering-rust-parity",
                ],
                "evidence": {
                    "quality_mixing_ready": bool(exp5634.get("quality_mixing_ready")),
                    "target_diagnostics_within_exp5633_bounds": bool(
                        exp5634.get("target_diagnostics_within_exp5633_bounds")
                    ),
                    "paired_deltas_and_intervals": exp5634.get("paired_deltas_and_intervals", {}),
                    "wall_time_provenance_only": exp5634.get("wall_time_provenance_only", {}),
                },
                "claim_boundary": (
                    "hard-instance quality and mixing only; wall time is provenance, "
                    "not speedup or hardware evidence"
                ),
            }
        )
    return rows


def promising_unpromoted_substrates(artifacts: Mapping[str, JsonMap]) -> list[JsonDict]:
    exp5627 = _payload(artifacts, EXP5627_QUALIFICATION_PATH)
    exp5628 = _payload(artifacts, EXP5628_KAN_CSL_PATH)
    exp5629 = _payload(artifacts, EXP5629_KAN_AUDIT_PATH)
    capstone = _payload(artifacts, EXP5635_CAPSTONE_PATH)
    csl_promotion = capstone.get("continuous_self_learning_promotion")
    return [
        {
            "key": "fr11_conformal_active_spline_kan_internal",
            "source_artifacts": [
                EXP5627_QUALIFICATION_PATH.as_posix(),
                EXP5628_KAN_CSL_PATH.as_posix(),
                EXP5629_KAN_AUDIT_PATH.as_posix(),
                EXP5635_CAPSTONE_PATH.as_posix(),
            ],
            "promoted": False,
            "independent_promotion": False,
            "terminal_state": "clean_internal_ready_but_unaudited",
            "blocking_gate": {
                "exp5629_status": _status_for_payload(
                    exp5629,
                    {"exists": True, "loadable": bool(exp5629)},
                ),
                "gate_check_summary": exp5629.get("gate_check_summary"),
                "schema_gate_failure_not_scientific_negative": True,
                "capstone_promoted": (
                    csl_promotion.get("promoted") if isinstance(csl_promotion, Mapping) else None
                ),
                "capstone_independent_certified": (
                    csl_promotion.get("independent_certified")
                    if isinstance(csl_promotion, Mapping)
                    else None
                ),
            },
            "evidence": {
                "conformal_qualification_ready_score": exp5627.get(
                    "conformal_qualification_ready_score"
                ),
                "headline_worst_group_coverage": _headline_group_coverage(exp5627),
                "qualification_exact_unsafe_accept_total": _unsafe_total(
                    exp5627.get("exact_unsafe_accept_count")
                ),
                "continuous_self_learning_ready": bool(
                    exp5628.get("continuous_self_learning_ready")
                ),
                "unsafe_false_accept_total": _unsafe_total(
                    exp5628.get("unsafe_false_accept_count")
                ),
                "full_conformal_kan_ale": (
                    exp5628.get("ale_by_arm", {})
                    .get("full_conformal_kan_controller", {})
                    .get("mean")
                    if isinstance(exp5628.get("ale_by_arm"), Mapping)
                    else None
                ),
                "best_fixed_nonoracle_ale": (
                    exp5628.get("ale_by_arm", {}).get("best_fixed_nonoracle", {}).get("mean")
                    if isinstance(exp5628.get("ale_by_arm"), Mapping)
                    else None
                ),
                "poison_rejection": exp5628.get("poison_rejection_rate"),
                "checkpoint_replay_exact": exp5628.get("checkpoint_replay_exact"),
                "delayed_regression_recovery": exp5628.get("delayed_regression_recovery"),
            },
            "claim_boundary": (
                "may seed schema corrigendum and independent audit, but cannot be "
                "treated as an independent FR-11 promotion"
            ),
        }
    ]


def retired_scopes(artifacts: Mapping[str, JsonMap]) -> list[JsonDict]:
    exp5625 = _payload(artifacts, EXP5625_TRANSITION_PATH)
    exp5629 = _payload(artifacts, EXP5629_KAN_AUDIT_PATH)
    exp5630 = _payload(artifacts, EXP5630_ARC_PROBE_PATH)
    exp5631 = _payload(artifacts, EXP5631_ARC_AB_PATH)
    exp5632 = _payload(artifacts, EXP5632_ARC_LEVEL_PATH)
    exp5634 = _payload(artifacts, EXP5634_TEMPERATURE_QUALITY_PATH)
    prior_retired = exp5625.get("retired_scopes", []) if isinstance(exp5625, Mapping) else []
    return [
        {
            "key": "fr11_independent_promotion_claim",
            "closed": True,
            "scientific_negative": False,
            "source_artifacts": [
                EXP5629_KAN_AUDIT_PATH.as_posix(),
                EXP5635_CAPSTONE_PATH.as_posix(),
            ],
            "reason": "The audit was schema-gate skipped, so no independent FR-11 promotion exists yet.",
            "evidence": {
                "blocked_at_layer": exp5629.get("blocked_at_layer"),
                "gate_check_summary": exp5629.get("gate_check_summary"),
                "gates_evaluated": exp5629.get("gates_evaluated", []),
            },
        },
        {
            "key": "arc_epistemic_object_probe",
            "closed": True,
            "source_artifacts": [EXP5630_ARC_PROBE_PATH.as_posix()],
            "reason": "The probe is terminally blocked by a degenerate or unreachable readiness score.",
            "evidence": {
                "epistemic_probe_ready_score": exp5630.get("epistemic_probe_ready_score"),
                "object_hypothesis_non_degenerate_count": exp5630.get(
                    "object_hypothesis_non_degenerate_count"
                ),
                "unsafe_model_accept_count": exp5630.get("unsafe_model_accept_count"),
                "informative_control_delta": exp5630.get("informative_control_delta"),
                "solve_provenance": exp5630.get("solve_provenance"),
            },
        },
        {
            "key": "arc_epistemic_probe_live_ab",
            "closed": True,
            "source_artifacts": [EXP5631_ARC_AB_PATH.as_posix()],
            "reason": "The known-level A/B was skipped because Exp5630 did not pass readiness.",
            "evidence": {
                "blocked_at_layer": exp5631.get("blocked_at_layer"),
                "gate_check_summary": exp5631.get("gate_check_summary"),
                "gates_evaluated": exp5631.get("gates_evaluated", []),
            },
        },
        {
            "key": "arc_live_level_credit_v508",
            "closed": True,
            "source_artifacts": [EXP5632_ARC_LEVEL_PATH.as_posix()],
            "reason": "The bounded live attempt executed but banked no new reproducible level.",
            "evidence": {
                "flagged_adversarial": bool(exp5632.get("flagged_adversarial")),
                "registry_count_before": exp5632.get("registry_count_before"),
                "registry_count_after": exp5632.get("registry_count_after"),
                "registry_delta": exp5632.get("registry_delta"),
                "new_reproducible_levels": exp5632.get("new_reproducible_levels", []),
                "offline_reproduced": bool(exp5632.get("offline_reproduced")),
                "selected_game": exp5632.get("selected_game"),
                "selected_level": exp5632.get("selected_level"),
            },
        },
        {
            "key": "cdls_timing_crossover_and_hardware_speedup",
            "closed": True,
            "source_artifacts": [
                EXP5625_TRANSITION_PATH.as_posix(),
                EXP5633_TEMPERATURE_EXACT_PATH.as_posix(),
                EXP5634_TEMPERATURE_QUALITY_PATH.as_posix(),
            ],
            "reason": "The promoted sampler evidence is exactness and quality only; timing remains banned.",
            "evidence": {
                "prior_retired_scopes": prior_retired,
                "exp5634_hardware_speedup_claimed": bool(exp5634.get("hardware_speedup_claimed")),
                "exp5634_timing_claimed": bool(exp5634.get("timing_claimed")),
                "wall_time_provenance_only": exp5634.get("wall_time_provenance_only", {}),
            },
        },
        {
            "key": "board_snn_tsu_claims",
            "closed": True,
            "source_artifacts": [EXP5635_CAPSTONE_PATH.as_posix(), VNEXT_RELATIVE_PATH.as_posix()],
            "reason": "No board, SNN, or TSU execution is required or promoted by the V508 terminal set.",
            "evidence": {
                "board_claim_allowed": False,
                "snn_claim_allowed": False,
                "tsu_claim_allowed": False,
            },
        },
    ]


def adversarial_flags_preserved(findings: Mapping[str, JsonMap]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for task_id, row in findings.items():
        if not row.get("flagged_adversarial"):
            continue
        pending = row.get("corrigendum_pending")
        flags = pending if isinstance(pending, list) else []
        rows.append(
            {
                "task_id": task_id,
                "artifact_path": row.get("artifact_path"),
                "status": row.get("status"),
                "flag_kinds": [
                    str(flag.get("kind")) for flag in flags if isinstance(flag, Mapping)
                ],
                "flag_severities": [
                    str(flag.get("severity")) for flag in flags if isinstance(flag, Mapping)
                ],
                "upgraded_to_clean": False,
                "claim_boundary": "flagged evidence remains visible and is never a clean substrate",
            }
        )
    return rows


def dependency_map() -> JsonDict:
    return {
        "fr11_schema_corrigendum_to_shadow_integration": {
            "chain": "FR-11 schema corrigendum->anytime-valid audit->shadow integration",
            "tasks": [
                "exp5638-fr11-gate-schema-corrigendum",
                "exp5639-anytime-valid-csl-independent-audit",
                "exp5640-fr11-shadow-pipeline-integration",
            ],
            "upstream_clean_internal_evidence": [
                "exp5627-online-conformal-kan-qualification",
                "exp5628-conformal-active-spline-kan-csl",
            ],
            "not_independent_promotion": "exp5629-conformal-kan-independent-audit",
            "gates": [
                {
                    "upstream": "exp5638-fr11-gate-schema-corrigendum",
                    "field": "gate_contract_ready_score",
                    "op": ">=",
                    "value": 1.0,
                },
                {
                    "upstream": "exp5638-fr11-gate-schema-corrigendum",
                    "field": "unsafe_false_accept_count_total",
                    "op": "==",
                    "value": 0,
                },
                {
                    "upstream": "exp5639-anytime-valid-csl-independent-audit",
                    "field": "fr11_independent_promotion_ready_score",
                    "op": ">=",
                    "value": 1.0,
                },
                {
                    "upstream": "exp5640-fr11-shadow-pipeline-integration",
                    "field": "shadow_integration_ready_score",
                    "op": ">=",
                    "value": 1.0,
                },
            ],
        },
        "executable_arc_model_to_live_attempt": {
            "chain": "executable ARC model->advisory known-level A/B->unconditional live attempt",
            "tasks": [
                "exp5641-arc-counterexample-executable-model",
                "exp5642-arc-executable-model-live-ab",
                "exp5643-arc-live-self-discovery-levelup-v509",
            ],
            "closed_scope_not_reopened": "arc_epistemic_object_probe",
            "gates": [
                {
                    "upstream": "exp5641-arc-counterexample-executable-model",
                    "field": "counterexample_replay_ready_score",
                    "op": ">=",
                    "value": 1.0,
                },
                {
                    "upstream": "exp5641-arc-counterexample-executable-model",
                    "field": "unsafe_patch_accept_count",
                    "op": "==",
                    "value": 0,
                },
                {
                    "upstream": "exp5642-arc-executable-model-live-ab",
                    "field": "live_ab_promotion_allowed",
                    "op": "==",
                    "value": True,
                    "advisory_only": True,
                },
                {
                    "upstream": "exp5643-arc-live-self-discovery-levelup-v509",
                    "field": "live_attempt_executed",
                    "op": "==",
                    "value": True,
                    "unconditional": True,
                },
            ],
        },
        "two_axis_tempering_to_rust_parity": {
            "chain": "two-axis invariant audit->quality->Rust parity",
            "tasks": [
                "exp5644-two-axis-parallel-tempering-exact-audit",
                "exp5645-two-axis-tempering-hard-constraint-quality",
                "exp5646-two-axis-tempering-rust-parity",
            ],
            "upstream_promoted_substrate": "one_axis_temperature_exchange_quality",
            "banned_claims_preserved": [
                "timing",
                "CPU/CUDA crossover",
                "board",
                "SNN",
                "TSU",
                "hardware speedup",
            ],
            "gates": [
                {
                    "upstream": "exp5644-two-axis-parallel-tempering-exact-audit",
                    "field": "two_axis_invariant_ready_score",
                    "op": ">=",
                    "value": 1.0,
                },
                {
                    "upstream": "exp5645-two-axis-tempering-hard-constraint-quality",
                    "field": "two_axis_quality_promoted",
                    "op": "==",
                    "value": True,
                },
                {
                    "upstream": "exp5646-two-axis-tempering-rust-parity",
                    "field": "rust_python_parity_ready_score",
                    "op": ">=",
                    "value": 1.0,
                },
            ],
        },
        "capstone_reconciliation": {
            "chain": "Exp5636-Exp5646->Exp5647 reconciliation",
            "tasks": [*EXPECTED_TASK_IDS[:-1], "exp5647-v509-capstone-reconciliation"],
            "gate": (
                "Exp5647 cannot promote blocked, skipped, development-proxy, flagged, "
                "unpromoted, retired, timing, board, SNN, TSU, or hardware-speedup evidence."
            ),
        },
    }


def _protected_file_checks(
    root: Path,
    *,
    roadmap_modified: bool,
    conductor_modified: bool,
) -> list[JsonDict]:
    return [
        {
            "path": ROADMAP_RELATIVE_PATH.as_posix(),
            "exists": (root / ROADMAP_RELATIVE_PATH).exists(),
            "git_status_clean": not roadmap_modified,
            "sha256": path_sha256(root / ROADMAP_RELATIVE_PATH),
        },
        {
            "path": CONDUCTOR_RELATIVE_PATH.as_posix(),
            "exists": (root / CONDUCTOR_RELATIVE_PATH).exists(),
            "git_status_clean": not conductor_modified,
            "sha256": path_sha256(root / CONDUCTOR_RELATIVE_PATH),
        },
    ]


def _failed_preconditions(
    missing_artifacts: Sequence[str],
    malformed_artifacts: Sequence[str],
    collision: JsonMap,
    *,
    roadmap_modified: bool,
    conductor_modified: bool,
) -> list[str]:
    failures = [f"missing_artifact:{path}" for path in missing_artifacts]
    failures.extend(f"malformed_artifact:{path}" for path in malformed_artifacts)
    failures.extend(
        f"current_task_id_collision_completed:{task_id}"
        for task_id in collision.get("completed_colliding_ids", [])
    )
    failures.extend(
        f"current_task_id_collision_retired:{task_id}"
        for task_id in collision.get("retired_colliding_ids", [])
    )
    if not collision.get("range_matches_roadmap"):
        failures.append("roadmap_task_range_mismatch")
    if roadmap_modified:
        failures.append("research-roadmap.yaml_modified")
    if conductor_modified:
        failures.append("scripts/research_conductor.py_modified")
    return failures


def _honest_verdict(status: str, failures: Sequence[str]) -> str:
    if status == "complete":
        return (
            "complete: archived .508 terminal evidence into .509 dependency map; "
            "current_task_range=exp5636-exp5647; fr11_internal_ready=True; "
            "fr11_independent_promoted=False; exp5629_schema_gate_failure=True; "
            "arc_epistemic_probe_retired=True; exp5632_flags_preserved=True; "
            "one_axis_temperature_exchange_promoted=True; timing_hardware_claims_closed=True."
        )
    first = failures[0] if failures else "unknown"
    return f"blocked: .509 transition receipt failed precondition {first}."


def build_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[str | Mapping[str, Any]] = DEFAULT_TESTS_RUN,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifacts, metadata, artifacts_read = read_artifacts(root)
    source_context, source_context_missing = _read_source_context(root)
    roadmap, _roadmap_meta = read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    roadmap_task_ids = extract_roadmap_tasks(roadmap)
    roadmap_doc_task_range = _task_range_from_text(_read_text(root, VNEXT_RELATIVE_PATH))
    roadmap_modified = _modification_status(root, ROADMAP_RELATIVE_PATH, modification_overrides)
    conductor_modified = _modification_status(root, CONDUCTOR_RELATIVE_PATH, modification_overrides)
    missing_artifacts = [
        path.as_posix()
        for path in UPSTREAM_ARTIFACT_PATHS
        if not metadata[path.as_posix()].get("exists")
    ]
    malformed_artifacts = [
        path.as_posix()
        for path in UPSTREAM_ARTIFACT_PATHS
        if metadata[path.as_posix()].get("exists") and not metadata[path.as_posix()].get("loadable")
    ]
    findings = terminal_findings(artifacts, metadata)
    collision = _collision_check(root, roadmap_task_ids)
    failures = _failed_preconditions(
        missing_artifacts,
        malformed_artifacts,
        collision,
        roadmap_modified=roadmap_modified,
        conductor_modified=conductor_modified,
    )
    status = "complete" if not failures else "blocked"
    tests = [dict(row) if isinstance(row, Mapping) else {"command": str(row)} for row in tests_run]
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "previous_milestone": PREVIOUS_MILESTONE,
        "current_milestone": CURRENT_MILESTONE,
        "previous_task_range": PREVIOUS_TASK_RANGE,
        "current_task_range": CURRENT_TASK_RANGE,
        "current_task_collision_check": collision,
        "artifact_metadata": metadata,
        "source_context": source_context,
        "source_context_missing": source_context_missing,
        "missing_artifacts": missing_artifacts,
        "malformed_artifacts": malformed_artifacts,
        "roadmap_task_ids": roadmap_task_ids,
        "roadmap_task_count": len(roadmap_task_ids),
        "roadmap_doc_task_range": roadmap_doc_task_range,
        "protected_file_checks": _protected_file_checks(
            root,
            roadmap_modified=roadmap_modified,
            conductor_modified=conductor_modified,
        ),
        "preconditions_checked": {
            "upstream_artifacts_expected": len(UPSTREAM_ARTIFACT_PATHS),
            "upstream_artifacts_read": len(artifacts_read),
            "roadmap_task_count": len(roadmap_task_ids),
            "roadmap_doc_task_range": roadmap_doc_task_range,
            "research_complete_checked": (root / RESEARCH_COMPLETE_RELATIVE_PATH).exists(),
            "conductor_log_checked": (root / CONDUCTOR_LOG_RELATIVE_PATH).exists(),
            "collision_free": bool(collision.get("collision_free")),
            "roadmap_yaml_unchanged": not roadmap_modified,
            "conductor_unchanged": not conductor_modified,
        },
        "failed_preconditions": failures,
        "tests_run": tests,
        "roadmap_yaml_unchanged": not roadmap_modified,
        "conductor_unchanged": not conductor_modified,
        "field_principles": dict(FIELD_PRINCIPLES),
        "artifacts_read": artifacts_read,
        "terminal_findings": findings,
        "promoted_substrates": promoted_substrates(artifacts, findings),
        "promising_unpromoted_substrates": promising_unpromoted_substrates(artifacts),
        "retired_scopes": retired_scopes(artifacts),
        "adversarial_flags_preserved": adversarial_flags_preserved(findings),
        "dependency_map": dependency_map(),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(status, failures),
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


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
    for field in BOOL_FIELDS:
        if field in payload and not isinstance(payload[field], bool):
            errors.append(field)
    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    terminal = payload.get("terminal_findings")
    if isinstance(terminal, Mapping) and not set(TASK_ARTIFACT_PATHS).issubset(terminal):
        errors.append("terminal_findings")
    promoted = payload.get("promoted_substrates")
    if payload.get("status") == "complete" and isinstance(promoted, list):
        keys = {str(row.get("key")) for row in promoted if isinstance(row, Mapping)}
        if (
            not {
                "one_axis_temperature_exchange_exact",
                "one_axis_temperature_exchange_quality",
            }
            <= keys
        ):
            errors.append("promoted_substrates")
    promising = payload.get("promising_unpromoted_substrates")
    if payload.get("status") == "complete" and isinstance(promising, list):
        keys = {str(row.get("key")) for row in promising if isinstance(row, Mapping)}
        if "fr11_conformal_active_spline_kan_internal" not in keys:
            errors.append("promising_unpromoted_substrates")
    retired = payload.get("retired_scopes")
    if isinstance(retired, list):
        keys = {str(row.get("key")) for row in retired if isinstance(row, Mapping)}
        if (
            not {
                "fr11_independent_promotion_claim",
                "arc_epistemic_object_probe",
                "arc_epistemic_probe_live_ab",
                "arc_live_level_credit_v508",
                "cdls_timing_crossover_and_hardware_speedup",
                "board_snn_tsu_claims",
            }
            <= keys
        ):
            errors.append("retired_scopes")
    flags = payload.get("adversarial_flags_preserved")
    if payload.get("status") == "complete" and isinstance(flags, list):
        ids = {str(row.get("task_id")) for row in flags if isinstance(row, Mapping)}
        if "exp5632-arc-live-self-discovery-levelup-v508" not in ids:
            errors.append("adversarial_flags_preserved")
    if payload.get("current_task_range") != CURRENT_TASK_RANGE:
        errors.append("current_task_range")
    collision = payload.get("current_task_collision_check")
    if (
        payload.get("status") == "complete"
        and isinstance(collision, Mapping)
        and collision.get("collision_free") is not True
    ):
        errors.append("current_task_collision_check")
    dependency = payload.get("dependency_map")
    if not isinstance(dependency, Mapping) or not {
        "fr11_schema_corrigendum_to_shadow_integration",
        "executable_arc_model_to_live_attempt",
        "two_axis_tempering_to_rust_parity",
    } <= set(dependency):
        errors.append("dependency_map")
    if payload.get("roadmap_yaml_unchanged") is not True:
        errors.append("roadmap_yaml_unchanged")
    if payload.get("conductor_unchanged") is not True:
        errors.append("conductor_unchanged")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    checksum = payload.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:"):
        errors.append("reproducibility_checksum")
    honest_verdict = payload.get("honest_verdict")
    if not isinstance(honest_verdict, str) or not honest_verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    return sorted(set(errors))


def write_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[str | Mapping[str, Any]] = DEFAULT_TESTS_RUN,
    modification_overrides: Mapping[Path | str, bool] | None = None,
) -> JsonDict:
    artifact = build_report(
        root=root,
        tests_run=tests_run,
        modification_overrides=modification_overrides,
    )
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover - validation failure paths are covered without raising.
        raise ValueError(f"invalid Exp5636 artifact fields: {', '.join(errors)}")
    write_json(root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the Exp5636 artifact")
    args = parser.parse_args(argv)
    artifact = write_report() if args.write else build_report()
    if not args.write:
        write_json(Path("/dev/stdout"), artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
