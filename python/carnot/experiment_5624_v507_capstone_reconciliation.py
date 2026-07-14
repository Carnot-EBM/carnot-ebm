"""Exp5624 V507 capstone reconciliation.

Spec refs: REQ-CAPSTONE-5624, SCENARIO-CAPSTONE-5624,
SCENARIO-CAPSTONE-5624-MISSING-MALFORMED,
SCENARIO-CAPSTONE-5624-FIELD-PRINCIPLES.

This module is an evidence ledger, not a new experiment. It reads the terminal
Exp5613-Exp5623 artifacts, keeps blocked, skipped, flagged, malformed, and
complete states separate, then records narrow promotion and retirement
decisions. The capstone makes no new scientific inference; it only aggregates
what the upstream files support.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5624_v507_capstone_reconciliation.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CAPSTONE_SPEC_RELATIVE_PATH = Path("openspec/capabilities/capstone/spec.md")

EXPERIMENT = "experiment_5624_v507_capstone_reconciliation"
EXPERIMENT_ID = "exp5624-v507-capstone-reconciliation"
MILESTONE = "2026.07.507"
RUN_DATE = "2026-07-14"
RANDOM_SEED = 5624
SCHEMA = "carnot.experiment_5624.v507_capstone_reconciliation.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
TERMINAL_PREFIXES = ("complete:", "blocked:")

SPEC_REFS = (
    "REQ-CAPSTONE-5624",
    "SCENARIO-CAPSTONE-5624",
    "SCENARIO-CAPSTONE-5624-MISSING-MALFORMED",
    "SCENARIO-CAPSTONE-5624-FIELD-PRINCIPLES",
)

EXPECTED_TASK_IDS = (
    "exp5613-transition-v507",
    "exp5614-v507-source-delta-ingestion",
    "exp5615-native-llamacpp-cuda-runtime-certificate",
    "exp5616-exact-nonstationary-constraint-stream",
    "exp5617-kan-critical-task-duration-map",
    "exp5618-predictive-window-kan-self-learning",
    "exp5619-arc-forward-inverse-transition-cycle",
    "exp5620-arc-cycle-guarded-live-update-ab",
    "exp5621-arc-live-self-discovery-levelup-v507",
    "exp5622-cdls-exact-kernel-audit",
    "exp5623-cdls-multiseed-cpu-cuda-crossover",
    "exp5624-v507-capstone-reconciliation",
)

TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5613-transition-v507": Path("results/experiment_5613_transition_v507.json"),
    "exp5614-v507-source-delta-ingestion": Path(
        "results/experiment_5614_v507_source_delta_ingestion.json"
    ),
    "exp5615-native-llamacpp-cuda-runtime-certificate": Path(
        "results/experiment_5615_native_llamacpp_cuda_runtime_certificate.json"
    ),
    "exp5616-exact-nonstationary-constraint-stream": Path(
        "results/experiment_5616_exact_nonstationary_constraint_stream.json"
    ),
    "exp5617-kan-critical-task-duration-map": Path(
        "results/experiment_5617_kan_critical_task_duration_map.json"
    ),
    "exp5618-predictive-window-kan-self-learning": Path(
        "results/experiment_5618_predictive_window_kan_self_learning.json"
    ),
    "exp5619-arc-forward-inverse-transition-cycle": Path(
        "results/experiment_5619_arc_forward_inverse_transition_cycle.json"
    ),
    "exp5620-arc-cycle-guarded-live-update-ab": Path(
        "results/experiment_5620_arc_cycle_guarded_live_update_ab.json"
    ),
    "exp5621-arc-live-self-discovery-levelup-v507": Path(
        "results/experiment_5621_arc_live_self_discovery_levelup_v507.json"
    ),
    "exp5622-cdls-exact-kernel-audit": Path("results/experiment_5622_cdls_exact_kernel_audit.json"),
    "exp5623-cdls-multiseed-cpu-cuda-crossover": Path(
        "results/experiment_5623_cdls_multiseed_cpu_cuda_crossover.json"
    ),
}
EXP5615_RESPONSES_RELATIVE_PATH = Path(
    "results/experiment_5615_native_llamacpp_cuda_runtime_certificate.responses.jsonl"
)
EXP5621_TRACE_RELATIVE_PATH = Path(
    "results/experiment_5621_arc_live_self_discovery_levelup_v507_trace.json"
)
EXP5623_STATS_RELATIVE_PATH = Path(
    "results/experiment_5623_cdls_multiseed_cpu_cuda_crossover_sufficient_statistics.json"
)
PRIMARY_ARTIFACT_PATHS = tuple(TASK_ARTIFACT_PATHS.values())
SIDECAR_ARTIFACT_PATHS = (
    EXP5615_RESPONSES_RELATIVE_PATH,
    EXP5621_TRACE_RELATIVE_PATH,
    EXP5623_STATS_RELATIVE_PATH,
)

SOURCE_CONTEXT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
    Path("research-complete.yaml"),
    Path("research-references.md"),
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
    "research-references.md",
    "ops/exclusion_manifest.yaml",
    "ops/arc_solve_registry.yaml",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "field_principles": "one-line annotations for every required capstone field.",
    "expected_task_ids": "fixed milestone denominator; must list Exp5613 through Exp5624 task ids.",
    "artifacts_found": "only readable files can support claims.",
    "terminal_status_by_task": (
        "blocked, skipped, flagged, malformed, missing, and complete stay distinct."
    ),
    "headline_claims": (
        "narrow artifact-backed conclusions; flagged artifacts cannot support positive claims."
    ),
    "promotion_decisions": "mechanisms promote only through preregistered gates.",
    "retirement_decisions": (
        "repeated failures stop reruns while new scientific nulls stay planning inputs."
    ),
    "native_runtime_verdict": "native CUDA smoke/build evidence is not verifier-quality evidence.",
    "continuous_self_learning_verdict": (
        "FR-11 outcome is explicit and respects upstream duration-map gates."
    ),
    "arc_transition_verdict": (
        "development-proxy and flagged ARC evidence is bounded before live-path promotion."
    ),
    "arc_registry_delta": "levels_after - levels_before and new_reproducible_levels must agree.",
    "hardware_sampling_verdict": "exactness and quality gates bound all timing/crossover claims.",
    "documents_reconciled": (
        "spec and ops reconciliation state, including stop-rule delegated files."
    ),
    "tests_run": "commands, exit codes, counts, warnings, and skipped checks actually observed.",
    "unresolved_gaps": "negative evidence becomes planning input.",
    "inference_substrate": "must equal aggregation_from_upstream_artifacts.",
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
    "promoted_tasks",
    "retired_tasks",
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
    "promoted_tasks",
    "retired_tasks",
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
    "native_runtime_verdict",
    "continuous_self_learning_verdict",
    "arc_transition_verdict",
    "arc_registry_delta_evidence",
    "hardware_sampling_verdict",
    "documents_reconciled",
)

DEFAULT_TESTS_RUN = (
    {
        "command": (
            ".venv/bin/pytest "
            "tests/python/test_experiment_5624_v507_capstone_reconciliation.py -q --no-cov -n 0"
        ),
        "exit_code": None,
        "status": "not_run_in_default_artifact",
        "counts": {},
        "warnings": [],
        "skipped_checks": [],
    },
    {
        "command": (
            ".venv/bin/coverage run --include=python/carnot/"
            "experiment_5624_v507_capstone_reconciliation.py -m pytest "
            "tests/python/test_experiment_5624_v507_capstone_reconciliation.py -q --no-cov -n 0"
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
            "experiment_5624_v507_capstone_reconciliation.py --fail-under=100"
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


def _read_jsonl_sidecar(path: Path) -> tuple[JsonDict, JsonDict]:
    metadata: JsonDict = {
        "exists": path.exists(),
        "loadable": False,
        "json_type": "jsonl",
        "sha256": path_sha256(path),
        "line_count": 0,
    }
    if not path.exists():
        metadata["error"] = "missing"
        return {}, metadata
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - filesystem race guard
        metadata["error"] = f"unreadable:{exc.__class__.__name__}"
        return {}, metadata
    line_count = 0
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        line_count += 1
        try:
            json.loads(line)
        except json.JSONDecodeError as exc:
            metadata.update(
                {
                    "error": "malformed_jsonl",
                    "line": line_no,
                    "column": exc.colno,
                    "line_count": line_count,
                }
            )
            return {}, metadata
    metadata.update({"loadable": True, "error": None, "line_count": line_count})
    return {"line_count": line_count}, metadata


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
        if rel_path.suffix == ".jsonl":
            payload, meta = _read_jsonl_sidecar(root / rel_path)
        else:
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
    return bool(
        status == "blocked"
        or verdict.startswith("blocked:")
        or verdict.startswith("blocked_")
        or verdict.startswith("blocked ")
    )


def _is_complete(payload: JsonMap) -> bool:
    verdict = _verdict(payload).lower()
    status = str(payload.get("status") or "").lower()
    return bool(
        status == "complete" or verdict.startswith("complete:") or verdict.startswith("complete_")
    )


def _is_flagged(payload: JsonMap) -> bool:
    return bool(payload.get("flagged_adversarial"))


def _clean_for_positive_claim(payload: JsonMap) -> bool:
    return bool(payload) and not (
        _is_flagged(payload) or _is_gate_skip(payload) or _is_blocked(payload)
    )


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


def _count(value: Any) -> int:
    if isinstance(value, Mapping):
        if "total" in value:
            return _int(value, "total")
        return len(value)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return len(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    return 0


def _zero_false_accepts(value: Any) -> bool:
    if isinstance(value, Mapping):
        if "total" in value:
            return _int(value, "total") == 0
        return bool(value) and all(_zero_false_accepts(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return all(_zero_false_accepts(item) for item in value)
    if isinstance(value, bool):
        return not value
    if isinstance(value, int | float):
        return value == 0
    return False


def _all_mapping_values_true(value: Any) -> bool:
    return bool(
        isinstance(value, Mapping) and value and all(item is True for item in value.values())
    )


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


def _arc_delta(exp5621: JsonMap) -> tuple[int, JsonDict]:
    before = _int(exp5621, "levels_before")
    after = _int(exp5621, "levels_after")
    new_levels = exp5621.get("new_reproducible_levels")
    new_count = len(new_levels) if isinstance(new_levels, list) else 0
    arithmetic_delta = after - before
    delta = arithmetic_delta if arithmetic_delta == new_count else min(arithmetic_delta, new_count)
    target = exp5621.get("target_selection_receipt")
    target_map = target if isinstance(target, Mapping) else {}
    return delta, {
        "levels_before": before,
        "levels_after": after,
        "arithmetic_delta": arithmetic_delta,
        "new_reproducible_level_count": new_count,
        "new_reproducible_levels": list(new_levels) if isinstance(new_levels, list) else [],
        "selected_game": target_map.get("selected_game"),
        "selected_level": target_map.get("selected_level"),
        "offline_reproduced": bool(exp5621.get("offline_reproduced")),
        "registry_updated": bool(exp5621.get("registry_updated")),
        "target_reached_live": bool(exp5621.get("target_reached_live")),
        "flagged_adversarial": bool(exp5621.get("flagged_adversarial")),
        "solve_provenance": exp5621.get("solve_provenance"),
    }


def _failed_gate_summary(exp5623: JsonMap) -> JsonDict:
    explicit = exp5623.get("failed_gate_summary_by_device")
    if isinstance(explicit, Mapping):
        return dict(explicit)
    summary: JsonDict = {}
    pairs = exp5623.get("quality_gate_results_by_pair")
    if not isinstance(pairs, Sequence) or isinstance(pairs, str | bytes | bytearray):
        return summary
    for row in pairs:
        if not isinstance(row, Mapping):
            continue
        device = str(row.get("device") or row.get("backend") or "unknown")
        failures = row.get("failed_gates")
        if isinstance(failures, Sequence) and not isinstance(failures, str | bytes | bytearray):
            for gate in failures:
                gate_name = str(gate)
                device_row = summary.setdefault(device, {})
                device_row[gate_name] = _int(device_row, gate_name) + 1
        elif row.get("included_in_speedups") is False:
            device_row = summary.setdefault(device, {})
            device_row["quality_gate_failed"] = _int(device_row, "quality_gate_failed") + 1
    return summary


def derive_claims(
    artifacts: Mapping[str, JsonMap], statuses: Mapping[str, JsonMap]
) -> tuple[
    JsonDict,
    JsonDict,
    JsonDict,
    JsonDict,
    JsonDict,
    JsonDict,
    int,
    JsonDict,
    JsonDict,
    list[JsonDict],
]:
    exp5613 = _payload(artifacts, TASK_ARTIFACT_PATHS["exp5613-transition-v507"])
    exp5614 = _payload(artifacts, TASK_ARTIFACT_PATHS["exp5614-v507-source-delta-ingestion"])
    exp5615 = _payload(
        artifacts, TASK_ARTIFACT_PATHS["exp5615-native-llamacpp-cuda-runtime-certificate"]
    )
    exp5616 = _payload(
        artifacts, TASK_ARTIFACT_PATHS["exp5616-exact-nonstationary-constraint-stream"]
    )
    exp5617 = _payload(artifacts, TASK_ARTIFACT_PATHS["exp5617-kan-critical-task-duration-map"])
    exp5618 = _payload(
        artifacts, TASK_ARTIFACT_PATHS["exp5618-predictive-window-kan-self-learning"]
    )
    exp5619 = _payload(
        artifacts, TASK_ARTIFACT_PATHS["exp5619-arc-forward-inverse-transition-cycle"]
    )
    exp5620 = _payload(artifacts, TASK_ARTIFACT_PATHS["exp5620-arc-cycle-guarded-live-update-ab"])
    exp5621 = _payload(
        artifacts, TASK_ARTIFACT_PATHS["exp5621-arc-live-self-discovery-levelup-v507"]
    )
    exp5622 = _payload(artifacts, TASK_ARTIFACT_PATHS["exp5622-cdls-exact-kernel-audit"])
    exp5623 = _payload(artifacts, TASK_ARTIFACT_PATHS["exp5623-cdls-multiseed-cpu-cuda-crossover"])

    cuda_capability = exp5615.get("cuda_build_capability")
    cuda_map = cuda_capability if isinstance(cuda_capability, Mapping) else {}
    native_cuda_ready = bool(cuda_map.get("native_cuda_ready"))
    native_certificate_ok = bool(
        _clean_for_positive_claim(exp5615)
        and _int(exp5615, "models_certified_count") == _int(exp5615, "models_certified_denominator")
        and _int(exp5615, "models_certified_denominator") == 3
        and _float(exp5615, "runtime_certificate_ready_score") >= 1.0
    )
    fixture_ok = bool(
        _clean_for_positive_claim(exp5616)
        and _float(exp5616, "fixture_ready_score") >= 1.0
        and _int(exp5616, "oracle_label_error_count") == 0
    )
    duration_switch_count = _count(exp5617.get("nondegenerate_switch_cases"))
    duration_false_accepts_zero = _zero_false_accepts(exp5617.get("unsafe_false_accept_count"))
    duration_map_ok = bool(
        _clean_for_positive_claim(exp5617)
        and _int(exp5617, "critical_task_duration") > 0
        and duration_switch_count >= 2
        and duration_false_accepts_zero
        and exp5617.get("lazy_identity_guard_passed") is True
    )
    duration_fit_r2 = _float(exp5617, "critical_duration_fit_r2")
    prereg_duration_fit_passed = duration_fit_r2 >= 0.5
    controller_gate = exp5618.get("controller_gate_receipt")
    kan_artifact_ready = bool(exp5618.get("continuous_self_learning_ready"))
    kan_safety_passed = bool(
        _zero_false_accepts(exp5618.get("unsafe_false_accept_count"))
        and exp5618.get("rollback_positive_control")
        and exp5618.get("delayed_regression_passed")
        and exp5618.get("no_model_weight_mutation")
        and exp5618.get("kan_spline_state_mutated")
    )
    kan_promotion_allowed = bool(
        _clean_for_positive_claim(exp5618)
        and kan_artifact_ready
        and _all_mapping_values_true(controller_gate)
        and prereg_duration_fit_passed
        and duration_switch_count >= 2
        and duration_false_accepts_zero
        and kan_safety_passed
    )
    arc_transition_ok = bool(
        _clean_for_positive_claim(exp5619)
        and exp5619.get("solve_provenance") != "development_proxy"
        and _float(exp5619, "cycle_verifier_positive_control_rate") >= 0.9
        and _int(exp5619, "unsafe_transition_accept_count") == 0
    )
    arc_branch_ok = bool(_clean_for_positive_claim(exp5620) and exp5620.get("live_branch_promoted"))
    arc_registry_delta, arc_registry_delta_evidence = _arc_delta(exp5621)
    arc_registry_ok = bool(
        _clean_for_positive_claim(exp5621)
        and exp5621.get("solve_provenance") == "live_agent_self_discovery"
        and arc_registry_delta > 0
        and exp5621.get("offline_reproduced")
        and exp5621.get("registry_updated")
    )
    cdls_exact_ok = bool(
        _clean_for_positive_claim(exp5622)
        and exp5622.get("correction_applied") is True
        and _float(exp5622, "kernel_audit_ready_score") >= 1.0
        and _float(exp5622, "transition_row_sum_error_max") <= 1e-12
        and _float(exp5622, "detailed_balance_residual_max") <= 1e-12
        and _float(exp5622, "exact_distribution_tv_max") <= 1e-10
        and _float(exp5622, "energy_histogram_tv_max") <= 1e-10
        and exp5622.get("broken_kernel_controls_rejected") is True
        and _int(exp5622, "quality_gate_specified_count") >= 4
    )
    cdls_crossover_ok = bool(
        _clean_for_positive_claim(exp5623)
        and exp5623.get("crossover_claim_allowed")
        and _count(exp5623.get("successful_matched_pairs")) > 0
    )

    headline_claims: JsonDict = {
        "source_delta": {
            "claim_allowed": bool(
                _clean_for_positive_claim(exp5614)
                and exp5614.get("planner_marker_found")
                and exp5614.get("closed_scopes_reopened") is False
            ),
            "claim": "no_new_non_duplicate_actionable_source_delta",
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5614-v507-source-delta-ingestion"].as_posix()
            ],
            "evidence": {
                "planner_marker_found": bool(exp5614.get("planner_marker_found")),
                "new_references_added": exp5614.get("new_references_added", []),
                "closed_scopes_reopened": bool(exp5614.get("closed_scopes_reopened")),
            },
        },
        "native_runtime_certificate": {
            "claim_allowed": native_certificate_ok,
            "claim": "no_native_three_model_runtime_certificate",
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5615-native-llamacpp-cuda-runtime-certificate"].as_posix(),
                EXP5615_RESPONSES_RELATIVE_PATH.as_posix(),
            ],
            "evidence": {
                "native_cuda_ready": native_cuda_ready,
                "models_certified_count": _int(exp5615, "models_certified_count"),
                "runtime_certificate_ready_score": exp5615.get("runtime_certificate_ready_score"),
            },
        },
        "exact_drift_fixture": {
            "claim_allowed": fixture_ok,
            "claim": "exact_nonstationary_constraint_stream_fixture_ready"
            if fixture_ok
            else "not_promoted",
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5616-exact-nonstationary-constraint-stream"].as_posix()
            ],
            "evidence": {
                "fixture_ready_score": exp5616.get("fixture_ready_score"),
                "oracle_label_error_count": exp5616.get("oracle_label_error_count"),
                "exact_oracle_label_count": exp5616.get("exact_oracle_label_count"),
                "dataset_row_count": exp5616.get("dataset_row_count"),
                "task_durations": exp5616.get("task_durations"),
            },
        },
        "critical_duration_map": {
            "claim_allowed": duration_map_ok,
            "claim": "critical_task_duration_d16_map_measured_not_predictive_fit",
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5617-kan-critical-task-duration-map"].as_posix()
            ],
            "evidence": {
                "critical_task_duration": exp5617.get("critical_task_duration"),
                "critical_duration_fit_r2": exp5617.get("critical_duration_fit_r2"),
                "nondegenerate_switch_case_count": duration_switch_count,
                "unsafe_false_accept_count": exp5617.get("unsafe_false_accept_count"),
                "lazy_identity_guard_passed": bool(exp5617.get("lazy_identity_guard_passed")),
            },
        },
        "predictive_kan_self_learning": {
            "claim_allowed": kan_promotion_allowed,
            "claim": "artifact_reports_ready_but_preregistered_duration_fit_gate_failed",
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5618-predictive-window-kan-self-learning"].as_posix()
            ],
            "evidence": {
                "continuous_self_learning_ready": kan_artifact_ready,
                "critical_duration_fit_r2": duration_fit_r2,
                "required_critical_duration_fit_r2": 0.5,
                "promotion_allowed": kan_promotion_allowed,
            },
        },
        "arc_transition_cycle": {
            "claim_allowed": arc_transition_ok,
            "claim": "no_arc_transition_cycle_promotion",
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5619-arc-forward-inverse-transition-cycle"].as_posix()
            ],
            "evidence": {
                "flagged_adversarial": bool(exp5619.get("flagged_adversarial")),
                "solve_provenance": exp5619.get("solve_provenance"),
                "cycle_verifier_positive_control_rate": exp5619.get(
                    "cycle_verifier_positive_control_rate"
                ),
                "unsafe_transition_accept_count": exp5619.get("unsafe_transition_accept_count"),
            },
        },
        "arc_live_branch": {
            "claim_allowed": arc_branch_ok,
            "claim": "guarded_live_branch_gate_skipped",
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5620-arc-cycle-guarded-live-update-ab"].as_posix()
            ],
            "evidence": {
                "status": statuses["exp5620-arc-cycle-guarded-live-update-ab"].get("status"),
                "blocked_at_layer": exp5620.get("blocked_at_layer"),
                "gate_check_summary": exp5620.get("gate_check_summary"),
            },
        },
        "arc_new_registry_levels": {
            "claim_allowed": arc_registry_ok,
            "claim": "no_new_reproducible_arc_level_banked",
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5621-arc-live-self-discovery-levelup-v507"].as_posix(),
                EXP5621_TRACE_RELATIVE_PATH.as_posix(),
            ],
            "evidence": arc_registry_delta_evidence,
        },
        "cdls_exact_kernel": {
            "claim_allowed": cdls_exact_ok,
            "claim": "corrected_cdls_exact_kernel_audit_ready" if cdls_exact_ok else "not_promoted",
            "source_artifacts": [TASK_ARTIFACT_PATHS["exp5622-cdls-exact-kernel-audit"].as_posix()],
            "evidence": {
                "correction_applied": bool(exp5622.get("correction_applied")),
                "kernel_audit_ready_score": exp5622.get("kernel_audit_ready_score"),
                "transition_row_sum_error_max": exp5622.get("transition_row_sum_error_max"),
                "detailed_balance_residual_max": exp5622.get("detailed_balance_residual_max"),
                "exact_distribution_tv_max": exp5622.get("exact_distribution_tv_max"),
                "energy_histogram_tv_max": exp5622.get("energy_histogram_tv_max"),
            },
        },
        "cdls_cpu_cuda_crossover": {
            "claim_allowed": cdls_crossover_ok,
            "claim": "no_quality_matched_cpu_cuda_crossover_claim",
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5623-cdls-multiseed-cpu-cuda-crossover"].as_posix(),
                EXP5623_STATS_RELATIVE_PATH.as_posix(),
            ],
            "evidence": {
                "successful_matched_pair_count": _count(exp5623.get("successful_matched_pairs")),
                "crossover_claim_allowed": bool(exp5623.get("crossover_claim_allowed")),
                "crossover_size": exp5623.get("crossover_size"),
                "board_speedup_claimed": bool(exp5623.get("board_speedup_claimed")),
            },
        },
    }

    promotion_decisions: JsonDict = {
        "native_runtime_certificate": {
            "decision": "do_not_promote_terminal_native_certificate_failure",
            "reason": "native CUDA/build smoke evidence exists, but zero of three models earned verifier-quality runtime certificates",
        },
        "exact_drift_fixture": {
            "decision": "promote_bounded" if fixture_ok else "do_not_promote",
            "reason": "exact oracle labels and fixture row counts are internally consistent"
            if fixture_ok
            else "fixture artifact missing, malformed, or not exact",
        },
        "critical_duration_map": {
            "decision": "promote_bounded" if duration_map_ok else "do_not_promote",
            "reason": "critical duration is measured, but fit quality is not enough for downstream predictive promotion",
        },
        "predictive_window_kan_self_learning": {
            "decision": "promote_bounded"
            if kan_promotion_allowed
            else "do_not_promote_preregistered_gate_failed",
            "reason": "Exp5617 critical_duration_fit_r2 is below the preregistered 0.5 gate",
        },
        "arc_transition_cycle_verifier": {
            "decision": "do_not_promote",
            "reason": "development-proxy artifact is adversarially flagged and positive-control acceptance is far below gate",
        },
        "arc_cycle_guarded_live_branch": {
            "decision": "do_not_promote",
            "reason": "conductor pre-gate skipped the live branch after the Exp5619 verifier gate failed",
        },
        "arc_live_registry_level": {
            "decision": "do_not_promote",
            "reason": "bp35 L9 produced no offline-reproduced registry delta and is adversarially flagged",
        },
        "cdls_exact_kernel": {
            "decision": "promote_bounded" if cdls_exact_ok else "do_not_promote",
            "reason": "corrected exact kernel residuals and broken-kernel controls passed"
            if cdls_exact_ok
            else "exact-kernel audit gates did not pass",
        },
        "cdls_cpu_cuda_crossover": {
            "decision": "do_not_promote",
            "reason": "zero quality-matched pairs entered speedups, so no timing or crossover claim is allowed",
        },
    }

    retirement_decisions: JsonDict = {
        "native_three_model_runtime_certificate": {
            "decision": "retire_terminal_same_verdict",
            "retire_if_same_verdict_applied": True,
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5615-native-llamacpp-cuda-runtime-certificate"].as_posix()
            ],
            "reason": "the same native runtime certificate direction again ended with zero certified models despite CUDA/build smoke evidence",
            "manifest_update": "delegated_by_stop_rule",
        },
        "predictive_window_kan_self_learning": {
            "decision": "do_not_retire_preregistered_gate_mismatch",
            "retire_if_same_verdict_applied": False,
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5618-predictive-window-kan-self-learning"].as_posix()
            ],
            "reason": "the artifact is positive internally, but the upstream duration-fit gate failed; treat as planning input, not terminal mechanism failure",
        },
        "arc_transition_cycle_verifier": {
            "decision": "do_not_retire_flagged_development_proxy",
            "retire_if_same_verdict_applied": False,
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5619-arc-forward-inverse-transition-cycle"].as_posix()
            ],
            "reason": "the artifact is adversarially flagged and development-proxy bounded, so it blocks promotion without retiring live ARC work",
        },
        "arc_live_levelup_bp35_l9": {
            "decision": "do_not_retire_new_rotated_target_null",
            "retire_if_same_verdict_applied": False,
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5621-arc-live-self-discovery-levelup-v507"].as_posix()
            ],
            "reason": "bp35 L9 is a rotated target null; retire only an unchanged repeat of the same route",
            "manifest_update": "delegated_by_stop_rule",
        },
        "cdls_exact_kernel": {
            "decision": "keep_promoted_bounded",
            "retire_if_same_verdict_applied": False,
            "source_artifacts": [TASK_ARTIFACT_PATHS["exp5622-cdls-exact-kernel-audit"].as_posix()],
            "reason": "exact corrected-kernel evidence passed and should bound future timing claims",
        },
        "cdls_cpu_cuda_crossover": {
            "decision": "close_current_timing_claim_until_quality_gate_fixed",
            "retire_if_same_verdict_applied": True,
            "source_artifacts": [
                TASK_ARTIFACT_PATHS["exp5623-cdls-multiseed-cpu-cuda-crossover"].as_posix()
            ],
            "reason": "the corrected exact kernel was available, but all matched CPU/CUDA timing pairs failed quality gates",
            "manifest_update": "delegated_by_stop_rule",
        },
    }

    native_runtime_verdict: JsonDict = {
        "certificate_ready": native_certificate_ok,
        "models_certified_count": _int(exp5615, "models_certified_count"),
        "models_certified_denominator": _int(exp5615, "models_certified_denominator"),
        "runtime_certificate_ready_score": exp5615.get("runtime_certificate_ready_score"),
        "native_cuda_ready": native_cuda_ready,
        "gpu_memory_delta_by_model": exp5615.get("gpu_memory_delta_by_model", {}),
        "offload_layers_by_model": exp5615.get("offload_layers_by_model", {}),
        "lossless_replay_rate": exp5615.get("lossless_replay_rate"),
        "stop_control_pass_rate": exp5615.get("stop_control_pass_rate"),
        "semantic_false_accept_count": exp5615.get("semantic_false_accept_count"),
        "orphan_process_count": exp5615.get("orphan_process_count"),
        "no_task_accuracy_computed": bool(exp5615.get("no_task_accuracy_computed")),
        "solve_verify_accuracy_inferred": bool(exp5615.get("solve_verify_accuracy_inferred")),
        "claim_boundary": "native CUDA smoke evidence only; no three-model verifier-quality runtime certificate",
    }
    continuous_self_learning_verdict: JsonDict = {
        "artifact_reports_ready": kan_artifact_ready,
        "promotion_allowed": kan_promotion_allowed,
        "duration_map_gate": {
            "critical_duration_fit_r2": duration_fit_r2,
            "required_critical_duration_fit_r2": 0.5,
            "critical_duration_fit_passed": prereg_duration_fit_passed,
            "nondegenerate_switch_case_count": duration_switch_count,
            "nondegenerate_switch_case_required": 2,
            "nondegenerate_switch_case_passed": duration_switch_count >= 2,
            "unsafe_false_accepts_zero": duration_false_accepts_zero,
        },
        "controller_gate_receipt": dict(controller_gate)
        if isinstance(controller_gate, Mapping)
        else {},
        "delta_ale_vs_best_fixed": exp5618.get("delta_ale_vs_best_fixed"),
        "forward_transfer_delta": exp5618.get("forward_transfer_delta"),
        "backward_retention_delta": exp5618.get("backward_retention_delta"),
        "forgetting_delta": exp5618.get("forgetting_delta"),
        "unsafe_false_accept_count": exp5618.get("unsafe_false_accept_count"),
        "rollback_positive_control": exp5618.get("rollback_positive_control"),
        "delayed_regression_passed": bool(exp5618.get("delayed_regression_passed")),
        "no_model_weight_mutation": bool(exp5618.get("no_model_weight_mutation")),
        "kan_spline_state_mutated": bool(exp5618.get("kan_spline_state_mutated")),
        "poison_update_disposition": exp5618.get("poison_update_disposition", {}),
        "llm_invoked": bool(exp5618.get("llm_invoked")),
        "llm_weight_training": bool(exp5618.get("llm_weight_training")),
        "claim_boundary": "FR-11 KAN artifact stayed bounded by upstream duration-map fit gate; no LLM or model-weight claim",
    }
    arc_transition_verdict: JsonDict = {
        "transition_cycle_promotion_allowed": arc_transition_ok,
        "live_branch_promotion_allowed": arc_branch_ok,
        "registry_level_promotion_allowed": arc_registry_ok,
        "transition_cycle": {
            "flagged_adversarial": bool(exp5619.get("flagged_adversarial")),
            "solve_provenance": exp5619.get("solve_provenance"),
            "cycle_verifier_positive_control_rate": exp5619.get(
                "cycle_verifier_positive_control_rate"
            ),
            "valid_transition_accept_rate": exp5619.get("valid_transition_accept_rate"),
            "corruption_reject_rate": exp5619.get("corruption_reject_rate"),
            "unsafe_transition_accept_count": exp5619.get("unsafe_transition_accept_count"),
            "abstention_rate": exp5619.get("abstention_rate"),
            "source_files_read": bool(exp5619.get("source_files_read")),
            "per_game_adapter_used": bool(exp5619.get("per_game_adapter_used")),
        },
        "guarded_live_branch": {
            "status": statuses["exp5620-arc-cycle-guarded-live-update-ab"].get("status"),
            "blocked_at_layer": exp5620.get("blocked_at_layer"),
            "gate_check_summary": exp5620.get("gate_check_summary"),
            "gates_evaluated": exp5620.get("gates_evaluated", []),
        },
        "live_levelup": arc_registry_delta_evidence,
        "claim_boundary": "ARC evidence remains development-proxy or flagged; live branch and registry promotion are not supported",
    }
    hardware_sampling_verdict: JsonDict = {
        "exact_kernel_ready": cdls_exact_ok,
        "exact_kernel_evidence": {
            "correction_applied": bool(exp5622.get("correction_applied")),
            "kernel_audit_ready_score": exp5622.get("kernel_audit_ready_score"),
            "transition_row_sum_error_max": exp5622.get("transition_row_sum_error_max"),
            "detailed_balance_residual_max": exp5622.get("detailed_balance_residual_max"),
            "exact_distribution_tv_max": exp5622.get("exact_distribution_tv_max"),
            "energy_histogram_tv_max": exp5622.get("energy_histogram_tv_max"),
            "quality_gate_specified_count": exp5622.get("quality_gate_specified_count"),
        },
        "crossover_claim_allowed": cdls_crossover_ok,
        "large_n_sampling_evidence": {
            "upstream_gate_receipt": exp5623.get("upstream_gate_receipt", {}),
            "preconditions": exp5623.get("preconditions", {}),
            "seeds": exp5623.get("seeds", []),
            "instance_sizes": exp5623.get("instance_sizes", []),
            "samples_per_pair": exp5623.get("samples_per_pair"),
            "quality_gate_result_count": _count(exp5623.get("quality_gate_results_by_pair")),
            "successful_matched_pair_count": _count(exp5623.get("successful_matched_pairs")),
            "crossover_size": exp5623.get("crossover_size"),
            "board_speedup_claimed": bool(exp5623.get("board_speedup_claimed")),
            "timing_interval_count": _count(exp5623.get("timing_intervals_by_size")),
            "failed_gate_summary_by_device": _failed_gate_summary(exp5623),
        },
        "claim_boundary": "corrected exactness evidence promotes; CPU/CUDA timing remains null because quality gates excluded every pair",
    }
    unresolved_gaps = [
        {
            "gap": "native_three_model_runtime_certificate_absent",
            "planning_input": "do not rerun this certificate shape without changed offload observability or model runtime instrumentation",
            "source_task": "exp5615-native-llamacpp-cuda-runtime-certificate",
        },
        {
            "gap": "predictive_kan_preregistered_fit_gate_failed",
            "planning_input": "FR-11 continuation needs a duration-map fit that clears the preregistered threshold before promotion",
            "source_task": "exp5618-predictive-window-kan-self-learning",
        },
        {
            "gap": "arc_transition_cycle_flagged_development_proxy",
            "planning_input": "replace the low-positive-control transition-cycle proxy before live ARC branch promotion",
            "source_task": "exp5619-arc-forward-inverse-transition-cycle",
        },
        {
            "gap": "arc_bp35_l9_no_new_reproducible_level",
            "planning_input": "standing ARC floor remains open, but this bp35 L9 route should not be repeated unchanged",
            "source_task": "exp5621-arc-live-self-discovery-levelup-v507",
        },
        {
            "gap": "cdls_cpu_cuda_quality_matched_pairs_absent",
            "planning_input": "fix quality/mixing criteria before another CPU/CUDA crossover claim",
            "source_task": "exp5623-cdls-multiseed-cpu-cuda-crossover",
        },
    ]
    if exp5613:
        unresolved_gaps.append(
            {
                "gap": "roadmap_transition_retirements_delegated",
                "planning_input": "apply manifest/status/changelog reconciliation in the separate conductor step required by the stop rule",
                "source_task": "exp5613-transition-v507",
            }
        )

    return (
        headline_claims,
        promotion_decisions,
        retirement_decisions,
        native_runtime_verdict,
        continuous_self_learning_verdict,
        arc_transition_verdict,
        arc_registry_delta,
        arc_registry_delta_evidence,
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
    spec_text = (
        spec_path.read_text(encoding="utf-8", errors="replace") if spec_path.exists() else ""
    )
    roadmap_path = root / ROADMAP_RELATIVE_PATH
    roadmap_text = (
        roadmap_path.read_text(encoding="utf-8", errors="replace") if roadmap_path.exists() else ""
    )
    return {
        "openspec_capstone_req_present": "REQ-CAPSTONE-5624" in spec_text,
        "updated_by_this_workflow": [
            CAPSTONE_SPEC_RELATIVE_PATH.as_posix(),
            "python/carnot/experiment_5624_v507_capstone_reconciliation.py",
            "tests/python/test_experiment_5624_v507_capstone_reconciliation.py",
            RESULT_RELATIVE_PATH.as_posix(),
        ],
        "delegated_by_stop_rule": list(DELEGATED_BY_STOP_RULE),
        "protected_files": {
            ROADMAP_RELATIVE_PATH.as_posix(): not roadmap_modified,
            CONDUCTOR_RELATIVE_PATH.as_posix(): not conductor_modified,
        },
        "research_roadmap_next_missing": ROADMAP_NEXT_RELATIVE_PATH.as_posix()
        in source_context_missing,
        "active_milestone": MILESTONE if MILESTONE in roadmap_text else None,
        "next_milestone_activated": False,
        "reconciliation_note": (
            "ops/status, ops/changelog, traceability, completion, references, exclusion, "
            "and ARC registry edits are recorded as delegated because the operator stop rule "
            "forbids touching them in this run."
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
        native_runtime_verdict,
        continuous_self_learning_verdict,
        arc_transition_verdict,
        arc_registry_delta,
        arc_registry_delta_evidence,
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
    promoted_tasks = [
        task_id
        for task_id in (
            "exp5616-exact-nonstationary-constraint-stream",
            "exp5617-kan-critical-task-duration-map",
            "exp5622-cdls-exact-kernel-audit",
        )
        if statuses[task_id].get("status") == "complete"
    ]
    retired_tasks = [
        task_id
        for task_id, row in {
            "exp5615-native-llamacpp-cuda-runtime-certificate": retirement_decisions[
                "native_three_model_runtime_certificate"
            ],
            "exp5623-cdls-multiseed-cpu-cuda-crossover": retirement_decisions[
                "cdls_cpu_cuda_crossover"
            ],
        }.items()
        if row.get("retire_if_same_verdict_applied")
    ]
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
        "promoted_tasks": promoted_tasks,
        "retired_tasks": retired_tasks,
        "current_task": EXPERIMENT_ID,
        "expected_task_ids": list(EXPECTED_TASK_IDS),
        "artifacts_found": [dict(row) for row in artifacts_found],
        "terminal_status_by_task": statuses,
        "headline_claims": headline_claims,
        "promotion_decisions": promotion_decisions,
        "retirement_decisions": retirement_decisions,
        "native_runtime_verdict": native_runtime_verdict,
        "continuous_self_learning_verdict": continuous_self_learning_verdict,
        "arc_transition_verdict": arc_transition_verdict,
        "arc_registry_delta": arc_registry_delta,
        "arc_registry_delta_evidence": arc_registry_delta_evidence,
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
            f"{status_prefix} .507 capstone aggregated {len(artifacts_found)} readable "
            f"artifact files across Exp5613-Exp5623; native_runtime_certificate="
            f"{native_runtime_verdict['certificate_ready']}; exact_fixture="
            f"{headline_claims['exact_drift_fixture']['claim_allowed']}; critical_duration_map="
            f"{headline_claims['critical_duration_map']['claim_allowed']}; predictive_kan_promoted="
            f"{continuous_self_learning_verdict['promotion_allowed']}; arc_registry_delta="
            f"{arc_registry_delta}; cdls_exact={hardware_sampling_verdict['exact_kernel_ready']}; "
            f"cdls_crossover={hardware_sampling_verdict['crossover_claim_allowed']}"
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
    native_runtime = payload.get("native_runtime_verdict")
    if not isinstance(native_runtime, Mapping) or not isinstance(
        native_runtime.get("certificate_ready"), bool
    ):
        errors.append("native_runtime_verdict")
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
        if not isinstance(protected, Mapping) or not protected.get(
            ROADMAP_RELATIVE_PATH.as_posix()
        ):
            errors.append("documents_reconciled")
        if not isinstance(protected, Mapping) or not protected.get(
            CONDUCTOR_RELATIVE_PATH.as_posix()
        ):
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
        raise ValueError(f"invalid Exp5624 artifact fields: {', '.join(errors)}")
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
    parser.add_argument("--write", action="store_true", help="write the Exp5624 artifact")
    parser.add_argument(
        "--tests-run-json",
        type=Path,
        default=None,
        help="optional JSON list of observed validation command records",
    )
    args = parser.parse_args(argv)
    tests_run = _load_tests_run(args.tests_run_json)
    artifact = (
        write_capstone(tests_run=tests_run) if args.write else run_capstone(tests_run=tests_run)
    )
    if not args.write:
        write_json(Path("/dev/stdout"), artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
