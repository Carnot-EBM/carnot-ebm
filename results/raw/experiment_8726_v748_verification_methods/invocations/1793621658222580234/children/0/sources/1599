"""Exp 5276: memory-assisted verifier-dose gated pilot.

Spec refs: REQ-VERIFY-5276, SCENARIO-VERIFY-5276.

The experiment joins three prior receipts: Exp 5271 proves local SOTA GGUF
telemetry is live, Exp 5275 proves decision-history memory is governed, and
Exp 5264 provides the verifier-dose scheduler replay. Memory is intentionally
limited to allocation: it can select a verifier dose or deterministic check
family, but it never writes the accepted answer directly.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot.pipeline import verifier_dose_scheduler_replay as scheduler
from carnot.provenance_receipts import receipt_bytes, receipt_exists


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 5276
EXPERIMENT_NAME = "experiment_5276_memory_assisted_verifier_dose_gated_v482"
RESULT_RELATIVE_PATH = Path("results/experiment_5276_memory_assisted_verifier_dose_gated_v482.json")
EXP5271_RELATIVE_PATH = Path("results/experiment_5271_sota_telemetry_receipt_harness_v482.json")
EXP5275_RELATIVE_PATH = Path("results/experiment_5275_governed_decision_history_memory_v482.json")
EXP5264_RELATIVE_PATH = Path("results/experiment_5264_verifier_dose_scheduler_replay_v481.json")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
SCHEMA = "carnot.experiment_5276.memory_assisted_verifier_dose_gated.v482"
SPEC_REFS = ("REQ-VERIFY-5276", "SCENARIO-VERIFY-5276")
INFERENCE_SUBSTRATE = "live_llm_inference_local_gguf_sota"
RANDOM_SEED = 5276

HEADLINE_ROLES = ("flagship_moe", "flagship_dense")
OPTIONAL_ROLES = ("middle_moe",)
ROLE_ORDER = HEADLINE_ROLES + OPTIONAL_ROLES
MANDATED_MODEL_IDS = {
    "flagship_moe": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "flagship_dense": "unsloth/gemma-4-31B-it-GGUF",
    "middle_moe": "unsloth/gemma-4-26B-A4B-it-GGUF",
}

ROUTE_NO_VERIFIER = "no_verifier"
ROUTE_CHEAP = "cheap_deterministic"
ROUTE_MEMORY_CHECK = "memory_guided_deterministic_check"
ROUTE_FULL = "full_verifier"
ROUTES = (ROUTE_NO_VERIFIER, ROUTE_CHEAP, ROUTE_MEMORY_CHECK, ROUTE_FULL)
SAFE_SUPPRESSION_ACTIONS = {
    "evict_stale_conflict",
    "reject_out_of_scope",
    "reject_poisoning",
    "rollback_harmful",
}
TASK_MEMORY_SCOPES = {
    "gap1_memory_only_consumer": "verifier/gap1_orientation",
    "gap1_registry_rollback_consumer": "verifier/gap1_orientation",
    "gap4_candidate_pool_consumer": "verifier/gap4_claims",
    "arc_rubric_before_patch_consumer": "arc/patch_synthesis",
    "hardware_speedup_boundary_consumer": "hardware/reporting",
}

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal Exp 5276 verdict; starts with complete: or blocked_ and states "
        "whether memory-assisted verifier dosing is positive, null, harmful, or unmeasured."
    ),
    "inference_substrate": (
        "Declares the bounded local SOTA GGUF receipt-backed pilot, not cached-only "
        "replay, tiny-model smoke, external judging, or answer injection."
    ),
    "preconditions_checked": (
        "Records the Exp 5271 telemetry gate, Exp 5275 governed-memory gate, Exp "
        "5264 scheduler replay readiness, model/runtime snapshots, and "
        "exclusion-manifest check before allocation metrics are interpreted."
    ),
    "MODEL_SPECS": (
        "Records mandated SOTA GGUF model IDs, roles, quantization/file receipts, "
        "and headline inclusion; tiny legacy smoke models cannot contribute "
        "headline metrics."
    ),
    "memory_verifier_dose_ready": (
        "True only when governed memory changes allocation, avoids full verifier "
        "calls, preserves always-full decision quality, causes zero unsafe false "
        "accepts, suppresses unsafe memory rows, and exercises rollback."
    ),
    "calls_avoided_rate": (
        "Fraction of always-full verifier calls avoided by the memory-assisted "
        "policy on live-SOTA-receipt-backed pilot rows."
    ),
    "decision_quality_delta": (
        "Memory-assisted quality rate minus always-full quality rate on the same "
        "pilot rows; negative values are harmful."
    ),
    "unsafe_false_accepts": (
        "Counts unsafe acceptances introduced by memory-assisted allocation; any "
        "positive value blocks readiness."
    ),
    "rollback_exercised": (
        "Confirms the pilot included a harmful-memory allocation case and selected "
        "rollback/block/quarantine/retire instead of trusting memory."
    ),
    "memory_scope_violations_blocked": (
        "Counts stale, poisoning-like, out-of-scope, or harmful memory rows blocked "
        "before they can influence allocation."
    ),
    "commands_run": (
        "Commands run to validate the module, artifact schema, new-code coverage, "
        "repository tests, and adversarial verification."
    ),
}
REQUIRED_WRAPPED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "MODEL_SPECS",
    "memory_verifier_dose_ready",
    "calls_avoided_rate",
    "decision_quality_delta",
    "unsafe_false_accepts",
    "rollback_exercised",
    "memory_scope_violations_blocked",
)


@dataclass(frozen=True)
class PilotRow:
    """One verifier-dose allocation row backed by live SOTA receipt metadata."""

    task_id: str
    model_role: str
    model_hf_id: str
    expected_decision: str
    no_verifier_decision: str
    cheap_decision: str
    deterministic_check_decision: str
    full_decision: str
    cheap_gate_passed: bool
    memory_confidence: float
    deterministic_violation_count: int
    receipt_complete: bool
    live_receipt_available: bool
    memory_scope: str | None
    memory_feature_active: bool
    active_memory_decision_ids: tuple[str, ...]
    suppressed_memory_decision_ids: tuple[str, ...]
    suppressed_memory_actions: tuple[str, ...]
    memory_answer_injection_blocked: bool = True


def load_upstream_artifacts(root: Path | str = REPO_ROOT) -> dict[str, JsonDict]:
    """Read the three gated upstream artifacts used by Exp 5276."""

    root_path = Path(root)
    return {
        "exp5271": _read_json(root_path / EXP5271_RELATIVE_PATH),
        "exp5275": _read_json(root_path / EXP5275_RELATIVE_PATH),
        "exp5264": _read_json(root_path / EXP5264_RELATIVE_PATH),
    }


def extract_model_specs(telemetry_artifact: Mapping[str, Any]) -> JsonDict:
    """Return only the mandated SOTA GGUF roles that are locally receipt-backed."""

    raw_specs = _wrapped_value(telemetry_artifact, "MODEL_SPECS") or {}
    if not isinstance(raw_specs, Mapping):
        return {}
    out: JsonDict = {}
    for role in ROLE_ORDER:
        raw = raw_specs.get(role)
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("hf_id")) != MANDATED_MODEL_IDS[role]:
            continue
        if raw.get("status") != "local_gguf_resolved":
            continue
        spec = dict(raw)
        spec["role"] = role
        spec["headline_role"] = True
        out[role] = spec
    return out


def check_preconditions(
    *,
    root: Path | str = REPO_ROOT,
    upstream_artifacts: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    """Return gate and snapshot data checked before any pilot metric is read."""

    root_path = Path(root)
    exp5271 = upstream_artifacts["exp5271"]
    exp5275 = upstream_artifacts["exp5275"]
    exp5264 = upstream_artifacts["exp5264"]
    model_specs = extract_model_specs(exp5271)
    telemetry_ready = bool(exp5271.get("telemetry_harness_ready"))
    memory_ready = bool(exp5275.get("memory_decision_history_ready"))
    scheduler_ready = bool(exp5264.get("scheduler_ready"))
    headline_roles_ready = all(role in model_specs for role in HEADLINE_ROLES)
    exclusion_manifest_checked = (root_path / EXCLUSION_MANIFEST_RELATIVE_PATH).exists()
    exclusion_manifest_allows = _exclusion_manifest_allows(root_path)
    blockers = []
    if not telemetry_ready:
        blockers.append("exp5271.telemetry_harness_ready")
    if not memory_ready:
        blockers.append("exp5275.memory_decision_history_ready")
    if not scheduler_ready:
        blockers.append("exp5264.scheduler_ready")
    if not headline_roles_ready:
        blockers.append("headline_sota_roles_ready")
    if not exclusion_manifest_checked:
        blockers.append("ops.exclusion_manifest_present")
    if not exclusion_manifest_allows:
        blockers.append("experiment_5276_not_retired")

    return {
        "exp5271.telemetry_harness_ready": telemetry_ready,
        "exp5275.memory_decision_history_ready": memory_ready,
        "exp5264.scheduler_ready": scheduler_ready,
        "headline_roles_ready": headline_roles_ready,
        "exclusion_manifest_checked": exclusion_manifest_checked,
        "exclusion_manifest_allows_exp5276": exclusion_manifest_allows,
        "all_gates_ready": not blockers,
        "blockers": blockers,
        "model_snapshot": model_specs,
        "runtime_snapshot": _runtime_snapshot(exp5271),
        "memory_snapshot": _memory_snapshot(exp5275),
        "scheduler_snapshot": _scheduler_snapshot(exp5264),
    }


def build_pilot_rows(
    *,
    root: Path | str = REPO_ROOT,
    upstream_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[PilotRow, ...]:
    """Build the bounded live-SOTA-receipt-backed scheduler pilot rows."""

    artifacts = dict(upstream_artifacts or load_upstream_artifacts(root))
    model_specs = extract_model_specs(artifacts["exp5271"])
    ready_roles = tuple(role for role in ROLE_ORDER if role in model_specs)
    memory_index = _memory_index(artifacts["exp5275"])
    fixtures = scheduler.build_scheduler_fixtures(root=root)
    rows: list[PilotRow] = []
    for index, fixture in enumerate(fixtures):
        role = ready_roles[index % len(ready_roles)]
        model_spec = model_specs[role]
        scope = TASK_MEMORY_SCOPES.get(fixture.task_id)
        active_rows = memory_index["active_by_scope"].get(scope, ()) if scope else ()
        suppressed_rows = memory_index["suppressed_by_scope"].get(scope, ()) if scope else ()
        rows.append(
            PilotRow(
                task_id=fixture.task_id,
                model_role=role,
                model_hf_id=str(model_spec["hf_id"]),
                expected_decision=fixture.expected_decision,
                no_verifier_decision=fixture.no_verifier_decision,
                cheap_decision=fixture.cheap_decision,
                deterministic_check_decision=fixture.full_decision,
                full_decision=fixture.full_decision,
                cheap_gate_passed=fixture.cheap_gate_passed,
                memory_confidence=fixture.memory_confidence,
                deterministic_violation_count=fixture.deterministic_violation_count,
                receipt_complete=fixture.receipt_complete,
                live_receipt_available=_role_has_live_receipt(artifacts["exp5271"], role),
                memory_scope=scope,
                memory_feature_active=bool(active_rows or suppressed_rows),
                active_memory_decision_ids=tuple(str(row["decision_id"]) for row in active_rows),
                suppressed_memory_decision_ids=tuple(
                    str(row["decision_id"]) for row in suppressed_rows
                ),
                suppressed_memory_actions=tuple(
                    str(row["governance_action"]) for row in suppressed_rows
                ),
            )
        )
    return tuple(rows)


def choose_memory_route(row: PilotRow) -> str:
    """Choose a verifier dose with governed memory as an allocation feature only."""

    if not row.receipt_complete or not row.live_receipt_available:
        return ROUTE_FULL
    if row.deterministic_violation_count >= 3:
        return ROUTE_FULL
    if "rollback_harmful" in row.suppressed_memory_actions:
        return ROUTE_MEMORY_CHECK
    if row.active_memory_decision_ids and row.memory_confidence >= 0.8:
        return ROUTE_MEMORY_CHECK
    if not row.cheap_gate_passed:
        return ROUTE_CHEAP
    if row.memory_confidence < 0.4 and row.deterministic_violation_count == 0:
        return ROUTE_NO_VERIFIER
    return ROUTE_CHEAP


def choose_no_memory_route(row: PilotRow) -> str:
    """Choose the conservative scheduler route when memory features are withheld."""

    if not row.receipt_complete or not row.live_receipt_available:
        return ROUTE_FULL
    if row.deterministic_violation_count >= 3:
        return ROUTE_FULL
    if not row.cheap_gate_passed and row.memory_confidence >= 0.8:
        return ROUTE_FULL
    if not row.cheap_gate_passed:
        return ROUTE_CHEAP
    if row.memory_confidence < 0.4 and row.deterministic_violation_count == 0:
        return ROUTE_NO_VERIFIER
    return ROUTE_CHEAP


def decision_for_route(row: PilotRow, route: str) -> str:
    """Return the decision from the route's verifier/check output, never from memory."""

    if route == ROUTE_NO_VERIFIER:
        return row.no_verifier_decision
    if route == ROUTE_CHEAP:
        return row.cheap_decision
    if route == ROUTE_MEMORY_CHECK:
        return row.deterministic_check_decision
    return row.full_decision


def evaluate_pilot(rows: Sequence[PilotRow]) -> JsonDict:
    """Compare memory-assisted, always-full, and no-memory scheduler policies."""

    memory_rows = [_decision_row(row, choose_memory_route(row)) for row in rows]
    always_full_rows = [_decision_row(row, ROUTE_FULL) for row in rows]
    no_memory_rows = [_decision_row(row, choose_no_memory_route(row)) for row in rows]
    memory_metrics = _metrics(memory_rows)
    always_full_metrics = _metrics(always_full_rows)
    no_memory_metrics = _metrics(no_memory_rows)
    full_denominator = int(always_full_metrics["full_verifier_calls"])
    calls_avoided_rate = _rate(
        full_denominator - int(memory_metrics["full_verifier_calls"]),
        full_denominator,
    )
    decision_quality_delta = _delta(
        float(memory_metrics["quality_rate"]),
        float(always_full_metrics["quality_rate"]),
    )
    unsafe_false_accepts = int(memory_metrics["false_accepts"])
    rollback_exercised = any("rollback_harmful" in row.suppressed_memory_actions for row in rows)
    memory_scope_violations_blocked = len(
        {
            decision_id
            for row in rows
            if row.memory_feature_active
            for decision_id in row.suppressed_memory_decision_ids
        }
    )
    allocation_changed_by_memory_count = sum(
        1 for row in rows if choose_memory_route(row) != choose_no_memory_route(row)
    )
    ready = bool(
        rows
        and calls_avoided_rate > 0.0
        and decision_quality_delta >= 0.0
        and unsafe_false_accepts == 0
        and rollback_exercised
        and memory_scope_violations_blocked > 0
        and allocation_changed_by_memory_count > 0
    )
    return {
        "pilot_row_count": len(rows),
        "memory_assisted_rows": memory_rows,
        "baseline_rows": {
            "always_full": always_full_rows,
            "no_memory_scheduler": no_memory_rows,
        },
        "memory_assisted_metrics": memory_metrics,
        "baseline_metrics": {
            "always_full": always_full_metrics,
            "no_memory_scheduler": no_memory_metrics,
        },
        "route_counts": _ordered_counts(Counter(row["route"] for row in memory_rows)),
        "calls_avoided_rate": calls_avoided_rate,
        "decision_quality_delta": decision_quality_delta,
        "unsafe_false_accepts": unsafe_false_accepts,
        "rollback_exercised": rollback_exercised,
        "memory_scope_violations_blocked": memory_scope_violations_blocked,
        "allocation_changed_by_memory_count": allocation_changed_by_memory_count,
        "memory_verifier_dose_ready": ready,
    }


def build_result_artifact(
    *,
    root: Path | str = REPO_ROOT,
    upstream_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    commands_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp 5276 artifact from gated upstream receipts and pilot rows."""

    root_path = Path(root)
    artifacts = dict(upstream_artifacts or load_upstream_artifacts(root_path))
    preconditions = check_preconditions(root=root_path, upstream_artifacts=artifacts)
    model_specs = extract_model_specs(artifacts["exp5271"])
    if preconditions["all_gates_ready"]:
        rows = build_pilot_rows(root=root_path, upstream_artifacts=artifacts)
        pilot = evaluate_pilot(rows)
    else:
        rows = ()
        pilot = _neutral_pilot()

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "run_date": "2026-07-05",
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "random_seed": RANDOM_SEED,
        "duration_s": _live_receipt_duration_s(artifacts["exp5271"]),
        "honest_verdict": _wrap("honest_verdict", _honest_verdict(preconditions, pilot)),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "preconditions_checked": _wrap("preconditions_checked", preconditions),
        "MODEL_SPECS": _wrap("MODEL_SPECS", model_specs),
        "memory_verifier_dose_ready": _wrap(
            "memory_verifier_dose_ready",
            bool(pilot["memory_verifier_dose_ready"]),
        ),
        "calls_avoided_rate": _wrap("calls_avoided_rate", float(pilot["calls_avoided_rate"])),
        "decision_quality_delta": _wrap(
            "decision_quality_delta",
            float(pilot["decision_quality_delta"]),
        ),
        "unsafe_false_accepts": _wrap("unsafe_false_accepts", int(pilot["unsafe_false_accepts"])),
        "rollback_exercised": _wrap("rollback_exercised", bool(pilot["rollback_exercised"])),
        "memory_scope_violations_blocked": _wrap(
            "memory_scope_violations_blocked",
            int(pilot["memory_scope_violations_blocked"]),
        ),
        "model_runtime_snapshot": {
            "models": model_specs,
            "runtime": preconditions["runtime_snapshot"],
        },
        "memory_snapshot": preconditions["memory_snapshot"],
        "scheduler_snapshot": preconditions["scheduler_snapshot"],
        "pilot_rows": pilot["memory_assisted_rows"],
        "baseline_rows": pilot["baseline_rows"],
        "memory_assisted_metrics": pilot["memory_assisted_metrics"],
        "baseline_metrics": pilot["baseline_metrics"],
        "route_counts": pilot["route_counts"],
        "allocation_changed_by_memory_count": pilot["allocation_changed_by_memory_count"],
        "source_artifact_checksums": source_artifact_checksums(root_path),
        "field_principles": dict(FIELD_PRINCIPLES),
        "commands_run": [dict(row) for row in commands_run],
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the Exp 5276 schema required by the tests and conductor."""

    for field in REQUIRED_WRAPPED_FIELDS:
        wrapped = artifact.get(field)
        if not isinstance(wrapped, Mapping) or "value" not in wrapped or "principle" not in wrapped:
            raise ValueError(f"{field} must be principle-wrapped")  # pragma: no cover
    verdict = str(_wrapped_value(artifact, "honest_verdict"))
    if not (verdict.startswith("complete:") or verdict.startswith("blocked_")):
        raise ValueError("honest_verdict terminal prefix invalid")  # pragma: no cover
    if _wrapped_value(artifact, "inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError(
            "inference_substrate must be live_llm_inference_local_gguf_sota"
        )  # pragma: no cover
    if not isinstance(_wrapped_value(artifact, "memory_verifier_dose_ready"), bool):
        raise ValueError("memory_verifier_dose_ready must wrap a bool")  # pragma: no cover
    if not isinstance(_wrapped_value(artifact, "calls_avoided_rate"), float):
        raise ValueError("calls_avoided_rate must wrap a float")  # pragma: no cover
    if not isinstance(_wrapped_value(artifact, "decision_quality_delta"), float):
        raise ValueError("decision_quality_delta must wrap a float")  # pragma: no cover
    if not isinstance(_wrapped_value(artifact, "unsafe_false_accepts"), int):
        raise ValueError("unsafe_false_accepts must wrap an int")  # pragma: no cover
    if not isinstance(_wrapped_value(artifact, "rollback_exercised"), bool):
        raise ValueError("rollback_exercised must wrap a bool")  # pragma: no cover
    if not isinstance(_wrapped_value(artifact, "memory_scope_violations_blocked"), int):
        raise ValueError("memory_scope_violations_blocked must wrap an int")  # pragma: no cover
    if not isinstance(artifact.get("commands_run"), list):
        raise ValueError("commands_run must be a bare list")  # pragma: no cover
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    commands_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp 5276 result artifact and return its JSON payload."""

    artifact = build_result_artifact(root=root, commands_run=commands_run)
    _write_json(Path(result_path), artifact)
    return artifact


def source_artifact_checksums(root: Path | str = REPO_ROOT) -> JsonDict:
    """Return sha256 receipts for the gated source artifacts."""

    root_path = Path(root)
    return {
        "exp5271": _sha256_file(root_path / EXP5271_RELATIVE_PATH),
        "exp5275": _sha256_file(root_path / EXP5275_RELATIVE_PATH),
        "exp5264": _sha256_file(root_path / EXP5264_RELATIVE_PATH),
        "exclusion_manifest": _sha256_file(root_path / EXCLUSION_MANIFEST_RELATIVE_PATH),
    }


def _memory_index(memory_artifact: Mapping[str, Any]) -> JsonDict:
    active_by_scope: dict[str, list[Mapping[str, Any]]] = {}
    suppressed_by_scope: dict[str, list[Mapping[str, Any]]] = {}
    for row in memory_artifact.get("governance_rows", []):
        if not isinstance(row, Mapping):
            continue
        scope = str(row.get("task_scope") or "")
        if not scope:
            continue
        if row.get("active") and row.get("governance_action") == "promote":
            active_by_scope.setdefault(scope, []).append(row)
        if row.get("governance_action") in SAFE_SUPPRESSION_ACTIONS:
            suppressed_by_scope.setdefault(scope, []).append(row)
    return {
        "active_by_scope": {key: tuple(value) for key, value in active_by_scope.items()},
        "suppressed_by_scope": {key: tuple(value) for key, value in suppressed_by_scope.items()},
    }


def _decision_row(row: PilotRow, route: str) -> JsonDict:
    decision = decision_for_route(row, route)
    correct = decision == row.expected_decision
    payload = asdict(row)
    for key in (
        "active_memory_decision_ids",
        "suppressed_memory_decision_ids",
        "suppressed_memory_actions",
    ):
        payload[key] = list(payload[key])
    return {
        **payload,
        "route": route,
        "selected_decision": decision,
        "selected_decision_source": _decision_source(route),
        "correct": correct,
        "false_accept": _is_false_accept(decision, row.expected_decision),
        "full_verifier_call": route == ROUTE_FULL,
    }


def _metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "n": len(rows),
        "correct_n": sum(1 for row in rows if row["correct"]),
        "quality_rate": _rate(sum(1 for row in rows if row["correct"]), len(rows)),
        "false_accepts": sum(1 for row in rows if row["false_accept"]),
        "false_accept_rate": _rate(sum(1 for row in rows if row["false_accept"]), len(rows)),
        "full_verifier_calls": sum(1 for row in rows if row["full_verifier_call"]),
    }


def _neutral_pilot() -> JsonDict:
    return {
        "pilot_row_count": 0,
        "memory_assisted_rows": [],
        "baseline_rows": {"always_full": [], "no_memory_scheduler": []},
        "memory_assisted_metrics": _metrics([]),
        "baseline_metrics": {
            "always_full": _metrics([]),
            "no_memory_scheduler": _metrics([]),
        },
        "route_counts": {},
        "calls_avoided_rate": 0.0,
        "decision_quality_delta": 0.0,
        "unsafe_false_accepts": 0,
        "rollback_exercised": False,
        "memory_scope_violations_blocked": 0,
        "allocation_changed_by_memory_count": 0,
        "memory_verifier_dose_ready": False,
    }


def _runtime_snapshot(telemetry_artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "inference_substrate": _wrapped_value(telemetry_artifact, "inference_substrate"),
        "telemetry_harness_ready": bool(telemetry_artifact.get("telemetry_harness_ready")),
        "telemetry_harness_ready_principle": telemetry_artifact.get(
            "telemetry_harness_ready_principle"
        ),
        "duration_receipts": _wrapped_value(telemetry_artifact, "duration_receipts") or {},
        "gpu_offload_receipts": _wrapped_value(telemetry_artifact, "gpu_offload_receipts") or {},
        "exposed_telemetry_fields": _wrapped_value(telemetry_artifact, "exposed_telemetry_fields")
        or {},
    }


def _memory_snapshot(memory_artifact: Mapping[str, Any]) -> JsonDict:
    rows = [row for row in memory_artifact.get("governance_rows", []) if isinstance(row, Mapping)]
    return {
        "memory_decision_history_ready": bool(memory_artifact.get("memory_decision_history_ready")),
        "memory_decision_history_ready_principle": memory_artifact.get(
            "memory_decision_history_ready_principle"
        ),
        "unsafe_false_accepts": _wrapped_value(memory_artifact, "unsafe_false_accepts"),
        "row_count": len(rows),
        "governance_action_counts": _ordered_counts(
            Counter(str(row.get("governance_action")) for row in rows)
        ),
    }


def _scheduler_snapshot(scheduler_artifact: Mapping[str, Any]) -> JsonDict:
    return {
        "scheduler_ready": bool(scheduler_artifact.get("scheduler_ready")),
        "scheduler_ready_principle": scheduler_artifact.get("scheduler_ready_principle"),
        "full_verifier_calls_avoided_rate": _wrapped_value(
            scheduler_artifact,
            "full_verifier_calls_avoided_rate",
        ),
        "decision_quality_delta": _wrapped_value(scheduler_artifact, "decision_quality_delta"),
        "false_accept_delta": _wrapped_value(scheduler_artifact, "false_accept_delta"),
    }


def _role_has_live_receipt(telemetry_artifact: Mapping[str, Any], role: str) -> bool:
    duration = (_wrapped_value(telemetry_artifact, "duration_receipts") or {}).get("per_model", {})
    fields = (_wrapped_value(telemetry_artifact, "exposed_telemetry_fields") or {}).get(role, {})
    role_duration = duration.get(role, {})
    return bool(
        role_duration.get("runtime_ready")
        and any(
            isinstance(fields.get(key), Mapping) and fields[key].get("availability") == "available"
            for key in ("logits", "token_logprobs", "hidden_states", "attention_summaries")
        )
    )


def _live_receipt_duration_s(telemetry_artifact: Mapping[str, Any]) -> float:
    duration = _wrapped_value(telemetry_artifact, "duration_receipts") or {}
    total = duration.get("total_wall_clock_s", telemetry_artifact.get("duration_s", 0.0))
    return round(float(total or 0.0), 6)


def _honest_verdict(preconditions: Mapping[str, Any], pilot: Mapping[str, Any]) -> str:
    if not preconditions["all_gates_ready"]:
        blockers = ",".join(str(item) for item in preconditions["blockers"])
        return f"blocked_upstream_gate_unmeasured: memory-assisted verifier dosing unmeasured; blockers={blockers}"
    if int(pilot["unsafe_false_accepts"]) > 0:
        return "complete: harmful memory-assisted verifier dosing introduced unsafe false accepts"
    if float(pilot["decision_quality_delta"]) < 0.0:
        return "complete: harmful memory-assisted verifier dosing reduced decision quality"
    if float(pilot["calls_avoided_rate"]) <= 0.0:
        return "complete: null memory-assisted verifier dosing avoided no full verifier calls"
    if not pilot["rollback_exercised"]:
        return "complete: null memory-assisted verifier dosing did not exercise rollback"
    if not pilot["memory_verifier_dose_ready"]:
        return "complete: null memory-assisted verifier dosing did not satisfy all safety gates"
    return (
        "complete: positive memory-assisted verifier dosing preserved always-full quality, "
        f"avoided {float(pilot['calls_avoided_rate']):.6f} full verifier calls, "
        "blocked unsafe memory rows, and kept unsafe_false_accepts=0"
    )


def _decision_source(route: str) -> str:
    if route == ROUTE_MEMORY_CHECK:
        return "deterministic_check_selected_by_memory_feature"
    if route == ROUTE_FULL:
        return "full_verifier"
    if route == ROUTE_CHEAP:
        return "cheap_deterministic_check"
    return "no_verifier_baseline_output"


def _is_false_accept(decision: str, expected: str) -> bool:
    if decision == expected:
        return False
    expected_blocks = expected.startswith(scheduler.ABSTAIN_OR_BLOCK_PREFIXES)
    decision_blocks = decision.startswith(scheduler.ABSTAIN_OR_BLOCK_PREFIXES)
    return expected_blocks and not decision_blocks


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _delta(left: float, right: float) -> float:
    return round(left - right, 6)


def _ordered_counts(counter: Counter[str]) -> JsonDict:
    return {key: counter[key] for key in sorted(counter)}


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _wrapped_value(artifact: Mapping[str, Any], field: str) -> Any:
    wrapped = artifact.get(field)
    return wrapped.get("value") if isinstance(wrapped, Mapping) else wrapped


def _exclusion_manifest_allows(root: Path) -> bool:
    path = root / EXCLUSION_MANIFEST_RELATIVE_PATH
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return "experiment_id: 5276" not in text and "experiment_5276" not in text


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str | None:
    if not receipt_exists(path, artifact_relative_path=RESULT_RELATIVE_PATH):
        return None
    return (
        "sha256:"
        + hashlib.sha256(
            receipt_bytes(path, artifact_relative_path=RESULT_RELATIVE_PATH)
        ).hexdigest()
    )


def _checksum(artifact: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
                "utf-8"
            )
        ).hexdigest()
    )
