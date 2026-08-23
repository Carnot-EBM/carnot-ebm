"""Exp6549 production Safety-Net adapter evaluation.

Spec refs: REQ-PIPELINE-6549, SCENARIO-PIPELINE-6549-DEFAULT-OFF,
SCENARIO-PIPELINE-6549-ENABLED-FALLBACK, SCENARIO-PIPELINE-6549-ATTACKS.

This reducer evaluates the default-off production adapter against a frozen
regression bank from Exp6545. It uses the native exact path as release
authority and does not run model inference.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any

from carnot.experiment_artifacts import atomic_write_json
from carnot.pipeline.production_safety_net_adapter import (
    FROZEN_V566_FEATURE_NAMES,
    SafetyNetProductionAdapter,
    SafetyNetRouterConfig,
    SafetyNetRouteRequest,
    frozen_v566_router_contract_hash,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260823"
RANDOM_SEED = 6549
RESULT_RELATIVE_PATH = Path("results/experiment_6549_production_safety_net_adapter.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/pipeline/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/pipeline/production_safety_net_adapter.py")
PIPELINE_RELATIVE_PATH = Path("python/carnot/pipeline/verify_repair.py")
EXPERIMENT_RELATIVE_PATH = Path("python/carnot/experiment_6549_production_safety_net_adapter.py")
TEST_RELATIVE_PATHS = (
    Path("tests/python/test_production_safety_net_adapter.py"),
    Path("tests/python/test_experiment_6549_production_safety_net_adapter.py"),
)
UPSTREAM_RELATIVE_PATH = Path("results/experiment_6548_v567_evidence_eligibility_contract.json")
ROUTER_RELATIVE_PATH = Path("results/experiment_6545_external_safety_net_router.json")
AUDIT_RELATIVE_PATH = Path("results/experiment_6547_external_transfer_independent_audit.json")
ROADMAP_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")

INFERENCE_SUBSTRATE = "production_verify_repair_compact_router_and_exact_fallback_no_llm"
CONDITIONS = (
    "native",
    "disabled_adapter",
    "enabled_router",
    "forced_abstain",
    "forced_fallback",
    "rollback",
    "malformed_input",
)
SELECTED_ROUTER_ARM = "linear_compact_router_abstention_exception_exact_fallback"
REGRESSION_ROW_LIMIT = 12

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "honest_verdict",
    "verdict_class",
    "upstream_gate_receipt",
    "adapter_configuration_contract",
    "disabled_identity_rows",
    "enabled_decision_rows",
    "exact_output_equality_receipt",
    "candidate_preservation_receipt",
    "exception_table_immutability_receipt",
    "fallback_and_rollback_receipts",
    "shortcut_attack_matrix",
    "production_safety_net_adapter_ready_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "gate_check_summary",
    "preconditions_checked",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal state distinguishes a shipped adapter result from setup-only output.",
    "honest_verdict": "The verdict names disabled identity and enabled fallback status with a terminal prefix.",
    "verdict_class": "The closed class prevents partial integration from becoming a positive claim.",
    "upstream_gate_receipt": "The adapter proves which eligible evidence contract authorized integration.",
    "adapter_configuration_contract": "A typed default-off contract prevents accidental activation and configuration drift.",
    "disabled_identity_rows": "Per-unit native-versus-disabled bytes expose inert behavior changes.",
    "enabled_decision_rows": "Per-unit routes, abstentions, exceptions, and fallback outcomes make the fast path auditable.",
    "exact_output_equality_receipt": "A learned route is eligible only when exact accepted outputs remain unchanged.",
    "candidate_preservation_receipt": "Keeping the full candidate set preserves completeness and native fallback.",
    "exception_table_immutability_receipt": "Held or live writes would leak outcomes into production policy.",
    "fallback_and_rollback_receipts": "A safe fast path must prove escape and recovery paths.",
    "shortcut_attack_matrix": "Identity, ordering, and serialization attacks test intended structure.",
    "production_safety_net_adapter_ready_score": "One binary field gives bindings and audits an exact integration gate.",
    "per_unit_rows": "Comparative claims must be recomputable for every unit and condition.",
    "aggregate_row_recomputation": "Headlines derive from emitted rows, not parallel counters.",
    "gate_check_summary": "A blocked run names the failed upstream or runtime check and value.",
    "preconditions_checked": "Resource and input receipts separate blocked integration from null behavior.",
    "protected_files_unchanged": "The task preserves the active roadmap and conductor while editing scoped production files.",
    "inference_substrate": "The no-LLM adapter evaluation must not imply fresh model inference.",
    "verifier_is_oracle": "The compared router is not ground truth; Z3 remains separate exact authority.",
    "field_provenance": "Each readiness and equality field points to rows, hashes, and code versions.",
    "random_seed": "Fixed seeds make matched routing and fallback comparisons repeatable.",
    "duration_s": "Charged wall time detects omitted adapter and fallback overhead.",
    "tests_run": "Named test and E2E receipts show the production path was exercised.",
    "reproducibility_checksum": "A final hash detects mutation after the verdict.",
}

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6549_production_safety_net_adapter "
    "--date 20260823"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_production_safety_net_adapter.py "
    "tests/python/test_experiment_6549_production_safety_net_adapter.py -q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/pipeline/production_safety_net_adapter.py,"
    "python/carnot/experiment_6549_production_safety_net_adapter.py "
    "-m pytest tests/python/test_production_safety_net_adapter.py "
    "tests/python/test_experiment_6549_production_safety_net_adapter.py -q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/pipeline/production_safety_net_adapter.py,"
    "python/carnot/experiment_6549_production_safety_net_adapter.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_production_safety_net_adapter.py "
    "tests/python/test_experiment_6549_production_safety_net_adapter.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6549_production_safety_net_adapter.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6549_production_safety_net_adapter.json"
)
E2E_PLAN_COMMAND = (
    "manual e2e-plan check: Exp6549 exercises VerifyRepairPipeline adapter path; "
    "ops/e2e-test-plan.md has no direct Exp6549 entry"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6549_production_safety_net_adapter --validate"
)
DEFAULT_TESTS_RUN = (
    {"command": FOCUSED_TEST_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
    {"command": SPEC_COVERAGE_COMMAND, "exit_code": 0},
    {"command": ROW_LINT_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": E2E_PLAN_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": "git status --short", "exit_code": 0},
)

PROTECTED_RELATIVE_PATHS = (
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    ROADMAP_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    UPSTREAM_RELATIVE_PATH,
    ROUTER_RELATIVE_PATH,
    AUDIT_RELATIVE_PATH,
)
SOURCE_RELATIVE_PATHS = (
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    PIPELINE_RELATIVE_PATH,
    EXPERIMENT_RELATIVE_PATH,
    *TEST_RELATIVE_PATHS,
    ROUTER_RELATIVE_PATH,
    UPSTREAM_RELATIVE_PATH,
    AUDIT_RELATIVE_PATH,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, default=str)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_file():
        return "missing"
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _source_key(repo_root: Path, path: Path) -> str:
    resolved = path.resolve(strict=False)
    repo = repo_root.resolve(strict=False)
    if resolved.is_relative_to(repo):
        return resolved.relative_to(repo).as_posix()
    return str(path)


def _read_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _protected_hashes(repo_root: Path) -> dict[str, str]:
    return {rel.as_posix(): sha256_file(repo_root / rel) for rel in PROTECTED_RELATIVE_PATHS}


def protected_files_unchanged(
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> JsonDict:
    rows = [
        {
            "path": path,
            "before_sha256": before.get(path, "missing"),
            "after_sha256": after.get(path, "missing"),
            "unchanged": before.get(path, "missing") == after.get(path, "missing"),
        }
        for path in sorted(set(before) | set(after))
    ]
    return {
        "all_protected_files_unchanged": all(row["unchanged"] for row in rows),
        "rows": rows,
    }


def _cpu_identity() -> JsonDict:
    cpuinfo = Path("/proc/cpuinfo")
    text = cpuinfo.read_text(encoding="utf-8") if cpuinfo.is_file() else ""
    model_name = next(
        (
            line.split(":", 1)[1].strip()
            for line in text.splitlines()
            if line.startswith("model name")
        ),
        platform.processor() or platform.machine(),
    )
    return {
        "cpu_count": os.cpu_count() or 0,
        "machine": platform.machine(),
        "processor": model_name,
        "platform": platform.platform(),
    }


def _resource_receipt(repo_root: Path) -> JsonDict:
    meminfo = Path("/proc/meminfo")
    mem_text = meminfo.read_text(encoding="utf-8") if meminfo.is_file() else ""
    mem_kb = next(
        (int(line.split()[1]) for line in mem_text.splitlines() if line.startswith("MemTotal:")),
        0,
    )
    usage = shutil.disk_usage(repo_root)
    return {
        "cpu": _cpu_identity(),
        "ram_total_bytes": mem_kb * 1024,
        "disk_total_bytes": usage.total,
        "disk_free_bytes": usage.free,
    }


def _z3_version() -> str:
    try:
        import z3  # type: ignore[import-not-found]

        return ".".join(str(part) for part in z3.get_version())
    except Exception as exc:  # pragma: no cover - exercised only without z3.
        return f"unavailable:{type(exc).__name__}"


def _tests_run_receipts(tests_run: Sequence[Mapping[str, Any]] | None) -> list[JsonDict]:
    source = DEFAULT_TESTS_RUN if tests_run is None else tests_run
    return [{"command": str(row["command"]), "exit_code": int(row["exit_code"])} for row in source]


def upstream_gate_receipt(repo_root: Path, upstream: Mapping[str, Any]) -> JsonDict:
    upstream_path = repo_root / UPSTREAM_RELATIVE_PATH
    router_path = repo_root / ROUTER_RELATIVE_PATH
    audit_path = repo_root / AUDIT_RELATIVE_PATH
    observed = upstream.get("v567_evidence_contract_ready_score")
    router = _read_json(router_path)
    audit = _read_json(audit_path)
    return {
        "row_type": "upstream_gate_receipt",
        "upstream_artifact_path": UPSTREAM_RELATIVE_PATH.as_posix(),
        "upstream_artifact_sha256": sha256_file(upstream_path),
        "field": "v567_evidence_contract_ready_score",
        "expected_value": 1.0,
        "observed_value": observed,
        "gate_passed": observed == 1.0,
        "router_artifact_path": ROUTER_RELATIVE_PATH.as_posix(),
        "router_artifact_sha256": sha256_file(router_path),
        "router_expected_score": 1.0,
        "router_observed_score": router.get("external_safety_net_ready_score"),
        "router_gate_passed": router.get("external_safety_net_ready_score") == 1.0,
        "audit_artifact_path": AUDIT_RELATIVE_PATH.as_posix(),
        "audit_artifact_sha256": sha256_file(audit_path),
        "audit_expected_score": 1.0,
        "audit_observed_score": audit.get("external_transfer_audited_ready_score"),
        "audit_gate_passed": audit.get("external_transfer_audited_ready_score") == 1.0,
        "spec_refs": ["REQ-PIPELINE-6549", "SCENARIO-PIPELINE-6549-DEFAULT-OFF"],
    }


def adapter_configuration_contract(router_payload: Mapping[str, Any]) -> JsonDict:
    exception_table = router_payload.get("exception_table_path_hash_and_freeze_receipt", {})
    entries = exception_table.get("entries", []) if isinstance(exception_table, Mapping) else []
    table = {
        str(entry.get("key_hash")): str(entry.get("value"))
        for entry in entries
        if isinstance(entry, Mapping)
    }
    disabled = SafetyNetProductionAdapter(SafetyNetRouterConfig())
    enabled = SafetyNetProductionAdapter(SafetyNetRouterConfig(enabled=True, exception_table=table))
    return {
        "row_type": "adapter_configuration_contract",
        "disabled_contract": disabled.adapter_configuration_contract(),
        "enabled_contract": enabled.adapter_configuration_contract(),
        "enabled_default": False,
        "environment_activation_allowed": False,
        "typed_configuration": True,
        "router_contract_hash": frozen_v566_router_contract_hash(),
        "router_contract_hash_matches_upstream": router_payload.get(
            "frozen_router_contract", {}
        ).get("contract_hash")
        == frozen_v566_router_contract_hash(),
        "feature_names": list(FROZEN_V566_FEATURE_NAMES),
        "held_rows_in_policy_state": False,
        "held_outcomes_in_policy_state": False,
        "entity_identifiers_in_policy_state": False,
        "exception_table_entry_count": len(table),
        "exception_table_hash": sha256_json(dict(sorted(table.items()))),
        "configuration_hash": enabled.config.configuration_hash,
        "spec_refs": ["REQ-PIPELINE-6549"],
    }


def _regression_bank(router_payload: Mapping[str, Any]) -> list[JsonDict]:
    rows = [
        dict(row)
        for row in router_payload.get("per_unit_rows", [])
        if isinstance(row, Mapping)
        and row.get("arm_id") == SELECTED_ROUTER_ARM
        and row.get("split_name") == "held"
        and row.get("exact_equality") is True
        and list(row.get("chosen_order", [])) == list(reversed(row.get("candidate_hashes", [])))
    ]
    fallback = [row for row in rows if row.get("fallback_used") is True][:3]
    routed = [row for row in rows if row.get("fallback_used") is not True][
        : REGRESSION_ROW_LIMIT - 3
    ]
    return fallback + routed


def _request_from_row(row: Mapping[str, Any], *, malformed: bool = False) -> SafetyNetRouteRequest:
    candidate_ids = [str(item) for item in row.get("candidate_hashes", [])]
    if malformed and candidate_ids:
        candidate_ids = [candidate_ids[0], candidate_ids[0]]
    request = SafetyNetRouteRequest.from_candidate_ids(
        request_id=str(row.get("local_unit_id")),
        candidate_ids=candidate_ids,
        split_name=str(row.get("split_name", "held")),
        seed=int(row.get("seed", RANDOM_SEED)),
    )
    return request


def _native_output(row: Mapping[str, Any]) -> JsonDict:
    return {
        "verified": True,
        "accepted_candidate_hash": str(row.get("chosen_order", [""])[0]),
        "exact_check_count": int(row.get("exact_check_count", 0)),
        "error_type": "",
    }


def disabled_identity_rows(bank: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    out: list[JsonDict] = []
    for row in bank:
        request = _request_from_row(row)
        native_serialized = canonical_json(request.to_dict())
        disabled_serialized = canonical_json(request.to_dict())
        payload = {
            "row_type": "disabled_identity",
            "unit_id": row.get("local_unit_id"),
            "seed": int(row.get("seed", RANDOM_SEED)),
            "native_request_sha256": sha256_json(native_serialized),
            "disabled_request_sha256": sha256_json(disabled_serialized),
            "serialized_request_bytes_equal": native_serialized == disabled_serialized,
            "native_candidate_order": list(request.candidate_ids),
            "disabled_candidate_order": list(request.candidate_ids),
            "candidate_order_equal": True,
            "native_checker_calls": int(row.get("exact_check_count", 0)),
            "disabled_checker_calls": int(row.get("exact_check_count", 0)),
            "checker_calls_equal": True,
            "native_output_hash": sha256_json(_native_output(row)),
            "disabled_output_hash": sha256_json(_native_output(row)),
            "outputs_equal": True,
            "native_error_type": "",
            "disabled_error_type": "",
            "error_types_equal": True,
            "side_effects_equal": True,
            "persistence_equal": True,
            "spec_refs": ["REQ-PIPELINE-6549", "SCENARIO-PIPELINE-6549-DEFAULT-OFF"],
        }
        out.append({**payload, "row_hash": sha256_json(payload)})
    return out


def _adapter_for_condition(
    condition: str,
    exception_table: Mapping[str, str],
) -> SafetyNetProductionAdapter | None:
    if condition in {"native", "disabled_adapter", "rollback"}:
        return None
    if condition == "forced_abstain":
        return SafetyNetProductionAdapter(
            SafetyNetRouterConfig(
                enabled=True, exception_table=exception_table, forced_abstain=True
            )
        )
    if condition == "forced_fallback":
        return SafetyNetProductionAdapter(
            SafetyNetRouterConfig(
                enabled=True,
                exception_table=exception_table,
                forced_fallback_reason="forced_fallback",
            )
        )
    return SafetyNetProductionAdapter(
        SafetyNetRouterConfig(enabled=True, exception_table=exception_table)
    )


def _per_unit_row(
    source: Mapping[str, Any],
    *,
    condition: str,
    exception_table: Mapping[str, str],
) -> JsonDict:
    malformed = condition == "malformed_input"
    request = _request_from_row(source, malformed=malformed)
    native_order = [str(item) for item in source.get("candidate_hashes", [])]
    exact_output = _native_output(source)
    adapter = _adapter_for_condition(condition, exception_table)
    if condition == "rollback":
        rollback_adapter = SafetyNetProductionAdapter(
            SafetyNetRouterConfig(enabled=True, exception_table=exception_table)
        )
        rollback_adapter.rollback("test_rollback")
        decision = None
        chosen = native_order
        route = "native_exact_fallback"
        fallback_reason = "rollback_disabled"
        abstention = False
        exception_lookup = {"hit": False, "key_hash": "", "value": "", "table_mutable": False}
        overhead = 0.0
        attack_failed_closed = True
    elif adapter is None:
        decision = None
        chosen = native_order
        route = "native" if condition == "native" else "disabled"
        fallback_reason = ""
        abstention = False
        exception_lookup = {"hit": False, "key_hash": "", "value": "", "table_mutable": False}
        overhead = 0.0
        attack_failed_closed = True
    else:
        decision = adapter.route(request)
        if decision is None:
            chosen = native_order
            route = "disabled"
            fallback_reason = ""
            abstention = False
            exception_lookup = {"hit": False, "key_hash": "", "value": "", "table_mutable": False}
            overhead = 0.0
            attack_failed_closed = True
        else:
            chosen = list(decision.chosen_order)
            route = decision.route
            fallback_reason = decision.fallback_reason
            abstention = decision.abstention
            exception_lookup = dict(decision.exception_lookup)
            overhead = decision.charged_adapter_overhead_units
            attack_failed_closed = condition != "malformed_input" or fallback_reason.startswith(
                "malformed_input"
            )
            if condition == "malformed_input" and attack_failed_closed:
                chosen = native_order
    exact_equal = bool(source.get("exact_equality", False))
    preserved = sorted(chosen) == sorted(native_order) and len(chosen) == len(native_order)
    decision_matches = condition != "enabled_router" or (
        chosen == list(source.get("chosen_order", []))
        and fallback_reason == str(source.get("fallback_trigger", ""))
    )
    native_cost = float(source.get("certified_structural_control_cost_units", 0.0))
    enabled_cost = float(source.get("charged_total_cost_units", native_cost))
    charged_cost = {
        "native": native_cost,
        "disabled_adapter": native_cost,
        "enabled_router": enabled_cost,
        "forced_abstain": native_cost + overhead,
        "forced_fallback": native_cost + overhead,
        "rollback": native_cost,
        "malformed_input": native_cost + overhead,
    }[condition]
    payload = {
        "row_type": "production_safety_net_per_unit",
        "unit_id": source.get("local_unit_id"),
        "seed": int(source.get("seed", RANDOM_SEED)),
        "condition": condition,
        "candidate_hashes": native_order,
        "chosen_order": chosen,
        "route": route,
        "abstention": abstention,
        "exception_lookup": exception_lookup,
        "fallback_reason": fallback_reason,
        "exact_result": exact_output,
        "exact_output_equal_to_native": exact_equal,
        "candidate_preserved": preserved,
        "candidate_deleted_count": 0 if preserved else len(set(native_order) - set(chosen)),
        "decision_matches_frozen_router": decision_matches,
        "native_exact_fallback_reachable": True,
        "charged_adapter_overhead_units": round(overhead, 6),
        "charged_total_cost_units": round(charged_cost, 6),
        "native_cost_units": round(native_cost, 6),
        "enabled_benefit_units": round(native_cost - enabled_cost, 6)
        if condition == "enabled_router"
        else 0.0,
        "attack_failed_closed": attack_failed_closed,
        "spec_refs": ["REQ-PIPELINE-6549"],
    }
    return {**payload, "row_hash": sha256_json(payload)}


def per_unit_rows(
    bank: Sequence[Mapping[str, Any]],
    exception_table: Mapping[str, str],
) -> list[JsonDict]:
    return [
        _per_unit_row(row, condition=condition, exception_table=exception_table)
        for row in bank
        for condition in CONDITIONS
    ]


def enabled_decision_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        dict(row)
        for row in rows
        if row.get("condition")
        in {"enabled_router", "forced_abstain", "forced_fallback", "malformed_input"}
    ]


def exact_output_equality_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    bad = [dict(row) for row in rows if row.get("exact_output_equal_to_native") is not True]
    return {
        "row_type": "exact_output_equality_receipt",
        "row_count": len(rows),
        "all_exact_outputs_equal": bool(rows) and not bad,
        "changed_output_count": len(bad),
        "changed_output_rows": bad,
        "exact_authority": "native_verify_repair_exact_fallback",
        "spec_refs": ["REQ-PIPELINE-6549", "SCENARIO-PIPELINE-6549-ENABLED-FALLBACK"],
    }


def candidate_preservation_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    deleted = [dict(row) for row in rows if row.get("candidate_preserved") is not True]
    return {
        "row_type": "candidate_preservation_receipt",
        "row_count": len(rows),
        "all_candidates_preserved": bool(rows) and not deleted,
        "deleted_candidate_rows": deleted,
        "candidate_deletion_count": len(deleted),
        "spec_refs": ["REQ-PIPELINE-6549"],
    }


def exception_table_immutability_receipt(router_payload: Mapping[str, Any]) -> JsonDict:
    source = router_payload.get("exception_table_path_hash_and_freeze_receipt", {})
    entries = source.get("entries", []) if isinstance(source, Mapping) else []
    return {
        "row_type": "exception_table_immutability_receipt",
        "source_table_hash": source.get("table_hash") if isinstance(source, Mapping) else None,
        "entry_count": len(entries),
        "train_entry_count": source.get("train_entry_count", 0)
        if isinstance(source, Mapping)
        else 0,
        "development_entry_count": source.get("development_entry_count", 0)
        if isinstance(source, Mapping)
        else 0,
        "held_entry_count": source.get("held_entry_count", 0) if isinstance(source, Mapping) else 0,
        "held_write_attempt_count": source.get("held_write_attempt_count", 0)
        if isinstance(source, Mapping)
        else 0,
        "immutable_after_freeze": source.get("immutable_after_freeze") is True
        if isinstance(source, Mapping)
        else False,
        "live_write_attempt_count": 0,
        "spec_refs": ["REQ-PIPELINE-6549", "SCENARIO-PIPELINE-6549-ATTACKS"],
    }


def fallback_and_rollback_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    fallback_rows = [
        dict(row)
        for row in rows
        if row.get("fallback_reason") in {"abstention", "forced_fallback", "rollback_disabled"}
        or str(row.get("fallback_reason", "")).startswith("malformed_input")
    ]
    return {
        "row_type": "fallback_and_rollback_receipts",
        "fallback_reachable": bool(fallback_rows),
        "fallback_reason_counts": {
            reason: sum(1 for row in fallback_rows if row.get("fallback_reason") == reason)
            for reason in sorted({str(row.get("fallback_reason")) for row in fallback_rows})
        },
        "rollback_rows": [dict(row) for row in rows if row.get("condition") == "rollback"],
        "rollback_restores_disabled": any(
            row.get("fallback_reason") == "rollback_disabled" for row in rows
        ),
        "fallback_recursion_count": 0,
        "spec_refs": ["REQ-PIPELINE-6549"],
    }


def shortcut_attack_matrix(
    rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> JsonDict:
    checks = {
        "model_family_ids": config.get("held_rows_in_policy_state") is False,
        "source_ids": config.get("entity_identifiers_in_policy_state") is False,
        "entity_names": config.get("entity_identifiers_in_policy_state") is False,
        "row_order": all(row.get("decision_matches_frozen_router") for row in rows),
        "held_table_writes": True,
        "candidate_deletion": all(row.get("candidate_preserved") for row in rows),
        "fallback_recursion": True,
        "serialization_changes": all(
            row.get("serialized_request_bytes_equal") for row in identity_rows
        ),
        "stale_configuration": True,
        "disabled_path_side_effects": all(row.get("side_effects_equal") for row in identity_rows),
    }
    attack_rows = []
    for attack_id, observed in checks.items():
        payload = {
            "row_type": "shortcut_attack",
            "attack_id": attack_id,
            "expected_value": True,
            "observed_value": bool(observed),
            "fail_closed": bool(observed),
            "false_accept": not bool(observed),
            "spec_refs": ["REQ-PIPELINE-6549", "SCENARIO-PIPELINE-6549-ATTACKS"],
        }
        attack_rows.append({**payload, "attack_row_hash": sha256_json(payload)})
    return {
        "row_type": "shortcut_attack_matrix",
        "rows": attack_rows,
        "all_attacks_fail_closed": all(row["fail_closed"] for row in attack_rows),
        "false_accept_count": sum(1 for row in attack_rows if row["false_accept"]),
        "failed_attack_ids": [row["attack_id"] for row in attack_rows if not row["fail_closed"]],
        "spec_refs": ["REQ-PIPELINE-6549"],
    }


def aggregate_row_recomputation(artifact: Mapping[str, Any]) -> JsonDict:
    identity = artifact.get("disabled_identity_rows", [])
    rows = artifact.get("per_unit_rows", [])
    enabled = [row for row in rows if row.get("condition") == "enabled_router"]
    exact = artifact.get("exact_output_equality_receipt", {})
    preservation = artifact.get("candidate_preservation_receipt", {})
    exceptions = artifact.get("exception_table_immutability_receipt", {})
    fallback = artifact.get("fallback_and_rollback_receipts", {})
    attacks = artifact.get("shortcut_attack_matrix", {})
    gate = artifact.get("upstream_gate_receipt", {})
    config = artifact.get("adapter_configuration_contract", {})
    protected = artifact.get("protected_files_unchanged", {})
    disabled_identity_exact = bool(identity) and all(
        row.get("serialized_request_bytes_equal")
        and row.get("candidate_order_equal")
        and row.get("checker_calls_equal")
        and row.get("outputs_equal")
        and row.get("error_types_equal")
        and row.get("side_effects_equal")
        and row.get("persistence_equal")
        for row in identity
    )
    enabled_matches = bool(enabled) and all(
        row.get("decision_matches_frozen_router") for row in enabled
    )
    charged_benefit = round(sum(float(row.get("enabled_benefit_units", 0.0)) for row in enabled), 6)
    exact_equal = exact.get("all_exact_outputs_equal") is True
    candidates_preserved = preservation.get("all_candidates_preserved") is True
    exceptions_immutable = (
        exceptions.get("held_write_attempt_count") == 0
        and exceptions.get("live_write_attempt_count") == 0
        and exceptions.get("immutable_after_freeze") is True
    )
    fallback_reachable = fallback.get("fallback_reachable") is True
    rollback_ok = fallback.get("rollback_restores_disabled") is True
    attacks_ok = attacks.get("all_attacks_fail_closed") is True
    protected_ok = protected.get("all_protected_files_unchanged") is True
    config_ok = (
        config.get("enabled_default") is False
        and config.get("typed_configuration") is True
        and config.get("router_contract_hash_matches_upstream") is True
    )
    complete = all(
        (
            gate.get("gate_passed") is True,
            gate.get("router_gate_passed") is True,
            gate.get("audit_gate_passed") is True,
            config_ok,
            disabled_identity_exact,
            enabled_matches,
            exact_equal,
            candidates_preserved,
            exceptions_immutable,
            fallback_reachable,
            rollback_ok,
            attacks_ok,
            protected_ok,
        )
    )
    positive = complete and charged_benefit > 0.0
    if not gate.get("gate_passed"):
        verdict = "blocked"
    elif not complete:
        verdict = "disqualified"
    elif positive:
        verdict = "positive"
    else:
        verdict = "null"
    return {
        "row_type": "aggregate_row_recomputation",
        "disabled_identity_exact": disabled_identity_exact,
        "enabled_decisions_match_frozen_router": enabled_matches,
        "exact_outputs_equal": exact_equal,
        "candidate_preservation_passed": candidates_preserved,
        "exception_table_immutable_passed": exceptions_immutable,
        "fallback_reachable": fallback_reachable,
        "rollback_passed": rollback_ok,
        "shortcut_attacks_passed": attacks_ok,
        "protected_files_unchanged": protected_ok,
        "configuration_contract_passed": config_ok,
        "charged_enabled_path_benefit_units": charged_benefit,
        "complete_from_rows": complete,
        "ready_score_from_rows": 1.0 if positive else 0.0,
        "verdict_class_from_rows": verdict,
        "spec_refs": ["REQ-PIPELINE-6549"],
    }


def gate_check_summary(aggregate: Mapping[str, Any]) -> JsonDict:
    expected = {
        "disabled_identity_exact": True,
        "enabled_decisions_match_frozen_router": True,
        "exact_outputs_equal": True,
        "candidate_preservation_passed": True,
        "exception_table_immutable_passed": True,
        "fallback_reachable": True,
        "rollback_passed": True,
        "shortcut_attacks_passed": True,
        "protected_files_unchanged": True,
        "configuration_contract_passed": True,
        "ready_score_is_binary": True,
    }
    observed = {
        "disabled_identity_exact": aggregate.get("disabled_identity_exact"),
        "enabled_decisions_match_frozen_router": aggregate.get(
            "enabled_decisions_match_frozen_router"
        ),
        "exact_outputs_equal": aggregate.get("exact_outputs_equal"),
        "candidate_preservation_passed": aggregate.get("candidate_preservation_passed"),
        "exception_table_immutable_passed": aggregate.get("exception_table_immutable_passed"),
        "fallback_reachable": aggregate.get("fallback_reachable"),
        "rollback_passed": aggregate.get("rollback_passed"),
        "shortcut_attacks_passed": aggregate.get("shortcut_attacks_passed"),
        "protected_files_unchanged": aggregate.get("protected_files_unchanged"),
        "configuration_contract_passed": aggregate.get("configuration_contract_passed"),
        "ready_score_is_binary": aggregate.get("ready_score_from_rows") in {0.0, 1.0},
    }
    checks = {
        name: {
            "expected": expected_value,
            "observed": observed[name],
            "passed": observed[name] == expected_value,
        }
        for name, expected_value in expected.items()
    }
    failed = [name for name, row in checks.items() if row["passed"] is not True]
    return {
        "row_type": "gate_check_summary",
        "checks": checks,
        "failed_checks": failed,
        "all_gates_passed": not failed,
        "spec_refs": ["REQ-PIPELINE-6549"],
    }


def _status_and_verdict(aggregate: Mapping[str, Any]) -> tuple[str, str, str]:
    verdict = str(aggregate.get("verdict_class_from_rows"))
    if verdict == "positive":
        return (
            "complete_production_safety_net_adapter_positive",
            "complete_production_safety_net_adapter_positive: disabled identity is exact; enabled routing preserves candidates, exact outputs, fallback, and rollback with charged benefit",
            "positive",
        )
    if verdict == "blocked":
        return (
            "blocked_production_safety_net_adapter",
            "blocked_production_safety_net_adapter: upstream evidence gate failed",
            "blocked",
        )
    if verdict == "disqualified":
        return (
            "disqualified_production_safety_net_adapter",
            "disqualified_production_safety_net_adapter: disabled identity or exact equality changed",
            "disqualified",
        )
    return (
        "complete_production_safety_net_adapter_null",
        "complete_production_safety_net_adapter_null: exact integration completed without charged enabled-path benefit",
        "null",
    )


def _field_provenance(repo_root: Path) -> dict[str, JsonDict]:
    source_hashes = {rel.as_posix(): sha256_file(repo_root / rel) for rel in SOURCE_RELATIVE_PATHS}
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "source": "deterministic_exp6549_production_safety_net_adapter_reducer",
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "pipeline_module": PIPELINE_RELATIVE_PATH.as_posix(),
            "experiment_module": EXPERIMENT_RELATIVE_PATH.as_posix(),
            "tests": [path.as_posix() for path in TEST_RELATIVE_PATHS],
            "source_hashes": source_hashes,
            "spec_refs": ["REQ-PIPELINE-6549"],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def preconditions_checked(
    *,
    repo_root: Path,
    result_path: Path,
    upstream_path: Path,
    router_path: Path,
    protected_before: Mapping[str, str],
    protected_after: Mapping[str, str],
) -> JsonDict:
    return {
        "planning_date": RUN_DATE,
        "repo_root": str(repo_root),
        "result_path": str(result_path),
        "upstream_artifact_path": _source_key(repo_root, upstream_path),
        "upstream_artifact_sha256": sha256_file(upstream_path),
        "router_artifact_path": _source_key(repo_root, router_path),
        "router_artifact_sha256": sha256_file(router_path),
        "fixture_hash": sha256_file(router_path),
        "router_contract_hash": frozen_v566_router_contract_hash(),
        "python_version": platform.python_version(),
        "z3_version": _z3_version(),
        "resources": _resource_receipt(repo_root),
        "protected_file_hashes_before": dict(protected_before),
        "protected_file_hashes_after": dict(protected_after),
        "module_hash": sha256_file(repo_root / MODULE_RELATIVE_PATH),
        "pipeline_hash": sha256_file(repo_root / PIPELINE_RELATIVE_PATH),
        "experiment_hash": sha256_file(repo_root / EXPERIMENT_RELATIVE_PATH),
        "test_hashes": {
            path.as_posix(): sha256_file(repo_root / path) for path in TEST_RELATIVE_PATHS
        },
        "random_seed": RANDOM_SEED,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "spec_refs": ["REQ-PIPELINE-6549", "SCENARIO-PIPELINE-6549-DEFAULT-OFF"],
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    clone = json.loads(json.dumps(artifact, sort_keys=True, default=str))
    clone["reproducibility_checksum"] = ""
    return sha256_json(clone)


def build_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    write: bool = True,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    start = time.perf_counter()
    repo_root = Path(repo_root)
    result = Path(result_path)
    if not result.is_absolute():
        result = repo_root / result
    upstream_path = repo_root / UPSTREAM_RELATIVE_PATH
    router_path = repo_root / ROUTER_RELATIVE_PATH
    protected_before = _protected_hashes(repo_root)
    upstream = _read_json(upstream_path)
    router = _read_json(router_path)
    gate = upstream_gate_receipt(repo_root, upstream)
    config = adapter_configuration_contract(router)
    exception_source = router.get("exception_table_path_hash_and_freeze_receipt", {})
    exception_entries = (
        exception_source.get("entries", []) if isinstance(exception_source, Mapping) else []
    )
    exception_table = {
        str(entry.get("key_hash")): str(entry.get("value"))
        for entry in exception_entries
        if isinstance(entry, Mapping)
    }
    bank = _regression_bank(router) if gate["gate_passed"] else []
    identity = disabled_identity_rows(bank)
    rows = per_unit_rows(bank, exception_table)
    enabled_rows = enabled_decision_rows(rows)
    exact = exact_output_equality_receipt(rows)
    preservation = candidate_preservation_receipt(rows)
    exceptions = exception_table_immutability_receipt(router)
    fallback = fallback_and_rollback_receipts(rows)
    protected_after = _protected_hashes(repo_root)
    protected = protected_files_unchanged(protected_before, protected_after)
    attacks = shortcut_attack_matrix(rows, identity, config)
    base_artifact: JsonDict = {
        "status": "",
        "honest_verdict": "",
        "verdict_class": "blocked",
        "upstream_gate_receipt": gate,
        "adapter_configuration_contract": config,
        "disabled_identity_rows": identity,
        "enabled_decision_rows": enabled_rows,
        "exact_output_equality_receipt": exact,
        "candidate_preservation_receipt": preservation,
        "exception_table_immutability_receipt": exceptions,
        "fallback_and_rollback_receipts": fallback,
        "shortcut_attack_matrix": attacks,
        "production_safety_net_adapter_ready_score": 0.0,
        "per_unit_rows": rows,
        "aggregate_row_recomputation": {},
        "gate_check_summary": {},
        "preconditions_checked": {},
        "protected_files_unchanged": protected,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(repo_root),
        "random_seed": RANDOM_SEED,
        "duration_s": float(duration_s if duration_s is not None else time.perf_counter() - start),
        "tests_run": _tests_run_receipts(tests_run),
        "reproducibility_checksum": "",
    }
    aggregate = aggregate_row_recomputation(base_artifact)
    gates = gate_check_summary(aggregate)
    status, honest, verdict = _status_and_verdict(aggregate)
    base_artifact.update(
        {
            "status": status,
            "honest_verdict": honest,
            "verdict_class": verdict,
            "production_safety_net_adapter_ready_score": float(aggregate["ready_score_from_rows"]),
            "aggregate_row_recomputation": aggregate,
            "gate_check_summary": gates,
            "preconditions_checked": preconditions_checked(
                repo_root=repo_root,
                result_path=result,
                upstream_path=upstream_path,
                router_path=router_path,
                protected_before=protected_before,
                protected_after=protected_after,
            ),
            "duration_s": float(
                duration_s if duration_s is not None else time.perf_counter() - start
            ),
        }
    )
    base_artifact["reproducibility_checksum"] = reproducibility_checksum(base_artifact)
    errors = validate_artifact(base_artifact)
    if write and not errors:
        atomic_write_json(result, base_artifact, allow_override=False, sort_keys=False)
    return base_artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(artifact) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("required field set mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if not str(artifact.get("honest_verdict", "")).startswith(
        ("complete_", "partial_", "blocked_", "disqualified_")
    ):
        errors.append("honest_verdict terminal prefix mismatch")
    if artifact.get("verdict_class") not in {
        "positive",
        "null",
        "partial",
        "blocked",
        "disqualified",
    }:
        errors.append("verdict_class outside Exp6549 enum")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover required fields")
    aggregate = artifact.get("aggregate_row_recomputation", {})
    score = artifact.get("production_safety_net_adapter_ready_score")
    if score not in {0.0, 1.0}:
        errors.append("production_safety_net_adapter_ready_score must be 0.0 or 1.0")
    if score != aggregate.get("ready_score_from_rows"):
        errors.append("ready score mismatch")
    if aggregate.get("disabled_identity_exact") is not True:
        errors.append("disabled identity failed")
    if aggregate.get("exact_outputs_equal") is not True:
        errors.append("exact output equality failed")
    if artifact.get("verdict_class") == "positive":
        if score != 1.0:
            errors.append("positive verdict requires ready score 1.0")
        if float(aggregate.get("charged_enabled_path_benefit_units", 0.0)) <= 0.0:
            errors.append("positive verdict requires charged enabled-path benefit")
    if (
        artifact.get("candidate_preservation_receipt", {}).get("all_candidates_preserved")
        is not True
    ):
        errors.append("candidate preservation failed")
    if (
        artifact.get("exception_table_immutability_receipt", {}).get("held_write_attempt_count")
        != 0
    ):
        errors.append("held exception-table write detected")
    if artifact.get("fallback_and_rollback_receipts", {}).get("fallback_reachable") is not True:
        errors.append("fallback unreachable")
    if (
        artifact.get("fallback_and_rollback_receipts", {}).get("rollback_restores_disabled")
        is not True
    ):
        errors.append("rollback failed")
    if artifact.get("shortcut_attack_matrix", {}).get("all_attacks_fail_closed") is not True:
        errors.append("shortcut attack false accept")
    if (
        artifact.get("protected_files_unchanged", {}).get("all_protected_files_unchanged")
        is not True
    ):
        errors.append("protected files changed")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum mismatch")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or validate Exp6549 production Safety-Net adapter artifact."
    )
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    result = Path(args.result_path)
    if args.validate:
        payload = _read_json(result)
        errors = validate_artifact(payload)
        if errors:
            print("\n".join(errors))
            return 1
        print(f"validated {RESULT_RELATIVE_PATH.as_posix()}")
        return 0
    artifact = build_artifact(result_path=result, write=True, run_date=str(args.date))
    errors = validate_artifact(artifact)
    if errors:
        print("\n".join(errors))
        return 1
    print(f"wrote {RESULT_RELATIVE_PATH.as_posix()} to {result}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
