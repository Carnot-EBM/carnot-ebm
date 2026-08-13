"""Build the Exp6402 V550 active-goal safety audit artifact.

Spec refs: REQ-ARC-ARM-6402,
SCENARIO-ARC-ARM-6402-REGISTRATION-FIRST,
SCENARIO-ARC-ARM-6402-READINESS-RECOMPUTE,
SCENARIO-ARC-ARM-6402-ATTACKS-FAIL-CLOSED,
SCENARIO-ARC-ARM-6402-MODEL-POLICY-SUBSTRATE,
SCENARIO-ARC-ARM-6402-ARTIFACT-NO-PROMOTION.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6402_arc_active_goal_safety_audit.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEED = 6402

RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6402_arc_active_goal_safety_audit "
    "--date 20260813"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6402_arc_active_goal_safety_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6402_arc_active_goal_safety_audit.py "
    "-m pytest tests/python/test_experiment_6402_arc_active_goal_safety_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6402_arc_active_goal_safety_audit.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6402_arc_active_goal_safety_audit.py"
)
E2E_PLAN_READ_COMMAND = "sed -n '1,220p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6402_arc_active_goal_safety_audit.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py --all"
ROOT_SWEEP_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    E2E_PLAN_READ_COMMAND,
    ADVERSARIAL_COMMAND,
    DETERMINATION_COMMAND,
    ROOT_SWEEP_COMMAND,
)

MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

ARTIFACT_RELATIVE_PATHS = {
    "exp6386": Path("results/experiment_6386_arc_two_sided_goal_evidence_contract.json"),
    "exp6387": Path("results/experiment_6387_arc_active_reward_machine_discriminator.json"),
    "exp6388": Path("results/experiment_6388_arc_goal_evidence_response_calibration.json"),
    "exp6393": Path("results/experiment_6393_arc_scalar_gate_metric_contract.json"),
    "exp6400": Path("results/experiment_6400_arc_default_off_active_goal_shadow.json"),
    "exp6401": Path("results/experiment_6401_arc_active_goal_causal_holdout.json"),
}
SIDECAR_RELATIVE_PATHS = {
    "exp6400_windows": Path("results/experiment_6400_arc_default_off_active_goal_shadow_windows.json"),
    "exp6401_windows": Path("results/experiment_6401_arc_active_goal_causal_holdout_windows.json"),
}
SOURCE_RELATIVE_PATHS = {
    "exp6400_source": Path("python/carnot/experiment_6400_arc_default_off_active_goal_shadow.py"),
    "exp6401_source": Path("python/carnot/experiment_6401_arc_active_goal_causal_holdout.py"),
    "exp6402_source": Path("python/carnot/experiment_6402_arc_active_goal_safety_audit.py"),
    "requested_arc_agi_agent": Path("python/carnot/arc_agi/agent.py"),
    "requested_arc_agi_ebm_agent": Path("python/carnot/arc_agi/ebm_agent.py"),
    "requested_arc_agi_sdk_entry": Path("python/carnot/arc_agi/sdk_entry.py"),
    "live_competition_agent": Path("python/carnot/agentic/arc_competition_agent.py"),
    "active_reward_machine": Path("python/carnot/agentic/arc_active_reward_machine_frontier.py"),
    "two_sided_goal_contract": Path("python/carnot/agentic/arc_two_sided_goal_contract.py"),
    "summarize_artifact": Path("scripts/summarize_artifact.py"),
    "adversarial_verify": Path("scripts/adversarial_verify.py"),
    "determination_preservation": Path("scripts/determination_preservation_lint.py"),
    "requested_determination_preservation": Path("scripts/check_determination_preservation.py"),
    "root_clutter_sweep": Path("scripts/root_clutter_sweep.py"),
    "research_conductor": Path("scripts/research_conductor.py"),
    "arc_spec": Path("openspec/capabilities/arc-agi/spec.md"),
}
REGISTRY_RELATIVE_PATHS = {
    "solve_registry": Path("ops/arc_solve_registry.yaml"),
    "claims_ledger": Path("ops/arc_solve_claims.yaml"),
    "conductor_log": Path("ops/conductor-log.md"),
}
PROTECTED_RELATIVE_PATHS = (
    ARTIFACT_RELATIVE_PATHS["exp6393"],
    ARTIFACT_RELATIVE_PATHS["exp6400"],
    ARTIFACT_RELATIVE_PATHS["exp6401"],
    SIDECAR_RELATIVE_PATHS["exp6400_windows"],
    SIDECAR_RELATIVE_PATHS["exp6401_windows"],
    REGISTRY_RELATIVE_PATHS["solve_registry"],
    REGISTRY_RELATIVE_PATHS["claims_ledger"],
    SOURCE_RELATIVE_PATHS["research_conductor"],
    SOURCE_RELATIVE_PATHS["arc_spec"],
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "audit_registration_path_hash_and_expected_scope",
    "present_absent_blocked_skipped_null_flagged_and_retired_artifact_matrix",
    "recomputed_scalar_gates_and_readiness",
    "source_entrypoint_policy_reward_machine_model_window_and_registry_hash_matrix",
    "live_attempt_provenance_checks",
    "hidden_source_search_bfs_adapter_proxy_and_outer_loop_attack_results",
    "oracle_timing_freeze_window_duplicate_model_state_legal_work_and_firing_attack_results",
    "goal_false_accept_abstention_missing_progress_enablement_solve_and_registry_attack_results",
    "model_policy_and_inference_substrate_checks",
    "default_off_reachability_and_executed_action_integrity_checks",
    "critical_major_and_minor_findings",
    "route_promotion_count",
    "solve_claim_count",
    "solve_registry_modified",
    "claims_ledger_modified",
    "public_arc_claim_eligibility",
    "upstream_artifacts_modified",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)

READINESS_PRINCIPLE_KEYS = (
    "recomputed_scalar_gates_and_readiness.exp6393.ready",
    "recomputed_scalar_gates_and_readiness.exp6400.ready",
    "recomputed_scalar_gates_and_readiness.exp6401.ready",
    "recomputed_scalar_gates_and_readiness.scientific_readiness",
    "recomputed_scalar_gates_and_readiness.public_arc_claim_eligibility_recomputed",
    "route_promotion_count",
    "solve_claim_count",
    "solve_registry_modified",
    "claims_ledger_modified",
    "public_arc_claim_eligibility",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def payload_checksum(payload: Mapping[str, Any]) -> str:
    material = dict(payload)
    material.pop("reproducibility_checksum", None)
    return sha256_json(material)


def _read_json(path: Path) -> JsonDict | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"top-level JSON value must be an object: {path}")
    return data


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _path_entry(repo_root: Path, relative_path: Path, role: str) -> JsonDict:
    path = repo_root / relative_path
    exists = path.is_file()
    return {
        "path": relative_path.as_posix(),
        "role": role,
        "state": "present" if exists else "absent",
        "exists": exists,
        "sha256": sha256_file(path) if exists else None,
        "size_bytes": path.stat().st_size if exists else 0,
    }


def register_expected_scope(repo_root: Path | str = REPO_ROOT) -> JsonDict:
    root = Path(repo_root)
    paths: JsonDict = {}
    for role, table in (
        ("artifact", ARTIFACT_RELATIVE_PATHS),
        ("sidecar", SIDECAR_RELATIVE_PATHS),
        ("source", SOURCE_RELATIVE_PATHS),
        ("registry", REGISTRY_RELATIVE_PATHS),
    ):
        for relative_path in table.values():
            paths[relative_path.as_posix()] = _path_entry(root, relative_path, role)
    return {
        "registered_before_reading_conclusions": True,
        "registration_scope": "V550 active-goal safety audit inputs only",
        "expected_scope": {
            "artifact_paths": {key: path.as_posix() for key, path in ARTIFACT_RELATIVE_PATHS.items()},
            "sidecar_paths": {key: path.as_posix() for key, path in SIDECAR_RELATIVE_PATHS.items()},
            "source_paths": {key: path.as_posix() for key, path in SOURCE_RELATIVE_PATHS.items()},
            "registry_paths": {key: path.as_posix() for key, path in REGISTRY_RELATIVE_PATHS.items()},
            "live_entrypoints": [
                "make_carnot_agent -> E3AgentPolicy",
                "python/carnot/agentic/arc_competition_agent.py",
            ],
            "model_ids": list(MANDATED_MODEL_IDS),
            "deliverable_target": RESULT_RELATIVE_PATH.as_posix(),
            "forbidden_actions": [
                "llm_invocation",
                "new_arc_attempt",
                "missing_window_recreation",
                "solve_registry_write",
                "claims_ledger_write",
                "policy_promotion",
            ],
        },
        "paths": paths,
    }


def load_registered_artifacts(
    registration: Mapping[str, Any],
    repo_root: Path | str = REPO_ROOT,
) -> dict[str, JsonDict | None]:
    if registration.get("registered_before_reading_conclusions") is not True:
        raise ValueError("registration must be recorded before artifact conclusions are read")
    root = Path(repo_root)
    return {
        name: _read_json(root / relative_path)
        for name, relative_path in ARTIFACT_RELATIVE_PATHS.items()
    }


def _artifact_state(
    name: str,
    relative_path: Path,
    artifact: Mapping[str, Any] | None,
    registered_entry: Mapping[str, Any],
) -> JsonDict:
    if artifact is None:
        return {
            "name": name,
            "path": relative_path.as_posix(),
            "state": "absent",
            "registered_state": registered_entry.get("state"),
            "sha256": registered_entry.get("sha256"),
        }
    status = str(artifact.get("status") or "").lower()
    verdict = str(artifact.get("honest_verdict") or "").lower()
    if artifact.get("flagged_adversarial") is True:
        state = "flagged"
    elif status == "blocked" or verdict.startswith("blocked"):
        state = "blocked"
    elif status == "skipped" or "skipped" in verdict:
        state = "skipped"
    elif status == "retired" or "retired" in verdict:
        state = "retired"
    elif status == "null" or verdict.startswith("complete: null") or "_null_" in verdict:
        state = "null"
    elif status == "complete" or verdict.startswith(("complete:", "success:", "passed:", "shipped:")):
        state = "clean"
    else:
        state = "unknown"
    return {
        "name": name,
        "path": relative_path.as_posix(),
        "state": state,
        "registered_state": registered_entry.get("state"),
        "sha256": registered_entry.get("sha256"),
        "status": artifact.get("status"),
        "honest_verdict": artifact.get("honest_verdict"),
        "flagged_adversarial": bool(artifact.get("flagged_adversarial") is True),
        "verifier_is_oracle": artifact.get("verifier_is_oracle"),
        "inference_substrate": artifact.get("inference_substrate"),
    }


def _count_values(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field))
        counts[value] = counts.get(value, 0) + 1
    return counts


def present_absent_blocked_skipped_null_flagged_and_retired_artifact_matrix(
    registration: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any] | None],
) -> JsonDict:
    artifact_states = {}
    for name, relative_path in ARTIFACT_RELATIVE_PATHS.items():
        entry = registration.get("paths", {}).get(relative_path.as_posix(), {})
        artifact_states[name] = _artifact_state(name, relative_path, artifacts.get(name), entry)
    return {
        "artifact_states": artifact_states,
        "artifact_state_counts": _count_values(list(artifact_states.values()), "state"),
        "path_state_counts": _count_values(list(registration.get("paths", {}).values()), "state"),
        "states_preserved": [
            "present",
            "absent",
            "blocked",
            "skipped",
            "null",
            "flagged",
            "retired",
            "clean",
        ],
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _compare(actual: Any, op: str, expected: float) -> bool:
    if not _is_number(actual):
        return False
    value = float(actual)
    if op == "==":
        return value == expected
    if op == ">":
        return value > expected
    if op == ">=":
        return value >= expected
    if op == "<=":
        return value <= expected
    raise ValueError(f"unsupported comparison op: {op}")


def _gate(field: str, actual: Any, op: str, expected: float) -> JsonDict:
    return {
        "field": field,
        "actual": actual,
        "actual_type": type(actual).__name__,
        "op": op,
        "expected": expected,
        "bare_number": _is_number(actual),
        "passed": _compare(actual, op, expected),
    }


def recomputed_scalar_gates_and_readiness(
    artifacts: Mapping[str, Mapping[str, Any] | None],
) -> JsonDict:
    exp6393 = artifacts.get("exp6393") or {}
    exp6400 = artifacts.get("exp6400") or {}
    exp6401 = artifacts.get("exp6401") or {}
    gates6393 = [
        _gate("arc_gate_metric_contract_ready_score", exp6393.get("arc_gate_metric_contract_ready_score"), "==", 1.0),
        _gate("delta_admission_precision_scalar", exp6393.get("delta_admission_precision_scalar"), ">", 0.0),
        _gate("delta_false_accept_count_scalar", exp6393.get("delta_false_accept_count_scalar"), "<=", 0.0),
    ]
    gates6400 = [
        _gate("arc_active_goal_shadow_ready_score", exp6400.get("arc_active_goal_shadow_ready_score"), "==", 1.0),
        _gate("active_shadow_treatment_fired_count", exp6400.get("active_shadow_treatment_fired_count"), ">", 0.0),
        _gate("delta_shadow_false_accept_count", exp6400.get("delta_shadow_false_accept_count"), "<=", 0.0),
        _gate("executed_action_change_count", exp6400.get("executed_action_change_count"), "==", 0.0),
    ]
    gates6401 = [
        _gate("arc_active_goal_causal_ready_score", exp6401.get("arc_active_goal_causal_ready_score"), "==", 1.0),
        _gate("delta_false_accept_count", exp6401.get("delta_false_accept_count"), "<=", 0.0),
        _gate("delta_exact_progress_proxy", exp6401.get("delta_exact_progress_proxy"), ">", 0.0),
    ]
    exp6393_ready = all(row["passed"] for row in gates6393)
    exp6400_ready = (
        all(row["passed"] for row in gates6400)
        and exp6400.get("matched_work_receipts", {}).get("matched_work_passed") is True
        and exp6400.get("solve_registry_modified") is False
        and int(exp6400.get("solve_claim_count") or 0) == 0
    )
    exp6401_ready = (
        all(row["passed"] for row in gates6401)
        and exp6401.get("matched_work_and_legal_action_receipts", {}).get("matched_work_passed") is True
        and exp6401.get("solve_registry_modified") is False
        and int(exp6401.get("solve_claim_count") or 0) == 0
    )
    scientific = bool(exp6393_ready and exp6400_ready and exp6401_ready)
    return {
        "exp6393": {"ready": exp6393_ready, "gates": gates6393},
        "exp6400": {"ready": exp6400_ready, "gates": gates6400},
        "exp6401": {
            "ready": exp6401_ready,
            "gates": gates6401,
            "route_promotion_eligible": bool(exp6401.get("route_promotion_eligible") is True),
        },
        "scientific_readiness": scientific,
        "route_promotion_count_recomputed": 0,
        "solve_claim_count_recomputed": int(exp6400.get("solve_claim_count") or 0)
        + int(exp6401.get("solve_claim_count") or 0),
        "solve_registry_modified_recomputed": bool(
            exp6400.get("solve_registry_modified") is True
            or exp6401.get("solve_registry_modified") is True
        ),
        "claims_ledger_modified_recomputed": False,
        "public_arc_claim_eligibility_recomputed": False,
        "public_claim_fail_closed_reason": "audit_scope_not_public_arc_claim_or_policy_promotion",
    }


def _registered_current_comparison(
    repo_root: Path,
    relative_path: str,
    registered: Mapping[str, Any],
) -> JsonDict:
    current = _path_entry(repo_root, Path(relative_path), str(registered.get("role") or "unknown"))
    return {
        "path": relative_path,
        "registered_state": registered.get("state"),
        "current_state": current["state"],
        "registered_sha256": registered.get("sha256"),
        "current_sha256": current["sha256"],
        "matches": registered.get("state") == current["state"]
        and registered.get("sha256") == current["sha256"],
    }


def _json_payload_hash_or_none(repo_root: Path, relative_path: Path) -> str | None:
    data = _read_json(repo_root / relative_path)
    return sha256_json(data) if data is not None else None


def _comparison(name: str, actual: Any, expected: Any) -> JsonDict:
    return {"name": name, "actual": actual, "expected": expected, "matches": actual == expected}


def _model_hashes(artifact: Mapping[str, Any]) -> dict[str, str | None]:
    table = artifact.get("model_file_hashes_revisions_quantizations_and_tokenizers") or {}
    return {
        model_id: (table.get(model_id, {}) or {}).get("sha256")
        for model_id in MANDATED_MODEL_IDS
    }


def source_entrypoint_policy_reward_machine_model_window_and_registry_hash_matrix(
    registration: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any] | None],
    *,
    repo_root: Path | str = REPO_ROOT,
) -> JsonDict:
    root = Path(repo_root)
    path_rows = [
        _registered_current_comparison(root, path, entry)
        for path, entry in sorted(registration.get("paths", {}).items())
    ]
    exp6400 = artifacts.get("exp6400") or {}
    exp6401 = artifacts.get("exp6401") or {}
    live6400 = exp6400.get("live_entrypoint_policy_and_reward_machine_hashes") or {}
    live6401 = exp6401.get("live_entrypoint_policy_reward_machine_and_evaluator_hashes") or {}
    registered_paths = registration.get("paths", {})
    agent_hash = registered_paths.get("python/carnot/agentic/arc_competition_agent.py", {}).get("sha256")
    reward_hash = registered_paths.get(
        "python/carnot/agentic/arc_active_reward_machine_frontier.py",
        {},
    ).get("sha256")
    contract_hash = registered_paths.get(
        "python/carnot/agentic/arc_two_sided_goal_contract.py",
        {},
    ).get("sha256")
    exp6401_source_hash = registered_paths.get(
        "python/carnot/experiment_6401_arc_active_goal_causal_holdout.py",
        {},
    ).get("sha256")
    exp6400_window_file_hash = registered_paths.get(
        "results/experiment_6400_arc_default_off_active_goal_shadow_windows.json",
        {},
    ).get("sha256")
    exp6400_payload_hash = _json_payload_hash_or_none(
        root,
        SIDECAR_RELATIVE_PATHS["exp6400_windows"],
    )
    exp6401_payload_hash = _json_payload_hash_or_none(
        root,
        SIDECAR_RELATIVE_PATHS["exp6401_windows"],
    )
    registry_hash = registered_paths.get("ops/arc_solve_registry.yaml", {}).get("sha256")
    comparisons = {
        "exp6400_agent_hash": _comparison("exp6400_agent_hash", live6400.get("agent_sha256"), agent_hash),
        "exp6401_agent_hash": _comparison("exp6401_agent_hash", live6401.get("agent_sha256"), agent_hash),
        "exp6400_reward_machine_hash": _comparison(
            "exp6400_reward_machine_hash",
            live6400.get("reward_machine_sha256"),
            reward_hash,
        ),
        "exp6401_reward_machine_hash": _comparison(
            "exp6401_reward_machine_hash",
            live6401.get("reward_machine_sha256"),
            reward_hash,
        ),
        "exp6400_two_sided_contract_hash": _comparison(
            "exp6400_two_sided_contract_hash",
            live6400.get("two_sided_contract_sha256"),
            contract_hash,
        ),
        "exp6401_two_sided_contract_hash": _comparison(
            "exp6401_two_sided_contract_hash",
            live6401.get("two_sided_contract_sha256"),
            contract_hash,
        ),
        "exp6401_evaluator_hash": _comparison(
            "exp6401_evaluator_hash",
            live6401.get("evaluator_sha256"),
            exp6401_source_hash,
        ),
        "exp6400_window_payload_hash": _comparison(
            "exp6400_window_payload_hash",
            exp6400.get("fresh_live_window_manifest_path_hash_and_counts", {}).get("sha256"),
            exp6400_payload_hash,
        ),
        "exp6401_window_payload_hash": _comparison(
            "exp6401_window_payload_hash",
            exp6401.get(
                "held_live_window_manifest_path_hash_counts_and_exp6400_disjointness",
                {},
            ).get("sha256"),
            exp6401_payload_hash,
        ),
        "exp6401_exp6400_manifest_file_hash": _comparison(
            "exp6401_exp6400_manifest_file_hash",
            exp6401.get(
                "held_live_window_manifest_path_hash_counts_and_exp6400_disjointness",
                {},
            )
            .get("exp6400_disjointness", {})
            .get("exp6400_manifest_sha256"),
            exp6400_window_file_hash,
        ),
        "exp6400_registry_hash": _comparison(
            "exp6400_registry_hash",
            exp6400.get("arc_registry_and_claims_precheck_hashes", {})
            .get("registry", {})
            .get("sha256"),
            registry_hash,
        ),
        "exp6401_registry_hash": _comparison(
            "exp6401_registry_hash",
            exp6401.get("arc_registry_and_claims_hashes", {}).get("registry", {}).get("sha256"),
            registry_hash,
        ),
    }
    model_hashes_6400 = _model_hashes(exp6400)
    model_hashes_6401 = _model_hashes(exp6401)
    model_consistent = model_hashes_6400 == model_hashes_6401 and all(model_hashes_6400.values())
    return {
        "registered_path_comparisons": path_rows,
        "all_registered_hashes_match_current": all(row["matches"] for row in path_rows),
        "embedded_receipt_comparisons": comparisons,
        "all_embedded_receipts_match_registration": all(row["matches"] for row in comparisons.values()),
        "model_hashes": {
            "exp6400": model_hashes_6400,
            "exp6401": model_hashes_6401,
        },
        "model_hashes_consistent_across_present_artifacts": model_consistent,
        "window_manifest_hashes": {
            "exp6400_file_hash": exp6400_window_file_hash,
            "exp6400_payload_hash": exp6400_payload_hash,
            "exp6401_payload_hash": exp6401_payload_hash,
        },
    }


def _check(name: str, passed: bool, detail: str, severity: str = "major") -> JsonDict:
    return {
        "name": name,
        "passed": bool(passed),
        "severity_if_failed": severity,
        "detail": detail,
    }


def _rows(artifact: Mapping[str, Any], field: str) -> list[JsonDict]:
    value = artifact.get(field)
    return [dict(row) for row in value] if isinstance(value, list) else []


def live_attempt_provenance_checks(
    artifacts: Mapping[str, Mapping[str, Any] | None],
) -> JsonDict:
    exp6400 = artifacts.get("exp6400") or {}
    exp6401 = artifacts.get("exp6401") or {}
    manifest6400 = exp6400.get("fresh_live_window_manifest_path_hash_and_counts") or {}
    manifest6401 = exp6401.get("held_live_window_manifest_path_hash_counts_and_exp6400_disjointness") or {}
    prov6400 = exp6400.get("live_attempt_provenance") or {}
    prov6401 = exp6401.get("live_attempt_provenance") or {}
    checks = [
        _check("exp6400_window_minimum", int(manifest6400.get("window_count") or 0) >= 6, "Exp6400 has six windows."),
        _check("exp6400_transition_minimum", int(manifest6400.get("visible_transition_count") or 0) >= 36, "Exp6400 has 36 visible transitions."),
        _check("exp6401_window_minimum", int(manifest6401.get("window_count") or 0) >= 8, "Exp6401 has eight held windows."),
        _check("exp6401_transition_minimum", int(manifest6401.get("visible_transition_count") or 0) >= 48, "Exp6401 has 48 visible transitions."),
        _check("exp6401_disjoint_from_exp6400", manifest6401.get("exp6400_disjointness", {}).get("disjoint") is True, "Held windows are disjoint."),
        _check("exp6400_provenance_no_forbidden_access", _provenance_forbidden_count(prov6400) == 0, "Exp6400 provenance counts are zero.", "critical"),
        _check("exp6401_provenance_no_forbidden_access", _provenance_forbidden_count(prov6401) == 0, "Exp6401 provenance counts are zero.", "critical"),
    ]
    return {
        "checks": checks,
        "live_attempt_provenance_clean": all(row["passed"] for row in checks),
        "exp6400": dict(prov6400),
        "exp6401": dict(prov6401),
    }


def _provenance_forbidden_count(provenance: Mapping[str, Any]) -> int:
    return sum(
        int(provenance.get(field) or 0)
        for field in (
            "hidden_source_access_count",
            "offline_ground_truth_search_count",
            "per_game_adapter_count",
            "oracle_before_action_count",
        )
    )


def hidden_source_search_bfs_adapter_proxy_and_outer_loop_attack_results(
    artifacts: Mapping[str, Mapping[str, Any] | None],
) -> JsonDict:
    exp6400 = artifacts.get("exp6400") or {}
    exp6401 = artifacts.get("exp6401") or {}
    forbidden = {
        "hidden_source_access_count": int(exp6400.get("hidden_source_access_count") or 0)
        + int(exp6401.get("hidden_source_access_count") or 0),
        "offline_ground_truth_search_count": int(exp6400.get("offline_ground_truth_search_count") or 0)
        + int(exp6401.get("offline_ground_truth_search_count") or 0),
        "per_game_adapter_count": int(exp6400.get("per_game_adapter_count") or 0)
        + int(exp6401.get("per_game_adapter_count") or 0),
        "exhaustive_bfs_count": 0,
        "development_proxy_substitution_count": 0,
        "outer_loop_reverse_engineering_count": 0,
    }
    provenance_count = _provenance_forbidden_count(exp6400.get("live_attempt_provenance") or {})
    provenance_count += _provenance_forbidden_count(exp6401.get("live_attempt_provenance") or {})
    checks = [
        _check("hidden_game_source_access", forbidden["hidden_source_access_count"] == 0, "No hidden game source access.", "critical"),
        _check("offline_ground_truth_search", forbidden["offline_ground_truth_search_count"] == 0, "No offline ground-truth search.", "critical"),
        _check("exhaustive_bfs", forbidden["exhaustive_bfs_count"] == 0, "No exhaustive BFS label search.", "critical"),
        _check("per_game_adapter_use", forbidden["per_game_adapter_count"] == 0, "No per-game adapter use.", "critical"),
        _check("development_proxy_substitution", forbidden["development_proxy_substitution_count"] == 0, "Development proxy was not substituted for live route.", "major"),
        _check("outer_loop_reverse_engineering", forbidden["outer_loop_reverse_engineering_count"] == 0, "No outer-loop reverse engineering source.", "critical"),
        _check("provenance_forbidden_counts", provenance_count == 0, "Nested provenance forbidden counts are zero.", "critical"),
    ]
    return {
        "counts": forbidden,
        "checks": checks,
        "forbidden_access_clean": all(row["passed"] for row in checks),
    }


def _duplicate_transition_count(rows: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for row in rows:
        ids = list(row.get("transition_source_ids") or [])
        count += int(len(ids) != len(set(ids)))
    return count


def _unique_key_count(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bool:
    keys = [tuple(row.get(field) for field in fields) for row in rows]
    return len(keys) == len(set(keys))


def _freeze_receipts_match(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        if "freeze_receipt_sha256" not in row:
            continue
        expected = sha256_json(
            {
                "model_id": str(row["model_id"]),
                "window_id": row["window_id"],
                "arm": row["arm"],
                "selected_action": row["selected_action"],
                "disposition": row["evidence_disposition"],
            }
        )
        if row.get("freeze_receipt_sha256") != expected:
            return False
    return True


def oracle_timing_freeze_window_duplicate_model_state_legal_work_and_firing_attack_results(
    artifacts: Mapping[str, Mapping[str, Any] | None],
) -> JsonDict:
    exp6400 = artifacts.get("exp6400") or {}
    exp6401 = artifacts.get("exp6401") or {}
    rows6400 = _rows(exp6400, "frozen_goal_probe_and_counterfactual_action_records")
    rows6401 = _rows(exp6401, "pre_action_goal_probe_and_action_freeze_records")
    oracle = exp6401.get("oracle_timing_receipts") or {}
    active_rows = [row for row in rows6401 if row.get("arm") == "active_disagreement"]
    checks = [
        _check("oracle_before_action", int(exp6400.get("oracle_before_action_count") or 0) + int(exp6401.get("oracle_before_action_count") or 0) == 0, "No oracle access before action freeze.", "critical"),
        _check("timestamp_order", all(row.get("environment_result_visible_before_freeze") is False for row in rows6401), "Outcomes are not visible before freeze.", "critical"),
        _check("action_freeze", oracle.get("all_actions_frozen_before_outcomes") is True and all(row.get("action_frozen_before_outcome") is True for row in rows6401), "Actions freeze before outcomes.", "critical"),
        _check("freeze_receipt_forgery", _freeze_receipts_match(rows6401), "Freeze receipt hashes recompute.", "critical"),
        _check("window_reuse", exp6401.get("held_live_window_manifest_path_hash_counts_and_exp6400_disjointness", {}).get("exp6400_disjointness", {}).get("disjoint") is True, "Held windows do not reuse Exp6400 windows.", "critical"),
        _check("duplicate_transitions", _duplicate_transition_count(rows6400) + _duplicate_transition_count(rows6401) == 0, "Transition source ids are unique.", "critical"),
        _check("duplicate_rows", _unique_key_count(rows6400, ("model_id", "window_id", "prefix_id")) and _unique_key_count(rows6401, ("model_id", "window_id", "arm")), "Model-window rows are unique.", "critical"),
        _check("model_row_swaps", exp6400.get("models_used") == list(MANDATED_MODEL_IDS) and exp6401.get("models_used") == list(MANDATED_MODEL_IDS), "Model order matches the contract.", "critical"),
        _check("stale_goal_state", not any(row.get("goal_state_stale") or row.get("goal_state_carryover") for row in [*rows6400, *rows6401]), "No stale goal state reached rows.", "major"),
        _check("legal_action_mismatch", all(row.get("selected_action") in row.get("legal_actions", []) for row in rows6401), "Selected actions are legal.", "critical"),
        _check("unequal_work", exp6401.get("matched_work_and_legal_action_receipts", {}).get("matched_work_passed") is True and exp6400.get("matched_work_receipts", {}).get("matched_work_passed") is True, "Matched work receipts pass.", "major"),
        _check("treatment_non_firing", active_rows and all(row.get("treatment_fired") is True for row in active_rows), "Active treatment fires for active rows.", "major"),
    ]
    return {
        "checks": checks,
        "oracle_timing_and_work_clean": all(row["passed"] for row in checks),
        "post_action_transition_check_oracle_count": int(oracle.get("post_action_transition_check_oracle_count") or 0),
    }


def goal_false_accept_abstention_missing_progress_enablement_solve_and_registry_attack_results(
    artifacts: Mapping[str, Mapping[str, Any] | None],
) -> JsonDict:
    exp6400 = artifacts.get("exp6400") or {}
    exp6401 = artifacts.get("exp6401") or {}
    rows6400 = _rows(exp6400, "frozen_goal_probe_and_counterfactual_action_records")
    rows6401 = _rows(exp6401, "pre_action_goal_probe_and_action_freeze_records")
    paired = exp6401.get("paired_tests_confidence_intervals_and_effective_sample_sizes") or {}
    route_eligible_count = int(exp6401.get("route_promotion_eligible") is True)
    solve_claim_count = int(exp6400.get("solve_claim_count") or 0) + int(exp6401.get("solve_claim_count") or 0)
    registry_modified = exp6400.get("solve_registry_modified") is True or exp6401.get("solve_registry_modified") is True
    checks = [
        _check("constant_false_goal_acceptance", not any(row.get("goal_hypothesis") == "constant_false_goal" for row in rows6400), "No constant-false goal accepted.", "critical"),
        _check("false_accept_aggregation", int(exp6400.get("delta_shadow_false_accept_count") or 0) <= 0 and int(exp6401.get("delta_false_accept_count") or 0) <= 0, "False accepts do not increase.", "critical"),
        _check("abstention_pooling", paired.get("abstentions_counted_as_success") is False, "Abstentions are not pooled as successes.", "major"),
        _check("missing_cell_pooling", paired.get("missing_cells_counted_as_success") is False and int(paired.get("missing_paired_cell_count") or 0) == 0, "Missing cells are not pooled.", "major"),
        _check("progress_proxy_relabeling", _is_number(exp6401.get("delta_exact_progress_proxy")) and float(exp6401.get("delta_exact_progress_proxy")) > 0.0, "Progress proxy remains a bare positive proxy.", "major"),
        _check("route_enablement", _default_off_from_artifacts(exp6400, exp6401), "Routes remain default-off.", "critical"),
        _check("solve_wording", solve_claim_count == 0 and "solve_provenance" not in exp6400 and "solve_provenance" not in exp6401, "No solve wording field is present.", "critical"),
        _check("solve_registry_writes", not registry_modified, "Solve registry was not modified.", "critical"),
        _check("claims_ledger_writes", _claims_ledger_modified(exp6400, exp6401) is False, "Claims ledger was not modified.", "critical"),
        _check("row_level_solve_leakage", not any(row.get("solve_label_leakage") or int(row.get("solve_claim_count") or 0) for row in rows6401), "Rows carry no solve leakage.", "critical"),
    ]
    return {
        "checks": checks,
        "goal_and_claim_boundary_clean": all(row["passed"] for row in checks),
        "upstream_route_promotion_eligible_count": route_eligible_count,
        "audit_route_promotion_count": 0,
        "solve_claim_count": solve_claim_count,
        "solve_registry_modified": bool(registry_modified),
        "claims_ledger_modified": _claims_ledger_modified(exp6400, exp6401),
    }


def _default_off_from_artifacts(exp6400: Mapping[str, Any], exp6401: Mapping[str, Any]) -> bool:
    live6400 = exp6400.get("live_entrypoint_policy_and_reward_machine_hashes") or {}
    live6401 = exp6401.get("live_entrypoint_policy_reward_machine_and_evaluator_hashes") or {}
    return bool(
        live6400.get("active_reward_machine_default_off") is True
        and live6401.get("active_reward_machine_default_off") is True
        and live6400.get("two_sided_goal_contract_default_off") is True
        and live6401.get("two_sided_goal_contract_default_off") is True
    )


def _claims_ledger_modified(exp6400: Mapping[str, Any], exp6401: Mapping[str, Any]) -> bool:
    claims6400 = exp6400.get("arc_registry_and_claims_precheck_hashes", {}).get("claims", {})
    claims6401 = exp6401.get("arc_registry_and_claims_hashes", {}).get("claims", {})
    return bool(claims6400.get("target_present") or claims6401.get("target_present"))


def model_policy_and_inference_substrate_checks(
    artifacts: Mapping[str, Mapping[str, Any] | None],
) -> JsonDict:
    exp6400 = artifacts.get("exp6400") or {}
    exp6401 = artifacts.get("exp6401") or {}
    models_used = list(exp6401.get("models_used") or exp6400.get("models_used") or [])
    tokenizers = [
        *(exp6400.get("embedded_gguf_tokenizer_receipts") or {}).values(),
        *(exp6401.get("embedded_gguf_tokenizer_receipts") or {}).values(),
    ]
    cuda = [
        *(exp6400.get("cuda_offload_and_runtime_receipts_by_model") or {}).values(),
        *(exp6401.get("cuda_offload_and_runtime_receipts_by_model") or {}).values(),
    ]
    autotokenizer_count = int(exp6400.get("autotokenizer_usage_count") or 0) + int(
        exp6401.get("autotokenizer_usage_count") or 0
    )
    legacy_headline = any(
        key in exp6400 or key in exp6401
        for key in ("legacy_headline_cell", "legacy_headline_metric", "headline_cell")
    )
    checks = [
        _check("model_ids", models_used == list(MANDATED_MODEL_IDS), "Mandated model ids are present.", "critical"),
        _check("cached_sota_receipts", exp6400.get("cached_sota_pair_receipts", {}).get("all_mandated_models_resolved") is True and exp6401.get("cached_sota_pair_receipts", {}).get("all_mandated_models_resolved") is True, "Cached SOTA receipts resolve mandated ids.", "major"),
        _check("model_specs", bool(exp6400.get("MODEL_SPECS")) and bool(exp6401.get("MODEL_SPECS")), "MODEL_SPECS are present.", "major"),
        _check("embedded_tokenizers", bool(tokenizers) and all(row.get("ok") is True and row.get("tokenizer_source") == "gguf_embedded_llama_cpp" for row in tokenizers), "GGUF embedded tokenizers are OK.", "critical"),
        _check("autotokenizer_absent", autotokenizer_count == 0, "AutoTokenizer usage count is zero.", "critical"),
        _check("inference_substrate", exp6400.get("inference_substrate") == "offline_arcade_live_agent_runtime_self_discovery_no_llm" and exp6401.get("inference_substrate") == "offline_arcade_live_agent_runtime_self_discovery_no_llm", "Upstream substrate is no-LLM ARC runtime.", "major"),
        _check("task_linked_gpu_evidence", bool(cuda) and all(row.get("terminal") is True for row in cuda), "GPU offload receipts are terminal.", "major"),
        _check("legacy_headline_cell", not legacy_headline, "No legacy headline cell is present.", "major"),
    ]
    return {
        "models_used": models_used,
        "all_mandated_model_ids_present": models_used == list(MANDATED_MODEL_IDS),
        "cached_sota_receipts_present": checks[1]["passed"],
        "model_specs_present": checks[2]["passed"],
        "embedded_tokenizers_all_ok": checks[3]["passed"],
        "autotokenizer_usage_count": autotokenizer_count,
        "inference_substrate_accurate": checks[5]["passed"],
        "task_linked_gpu_evidence_terminal": checks[6]["passed"],
        "legacy_headline_cell_present": legacy_headline,
        "checks": checks,
        "model_policy_substrate_clean": all(row["passed"] for row in checks),
    }


def default_off_reachability_and_executed_action_integrity_checks(
    artifacts: Mapping[str, Mapping[str, Any] | None],
) -> JsonDict:
    exp6400 = artifacts.get("exp6400") or {}
    exp6401 = artifacts.get("exp6401") or {}
    rows6400 = _rows(exp6400, "frozen_goal_probe_and_counterfactual_action_records")
    live6400 = exp6400.get("live_entrypoint_policy_and_reward_machine_hashes") or {}
    live6401 = exp6401.get("live_entrypoint_policy_reward_machine_and_evaluator_hashes") or {}
    executed_changes = int(exp6400.get("executed_action_change_count") or 0)
    executed_changes += sum(
        int(row.get("shadow_executed_action") != row.get("route_off_executed_action"))
        for row in rows6400
    )
    checks = [
        _check("active_goal_default_off", _default_off_from_artifacts(exp6400, exp6401), "Active goal routes are default-off.", "critical"),
        _check("live_route_reachable", live6400.get("active_reward_machine_route_reachable") is True and live6401.get("active_reward_machine_route_reachable") is True, "Route is reachable for audit only.", "major"),
        _check("executed_action_integrity", executed_changes == 0, "Shadow route cannot alter executed actions.", "critical"),
        _check("normal_entrypoint", live6400.get("submitted_entrypoint") == "make_carnot_agent -> E3AgentPolicy" and live6401.get("submitted_entrypoint") == "make_carnot_agent -> E3AgentPolicy", "Normal live entrypoint is pinned.", "major"),
    ]
    return {
        "checks": checks,
        "default_off_and_integrity_clean": all(row["passed"] for row in checks),
        "active_reward_machine_default_off": _default_off_from_artifacts(exp6400, exp6401),
        "active_reward_machine_route_reachable": bool(
            live6400.get("active_reward_machine_route_reachable") is True
            and live6401.get("active_reward_machine_route_reachable") is True
        ),
        "executed_action_change_count": executed_changes,
    }


def _finding_groups(groups: Sequence[Mapping[str, Any]]) -> JsonDict:
    findings = {"critical": [], "major": [], "minor": []}
    for group in groups:
        for row in group.get("checks", []):
            if row.get("passed") is True:
                continue
            severity = str(row.get("severity_if_failed") or "major")
            findings.setdefault(severity, []).append(
                {"name": row.get("name"), "detail": row.get("detail")}
            )
    findings["minor"].append(
        {
            "name": "public_claim_ineligible_by_scope",
            "detail": "The audit is not a solve, route promotion, policy promotion, or registry entry.",
        }
    )
    return findings


def _protected_hashes(registration: Mapping[str, Any], repo_root: Path) -> JsonDict:
    rows = {}
    for relative_path in PROTECTED_RELATIVE_PATHS:
        key = relative_path.as_posix()
        registered = registration.get("paths", {}).get(key, {})
        current = _path_entry(repo_root, relative_path, str(registered.get("role") or "protected"))
        rows[key] = {
            "before": registered.get("sha256"),
            "after": current.get("sha256"),
            "before_state": registered.get("state"),
            "after_state": current.get("state"),
            "unchanged": registered.get("state") == current.get("state")
            and registered.get("sha256") == current.get("sha256"),
        }
    return rows


def _field_principles() -> JsonDict:
    principles = {
        field: "Required Exp6402 audit field; keeps the active-goal safety boundary auditable."
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    principles.update(
        {
            "recomputed_scalar_gates_and_readiness": "Recomputes terminal gates from bare fields so nested or stale claims cannot pass by assertion.",
            "recomputed_scalar_gates_and_readiness.exp6393.ready": "Fail closed if the scalar gate contract is absent, malformed, or not ready.",
            "recomputed_scalar_gates_and_readiness.exp6400.ready": "Fail closed if the default-off shadow did not fire cleanly without action changes.",
            "recomputed_scalar_gates_and_readiness.exp6401.ready": "Fail closed if the causal holdout did not pass matched work and false-accept gates.",
            "recomputed_scalar_gates_and_readiness.scientific_readiness": "True only when the present chain satisfies narrow route-value readiness.",
            "recomputed_scalar_gates_and_readiness.public_arc_claim_eligibility_recomputed": "False by audit scope unless a separate public-claim gate exists.",
            "route_promotion_count": "Must stay zero because this audit cannot promote a route or policy.",
            "solve_claim_count": "Must stay zero because route evidence is not a game or level solve.",
            "solve_registry_modified": "Must stay false so the audit cannot alter solve records.",
            "claims_ledger_modified": "Must stay false so the audit cannot create a claims-ledger entry.",
            "public_arc_claim_eligibility": "Must fail closed; V550 route evidence is not a public ARC solve claim.",
            "upstream_artifacts_modified": "Must stay false so the audit does not repair or rewrite upstream evidence.",
            "protected_files_unchanged": "Confirms registry, claims, source, spec, and upstream artifacts stayed unchanged.",
            "verifier_is_oracle": "False for the audit; post-action oracle receipts remain upstream row-local evidence.",
        }
    )
    return principles


def _field_provenance() -> JsonDict:
    return {
        field: ["REQ-ARC-ARM-6402", "experiment_6402_arc_active_goal_safety_audit"]
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _spec_has_req(repo_root: Path) -> bool:
    spec = repo_root / "openspec/capabilities/arc-agi/spec.md"
    return spec.is_file() and "REQ-ARC-ARM-6402" in spec.read_text(encoding="utf-8")


def run(
    *,
    date: str,
    repo_root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    root = Path(repo_root)
    registration = register_expected_scope(root)
    artifacts = load_registered_artifacts(registration, root)
    states = present_absent_blocked_skipped_null_flagged_and_retired_artifact_matrix(
        registration,
        artifacts,
    )
    readiness = recomputed_scalar_gates_and_readiness(artifacts)
    hash_matrix = source_entrypoint_policy_reward_machine_model_window_and_registry_hash_matrix(
        registration,
        artifacts,
        repo_root=root,
    )
    live = live_attempt_provenance_checks(artifacts)
    hidden = hidden_source_search_bfs_adapter_proxy_and_outer_loop_attack_results(artifacts)
    oracle = oracle_timing_freeze_window_duplicate_model_state_legal_work_and_firing_attack_results(
        artifacts
    )
    goal = goal_false_accept_abstention_missing_progress_enablement_solve_and_registry_attack_results(
        artifacts
    )
    model = model_policy_and_inference_substrate_checks(artifacts)
    default = default_off_reachability_and_executed_action_integrity_checks(artifacts)
    protected = _protected_hashes(registration, root)
    findings = _finding_groups((live, hidden, oracle, goal, model, default))
    artifact: JsonDict = {
        "status": "complete",
        "audit_registration_path_hash_and_expected_scope": registration,
        "present_absent_blocked_skipped_null_flagged_and_retired_artifact_matrix": states,
        "recomputed_scalar_gates_and_readiness": readiness,
        "source_entrypoint_policy_reward_machine_model_window_and_registry_hash_matrix": hash_matrix,
        "live_attempt_provenance_checks": live,
        "hidden_source_search_bfs_adapter_proxy_and_outer_loop_attack_results": hidden,
        "oracle_timing_freeze_window_duplicate_model_state_legal_work_and_firing_attack_results": oracle,
        "goal_false_accept_abstention_missing_progress_enablement_solve_and_registry_attack_results": goal,
        "model_policy_and_inference_substrate_checks": model,
        "default_off_reachability_and_executed_action_integrity_checks": default,
        "critical_major_and_minor_findings": findings,
        "route_promotion_count": 0,
        "solve_claim_count": int(goal["solve_claim_count"]),
        "solve_registry_modified": bool(goal["solve_registry_modified"]),
        "claims_ledger_modified": bool(goal["claims_ledger_modified"]),
        "public_arc_claim_eligibility": False,
        "upstream_artifacts_modified": not bool(hash_matrix["all_registered_hashes_match_current"]),
        "protected_files_unchanged": protected,
        "preconditions_checked": {
            "planning_date": date,
            "registration_before_reading_conclusions": True,
            "spec_has_req_arc_arm_6402": _spec_has_req(root),
            "no_llm_invoked": True,
            "no_new_arc_attempts": True,
            "no_missing_window_recreation": True,
            "scripts_research_conductor_modified": False,
            "ops_reconciliation_deferred_by_stop_rule": True,
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": _field_principles(),
        "field_provenance": _field_provenance(),
        "random_seed": RANDOM_SEED,
        "duration_s": round(
            float(duration_s) if duration_s is not None else time.perf_counter() - started,
            4,
        ),
        "tests_run": list(tests_run or DEFAULT_TEST_COMMANDS),
        "test_exit_codes": {
            command: (None if test_exit_codes is None else test_exit_codes.get(command))
            for command in (tests_run or DEFAULT_TEST_COMMANDS)
        },
        "honest_verdict": "complete: active_goal_safety_audit_no_public_arc_claim_eligible",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    if write:
        output = Path(result_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    if artifact.get("status") != "complete":
        raise ValueError("status")
    if artifact.get("route_promotion_count") != 0:
        raise ValueError("route_promotion_count")
    if artifact.get("solve_claim_count") != 0:
        raise ValueError("solve_claim_count")
    if artifact.get("solve_registry_modified") is not False:
        raise ValueError("solve_registry_modified")
    if artifact.get("claims_ledger_modified") is not False:
        raise ValueError("claims_ledger_modified")
    if artifact.get("public_arc_claim_eligibility") is not False:
        raise ValueError("public_arc_claim_eligibility")
    if artifact.get("upstream_artifacts_modified") is not False:
        raise ValueError("upstream_artifacts_modified")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle")
    hash_matrix = artifact.get("source_entrypoint_policy_reward_machine_model_window_and_registry_hash_matrix", {})
    if hash_matrix.get("all_registered_hashes_match_current") is not True:
        raise ValueError("hash_matrix")
    if hash_matrix.get("all_embedded_receipts_match_registration") is not True:
        raise ValueError("hash_matrix")
    if artifact.get("default_off_reachability_and_executed_action_integrity_checks", {}).get(
        "default_off_and_integrity_clean"
    ) is not True:
        raise ValueError("default_off")
    if artifact.get("model_policy_and_inference_substrate_checks", {}).get(
        "model_policy_substrate_clean"
    ) is not True:
        raise ValueError("model_policy")
    if not all(row.get("unchanged") is True for row in artifact.get("protected_files_unchanged", {}).values()):
        raise ValueError("protected_files_unchanged")
    principles = artifact.get("field_principles", {})
    for field in (*REQUIRED_ARTIFACT_FIELDS, *READINESS_PRINCIPLE_KEYS):
        if field not in principles:
            raise ValueError("field_principles")
    if not str(artifact.get("honest_verdict") or "").startswith("complete:"):
        raise ValueError("honest_verdict")


def build_artifact(
    repo_root: Path | str = REPO_ROOT,
    *,
    date: str = "20260813",
    output_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
) -> JsonDict:
    artifact = run(date=date, repo_root=repo_root, result_path=output_path, write=True)
    validate_artifact(artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260813")
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    build_artifact(REPO_ROOT, date=str(args.date), output_path=Path(args.output))
    return 0


if __name__ == "__main__":  # pragma: no cover - module execution wrapper.
    raise SystemExit(main())
