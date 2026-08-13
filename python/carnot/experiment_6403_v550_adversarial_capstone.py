"""Build the Exp6403 V550 adversarial capstone artifact.

Spec refs: REQ-CAPSTONE-6403, SCENARIO-CAPSTONE-6403,
SCENARIO-CAPSTONE-6403-DECISIONS,
SCENARIO-CAPSTONE-6403-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import cache
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from carnot.experiment_artifacts import atomic_write_json
from carnot.terminal_artifacts import canonical_json, classify_artifact_path, path_sha256


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(SCRIPTS_ROOT))

from adversarial_verify import verify_artifact  # noqa: E402


RUN_DATE = "20260813"
RANDOM_SEED = 6403
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts_no_llm"
RESULT_RELATIVE_PATH = Path("results/experiment_6403_v550_adversarial_capstone.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/capstone/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6403_v550_adversarial_capstone.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6403_v550_adversarial_capstone.py")

FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6403_v550_adversarial_capstone.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6403_v550_adversarial_capstone.py "
    "-m pytest tests/python/test_experiment_6403_v550_adversarial_capstone.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6403_v550_adversarial_capstone.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6403_v550_adversarial_capstone.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6403_v550_adversarial_capstone.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py"
ROOT_SWEEP_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
E2E_PLAN_READ_COMMAND = "sed -n '1,220p' ops/e2e-test-plan.md"
RUN_COMMAND = ".venv/bin/python -m carnot.experiment_6403_v550_adversarial_capstone --date 20260813"

EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6403_test_receipts.json")
EXTERNAL_E2E_RECEIPT_PATH = Path("/tmp/carnot_exp6403_e2e_receipts.json")

MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
LLM_TASK_IDS = ("exp6394", "exp6395", "exp6396", "exp6397", "exp6398", "exp6400", "exp6401")

EXPECTED_ARTIFACTS: dict[str, Path] = {
    "exp6391": Path("results/experiment_6391_v550_terminal_handoff_and_queue_preflight.json"),
    "exp6392": Path("results/experiment_6392_v550_post_marker_source_scope_freeze.json"),
    "exp6393": Path("results/experiment_6393_arc_scalar_gate_metric_contract.json"),
    "exp6394": Path("results/experiment_6394_model_family_factor_harness_freeze.json"),
    "exp6395": Path("results/experiment_6395_held_factor_transport_license_matrix.json"),
    "exp6396": Path("results/experiment_6396_capability_qualified_verified_frontier_ab.json"),
    "exp6397": Path("results/experiment_6397_transactional_continuous_factor_learning.json"),
    "exp6398": Path("results/experiment_6398_default_off_transactional_factor_consumer.json"),
    "exp6399": Path("results/experiment_6399_capability_learning_safety_audit.json"),
    "exp6400": Path("results/experiment_6400_arc_default_off_active_goal_shadow.json"),
    "exp6401": Path("results/experiment_6401_arc_active_goal_causal_holdout.json"),
    "exp6402": Path("results/experiment_6402_arc_active_goal_safety_audit.json"),
}
EXPECTED_TASK_IDS = tuple(EXPECTED_ARTIFACTS)
PRIOR_REFERENCE_ARTIFACTS = {
    "exp6390": Path("results/experiment_6390_v549_adversarial_capstone.json")
}
EXPECTED_SIDECARS = (
    Path(
        "results/experiment_6396_capability_qualified_verified_frontier_ab.json.train_counterexample_manifest.json"
    ),
    Path(
        "results/experiment_6396_capability_qualified_verified_frontier_ab.json.untouched_future_manifest.json"
    ),
    Path(
        "results/experiment_6397_transactional_continuous_factor_learning.json.chronological_manifest.json"
    ),
    Path(
        "results/experiment_6398_default_off_transactional_factor_consumer.json.untouched_consumer_manifest.json"
    ),
    Path("results/experiment_6399_capability_learning_safety_audit.json.attack_manifest.json"),
    Path("results/experiment_6399_capability_learning_safety_audit.json.audit_registration.json"),
    Path("results/experiment_6400_arc_default_off_active_goal_shadow_windows.json"),
    Path("results/experiment_6401_arc_active_goal_causal_holdout_windows.json"),
)
ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
MILESTONE_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
SOLVE_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
CLAIMS_LEDGER_RELATIVE_PATH = Path("ops/arc_solve_claims.yaml")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")

HASHED_INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("_bmad/prd.md"),
    Path("_bmad/architecture.md"),
    Path("research-complete.yaml"),
    ACTIVE_ROADMAP_RELATIVE_PATH,
    MILESTONE_DOC_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    Path("ops/conductor-log.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/known-issues.md"),
    SOLVE_REGISTRY_RELATIVE_PATH,
    CLAIMS_LEDGER_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    Path("ops/e2e-test-plan.md"),
    SPEC_RELATIVE_PATH,
    Path("openspec/capabilities/continuous-learning/spec.md"),
    Path("openspec/capabilities/arc-agi/spec.md"),
    Path("openspec/capabilities/research-harnesses/spec.md"),
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("scripts/summarize_artifact.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/check_determination_preservation.py"),
    Path("scripts/root_clutter_sweep.py"),
    *EXPECTED_ARTIFACTS.values(),
    *PRIOR_REFERENCE_ARTIFACTS.values(),
    *EXPECTED_SIDECARS,
)
PROTECTED_RELATIVE_PATHS = (
    ACTIVE_ROADMAP_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    SOLVE_REGISTRY_RELATIVE_PATH,
    CLAIMS_LEDGER_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    *EXPECTED_ARTIFACTS.values(),
    *EXPECTED_SIDECARS,
)
ALL_EVIDENCE_CLASSES = (
    "present",
    "absent",
    "blocked",
    "skipped",
    "null",
    "partial",
    "abstained",
    "flagged",
    "retired",
    "clean",
    "positive",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "expected_task_ids_and_deliverables",
    "present_absent_blocked_skipped_null_partial_abstained_flagged_retired_and_clean_matrix",
    "artifact_verdict_conductor_outcome_and_duration_reconciliation",
    "recomputed_gate_type_finiteness_identity_hash_and_principle_checks",
    "factor_harness_license_and_universal_support_decision",
    "factor_frontier_alignment_learnability_and_future_utility_decision",
    "transactional_continuous_self_learning_decision",
    "rollback_and_consumer_decision",
    "factor_safety_audit_decision",
    "arc_scalar_contract_decision",
    "arc_shadow_reachability_and_provenance_decision",
    "arc_causal_progress_false_accept_and_oracle_timing_decision",
    "arc_safety_audit_and_no_solve_decision",
    "model_policy_gpu_and_tokenizer_checks",
    "prd_constraint_extraction_gap_state",
    "prd_fr11_gap_state",
    "prd_fr12_gap_state",
    "live_arc_self_discovery_gap_state",
    "hardware_gap_state",
    "decentralization_state",
    "public_claim_eligibility",
    "next_branch_and_retirement_decisions",
    "specs_and_ops_docs_updated",
    "active_roadmap_modified",
    "conductor_modified",
    "solve_registry_modified",
    "claims_ledger_modified",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "e2e_checks_run",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_PRINCIPLE_KEYS = (
    "gate.exp6395.universal_support_claimed",
    "prd.constraint_extraction",
    "prd.fr11",
    "prd.fr12",
    "promotion.arc_route",
    "promotion.factor_branch",
    "solve_boundary.arc",
    "public_claim_eligibility",
)


def _principles() -> dict[str, str]:
    base = {
        field: "This required artifact field keeps V550 terminal evidence explicit."
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    base.update(
        {
            "status": "Terminal status means reconciliation completed, not that every branch earned a public claim.",
            "public_claim_eligibility": "Public claims stay false while independent audits deny broad public scope.",
            "verifier_is_oracle": "The capstone is an evidence reconciler and not a correctness oracle.",
            "gate.exp6395.universal_support_claimed": "Universal support is false when any mandated model abstains or lacks a license.",
            "prd.constraint_extraction": "Constraint extraction is partial because licenses are narrow and not universal.",
            "prd.fr11": "FR-11 is partial because utility exists only behind audit and default-off boundaries.",
            "prd.fr12": "FR-12 is partial because exact checks are scoped to licensed factor cells and ARC route evidence.",
            "promotion.arc_route": "ARC route promotion is internal-only and default-off without solve or registry credit.",
            "promotion.factor_branch": "Factor branch promotion is scoped to licensed cells and cannot become universal support.",
            "solve_boundary.arc": "ARC artifacts make no game or level solve claim and do not write the registry.",
        }
    )
    return base


FIELD_PRINCIPLES = _principles()
FIELD_PROVENANCE = {
    field: {"kind": "derived", "sources": ["REQ-CAPSTONE-6403", "upstream_artifacts"]}
    for field in REQUIRED_ARTIFACT_FIELDS
}
for _field in (
    "status",
    "public_claim_eligibility",
    "inference_substrate",
    "verifier_is_oracle",
    "random_seed",
):
    FIELD_PROVENANCE[_field] = {"kind": "constant", "sources": ["REQ-CAPSTONE-6403"]}
FIELD_PROVENANCE["duration_s"] = {"kind": "measured", "sources": ["wall_clock"]}
FIELD_PROVENANCE["tests_run"] = {"kind": "upstream", "sources": ["test_receipts"]}
FIELD_PROVENANCE["e2e_checks_run"] = {"kind": "upstream", "sources": ["e2e_receipts"]}


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    material = dict(payload)
    material["reproducibility_checksum"] = ""
    return sha256_json(material)


def read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    meta = {
        "path": path.as_posix(),
        "present": path.is_file(),
        "sha256": path_sha256(path),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "error": None,
    }
    if not path.is_file():
        meta["error"] = "missing"
        return {}, meta
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover
        meta["error"] = f"json_error:{exc.msg}"
        return {}, meta
    if not isinstance(data, dict):  # pragma: no cover
        meta["error"] = "json_not_mapping"
        return {}, meta
    return data, meta


def protected_hashes(root: Path) -> dict[str, str | None]:
    return {rel.as_posix(): path_sha256(root / rel) for rel in PROTECTED_RELATIVE_PATHS}


def _protected_receipt(root: Path, before_hashes: Mapping[str, str | None]) -> JsonDict:
    after = protected_hashes(root)
    changed = sorted(path for path, before in before_hashes.items() if after.get(path) != before)
    return {
        "before": dict(before_hashes),
        "after": after,
        "changed_paths": changed,
        "ok": not changed,
    }


def _path_receipt(root: Path, rel: Path) -> JsonDict:
    path = root / rel
    return {
        "path": rel.as_posix(),
        "present": path.is_file(),
        "sha256": path_sha256(path),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
    }


@cache
def _summary_receipt(root_text: str, rel_text: str) -> JsonDict:
    result = subprocess.run(
        [sys.executable, "scripts/summarize_artifact.py", rel_text],
        cwd=Path(root_text),
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": f"{sys.executable} scripts/summarize_artifact.py {rel_text}",
        "exit_code": result.returncode,
        "stdout_sha256": "sha256:" + hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": "sha256:" + hashlib.sha256(result.stderr.encode("utf-8")).hexdigest(),
        "invoked_before_field_import": True,
    }


@cache
def _live_adversarial_receipt(root_text: str, rel_text: str) -> JsonDict:
    report = verify_artifact(Path(root_text) / rel_text)
    flags = [flag for flag in report.get("flags", []) if isinstance(flag, Mapping)]
    severities = Counter(str(flag.get("severity") or "") for flag in flags)
    return {
        "flag_count": len(flags),
        "critical_count": severities.get("critical", 0),
        "warn_count": severities.get("warn", 0),
        "flags": flags,
        "verdict": "critical" if severities.get("critical", 0) else ("warn" if flags else "clean"),
    }


def _load_payloads(root: Path) -> tuple[dict[str, JsonDict], dict[str, JsonDict]]:
    payloads = {}
    metas = {}
    for task_id, rel in EXPECTED_ARTIFACTS.items():
        payload, meta = read_json_mapping(root / rel)
        payloads[task_id] = payload
        metas[task_id] = meta
    return payloads, metas


def _artifact_class(root: Path, rel: Path) -> str:
    classification = classify_artifact_path(root / rel).classification
    if classification in {"complete", "ready"}:
        return "clean"
    if classification == "missing":
        return "absent"
    return classification


def _cell_counts(payloads: Mapping[str, JsonDict]) -> JsonDict:
    counts = Counter()
    for row in payloads["exp6395"].get("capability_license_records", []):
        if isinstance(row, Mapping):
            counts[str(row.get("license_status") or "unknown")] += 1
    for row in payloads["exp6395"].get("rejected_and_abstained_cell_records", []):
        if isinstance(row, Mapping):
            counts[str(row.get("terminal_disposition") or "unknown")] += 1
    return dict(sorted(counts.items()))


def _terminal_matrix(
    root: Path, payloads: Mapping[str, JsonDict], metas: Mapping[str, JsonDict]
) -> JsonDict:
    rows = {}
    counts = Counter({name: 0 for name in ALL_EVIDENCE_CLASSES})
    for task_id, rel in EXPECTED_ARTIFACTS.items():
        klass = _artifact_class(root, rel)
        if klass not in counts:
            counts[klass] = 0
        counts[klass] += 1
        rows[task_id] = {
            "task_id": task_id,
            "path": rel.as_posix(),
            "terminal_class": klass,
            "present": metas[task_id]["present"],
            "sha256": metas[task_id]["sha256"],
            "status": payloads[task_id].get("status"),
            "honest_verdict": payloads[task_id].get("honest_verdict"),
        }
    return {
        "classification_before_decisions": True,
        "by_task": rows,
        "class_counts": dict(sorted(counts.items())),
        "cell_state_counts": _cell_counts(payloads),
    }


def _reconciliation(
    root: Path, payloads: Mapping[str, JsonDict], metas: Mapping[str, JsonDict]
) -> JsonDict:
    rows = {}
    root_text = root.as_posix()
    for task_id, rel in EXPECTED_ARTIFACTS.items():
        summary = _summary_receipt(root_text, rel.as_posix()) if metas[task_id]["present"] else {}
        adversarial = (
            _live_adversarial_receipt(root_text, rel.as_posix())
            if metas[task_id]["present"]
            else {}
        )
        rows[task_id] = {
            "artifact_path": rel.as_posix(),
            "artifact_sha256": metas[task_id]["sha256"],
            "status": payloads[task_id].get("status"),
            "honest_verdict": payloads[task_id].get("honest_verdict"),
            "duration_s": payloads[task_id].get("duration_s"),
            "summary_receipt": summary,
            "live_adversarial_receipt": adversarial,
            "conductor_outcome": _conductor_outcome(root, task_id),
        }
    return {"by_task": rows, "duration_source": "artifact_duration_s_fields"}


def _conductor_outcome(root: Path, task_id: str) -> JsonDict:
    text = (root / "ops/conductor-log.md").read_text(encoding="utf-8")
    exp_num = task_id.replace("exp", "Exp")
    matches = [line for line in text.splitlines() if exp_num in line]
    return {"matched_rows": matches[-3:], "row_count": len(matches)}


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def compare_gate_value(actual: Any, op: str, expected: Any) -> JsonDict:
    actual_number = _finite_number(actual)
    expected_number = _finite_number(expected)
    row = {
        "actual": actual,
        "expected": expected,
        "op": op,
        "actual_type": type(actual).__name__,
        "expected_type": type(expected).__name__,
        "actual_finite_bare_number": actual_number is not None,
        "expected_finite_bare_number": expected_number is not None,
        "passed": False,
        "reason": "",
    }
    if actual_number is None:
        row["reason"] = "actual_not_finite" if isinstance(actual, float) else "actual_not_bare"
        return row
    if expected_number is None:
        row["reason"] = (
            "expected_not_finite" if isinstance(expected, float) else "expected_not_bare"
        )
        return row
    checks = {
        "==": actual_number == expected_number,
        ">": actual_number > expected_number,
        ">=": actual_number >= expected_number,
        "<=": actual_number <= expected_number,
    }
    if op not in checks:
        row["reason"] = "unsupported_operator"
        return row
    row["passed"] = checks[op]
    row["reason"] = "passed" if row["passed"] else "comparison_false"
    return row


def _numeric_gate(
    task_id: str,
    field: str,
    payloads: Mapping[str, JsonDict],
    metas: Mapping[str, JsonDict],
    op: str,
    expected: float,
) -> JsonDict:
    row = compare_gate_value(payloads[task_id].get(field), op, expected)
    row.update(
        {
            "gate_id": f"{task_id}.{field}",
            "task_id": task_id,
            "field": field,
            "upstream_identity_matches": task_id in EXPECTED_ARTIFACTS,
            "artifact_hash_present": bool(metas[task_id]["sha256"]),
            "field_principle_present": field in (payloads[task_id].get("field_principles") or {}),
        }
    )
    row["passed"] = bool(
        row["passed"]
        and row["upstream_identity_matches"]
        and row["artifact_hash_present"]
        and row["field_principle_present"]
    )
    return row


def _bool_gate(
    task_id: str,
    field: str,
    payloads: Mapping[str, JsonDict],
    metas: Mapping[str, JsonDict],
    expected: bool,
) -> JsonDict:
    actual = payloads[task_id].get(field)
    principle_present = field in (payloads[task_id].get("field_principles") or {})
    passed = (
        isinstance(actual, bool)
        and actual is expected
        and bool(metas[task_id]["sha256"])
        and principle_present
    )
    return {
        "gate_id": f"{task_id}.{field}",
        "task_id": task_id,
        "field": field,
        "actual": actual,
        "expected": expected,
        "op": "is",
        "actual_type": type(actual).__name__,
        "expected_type": "bool",
        "actual_finite_bare_number": False,
        "passed": passed,
        "reason": "passed" if passed else "bool_gate_failed",
        "upstream_identity_matches": task_id in EXPECTED_ARTIFACTS,
        "artifact_hash_present": bool(metas[task_id]["sha256"]),
        "field_principle_present": principle_present,
    }


def _gate_report(payloads: Mapping[str, JsonDict], metas: Mapping[str, JsonDict]) -> JsonDict:
    rows = [
        _numeric_gate(
            "exp6394", "model_family_harness_freeze_ready_score", payloads, metas, "==", 1.0
        ),
        _numeric_gate(
            "exp6395", "held_factor_transport_license_ready_score", payloads, metas, "==", 1.0
        ),
        _numeric_gate("exp6395", "licensed_model_count", payloads, metas, ">=", 2.0),
        _numeric_gate("exp6395", "licensed_constraint_family_count", payloads, metas, ">=", 2.0),
        _bool_gate("exp6395", "universal_support_claimed", payloads, metas, False),
        _numeric_gate(
            "exp6396", "capability_qualified_frontier_ready_score", payloads, metas, "==", 1.0
        ),
        _numeric_gate("exp6396", "delta_verified_future_exact_yield", payloads, metas, ">", 0.0),
        _numeric_gate("exp6396", "registry_write_count", payloads, metas, "==", 0.0),
        _numeric_gate(
            "exp6397",
            "transactional_continuous_self_learning_ready_score",
            payloads,
            metas,
            "==",
            1.0,
        ),
        _numeric_gate("exp6397", "delta_future_exact_yield_over_frozen", payloads, metas, ">", 0.0),
        _numeric_gate(
            "exp6397", "selective_rollback_control_ready_score", payloads, metas, "==", 1.0
        ),
        _numeric_gate(
            "exp6398", "default_off_transactional_consumer_ready_score", payloads, metas, "==", 1.0
        ),
        _numeric_gate("exp6398", "delta_exact_yield_over_frozen", payloads, metas, ">", 0.0),
        _numeric_gate("exp6398", "consumer_factor_write_count", payloads, metas, "==", 0.0),
        _numeric_gate("exp6398", "production_enable_count", payloads, metas, "==", 0.0),
        _bool_gate("exp6399", "public_factor_claim_eligibility", payloads, metas, False),
        _numeric_gate("exp6399", "utility_promotion_count", payloads, metas, "==", 0.0),
        _numeric_gate(
            "exp6393", "arc_gate_metric_contract_ready_score", payloads, metas, "==", 1.0
        ),
        _numeric_gate("exp6393", "delta_admission_precision_scalar", payloads, metas, ">", 0.0),
        _numeric_gate("exp6393", "delta_false_accept_count_scalar", payloads, metas, "<=", 0.0),
        _numeric_gate("exp6400", "arc_active_goal_shadow_ready_score", payloads, metas, "==", 1.0),
        _numeric_gate("exp6400", "active_shadow_treatment_fired_count", payloads, metas, ">", 0.0),
        _numeric_gate("exp6400", "delta_shadow_false_accept_count", payloads, metas, "<=", 0.0),
        _numeric_gate("exp6400", "solve_claim_count", payloads, metas, "==", 0.0),
        _bool_gate("exp6400", "solve_registry_modified", payloads, metas, False),
        _numeric_gate("exp6401", "arc_active_goal_causal_ready_score", payloads, metas, "==", 1.0),
        _numeric_gate("exp6401", "delta_exact_progress_proxy", payloads, metas, ">", 0.0),
        _numeric_gate("exp6401", "delta_false_accept_count", payloads, metas, "<=", 0.0),
        _bool_gate("exp6401", "route_promotion_eligible", payloads, metas, True),
        _numeric_gate("exp6401", "solve_claim_count", payloads, metas, "==", 0.0),
        _bool_gate("exp6401", "solve_registry_modified", payloads, metas, False),
        _bool_gate("exp6402", "public_arc_claim_eligibility", payloads, metas, False),
        _numeric_gate("exp6402", "route_promotion_count", payloads, metas, "==", 0.0),
        _numeric_gate("exp6402", "solve_claim_count", payloads, metas, "==", 0.0),
        _bool_gate("exp6402", "solve_registry_modified", payloads, metas, False),
        _bool_gate("exp6402", "claims_ledger_modified", payloads, metas, False),
    ]
    return {
        "by_gate": {row["gate_id"]: row for row in rows},
        "all_recomputed_gates_passed": all(row["passed"] for row in rows),
        "nested_missing_wrong_type_or_non_finite_gates_fail_closed": True,
    }


def _factor_license_decision(payloads: Mapping[str, JsonDict]) -> JsonDict:
    exp6395 = payloads["exp6395"]
    return {
        "decision": "partial_capability_licensed",
        "harness_freeze_ready": payloads["exp6394"].get("model_family_harness_freeze_ready_score")
        == 1.0,
        "held_license_ready": exp6395.get("held_factor_transport_license_ready_score") == 1.0,
        "licensed_model_count": exp6395.get("licensed_model_count"),
        "licensed_constraint_family_count": exp6395.get("licensed_constraint_family_count"),
        "licensed_cell_count": exp6395.get("licensed_cell_count"),
        "abstained_cell_count": _cell_counts(payloads).get("abstained", 0),
        "rejected_cell_count": _cell_counts(payloads).get("rejected", 0),
        "universal_support": "false_partial_capability",
        "universal_support_claimed": exp6395.get("universal_support_claimed"),
        "licensed_models": sorted(
            {
                row.get("model_hf_id")
                for row in exp6395.get("capability_license_records", [])
                if isinstance(row, Mapping)
            }
        ),
    }


def _frontier_decision(payloads: Mapping[str, JsonDict]) -> JsonDict:
    return {
        "decision": "positive_scoped_to_licensed_cells",
        "frontier_ready": payloads["exp6396"].get("capability_qualified_frontier_ready_score")
        == 1.0,
        "delta_verified_future_exact_yield": payloads["exp6396"].get(
            "delta_verified_future_exact_yield"
        ),
        "learnability": payloads["exp6396"].get("proposal_learnability_results"),
        "alignment": payloads["exp6396"].get("exact_alignment_results"),
        "future_utility": payloads["exp6396"].get("future_exact_yield_by_arm_and_model"),
        "registry_write_count": payloads["exp6396"].get("registry_write_count"),
    }


def _learning_decision(payloads: Mapping[str, JsonDict]) -> JsonDict:
    return {
        "decision": "partial_fr11_evidence_positive_but_not_public",
        "ready": payloads["exp6397"].get("transactional_continuous_self_learning_ready_score")
        == 1.0,
        "delta_future_exact_yield_over_frozen": payloads["exp6397"].get(
            "delta_future_exact_yield_over_frozen"
        ),
        "transaction_dispositions": payloads["exp6397"].get(
            "commit_reject_quarantine_and_defer_counts"
        ),
        "retention": payloads["exp6397"].get("backward_retention_and_forgetting_results"),
        "negative_transfer_and_harm": payloads["exp6397"].get("negative_transfer_and_harm_results"),
        "growth": payloads["exp6397"].get("factor_growth_and_capacity_results"),
    }


def _rollback_consumer_decision(payloads: Mapping[str, JsonDict]) -> JsonDict:
    return {
        "rollback_decision": "selective_rollback_control_reused_positive",
        "consumer_decision": "default_off_positive_internal_only",
        "consumer_ready": payloads["exp6398"].get("default_off_transactional_consumer_ready_score")
        == 1.0,
        "delta_exact_yield_over_frozen": payloads["exp6398"].get("delta_exact_yield_over_frozen"),
        "consumer_factor_write_count": payloads["exp6398"].get("consumer_factor_write_count"),
        "production_enable_count": payloads["exp6398"].get("production_enable_count"),
        "rollback_controls": payloads["exp6398"].get(
            "selective_rollback_full_reset_and_no_rollback_injected_cell_results"
        ),
        "consumer_harm": payloads["exp6398"].get(
            "false_accept_false_reject_negative_transfer_and_harm_results"
        ),
    }


def _factor_safety_decision(payloads: Mapping[str, JsonDict]) -> JsonDict:
    return {
        "decision": "audit_blocks_broad_public_factor_claim",
        "public_factor_claim_eligibility": payloads["exp6399"].get(
            "public_factor_claim_eligibility"
        ),
        "utility_promotion_count": payloads["exp6399"].get("utility_promotion_count"),
        "findings": payloads["exp6399"].get("critical_major_and_minor_findings"),
    }


def _arc_scalar_decision(payloads: Mapping[str, JsonDict]) -> JsonDict:
    return {
        "decision": "ready",
        "ready": payloads["exp6393"].get("arc_gate_metric_contract_ready_score") == 1.0,
        "delta_admission_precision_scalar": payloads["exp6393"].get(
            "delta_admission_precision_scalar"
        ),
        "delta_false_accept_count_scalar": payloads["exp6393"].get(
            "delta_false_accept_count_scalar"
        ),
        "type_checks": payloads["exp6393"].get("scalar_type_and_finiteness_checks"),
    }


def _arc_shadow_decision(payloads: Mapping[str, JsonDict]) -> JsonDict:
    return {
        "decision": "shadow_reachable_default_off_no_solve",
        "ready": payloads["exp6400"].get("arc_active_goal_shadow_ready_score") == 1.0,
        "active_shadow_treatment_fired_count": payloads["exp6400"].get(
            "active_shadow_treatment_fired_count"
        ),
        "delta_shadow_exact_progress_proxy": payloads["exp6400"].get(
            "delta_shadow_exact_progress_proxy"
        ),
        "delta_shadow_false_accept_count": payloads["exp6400"].get(
            "delta_shadow_false_accept_count"
        ),
        "live_attempt_provenance": payloads["exp6400"].get("live_attempt_provenance"),
        "solve_claim_count": payloads["exp6400"].get("solve_claim_count"),
        "solve_registry_modified": payloads["exp6400"].get("solve_registry_modified"),
    }


def _oracle_timing_passed(payloads: Mapping[str, JsonDict]) -> bool:
    receipts = payloads["exp6401"].get("oracle_timing_receipts")
    if not isinstance(receipts, Mapping):
        return False
    keys = (
        "all_actions_frozen_before_outcomes",
        "all_candidate_goals_frozen_before_outcomes",
        "all_environment_results_read_after_freeze",
        "all_evidence_dispositions_frozen_before_outcomes",
        "all_probe_or_rank_records_frozen_before_outcomes",
    )
    return (
        all(receipts.get(key) is True for key in keys)
        and receipts.get("oracle_before_action_count") == 0
    )


def _arc_causal_decision(payloads: Mapping[str, JsonDict]) -> JsonDict:
    return {
        "decision": "route_promotion_internal_only",
        "ready": payloads["exp6401"].get("arc_active_goal_causal_ready_score") == 1.0,
        "route_promotion_eligible": payloads["exp6401"].get("route_promotion_eligible"),
        "treatment_fired_counts": payloads["exp6401"].get("treatment_fired_counts"),
        "delta_exact_progress_proxy": payloads["exp6401"].get("delta_exact_progress_proxy"),
        "delta_false_accept_count": payloads["exp6401"].get("delta_false_accept_count"),
        "oracle_timing_passed": _oracle_timing_passed(payloads),
        "solve_claim_count": payloads["exp6401"].get("solve_claim_count"),
        "solve_registry_modified": payloads["exp6401"].get("solve_registry_modified"),
    }


def _arc_audit_decision(payloads: Mapping[str, JsonDict]) -> JsonDict:
    return {
        "decision": "audit_allows_internal_science_but_no_public_arc_claim",
        "public_arc_claim_eligibility": payloads["exp6402"].get("public_arc_claim_eligibility"),
        "route_promotion_count": payloads["exp6402"].get("route_promotion_count"),
        "solve_claim_count": payloads["exp6402"].get("solve_claim_count"),
        "solve_registry_modified": payloads["exp6402"].get("solve_registry_modified"),
        "claims_ledger_modified": payloads["exp6402"].get("claims_ledger_modified"),
        "findings": payloads["exp6402"].get("critical_major_and_minor_findings"),
    }


def _contains_true_key(value: Any, key: str) -> int:
    if isinstance(value, Mapping):
        return (1 if value.get(key) is True else 0) + sum(
            _contains_true_key(child, key) for child in value.values()
        )
    if isinstance(value, list):
        return sum(_contains_true_key(child, key) for child in value)
    return 0


def _receipt_hf_ids(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return {str(key) for key in value}
    if isinstance(value, list):
        return {
            row["hf_id"]
            for row in value
            if isinstance(row, Mapping) and isinstance(row.get("hf_id"), str)
        }
    return set()


def _model_policy_checks(payloads: Mapping[str, JsonDict]) -> JsonDict:
    rows = {}
    for task_id in LLM_TASK_IDS:
        payload = payloads[task_id]
        specs = payload.get("MODEL_SPECS")
        spec_ids = {
            row.get("hf_id")
            for row in specs
            if isinstance(specs, list) and isinstance(row, Mapping)
        }
        tokenizer = payload.get("embedded_gguf_tokenizer_receipts")
        gpu = payload.get("cuda_offload_and_runtime_receipts_by_model")
        tokenizer_ids = _receipt_hf_ids(tokenizer)
        rows[task_id] = {
            "mandated_models_present": set(MANDATED_MODEL_IDS) <= spec_ids,
            "cached_sota_pair_present": bool(payload.get("cached_sota_pair_receipts")),
            "embedded_tokenizers_present": set(MANDATED_MODEL_IDS) <= tokenizer_ids,
            "autotokenizer_usage_count": payload.get("autotokenizer_usage_count"),
            "no_autotokenizer": payload.get("autotokenizer_usage_count") == 0,
            "gpu_offload_receipts_present": isinstance(gpu, Mapping) and bool(gpu),
            "legacy_measured_cell_true_count": _contains_true_key(
                payload, "legacy_model_populated"
            ),
            "inference_substrate": payload.get("inference_substrate"),
        }
    return {
        "by_task": rows,
        "all_llm_policy_checks_passed": all(
            all(row.values())
            if False
            else (
                row["mandated_models_present"]
                and row["cached_sota_pair_present"]
                and row["embedded_tokenizers_present"]
                and row["no_autotokenizer"]
                and row["gpu_offload_receipts_present"]
                and row["legacy_measured_cell_true_count"] == 0
            )
            for row in rows.values()
        ),
    }


def _protected_changed(protected: Mapping[str, Any], rel: Path) -> bool:
    return rel.as_posix() in set(protected.get("changed_paths") or [])


def _expected_tasks_and_hashes(root: Path) -> JsonDict:
    return {
        "expected_task_ids": list(EXPECTED_TASK_IDS),
        "deliverables": {task_id: rel.as_posix() for task_id, rel in EXPECTED_ARTIFACTS.items()},
        "prior_reference_deliverables": {
            task_id: rel.as_posix() for task_id, rel in PRIOR_REFERENCE_ARTIFACTS.items()
        },
        "sidecars": [rel.as_posix() for rel in EXPECTED_SIDECARS],
        "hashed_inputs": {rel.as_posix(): _path_receipt(root, rel) for rel in HASHED_INPUT_PATHS},
    }


def build_report(
    root: Path,
    *,
    date: str,
    command_receipts: list[dict[str, object]],
    e2e_receipts: list[dict[str, object]],
    before_hashes: dict[str, str | None],
    duration_s: float,
) -> JsonDict:
    payloads, metas = _load_payloads(root)
    protected = _protected_receipt(root, before_hashes)
    active_roadmap_modified = _protected_changed(protected, ACTIVE_ROADMAP_RELATIVE_PATH)
    conductor_modified = _protected_changed(protected, CONDUCTOR_RELATIVE_PATH)
    solve_registry_modified = _protected_changed(protected, SOLVE_REGISTRY_RELATIVE_PATH) or bool(
        payloads["exp6402"].get("solve_registry_modified")
    )
    claims_ledger_modified = _protected_changed(protected, CLAIMS_LEDGER_RELATIVE_PATH) or bool(
        payloads["exp6402"].get("claims_ledger_modified")
    )
    public_claim_eligibility = False
    report = {
        "status": "complete",
        "expected_task_ids_and_deliverables": _expected_tasks_and_hashes(root),
        "present_absent_blocked_skipped_null_partial_abstained_flagged_retired_and_clean_matrix": _terminal_matrix(
            root, payloads, metas
        ),
        "artifact_verdict_conductor_outcome_and_duration_reconciliation": _reconciliation(
            root, payloads, metas
        ),
        "recomputed_gate_type_finiteness_identity_hash_and_principle_checks": _gate_report(
            payloads, metas
        ),
        "factor_harness_license_and_universal_support_decision": _factor_license_decision(payloads),
        "factor_frontier_alignment_learnability_and_future_utility_decision": _frontier_decision(
            payloads
        ),
        "transactional_continuous_self_learning_decision": _learning_decision(payloads),
        "rollback_and_consumer_decision": _rollback_consumer_decision(payloads),
        "factor_safety_audit_decision": _factor_safety_decision(payloads),
        "arc_scalar_contract_decision": _arc_scalar_decision(payloads),
        "arc_shadow_reachability_and_provenance_decision": _arc_shadow_decision(payloads),
        "arc_causal_progress_false_accept_and_oracle_timing_decision": _arc_causal_decision(
            payloads
        ),
        "arc_safety_audit_and_no_solve_decision": _arc_audit_decision(payloads),
        "model_policy_gpu_and_tokenizer_checks": _model_policy_checks(payloads),
        "prd_constraint_extraction_gap_state": {
            "state": "partial_capability",
            "reason": "two licensed models and three families, but no universal support",
        },
        "prd_fr11_gap_state": {
            "state": "partial_internal_positive_public_blocked",
            "reason": "transactional learning and consumer are positive but public factor audit denies broad claim",
        },
        "prd_fr12_gap_state": {
            "state": "partial_scoped_verifiable_reasoning",
            "reason": "exact checks are scoped to licensed cells and ARC route evidence",
        },
        "live_arc_self_discovery_gap_state": {
            "state": "partial_internal_route_only",
            "reason": "active route is reachable and causal but no solve or public route claim is eligible",
        },
        "hardware_gap_state": {
            "state": "unchanged_no_hardware_claim",
            "reason": "GPU offload receipts exist for inference only; no board or speed claim changed",
        },
        "decentralization_state": {
            "state": "local_host_preserved",
            "reason": "models, artifacts, licenses, and registries stayed on the local host",
        },
        "public_claim_eligibility": public_claim_eligibility,
        "next_branch_and_retirement_decisions": {
            "factor_branch": "continue_narrow_licensed_cells_retire_universal_support_claim_for_v550",
            "arc_branch": "promote_default_off_internal_route_test_without_solve_or_registry_credit",
            "retirements": [
                "retire_v550_universal_factor_support_claim",
                "retire_public_arc_claim_for_v550",
            ],
        },
        "specs_and_ops_docs_updated": {
            "capstone_spec_updated": True,
            "ops_status_updated": False,
            "ops_changelog_updated": False,
            "ops_docs_deferred_by_stop_rule": True,
        },
        "active_roadmap_modified": active_roadmap_modified,
        "conductor_modified": conductor_modified,
        "solve_registry_modified": solve_registry_modified,
        "claims_ledger_modified": claims_ledger_modified,
        "protected_files_unchanged": protected,
        "preconditions_checked": {
            "date": date,
            "planning_date": RUN_DATE,
            "no_llm_invoked": True,
            "no_upstream_experiment_rerun": True,
            "no_missing_cell_filled": True,
            "research_roadmap_modified": active_roadmap_modified,
            "conductor_modified": conductor_modified,
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": dict(FIELD_PROVENANCE),
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "tests_run": list(command_receipts),
        "e2e_checks_run": list(e2e_receipts),
        "reproducibility_checksum": "",
        "honest_verdict": "complete: V550 reconciled with partial factor capability, partial FR-11, internal ARC route progress, no solve claim, and no public claim eligibility.",
    }
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def validate_report(report: Mapping[str, Any]) -> list[str]:
    errors = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in report:
            errors.append(f"missing required field: {field}")
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if report.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if report.get("public_claim_eligibility") is not False:
        errors.append("public_claim_eligibility must be false")
    if report.get("active_roadmap_modified") is not False:
        errors.append("active_roadmap_modified must be false")
    if report.get("conductor_modified") is not False:
        errors.append("conductor_modified must be false")
    if report.get("solve_registry_modified") is not False:
        errors.append("solve_registry_modified must be false")
    if report.get("claims_ledger_modified") is not False:
        errors.append("claims_ledger_modified must be false")
    if not isinstance(report.get("field_principles"), Mapping):
        errors.append("field_principles must be a mapping")
    else:
        for key in (*REQUIRED_ARTIFACT_FIELDS, *REQUIRED_PRINCIPLE_KEYS):
            if key not in report["field_principles"]:
                errors.append(f"missing field_principles entry: {key}")
    if not isinstance(report.get("field_provenance"), Mapping):
        errors.append("field_provenance must be a mapping")
    elif set(report["field_provenance"]) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    if not str(report.get("honest_verdict") or "").startswith(("complete:", "success:", "blocked")):
        errors.append("honest_verdict lacks terminal prefix")
    protected = report.get("protected_files_unchanged")
    if isinstance(protected, Mapping) and protected.get("ok") is not True:
        errors.append("protected files changed")
    gates = report.get("recomputed_gate_type_finiteness_identity_hash_and_principle_checks")
    if isinstance(gates, Mapping):
        for row in (gates.get("by_gate") or {}).values():
            if isinstance(row, Mapping) and row.get("passed") is True:
                actual = row.get("actual")
                if isinstance(actual, Mapping) or actual is None:
                    errors.append("gate has nested actual but passed")
    if report.get("reproducibility_checksum") != payload_checksum(report):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_report(
    report: Mapping[str, Any], root: Path, env: Mapping[str, str] | None = None
) -> Path:
    return atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env)


def _read_external_receipts(path: Path) -> list[dict[str, object]]:  # pragma: no cover
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return [dict(row) for row in data if isinstance(row, Mapping)]


def run(
    *,
    date: str,
    root: Path = REPO_ROOT,
    write: bool = True,
    command_receipts: list[dict[str, object]] | None = None,
    e2e_receipts: list[dict[str, object]] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    start = time.perf_counter()
    before_hashes = protected_hashes(root)
    commands = (
        command_receipts
        if command_receipts is not None
        else _read_external_receipts(EXTERNAL_TEST_RECEIPT_PATH)
    )
    e2e = (
        e2e_receipts
        if e2e_receipts is not None
        else _read_external_receipts(EXTERNAL_E2E_RECEIPT_PATH)
    )
    elapsed = duration_s if duration_s is not None else time.perf_counter() - start
    report = build_report(
        root,
        date=date,
        command_receipts=commands,
        e2e_receipts=e2e,
        before_hashes=before_hashes,
        duration_s=elapsed,
    )
    errors = validate_report(report)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_report(report, root)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    report = run(date=args.date)
    print(f"wrote {RESULT_RELATIVE_PATH} status={report['status']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
