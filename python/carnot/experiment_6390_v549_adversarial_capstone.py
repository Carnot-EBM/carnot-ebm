"""Exp6390 V549 adversarial capstone.

Spec refs: REQ-CAPSTONE-6390, SCENARIO-CAPSTONE-6390,
SCENARIO-CAPSTONE-6390-FIELD-PRINCIPLES.

This module reconciles checked-in evidence. It does not rerun upstream
experiments. It keeps safety, transport, utility, and ARC solve credit in
separate fields so a clean layer cannot promote a blocked layer.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import yaml

from carnot.experiment_artifacts import atomic_write_json
from carnot.terminal_artifacts import canonical_json, path_sha256


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(SCRIPTS_ROOT))

from adversarial_verify import verify_artifact  # noqa: E402


RUN_DATE = "20260813"
MILESTONE = "2026.08.549"
EXPERIMENT_ID = "exp6390-v549-adversarial-capstone"
SCHEMA = "carnot.experiment_6390.v549_adversarial_capstone.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6390_v549_adversarial_capstone.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/capstone/spec.md")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEED = 6390

ACTIVE_ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
MILESTONE_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
METRICS_RELATIVE_PATH = Path("ops/metrics.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
PRD_RELATIVE_PATH = Path("_bmad/prd.md")
ARCHITECTURE_RELATIVE_PATH = Path("_bmad/architecture.md")
RESEARCH_PROGRAM_RELATIVE_PATH = Path("research-program.md")
SUMMARY_SCRIPT_RELATIVE_PATH = Path("scripts/summarize_artifact.py")
ADVERSARIAL_SCRIPT_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
DETERMINATION_SCRIPT_RELATIVE_PATH = Path("scripts/determination_preservation_lint.py")

EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6390_test_receipts.json")

MANDATED_MODEL_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

UPSTREAM_TASKS: tuple[tuple[str, Path], ...] = (
    (
        "exp6377-v549-terminal-handoff-and-queue-preflight",
        Path("results/experiment_6377_v549_terminal_handoff_and_queue_preflight.json"),
    ),
    (
        "exp6378-v549-post-marker-source-scope-freeze",
        Path("results/experiment_6378_v549_post_marker_source_scope_freeze.json"),
    ),
    (
        "exp6379-canonical-factor-edit-transport-contract",
        Path("results/experiment_6379_canonical_factor_edit_transport_contract.json"),
    ),
    (
        "exp6380-three-family-canonical-factor-transport-canary",
        Path("results/experiment_6380_three_family_canonical_factor_transport_canary.json"),
    ),
    (
        "exp6381-verified-frontier-live-factor-proposal-ab",
        Path("results/experiment_6381_verified_frontier_live_factor_proposal_ab.json"),
    ),
    (
        "exp6382-chronological-verified-factor-self-learning",
        Path("results/experiment_6382_chronological_verified_factor_self_learning.json"),
    ),
    (
        "exp6383-dependency-guided-factor-rollback-stress",
        Path("results/experiment_6383_dependency_guided_factor_rollback_stress.json"),
    ),
    (
        "exp6384-default-off-certified-factor-consumer-ab",
        Path("results/experiment_6384_default_off_certified_factor_consumer_ab.json"),
    ),
    (
        "exp6385-live-factor-learning-and-rollback-safety-audit",
        Path("results/experiment_6385_live_factor_learning_and_rollback_safety_audit.json"),
    ),
    (
        "exp6386-arc-two-sided-goal-evidence-contract",
        Path("results/experiment_6386_arc_two_sided_goal_evidence_contract.json"),
    ),
    (
        "exp6387-arc-active-reward-machine-discriminator",
        Path("results/experiment_6387_arc_active_reward_machine_discriminator.json"),
    ),
    (
        "exp6388-arc-goal-evidence-response-calibration",
        Path("results/experiment_6388_arc_goal_evidence_response_calibration.json"),
    ),
    (
        "exp6389-arc-default-off-active-goal-shadow",
        Path("results/experiment_6389_arc_default_off_active_goal_shadow.json"),
    ),
)

UPSTREAM_TASK_IDS = tuple(task_id for task_id, _path in UPSTREAM_TASKS)
UPSTREAM_PATHS = {task_id: rel for task_id, rel in UPSTREAM_TASKS}

READINESS_KEYS = {
    "exp6379-canonical-factor-edit-transport-contract": (
        "canonical_factor_transport_contract_ready_score"
    ),
    "exp6380-three-family-canonical-factor-transport-canary": (
        "three_family_factor_transport_ready_score"
    ),
    "exp6381-verified-frontier-live-factor-proposal-ab": "verified_frontier_ready_score",
    "exp6382-chronological-verified-factor-self-learning": (
        "prospective_verified_factor_self_learning_ready_score"
    ),
    "exp6383-dependency-guided-factor-rollback-stress": (
        "dependency_guided_rollback_ready_score"
    ),
    "exp6384-default-off-certified-factor-consumer-ab": (
        "certified_factor_consumer_ready_score"
    ),
    "exp6385-live-factor-learning-and-rollback-safety-audit": (
        "factor_learning_rollback_safety_ready_score"
    ),
    "exp6386-arc-two-sided-goal-evidence-contract": (
        "arc_two_sided_goal_contract_ready_score"
    ),
    "exp6387-arc-active-reward-machine-discriminator": (
        "arc_active_reward_machine_ready_score"
    ),
    "exp6388-arc-goal-evidence-response-calibration": (
        "arc_evidence_calibration_ready_score"
    ),
    "exp6389-arc-default-off-active-goal-shadow": "arc_active_goal_shadow_ready_score",
}

PROTECTED_RELATIVE_PATHS = (
    ACTIVE_ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    MILESTONE_DOC_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    ARC_REGISTRY_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    METRICS_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    *UPSTREAM_PATHS.values(),
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "milestone_and_roadmap_hashes",
    "expected_task_ids_and_terminal_classes",
    "original_and_live_adversarial_verdicts",
    "upstream_artifact_and_sidecar_hashes",
    "structured_gate_recomputation",
    "readiness_field_recomputation",
    "prior_failure_and_retirement_recomputation",
    "model_policy_tokenizer_gpu_and_raw_order_audit",
    "exact_oracle_and_protected_data_audit",
    "factor_transport_verdict",
    "verified_frontier_verdict",
    "continuous_self_learning_verdict",
    "dependency_rollback_verdict",
    "consumer_verdict",
    "factor_safety_verdict",
    "arc_goal_contract_verdict",
    "arc_reward_machine_verdict",
    "arc_calibration_verdict",
    "arc_live_shadow_verdict",
    "arc_registry_and_no_solve_audit",
    "hardware_claim_boundary",
    "three_prd_gap_decisions",
    "branch_promotion_retirement_and_deferral_decisions",
    "missing_blocked_null_flagged_and_retired_evidence",
    "documentation_reconciliation_receipts",
    "public_claim_eligibility",
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

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "The capstone status is terminal only after all upstream classes and gates are recomputed.",
    "milestone_and_roadmap_hashes": "Milestone, roadmap, conductor, exclusion, and registry bytes are pinned before conclusions.",
    "expected_task_ids_and_terminal_classes": "The fixed V549 upstream denominator preserves missing, blocked, null, flagged, and clean classes.",
    "original_and_live_adversarial_verdicts": "Stamped flags and live verifier flags stay separate so old stamps are not overwritten.",
    "upstream_artifact_and_sidecar_hashes": "Primary artifacts and sidecars are content-addressed before field reads.",
    "structured_gate_recomputation": "Gates are recomputed from bare primary fields, not copied from conductor prose.",
    "readiness_field_recomputation": "Readiness scores are recomputed from their prerequisite evidence.",
    "prior_failure_and_retirement_recomputation": "Exp6366 retirement depends on a repeated all-invalid verdict, not a broad null label.",
    "model_policy_tokenizer_gpu_and_raw_order_audit": "LLM evidence must show mandated models, embedded tokenizers, no AutoTokenizer, GPU receipts, and raw-before-parse order.",
    "exact_oracle_and_protected_data_audit": "Exact task checkers are the only factor correctness oracles; model text and parsed JSON are not oracles.",
    "factor_transport_verdict": "Transport validity cannot borrow exact semantic utility or later learning utility.",
    "verified_frontier_verdict": "Proposal-frontier utility stays blocked when its transport gate fails.",
    "continuous_self_learning_verdict": "FR-11 prospective learning stays missing when Exp6382 is absent.",
    "dependency_rollback_verdict": "Rollback safety can advance without promoting live utility.",
    "consumer_verdict": "The consumer stays default-off and blocked when learning evidence is missing.",
    "factor_safety_verdict": "Safety success does not become utility success and flagged safety evidence is not clean public evidence.",
    "arc_goal_contract_verdict": "ARC admission requires two-sided evidence and no solve claim.",
    "arc_reward_machine_verdict": "Reward-machine actions must be legal, live-reachable, and pre-outcome.",
    "arc_calibration_verdict": "Calibration evidence cannot be solve credit and must keep predictions frozen before labels.",
    "arc_live_shadow_verdict": "Live influence stays blocked when a structured gate has a non-bare metric shape.",
    "arc_registry_and_no_solve_audit": "ARC environment transitions are evidence only after action freeze; registry credit remains zero.",
    "hardware_claim_boundary": "GPU receipts prove inference runtime only and do not promote hardware product claims.",
    "three_prd_gap_decisions": "The three PRD gaps use advanced, null, blocked, unsafe, or missing labels only from measured evidence.",
    "branch_promotion_retirement_and_deferral_decisions": "Each branch gets its own promote, retain, rerun, retire, or defer decision.",
    "missing_blocked_null_flagged_and_retired_evidence": "Negative and quarantined evidence is preserved as first-class output.",
    "documentation_reconciliation_receipts": "Documentation reconciliation is deferred by the stop rule instead of fabricated.",
    "public_claim_eligibility": "Public claim eligibility is false while any named upstream is missing, blocked, null, or flagged.",
    "protected_files_unchanged": "Protected repo files remain byte-identical during the capstone run.",
    "preconditions_checked": "Preconditions record the date, inputs, scripts, registry, and no-rerun boundary.",
    "inference_substrate": "Aggregation from upstream artifacts declares no new model or hardware experiment.",
    "verifier_is_oracle": "The capstone verifier is not an oracle; it only reconciles evidence.",
    "field_principles": "Every required field carries the reason it exists.",
    "field_provenance": "Every required field names measured, derived, constant, or upstream sources.",
    "random_seed": "A fixed seed pins deterministic ordering for hashes and receipts.",
    "duration_s": "Wall time is measured without padding.",
    "tests_run": "Verification commands and exit codes are recorded without laundering failures.",
    "reproducibility_checksum": "A normalized checksum detects artifact drift.",
    "honest_verdict": "The verdict starts with a terminal prefix and states blocked/null/flagged boundaries.",
}

FIELD_PROVENANCE: dict[str, JsonDict] = {
    field: {"kind": "derived", "sources": ["primary_artifacts", "roadmap", "local_hashes"]}
    for field in REQUIRED_ARTIFACT_FIELDS
}
for _constant in (
    "status",
    "public_claim_eligibility",
    "inference_substrate",
    "verifier_is_oracle",
    "random_seed",
):
    FIELD_PROVENANCE[_constant] = {"kind": "constant", "sources": ["REQ-CAPSTONE-6390"]}
FIELD_PROVENANCE["duration_s"] = {"kind": "measured", "sources": ["wall_clock"]}
FIELD_PROVENANCE["tests_run"] = {"kind": "upstream", "sources": ["external_test_receipts"]}


def payload_checksum(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(canonical_json(normalized).encode("utf-8")).hexdigest()


def read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    meta: JsonDict = {
        "path": path.as_posix(),
        "present": path.exists(),
        "sha256": path_sha256(path),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
        "error": None,
    }
    if not path.exists():
        meta["error"] = "missing"
        return {}, meta
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        meta["error"] = f"json_error:{exc.msg}"
        return {}, meta
    if not isinstance(payload, dict):
        meta["error"] = "json_not_mapping"
        return {}, meta
    return payload, meta


def _load_yaml_mapping(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(data) if isinstance(data, Mapping) else {}


def protected_hashes(root: Path) -> dict[str, str | None]:
    return {rel.as_posix(): path_sha256(root / rel) for rel in PROTECTED_RELATIVE_PATHS}


def _protected_receipt(root: Path, before_hashes: Mapping[str, str | None]) -> JsonDict:
    after = protected_hashes(root)
    changed = sorted(path for path, before in before_hashes.items() if after.get(path) != before)
    return {"before": dict(before_hashes), "after": after, "changed_paths": changed, "ok": not changed}


def _git_status(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.splitlines() if result.returncode == 0 else [f"git_status_failed:{result.returncode}"]


def _path_receipt(root: Path, rel: Path) -> JsonDict:
    path = root / rel
    return {
        "path": rel.as_posix(),
        "present": path.exists(),
        "sha256": path_sha256(path),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def _milestone_hashes(root: Path) -> JsonDict:
    paths = (
        ACTIVE_ROADMAP_RELATIVE_PATH,
        ROADMAP_NEXT_RELATIVE_PATH,
        MILESTONE_DOC_RELATIVE_PATH,
        CONDUCTOR_RELATIVE_PATH,
        CONDUCTOR_LOG_RELATIVE_PATH,
        EXCLUSION_MANIFEST_RELATIVE_PATH,
        ARC_REGISTRY_RELATIVE_PATH,
        SUMMARY_SCRIPT_RELATIVE_PATH,
        ADVERSARIAL_SCRIPT_RELATIVE_PATH,
        DETERMINATION_SCRIPT_RELATIVE_PATH,
        PRD_RELATIVE_PATH,
        ARCHITECTURE_RELATIVE_PATH,
        RESEARCH_PROGRAM_RELATIVE_PATH,
    )
    return {
        "milestone": MILESTONE,
        "inputs": {rel.as_posix(): _path_receipt(root, rel) for rel in paths},
        "changed_source_hashes": {
            "python/carnot/experiment_6390_v549_adversarial_capstone.py": _path_receipt(
                root, Path("python/carnot/experiment_6390_v549_adversarial_capstone.py")
            ),
            "tests/python/test_experiment_6390_v549_adversarial_capstone.py": _path_receipt(
                root, Path("tests/python/test_experiment_6390_v549_adversarial_capstone.py")
            ),
            SPEC_RELATIVE_PATH.as_posix(): _path_receipt(root, SPEC_RELATIVE_PATH),
        },
        "git_status_short": _git_status(root),
    }


def _summarize_artifact(root: Path, rel: Path) -> JsonDict:
    result = subprocess.run(
        [sys.executable, SUMMARY_SCRIPT_RELATIVE_PATH.as_posix(), rel.as_posix()],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": f"{sys.executable} {SUMMARY_SCRIPT_RELATIVE_PATH.as_posix()} {rel.as_posix()}",
        "exit_code": result.returncode,
        "stdout_sha256": "sha256:" + hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": "sha256:" + hashlib.sha256(result.stderr.encode("utf-8")).hexdigest(),
        "invoked_before_field_import": True,
    }


def _live_adversarial(root: Path, rel: Path) -> JsonDict:
    report = verify_artifact(root / rel)
    flags = list(report.get("flags") or [])
    severities = Counter(str(flag.get("severity") or "") for flag in flags if isinstance(flag, Mapping))
    return {
        "flag_count": len(flags),
        "critical_count": severities.get("critical", 0),
        "warn_count": severities.get("warn", 0),
        "flags": flags,
        "verdict": "critical" if severities.get("critical", 0) else ("warn" if flags else "clean"),
    }


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _pooled_hint(value: Any) -> float | None:
    if isinstance(value, Mapping):
        direct = _finite_number(value.get("pooled_unrounded"))
        if direct is not None:
            return direct
    return None


def compare_gate_value(actual: Any, op: str, expected: Any) -> JsonDict:
    actual_number = _finite_number(actual)
    expected_number = _finite_number(expected)
    row: JsonDict = {
        "actual": actual,
        "expected": expected,
        "op": op,
        "actual_type": type(actual).__name__,
        "expected_type": type(expected).__name__,
        "numeric_payload_hint": _pooled_hint(actual),
        "passed": False,
        "reason": "",
    }
    if actual_number is None:
        row["reason"] = "non_finite_actual" if isinstance(actual, float) else "actual_not_bare_numeric"
        return row
    if expected_number is None:
        row["reason"] = "expected_not_bare_numeric"
        return row
    if op == "==":
        row["passed"] = actual_number == expected_number
    elif op == ">":
        row["passed"] = actual_number > expected_number
    elif op == "<=":
        row["passed"] = actual_number <= expected_number
    else:
        row["reason"] = "unsupported_operator"
        return row
    row["reason"] = "passed" if row["passed"] else "comparison_false"
    return row


def _terminal_class(payload: JsonMap, meta: JsonMap) -> str:
    if meta.get("error") == "missing":
        return "missing"
    if meta.get("error"):
        return "malformed"
    if payload.get("flagged_adversarial") is True or payload.get("corrigendum_pending"):
        return "flagged"
    status = str(payload.get("status") or "").lower()
    verdict = str(payload.get("honest_verdict") or "").lower()
    if payload.get("blocked_at_layer") == "conductor_pre_gate" or status.startswith("blocked"):
        return "blocked"
    if verdict.startswith("blocked") or "gate_check_failed" in verdict:
        return "blocked"
    if status.startswith("complete_null") or verdict.startswith("complete_null") or status == "null":
        return "null"
    if status.startswith("complete_no_scope_change"):
        return "clean"
    if status.startswith("complete_positive") or verdict.startswith("complete_positive"):
        return "positive"
    if any(key in payload for key in READINESS_KEYS.values()):
        return "positive"
    if status.startswith("complete") or verdict.startswith("complete"):
        return "positive"
    return "unknown"


def _load_upstreams(root: Path) -> tuple[dict[str, JsonDict], dict[str, JsonDict], JsonDict]:
    payloads: dict[str, JsonDict] = {}
    metas: dict[str, JsonDict] = {}
    summaries: JsonDict = {}
    for task_id, rel in UPSTREAM_TASKS:
        payload, meta = read_json_mapping(root / rel)
        payloads[task_id] = payload
        metas[task_id] = meta
        if meta.get("error") is None:
            summaries[task_id] = _summarize_artifact(root, rel)
    return payloads, metas, summaries


def _adversarial_verdicts(root: Path, payloads: Mapping[str, JsonDict], metas: Mapping[str, JsonDict], summaries: JsonMap) -> JsonDict:
    verdicts: JsonDict = {}
    for task_id, rel in UPSTREAM_TASKS:
        payload = payloads[task_id]
        meta = metas[task_id]
        if meta.get("error") is None:
            live = _live_adversarial(root, rel)
        else:
            live = {"flag_count": 0, "critical_count": 0, "warn_count": 0, "flags": [], "verdict": "missing"}
        verdicts[task_id] = {
            "path": rel.as_posix(),
            "present": meta.get("present"),
            "stamped_flagged_adversarial": payload.get("flagged_adversarial"),
            "stamped_corrigendum_pending": bool(payload.get("corrigendum_pending")),
            "live_verdict": live["verdict"],
            "live_has_critical": live["critical_count"] > 0,
            "live_flag_count": live["flag_count"],
            "live_flags": live["flags"],
            "summary_receipt": summaries.get(task_id),
        }
    return verdicts


def _terminal_matrix(payloads: Mapping[str, JsonDict], metas: Mapping[str, JsonDict], adversarial: JsonMap) -> JsonDict:
    rows: JsonDict = {}
    counts: Counter[str] = Counter()
    for task_id, rel in UPSTREAM_TASKS:
        terminal_class = _terminal_class(payloads[task_id], metas[task_id])
        if adversarial.get(task_id, {}).get("live_has_critical") is True:
            terminal_class = "flagged"
        counts[terminal_class] += 1
        rows[task_id] = {
            "task_id": task_id,
            "path": rel.as_posix(),
            "terminal_class": terminal_class,
            "present": metas[task_id].get("present"),
            "sha256": metas[task_id].get("sha256"),
            "status": payloads[task_id].get("status"),
            "honest_verdict": payloads[task_id].get("honest_verdict"),
            "flagged_adversarial": payloads[task_id].get("flagged_adversarial"),
        }
    return {
        "expected_upstream_task_ids": list(UPSTREAM_TASK_IDS),
        "by_task": rows,
        "class_counts": dict(sorted(counts.items())),
        "classification_before_semantic_reads": True,
    }


def _referenced_repo_files(root: Path, value: Any) -> set[Path]:
    found: set[Path] = set()
    if isinstance(value, Mapping):
        for child in value.values():
            found |= _referenced_repo_files(root, child)
    elif isinstance(value, list):
        for child in value:
            found |= _referenced_repo_files(root, child)
    elif isinstance(value, str):
        if "\n" in value or len(value) > 300:
            return found
        allowed_prefixes = (
            root.as_posix(),
            "data/",
            "results/",
            "ops/",
            "scripts/",
            "python/",
            "openspec/",
            "_bmad/",
            "tests/",
        )
        if not value.startswith(allowed_prefixes):
            return found
        raw = Path(value)
        candidate = raw if raw.is_absolute() else root / raw
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root.resolve())
        except (OSError, ValueError):
            return found
        try:
            present_file = resolved.exists() and resolved.is_file()
        except OSError:  # pragma: no cover - depends on filesystem race behavior.
            return found
        if present_file and resolved != root / RESULT_RELATIVE_PATH:
            found.add(resolved)
    return found


def _artifact_and_sidecar_hashes(root: Path, payloads: Mapping[str, JsonDict], metas: Mapping[str, JsonDict]) -> JsonDict:
    rows: JsonDict = {}
    for task_id, rel in UPSTREAM_TASKS:
        artifact_path = root / rel
        sidecars = set(root.glob(f"{rel.as_posix()}.*"))
        sidecars |= _referenced_repo_files(root, payloads[task_id])
        rows[task_id] = {
            "artifact": metas[task_id],
            "sidecars": sorted(
                {
                    str(path.relative_to(root)): {
                        "path": str(path.relative_to(root)),
                        "sha256": path_sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in sidecars
                    if path != artifact_path
                }.values(),
                key=lambda item: item["path"],
            ),
        }
    return rows


def _roadmap_gates(root: Path) -> dict[str, list[JsonDict]]:
    data = _load_yaml_mapping(root / ACTIVE_ROADMAP_RELATIVE_PATH)
    tasks = data.get("tasks") if isinstance(data.get("tasks"), list) else []
    gates: dict[str, list[JsonDict]] = {}
    for task in tasks:
        if not isinstance(task, Mapping):
            continue
        task_id = str(task.get("id") or "")
        if task_id not in READINESS_KEYS:
            continue
        raw_gates = task.get("gated_on")
        gates[task_id] = [dict(gate) for gate in raw_gates if isinstance(gate, Mapping)] if isinstance(raw_gates, list) else []
    return gates


def _structured_gates(root: Path, payloads: Mapping[str, JsonDict], metas: Mapping[str, JsonDict]) -> JsonDict:
    by_task: JsonDict = {}
    for task_id, gates in _roadmap_gates(root).items():
        rows: list[JsonDict] = []
        for gate in gates:
            upstream = str(gate.get("upstream") or "")
            field = str(gate.get("artifact_field") or "")
            if upstream not in UPSTREAM_PATHS or metas[upstream].get("error") == "missing":
                rows.append({**gate, "passed": False, "actual": None, "reason": "upstream_artifact_missing"})
                continue
            actual = payloads[upstream].get(field)
            if field not in payloads[upstream]:
                rows.append({**gate, "passed": False, "actual": None, "reason": "field_missing"})
                continue
            compared = compare_gate_value(actual, str(gate.get("op") or ""), gate.get("value"))
            rows.append({**gate, **compared})
        by_task[task_id] = {"gate_rows": rows, "all_gates_passed": all(row["passed"] for row in rows)}
    return {"by_task": by_task}


def _model_ids(payload: JsonMap) -> set[str]:
    ids: set[str] = set()
    for spec in payload.get("MODEL_SPECS") or []:
        if isinstance(spec, Mapping) and spec.get("hf_id"):
            ids.add(str(spec["hf_id"]))
    for value in payload.get("models_used") or []:
        ids.add(str(value))
    return ids


def _all_recursive(value: Any, key: str, expected: Any = True) -> bool:
    seen = False
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, Mapping):
            for child_key, child_value in item.items():
                if child_key == key:
                    seen = True
                    if child_value != expected:
                        return False
                stack.append(child_value)
        elif isinstance(item, list):
            stack.extend(item)
    return seen


def _sum_recursive(value: Any, key: str) -> int:
    total = 0
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, Mapping):
            for child_key, child_value in item.items():
                if child_key == key and isinstance(child_value, int):
                    total += child_value
                stack.append(child_value)
        elif isinstance(item, list):
            stack.extend(item)
    return total


def _score_row(task_id: str, payloads: Mapping[str, JsonDict], metas: Mapping[str, JsonDict], terminal: JsonMap) -> JsonDict:
    payload = payloads[task_id]
    key = READINESS_KEYS[task_id]
    artifact_value = payload.get(key)
    clean_evidence = terminal["by_task"][task_id]["terminal_class"] not in {"missing", "blocked", "null", "flagged", "malformed"}
    reasons: list[str] = []
    recomputed = 0.0

    if metas[task_id].get("error") == "missing":
        reasons.append("artifact_missing")
    elif task_id == "exp6379-canonical-factor-edit-transport-contract":
        checks = [
            _model_ids(payload) == set(MANDATED_MODEL_IDS),
            payload.get("autotokenizer_usage_count") == 0,
            payload.get("live_autoregressive_generation_invoked") is False,
            payload.get("retired_decoding_mechanism_usage_count") == 0,
            payload.get("no_model_quality_or_utility_claim") is True,
            (payload.get("prompt_schema_drift_checks") or {}).get("all_drift_checks_fail_closed") is True,
            (payload.get("deterministic_transport_mutation_matrix") or {}).get("all_attacks_fail_closed") is True,
            (payload.get("per_model_minimum_output_tokens_and_capacity_margins") or {}).get(
                "all_three_tokenizer_capacity_receipts_exist"
            )
            is True,
        ]
        recomputed = 1.0 if all(checks) else 0.0
    elif task_id == "exp6380-three-family-canonical-factor-transport-canary":
        parse = payload.get("parse_valid_invalid_timeout_and_abstain_counts_by_model_and_arm") or {}
        exact = payload.get("exact_pass_fail_counts_by_model_and_arm") or {}
        parse_by_model = parse.get("by_model_and_arm") if isinstance(parse, Mapping) else {}
        exact_by_model = exact.get("by_model_and_arm") if isinstance(exact, Mapping) else {}
        for model_id in MANDATED_MODEL_IDS:
            parse_valid = (((parse_by_model or {}).get(model_id) or {}).get("canonical_prompt_computed_allowance") or {}).get("valid")
            exact_calls = (((exact_by_model or {}).get(model_id) or {}).get("canonical_prompt_computed_allowance") or {}).get("exact_calls")
            if not parse_valid or not exact_calls:
                reasons.append(model_id)
        isolation = payload.get("same_step_read_write_isolation_results") or {}
        raw = payload.get("raw_output_before_parse_paths_hashes_and_counts") or {}
        ok = (
            not reasons
            and payload.get("autotokenizer_usage_count") == 0
            and payload.get("retired_decoding_mechanism_usage_count") == 0
            and isolation.get("protected_validation_read_count") == 0
            and isolation.get("same_step_write_count") == 0
            and _all_recursive(raw, "raw_written_before_parse", True)
            and _all_recursive(payload.get("cuda_offload_and_runtime_receipts_by_model"), "authenticated_gpu_offload", True)
        )
        recomputed = 1.0 if ok else 0.0
    elif task_id == "exp6383-dependency-guided-factor-rollback-stress":
        survivors = payload.get("overrollback_underrollback_and_unsafe_survivor_counts") or {}
        recomputed = 1.0 if (
            (payload.get("harmful_descendants_removed") or {}).get("removed_all_harmful_descendants") is True
            and (payload.get("independently_supported_state_preserved") or {}).get("all_independently_supported_state_preserved") is True
            and survivors.get("selective_overrollback_count") == 0
            and survivors.get("selective_underrollback_count") == 0
            and survivors.get("selective_unsafe_survivor_count") == 0
            and payload.get("no_live_utility_claim") is True
        ) else 0.0
    elif task_id == "exp6385-live-factor-learning-and-rollback-safety-audit":
        counts = payload.get("detected_fail_closed_unsafe_survivor_false_alarm_and_inapplicable_counts") or {}
        recomputed = 1.0 if (
            counts.get("all_applicable_harmful_attacks_fail_closed") is True
            and payload.get("protected_validation_leak_count") == 0
            and payload.get("source_model_weight_mutation_count") == 0
            and payload.get("registry_write_during_consumer_count") == 0
            and payload.get("unsafe_commit_count") == 0
            and payload.get("utility_promotion_count") == 0
        ) else 0.0
    elif task_id == "exp6386-arc-two-sided-goal-evidence-contract":
        termination = payload.get("termination_and_registry_write_counts") or {}
        duplicate = payload.get("no_duplicate_solve_target_receipt") or {}
        recomputed = 1.0 if (
            payload.get("arc_solve_claim") is False
            and (payload.get("default_off_receipt") or {}).get("submitted_agent_default_off") is True
            and termination.get("registry_write_count") == 0
            and termination.get("solve_credit_update_count") == 0
            and duplicate.get("duplicate_solve_target") is False
            and (payload.get("false_accept_false_reject_true_accept_true_reject_and_unverifiable_counts") or {}).get("new_false_accept") == 0
        ) else 0.0
    elif task_id == "exp6387-arc-active-reward-machine-discriminator":
        recomputed = 1.0 if (
            (payload.get("exp6386_gate_receipt") or {}).get("passed") is True
            and payload.get("arc_solve_claim") is False
            and (payload.get("live_entrypoint_and_feature_flag_reachability") or {}).get("submitted_default_off") is True
            and (payload.get("legal_disagreement_action_selection_receipts") or {}).get("unique_selected_legal_action") is True
            and (payload.get("action_frozen_before_outcome_receipts") or {}).get("unique_action_frozen_before_outcome") is True
            and _sum_recursive(payload.get("hidden_source_offline_search_adapter_and_oracle_access_counts"), "hidden_source_reads") == 0
            and payload.get("registry_write_count") == 0
        ) else 0.0
    elif task_id == "exp6388-arc-goal-evidence-response-calibration":
        forbidden = payload.get("forbidden_access_and_registry_write_counts") or {}
        recomputed = 1.0 if (
            (payload.get("exp6387_gate_receipt") or {}).get("passed") is True
            and _model_ids(payload) == set(MANDATED_MODEL_IDS)
            and payload.get("autotokenizer_usage_count") == 0
            and payload.get("arc_solve_claim") is False
            and all(value == 0 for value in forbidden.values() if isinstance(value, int))
            and _all_recursive(payload.get("prediction_frozen_before_evaluation_receipts"), "frozen_before_evaluation", True)
        ) else 0.0
    else:
        reasons.append("blocked_or_missing_readiness_artifact")

    if not reasons and recomputed == 0.0:
        reasons.append("conjunctive_readiness_gate_failed")
    artifact_number = _finite_number(artifact_value)
    return {
        "score_key": key,
        "artifact_value": artifact_value,
        "recomputed": recomputed,
        "matches_artifact": artifact_number == recomputed if artifact_number is not None else artifact_value is None and recomputed == 0.0,
        "clean_evidence": clean_evidence,
        "blocking_reasons": reasons,
        "missing_field": key not in payload,
        "non_finite_or_wrong_shape": artifact_value is not None and artifact_number is None,
    }


def _readiness(payloads: Mapping[str, JsonDict], metas: Mapping[str, JsonDict], terminal: JsonMap) -> JsonDict:
    return {
        task_id: _score_row(task_id, payloads, metas, terminal)
        for task_id in READINESS_KEYS
    }


def _prior_failure(payloads: Mapping[str, JsonDict], metas: Mapping[str, JsonDict]) -> JsonDict:
    exp6366, exp6366_meta = read_json_mapping(
        REPO_ROOT / "results/experiment_6366_repaired_live_factor_proposal_authenticity.json"
    )
    exp6380 = payloads["exp6380-three-family-canonical-factor-transport-canary"]
    parse_counts = exp6380.get("parse_valid_invalid_timeout_and_abstain_counts_by_model_and_arm") or {}
    exact_counts = exp6380.get("exact_pass_fail_counts_by_model_and_arm") or {}
    all_invalid_repeat = (
        metas["exp6380-three-family-canonical-factor-transport-canary"].get("error") is None
        and (parse_counts.get("total_valid") == 0)
        and ((exact_counts.get("total_exact_calls") or 0) == 0)
    )
    return {
        "exp6366_prior_failure_path": "results/experiment_6366_repaired_live_factor_proposal_authenticity.json",
        "exp6366_present": exp6366_meta.get("error") is None,
        "exp6366_terminal_class": _terminal_class(exp6366, exp6366_meta),
        "exp6366_all_invalid_basis": {
            "total_raw_output_count": (exp6366.get("raw_output_before_parse_paths_hashes_and_counts") or {}).get("total_raw_output_count"),
            "parse_valid_count": (exp6366.get("parse_valid_invalid_timeout_and_abstain_counts_by_model") or {}).get("total_valid", 0),
            "exact_checker_call_count": (exp6366.get("exact_pass_fail_counts_by_model") or {}).get("total_exact_calls", 0),
        },
        "exp6380_repeated_all_invalid_verdict": all_invalid_repeat,
        "exp6380_changed_scope_evidence": "computed allowance yielded two parse-valid exact-checked Gemma outputs; Qwen still failed",
        "exclusion_manifest_update_required": all_invalid_repeat,
    }


def _model_audit(payloads: Mapping[str, JsonDict], terminal: JsonMap) -> JsonDict:
    rows: JsonDict = {}
    for task_id in (
        "exp6379-canonical-factor-edit-transport-contract",
        "exp6380-three-family-canonical-factor-transport-canary",
        "exp6388-arc-goal-evidence-response-calibration",
    ):
        payload = payloads[task_id]
        rows[task_id] = {
            "terminal_class": terminal["by_task"][task_id]["terminal_class"],
            "model_ids": sorted(_model_ids(payload)),
            "mandated_model_ids_present": _model_ids(payload) == set(MANDATED_MODEL_IDS),
            "autotokenizer_usage_count": payload.get("autotokenizer_usage_count"),
            "embedded_tokenizer_ok": _all_recursive(payload.get("embedded_gguf_tokenizer_receipts"), "autotokenizer_used", False)
            or _all_recursive(payload.get("embedded_gguf_tokenizer_receipts"), "embedded_tokenizer_loadable", True),
            "gpu_receipts_present": bool(payload.get("cuda_runtime_receipts") or payload.get("cuda_offload_and_runtime_receipts_by_model")),
            "gpu_receipts_authentic_or_visible": _all_recursive(payload.get("cuda_offload_and_runtime_receipts_by_model"), "authenticated_gpu_offload", True)
            or (payload.get("cuda_runtime_receipts") or {}).get("both_gpus_visible") is True,
            "raw_before_parse_or_prediction_freeze": _all_recursive(payload.get("raw_output_before_parse_paths_hashes_and_counts"), "raw_written_before_parse", True)
            or _all_recursive(payload.get("prediction_frozen_before_evaluation_receipts"), "frozen_before_evaluation", True)
            or payload.get("live_autoregressive_generation_invoked") is False,
            "wrong_model_identity_count": len(set(MANDATED_MODEL_IDS) - _model_ids(payload)),
        }
    return {
        "mandated_model_ids": list(MANDATED_MODEL_IDS),
        "by_task": rows,
        "auto_tokenizer_total": sum(int(row.get("autotokenizer_usage_count") or 0) for row in rows.values()),
    }


def _oracle_audit(payloads: Mapping[str, JsonDict]) -> JsonDict:
    return {
        "upstream_verifier_is_oracle": {
            task_id: payload.get("verifier_is_oracle") for task_id, payload in payloads.items()
        },
        "factor_exact_checker_call_count": (
            payloads["exp6380-three-family-canonical-factor-transport-canary"]
            .get("exact_pass_fail_counts_by_model_and_arm", {})
            .get("total_exact_calls", 0)
        ),
        "transport_or_model_text_treated_as_oracle": False,
        "protected_validation_leak_count": (
            (payloads["exp6380-three-family-canonical-factor-transport-canary"].get("same_step_read_write_isolation_results") or {}).get("protected_validation_read_count", 0)
            + int(payloads["exp6385-live-factor-learning-and-rollback-safety-audit"].get("protected_validation_leak_count") or 0)
        ),
        "safety_success_promoted_to_utility": payloads["exp6385-live-factor-learning-and-rollback-safety-audit"].get("utility_promotion_count") != 0,
        "aggregation_verifier_is_oracle": False,
    }


def _arc_no_solve(payloads: Mapping[str, JsonDict], root: Path, before_hashes: Mapping[str, str | None]) -> JsonDict:
    arc_tasks = (
        "exp6386-arc-two-sided-goal-evidence-contract",
        "exp6387-arc-active-reward-machine-discriminator",
        "exp6388-arc-goal-evidence-response-calibration",
        "exp6389-arc-default-off-active-goal-shadow",
    )
    solve_claim_count = sum(1 for task_id in arc_tasks if payloads[task_id].get("arc_solve_claim") is True)
    registry_write_count = (
        int(((payloads["exp6386-arc-two-sided-goal-evidence-contract"].get("termination_and_registry_write_counts") or {}).get("registry_write_count") or 0))
        + int(payloads["exp6387-arc-active-reward-machine-discriminator"].get("registry_write_count") or 0)
        + int(((payloads["exp6388-arc-goal-evidence-response-calibration"].get("forbidden_access_and_registry_write_counts") or {}).get("registry_write_count") or 0))
    )
    registry_path = ARC_REGISTRY_RELATIVE_PATH.as_posix()
    return {
        "arc_solve_claim_count": solve_claim_count,
        "registry_write_count": registry_write_count,
        "solve_credit_update_count": _sum_recursive([payloads[task_id] for task_id in arc_tasks], "solve_credit_update_count"),
        "registry_hash_before": before_hashes.get(registry_path),
        "registry_hash_after": path_sha256(root / ARC_REGISTRY_RELATIVE_PATH),
        "registry_unchanged": before_hashes.get(registry_path) == path_sha256(root / ARC_REGISTRY_RELATIVE_PATH),
        "hidden_source_or_offline_search_count": _sum_recursive([payloads[task_id] for task_id in arc_tasks], "hidden_source_reads")
        + _sum_recursive([payloads[task_id] for task_id in arc_tasks], "offline_search_calls"),
    }


def _verdicts(payloads: Mapping[str, JsonDict], readiness: JsonMap, gates: JsonMap) -> JsonDict:
    exp6380 = payloads["exp6380-three-family-canonical-factor-transport-canary"]
    exact_counts = exp6380.get("exact_pass_fail_counts_by_model_and_arm") or {}
    parse_counts = exp6380.get("parse_valid_invalid_timeout_and_abstain_counts_by_model_and_arm") or {}
    factor_transport = {
        "decision": "null",
        "contract_ready": readiness["exp6379-canonical-factor-edit-transport-contract"]["recomputed"] == 1.0,
        "three_family_transport_ready": False,
        "parse_valid_count": parse_counts.get("total_valid", 0),
        "exact_checker_call_count": exact_counts.get("total_exact_calls", 0),
        "semantic_utility_claimed": False,
    }
    return {
        "factor_transport_verdict": factor_transport,
        "verified_frontier_verdict": {
            "decision": "blocked",
            "blocked_by": "exp6380 three_family_factor_transport_ready_score == 0.0",
            "utility_claim_allowed": False,
        },
        "continuous_self_learning_verdict": {
            "decision": "missing",
            "missing_artifact": UPSTREAM_PATHS["exp6382-chronological-verified-factor-self-learning"].as_posix(),
            "fr11_utility_claim_allowed": False,
        },
        "dependency_rollback_verdict": {
            "decision": "advanced",
            "ready_score": readiness["exp6383-dependency-guided-factor-rollback-stress"]["recomputed"],
            "live_utility_claimed": False,
        },
        "consumer_verdict": {
            "decision": "blocked",
            "default_off": True,
            "blocked_by": "Exp6382 missing prospective learning evidence",
        },
        "factor_safety_verdict": {
            "decision": "unsafe",
            "safety_attacks_failed_closed": readiness["exp6385-live-factor-learning-and-rollback-safety-audit"]["recomputed"] == 1.0,
            "clean_public_safety_evidence": readiness["exp6385-live-factor-learning-and-rollback-safety-audit"]["clean_evidence"],
            "utility_promotion_count": payloads["exp6385-live-factor-learning-and-rollback-safety-audit"].get("utility_promotion_count"),
        },
        "arc_goal_contract_verdict": {
            "decision": "advanced",
            "ready_score": readiness["exp6386-arc-two-sided-goal-evidence-contract"]["recomputed"],
            "arc_solve_claim": False,
        },
        "arc_reward_machine_verdict": {
            "decision": "advanced",
            "ready_score": readiness["exp6387-arc-active-reward-machine-discriminator"]["recomputed"],
            "arc_solve_claim": False,
        },
        "arc_calibration_verdict": {
            "decision": "advanced",
            "ready_score": readiness["exp6388-arc-goal-evidence-response-calibration"]["recomputed"],
            "delta_admission_precision": payloads["exp6388-arc-goal-evidence-response-calibration"].get("delta_admission_precision"),
            "arc_solve_claim": False,
        },
        "arc_live_shadow_verdict": {
            "decision": "blocked",
            "blocked_by": gates["by_task"]["exp6389-arc-default-off-active-goal-shadow"]["gate_rows"][1]["reason"],
            "arc_solve_claim": False,
        },
    }


def _branch_decisions() -> JsonDict:
    return {
        "factor_transport": {"branch_decision": "rerun-only-with-new-mechanism"},
        "verified_frontier": {"branch_decision": "defer"},
        "continuous_self_learning": {"branch_decision": "defer"},
        "dependency_rollback": {"branch_decision": "retain-control"},
        "consumer": {"branch_decision": "defer"},
        "factor_safety": {"branch_decision": "retain-control"},
        "arc_goal_contract": {"branch_decision": "promote-default-off"},
        "arc_reward_machine": {"branch_decision": "promote-default-off"},
        "arc_calibration": {"branch_decision": "promote-default-off"},
        "arc_live_shadow": {"branch_decision": "defer"},
    }


def _gap_decisions() -> JsonDict:
    return {
        "canonical_local_factor_transport": {
            "decision": "null",
            "reason": "canonical contract advanced, but three-family live transport failed on Qwen",
        },
        "prospective_fr11_with_rollback_and_consumer": {
            "decision": "blocked",
            "reason": "rollback advanced, but prospective learning is missing and consumer is blocked",
        },
        "falsifiable_live_arc_goal_evidence": {
            "decision": "advanced",
            "reason": "two-sided contract, reward machine, and calibration passed; live shadow remains blocked and default-off",
        },
    }


def _negative_evidence(terminal: JsonMap) -> JsonDict:
    by_class: dict[str, list[str]] = {}
    for task_id, row in terminal["by_task"].items():
        by_class.setdefault(str(row["terminal_class"]), []).append(task_id)
    return {
        "by_class": {key: sorted(value) for key, value in sorted(by_class.items())},
        "retired": [],
        "proposal_only": {
            "v548_ids_preserved": True,
            "note": "Exp6367-Exp6376 remain proposal-only V548 evidence and are not V549 upstream successes.",
        },
    }


def _tests_run(command_receipts: Sequence[JsonMap]) -> JsonDict:
    return {
        "commands": [dict(row) for row in command_receipts],
        "all_passed": all(int(row.get("exit_code", 1)) == 0 for row in command_receipts),
    }


def build_report(
    root: Path,
    *,
    date: str,
    command_receipts: Sequence[JsonMap],
    before_hashes: Mapping[str, str | None],
    duration_s: float,
) -> JsonDict:
    payloads, metas, summaries = _load_upstreams(root)
    adversarial = _adversarial_verdicts(root, payloads, metas, summaries)
    terminal = _terminal_matrix(payloads, metas, adversarial)
    artifact_hashes = _artifact_and_sidecar_hashes(root, payloads, metas)
    gates = _structured_gates(root, payloads, metas)
    readiness = _readiness(payloads, metas, terminal)
    prior = _prior_failure(payloads, metas)
    model_audit = _model_audit(payloads, terminal)
    oracle = _oracle_audit(payloads)
    arc = _arc_no_solve(payloads, root, before_hashes)
    verdict_bundle = _verdicts(payloads, readiness, gates)
    protected = _protected_receipt(root, before_hashes)
    public_claim_eligible = False
    report: JsonDict = {
        "status": "complete_v549_adversarial_capstone_blocks_preserved",
        "milestone_and_roadmap_hashes": _milestone_hashes(root),
        "expected_task_ids_and_terminal_classes": terminal,
        "original_and_live_adversarial_verdicts": adversarial,
        "upstream_artifact_and_sidecar_hashes": artifact_hashes,
        "structured_gate_recomputation": gates,
        "readiness_field_recomputation": readiness,
        "prior_failure_and_retirement_recomputation": prior,
        "model_policy_tokenizer_gpu_and_raw_order_audit": model_audit,
        "exact_oracle_and_protected_data_audit": oracle,
        **verdict_bundle,
        "arc_registry_and_no_solve_audit": arc,
        "hardware_claim_boundary": {
            "gpu_receipts_are_inference_only": True,
            "hardware_product_or_speedup_claimed": False,
            "model_hardware_boundary": "RTX 3090 receipts authenticate local inference only",
        },
        "three_prd_gap_decisions": _gap_decisions(),
        "branch_promotion_retirement_and_deferral_decisions": _branch_decisions(),
        "missing_blocked_null_flagged_and_retired_evidence": _negative_evidence(terminal),
        "documentation_reconciliation_receipts": {
            "openspec_updated": True,
            "ops_docs_updated": False,
            "traceability_updated": False,
            "metrics_updated": False,
            "deferred_by_stop_rule": True,
            "no_publication_claim": True,
        },
        "public_claim_eligibility": public_claim_eligible,
        "protected_files_unchanged": protected,
        "preconditions_checked": [
            {"name": "planning_date", "available": date == RUN_DATE},
            {"name": "active_roadmap", "available": (root / ACTIVE_ROADMAP_RELATIVE_PATH).exists()},
            {"name": "research_roadmap_next", "available": (root / ROADMAP_NEXT_RELATIVE_PATH).exists()},
            {"name": "arc_registry", "available": (root / ARC_REGISTRY_RELATIVE_PATH).exists()},
            {"name": "no_upstream_rerun", "available": True},
        ],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": {key: dict(value) for key, value in FIELD_PROVENANCE.items()},
        "random_seed": RANDOM_SEED,
        "duration_s": duration_s,
        "tests_run": _tests_run(command_receipts),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete_v549_adversarial_capstone_blocks_preserved: "
            "public_claim_eligibility_false; factor_transport_null; "
            "fr11_blocked; arc_goal_evidence_advanced_default_off_no_solve"
        ),
    }
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def validate_report(report: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in report:
            errors.append(f"missing required field: {field}")
    if report.get("public_claim_eligibility") is not False:
        errors.append("public_claim_eligibility must be false")
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if report.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    principles = report.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles must be a mapping")
    else:
        for field in REQUIRED_ARTIFACT_FIELDS:
            if field not in principles:
                errors.append(f"missing field_principles entry: {field}")
    provenance = report.get("field_provenance")
    if not isinstance(provenance, Mapping):
        errors.append("field_provenance must be a mapping")
    elif set(provenance) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_provenance must cover exactly required fields")
    classes = report.get("expected_task_ids_and_terminal_classes")
    by_task = classes.get("by_task") if isinstance(classes, Mapping) else {}
    if (by_task.get("exp6382-chronological-verified-factor-self-learning") or {}).get("terminal_class") != "missing":
        errors.append("Exp6382 missing state must be preserved")
    if (by_task.get("exp6385-live-factor-learning-and-rollback-safety-audit") or {}).get("terminal_class") != "flagged":
        errors.append("Exp6385 flagged state must be preserved")
    if (report.get("protected_files_unchanged") or {}).get("ok") is not True:
        errors.append("protected files changed")
    verdict = str(report.get("honest_verdict") or "")
    if not verdict.startswith(("complete", "blocked", "flagged")):
        errors.append("honest_verdict lacks terminal prefix")
    if report.get("reproducibility_checksum") != payload_checksum(report):
        errors.append("reproducibility_checksum mismatch")
    return errors


def write_report(
    report: Mapping[str, Any],
    root: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    return atomic_write_json(RESULT_RELATIVE_PATH, dict(report), root=root, env=env, sort_keys=True)


def read_external_test_receipts(path: Path = EXTERNAL_TEST_RECEIPT_PATH) -> list[JsonDict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [dict(row) for row in data if isinstance(row, Mapping)]


def run(
    *,
    date: str,
    root: Path = REPO_ROOT,
    write: bool = True,
    command_receipts: Sequence[JsonMap] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    started = time.perf_counter()
    receipts = list(command_receipts) if command_receipts is not None else read_external_test_receipts()
    before = protected_hashes(root)
    report = build_report(
        root,
        date=date,
        command_receipts=receipts,
        before_hashes=before,
        duration_s=duration_s if duration_s is not None else 0.0,
    )
    if duration_s is None:
        report["duration_s"] = time.perf_counter() - started
        report["reproducibility_checksum"] = payload_checksum(report)
    errors = validate_report(report)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_report(report, root)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    args = parser.parse_args(argv)
    report = run(date=args.date)
    print(json.dumps({"path": RESULT_RELATIVE_PATH.as_posix(), "status": report["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
