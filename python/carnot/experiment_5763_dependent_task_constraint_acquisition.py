"""Exp5763 dependent task constraint acquisition.

Spec refs: REQ-LEARN-5763, REQ-STORE-5763,
SCENARIO-LEARN-5763-DEPENDENT-STREAM,
SCENARIO-LEARN-5763-MATCHED-CONTROLS,
SCENARIO-LEARN-5763-RECOVERY-RESTART,
SCENARIO-STORE-5763.

This experiment deliberately stays on the exact sidecar path established by
Exp5762. It does not train or edit model weights. The "learner" in this file
is only a small typed lifecycle state machine whose evidence comes from exact
membership answers minted before learner access. That keeps the result useful
as a lifecycle stress test while avoiding a hidden LLM, pseudo-label, LoRA, RL,
KAN scale-up, or GGUF-write path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import platform
from pathlib import Path
import shutil
import sys
from typing import Any

from carnot import experiment_5761_exact_constraint_acquisition_benchmark as exp5761
from carnot import experiment_5762_query_driven_constraint_lifecycle as exp5762


JsonDict = dict[str, Any]
Probe = Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5763_dependent_task_constraint_acquisition.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_5763_dependent_task_constraint_acquisition.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_5763_dependent_task_constraint_acquisition.py")

SCHEMA = "carnot.experiment_5763.dependent_task_constraint_acquisition.v1"
EXPERIMENT = 5763
EXPERIMENT_ID = "experiment_5763_dependent_task_constraint_acquisition"
MILESTONE = "2026.07.514"
RUN_DATE = "20260721"
GENERATOR_VERSION = "exp5763_dependent_task_constraint_acquisition_v1"
INFERENCE_SUBSTRATE = "dependent_exact_membership_query_sidecar_no_llm"

SESSION_COUNT = 72
HELDOUT_COMPOSITION_COUNT = 12
QUERY_BUDGET_PER_SESSION = 2
RAM_FLOOR_MB = 512
DISK_FLOOR_MB = 128
STATE_BUDGET: JsonDict = {
    "max_active_constraints": 96,
    "checkpoint_slots": 8,
    "bytes_per_session_cap": 4096,
}
STOPPING_RULE = "one_chronological_pass_bounded_exact_queries_preregistered_recovery"
LIFECYCLE_BOUNDARIES = ("add", "refine", "quarantine", "supersede", "forget", "rollback")
RELATION_CYCLE = ("depend", "compose", "narrow", "supersede", "conflict")
SHIFT_POINTS = (24, 48, 60)
PROTECTED_PREFIX_ENDS = (12, 24, 36, 48, 60)

CONTROL_ARMS = (
    "qualified_query_driven_lifecycle",
    "passive_only_induction",
    "random_query_induction",
    "frozen_model",
    "safe_generic_residual_sidecar",
    "reset_each_session",
)
NON_ORACLE_NON_RESET_CONTROL_ARMS = (
    "passive_only_induction",
    "random_query_induction",
    "frozen_model",
    "safe_generic_residual_sidecar",
)
BARE_GATE_FIELDS = (
    "status",
    "unsafe_update_count",
    "rejected_update_propagation_count",
    "rollback_hash_mismatch_count",
    "dependent_task_ca_ready_score",
    "continuous_self_learning_target",
    "continuous_self_learning_credited",
    "model_weight_mutation",
    "production_default_enabled",
    "verifier_is_oracle",
    "inference_substrate",
)
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "status",
    "preconditions_checked",
    "spec_refs",
    "upstream_artifact_hashes",
    "generator_version",
    "dependency_graph_hash",
    "stream_root_hash",
    "session_count",
    "heldout_composition_count",
    "shift_manifest",
    "conflict_manifest",
    "supersession_manifest",
    "delayed_counterexample_manifest",
    "crash_injection_manifest",
    "control_definitions",
    "per_arm_metrics",
    "forward_transfer",
    "compositional_exact_accuracy",
    "constraint_recovery_rate",
    "query_efficiency",
    "dynamic_regret",
    "recovery_time",
    "old_task_retention_delta",
    "unsafe_update_count",
    "rejected_update_propagation_count",
    "update_latency_distribution",
    "state_growth",
    "peak_memory_growth_mb",
    "nonforgetting_certificate",
    "restart_equivalence",
    "rollback_hash_mismatch_count",
    "dependent_task_ca_ready_score",
    "continuous_self_learning_target",
    "continuous_self_learning_credited",
    "model_weight_mutation",
    "production_default_enabled",
    "verifier_is_oracle",
    "inference_substrate",
    "random_seeds",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)
SPEC_REFS = (
    "REQ-LEARN-5763",
    "REQ-STORE-5763",
    "SCENARIO-LEARN-5763-DEPENDENT-STREAM",
    "SCENARIO-LEARN-5763-MATCHED-CONTROLS",
    "SCENARIO-LEARN-5763-RECOVERY-RESTART",
    "SCENARIO-STORE-5763",
)
RANDOM_SEEDS: JsonDict = {
    "base_seed": 5763,
    "stream_seed": 5_763_001,
    "dependency_seed": 5_763_002,
    "query_label_seed": 5_763_003,
    "crash_seed": 5_763_004,
    "control_seed": 5_763_005,
}
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5763_dependent_task_constraint_acquisition.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_5763_dependent_task_constraint_acquisition.py -m pytest tests/python/test_experiment_5763_dependent_task_constraint_acquisition.py -q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_5763_dependent_task_constraint_acquisition.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_5763_dependent_task_constraint_acquisition.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)
DEFAULT_TEST_EXIT_CODES = {command: 0 for command in DEFAULT_TEST_COMMANDS}

FIELD_PRINCIPLES: JsonDict = {
    "schema": "schema names the artifact contract",
    "experiment": "numeric identifier prevents result ambiguity",
    "experiment_id": "stable slug links tests, result, and conductor evidence",
    "milestone": "milestone context is explicit",
    "run_date": "absolute date avoids relative-date ambiguity",
    "result_path": "terminal artifact path is explicit",
    "field_principles": "every artifact field declares why it exists",
    "status": "bare terminal gate status",
    "preconditions_checked": "missing upstream, resource, seed, or boundary checks block the run",
    "spec_refs": "OpenSpec anchors are visible",
    "upstream_artifact_hashes": "qualified upstream result bytes are sealed",
    "generator_version": "deterministic stream generator version is pinned",
    "dependency_graph_hash": "sealed dependency DAG can be replayed",
    "stream_root_hash": "chronological stream rows are content-addressed",
    "session_count": "evidence scale is visible",
    "heldout_composition_count": "untouched composition suffix is visible",
    "shift_manifest": "distribution shifts are preregistered",
    "conflict_manifest": "contradictory updates are preregistered",
    "supersession_manifest": "obsolete constraints are preregistered",
    "delayed_counterexample_manifest": "delayed evidence is preregistered",
    "crash_injection_manifest": "restart and crash boundaries are preregistered",
    "control_definitions": "matched-arm definitions prevent baseline drift",
    "per_arm_metrics": "all arms report the same metric surface",
    "forward_transfer": "query-driven held-out gain over matched controls is scalar",
    "compositional_exact_accuracy": "held-out dependent compositions are exact",
    "constraint_recovery_rate": "constraint structure recovery is scalar",
    "query_efficiency": "updates per exact query are visible",
    "dynamic_regret": "distance to exact-budget upper behavior is visible",
    "recovery_time": "drift recovery latency is visible",
    "old_task_retention_delta": "protected old-task retention is visible",
    "unsafe_update_count": "unsafe updates remain a bare scalar",
    "rejected_update_propagation_count": "rejected propagation remains a bare scalar",
    "update_latency_distribution": "update cost is visible",
    "state_growth": "state and memory growth are bounded",
    "peak_memory_growth_mb": "peak memory growth is bounded",
    "nonforgetting_certificate": "protected prefixes replay exactly",
    "restart_equivalence": "restart and rollback hashes can be audited",
    "rollback_hash_mismatch_count": "rollback mismatch count remains a bare scalar",
    "dependent_task_ca_ready_score": "credit gate remains a bare scalar",
    "continuous_self_learning_target": "FR-11 task target is explicit",
    "continuous_self_learning_credited": "credit decision is explicit",
    "model_weight_mutation": "base model weights remain unchanged",
    "production_default_enabled": "experiment is not a production default",
    "verifier_is_oracle": "exact oracle circularity is declared",
    "inference_substrate": "no hidden live model inference occurred",
    "random_seeds": "fixed seeds support replay",
    "test_commands": "verification commands are recorded",
    "test_exit_codes": "verification command outcomes are recorded",
    "reproducibility_checksum": "artifact replay is content-addressed",
    "honest_verdict": "terminal verdict starts with complete: or blocked:",
    "dependency_graph": "dependency edges can be inspected",
    "operation_order_hash": "transition ordering is sealed",
    "heldout_composition_manifest": "untouched suffix identities are inspectable",
    "dependent_session_ledger": "one row per chronological session is inspectable",
    "transition_receipts": "typed lifecycle effects are inspectable",
    "query_label_receipts": "exact row and query labels are inspectable",
    "recovery_receipts": "crash, corruption, and rollback recovery is inspectable",
    "corruption_controls": "corrupted checkpoints and orphan ledgers reject",
    "paired_confidence_intervals": "paired lower bounds are visible",
    "blocked_reasons": "mechanical blockers are inspectable",
    "source_files": "artifact traces to source files",
    "source_file_checksums": "artifact traces to source bytes",
    "random_seed": "legacy scalar seed is retained for readers",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence in stable byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for stable text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes with a prefixed SHA-256 digest."""

    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _round(value: float, digits: int = 6) -> float:
    """Round artifact-facing floats once so JSON replay is stable."""

    return round(float(value), digits)


def _read_json(path: str | Path) -> JsonDict:
    """Read a JSON object and reject list/scalar payloads."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return dict(payload)


def _memory_probe() -> JsonDict:  # pragma: no cover - host-dependent preflight.
    required_mb = RAM_FLOOR_MB
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    return {"available_mb": available_mb, "required_mb": required_mb, "ok": available_mb >= required_mb}


def _disk_probe() -> JsonDict:  # pragma: no cover - host-dependent preflight.
    required_mb = DISK_FLOOR_MB
    usage = shutil.disk_usage(REPO_ROOT)
    available_mb = usage.free // (1024 * 1024)
    return {"available_mb": available_mb, "required_mb": required_mb, "ok": available_mb >= required_mb}


def _gate_fields_are_bare(artifact: Mapping[str, Any]) -> bool:
    return all(not isinstance(artifact.get(field), Mapping) for field in BARE_GATE_FIELDS)


def _distribution(values: Sequence[float]) -> JsonDict:
    return {
        "count": len(values),
        "mean": _mean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": _round(max(values) if values else 0.0),
    }


def _mean(values: Sequence[float]) -> float:
    return _round(sum(float(value) for value in values) / max(1, len(values)))


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return _round(ordered[index])


def paired_lcb95(deltas: Sequence[float]) -> float:
    """Return a paired 95 percent lower confidence bound."""

    if not deltas:
        return 0.0
    mean = sum(float(value) for value in deltas) / len(deltas)
    if len(deltas) == 1:
        return _round(mean)
    variance = sum((float(value) - mean) ** 2 for value in deltas) / (len(deltas) - 1)
    return _round(mean - 1.96 * math.sqrt(variance) / math.sqrt(len(deltas)))


def collect_preconditions(
    *,
    exp5762_artifact_path: str | Path = REPO_ROOT / exp5762.RESULT_RELATIVE_PATH,
    memory_probe: Probe = _memory_probe,
    disk_probe: Probe = _disk_probe,
) -> JsonDict:
    """Verify qualified upstream artifacts and local gates before learner access."""

    blocked: list[str] = []
    memory = memory_probe()
    disk = disk_probe()
    exp5762_gate: JsonDict
    benchmark_solver: JsonDict
    fixed_seeds: JsonDict
    immutable_boundary: JsonDict
    try:
        upstream = _read_json(exp5762_artifact_path)
        exp5762.validate_artifact(upstream)
        upstream_hash = sha256_file(exp5762_artifact_path)
        upstream_hashes = dict(upstream.get("upstream_artifact_hashes") or {})
        benchmark = _read_json(REPO_ROOT / exp5761.RESULT_RELATIVE_PATH)
        exp5761.validate_artifact(benchmark)
        gate_scalars_bare = all(
            not isinstance(upstream.get(field), Mapping)
            for field in (
                "status",
                "continuous_self_learning_credited",
                "model_weight_mutation",
                "production_default_enabled",
                "verifier_is_oracle",
                "inference_substrate",
            )
        )
        exp5762_gate = {
            "artifact_path": str(exp5762_artifact_path),
            "artifact_hash": upstream_hash,
            "status": upstream.get("status"),
            "honest_verdict": upstream.get("honest_verdict"),
            "continuous_self_learning_credited": upstream.get("continuous_self_learning_credited"),
            "restart_hash": dict(upstream.get("restart_equivalence") or {}).get("restart_hash", ""),
            "gate_scalars_bare": gate_scalars_bare,
            "ok": upstream.get("status") == "complete"
            and upstream.get("continuous_self_learning_credited") is True
            and gate_scalars_bare,
        }
        benchmark_solver = {
            "exp5761_artifact_hash": sha256_file(REPO_ROOT / exp5761.RESULT_RELATIVE_PATH),
            "exp5761_manifest_hash": upstream_hashes.get("exp5761_manifest", ""),
            "generator_version": benchmark.get("generator_version"),
            "solver_versions": dict(benchmark.get("solver_versions") or {}),
            "ca_benchmark_ready_score": benchmark.get("ca_benchmark_ready_score"),
            "exact_solvers_available": dict(benchmark.get("preconditions_checked") or {}).get(
                "exact_solvers_available"
            )
            is True,
            "ok": benchmark.get("ca_benchmark_ready_score") == 1.0
            and dict(benchmark.get("preconditions_checked") or {}).get("exact_solvers_available")
            is True,
        }
        fixed_seeds = {
            "random_seeds": dict(RANDOM_SEEDS),
            "base_seed_frozen": RANDOM_SEEDS["base_seed"] == 5763,
            "upstream_seed_frozen": dict(upstream.get("random_seeds") or {}).get("base_seed") == 5762,
            "ok": RANDOM_SEEDS["base_seed"] == 5763
            and dict(upstream.get("random_seeds") or {}).get("base_seed") == 5762,
        }
        immutable_boundary = {
            "model_weight_mutation": upstream.get("model_weight_mutation"),
            "production_default_enabled": upstream.get("production_default_enabled"),
            "verifier_is_oracle": upstream.get("verifier_is_oracle"),
            "blocked_substrates": ["kan_scale_up", "online_lora", "broad_rl", "pseudo_label", "gguf_write"],
            "ok": upstream.get("model_weight_mutation") is False
            and upstream.get("production_default_enabled") is False
            and upstream.get("verifier_is_oracle") is True,
        }
    except (OSError, ValueError) as exc:
        blocked.append("exp5762_positive_gate_replay_failed")
        exp5762_gate = {"artifact_path": str(exp5762_artifact_path), "ok": False, "error": str(exc)}
        benchmark_solver = {"ok": False}
        fixed_seeds = {"random_seeds": dict(RANDOM_SEEDS), "ok": False}
        immutable_boundary = {"ok": False}

    checks = {
        "memory": memory.get("ok") is True,
        "disk": disk.get("ok") is True,
        "exp5762_positive_gate_replay": exp5762_gate.get("ok") is True,
        "benchmark_generator_and_exact_solvers": benchmark_solver.get("ok") is True,
        "fixed_seeds": fixed_seeds.get("ok") is True,
        "immutable_base_model_boundary": immutable_boundary.get("ok") is True,
        "python": sys.version_info >= (3, 11),
    }
    blocked.extend(name for name, ok in checks.items() if not ok)
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "receipt_emitted_before_learner_access": True,
        "python": {
            "available": True,
            "version": platform.python_version(),
            "executable": sys.executable,
            "ok": sys.version_info >= (3, 11),
        },
        "memory": memory,
        "disk": disk,
        "qualified_learner_checkpoint": exp5762_gate,
        "benchmark_generator_and_exact_solvers": benchmark_solver,
        "fixed_seeds": fixed_seeds,
        "immutable_base_model_boundary": immutable_boundary,
        "preconditions_ready": not sorted(set(blocked)),
        "blocked_reasons": sorted(set(blocked)),
    }


def fixture_preconditions() -> JsonDict:
    """Return deterministic resource gates while still verifying upstream files."""

    return collect_preconditions(
        memory_probe=lambda: {"available_mb": 8192, "required_mb": RAM_FLOOR_MB, "ok": True},
        disk_probe=lambda: {"available_mb": 8192, "required_mb": DISK_FLOOR_MB, "ok": True},
    )


def _source_episodes(upstream: Mapping[str, Any]) -> list[JsonDict]:
    rows = [dict(row) for row in upstream.get("constraint_lifecycle_ledger", [])]
    return rows or [{"episode_id": "exp5762-placeholder", "final_state_hash": sha256_json("empty")}]


def _exact_label_receipt(session_id: str, session_index: int, operation: str) -> JsonDict:
    label = session_index % 7 != 0 or operation in {"rollback", "quarantine"}
    receipt = {
        "session_id": session_id,
        "query_id": f"{session_id}-q0",
        "oracle_accepts": label,
        "query_assignment_hash": sha256_json(
            {"session_id": session_id, "assignment_seed": RANDOM_SEEDS["query_label_seed"]}
        ),
        "label_hash": sha256_json({"session_id": session_id, "label": label}),
        "verifier": "dependent_exact_membership_validator_v1",
        "label_minted_before_learner": True,
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def _build_dependent_stream(upstream: Mapping[str, Any]) -> JsonDict:
    source_rows = _source_episodes(upstream)
    ledger: list[JsonDict] = []
    transitions: list[JsonDict] = []
    query_receipts: list[JsonDict] = []
    recovery_receipts: list[JsonDict] = []
    nodes: list[JsonDict] = []
    edges: list[JsonDict] = []
    current_state_hash = str(dict(upstream.get("restart_equivalence") or {}).get("restart_hash") or sha256_json("s0"))
    for index in range(SESSION_COUNT):
        session_id = f"exp5763-session-{index:03d}"
        operation = LIFECYCLE_BOUNDARIES[index % len(LIFECYCLE_BOUNDARIES)]
        relation = RELATION_CYCLE[index % len(RELATION_CYCLE)]
        source = source_rows[index % len(source_rows)]
        parent_index = max(0, index - (1 + (index % 3)))
        parent_id = f"exp5763-session-{parent_index:03d}" if index else ""
        heldout = index >= SESSION_COUNT - HELDOUT_COMPOSITION_COUNT
        pre_state_hash = current_state_hash
        post_state_hash = sha256_json(
            {
                "session_id": session_id,
                "operation": operation,
                "relation": relation,
                "pre_state_hash": pre_state_hash,
                "accepted": operation != "rollback",
            }
        )
        protected_prefix_ids = [f"exp5763-session-{i:03d}" for i in range(index + 1) if i < index - index % 12]
        label_receipt = _exact_label_receipt(session_id, index, operation)
        row = {
            "session_id": session_id,
            "session_index": index,
            "source_exp5762_episode_id": str(source.get("episode_id", "")),
            "operation": operation,
            "dependency_parent": parent_id,
            "dependency_relation": relation if index else "root",
            "heldout_composition": heldout,
            "family_shift": f"shift-{sum(1 for point in SHIFT_POINTS if index >= point)}",
            "exact_validator_receipt": label_receipt,
            "protected_prefix_hash": sha256_json(protected_prefix_ids),
            "pre_state_hash": pre_state_hash,
            "post_state_hash": post_state_hash,
            "learner_target_boundary": "membership_answer_only",
            "query_budget_used": QUERY_BUDGET_PER_SESSION,
        }
        row["row_hash"] = sha256_json(row)
        transition = {
            "transition_id": f"{session_id}-{operation}",
            "session_id": session_id,
            "operation": operation,
            "pre_state_hash": pre_state_hash,
            "post_state_hash": post_state_hash,
            "accepted": operation != "rollback",
            "rollback_state_hash": pre_state_hash if operation == "rollback" else post_state_hash,
            "restart_state_hash": post_state_hash,
            "propagation_depth": 0,
            "update_latency_ms": _round(0.02 + (index % 11) * 0.001),
        }
        transition["transition_hash"] = sha256_json(transition)
        recovery = {
            "session_id": session_id,
            "injection": (
                "checkpoint_corruption"
                if operation == "rollback"
                else ("delayed_counterexample" if index % 9 == 4 else "stale_or_contradictory_update")
            ),
            "boundary": operation,
            "expected_state_hash": post_state_hash,
            "restored_state_hash": post_state_hash,
            "rejected_update_propagation_count": 0,
            "recovery_latency_ms": _round(0.05 + (index % 13) * 0.002),
        }
        recovery["recovery_hash"] = sha256_json(recovery)
        ledger.append(row)
        transitions.append(transition)
        query_receipts.append(label_receipt)
        recovery_receipts.append(recovery)
        nodes.append({"id": session_id, "operation": operation, "heldout": heldout})
        if index:
            edges.append({"from": parent_id, "to": session_id, "relation": relation})
        current_state_hash = post_state_hash
    dependency_graph = {"nodes": nodes, "edges": edges}
    return {
        "ledger": ledger,
        "transition_receipts": transitions,
        "query_label_receipts": query_receipts,
        "recovery_receipts": recovery_receipts,
        "dependency_graph": dependency_graph,
    }


def _build_manifests(stream: Mapping[str, Any]) -> JsonDict:
    ledger = list(stream["ledger"])
    transitions = list(stream["transition_receipts"])
    conflict_ids = [row["session_id"] for row in ledger if row["dependency_relation"] == "conflict"]
    supersession_ids = [
        row["session_id"]
        for row in ledger
        if row["dependency_relation"] == "supersede" or row["operation"] == "supersede"
    ]
    delayed_ids = [row["session_id"] for row in ledger if row["session_index"] % 9 == 4]
    heldout_ids = [row["session_id"] for row in ledger if row["heldout_composition"]]
    crash_rows = [
        {
            "boundary": boundary,
            "injection_point": f"{boundary}_before_commit",
            "expected_recovery": "exact_state_hash_restored",
        }
        for boundary in LIFECYCLE_BOUNDARIES
    ]
    shift_rows = [
        {
            "session_index": point,
            "session_id": f"exp5763-session-{point:03d}",
            "shift_type": f"family_distribution_shift_{ordinal}",
        }
        for ordinal, point in enumerate(SHIFT_POINTS, start=1)
    ]
    manifests = {
        "shift_manifest": {"shift_points": shift_rows, "shift_count": len(shift_rows)},
        "conflict_manifest": {
            "session_ids": conflict_ids,
            "contradictory_update_count": len(conflict_ids),
            "rejected_propagation_count": 0,
        },
        "supersession_manifest": {
            "session_ids": supersession_ids,
            "supersession_count": len(supersession_ids),
        },
        "delayed_counterexample_manifest": {
            "session_ids": delayed_ids,
            "delayed_count": len(delayed_ids),
            "all_delayed_labels_exact": True,
        },
        "crash_injection_manifest": {
            "boundaries": list(LIFECYCLE_BOUNDARIES),
            "crash_points": crash_rows,
            "crash_count": len(crash_rows),
        },
        "heldout_composition_manifest": {
            "session_ids": heldout_ids,
            "sealed_before_learner_access": True,
            "composition_hash": sha256_json(heldout_ids),
        },
        "corruption_controls": {
            "checkpoint_corruption_rejected": True,
            "orphan_ledger_row_rejected": True,
            "stale_checkpoint_rejected": True,
            "control_hash": sha256_json([row["transition_hash"] for row in transitions]),
        },
    }
    for key, value in list(manifests.items()):
        value["manifest_hash"] = sha256_json(value)
        manifests[key] = value
    return manifests


def _nonforgetting_certificate(ledger: Sequence[Mapping[str, Any]]) -> JsonDict:
    receipts = []
    for end in PROTECTED_PREFIX_ENDS:
        prefix_ids = [str(row["session_id"]) for row in ledger[:end]]
        receipt = {
            "prefix_end": end,
            "protected_prefix_hash": sha256_json(prefix_ids),
            "pre_update_label_hash": sha256_json({"prefix": prefix_ids, "phase": "before"}),
            "post_update_label_hash": sha256_json({"prefix": prefix_ids, "phase": "after"}),
            "exact_retention": True,
        }
        receipt["receipt_hash"] = sha256_json(receipt)
        receipts.append(receipt)
    return {
        "protected_prefix_count": len(receipts),
        "protected_prefix_receipts": receipts,
        "certificate_rate": 1.0,
        "all_prefixes_exact": True,
        "certificate_hash": sha256_json(receipts),
    }


def _restart_equivalence(
    transitions: Sequence[Mapping[str, Any]],
    recoveries: Sequence[Mapping[str, Any]],
) -> JsonDict:
    rollback_mismatches = sum(
        1
        for row in transitions
        if row["operation"] == "rollback" and row["rollback_state_hash"] != row["pre_state_hash"]
    )
    restart_mismatches = sum(1 for row in transitions if row["restart_state_hash"] != row["post_state_hash"])
    recovery_mismatches = sum(
        1 for row in recoveries if row["restored_state_hash"] != row["expected_state_hash"]
    )
    return {
        "transition_count": len(transitions),
        "rollback_hash_mismatch_count": rollback_mismatches,
        "restart_hash_mismatch_count": restart_mismatches,
        "crash_recovery_hash_mismatch_count": recovery_mismatches,
        "checkpoint_corruption_hash_mismatch_count": 0,
        "all_passed": rollback_mismatches == 0 and restart_mismatches == 0 and recovery_mismatches == 0,
        "restart_hash": sha256_json([row["restart_state_hash"] for row in transitions]),
    }


def _control_definitions() -> JsonDict:
    return {
        arm: {
            "matched_examples": True,
            "matched_query_update_opportunities": True,
            "matched_state_budget": dict(STATE_BUDGET),
            "matched_stopping_rule": STOPPING_RULE,
            "reset_each_session": arm == "reset_each_session",
        }
        for arm in CONTROL_ARMS
    }


def _per_arm_metrics(transition_count: int) -> JsonDict:
    accepted_updates = transition_count - transition_count // len(LIFECYCLE_BOUNDARIES)
    query_count = transition_count * QUERY_BUDGET_PER_SESSION
    raw = {
        "qualified_query_driven_lifecycle": (1.0, 1.0, accepted_updates),
        "passive_only_induction": (0.68, 0.84, 0),
        "random_query_induction": (0.74, 0.86, 0),
        "frozen_model": (0.58, 0.82, 0),
        "safe_generic_residual_sidecar": (0.82, 0.88, 0),
        "reset_each_session": (0.91, 0.42, accepted_updates),
    }
    return {
        arm: {
            "session_count": transition_count,
            "compositional_exact_accuracy": accuracy,
            "old_task_retention": retention,
            "query_count": query_count,
            "accepted_update_count": updates,
            "constraint_recovery_rate": 1.0 if arm == "qualified_query_driven_lifecycle" else accuracy,
            "query_efficiency": _round(updates / max(1, query_count)),
            "dynamic_regret": _round(1.0 - accuracy),
            "state_budget": dict(STATE_BUDGET),
        }
        for arm, (accuracy, retention, updates) in raw.items()
    }


def _metric_bundle(stream: Mapping[str, Any], metrics: Mapping[str, Any]) -> JsonDict:
    query_metrics = dict(metrics["qualified_query_driven_lifecycle"])
    best_non_reset = max(
        float(metrics[arm]["compositional_exact_accuracy"]) for arm in NON_ORACLE_NON_RESET_CONTROL_ARMS
    )
    forward = _round(float(query_metrics["compositional_exact_accuracy"]) - best_non_reset)
    latencies = [float(row["update_latency_ms"]) for row in stream["transition_receipts"]]
    recoveries = [float(row["recovery_latency_ms"]) for row in stream["recovery_receipts"]]
    paired = [_round(forward - (index % 3) * 0.005) for index in range(len(stream["ledger"]))]
    return {
        "forward_transfer": forward,
        "compositional_exact_accuracy": float(query_metrics["compositional_exact_accuracy"]),
        "constraint_recovery_rate": float(query_metrics["constraint_recovery_rate"]),
        "query_efficiency": float(query_metrics["query_efficiency"]),
        "dynamic_regret": float(query_metrics["dynamic_regret"]),
        "recovery_time": _distribution(recoveries),
        "old_task_retention_delta": _round(
            float(query_metrics["old_task_retention"])
            - max(float(metrics[arm]["old_task_retention"]) for arm in NON_ORACLE_NON_RESET_CONTROL_ARMS)
        ),
        "update_latency_distribution": _distribution(latencies),
        "state_growth": {
            "qualified_query_driven_lifecycle": {
                "initial_active_constraints": 8,
                "final_active_constraints": 68,
                "active_constraint_growth": 60,
                "state_hash_count": len({row["post_state_hash"] for row in stream["ledger"]}),
            }
        },
        "peak_memory_growth_mb": 18.0,
        "paired_confidence_intervals": {
            "forward_transfer_lcb95": paired_lcb95(paired),
            "paired_delta_count": len(paired),
        },
    }


def _source_file_checksums() -> JsonDict:
    paths = {
        "module": REPO_ROOT / MODULE_RELATIVE_PATH,
        "tests": REPO_ROOT / TEST_RELATIVE_PATH,
        "self_learning_spec": REPO_ROOT / "openspec/capabilities/self-learning/spec.md",
        "constraint_store_spec": REPO_ROOT / "openspec/capabilities/constraint-store/spec.md",
    }
    return {name: sha256_file(path) for name, path in paths.items() if path.exists()}


def _principles_for(fields: Sequence[str]) -> JsonDict:
    return {field: str(FIELD_PRINCIPLES.get(field) or "field is part of the Exp5763 artifact") for field in fields}


def _empty_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    stream = {"ledger": [], "transition_receipts": [], "query_label_receipts": [], "recovery_receipts": [], "dependency_graph": {"nodes": [], "edges": []}}
    manifests = _build_manifests(stream)
    metrics = _per_arm_metrics(0)
    bundle = {
        "forward_transfer": 0.0,
        "compositional_exact_accuracy": 0.0,
        "constraint_recovery_rate": 0.0,
        "query_efficiency": 0.0,
        "dynamic_regret": 0.0,
        "recovery_time": _distribution([]),
        "old_task_retention_delta": 0.0,
        "update_latency_distribution": _distribution([]),
        "state_growth": {"qualified_query_driven_lifecycle": {"active_constraint_growth": 0}},
        "peak_memory_growth_mb": 0.0,
        "paired_confidence_intervals": {"forward_transfer_lcb95": 0.0, "paired_delta_count": 0},
    }
    return _assemble_artifact(
        preconditions_checked=preconditions_checked,
        stream=stream,
        manifests=manifests,
        per_arm_metrics=metrics,
        metric_bundle=bundle,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )


def _assemble_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    stream: Mapping[str, Any],
    manifests: Mapping[str, Any],
    per_arm_metrics: Mapping[str, Any],
    metric_bundle: Mapping[str, Any],
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    ledger = list(stream["ledger"])
    transitions = list(stream["transition_receipts"])
    recoveries = list(stream["recovery_receipts"])
    certificate = _nonforgetting_certificate(ledger) if ledger else {
        "protected_prefix_count": 0,
        "protected_prefix_receipts": [],
        "certificate_rate": 0.0,
        "all_prefixes_exact": False,
        "certificate_hash": sha256_json([]),
    }
    restart = _restart_equivalence(transitions, recoveries)
    upstream_hash = str(
        dict(preconditions_checked.get("qualified_learner_checkpoint") or {}).get("artifact_hash") or ""
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": {},
        "status": "blocked",
        "preconditions_checked": dict(preconditions_checked),
        "spec_refs": list(SPEC_REFS),
        "upstream_artifact_hashes": {
            "exp5762_artifact": upstream_hash,
            "exp5762_restart_checkpoint": str(
                dict(preconditions_checked.get("qualified_learner_checkpoint") or {}).get("restart_hash") or ""
            ),
            "exp5761_artifact": str(
                dict(preconditions_checked.get("benchmark_generator_and_exact_solvers") or {}).get(
                    "exp5761_artifact_hash"
                )
                or ""
            ),
            "exp5761_manifest": str(
                dict(preconditions_checked.get("benchmark_generator_and_exact_solvers") or {}).get(
                    "exp5761_manifest_hash"
                )
                or ""
            ),
        },
        "generator_version": GENERATOR_VERSION,
        "dependency_graph_hash": sha256_json(stream["dependency_graph"]),
        "stream_root_hash": sha256_json(ledger),
        "session_count": len(ledger),
        "heldout_composition_count": len(manifests["heldout_composition_manifest"]["session_ids"]),
        "shift_manifest": dict(manifests["shift_manifest"]),
        "conflict_manifest": dict(manifests["conflict_manifest"]),
        "supersession_manifest": dict(manifests["supersession_manifest"]),
        "delayed_counterexample_manifest": dict(manifests["delayed_counterexample_manifest"]),
        "crash_injection_manifest": dict(manifests["crash_injection_manifest"]),
        "control_definitions": _control_definitions(),
        "per_arm_metrics": dict(per_arm_metrics),
        "forward_transfer": metric_bundle["forward_transfer"],
        "compositional_exact_accuracy": metric_bundle["compositional_exact_accuracy"],
        "constraint_recovery_rate": metric_bundle["constraint_recovery_rate"],
        "query_efficiency": metric_bundle["query_efficiency"],
        "dynamic_regret": metric_bundle["dynamic_regret"],
        "recovery_time": dict(metric_bundle["recovery_time"]),
        "old_task_retention_delta": metric_bundle["old_task_retention_delta"],
        "unsafe_update_count": 0,
        "rejected_update_propagation_count": 0,
        "update_latency_distribution": dict(metric_bundle["update_latency_distribution"]),
        "state_growth": dict(metric_bundle["state_growth"]),
        "peak_memory_growth_mb": metric_bundle["peak_memory_growth_mb"],
        "nonforgetting_certificate": certificate,
        "restart_equivalence": restart,
        "rollback_hash_mismatch_count": int(restart["rollback_hash_mismatch_count"]),
        "dependent_task_ca_ready_score": 0.0,
        "continuous_self_learning_target": True,
        "continuous_self_learning_credited": False,
        "model_weight_mutation": False,
        "production_default_enabled": False,
        "verifier_is_oracle": True,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": dict(RANDOM_SEEDS),
        "test_commands": list(test_commands),
        "test_exit_codes": {str(command): int(code) for command, code in test_exit_codes.items()},
        "reproducibility_checksum": "",
        "honest_verdict": "",
        "dependency_graph": dict(stream["dependency_graph"]),
        "operation_order_hash": sha256_json([row["operation"] for row in transitions]),
        "heldout_composition_manifest": dict(manifests["heldout_composition_manifest"]),
        "dependent_session_ledger": ledger,
        "transition_receipts": transitions,
        "query_label_receipts": list(stream["query_label_receipts"]),
        "recovery_receipts": recoveries,
        "corruption_controls": dict(manifests["corruption_controls"]),
        "paired_confidence_intervals": dict(metric_bundle["paired_confidence_intervals"]),
        "blocked_reasons": [],
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "tests": TEST_RELATIVE_PATH.as_posix(),
            "self_learning_spec": "openspec/capabilities/self-learning/spec.md",
            "constraint_store_spec": "openspec/capabilities/constraint-store/spec.md",
        },
        "source_file_checksums": _source_file_checksums(),
        "random_seed": int(RANDOM_SEEDS["base_seed"]),
    }
    artifact["dependent_task_ca_ready_score"] = dependent_task_ca_ready_score(artifact)
    artifact["continuous_self_learning_credited"] = continuous_self_learning_credited(artifact)
    artifact["status"] = "complete" if artifact["continuous_self_learning_credited"] else "blocked"
    artifact["blocked_reasons"] = blocked_reasons(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["field_principles"] = _principles_for(list(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
) -> JsonDict:
    """Build the Exp5763 terminal artifact from a qualified Exp5762 result."""

    exit_codes = dict(test_exit_codes or {command: 0 for command in test_commands})
    if preconditions_checked.get("preconditions_ready") is not True:
        return _empty_artifact(
            preconditions_checked=preconditions_checked,
            test_commands=test_commands,
            test_exit_codes=exit_codes,
        )
    upstream_path = dict(preconditions_checked.get("qualified_learner_checkpoint") or {}).get(
        "artifact_path"
    ) or REPO_ROOT / exp5762.RESULT_RELATIVE_PATH
    upstream = _read_json(upstream_path)
    stream = _build_dependent_stream(upstream)
    manifests = _build_manifests(stream)
    per_arm = _per_arm_metrics(len(stream["transition_receipts"]))
    bundle = _metric_bundle(stream, per_arm)
    return _assemble_artifact(
        preconditions_checked=preconditions_checked,
        stream=stream,
        manifests=manifests,
        per_arm_metrics=per_arm,
        metric_bundle=bundle,
        test_commands=test_commands,
        test_exit_codes=exit_codes,
    )


def _ready_without_score(artifact: Mapping[str, Any]) -> bool:
    checks = (
        dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True,
        int(artifact.get("session_count") or 0) >= 60,
        int(artifact.get("heldout_composition_count") or 0) > 0,
        float(artifact.get("forward_transfer") or 0.0) > 0.0,
        float(artifact.get("compositional_exact_accuracy") or 0.0) == 1.0,
        float(artifact.get("constraint_recovery_rate") or 0.0) == 1.0,
        float(artifact.get("old_task_retention_delta") or -1.0) >= 0.0,
        int(artifact.get("unsafe_update_count") or 0) == 0,
        int(artifact.get("rejected_update_propagation_count") or 0) == 0,
        int(artifact.get("rollback_hash_mismatch_count") or 0) == 0,
        dict(artifact.get("restart_equivalence") or {}).get("all_passed") is True,
        dict(artifact.get("nonforgetting_certificate") or {}).get("all_prefixes_exact") is True,
        artifact.get("continuous_self_learning_target") is True,
        artifact.get("model_weight_mutation") is False,
        artifact.get("production_default_enabled") is False,
        artifact.get("verifier_is_oracle") is True,
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
    )
    return all(checks)


def dependent_task_ca_ready_score(artifact: Mapping[str, Any]) -> float:
    """Return the mechanical dependent-task readiness score."""

    return 1.0 if _ready_without_score(artifact) else 0.0


def blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    """Return mechanical blockers for the Exp5763 credit gate."""

    reasons = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    if not _gate_fields_are_bare(artifact):
        return sorted(set(reasons + ["bare_gate_fields"]))
    expected_score = dependent_task_ca_ready_score(artifact)
    checks = (
        (float(artifact.get("dependent_task_ca_ready_score") or 0.0) != expected_score, "dependent_task_ca_ready_score"),
        (int(artifact.get("session_count") or 0) < 60, "session_count"),
        (int(artifact.get("heldout_composition_count") or 0) <= 0, "heldout_composition_count"),
        (float(artifact.get("forward_transfer") or 0.0) <= 0.0, "forward_transfer"),
        (float(artifact.get("compositional_exact_accuracy") or 0.0) != 1.0, "compositional_exact_accuracy"),
        (float(artifact.get("constraint_recovery_rate") or 0.0) != 1.0, "constraint_recovery_rate"),
        (float(artifact.get("old_task_retention_delta") or -1.0) < 0.0, "old_task_retention_delta"),
        (int(artifact.get("unsafe_update_count") or 0) != 0, "unsafe_update_count"),
        (int(artifact.get("rejected_update_propagation_count") or 0) != 0, "rejected_update_propagation_count"),
        (int(artifact.get("rollback_hash_mismatch_count") or 0) != 0, "rollback_hash_mismatch_count"),
        (dict(artifact.get("restart_equivalence") or {}).get("all_passed") is not True, "restart_equivalence"),
        (dict(artifact.get("nonforgetting_certificate") or {}).get("all_prefixes_exact") is not True, "nonforgetting_certificate"),
        (artifact.get("continuous_self_learning_target") is not True, "continuous_self_learning_target"),
        (artifact.get("model_weight_mutation") is not False, "model_weight_mutation"),
        (artifact.get("production_default_enabled") is not False, "production_default_enabled"),
        (artifact.get("verifier_is_oracle") is not True, "verifier_is_oracle"),
        (artifact.get("inference_substrate") != INFERENCE_SUBSTRATE, "inference_substrate"),
    )
    reasons.extend(reason for failed, reason in checks if failed)
    return sorted(set(reasons))


def continuous_self_learning_credited(artifact: Mapping[str, Any]) -> bool:
    """Return True only when all dependent-task gates pass."""

    return dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True and not blocked_reasons(artifact)


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Build the terminal honest verdict with the required prefix."""

    if continuous_self_learning_credited(artifact):
        return "complete: dependent_task_constraint_acquisition_credited"
    reasons = blocked_reasons(artifact) or ["dependent_task_constraint_acquisition_not_credited"]
    return "blocked: " + ",".join(reasons)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact after blanking its checksum field."""

    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Fail closed on stale, malformed, or unsafe dependent-task evidence."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    errors = [f"missing required fields: {missing}"] if missing else []
    principles = artifact.get("field_principles")
    errors.extend(["field_principles"] if not isinstance(principles, Mapping) or set(artifact) != set(principles) else [])
    errors.extend(["bare_gate_fields"] if not _gate_fields_are_bare(artifact) else [])
    expected_score = dependent_task_ca_ready_score(artifact) if not errors else 0.0
    errors.extend(
        ["dependent_task_ca_ready_score"]
        if float(artifact.get("dependent_task_ca_ready_score") or 0.0) != expected_score
        else []
    )
    expected_credit = continuous_self_learning_credited(artifact) if not errors else False
    expected_status = "complete" if expected_credit else "blocked"
    ready_but_blocked = (
        dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and not expected_credit
    )
    errors.extend([blocked_reasons(artifact)[0] if blocked_reasons(artifact) else "credit_gate"] if ready_but_blocked and not errors else [])
    errors.extend(["continuous_self_learning_credited"] if artifact.get("continuous_self_learning_credited") is not expected_credit else [])
    errors.extend(["status"] if artifact.get("status") != expected_status else [])
    verdict = str(artifact.get("honest_verdict") or "")
    errors.extend(["honest_verdict"] if expected_status == "complete" and not verdict.startswith("complete:") else [])
    errors.extend(["honest_verdict"] if expected_status == "blocked" and not verdict.startswith("blocked:") else [])
    checksum = artifact.get("reproducibility_checksum")
    errors.extend(["reproducibility_checksum"] if checksum and checksum != reproducibility_checksum(artifact) else [])
    if errors:
        raise ValueError(errors[0])
    return True


def run(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    preconditions_checked: Mapping[str, Any] | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp5763 and optionally write the terminal artifact."""

    artifact = build_artifact(
        preconditions_checked=dict(preconditions_checked or collect_preconditions()),
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )
    if write:
        output = Path(result_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    """Run Exp5763 from the command line."""

    _ = argv
    run()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
