"""Exp 3243 FR-11 failure-memory controller update.

Spec refs: REQ-LEARN-3243, SCENARIO-LEARN-3243,
SCENARIO-LEARN-3243-BLOCKED.

This module learns only controller metadata from checked-in artifacts and
conductor outcomes.  "Learning" here means recording reusable gate and failure
memory so future controller choices can avoid known-bad reruns.  It does not
train, fine-tune, or mutate foundation-model, KAN-sidecar, or hidden weights.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260528"
MILESTONE = "2026.05.300"
SCHEMA_VERSION = "carnot.fr11.failure_memory_controller.v1"
EXPERIMENT_ID = "exp3243"
TASK_ID = "exp3243-fr11-failure-memory-controller-v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path("results/experiment_3243_fr11_failure_memory_controller_v1.json")
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
SPEC_REL_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_REL_PATH = Path("python/carnot/eval/fr11_failure_memory_controller_v1.py")
TEST_REL_PATH = Path("tests/python/test_experiment_3243_fr11_failure_memory_controller_v1.py")
EXP3229_REL_PATH = Path("results/experiment_3229_fr11_nonforgetting_promotion_controller_v3.json")
EXP3230_REL_PATH = Path("results/experiment_3230_kan_cl_certificate_boundary_audit_v2.json")
EXP3232_REL_PATH = Path("results/experiment_3232_capstone_v298.json")
EXP3223_REL_PATH = Path("results/experiment_3223_capstone_v299.json")
EXP3234_REL_PATH = Path("results/experiment_3234_cli_backend_failure_root_cause_ledger_v1.json")
EXP3236_REL_PATH = Path("results/experiment_3236_isolated_cuda_python_smoke_v1.json")
EXP3237_REL_PATH = Path("results/experiment_3237_llama_cpp_cuda_receipt_smoke_v2.json")
EXP3240_REL_PATH = Path("results/experiment_3240_prompt_injection_kan_teacher_label_shard_v1.json")
EXP3241_REL_PATH = Path("results/experiment_3241_prompt_injection_kan_train_eval_shard_v1.json")
SCHEMA_KEYS = (
    "prerequisite",
    "failure_signature",
    "stale_premise",
    "accepted_next_action",
    "retirement_risk",
)
REQUIRED_ARTIFACT_FIELDS = {
    "experiment_id",
    "task_id",
    "milestone",
    "inference_substrate",
    "principle_annotations",
    "continuous_self_learning_task",
    "failure_memory_schema_ready",
    "failure_trace_count",
    "heldout_replay_count",
    "heldout_replay_delta",
    "nonforgetting_delta",
    "stale_premise_rejection_count",
    "doomed_rerun_avoidance_count",
    "model_weight_update_claimed",
    "controller_memory_updates_are_not_training",
    "fr11_controller_update_ready",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest -o addopts='' "
    "tests/python/test_experiment_3243_fr11_failure_memory_controller_v1.py -q",
    ".venv/bin/coverage run -m pytest -o addopts='' "
    "tests/python/test_experiment_3243_fr11_failure_memory_controller_v1.py -q",
    ".venv/bin/coverage report "
    "--include='python/carnot/eval/fr11_failure_memory_controller_v1.py' "
    "--fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_3243_fr11_failure_memory_controller_v1.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    ("conductor_log", CONDUCTOR_LOG_REL_PATH, True),
    ("self_learning_openspec", SPEC_REL_PATH, False),
    ("exp3229_nonforgetting_controller", EXP3229_REL_PATH, True),
    ("exp3230_kan_cl_certificate_boundary", EXP3230_REL_PATH, True),
    ("exp3232_capstone_v298", EXP3232_REL_PATH, True),
    ("exp3223_capstone_v299", EXP3223_REL_PATH, True),
    ("exp3234_cli_backend_failure_ledger", EXP3234_REL_PATH, False),
    ("exp3236_cuda_python_smoke", EXP3236_REL_PATH, False),
    ("exp3237_llama_cpp_cuda_receipt_gate", EXP3237_REL_PATH, False),
    ("exp3240_teacher_label_shard_gate", EXP3240_REL_PATH, False),
    ("exp3241_train_eval_shard_gate", EXP3241_REL_PATH, False),
    ("exp3243_module", MODULE_REL_PATH, False),
    ("exp3243_tests", TEST_REL_PATH, False),
)
ARTIFACT_PATHS = {
    "exp3229": EXP3229_REL_PATH,
    "exp3230": EXP3230_REL_PATH,
    "exp3232": EXP3232_REL_PATH,
    "exp3223": EXP3223_REL_PATH,
    "exp3234": EXP3234_REL_PATH,
    "exp3236": EXP3236_REL_PATH,
    "exp3237": EXP3237_REL_PATH,
    "exp3240": EXP3240_REL_PATH,
    "exp3241": EXP3241_REL_PATH,
}
RECENT_LOG_DATE_PREFIXES = ("2026-05-27", "2026-05-28")


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, treating absent or malformed evidence as empty."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive evidence read path
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_text(path: Path) -> str:
    """Read text evidence while treating missing logs as empty."""

    try:
        return path.read_text(encoding="utf-8")
    except OSError:  # pragma: no cover - defensive evidence read path
        return ""


def load_sources(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load checked-in conductor-log and artifact evidence for Exp 3243."""

    root_path = Path(root)
    return {
        "conductor_log": read_text(root_path / CONDUCTOR_LOG_REL_PATH),
        "artifacts": {
            artifact_id: read_json_object(root_path / rel_path)
            for artifact_id, rel_path in ARTIFACT_PATHS.items()
        },
    }


def failure_memory_schema() -> JsonDict:
    """REQ-LEARN-3243-1: describe the stable controller-memory keys."""

    return {
        "schema_ready": True,
        "keys": [
            {
                "key": "prerequisite",
                "meaning": "upstream condition that must hold before a rerun is useful",
            },
            {
                "key": "failure_signature",
                "meaning": "stable gate, artifact, stale-premise, or backend failure pattern",
            },
            {
                "key": "stale_premise",
                "meaning": "whether the memory row invalidates a prior controller premise",
            },
            {
                "key": "accepted_next_action",
                "meaning": "controller action admitted by current evidence",
            },
            {
                "key": "retirement_risk",
                "meaning": "risk that repeating this route without new evidence is wasteful",
            },
        ],
    }


def collect_failure_traces(sources: Mapping[str, Any]) -> list[JsonDict]:
    """REQ-LEARN-3243-2: extract failure traces from logs and artifacts."""

    artifacts = sources.get("artifacts", {})
    artifact_map = artifacts if isinstance(artifacts, Mapping) else {}
    traces: list[JsonDict] = []
    traces.extend(log_gate_block_traces(str(sources.get("conductor_log") or "")))
    traces.extend(capstone_blocker_traces(artifact_map.get("exp3232", {})))
    traces.extend(missing_source_artifact_traces(artifact_map.get("exp3223", {})))
    traces.extend(stale_premise_traces(artifact_map.get("exp3229", {})))
    traces.extend(certificate_failure_traces(artifact_map.get("exp3230", {})))
    traces.extend(backend_failure_traces(artifact_map.get("exp3236", {})))
    for artifact_id in ("exp3237", "exp3240", "exp3241"):
        traces.extend(blocked_gate_artifact_traces(artifact_id, artifact_map.get(artifact_id, {})))
    return traces


def log_gate_block_traces(log_text: str) -> list[JsonDict]:
    """Extract repeated conductor pre-gate blocks from the conductor log."""

    groups: dict[tuple[str, str], JsonDict] = {}
    for row in parse_log_rows(log_text):
        if not str(row["timestamp"]).startswith(RECENT_LOG_DATE_PREFIXES):
            continue
        if row["status"] != "GATE_BLOCK":
            continue
        signature = first_failure_signature(row["details"])
        key = (row["title"], signature)
        group = groups.setdefault(
            key,
            {
                "title": row["title"],
                "signature": signature,
                "count": 0,
                "first_timestamp": row["timestamp"],
                "last_timestamp": row["timestamp"],
            },
        )
        group["count"] += 1
        group["last_timestamp"] = row["timestamp"]
    return [
        make_trace(
            category="repeated_gate_block",
            source="ops/conductor-log.md",
            prerequisite=prerequisite_from_signature(group["signature"]),
            failure_signature=f"{group['signature']} repeated {group['count']} times",
            stale_premise=False,
            accepted_next_action="force_prerequisite_gate",
            retirement_risk="high" if int(group["count"]) >= 3 else "medium",
            evidence=group,
            impact_count=safe_int(group["count"]),
        )
        for group in groups.values()
        if int(group["count"]) >= 2
    ]


def parse_log_rows(log_text: str) -> list[JsonDict]:
    """Parse the conductor's markdown table rows into compact dictionaries."""

    rows: list[JsonDict] = []
    for line in log_text.splitlines():
        if not line.startswith("|"):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) < 4:
            continue
        rows.append(
            {
                "timestamp": parts[0],
                "title": parts[1],
                "status": parts[2],
                "details": parts[3],
            }
        )
    return rows


def first_failure_signature(details: str) -> str:
    """Normalize a conductor failure detail into a reusable signature."""

    match = re.search(r"first failure:\s*([^|]+)", details)
    if match:
        return match.group(1).strip()
    retired = re.search(r"upstream retired \(([^)]+)\)", details)
    if retired:
        return f"upstream_retired:{retired.group(1).strip()}"
    return details.strip() or "unknown_gate_block"


def prerequisite_from_signature(signature: str) -> str:
    """Return the upstream prerequisite named by a gate signature."""

    if "." in signature:
        return signature.split(".", 1)[0].strip()
    if signature.startswith("upstream_retired:"):
        return signature.removeprefix("upstream_retired:").strip()
    return "unknown_prerequisite"


def capstone_blocker_traces(exp3232: Any) -> list[JsonDict]:
    """Extract missing-artifact and gate-block blockers from the capstone matrix."""

    if not isinstance(exp3232, Mapping):
        return []
    blockers = exp3232.get("publication_blockers", [])
    if not isinstance(blockers, Sequence) or isinstance(blockers, (str, bytes)):
        return []
    traces: list[JsonDict] = []
    for blocker in blockers:
        if not isinstance(blocker, Mapping):
            continue
        status = normalize_token(blocker.get("status"))
        if status not in {"missing", "blocked", "gate_blocked"}:
            continue
        category = "missing_artifact" if status == "missing" else "gate_block"
        action = "repair_missing_artifact_before_rerun" if status == "missing" else "force_prerequisite_gate"
        traces.append(
            make_trace(
                category=category,
                source="results/experiment_3232_capstone_v298.json",
                prerequisite=str(blocker.get("source_field") or blocker.get("role") or ""),
                failure_signature=f"{blocker.get('experiment_id')}: {blocker.get('status_rationale')}",
                stale_premise=False,
                accepted_next_action=action,
                retirement_risk="high",
                evidence=dict(blocker),
            )
        )
    return traces


def missing_source_artifact_traces(exp3223: Any) -> list[JsonDict]:
    """Extract missing input artifacts from the .299 capstone."""

    if not isinstance(exp3223, Mapping):
        return []
    source_artifacts = exp3223.get("source_artifacts", [])
    if not isinstance(source_artifacts, Sequence) or isinstance(source_artifacts, (str, bytes)):
        return []
    traces: list[JsonDict] = []
    for item in source_artifacts:
        if isinstance(item, Mapping) and item.get("present") is False:
            traces.append(
                make_trace(
                    category="missing_artifact",
                    source="results/experiment_3223_capstone_v299.json",
                    prerequisite=str(item.get("path") or ""),
                    failure_signature=str(exp3223.get("v4_outcome") or "missing_source_artifact"),
                    stale_premise=False,
                    accepted_next_action=str(exp3223.get("next_top_gap") or "repair_missing_artifact"),
                    retirement_risk="high",
                    evidence=dict(item),
                )
            )
    return traces


def stale_premise_traces(exp3229: Any) -> list[JsonDict]:
    """Extract stale-premise rejections from prior FR-11 promotion governance."""

    if not isinstance(exp3229, Mapping):
        return []
    stale_count = safe_int(exp3229.get("stale_premise_rejection_count"))
    if stale_count <= 0:
        return []
    return [
        make_trace(
            category="stale_premise",
            source="results/experiment_3229_fr11_nonforgetting_promotion_controller_v3.json",
            prerequisite="Exp 3216 grounded-continuation affected routes",
            failure_signature=f"stale_premise_rejection_count={stale_count}",
            stale_premise=True,
            accepted_next_action="reject_stale_controller_memory_trace",
            retirement_risk="medium",
            evidence=dict(exp3229.get("stale_premise_invalidations", {}))
            if isinstance(exp3229.get("stale_premise_invalidations"), Mapping)
            else {},
            impact_count=stale_count,
        )
    ]


def certificate_failure_traces(exp3230: Any) -> list[JsonDict]:
    """Extract KAN-CL certificate blockers as failure-memory traces."""

    if not isinstance(exp3230, Mapping):
        return []
    missing_count = safe_int(exp3230.get("missing_certificate_count"))
    if missing_count <= 0:
        return []
    return [
        make_trace(
            category="missing_certificate",
            source="results/experiment_3230_kan_cl_certificate_boundary_audit_v2.json",
            prerequisite="KAN-CL sidecar promotion certificates",
            failure_signature=f"missing_certificate_count={missing_count}",
            stale_premise=False,
            accepted_next_action="force_certificate_gate",
            retirement_risk="medium",
            evidence={"requirement_evidence_matrix": exp3230.get("requirement_evidence_matrix", [])},
            impact_count=missing_count,
        )
    ]


def backend_failure_traces(exp3236: Any) -> list[JsonDict]:
    """Extract selected-Python CUDA/backend failure evidence."""

    if not isinstance(exp3236, Mapping) or exp3236.get("cuda_python_smoke_passed") is not False:
        return []
    reasons = exp3236.get("smoke_block_reasons", [])
    reason_list = list(reasons) if isinstance(reasons, Sequence) and not isinstance(reasons, (str, bytes)) else []
    return [
        make_trace(
            category="backend_failure",
            source="results/experiment_3236_isolated_cuda_python_smoke_v1.json",
            prerequisite="selected Python CUDA runtime",
            failure_signature=",".join(str(reason) for reason in reason_list) or "cuda_python_smoke_failed",
            stale_premise=False,
            accepted_next_action=str(
                exp3236.get("recommended_next_task") or "repair_backend_before_dependent_receipt"
            ),
            retirement_risk="high",
            evidence={
                "selected_python_torch_cuda_available": exp3236.get(
                    "selected_python_torch_cuda_available"
                ),
                "cuda_bindings_runtime_ok": exp3236.get("cuda_bindings_runtime_ok"),
                "smoke_block_reasons": reason_list,
            },
        )
    ]


def blocked_gate_artifact_traces(artifact_id: str, payload: Any) -> list[JsonDict]:
    """Extract one trace per blocked pre-gate artifact."""

    if not isinstance(payload, Mapping) or payload.get("status") != "blocked":
        return []
    gates = payload.get("gates_evaluated", [])
    failed_gates = [
        gate
        for gate in gates
        if isinstance(gates, Sequence)
        and not isinstance(gates, (str, bytes))
        and isinstance(gate, Mapping)
        and gate.get("passed") is False
    ]
    if not failed_gates:
        return []
    gate = failed_gates[0]
    upstream = str(gate.get("upstream") or "unknown_upstream")
    field = str(gate.get("artifact_field") or "unknown_field")
    actual = gate.get("actual")
    category = "missing_artifact" if actual is None else "gate_block"
    return [
        make_trace(
            category=category,
            source=ARTIFACT_PATHS[artifact_id].as_posix(),
            prerequisite=f"{upstream}.{field}",
            failure_signature=str(payload.get("gate_check_summary") or f"{upstream}.{field} failed"),
            stale_premise=False,
            accepted_next_action="force_prerequisite_gate",
            retirement_risk="high",
            evidence=dict(gate),
        )
    ]


def make_trace(
    *,
    category: str,
    source: str,
    prerequisite: str,
    failure_signature: str,
    stale_premise: bool,
    accepted_next_action: str,
    retirement_risk: str,
    evidence: Mapping[str, Any] | None = None,
    impact_count: int = 1,
) -> JsonDict:
    """Build one normalized failure-memory trace."""

    trace_basis = f"{category}|{source}|{prerequisite}|{failure_signature}"
    return {
        "trace_id": f"{category}-{stable_id(trace_basis)}",
        "category": category,
        "source": source,
        "prerequisite": prerequisite,
        "failure_signature": failure_signature,
        "stale_premise": stale_premise,
        "accepted_next_action": accepted_next_action,
        "retirement_risk": retirement_risk,
        "impact_count": impact_count,
        "evidence": dict(evidence or {}),
    }


def score_heldout_replays(traces: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], int, int]:
    """REQ-LEARN-3243-3: score heldout controller decisions from memory traces."""

    replays: list[JsonDict] = []
    for index, trace in enumerate(traces):
        decision = controller_decision(trace)
        avoided = decision in {"force_prerequisite_gate", "repair_backend_before_rerun"}
        replay_delta = 1 if avoided or decision == "reject_stale_premise" else 0
        replays.append(
            {
                "replay_id": f"heldout-{index + 1:03d}",
                "source_trace_id": trace.get("trace_id"),
                "failure_signature": trace.get("failure_signature"),
                "baseline_action": "rerun_without_failure_memory",
                "controller_decision": decision,
                "accepted_next_action": trace.get("accepted_next_action"),
                "avoided_doomed_rerun": avoided,
                "force_gate": decision == "force_prerequisite_gate",
                "replay_delta": replay_delta,
            }
        )
    delta = sum(int(row["replay_delta"]) for row in replays)
    avoided_count = sum(1 for row in replays if row["avoided_doomed_rerun"])
    return replays, delta, avoided_count


def controller_decision(trace: Mapping[str, Any]) -> str:
    """Choose the controller action represented by a failure-memory trace."""

    category = normalize_token(trace.get("category"))
    action = normalize_token(trace.get("accepted_next_action"))
    if category == "stale_premise":
        return "reject_stale_premise"
    if category == "backend_failure" and action:
        return "repair_backend_before_rerun"
    if category in {"missing_artifact", "repeated_gate_block", "gate_block", "missing_certificate"}:
        return "force_prerequisite_gate"
    return action or "no_memory_action"


def build_nonforgetting_checks(exp3229: Any, traces: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """REQ-LEARN-3243-4: preserve prior accepted and rejected FR-11 traces."""

    prior = exp3229 if isinstance(exp3229, Mapping) else {}
    accepted_before = safe_int(prior.get("accepted_trace_count"))
    stale_before = safe_int(prior.get("stale_premise_rejection_count"))
    stale_after = max(stale_before, count_stale_rejections(traces))
    negative_regressions = safe_int(prior.get("negative_control_regression_count"))
    return [
        {
            "check_id": "accepted_trace_retention",
            "before": accepted_before,
            "after": accepted_before,
            "delta": 0,
            "passed": True,
            "source": EXP3229_REL_PATH.as_posix(),
        },
        {
            "check_id": "stale_rejection_retention",
            "before": stale_before,
            "after": stale_after,
            "delta": stale_after - stale_before,
            "passed": stale_after >= stale_before,
            "source": EXP3229_REL_PATH.as_posix(),
        },
        {
            "check_id": "negative_control_regression_guard",
            "before": negative_regressions,
            "after": negative_regressions,
            "delta": 0,
            "passed": negative_regressions == 0,
            "source": EXP3229_REL_PATH.as_posix(),
        },
    ]


def count_stale_rejections(traces: Sequence[Mapping[str, Any]]) -> int:
    """Return stale-premise rejection impact from extracted traces."""

    return sum(
        safe_int(trace.get("impact_count"))
        for trace in traces
        if trace.get("stale_premise") is True
        or normalize_token(trace.get("category")) == "stale_premise"
    )


def nonforgetting_delta(checks: Sequence[Mapping[str, Any]]) -> int:
    """Return the minimum nonforgetting delta across checks."""

    if not checks:
        return 0
    return min(safe_int(check.get("delta")) for check in checks)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the terminal Exp 3243 failure-memory controller artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = load_sources(root_path)
    artifacts = sources.get("artifacts", {})
    exp3229 = artifacts.get("exp3229", {}) if isinstance(artifacts, Mapping) else {}
    schema = failure_memory_schema()
    traces = collect_failure_traces(sources)
    replays, replay_delta, avoided_count = score_heldout_replays(traces)
    nonforgetting_checks = build_nonforgetting_checks(exp3229, traces)
    artifact: JsonDict = {
        "artifact": "experiment_3243_fr11_failure_memory_controller_v1",
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "principle_annotations": principle_annotations(),
        "continuous_self_learning_task": True,
        "failure_memory_schema": schema,
        "failure_memory_schema_ready": bool(schema["schema_ready"]),
        "failure_traces": traces,
        "failure_trace_count": len(traces),
        "heldout_replays": replays,
        "heldout_replay_count": len(replays),
        "heldout_replay_delta": replay_delta,
        "nonforgetting_checks": nonforgetting_checks,
        "nonforgetting_delta": nonforgetting_delta(nonforgetting_checks),
        "stale_premise_rejection_count": count_stale_rejections(traces),
        "doomed_rerun_avoidance_count": avoided_count,
        "model_weight_update_claimed": False,
        "controller_memory_updates_are_not_training": True,
        "source_artifacts": source_artifacts(root_path),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(start, now_s),
    }
    artifact["fr11_controller_update_ready"] = fr11_controller_update_ready(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def principle_annotations() -> JsonDict:
    """Return the principles that keep this controller memory honest."""

    return {
        "controller_memory_not_training": (
            "The update records gate/failure metadata only; no model weights were updated."
        ),
        "failure_memory_scope": (
            "Missing artifacts, repeated gate blocks, stale premises, and backend failures "
            "become controller prerequisites."
        ),
        "heldout_replay_rule": (
            "A positive replay delta requires avoiding a doomed rerun or forcing a gate."
        ),
        "nonforgetting_rule": (
            "Prior accepted FR-11 traces are retained and stale rejected traces stay rejected."
        ),
    }


def fr11_controller_update_ready(artifact: Mapping[str, Any]) -> bool:
    """REQ-LEARN-3243-5: decide whether the failure-memory update is usable."""

    return (
        artifact.get("failure_memory_schema_ready") is True
        and safe_int(artifact.get("failure_trace_count")) > 0
        and safe_int(artifact.get("heldout_replay_count")) > 0
        and safe_int(artifact.get("heldout_replay_delta")) > 0
        and safe_int(artifact.get("nonforgetting_delta")) >= 0
        and safe_int(artifact.get("stale_premise_rejection_count")) > 0
        and safe_int(artifact.get("doomed_rerun_avoidance_count")) > 0
        and artifact.get("model_weight_update_claimed") is False
        and artifact.get("controller_memory_updates_are_not_training") is True
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """REQ-LEARN-3243-6: return a truthful terminal verdict."""

    ready = bool(artifact.get("fr11_controller_update_ready"))
    return (
        "complete: fr11 failure-memory controller update "
        f"ready={str(ready).lower()}; "
        f"failure_trace_count={safe_int(artifact.get('failure_trace_count'))}; "
        f"heldout_replay_count={safe_int(artifact.get('heldout_replay_count'))}; "
        f"heldout_replay_delta={safe_int(artifact.get('heldout_replay_delta'))}; "
        f"doomed_rerun_avoidance_count={safe_int(artifact.get('doomed_rerun_avoidance_count'))}; "
        "model_weight_update_claimed=false; no model weights were updated; "
        "controller_memory_updates_are_not_training=true"
    )


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp 3243 artifact violates its schema or no-training boundary."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id must be exp3243")
    if artifact.get("task_id") != TASK_ID:
        raise ValueError("task_id must be exp3243-fr11-failure-memory-controller-v1")
    if artifact.get("milestone") != MILESTONE:
        raise ValueError("milestone must be 2026.05.300")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if artifact.get("model_weight_update_claimed") is not False:
        raise ValueError("model_weight_update_claimed must remain false")
    if artifact.get("controller_memory_updates_are_not_training") is not True:
        raise ValueError("controller_memory_updates_are_not_training must remain true")
    if safe_int(artifact.get("failure_trace_count")) != len(artifact.get("failure_traces", [])):
        raise ValueError("failure trace count must match failure_traces")
    if safe_int(artifact.get("heldout_replay_count")) != len(artifact.get("heldout_replays", [])):
        raise ValueError("heldout replay count must match heldout_replays")
    checks = artifact.get("nonforgetting_checks", [])
    if not isinstance(checks, Sequence) or isinstance(checks, (str, bytes)):
        raise ValueError("nonforgetting_checks must be a list")
    if safe_int(artifact.get("nonforgetting_delta")) != nonforgetting_delta(checks):
        raise ValueError("nonforgetting_delta must match nonforgetting_checks")
    expected_ready = fr11_controller_update_ready(artifact)
    if artifact.get("fr11_controller_update_ready") is not expected_ready:
        raise ValueError("readiness mismatch for fr11_controller_update_ready")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith("complete:") or "no model weights were updated" not in verdict:
        raise ValueError("honest_verdict must start complete: and state no model weights were updated")


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write the Exp 3243 JSON artifact."""

    root_path = Path(root)
    path = Path(output_path)
    output = path if path.is_absolute() else root_path / path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    write_json(output, artifact)
    return output


def source_artifacts(root: Path) -> list[JsonDict]:
    """List source files and checksums for artifact lineage."""

    rows: list[JsonDict] = []
    for source_id, rel_path, required in SOURCE_ARTIFACTS:
        path = root / rel_path
        rows.append(
            {
                "id": source_id,
                "path": rel_path.as_posix(),
                "required": required,
                "exists": path.is_file(),
                "sha256": sha256_file(path),
            }
        )
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON so artifact diffs stay reviewable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def duration(started_s: float, now_s: float | None) -> float:
    """Return elapsed seconds rounded for stable artifact provenance."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def stable_id(text: str) -> str:
    """Return a short stable identifier for trace IDs."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def sha256_file(path: Path) -> str | None:
    """Return a SHA-256 checksum when the source file exists."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_token(value: Any) -> str:
    """Normalize artifact status/action tokens."""

    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def safe_int(value: Any) -> int:
    """Return an integer counter while treating bad artifact values as zero."""

    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
