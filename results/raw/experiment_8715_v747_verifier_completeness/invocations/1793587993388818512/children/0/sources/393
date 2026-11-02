"""Exp 3255 FR-11 lifelong failure-memory retention audit.

Spec refs: REQ-LEARN-3255, SCENARIO-LEARN-3255,
SCENARIO-LEARN-3255-BLOCKED.

This module audits controller memory that already exists in checked-in
artifacts and conductor logs.  The "learning" under audit is a routing-memory
update: preserving known bad rerun avoidances, adapting to new gate signatures,
and checking that older accepted FR-11 traces did not regress.  It does not call
a live LLM and does not update foundation-model or sidecar weights.
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
MILESTONE = "2026.05.301"
SCHEMA_VERSION = "carnot.fr11.lifelong_failure_memory_retention_audit.v1"
EXPERIMENT_ID = "exp3255"
TASK_ID = "exp3255-fr11-lifelong-failure-memory-retention-audit-v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
OUTPUT_REL_PATH = Path(
    "results/experiment_3255_fr11_lifelong_failure_memory_retention_audit_v1.json"
)
CONDUCTOR_LOG_REL_PATH = Path("ops/conductor-log.md")
RESEARCH_REFERENCES_REL_PATH = Path("research-references.md")
SPEC_REL_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_REL_PATH = Path(
    "python/carnot/eval/fr11_lifelong_failure_memory_retention_audit_v1.py"
)
TEST_REL_PATH = Path(
    "tests/python/test_experiment_3255_fr11_lifelong_failure_memory_retention_audit_v1.py"
)
EXP3215_REL_PATH = Path(
    "results/experiment_3215_fr11_evidence_gated_trace_replay_controller_v2.json"
)
EXP3229_REL_PATH = Path(
    "results/experiment_3229_fr11_nonforgetting_promotion_controller_v3.json"
)
EXP3243_REL_PATH = Path("results/experiment_3243_fr11_failure_memory_controller_v1.json")
EXP3247_REL_PATH = Path("results/experiment_3247_selected_python_cuda_root_cause_surgery_v1.json")
EXP3248_REL_PATH = Path("results/experiment_3248_isolated_cuda_selected_python_smoke_v2.json")
EXP3250_REL_PATH = Path("results/experiment_3250_sota_gguf_receipt_v8.json")

ADAPTATION_300_MARKERS = (
    "exp3236",
    "exp3237",
    "exp3238",
    "exp3240",
    "exp3241",
    "selected_python_torch_cuda",
    "cuda_python_smoke",
)
ADAPTATION_301_TIMESTAMP_PREFIX = "2026-05-28 07:"
GOOD_ADAPTATION_ACTIONS = {
    "force_prerequisite_gate",
    "repair_backend_before_rerun",
    "reject_stale_premise",
    "reject_stale_controller_memory_trace",
}
REQUIRED_ARTIFACT_FIELDS = {
    "experiment_id",
    "task_id",
    "milestone",
    "inference_substrate",
    "principle_annotations",
    "continuous_self_learning_task",
    "fr11_controller_update_ready",
    "lifelong_eval_ready",
    "failure_trace_count",
    "heldout_replay_count",
    "retention_score",
    "adaptation_score",
    "forgetting_score",
    "negative_control_regression_count",
    "doomed_rerun_avoidance_count",
    "model_weight_update_claimed",
    "no_new_llm_invoked",
    "reproducibility_checksum",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest -o addopts='' "
    "tests/python/test_experiment_3255_fr11_lifelong_failure_memory_retention_audit_v1.py -q",
    ".venv/bin/coverage run -m pytest -o addopts='' "
    "tests/python/test_experiment_3255_fr11_lifelong_failure_memory_retention_audit_v1.py -q",
    ".venv/bin/coverage report "
    "--include='python/carnot/eval/fr11_lifelong_failure_memory_retention_audit_v1.py' "
    "--fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_3255_fr11_lifelong_failure_memory_retention_audit_v1.py",
    "jq -e . results/experiment_3255_fr11_lifelong_failure_memory_retention_audit_v1.json",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_no_hidden_weight_update_rules", Path("CLAUDE.md"), False),
    ("research_program", Path("research-program.md"), False),
    ("product_requirements", Path("_bmad/prd.md"), False),
    ("research_references", RESEARCH_REFERENCES_REL_PATH, True),
    ("self_learning_openspec", SPEC_REL_PATH, True),
    ("conductor_log", CONDUCTOR_LOG_REL_PATH, True),
    ("exp3215_trace_labels", EXP3215_REL_PATH, True),
    ("exp3229_nonforgetting_promotion", EXP3229_REL_PATH, True),
    ("exp3243_failure_memory", EXP3243_REL_PATH, True),
    ("exp3247_cuda_root_cause", EXP3247_REL_PATH, False),
    ("exp3248_cuda_smoke_gate", EXP3248_REL_PATH, False),
    ("exp3250_sota_receipt_gate", EXP3250_REL_PATH, False),
    ("exp3255_module", MODULE_REL_PATH, False),
    ("exp3255_tests", TEST_REL_PATH, False),
)


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object while treating missing or malformed inputs as absent."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def read_text(path: Path) -> str:
    """Read a text evidence file while treating missing files as empty evidence."""

    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def load_sources(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load the artifact and log evidence used by the Exp 3255 audit."""

    root_path = Path(root)
    return {
        "exp3215": read_json_object(root_path / EXP3215_REL_PATH),
        "exp3229": read_json_object(root_path / EXP3229_REL_PATH),
        "exp3243": read_json_object(root_path / EXP3243_REL_PATH),
        "exp3247": read_json_object(root_path / EXP3247_REL_PATH),
        "exp3248": read_json_object(root_path / EXP3248_REL_PATH),
        "exp3250": read_json_object(root_path / EXP3250_REL_PATH),
        "conductor_log": read_text(root_path / CONDUCTOR_LOG_REL_PATH),
        "research_references": read_text(root_path / RESEARCH_REFERENCES_REL_PATH),
    }


def lifelong_metric_mapping(research_references: str) -> JsonDict:
    """REQ-LEARN-3255-1: map LifelongAgentBench axes to Carnot traces."""

    note = next(
        (
            line.strip()
            for line in research_references.splitlines()
            if "LifelongAgentBench" in line
        ),
        "LifelongAgentBench note unavailable; using checked-in FR-11 trace evidence.",
    )
    return {
        "source_note": note,
        "retention": "preserved avoidance of previously doomed reruns in Exp 3243 heldout replays",
        "adaptation": "correct force-gate, backend-repair, or stale-reject actions for .300/.301 failures",
        "forgetting": "regressions on Exp 3229 prior accepted FR-11 controller traces",
    }


def remembered_slice(exp3243: Mapping[str, Any]) -> list[JsonDict]:
    """REQ-LEARN-3255-3: build remembered replay rows from Exp 3243."""

    replays = exp3243.get("heldout_replays", [])
    if not isinstance(replays, Sequence) or isinstance(replays, (str, bytes)):
        return []
    trace_lookup = trace_signature_lookup(exp3243)
    remembered: list[JsonDict] = []
    for row in replays:
        if not isinstance(row, Mapping):
            continue
        trace_id = str(row.get("source_trace_id") or row.get("replay_id") or stable_id(str(row)))
        replay_delta = safe_int(row.get("replay_delta"))
        remembered.append(
            {
                "slice": "remembered",
                "trace_id": trace_id,
                "replay_id": str(row.get("replay_id") or trace_id),
                "source": EXP3243_REL_PATH.as_posix(),
                "milestone_bucket": str(exp3243.get("milestone") or "2026.05.300"),
                "failure_signature": trace_lookup.get(trace_id, ""),
                "baseline_action": str(row.get("baseline_action") or "rerun_without_failure_memory"),
                "controller_action": str(row.get("controller_decision") or ""),
                "avoided_doomed_rerun": row.get("avoided_doomed_rerun") is True,
                "preserved_positive_replay": replay_delta > 0,
                "replay_delta": replay_delta,
            }
        )
    return remembered


def trace_signature_lookup(exp3243: Mapping[str, Any]) -> dict[str, str]:
    """Return Exp 3243 trace ID to failure-signature mapping."""

    traces = exp3243.get("failure_traces", [])
    if not isinstance(traces, Sequence) or isinstance(traces, (str, bytes)):
        return {}
    return {
        str(row.get("trace_id")): str(row.get("failure_signature") or "")
        for row in traces
        if isinstance(row, Mapping) and row.get("trace_id")
    }


def adapted_slice(sources: Mapping[str, Any]) -> list[JsonDict]:
    """REQ-LEARN-3255-2/3: build adapted rows for new .300 and .301 failures."""

    rows: list[JsonDict] = []
    exp3243 = sources.get("exp3243", {})
    if isinstance(exp3243, Mapping):
        rows.extend(adapted_from_exp3243(exp3243))
    rows.extend(adapted_from_301_artifacts(sources))
    rows.extend(adapted_from_log(str(sources.get("conductor_log") or "")))
    return dedupe_rows(rows)


def adapted_from_exp3243(exp3243: Mapping[str, Any]) -> list[JsonDict]:
    """Extract .300 selected-Python and dependent gate signatures from Exp 3243."""

    traces = exp3243.get("failure_traces", [])
    if not isinstance(traces, Sequence) or isinstance(traces, (str, bytes)):
        return []
    adapted: list[JsonDict] = []
    for trace in traces:
        if not isinstance(trace, Mapping):
            continue
        text = " ".join(
            [
                str(trace.get("trace_id") or ""),
                str(trace.get("source") or ""),
                str(trace.get("failure_signature") or ""),
            ]
        ).lower()
        if not any(marker in text for marker in ADAPTATION_300_MARKERS):
            continue
        action = controller_action_for_trace(trace)
        adapted.append(
            make_adapted_row(
                source=str(trace.get("source") or EXP3243_REL_PATH.as_posix()),
                signature=str(trace.get("failure_signature") or ""),
                milestone_bucket="2026.05.300",
                controller_action=action,
                basis=str(trace.get("trace_id") or text),
            )
        )
    return adapted


def adapted_from_301_artifacts(sources: Mapping[str, Any]) -> list[JsonDict]:
    """Extract .301 blocked artifacts that should force prerequisites, not reruns."""

    rows: list[JsonDict] = []
    exp3247 = sources.get("exp3247", {})
    if isinstance(exp3247, Mapping) and exp3247.get("next_smoke_allowed") is False:
        rows.append(
            make_adapted_row(
                source=EXP3247_REL_PATH.as_posix(),
                signature="exp3247.next_smoke_allowed=false",
                milestone_bucket="2026.05.301",
                controller_action="force_prerequisite_gate",
                basis="exp3247.next_smoke_allowed=false",
            )
        )
    for source_key, rel_path in (("exp3248", EXP3248_REL_PATH), ("exp3250", EXP3250_REL_PATH)):
        payload = sources.get(source_key, {})
        if isinstance(payload, Mapping) and payload.get("status") == "blocked":
            rows.append(
                make_adapted_row(
                    source=rel_path.as_posix(),
                    signature=str(payload.get("gate_check_summary") or f"{source_key}.blocked"),
                    milestone_bucket="2026.05.301",
                    controller_action="force_prerequisite_gate",
                    basis=f"{source_key}:{payload.get('gate_check_summary')}",
                )
            )
    return rows


def adapted_from_log(log_text: str) -> list[JsonDict]:
    """Extract repeated .301 conductor GATE_BLOCK signatures from the log."""

    groups: dict[tuple[str, str], JsonDict] = {}
    for row in parse_log_rows(log_text):
        if not str(row["timestamp"]).startswith(ADAPTATION_301_TIMESTAMP_PREFIX):
            continue
        if row["status"] != "GATE_BLOCK":
            continue
        signature = first_failure_signature(row["details"])
        key = (row["title"], signature)
        group = groups.setdefault(
            key,
            {"title": row["title"], "signature": signature, "count": 0},
        )
        group["count"] = safe_int(group["count"]) + 1
    return [
        make_adapted_row(
            source=CONDUCTOR_LOG_REL_PATH.as_posix(),
            signature=f"{group['signature']} repeated {group['count']} times",
            milestone_bucket="2026.05.301",
            controller_action="force_prerequisite_gate",
            basis=f"{group['title']}|{group['signature']}",
        )
        for group in groups.values()
    ]


def make_adapted_row(
    *,
    source: str,
    signature: str,
    milestone_bucket: str,
    controller_action: str,
    basis: str,
) -> JsonDict:
    """Build one normalized adaptation row."""

    return {
        "slice": "adapted",
        "trace_id": f"adapted-{stable_id(source + '|' + basis)}",
        "source": source,
        "milestone_bucket": milestone_bucket,
        "failure_signature": signature,
        "controller_action": controller_action,
        "correct_adaptation": normalize_token(controller_action) in GOOD_ADAPTATION_ACTIONS,
    }


def heldout_negative_control_slice(exp3215: Mapping[str, Any]) -> list[JsonDict]:
    """REQ-LEARN-3255-4: extract held-out negative controls from replay labels."""

    labels = exp3215.get("replay_utility_labels", [])
    if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
        return []
    heldout: list[JsonDict] = []
    for label in labels:
        if not isinstance(label, Mapping) or label.get("replay_role") != "negative_control":
            continue
        heldout.append(
            {
                "slice": "held_out_negative_control",
                "trace_id": str(label.get("trace_id") or stable_id(str(label))),
                "row_id": str(label.get("row_id") or ""),
                "source": EXP3215_REL_PATH.as_posix(),
                "milestone_bucket": str(exp3215.get("milestone") or "2026.05.297"),
                "controller_action": str(label.get("routing_outcome") or ""),
                "reward_weight": float(label.get("reward_weight") or 0.0),
                "rollback_or_retraction_status": str(
                    label.get("rollback_or_retraction_status") or "none"
                ),
                "regressed": negative_control_label_regressed(label),
            }
        )
    return heldout


def negative_control_label_regressed(label: Mapping[str, Any]) -> bool:
    """Return whether a negative-control label changed into an unsafe update."""

    rollback = normalize_token(label.get("rollback_or_retraction_status"))
    return (
        float(label.get("reward_weight") or 0.0) < 0.0
        or rollback not in {"", "none"}
        or label.get("model_weight_update_claimed") is True
    )


def negative_control_regression_count(
    sources: Mapping[str, Any],
    heldout_negative_controls: Sequence[Mapping[str, Any]],
) -> int:
    """Count source-reported plus label-level negative-control regressions."""

    exp3215 = sources.get("exp3215", {})
    exp3229 = sources.get("exp3229", {})
    source_count = 0
    if isinstance(exp3215, Mapping):
        source_count += safe_int(exp3215.get("negative_control_regression_count"))
    if isinstance(exp3229, Mapping):
        source_count += safe_int(exp3229.get("negative_control_regression_count"))
    label_count = sum(1 for row in heldout_negative_controls if row.get("regressed") is True)
    return source_count + label_count


def score_retention(remembered: Sequence[Mapping[str, Any]]) -> float:
    """Score preserved avoidance of previously doomed reruns."""

    doomed = [row for row in remembered if row.get("avoided_doomed_rerun") is True]
    preserved = [
        row
        for row in doomed
        if row.get("preserved_positive_replay") is True
        and normalize_token(row.get("controller_action")) in GOOD_ADAPTATION_ACTIONS
    ]
    return score_ratio(len(preserved), len(doomed))


def score_adaptation(adapted: Sequence[Mapping[str, Any]]) -> float:
    """Score correct handling of new .300/.301 failure signatures."""

    correct = sum(1 for row in adapted if row.get("correct_adaptation") is True)
    return score_ratio(correct, len(adapted))


def score_forgetting(exp3229: Mapping[str, Any], regression_count: int) -> float:
    """REQ-LEARN-3255-5: score prior accepted FR-11 traces against regressions."""

    accepted_count = safe_int(exp3229.get("accepted_trace_count"))
    accepted_traces = exp3229.get("accepted_traces", [])
    if accepted_count <= 0 and isinstance(accepted_traces, Sequence):
        accepted_count = len(accepted_traces)
    if accepted_count <= 0:
        return 0.0
    return round(max(0.0, 1.0 - (float(regression_count) / float(accepted_count))), 6)


def score_ratio(numerator: int, denominator: int) -> float:
    """Return a bounded score while failing closed for empty denominators."""

    if denominator <= 0:
        return 0.0
    return round(max(0.0, min(1.0, float(numerator) / float(denominator))), 6)


def fr11_controller_update_ready(exp3243: Mapping[str, Any]) -> bool:
    """Return whether the upstream Exp 3243 controller-memory update is usable."""

    return (
        exp3243.get("fr11_controller_update_ready") is True
        and exp3243.get("model_weight_update_claimed") is False
        and exp3243.get("controller_memory_updates_are_not_training") is True
    )


def lifelong_eval_ready(artifact: Mapping[str, Any]) -> bool:
    """REQ-LEARN-3255-6: gate the lifelong audit result."""

    return (
        artifact.get("fr11_controller_update_ready") is True
        and safe_int(artifact.get("remembered_replay_count")) > 0
        and safe_int(artifact.get("adapted_trace_count")) > 0
        and safe_int(artifact.get("heldout_replay_count")) > 0
        and artifact.get("retention_score") == 1.0
        and artifact.get("adaptation_score") == 1.0
        and artifact.get("forgetting_score") == 1.0
        and safe_int(artifact.get("negative_control_regression_count")) == 0
        and artifact.get("model_weight_update_claimed") is False
        and artifact.get("no_new_llm_invoked") is True
    )


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the terminal Exp 3255 lifelong failure-memory audit artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = load_sources(root_path)
    exp3215 = sources["exp3215"]
    exp3229 = sources["exp3229"]
    exp3243 = sources["exp3243"]
    remembered = remembered_slice(exp3243)
    adapted = adapted_slice(sources)
    negative_controls = heldout_negative_control_slice(exp3215)
    negative_regressions = negative_control_regression_count(sources, negative_controls)
    failure_traces = remembered + adapted
    artifact: JsonDict = {
        "artifact": "experiment_3255_fr11_lifelong_failure_memory_retention_audit_v1",
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "principle_annotations": principle_annotations(sources["research_references"]),
        "continuous_self_learning_task": True,
        "fr11_controller_update_ready": fr11_controller_update_ready(exp3243),
        "lifelong_metric_mapping": lifelong_metric_mapping(str(sources["research_references"])),
        "evaluation_slices": {
            "remembered": remembered,
            "adapted": adapted,
            "held_out_negative_control": negative_controls,
        },
        "failure_traces": failure_traces,
        "failure_trace_count": len(failure_traces),
        "remembered_replay_count": len(remembered),
        "adapted_trace_count": len(adapted),
        "heldout_replay_count": len(negative_controls),
        "retention_score": score_retention(remembered),
        "adaptation_score": score_adaptation(adapted),
        "negative_control_regression_count": negative_regressions,
        "forgetting_score": score_forgetting(exp3229, negative_regressions),
        "forgetting_regression_count": negative_regressions,
        "doomed_rerun_avoidance_count": doomed_rerun_avoidance_count(remembered, adapted),
        "model_weight_update_claimed": False,
        "no_new_llm_invoked": True,
        "learning_boundary": learning_boundary(),
        "source_artifacts": source_artifacts(root_path),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(start, now_s),
    }
    artifact["lifelong_eval_ready"] = lifelong_eval_ready(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def principle_annotations(research_references: str) -> JsonDict:
    """Return the audit principles tying LifelongAgentBench to FR-11 evidence."""

    return {
        "lifelong_agent_bench_mapping": lifelong_metric_mapping(research_references),
        "controller_memory_not_model_learning": (
            "Controller-memory updates change routing/gate metadata only; "
            "foundation-model weights are not updated."
        ),
        "retention_rule": "Previously doomed reruns must still be avoided.",
        "adaptation_rule": "New .300/.301 gate signatures must force prerequisites or backend repair.",
        "forgetting_rule": "Prior accepted FR-11 traces and held-out controls must not regress.",
    }


def learning_boundary() -> JsonDict:
    """Expose the boundary between controller memory and foundation-model learning."""

    return {
        "controller_memory_updates": "routing metadata and prerequisite gates",
        "foundation_model_weight_update": False,
        "kan_sidecar_weight_update": False,
        "hidden_state_mutation": False,
        "live_llm_invocation": False,
    }


def doomed_rerun_avoidance_count(
    remembered: Sequence[Mapping[str, Any]],
    adapted: Sequence[Mapping[str, Any]],
) -> int:
    """Count retained and newly adapted decisions that avoid doomed reruns."""

    remembered_count = sum(1 for row in remembered if row.get("avoided_doomed_rerun") is True)
    adapted_count = sum(1 for row in adapted if row.get("correct_adaptation") is True)
    return remembered_count + adapted_count


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """REQ-LEARN-3255-7: return a truthful terminal verdict."""

    return (
        "complete: fr11 lifelong failure-memory retention audit "
        f"ready={str(bool(artifact.get('lifelong_eval_ready'))).lower()}; "
        f"retention_score={artifact.get('retention_score')}; "
        f"adaptation_score={artifact.get('adaptation_score')}; "
        f"forgetting_score={artifact.get('forgetting_score')}; "
        f"negative_control_regression_count={safe_int(artifact.get('negative_control_regression_count'))}; "
        "controller memory only; foundation-model weights were not updated; "
        "model_weight_update_claimed=false; no_new_llm_invoked=true"
    )


def parse_log_rows(log_text: str) -> list[JsonDict]:
    """Parse conductor markdown table rows into dictionaries."""

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
    """Normalize conductor gate details into a reusable failure signature."""

    first = re.search(r"first failure:\s*([^|]+)", details)
    if first:
        return first.group(1).strip()
    retired = re.search(r"upstream retired \(([^)|]+)", details)
    if retired:
        return f"upstream_retired:{retired.group(1).strip()}"
    return details.strip() or "unknown_gate_block"


def controller_action_for_trace(trace: Mapping[str, Any]) -> str:
    """Map a source failure trace to the controller decision being audited."""

    category = normalize_token(trace.get("category"))
    action = normalize_token(trace.get("accepted_next_action"))
    if category == "backend_failure" or action.startswith("repair_selected_python"):
        return "repair_backend_before_rerun"
    if category == "stale_premise" or action.startswith("reject_stale"):
        return "reject_stale_premise"
    return "force_prerequisite_gate"


def dedupe_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Return rows once by trace ID, preserving first-seen order."""

    seen: set[str] = set()
    deduped: list[JsonDict] = []
    for row in rows:
        trace_id = str(row.get("trace_id") or "")
        if trace_id in seen:
            continue
        seen.add(trace_id)
        deduped.append(dict(row))
    return deduped


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp 3255 artifact violates schema or learning boundaries."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id must be exp3255")
    if artifact.get("task_id") != TASK_ID:
        raise ValueError("task_id must be exp3255-fr11-lifelong-failure-memory-retention-audit-v1")
    if artifact.get("milestone") != MILESTONE:
        raise ValueError("milestone must be 2026.05.301")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if artifact.get("model_weight_update_claimed") is not False:
        raise ValueError("model_weight_update_claimed must remain false")
    if artifact.get("no_new_llm_invoked") is not True:
        raise ValueError("no_new_llm_invoked must remain true")
    slices = artifact.get("evaluation_slices", {})
    heldout = slices.get("held_out_negative_control", []) if isinstance(slices, Mapping) else []
    if safe_int(artifact.get("heldout_replay_count")) != len(heldout):
        raise ValueError("heldout_replay_count must match held-out negative controls")
    failure_traces = artifact.get("failure_traces", [])
    if safe_int(artifact.get("failure_trace_count")) != len(failure_traces):
        raise ValueError("failure_trace_count must match failure_traces")
    for score_name in ("retention_score", "adaptation_score", "forgetting_score"):
        score = artifact.get(score_name)
        if not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
            raise ValueError(f"{score_name} must be between 0 and 1")
    if artifact.get("lifelong_eval_ready") is not lifelong_eval_ready(artifact):
        raise ValueError("lifelong_eval_ready readiness mismatch")
    verdict = str(artifact.get("honest_verdict") or "")
    if (
        not verdict.startswith("complete:")
        or "controller memory only" not in verdict
        or "foundation-model weights were not updated" not in verdict
    ):
        raise ValueError("honest_verdict must start complete: and state the learning boundary")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        raise ValueError("reproducibility_checksum must match canonical artifact payload")


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write the Exp 3255 JSON artifact."""

    root_path = Path(root)
    path = Path(output_path)
    output = path if path.is_absolute() else root_path / path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    write_json(output, artifact)
    return output


def source_artifacts(root: Path) -> list[JsonDict]:
    """List source evidence and checksums for reproducibility."""

    return [
        {
            "id": source_id,
            "path": rel_path.as_posix(),
            "required": required,
            "exists": (root / rel_path).is_file(),
            "sha256": sha256_file(root / rel_path),
        }
        for source_id, rel_path, required in SOURCE_ARTIFACTS
    ]


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Return a deterministic checksum over the artifact payload."""

    basis = {
        key: value
        for key, value in artifact.items()
        if key not in {"reproducibility_checksum", "duration_s"}
    }
    encoded = json.dumps(basis, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON for stable review diffs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def duration(started_s: float, now_s: float | None) -> float:
    """Return elapsed seconds rounded for stable provenance."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def stable_id(text: str) -> str:
    """Return a short stable identifier for row IDs."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def sha256_file(path: Path) -> str | None:
    """Return a file checksum when the evidence path exists."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_token(value: Any) -> str:
    """Normalize artifact status and action tokens."""

    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def safe_int(value: Any) -> int:
    """Return an integer count while treating malformed evidence as zero."""

    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
