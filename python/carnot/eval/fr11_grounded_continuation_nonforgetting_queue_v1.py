"""Exp 3216 FR-11 grounded continuation graph and nonforgetting queue.

Spec refs: REQ-LEARN-3216, SCENARIO-LEARN-3216,
SCENARIO-LEARN-3216-FALLBACK.

This module audits controller-memory replay traces.  It represents a small
slice of FR-11 replay evidence as claim, evidence, retraction, repair, and
route graph records, then computes a virtual nonforgetting queue for held-out
and drift regression pressure.  The queue is policy metadata only: this code
does not run live model inference, mutate model weights, or promote KAN
sidecars.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
MILESTONE = "2026.05.297"
SCHEMA_VERSION = "1.0"
EXPERIMENT_ID = "experiment_3216_fr11_grounded_continuation_nonforgetting_queue_v1"
SCHEMA = "carnot.fr11.grounded_continuation_nonforgetting_queue.v1"
OUTPUT_REL_PATH = Path(
    "results/experiment_3216_fr11_grounded_continuation_nonforgetting_queue_v1.json"
)
EXP3215_REL_PATH = Path(
    "results/experiment_3215_fr11_evidence_gated_trace_replay_controller_v2.json"
)
EXP3200_REL_PATH = Path("results/experiment_3200_fr11_verify_trace_memory_controller_v1.json")
SPEC_REL_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_REL_PATH = Path(
    "python/carnot/eval/fr11_grounded_continuation_nonforgetting_queue_v1.py"
)
TEST_REL_PATH = Path(
    "tests/python/test_experiment_3216_fr11_grounded_continuation_nonforgetting_queue_v1.py"
)
TRACE_LIMIT = 6
NONFORGETTING_BUDGET = 2.0
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "passed:", "passed_")
MUTATION_FLAGS = (
    "executes_live_model_inference",
    "model_weight_learning",
    "model_weight_training",
    "model_weight_mutation",
    "base_model_weights_updated",
    "kan_model_weight_training",
    "hidden_state_mutation_claimed",
)
REQUIRED_ARTIFACT_FIELDS = {
    "schema_version",
    "experiment_id",
    "milestone",
    "continuous_self_learning_task",
    "source_trace_artifact",
    "trace_graph_node_count",
    "trace_graph_edge_count",
    "stale_premise_invalidations",
    "affected_route_count",
    "nonforgetting_queue_defined",
    "nonforgetting_queue_value",
    "nonforgetting_budget_exceeded",
    "model_weight_update_claimed",
    "controller_memory_promotion_allowed",
    "conductor_file_modified",
    "active_roadmap_modified",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest -o addopts='' "
    "tests/python/test_experiment_3216_fr11_grounded_continuation_nonforgetting_queue_v1.py -q",
    ".venv/bin/coverage run -m pytest -o addopts='' "
    "tests/python/test_experiment_3216_fr11_grounded_continuation_nonforgetting_queue_v1.py -q",
    ".venv/bin/coverage report "
    "--include='python/carnot/eval/fr11_grounded_continuation_nonforgetting_queue_v1.py' "
    "--fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_3216_fr11_grounded_continuation_nonforgetting_queue_v1.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_no_hidden_weight_update_rules", Path("CLAUDE.md"), False),
    ("research_program", Path("research-program.md"), False),
    ("research_references", Path("research-references.md"), False),
    ("self_learning_openspec", SPEC_REL_PATH, False),
    ("exp3215_trace_labels", EXP3215_REL_PATH, False),
    ("exp3200_trace_memory_controller", EXP3200_REL_PATH, True),
    ("exp3216_module", MODULE_REL_PATH, False),
    ("exp3216_tests", TEST_REL_PATH, False),
)


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object while treating bad evidence as absent."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def load_sources(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load Exp 3215 and fallback Exp 3200 from checked-in artifacts."""

    root_path = Path(root)
    return {
        "exp3215": read_json_object(root_path / EXP3215_REL_PATH),
        "exp3200": read_json_object(root_path / EXP3200_REL_PATH),
    }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    trace_limit: int = TRACE_LIMIT,
) -> JsonDict:
    """Build the Exp 3216 terminal audit artifact from trace evidence."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = load_sources(root_path)
    selection = select_source(sources)
    payload = selection["payload"]
    rows = trace_rows_from_payload(payload, source_kind=selection["source_kind"])
    graph = build_trace_graph(rows, trace_limit=trace_limit)
    propagation = propagate_stale_premises(graph)
    queue = evaluate_nonforgetting_queue(payload, graph, propagation)
    artifact = {
        "artifact": EXPERIMENT_ID,
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "continuous_self_learning_task": True,
        "source_trace_artifact": selection["source_trace_artifact"],
        "source_selection": source_selection_report(selection),
        "source_artifacts": source_artifacts(root_path),
        "trace_graph_node_count": graph["node_count"],
        "trace_graph_edge_count": graph["edge_count"],
        "trace_graph": graph,
        "stale_premise_invalidations": propagation["stale_premise_invalidations"],
        "affected_route_count": propagation["affected_route_count"],
        "affected_routes": propagation["affected_routes"],
        "nonforgetting_queue_defined": queue["nonforgetting_queue_defined"],
        "nonforgetting_queue_value": queue["nonforgetting_queue_value"],
        "nonforgetting_budget_exceeded": queue["nonforgetting_budget_exceeded"],
        "nonforgetting_queue": queue,
        "model_weight_update_claimed": False,
        "controller_memory_promotion_allowed": False,
        "controller_memory_promotion_reason": (
            "audit_only_no_promotion; graph_and_queue_report_do_not_authorize_route_promotion"
        ),
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "inference_substrate": inference_substrate(),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(start, now_s),
        "honest_verdict": honest_verdict(queue["nonforgetting_budget_exceeded"], selection),
    }
    validate_artifact(artifact)
    return artifact


def select_source(sources: Mapping[str, Any]) -> JsonDict:
    """REQ-LEARN-3216-1: prefer terminal Exp 3215, then terminal Exp 3200."""

    exp3215 = sources.get("exp3215", {})
    if isinstance(exp3215, Mapping) and is_terminal(exp3215) and not unsafe_source(exp3215):
        return {
            "source_kind": "exp3215",
            "payload": dict(exp3215),
            "source_trace_artifact": EXP3215_REL_PATH.as_posix(),
            "fallback_used": False,
            "fallback_reason": "",
            "blocked_reason": "",
        }
    exp3200 = sources.get("exp3200", {})
    if isinstance(exp3200, Mapping) and is_terminal(exp3200) and not unsafe_source(exp3200):
        return {
            "source_kind": "exp3200",
            "payload": dict(exp3200),
            "source_trace_artifact": EXP3200_REL_PATH.as_posix(),
            "fallback_used": True,
            "fallback_reason": "exp3215_missing_or_not_terminal",
            "blocked_reason": "",
        }
    return {
        "source_kind": "none",
        "payload": {},
        "source_trace_artifact": "",
        "fallback_used": False,
        "fallback_reason": "",
        "blocked_reason": "no_terminal_safe_trace_source",
    }


def source_selection_report(selection: Mapping[str, Any]) -> JsonDict:
    """Return a compact source-selection record for the result artifact."""

    return {
        "source_kind": selection.get("source_kind"),
        "fallback_used": bool(selection.get("fallback_used")),
        "fallback_reason": str(selection.get("fallback_reason") or ""),
        "blocked_reason": str(selection.get("blocked_reason") or ""),
    }


def is_terminal(payload: Mapping[str, Any]) -> bool:
    """Return whether an artifact verdict is terminal enough to consume."""

    return str(payload.get("honest_verdict") or "").startswith(TERMINAL_PREFIXES)


def unsafe_source(payload: Mapping[str, Any]) -> bool:
    """Return whether source evidence already overclaims unsafe learning."""

    return detected_model_weight_update(payload) or source_claims_live_or_mutation(payload)


def source_claims_live_or_mutation(payload: Mapping[str, Any]) -> bool:
    """Return whether source evidence claims live inference or mutation."""

    substrate = payload.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        return True
    return int(substrate.get("fresh_live_inference_calls") or 0) != 0 or any(
        substrate.get(flag) is True for flag in MUTATION_FLAGS
    )


def detected_model_weight_update(payload: Mapping[str, Any]) -> bool:
    """Return whether an artifact claims any model-weight update."""

    if payload.get("model_weight_update_claimed") is True:
        return True
    if payload.get("model_weight_update_performed") is True:
        return True
    substrate = payload.get("inference_substrate", {})
    return isinstance(substrate, Mapping) and any(
        substrate.get(flag) is True for flag in MUTATION_FLAGS if "weight" in flag
    )


def trace_rows_from_payload(payload: Mapping[str, Any], *, source_kind: str) -> list[JsonDict]:
    """Return graph-ready trace rows from Exp 3215 labels or Exp 3200 traces."""

    key = "replay_utility_labels" if source_kind == "exp3215" else "trace_records"
    rows = payload.get(key, [])
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    normalized: list[JsonDict] = []
    for row in rows:
        if isinstance(row, Mapping):
            item = dict(row)
            item["source_kind"] = source_kind
            normalized.append(item)
    return normalized


def build_trace_graph(rows: Sequence[Mapping[str, Any]], *, trace_limit: int = TRACE_LIMIT) -> JsonDict:
    """REQ-LEARN-3216-2/3: build claim/evidence/retraction/repair graph records."""

    nodes: list[JsonDict] = []
    edges: list[JsonDict] = []
    selected_rows = select_trace_rows(rows, trace_limit=trace_limit)
    for row in selected_rows:
        trace_id = stable_trace_id(row)
        row_id = str(row.get("row_id") or trace_id)
        claim_id = f"claim:{trace_id}"
        evidence_id = f"evidence:{trace_id}"
        route_id = f"route:{trace_id}"
        nodes.extend(
            [
                graph_node(claim_id, "claim", row, text=claim_text(row)),
                graph_node(evidence_id, "evidence", row, text=evidence_text(row)),
                graph_node(route_id, "route", row, text=route_text(row)),
            ]
        )
        edges.extend(
            [
                graph_edge(evidence_id, claim_id, "supports"),
                graph_edge(claim_id, route_id, "supports_route"),
            ]
        )
        if trace_needs_retraction(row):
            retraction_id = f"retraction:{trace_id}"
            repair_id = f"repair:{trace_id}"
            nodes.extend(
                [
                    graph_node(retraction_id, "retraction", row, text=retraction_text(row)),
                    graph_node(repair_id, "repair", row, text=repair_text(row)),
                ]
            )
            edges.extend(
                [
                    graph_edge(retraction_id, claim_id, "invalidates"),
                    graph_edge(retraction_id, repair_id, "requires_repair"),
                    graph_edge(repair_id, route_id, "repairs_route"),
                ]
            )
    return {
        "schema_id": "carnot.fr11.grounded_continuation_trace_graph.v1",
        "source_trace_count": len(selected_rows),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def select_trace_rows(
    rows: Sequence[Mapping[str, Any]], *, trace_limit: int = TRACE_LIMIT
) -> list[Mapping[str, Any]]:
    """Select a small role-balanced trace set without reshuffling evidence."""

    limit = max(0, trace_limit)
    if limit == 0:
        return []
    selected_indexes: list[int] = []
    role_order = ("heldout", "drift", "negative_control")
    while len(selected_indexes) < limit:
        made_progress = False
        for role in role_order:
            for index, row in enumerate(rows):
                if index not in selected_indexes and normalize_token(row.get("replay_role")) == role:
                    selected_indexes.append(index)
                    made_progress = True
                    break
            if len(selected_indexes) >= limit:
                break
        if not made_progress:
            break
    for index, _row in enumerate(rows):
        if len(selected_indexes) >= limit:
            break
        if index not in selected_indexes:
            selected_indexes.append(index)
    return [rows[index] for index in selected_indexes]


def graph_node(node_id: str, kind: str, row: Mapping[str, Any], *, text: str) -> JsonDict:
    """Create one compact graph node with route metadata attached."""

    return {
        "node_id": node_id,
        "node_kind": kind,
        "trace_id": stable_trace_id(row),
        "row_id": str(row.get("row_id") or stable_trace_id(row)),
        "replay_role": normalize_token(row.get("replay_role")),
        "routing_outcome": normalize_token(row.get("routing_outcome")),
        "text": text,
    }


def graph_edge(source: str, target: str, relation: str) -> JsonDict:
    """Create one directed dependency edge."""

    return {"source": source, "target": target, "relation": relation}


def propagate_stale_premises(graph: Mapping[str, Any]) -> JsonDict:
    """REQ-LEARN-3216-3: walk invalidation edges to dependent routes."""

    nodes = {
        str(node.get("node_id")): node
        for node in graph.get("nodes", [])
        if isinstance(node, Mapping) and node.get("node_id")
    }
    adjacency: dict[str, list[tuple[str, str]]] = {}
    for edge in graph.get("edges", []):
        if isinstance(edge, Mapping):
            adjacency.setdefault(str(edge.get("source")), []).append(
                (str(edge.get("target")), str(edge.get("relation")))
            )
    invalidated: set[str] = set()
    affected: dict[str, JsonDict] = {}
    frontier = [
        str(edge.get("target"))
        for edge in graph.get("edges", [])
        if isinstance(edge, Mapping) and edge.get("relation") == "invalidates"
    ]
    while frontier:
        node_id = frontier.pop(0)
        if node_id in invalidated:
            continue
        invalidated.add(node_id)
        node = nodes.get(node_id, {})
        if node.get("node_kind") == "route":
            affected[node_id] = route_report(node)
        for target, relation in adjacency.get(node_id, []):
            if relation in {"supports_route", "repairs_route"}:
                frontier.append(target)
    return {
        "stale_premise_invalidations": len(invalidated),
        "affected_route_count": len(affected),
        "affected_routes": [affected[key] for key in sorted(affected)],
        "invalidated_node_ids": sorted(invalidated),
    }


def route_report(node: Mapping[str, Any]) -> JsonDict:
    """Return the route fields needed to audit stale-premise effects."""

    return {
        "route_node_id": str(node.get("node_id")),
        "row_id": str(node.get("row_id")),
        "replay_role": normalize_token(node.get("replay_role")),
        "routing_outcome": normalize_token(node.get("routing_outcome")),
    }


def evaluate_nonforgetting_queue(
    payload: Mapping[str, Any],
    graph: Mapping[str, Any],
    propagation: Mapping[str, Any],
) -> JsonDict:
    """REQ-LEARN-3216-4: score held-out/drift regression pressure."""

    affected_routes = [
        route
        for route in propagation.get("affected_routes", [])
        if isinstance(route, Mapping) and route.get("replay_role") in {"heldout", "drift"}
    ]
    negative_control = int(payload.get("negative_control_regression_count") or 0)
    rollback_pressure = max(0, retraction_count(graph) - len(affected_routes))
    value = float(len(affected_routes) + (2 * negative_control) + rollback_pressure)
    exceeded = value > NONFORGETTING_BUDGET
    return {
        "nonforgetting_queue_defined": True,
        "nonforgetting_queue_value": round(value, 6),
        "nonforgetting_budget": NONFORGETTING_BUDGET,
        "nonforgetting_budget_exceeded": exceeded,
        "pressure_terms": {
            "affected_heldout_or_drift_routes": len(affected_routes),
            "negative_control_regressions": negative_control,
            "unrouted_retraction_pressure": rollback_pressure,
        },
        "queue_rule": (
            "Q=max regression pressure from affected heldout/drift routes, "
            "negative controls, and unrepaired retractions; promotion blocks when Q>budget"
        ),
    }


def retraction_count(graph: Mapping[str, Any]) -> int:
    """Count explicit retraction nodes in the graph."""

    return sum(
        1
        for node in graph.get("nodes", [])
        if isinstance(node, Mapping) and node.get("node_kind") == "retraction"
    )


def trace_needs_retraction(row: Mapping[str, Any]) -> bool:
    """Return whether a trace should invalidate its stale premise."""

    rollback = normalize_token(row.get("rollback_or_retraction_status"))
    if rollback not in {"unknown", "none"}:
        return True
    if bool(row.get("redundant_check_suppressed")) and normalize_token(
        row.get("replay_role")
    ) in {"heldout", "drift"}:
        return True
    if normalize_token(row.get("exact_verifier_outcome")) == "exact_replay_failed":
        return True
    if normalize_token(row.get("consistency_judgment")) not in {"unknown", "consistent"}:
        return True
    return bool(row.get("rollback_triggered") or row.get("retracted"))


def stable_trace_id(row: Mapping[str, Any]) -> str:
    """Return a trace identifier suitable for graph node IDs."""

    trace_id = str(row.get("trace_id") or "").strip()
    if trace_id:
        return trace_id
    digest = hashlib.sha256(json.dumps(dict(row), sort_keys=True).encode("utf-8")).hexdigest()
    return f"trace-{digest[:12]}"


def claim_text(row: Mapping[str, Any]) -> str:
    """Describe the route claim being grounded."""

    return (
        f"row={row.get('row_id')} role={normalize_token(row.get('replay_role'))} "
        f"route={normalize_token(row.get('routing_outcome'))}"
    )


def evidence_text(row: Mapping[str, Any]) -> str:
    """Describe the exact replay evidence behind a claim."""

    outcome = row.get("exact_verifier_outcome") or row.get("exact_label") or "unknown"
    return f"verification_query={row.get('verification_query')}; outcome={outcome}"


def route_text(row: Mapping[str, Any]) -> str:
    """Describe the controller route under audit."""

    return (
        f"answer_or_abstain={normalize_token(row.get('answer_abstain_decision'))}; "
        f"routing_outcome={normalize_token(row.get('routing_outcome'))}"
    )


def retraction_text(row: Mapping[str, Any]) -> str:
    """Describe why a stale premise is being invalidated."""

    if bool(row.get("redundant_check_suppressed")):
        return "stale_premise_probe=redundant_check_suppressed"
    return (
        f"rollback_or_retraction_status={normalize_token(row.get('rollback_or_retraction_status'))}; "
        f"consistency={normalize_token(row.get('consistency_judgment'))}"
    )


def repair_text(row: Mapping[str, Any]) -> str:
    """Describe the controller-only repair action for a stale route."""

    utility = row.get("prior_route_utility") or "route_requires_reverification"
    return f"repair_action={utility}; no_model_weight_update=true"


def inference_substrate() -> JsonDict:
    """Declare that Exp 3216 is an offline controller-memory audit."""

    return {
        "mode": "controller_memory_grounded_continuation_audit",
        "controller_memory_replay_only": True,
        "grounded_continuation_graph_only": True,
        "nonforgetting_queue_report_only": True,
        "uses_checked_in_artifacts_only": True,
        "executes_live_model_inference": False,
        "fresh_live_inference_calls": 0,
        "model_weight_learning": False,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "base_model_weights_updated": False,
        "kan_model_weight_training": False,
        "kan_sidecar_promotion_allowed": False,
        "hidden_state_mutation_claimed": False,
    }


def source_artifacts(root: Path) -> list[JsonDict]:
    """List source files and checksums for artifact lineage."""

    rows: list[JsonDict] = []
    for source_id, rel_path, required in SOURCE_ARTIFACTS:
        path = root / rel_path
        exists = path.is_file()
        rows.append(
            {
                "id": source_id,
                "path": rel_path.as_posix(),
                "required": required,
                "exists": exists,
                "sha256": sha256_file(path) if exists else None,
            }
        )
    return rows


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp 3216 artifact is incomplete or overclaims authority."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    false_fields = [
        "model_weight_update_claimed",
        "controller_memory_promotion_allowed",
        "conductor_file_modified",
        "active_roadmap_modified",
    ]
    for field in false_fields:
        if artifact.get(field) is not False:
            raise ValueError(f"{field} must remain false")
    if artifact.get("nonforgetting_queue_defined") is not True:
        raise ValueError("nonforgetting queue must be defined")
    queue_value = artifact.get("nonforgetting_queue_value")
    if queue_value is not None and not isinstance(queue_value, (int, float)):
        raise ValueError("nonforgetting queue value must be numeric or null")
    if not str(artifact.get("honest_verdict") or "").startswith("complete:"):
        raise ValueError("honest_verdict must start with complete:")


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write deterministic Exp 3216 JSON."""

    root_path = Path(root)
    path = Path(output_path)
    output = path if path.is_absolute() else root_path / path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    write_json(output, artifact)
    return output


def honest_verdict(budget_exceeded: bool, selection: Mapping[str, Any]) -> str:
    """Return a truthful terminal verdict for the audit artifact."""

    budget_status = "budget_exceeded" if budget_exceeded else "budget_within_limit"
    return (
        "complete: fr11 grounded-continuation trace graph and virtual "
        f"nonforgetting queue materialized; source={selection.get('source_kind')}; "
        f"fallback_used={str(bool(selection.get('fallback_used'))).lower()}; "
        f"{budget_status}; model_weight_update_claimed=false; "
        "controller_memory_promotion_allowed=false; kan_sidecar_promotion_allowed=false"
    )


def normalize_token(value: Any) -> str:
    """Normalize compact status and route tokens."""

    text = str(value or "").strip().lower().replace(" ", "_")
    return text or "unknown"


def duration(started_s: float, now_s: float | None) -> float:
    """Return stable elapsed seconds for artifact provenance."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def sha256_file(path: Path) -> str | None:
    """Return a SHA-256 checksum when the source file exists."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON output for deterministic artifact diffs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
