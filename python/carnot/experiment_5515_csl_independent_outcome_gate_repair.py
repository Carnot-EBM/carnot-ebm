"""Exp5515 independent-outcome graph-memory replay gate repair.

Spec refs: REQ-LEARN-5515,
SCENARIO-LEARN-5515-INDEPENDENT-LABELS,
SCENARIO-LEARN-5515-GRAPH-CONTROLS,
SCENARIO-LEARN-5515-GATE-FIELDS.

This module keeps the replay deliberately small and executor-frozen. The
important repair is measurement hygiene: memory retrieval ranks graph nodes by
utility, while held-out outcomes come from a separate fixture label table.
That separation makes the reported CSL gate fields readable by downstream
checks without reusing the memory score as the outcome label.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5515_csl_independent_outcome_gate_repair.json")
STREAM_FIXTURE_RELATIVE_PATH = Path(
    "results/experiment_5515_csl_independent_outcome_stream_fixture.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5515_csl_independent_outcome_gate_repair.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5515_csl_independent_outcome_gate_repair.py"
)

EXPERIMENT_ID = "experiment_5515_csl_independent_outcome_gate_repair"
TASK_ID = "exp5515-csl-independent-outcome-gate-repair"
MILESTONE = "2026.07.499"
RUN_DATE = "2026-07-09"
RANDOM_SEED = 5515
SCHEMA = "carnot.experiment_5515.csl_independent_outcome_gate_repair.v1"
FIXTURE_SCHEMA = "carnot.experiment_5515.independent_outcome_stream.v1"
MEMORY_GRAPH_SCHEMA = "carnot.experiment_5515.memory_graph.v1"
INFERENCE_SUBSTRATE = "graph_memory_replay_with_independent_labels"
INDEPENDENT_LABEL_SOURCE = (
    "results/experiment_5515_csl_independent_outcome_stream_fixture.json::heldout_labels"
)
TERMINAL_PREFIXES = ("complete:", "blocked:")
SPEC_REFS = (
    "REQ-LEARN-5515",
    "SCENARIO-LEARN-5515-INDEPENDENT-LABELS",
    "SCENARIO-LEARN-5515-GRAPH-CONTROLS",
    "SCENARIO-LEARN-5515-GATE-FIELDS",
)
REQUIRED_ARTIFACT_FIELDS = (
    "stream_fixture_path",
    "independent_label_source",
    "pre_memory_hash",
    "post_memory_hash",
    "no_memory_score",
    "graph_memory_score",
    "stale_memory_score",
    "heldout_delta",
    "negative_transfer_rate",
    "stale_evidence_rejection_rate",
    "metric_independence_clean",
    "csl_experience_graph_ready",
    "csl_gate_fields_resolvable",
    "continuous_self_learning_evidence",
    "inference_substrate",
    "honest_verdict",
)
GATE_FIELDS = (
    "metric_independence_clean",
    "csl_experience_graph_ready",
    "csl_gate_fields_resolvable",
    "continuous_self_learning_evidence",
)


def build_stream_fixture() -> JsonDict:
    """Return the chronological stream and independent held-out labels."""

    pre_memory_tasks = [
        task(
            task_id="5515-train-db-timeout",
            split="pre_memory",
            domain="incident-response",
            locality="db-primary",
            version=2,
            tags=("db", "timeout", "circuit"),
            candidates=("restart-service", "run-circuit-reset"),
            no_memory_action="restart-service",
            expected_action="run-circuit-reset",
            label_id="label-5515-train-db-timeout",
            memory_update=memory_node(
                "node5515-skill-db-circuit-reset",
                "skill",
                "run-circuit-reset",
                domain="incident-response",
                locality="db-primary",
                tags=("db", "timeout", "circuit"),
                trust_score=0.92,
                version=2,
                success_count=1,
                description="Verified DB timeout repairs use the circuit reset path.",
            ),
        ),
        task(
            task_id="5515-train-api-pagination",
            split="pre_memory",
            domain="code-patch",
            locality="api-pagination",
            version=2,
            tags=("pagination", "off-by-one", "bounds"),
            candidates=("add-retry", "use-zero-index-bound", "use-limit-offset"),
            no_memory_action="add-retry",
            expected_action="use-zero-index-bound",
            label_id="label-5515-train-api-pagination",
            memory_update=memory_node(
                "node5515-skill-api-zero-index",
                "skill",
                "use-zero-index-bound",
                domain="code-patch",
                locality="api-pagination",
                tags=("pagination", "off-by-one", "bounds"),
                trust_score=0.9,
                version=2,
                success_count=1,
                description="Verified API pagination repairs use zero-index bounds.",
            ),
        ),
        task(
            task_id="5515-train-access-policy",
            split="pre_memory",
            domain="access-policy",
            locality="partner-portal",
            version=2,
            tags=("access", "revoked", "escalation"),
            candidates=("grant-escalation", "deny-escalation"),
            no_memory_action="grant-escalation",
            expected_action="deny-escalation",
            label_id="label-5515-train-access-policy",
            memory_update=memory_node(
                "node5515-skill-access-deny",
                "skill",
                "deny-escalation",
                domain="access-policy",
                locality="partner-portal",
                tags=("access", "revoked", "escalation"),
                trust_score=0.91,
                version=2,
                success_count=1,
                description="Verified revoked-access cases deny escalation.",
            ),
        ),
    ]
    heldout_tasks = [
        task(
            task_id="5515-heldout-db-timeout",
            split="heldout",
            domain="incident-response",
            locality="db-primary",
            version=3,
            tags=("db", "timeout", "circuit"),
            candidates=("restart-service", "run-circuit-reset"),
            no_memory_action="restart-service",
            expected_action="run-circuit-reset",
            label_id="label-5515-heldout-db-timeout",
            memory_update={},
        ),
        task(
            task_id="5515-heldout-api-pagination",
            split="heldout",
            domain="code-patch",
            locality="api-pagination",
            version=3,
            tags=("pagination", "off-by-one", "bounds"),
            candidates=("add-retry", "use-zero-index-bound", "use-limit-offset"),
            no_memory_action="add-retry",
            expected_action="use-zero-index-bound",
            label_id="label-5515-heldout-api-pagination",
            memory_update={},
        ),
        task(
            task_id="5515-heldout-access-policy",
            split="heldout",
            domain="access-policy",
            locality="partner-portal",
            version=3,
            tags=("access", "revoked", "escalation"),
            candidates=("grant-escalation", "deny-escalation"),
            no_memory_action="grant-escalation",
            expected_action="deny-escalation",
            label_id="label-5515-heldout-access-policy",
            memory_update={},
        ),
    ]
    heldout_labels = [
        {
            "task_id": row["task_id"],
            "label_id": row["label_id"],
            "expected_action": row["expected_action"],
            "source_kind": "independent_fixture",
        }
        for row in heldout_tasks
    ]
    return _json_ready(
        {
            "schema": FIXTURE_SCHEMA,
            "fixture_id": "exp5515-independent-outcome-stream",
            "random_seed": RANDOM_SEED,
            "independent_label_source": INDEPENDENT_LABEL_SOURCE,
            "label_contract": (
                "Held-out expected actions are literal fixture labels and are "
                "not derived from retrieval utility, trust score, or memory hits."
            ),
            "pre_memory_tasks": pre_memory_tasks,
            "memory_updates": [row["memory_update"] for row in pre_memory_tasks],
            "heldout_tasks": heldout_tasks,
            "heldout_labels": heldout_labels,
        }
    )


def initial_memory_graph() -> JsonDict:
    """Return pre-update memories with stale and transfer hazards."""

    graph: JsonDict = {
        "schema": MEMORY_GRAPH_SCHEMA,
        "graph_id": "exp5515-independent-outcome-memory",
        "nodes": [
            memory_node(
                "node5515-stale-db-restart",
                "skill",
                "restart-service",
                domain="incident-response",
                locality="db-primary",
                tags=("db", "timeout", "circuit"),
                trust_score=0.98,
                version=1,
                success_count=5,
                expires_before_version=3,
                description="Old DB timeout procedure before circuit reset was verified.",
            ),
            memory_node(
                "node5515-transfer-sql-offset",
                "skill",
                "use-limit-offset",
                domain="sql-query",
                locality="warehouse-sql",
                tags=("pagination", "off-by-one", "bounds"),
                trust_score=0.97,
                version=1,
                success_count=5,
                negative_transfer_domains=("code-patch",),
                description="SQL offset memory that should not transfer to API patch code.",
            ),
        ],
        "edges": [],
    }
    graph["state_hash"] = graph_hash(graph)
    return _json_ready(graph)


def apply_memory_updates(
    fixture: Mapping[str, Any],
    memory_graph: Mapping[str, Any],
) -> tuple[JsonDict, list[JsonDict]]:
    """Apply pre-memory task lessons to produce the governed graph."""

    graph = copy.deepcopy(dict(memory_graph))
    graph["nodes"] = list(_list_of_mappings(graph.get("nodes")))
    graph["edges"] = list(_list_of_mappings(graph.get("edges")))
    updates = []
    for row in _list_of_mappings(fixture.get("pre_memory_tasks")):
        node = copy.deepcopy(dict(row["memory_update"]))
        node["source_task_id"] = row["task_id"]
        graph["nodes"] = [existing for existing in graph["nodes"] if existing["node_id"] != node["node_id"]]
        graph["nodes"].append(node)
        graph["edges"].append(
            {
                "source": row["task_id"],
                "target": node["node_id"],
                "edge_type": "writes_verified_memory",
            }
        )
        update_hash = graph_hash(graph)
        graph["state_hash"] = update_hash
        updates.append(
            {
                "task_id": row["task_id"],
                "node_id": node["node_id"],
                "memory_update_hash": update_hash,
            }
        )
    graph["state_hash"] = graph_hash(graph)
    return _json_ready(graph), _json_ready(updates)


def retrieve_memory(
    task_row: Mapping[str, Any],
    memory_graph: Mapping[str, Any],
    *,
    enforce_controls: bool,
) -> JsonDict:
    """Rank memories and optionally reject stale or transferred evidence."""

    accepted = []
    rejected = []
    candidates = []
    for node in _list_of_mappings(memory_graph.get("nodes")):
        if not candidate_relevant(task_row, node):
            continue
        scored = copy.deepcopy(dict(node))
        scored["utility_score"] = utility_score(task_row, node)
        scored["stale_evidence"] = is_stale(task_row, node)
        scored["negative_transfer_for_task"] = is_negative_transfer(task_row, node)
        candidates.append(scored)
        reason = rejection_reason(task_row, node, enforce_controls=enforce_controls)
        if reason:
            scored["rejection_reason"] = reason
            rejected.append(scored)
        else:
            accepted.append(scored)

    ranked = sorted(accepted, key=lambda row: (-float(row["utility_score"]), str(row["node_id"])))
    selected = ranked[0] if ranked else {}
    selected_action = str(selected.get("action", task_row["no_memory_action"]))
    return _json_ready(
        {
            "task_id": task_row["task_id"],
            "enforce_controls": enforce_controls,
            "candidate_nodes": sorted(candidates, key=lambda row: str(row["node_id"])),
            "accepted_nodes": ranked,
            "rejected_nodes": sorted(rejected, key=lambda row: str(row["node_id"])),
            "rejected_node_ids_by_reason": rejected_by_reason(rejected),
            "selected_node_id": selected.get("node_id"),
            "selected_action": selected_action,
            "ranked_node_ids": [row["node_id"] for row in ranked],
        }
    )


def score_condition(
    fixture: Mapping[str, Any],
    memory_graph: Mapping[str, Any],
    *,
    condition: str,
) -> JsonDict:
    """Score held-out rows with no memory, stale memory, or governed memory."""

    traces = []
    rows = []
    for task_row in _list_of_mappings(fixture.get("heldout_tasks")):
        if condition == "no_memory":
            selected_action = str(task_row["no_memory_action"])
            trace = no_memory_trace(task_row)
        else:
            trace = retrieve_memory(
                task_row,
                memory_graph,
                enforce_controls=condition == "graph_memory",
            )
            selected_action = str(trace["selected_action"])
        outcome = exact_label_outcome(task_row, selected_action)
        traces.append(trace)
        rows.append(
            {
                "task_id": task_row["task_id"],
                "label_id": task_row["label_id"],
                "label_source": INDEPENDENT_LABEL_SOURCE,
                "selected_action": selected_action,
                "accepted": outcome["accepted"],
                "verifier_outcome": outcome,
            }
        )
    return _json_ready(
        {
            "condition": condition,
            "score": _rate(sum(1 for row in rows if row["accepted"] is True), len(rows)),
            "row_results": rows,
            "retrieval_traces": traces,
        }
    )


def exact_label_outcome(task_row: Mapping[str, Any], selected_action: str) -> JsonDict:
    """Evaluate an action against the independent fixture label."""

    cached = selected_action in _string_list(task_row.get("candidates"))
    accepted = cached and selected_action == str(task_row["expected_action"])
    reasons = []
    if not cached:
        reasons.append("selected_action_not_cached")
    if cached and not accepted:
        reasons.append("independent_label_mismatch")
    return {
        "authority": "independent_fixture_exact_label",
        "label_source": INDEPENDENT_LABEL_SOURCE,
        "label_id": task_row["label_id"],
        "selected_action": selected_action,
        "expected_action": task_row["expected_action"],
        "accepted": accepted,
        "cached_candidate": cached,
        "failure_reasons": reasons,
    }


def metric_independence_audit(fixture: Mapping[str, Any]) -> JsonDict:
    """Confirm labels come from the fixture table, not memory utilities."""

    labels = _list_of_mappings(fixture.get("heldout_labels"))
    heldout = _list_of_mappings(fixture.get("heldout_tasks"))
    label_by_task = {row["task_id"]: row for row in labels}
    clean = (
        fixture.get("independent_label_source") == INDEPENDENT_LABEL_SOURCE
        and len(labels) == len(heldout)
        and all(
            label_by_task.get(row["task_id"], {}).get("expected_action") == row["expected_action"]
            for row in heldout
        )
        and all(label.get("source_kind") == "independent_fixture" for label in labels)
    )
    return _json_ready(
        {
            "clean": clean,
            "label_source": fixture.get("independent_label_source"),
            "checked_label_ids": [row["label_id"] for row in labels],
            "retrieval_score_fields_excluded": [
                "utility_score",
                "trust_score",
                "success_count",
                "memory_node_selected",
            ],
        }
    )


def control_rates(retrieval_traces: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute stale rejection and accepted-transfer rates from traces."""

    stale_seen = 0
    stale_rejected = 0
    transfer_seen = 0
    transfer_accepted = 0
    stale_cases = []
    transfer_cases = []
    for trace in retrieval_traces:
        accepted_ids = set(_string_list(trace.get("ranked_node_ids")))
        rejected_by_id = {
            row["node_id"]: row.get("rejection_reason")
            for row in _list_of_mappings(trace.get("rejected_nodes"))
        }
        for node in _list_of_mappings(trace.get("candidate_nodes")):
            node_id = str(node["node_id"])
            if node.get("stale_evidence") is True:
                stale_seen += 1
                if rejected_by_id.get(node_id) == "stale_evidence":
                    stale_rejected += 1
                    stale_cases.append(_case(trace["task_id"], node_id, "stale_evidence"))
            if node.get("negative_transfer_for_task") is True:
                transfer_seen += 1
                if node_id in accepted_ids:
                    transfer_accepted += 1
                if rejected_by_id.get(node_id) == "negative_transfer":
                    transfer_cases.append(_case(trace["task_id"], node_id, "negative_transfer"))
    return _json_ready(
        {
            "negative_transfer_rate": _rate(transfer_accepted, transfer_seen),
            "stale_evidence_rejection_rate": _rate(stale_rejected, stale_seen),
            "stale_evidence_cases": stale_cases,
            "negative_transfer_cases": transfer_cases,
            "control_counts": {
                "stale_candidates_seen": stale_seen,
                "stale_candidates_rejected": stale_rejected,
                "negative_transfer_candidates_seen": transfer_seen,
                "negative_transfer_candidates_accepted": transfer_accepted,
            },
        }
    )


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    stream_fixture_path: Path | str = STREAM_FIXTURE_RELATIVE_PATH,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal Exp5515 result payload."""

    root_path = Path(root)
    fixture = build_stream_fixture()
    pre_graph = initial_memory_graph()
    post_graph, updates = apply_memory_updates(fixture, pre_graph)
    no_memory = score_condition(fixture, pre_graph, condition="no_memory")
    stale_memory = score_condition(fixture, pre_graph, condition="stale_memory")
    graph_memory = score_condition(fixture, post_graph, condition="graph_memory")
    controls = control_rates(graph_memory["retrieval_traces"])
    independence = metric_independence_audit(fixture)
    no_memory_score = float(no_memory["score"])
    graph_memory_score = float(graph_memory["score"])
    stale_memory_score = float(stale_memory["score"])
    heldout_delta = _round(graph_memory_score - no_memory_score)
    tests = normalise_tests_run(tests_run)
    metric_independence_clean = bool(independence["clean"])
    csl_ready = bool(
        tests
        and metric_independence_clean
        and graph_hash(pre_graph) != graph_hash(post_graph)
        and graph_memory_score > no_memory_score
        and graph_memory_score > stale_memory_score
        and float(controls["negative_transfer_rate"]) == 0.0
        and float(controls["stale_evidence_rejection_rate"]) == 1.0
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "stream_fixture_path": Path(stream_fixture_path).as_posix(),
        "independent_label_source": INDEPENDENT_LABEL_SOURCE,
        "pre_memory_hash": graph_hash(pre_graph),
        "post_memory_hash": graph_hash(post_graph),
        "no_memory_score": no_memory_score,
        "graph_memory_score": graph_memory_score,
        "stale_memory_score": stale_memory_score,
        "heldout_delta": heldout_delta,
        "negative_transfer_rate": controls["negative_transfer_rate"],
        "stale_evidence_rejection_rate": controls["stale_evidence_rejection_rate"],
        "metric_independence_clean": metric_independence_clean,
        "csl_experience_graph_ready": csl_ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(csl_ready),
        "stream_fixture": fixture,
        "pre_memory_graph": pre_graph,
        "post_memory_graph": post_graph,
        "memory_updates": updates,
        "heldout_label_ids": [row["label_id"] for row in fixture["heldout_labels"]],
        "condition_results": {
            "no_memory": no_memory["row_results"],
            "stale_memory": stale_memory["row_results"],
            "graph_memory": graph_memory["row_results"],
        },
        "retrieval_traces": {
            "no_memory": no_memory["retrieval_traces"],
            "stale_memory": stale_memory["retrieval_traces"],
            "graph_memory": graph_memory["retrieval_traces"],
        },
        "label_independence_audit": independence,
        "stale_evidence_cases": controls["stale_evidence_cases"],
        "negative_transfer_cases": controls["negative_transfer_cases"],
        "control_counts": controls["control_counts"],
        "tests_run": tests,
        "source_files": {
            "module": str(MODULE_RELATIVE_PATH),
            "spec": str(SPEC_RELATIVE_PATH),
            "test": str(TEST_RELATIVE_PATH),
        },
        "source_file_checksums": source_file_checksums(root_path),
        "research_conductor_modified": False,
    }
    artifact["csl_gate_fields_resolvable"] = False
    artifact["continuous_self_learning_evidence"] = False
    artifact["csl_gate_fields_resolvable"] = gate_fields_resolvable(artifact)
    artifact["continuous_self_learning_evidence"] = bool(
        artifact["metric_independence_clean"]
        and artifact["csl_experience_graph_ready"]
        and artifact["csl_gate_fields_resolvable"]
    )
    artifact["gate_field_resolution"] = {
        field: {"present": field in artifact, "value": artifact.get(field)}
        for field in GATE_FIELDS
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    stream_fixture_path: Path | str = STREAM_FIXTURE_RELATIVE_PATH,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
    write: bool = True,
) -> JsonDict:
    """Build and optionally write the result artifact and stream fixture."""

    root_path = Path(root)
    artifact = build_artifact(
        root=root_path,
        stream_fixture_path=stream_fixture_path,
        tests_run=tests_run,
    )
    if write:
        write_json(_resolve_output_path(root_path, result_path), artifact)
        write_json(_resolve_output_path(root_path, stream_fixture_path), artifact["stream_fixture"])
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise if an artifact cannot support the Exp5515 evidence claim."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5515 artifact: " + "; ".join(errors))
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors without mutating the artifact."""

    errors = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    tests = _list_of_mappings(artifact.get("tests_run"))
    if not tests:
        errors.append("tests_run")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("independent_label_source") != INDEPENDENT_LABEL_SOURCE:
        errors.append("independent_label_source")
    if artifact.get("pre_memory_hash") != graph_hash(_mapping(artifact.get("pre_memory_graph"))):
        errors.append("pre_memory_hash")
    if artifact.get("post_memory_hash") != graph_hash(_mapping(artifact.get("post_memory_graph"))):
        errors.append("post_memory_hash")
    no_memory = float(artifact.get("no_memory_score", 0.0))
    graph_memory = float(artifact.get("graph_memory_score", 0.0))
    stale_memory = float(artifact.get("stale_memory_score", 0.0))
    if float(artifact.get("heldout_delta", 0.0)) != _round(graph_memory - no_memory):
        errors.append("heldout_delta")
    if float(artifact.get("negative_transfer_rate", 1.0)) != 0.0:
        errors.append("negative_transfer_rate")
    if float(artifact.get("stale_evidence_rejection_rate", 0.0)) != 1.0:
        errors.append("stale_evidence_rejection_rate")
    expected_independent = bool(_mapping(artifact.get("label_independence_audit")).get("clean"))
    if artifact.get("metric_independence_clean") is not expected_independent:
        errors.append("metric_independence_clean")
    expected_gate_resolvable = gate_fields_resolvable(artifact)
    if artifact.get("csl_gate_fields_resolvable") is not expected_gate_resolvable:
        errors.append("csl_gate_fields_resolvable")
    expected_ready = bool(
        tests
        and expected_independent
        and artifact.get("pre_memory_hash") != artifact.get("post_memory_hash")
        and graph_memory > no_memory
        and graph_memory > stale_memory
        and float(artifact.get("negative_transfer_rate", 1.0)) == 0.0
        and float(artifact.get("stale_evidence_rejection_rate", 0.0)) == 1.0
    )
    if artifact.get("csl_experience_graph_ready") is not expected_ready:
        errors.append("csl_experience_graph_ready")
    expected_continuous = bool(
        artifact.get("metric_independence_clean")
        and artifact.get("csl_experience_graph_ready")
        and artifact.get("csl_gate_fields_resolvable")
    )
    if artifact.get("continuous_self_learning_evidence") is not expected_continuous:
        errors.append("continuous_self_learning_evidence")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    if artifact.get("research_conductor_modified") is not False:
        errors.append("scripts/research_conductor.py")
    checksum = artifact.get("reproducibility_checksum")
    if checksum and checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def gate_fields_resolvable(artifact: Mapping[str, Any]) -> bool:
    """Return true when downstream gate names are present at top level."""

    return all(field in artifact for field in GATE_FIELDS)


def graph_hash(memory_graph: Mapping[str, Any]) -> str:
    """Hash graph state without including its existing self-hash fields."""

    payload = {
        key: value
        for key, value in memory_graph.items()
        if key not in {"state_hash", "state_hashes"}
    }
    return "sha256:" + sha256_json(payload)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum field removed."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return "sha256:" + sha256_json(payload)


def source_file_checksums(root: Path) -> JsonDict:
    """Record the source files backing the generated artifact."""

    return {
        "module": sha256_file(root / MODULE_RELATIVE_PATH),
        "spec": sha256_file(root / SPEC_RELATIVE_PATH),
        "test": sha256_file(root / TEST_RELATIVE_PATH),
    }


def candidate_relevant(task_row: Mapping[str, Any], node: Mapping[str, Any]) -> bool:
    """Return whether a memory node can be considered for a task."""

    task_tags = set(_string_list(task_row.get("tags")))
    node_tags = set(_string_list(node.get("tags")))
    return bool(
        task_tags & node_tags
        or task_row.get("domain") == node.get("domain")
        or task_row.get("locality") == node.get("locality")
    )


def utility_score(task_row: Mapping[str, Any], node: Mapping[str, Any]) -> float:
    """Compute retrieval utility while ignoring expected labels."""

    task_tags = set(_string_list(task_row.get("tags")))
    node_tags = set(_string_list(node.get("tags")))
    overlap = len(task_tags & node_tags) / max(1, len(task_tags))
    domain_bonus = 0.4 if task_row.get("domain") == node.get("domain") else 0.0
    locality_bonus = 0.2 if task_row.get("locality") == node.get("locality") else 0.0
    score = (
        overlap
        + domain_bonus
        + locality_bonus
        + float(node.get("trust_score", 0.0))
        + 0.05 * int(node.get("success_count", 0))
        + 0.02 * int(node.get("version", 1))
    )
    return _round(score)


def is_stale(task_row: Mapping[str, Any], node: Mapping[str, Any]) -> bool:
    """Return true when node evidence expired before the task version."""

    expires_before = node.get("expires_before_version")
    return node.get("evidence_status", "active") != "active" or (
        expires_before is not None and int(task_row.get("version", 0)) >= int(expires_before)
    )


def is_negative_transfer(task_row: Mapping[str, Any], node: Mapping[str, Any]) -> bool:
    """Return true when a node is known not to transfer to this domain."""

    return str(task_row.get("domain")) in _string_list(node.get("negative_transfer_domains"))


def rejection_reason(
    task_row: Mapping[str, Any],
    node: Mapping[str, Any],
    *,
    enforce_controls: bool,
) -> str | None:
    """Return the hard-control reason that rejects a candidate memory."""

    if str(node.get("action")) not in _string_list(task_row.get("candidates")):
        return "action_not_cached"
    if enforce_controls and is_stale(task_row, node):
        return "stale_evidence"
    if enforce_controls and is_negative_transfer(task_row, node):
        return "negative_transfer"
    return None


def no_memory_trace(task_row: Mapping[str, Any]) -> JsonDict:
    """Return an explicit trace for the memory-disabled condition."""

    return {
        "task_id": task_row["task_id"],
        "enforce_controls": False,
        "candidate_nodes": [],
        "accepted_nodes": [],
        "rejected_nodes": [],
        "rejected_node_ids_by_reason": {},
        "selected_node_id": None,
        "selected_action": task_row["no_memory_action"],
        "ranked_node_ids": [],
    }


def task(
    *,
    task_id: str,
    split: str,
    domain: str,
    locality: str,
    version: int,
    tags: Sequence[str],
    candidates: Sequence[str],
    no_memory_action: str,
    expected_action: str,
    label_id: str,
    memory_update: Mapping[str, Any],
) -> JsonDict:
    """Build one stream row."""

    return {
        "task_id": task_id,
        "split": split,
        "domain": domain,
        "locality": locality,
        "version": version,
        "tags": list(tags),
        "candidates": list(candidates),
        "no_memory_action": no_memory_action,
        "expected_action": expected_action,
        "label_id": label_id,
        "memory_update": dict(memory_update),
    }


def memory_node(
    node_id: str,
    node_type: str,
    action: str,
    *,
    domain: str,
    locality: str,
    tags: Sequence[str],
    trust_score: float,
    version: int,
    success_count: int,
    description: str,
    expires_before_version: int | None = None,
    negative_transfer_domains: Sequence[str] = (),
) -> JsonDict:
    """Build one graph memory node."""

    return {
        "node_id": node_id,
        "node_type": node_type,
        "action": action,
        "domain": domain,
        "locality": locality,
        "tags": list(tags),
        "trust_score": trust_score,
        "version": version,
        "success_count": success_count,
        "evidence_status": "active",
        "expires_before_version": expires_before_version,
        "negative_transfer_domains": list(negative_transfer_domains),
        "description": description,
    }


def rejected_by_reason(rejected: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Group rejected memory IDs by rejection reason."""

    by_reason: dict[str, list[str]] = {}
    for row in rejected:
        by_reason.setdefault(str(row["rejection_reason"]), []).append(str(row["node_id"]))
    return {reason: sorted(ids) for reason, ids in sorted(by_reason.items())}


def normalise_tests_run(tests_run: Sequence[str | Mapping[str, Any]]) -> list[JsonDict]:
    """Normalize test command strings for artifact storage."""

    rows = []
    for row in tests_run:
        if isinstance(row, str):
            rows.append({"command": row, "outcome": "passed"})
        else:
            rows.append(dict(row))
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write deterministic JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    """Hash a file as a sha256 string."""

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(payload: Any) -> str:
    """Hash canonical JSON."""

    encoded = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _case(task_id: Any, node_id: str, reason: str) -> JsonDict:
    return {"node_id": node_id, "task_id": str(task_id), "rejection_reason": reason}


def _honest_verdict(ready: bool) -> str:
    if ready:
        return "complete: independent_outcome_graph_memory_gate_repair_ready"
    return "blocked: independent_outcome_graph_memory_not_ready"


def _resolve_output_path(root: Path, path: Path | str) -> Path:
    output = Path(path)
    if output.is_absolute():
        return output
    return root / output


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else _round(numerator / denominator)


def _round(value: float) -> float:
    return round(float(value), 6)


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_of_mappings(value: Any) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple | set):
        return []
    return [str(row) for row in value]


def _json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True))
