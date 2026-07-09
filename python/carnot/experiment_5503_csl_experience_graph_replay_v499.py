"""Exp5503: executor-frozen experience-graph replay for CSL.

Spec refs: REQ-LEARN-5503, SCENARIO-LEARN-5503-GRAPH-UPDATE,
SCENARIO-LEARN-5503-RETRIEVAL-CONTROLS, SCENARIO-LEARN-5503-BASELINE,
SCENARIO-LEARN-5503-ARTIFACT.

This fixture treats memory as an external graph that a frozen executor can
query and update. The held-out score is an exact verifier pass rate over cached
candidate actions, so it cannot collapse into the utility scalar used for
memory retrieval ranking.
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
RESULT_RELATIVE_PATH = Path("results/experiment_5503_csl_experience_graph_replay_v499.json")
REPLAY_FIXTURE_RELATIVE_PATH = Path(
    "results/experiment_5503_csl_experience_graph_replay_fixture_v499.json"
)
MEMORY_GRAPH_RELATIVE_PATH = Path(
    "results/experiment_5503_csl_experience_graph_memory_v499.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5503_csl_experience_graph_replay_v499.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5503_csl_experience_graph_replay_v499.py"
)

EXPERIMENT_ID = "experiment_5503_csl_experience_graph_replay_v499"
TASK_ID = "exp5503-csl-experience-graph-replay-v499"
MILESTONE = "2026.07.499"
RUN_DATE = "2026-07-09"
RANDOM_SEED = 5503
SCHEMA = "carnot.experiment_5503.csl_experience_graph_replay.v499"
FIXTURE_SCHEMA = "carnot.experiment_5503.replay_fixture.v1"
MEMORY_GRAPH_SCHEMA = "carnot.experiment_5503.memory_graph.v1"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
EXACT_VERIFIER = "cached_candidate_exact_action_verifier"
TERMINAL_PREFIXES = ("complete:", "blocked:")
SPEC_REFS = (
    "REQ-LEARN-5503",
    "SCENARIO-LEARN-5503-GRAPH-UPDATE",
    "SCENARIO-LEARN-5503-RETRIEVAL-CONTROLS",
    "SCENARIO-LEARN-5503-BASELINE",
    "SCENARIO-LEARN-5503-ARTIFACT",
)
REQUIRED_ARTIFACT_FIELDS = (
    "replay_fixture_path",
    "memory_graph_path",
    "test_paths",
    "num_stream_tasks",
    "memory_state_hashes",
    "no_memory_baseline_score",
    "graph_memory_score",
    "heldout_delta",
    "negative_transfer_rate",
    "stale_evidence_rejection_rate",
    "metric_independence_notes",
    "csl_experience_graph_ready",
    "model_weights_mutated",
    "inference_substrate",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "replay_fixture_path": "External chronological stream fixture used for replay.",
    "memory_graph_path": "External skill and failure graph after all updates.",
    "test_paths": "Focused tests that anchor REQ-LEARN-5503 behavior.",
    "num_stream_tasks": "Number of stream interactions processed by the frozen executor.",
    "memory_state_hashes": "Deterministic graph hashes after each episode update.",
    "no_memory_baseline_score": "Exact held-out score without memory access.",
    "graph_memory_score": "Exact held-out score with governed graph-memory retrieval.",
    "heldout_delta": "Exact graph-memory minus no-memory pass-rate delta.",
    "negative_transfer_rate": "Accepted negative-transfer candidates divided by candidates seen.",
    "stale_evidence_rejection_rate": "Rejected stale candidates divided by stale candidates seen.",
    "metric_independence_notes": "Why retrieval utility is not the held-out score scalar.",
    "csl_experience_graph_ready": "Terminal readiness gate for executor-frozen memory replay.",
    "model_weights_mutated": "Frozen model and adapter boundary.",
    "inference_substrate": "Cached-candidate verifier substrate declaration.",
    "honest_verdict": "Terminal status beginning with complete: or blocked:.",
}


def build_replay_fixture() -> JsonDict:
    """Return a tiny stream where memory can be learned before held-out rows.

    The rows use cached action candidates so the executor never calls a model.
    Train rows create reusable skill or failure nodes; held-out rows check that
    retrieval uses those nodes while rejecting stale, conflicting, or transferred
    evidence.
    """

    stream_tasks = [
        _task(
            task_id="5503-train-dock-crate",
            split="train",
            domain="logistics",
            locality="dock-7",
            version=1,
            tags=("dock", "crate", "release"),
            candidates=("crate-red", "crate-blue"),
            no_memory_action="crate-red",
            expected_action="crate-blue",
            lesson_node=_lesson_node(
                "node5503-failure-dock-crate-red",
                "failure",
                "crate-blue",
                domain="logistics",
                locality="dock-7",
                tags=("dock", "crate", "release"),
                trust_score=0.84,
                description="The red-crate shortcut failed; use the blue crate for dock-7 releases.",
            ),
        ),
        _task(
            task_id="5503-heldout-dock-crate",
            split="heldout",
            domain="logistics",
            locality="dock-7",
            version=1,
            tags=("dock", "crate", "release"),
            candidates=("crate-red", "crate-blue"),
            no_memory_action="crate-red",
            expected_action="crate-blue",
            lesson_node=_lesson_node(
                "node5503-skill-dock-crate-blue",
                "skill",
                "crate-blue",
                domain="logistics",
                locality="dock-7",
                tags=("dock", "crate", "release"),
                trust_score=0.9,
                description="Dock-7 release routing now prefers the blue crate.",
            ),
        ),
        _task(
            task_id="5503-train-python-loop",
            split="train",
            domain="python-loop",
            locality="repo-alpha",
            version=1,
            tags=("bounds", "iteration", "repair"),
            candidates=("use-sql-offset", "use-range-len"),
            no_memory_action="use-range-len",
            expected_action="use-range-len",
            lesson_node=_lesson_node(
                "node5503-skill-python-range-len",
                "skill",
                "use-range-len",
                domain="python-loop",
                locality="repo-alpha",
                tags=("bounds", "iteration", "repair"),
                trust_score=0.88,
                description="Python loop repairs should use range(len(items)) for bound checks.",
            ),
        ),
        _task(
            task_id="5503-heldout-python-loop",
            split="heldout",
            domain="python-loop",
            locality="repo-alpha",
            version=1,
            tags=("bounds", "iteration", "repair"),
            candidates=("use-sql-offset", "use-range-len"),
            no_memory_action="use-sql-offset",
            expected_action="use-range-len",
            lesson_node=_lesson_node(
                "node5503-skill-python-heldout-range-len",
                "skill",
                "use-range-len",
                domain="python-loop",
                locality="repo-alpha",
                tags=("bounds", "iteration", "repair"),
                trust_score=0.9,
                description="Held-out Python bound repair confirmed the range-len skill.",
            ),
        ),
        _task(
            task_id="5503-train-rx4-handoff",
            split="train",
            domain="incident-routing",
            locality="rx-4",
            version=3,
            tags=("handoff", "rx4", "queue"),
            candidates=("queue-alpha", "queue-beta"),
            no_memory_action="queue-beta",
            expected_action="queue-beta",
            lesson_node=_lesson_node(
                "node5503-skill-rx4-beta",
                "skill",
                "queue-beta",
                domain="incident-routing",
                locality="rx-4",
                tags=("handoff", "rx4", "queue"),
                trust_score=0.91,
                description="RX-4 handoff moved ownership to queue beta.",
            ),
        ),
        _task(
            task_id="5503-heldout-rx4-handoff",
            split="heldout",
            domain="incident-routing",
            locality="rx-4",
            version=4,
            tags=("handoff", "rx4", "queue"),
            candidates=("queue-alpha", "queue-beta"),
            no_memory_action="queue-alpha",
            expected_action="queue-beta",
            lesson_node=_lesson_node(
                "node5503-skill-rx4-heldout-beta",
                "skill",
                "queue-beta",
                domain="incident-routing",
                locality="rx-4",
                tags=("handoff", "rx4", "queue"),
                trust_score=0.92,
                description="Held-out RX-4 replay confirmed stale alpha evidence must be ignored.",
            ),
        ),
        _task(
            task_id="5503-heldout-dock-gate",
            split="heldout",
            domain="controller-action",
            locality="dock-7",
            version=2,
            tags=("gate", "amber", "controller"),
            candidates=("gate-A", "gate-B"),
            no_memory_action="gate-A",
            expected_action="gate-B",
            lesson_node=_lesson_node(
                "node5503-skill-gate-b-heldout",
                "skill",
                "gate-B",
                domain="controller-action",
                locality="dock-7",
                tags=("gate", "amber", "controller"),
                trust_score=0.9,
                description="Held-out gate replay confirmed newer gate-B evidence.",
            ),
        ),
    ]
    return _json_ready(
        {
            "schema": FIXTURE_SCHEMA,
            "fixture_id": "exp5503-csl-experience-graph-stream",
            "random_seed": RANDOM_SEED,
            "stream_tasks": stream_tasks,
            "heldout_task_ids": [
                task["task_id"] for task in stream_tasks if task["split"] == "heldout"
            ],
        }
    )


def initial_memory_graph() -> JsonDict:
    """Return seed memories that exercise stale, conflict, and transfer gates."""

    graph = {
        "schema": MEMORY_GRAPH_SCHEMA,
        "graph_id": "exp5503-experience-graph",
        "nodes": [
            _memory_node(
                "node5503-stale-rx4-alpha",
                "skill",
                "queue-alpha",
                domain="incident-routing",
                locality="rx-4",
                tags=("handoff", "rx4", "queue"),
                trust_score=0.89,
                version=1,
                success_count=3,
                expires_before_version=3,
                description="Old RX-4 queue-alpha owner before the handoff.",
            ),
            _memory_node(
                "node5503-transfer-sql-offset",
                "skill",
                "use-sql-offset",
                domain="sql-pagination",
                locality="warehouse-db",
                tags=("bounds", "iteration", "repair"),
                trust_score=0.83,
                version=1,
                success_count=4,
                negative_transfer_domains=("python-loop",),
                description="SQL offset repair that should not transfer to Python loop bounds.",
            ),
            _memory_node(
                "node5503-conflict-gate-old",
                "skill",
                "gate-A",
                domain="controller-action",
                locality="dock-7",
                tags=("gate", "amber", "controller"),
                trust_score=0.61,
                version=1,
                success_count=2,
                conflict_group="dock-7-amber-gate",
                description="Older dock-7 amber gate action.",
            ),
            _memory_node(
                "node5503-conflict-gate-new",
                "skill",
                "gate-B",
                domain="controller-action",
                locality="dock-7",
                tags=("gate", "amber", "controller"),
                trust_score=0.92,
                version=2,
                success_count=3,
                conflict_group="dock-7-amber-gate",
                description="Newer verified dock-7 amber gate action.",
            ),
        ],
        "edges": [
            {
                "source": "node5503-conflict-gate-new",
                "target": "node5503-conflict-gate-old",
                "edge_type": "replaces",
            }
        ],
    }
    graph["state_hash"] = memory_state_hash(graph)
    return _json_ready(graph)


def retrieve_memory(task: Mapping[str, Any], memory_graph: Mapping[str, Any]) -> JsonDict:
    """Rank eligible graph nodes and reject unsafe memories before selection."""

    accepted: list[JsonDict] = []
    rejected: list[JsonDict] = []
    utility_scores: dict[str, float] = {}
    for node in _list_of_mappings(memory_graph.get("nodes")):
        if not _candidate_relevant(task, node):
            continue
        scored = copy.deepcopy(dict(node))
        scored["utility_score"] = utility_score(task, node)
        utility_scores[str(node["node_id"])] = scored["utility_score"]
        reason = rejection_reason(task, node)
        if reason:
            scored["rejection_reason"] = reason
            rejected.append(scored)
        else:
            accepted.append(scored)

    accepted, conflict_rejections = resolve_conflicts(accepted)
    rejected.extend(conflict_rejections)
    ranked = sorted(accepted, key=lambda row: (-float(row["utility_score"]), str(row["node_id"])))
    selected = ranked[0] if ranked else {}
    selected_action = str(selected.get("action", task["no_memory_action"]))
    selected_node_id = selected.get("node_id")
    return _json_ready(
        {
            "task_id": task["task_id"],
            "ranked_node_ids": [row["node_id"] for row in ranked],
            "accepted_node_ids": [row["node_id"] for row in ranked],
            "rejected_node_ids_by_reason": _rejected_by_reason(rejected),
            "selected_node_id": selected_node_id,
            "selected_action": selected_action,
            "utility_scores": utility_scores,
            "accepted_nodes": ranked,
            "rejected_nodes": sorted(rejected, key=lambda row: str(row["node_id"])),
        }
    )


def utility_score(task: Mapping[str, Any], node: Mapping[str, Any]) -> float:
    """Compute retrieval utility; this is deliberately not the outcome metric."""

    task_tags = set(_string_list(task.get("tags")))
    node_tags = set(_string_list(node.get("tags")))
    overlap = len(task_tags & node_tags) / max(1, len(task_tags))
    domain_bonus = 0.4 if node.get("domain") == task.get("domain") else 0.0
    locality_bonus = 0.2 if node.get("locality") == task.get("locality") else 0.0
    failure_bonus = 0.12 if node.get("node_type") == "failure" else 0.0
    score = (
        overlap
        + domain_bonus
        + locality_bonus
        + failure_bonus
        + float(node.get("trust_score", 0.0))
        + 0.08 * int(node.get("success_count", 0))
        - 0.04 * int(node.get("failure_count", 0))
        + 0.03 * int(node.get("version", 1))
    )
    return round(score, 6)


def rejection_reason(task: Mapping[str, Any], node: Mapping[str, Any]) -> str | None:
    """Return the first hard-control reason that blocks a candidate node."""

    task_version = int(task.get("version", 0))
    expires_before = node.get("expires_before_version")
    if node.get("evidence_status", "active") != "active" or (
        expires_before is not None and task_version >= int(expires_before)
    ):
        return "stale_evidence"
    if str(task.get("domain")) in _string_list(node.get("negative_transfer_domains")):
        return "negative_transfer"
    if str(node.get("action")) not in _string_list(task.get("candidates")):
        return "action_not_cached"
    return None


def resolve_conflicts(nodes: Sequence[Mapping[str, Any]]) -> tuple[list[JsonDict], list[JsonDict]]:
    """Keep the highest-utility member of each conflict group."""

    winners: list[JsonDict] = []
    rejected: list[JsonDict] = []
    grouped: dict[str, list[JsonDict]] = {}
    for node in nodes:
        group = str(node.get("conflict_group") or node["node_id"])
        grouped.setdefault(group, []).append(copy.deepcopy(dict(node)))
    for rows in grouped.values():
        ordered = sorted(rows, key=lambda row: (-float(row["utility_score"]), str(row["node_id"])))
        winners.append(ordered[0])
        for loser in ordered[1:]:
            loser["rejection_reason"] = "conflict_lower_utility"
            rejected.append(loser)
    return winners, rejected


def exact_verifier(task: Mapping[str, Any], selected_action: str) -> JsonDict:
    """Score a cached action by exact match against the task's expected action."""

    cached = selected_action in _string_list(task.get("candidates"))
    accepted = cached and selected_action == str(task["expected_action"])
    reasons: list[str] = []
    if not cached:
        reasons.append("selected_action_not_cached")
    if cached and not accepted:
        reasons.append("expected_action_mismatch")
    return {
        "authority": EXACT_VERIFIER,
        "accepted": accepted,
        "selected_action": selected_action,
        "expected_action": str(task["expected_action"]),
        "cached_candidate": cached,
        "failure_reasons": reasons,
    }


def apply_episode_update(
    memory_graph: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    selected_action: str,
    selected_node_id: Any,
    verifier_outcome: Mapping[str, Any],
) -> tuple[JsonDict, JsonDict, str]:
    """Write the task's deterministic lesson node and return the new graph hash."""

    updated = copy.deepcopy(dict(memory_graph))
    learned_node = copy.deepcopy(dict(task["lesson_node"]))
    learned_node["source_task_id"] = str(task["task_id"])
    learned_node["learned_from_selected_action"] = selected_action
    learned_node["verifier_accepted"] = bool(verifier_outcome["accepted"])
    learned_node["success_count"] = 1 if verifier_outcome["accepted"] else 0
    learned_node["failure_count"] = 0 if verifier_outcome["accepted"] else 1
    learned_node["evidence_status"] = "active"
    updated["nodes"] = [
        node
        for node in _list_of_mappings(updated.get("nodes"))
        if node.get("node_id") != learned_node["node_id"]
    ]
    updated["nodes"].append(learned_node)
    updated["edges"] = _list_of_mappings(updated.get("edges"))
    updated["edges"].append(
        {
            "source": str(learned_node["node_id"]),
            "target": str(selected_node_id or "no-memory-baseline"),
            "edge_type": "learned_after_action",
        }
    )
    update_hash = memory_state_hash(updated)
    updated["state_hash"] = update_hash
    return _json_ready(updated), _json_ready(learned_node), update_hash


def run_graph_replay(
    replay_fixture: Mapping[str, Any] | None = None,
    memory_graph: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Process the stream once, updating memory after every interaction."""

    fixture = copy.deepcopy(dict(replay_fixture or build_replay_fixture()))
    graph = copy.deepcopy(dict(memory_graph or initial_memory_graph()))
    tasks = _list_of_mappings(fixture["stream_tasks"])
    episode_records: list[JsonDict] = []
    memory_state_hashes: list[str] = []
    retrieval_decisions: list[JsonDict] = []

    for index, task in enumerate(tasks):
        retrieval = retrieve_memory(task, graph)
        verifier = exact_verifier(task, str(retrieval["selected_action"]))
        graph, learned_node, update_hash = apply_episode_update(
            graph,
            task,
            selected_action=str(retrieval["selected_action"]),
            selected_node_id=retrieval.get("selected_node_id"),
            verifier_outcome=verifier,
        )
        memory_state_hashes.append(update_hash)
        next_task = tasks[index + 1] if index + 1 < len(tasks) else None
        next_decision = (
            retrieve_memory(next_task, graph)
            if next_task is not None
            else _empty_next_decision()
        )
        retrieval_decisions.append(retrieval)
        episode_records.append(
            {
                "episode_id": index + 1,
                "task_id": task["task_id"],
                "split": task["split"],
                "task_state": _task_state(task),
                "selected_action": retrieval["selected_action"],
                "selected_memory_node_id": retrieval.get("selected_node_id"),
                "retrieval_decision": retrieval,
                "verifier_outcome": verifier,
                "learned_node": learned_node,
                "memory_update_hash": update_hash,
                "next_task_retrieval_decision": next_decision,
            }
        )

    heldout_ids = set(_string_list(fixture["heldout_task_ids"]))
    heldout_results = [
        {
            "task_id": episode["task_id"],
            "selected_action": episode["selected_action"],
            "accepted": episode["verifier_outcome"]["accepted"],
        }
        for episode in episode_records
        if episode["task_id"] in heldout_ids
    ]
    graph_score = _rate(
        sum(1 for row in heldout_results if row["accepted"] is True),
        len(heldout_results),
    )
    controls = _control_rates(retrieval_decisions)
    graph["state_hashes"] = memory_state_hashes
    graph["episode_count"] = len(episode_records)
    graph["state_hash"] = memory_state_hash(graph)
    return _json_ready(
        {
            "episode_records": episode_records,
            "memory_graph": graph,
            "memory_state_hashes": memory_state_hashes,
            "num_stream_tasks": len(tasks),
            "heldout_results": heldout_results,
            "graph_memory_score": graph_score,
            "negative_transfer_rate": controls["negative_transfer_rate"],
            "stale_evidence_rejection_rate": controls["stale_evidence_rejection_rate"],
            "control_counts": controls["control_counts"],
        }
    )


def score_no_memory(replay_fixture: Mapping[str, Any]) -> JsonDict:
    """Score the exact same held-out rows with memory retrieval disabled."""

    fixture = copy.deepcopy(dict(replay_fixture))
    heldout_ids = set(_string_list(fixture["heldout_task_ids"]))
    row_results = []
    for task in _list_of_mappings(fixture["stream_tasks"]):
        if task["task_id"] not in heldout_ids:
            continue
        verifier = exact_verifier(task, str(task["no_memory_action"]))
        row_results.append(
            {
                "task_id": task["task_id"],
                "selected_action": task["no_memory_action"],
                "accepted": verifier["accepted"],
                "verifier_outcome": verifier,
            }
        )
    return _json_ready(
        {
            "score": _rate(sum(1 for row in row_results if row["accepted"] is True), len(row_results)),
            "row_results": row_results,
        }
    )


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    replay_fixture_path: Path | str = REPLAY_FIXTURE_RELATIVE_PATH,
    memory_graph_path: Path | str = MEMORY_GRAPH_RELATIVE_PATH,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal result payload without writing files."""

    root_path = Path(root)
    fixture = build_replay_fixture()
    replay = run_graph_replay(fixture, initial_memory_graph())
    baseline = score_no_memory(fixture)
    no_memory_score = float(baseline["score"])
    graph_score = float(replay["graph_memory_score"])
    heldout_delta = _round(graph_score - no_memory_score)
    tests = _normalise_tests_run(tests_run)
    ready = bool(
        tests
        and heldout_delta > 0.0
        and graph_score > no_memory_score
        and float(replay["negative_transfer_rate"]) == 0.0
        and float(replay["stale_evidence_rejection_rate"]) == 1.0
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "replay_fixture_path": Path(replay_fixture_path).as_posix(),
        "memory_graph_path": Path(memory_graph_path).as_posix(),
        "test_paths": [str(TEST_RELATIVE_PATH)],
        "num_stream_tasks": replay["num_stream_tasks"],
        "memory_state_hashes": replay["memory_state_hashes"],
        "no_memory_baseline_score": no_memory_score,
        "graph_memory_score": graph_score,
        "heldout_delta": heldout_delta,
        "negative_transfer_rate": replay["negative_transfer_rate"],
        "stale_evidence_rejection_rate": replay["stale_evidence_rejection_rate"],
        "metric_independence_notes": (
            "Retrieval ranks nodes by utility_score from tag overlap, trust, counts, "
            "and freshness; heldout_delta is not the retrieval utility and is computed "
            "only from exact cached-candidate verifier outcomes on held-out task IDs."
        ),
        "csl_experience_graph_ready": ready,
        "model_weights_mutated": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": _honest_verdict(ready, heldout_delta),
        "replay_fixture": fixture,
        "memory_graph": replay["memory_graph"],
        "episode_records": replay["episode_records"],
        "heldout_task_ids": fixture["heldout_task_ids"],
        "graph_memory_results": replay["heldout_results"],
        "no_memory_baseline_results": baseline["row_results"],
        "control_counts": replay["control_counts"],
        "tests_run": tests,
        "source_files": {
            "module": str(MODULE_RELATIVE_PATH),
            "spec": str(SPEC_RELATIVE_PATH),
            "test": str(TEST_RELATIVE_PATH),
        },
        "source_file_checksums": source_file_checksums(root_path),
        "research_conductor_modified": False,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    replay_fixture_path: Path | str = REPLAY_FIXTURE_RELATIVE_PATH,
    memory_graph_path: Path | str = MEMORY_GRAPH_RELATIVE_PATH,
    tests_run: Sequence[str | Mapping[str, Any]] = (),
    write: bool = True,
) -> JsonDict:
    """Build the artifact and optionally write result, fixture, and graph JSON."""

    root_path = Path(root)
    artifact = build_artifact(
        root=root_path,
        replay_fixture_path=replay_fixture_path,
        memory_graph_path=memory_graph_path,
        tests_run=tests_run,
    )
    if write:
        _write_json(_resolve_output_path(root_path, result_path), artifact)
        _write_json(
            _resolve_output_path(root_path, replay_fixture_path),
            artifact["replay_fixture"],
        )
        _write_json(
            _resolve_output_path(root_path, memory_graph_path),
            artifact["memory_graph"],
        )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when the artifact cannot support an experience-graph-ready claim."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5503 artifact: " + "; ".join(errors))
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return deliverable validation errors without mutating the artifact."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    tests_run = _list_of_mappings(artifact.get("tests_run"))
    if not tests_run:
        errors.append("tests_run")
    if artifact.get("model_weights_mutated") is not False:
        errors.append("model_weights_mutated")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    no_memory = float(artifact.get("no_memory_baseline_score", 0.0))
    graph_memory = float(artifact.get("graph_memory_score", 0.0))
    expected_delta = _round(graph_memory - no_memory)
    if float(artifact.get("heldout_delta", 0.0)) != expected_delta:
        errors.append("heldout_delta")
    if float(artifact.get("negative_transfer_rate", 1.0)) != 0.0:
        errors.append("negative_transfer_rate")
    if float(artifact.get("stale_evidence_rejection_rate", 0.0)) != 1.0:
        errors.append("stale_evidence_rejection_rate")
    if len(_string_list(artifact.get("memory_state_hashes"))) != int(
        artifact.get("num_stream_tasks", -1)
    ):
        errors.append("memory_state_hashes")
    expected_ready = bool(
        tests_run
        and expected_delta > 0.0
        and graph_memory > no_memory
        and artifact.get("model_weights_mutated") is False
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and float(artifact.get("negative_transfer_rate", 1.0)) == 0.0
        and float(artifact.get("stale_evidence_rejection_rate", 0.0)) == 1.0
    )
    if artifact.get("csl_experience_graph_ready") is not expected_ready:
        errors.append("csl_experience_graph_ready")
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    if artifact.get("research_conductor_modified") is not False:
        errors.append("scripts/research_conductor.py")
    checksum = artifact.get("reproducibility_checksum")
    if checksum and checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def memory_state_hash(memory_graph: Mapping[str, Any]) -> str:
    """Hash graph state while excluding self-referential hash fields."""

    payload = {
        key: value
        for key, value in memory_graph.items()
        if key not in {"state_hash", "state_hashes"}
    }
    return "sha256:" + _sha256_json(payload)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the deliverable with its checksum field removed."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return "sha256:" + _sha256_json(payload)


def source_file_checksums(root: Path) -> JsonDict:
    """Record source hashes so the replay artifact names the code it used."""

    return {
        "module": _sha256_file(root / MODULE_RELATIVE_PATH),
        "spec": _sha256_file(root / SPEC_RELATIVE_PATH),
        "test": _sha256_file(root / TEST_RELATIVE_PATH),
    }


def _task(
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
    lesson_node: Mapping[str, Any],
) -> JsonDict:
    return {
        "task_id": task_id,
        "split": split,
        "state": {
            "domain": domain,
            "locality": locality,
            "version": version,
            "tags": list(tags),
        },
        "domain": domain,
        "locality": locality,
        "version": version,
        "tags": list(tags),
        "candidates": list(candidates),
        "no_memory_action": no_memory_action,
        "expected_action": expected_action,
        "lesson_node": dict(lesson_node),
    }


def _lesson_node(
    node_id: str,
    node_type: str,
    action: str,
    *,
    domain: str,
    locality: str,
    tags: Sequence[str],
    trust_score: float,
    description: str,
) -> JsonDict:
    return _memory_node(
        node_id,
        node_type,
        action,
        domain=domain,
        locality=locality,
        tags=tags,
        trust_score=trust_score,
        version=1,
        success_count=0,
        description=description,
    )


def _memory_node(
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
    failure_count: int = 0,
    expires_before_version: int | None = None,
    conflict_group: str | None = None,
    negative_transfer_domains: Sequence[str] = (),
) -> JsonDict:
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
        "failure_count": failure_count,
        "evidence_status": "active",
        "expires_before_version": expires_before_version,
        "conflict_group": conflict_group,
        "negative_transfer_domains": list(negative_transfer_domains),
        "description": description,
    }


def _candidate_relevant(task: Mapping[str, Any], node: Mapping[str, Any]) -> bool:
    task_tags = set(_string_list(task.get("tags")))
    node_tags = set(_string_list(node.get("tags")))
    return bool(
        task_tags & node_tags
        or task.get("domain") == node.get("domain")
        or task.get("locality") == node.get("locality")
    )


def _task_state(task: Mapping[str, Any]) -> JsonDict:
    return {
        "domain": task["domain"],
        "locality": task["locality"],
        "version": task["version"],
        "tags": list(task["tags"]),
        "candidate_actions": list(task["candidates"]),
    }


def _control_rates(retrieval_decisions: Sequence[Mapping[str, Any]]) -> JsonDict:
    stale_seen = 0
    stale_rejected = 0
    transfer_seen = 0
    transfer_accepted = 0
    for decision in retrieval_decisions:
        rejected = _mapping(decision.get("rejected_node_ids_by_reason"))
        stale_rejected += len(_string_list(rejected.get("stale_evidence")))
        stale_seen += len(_string_list(rejected.get("stale_evidence")))
        transfer_seen += len(_string_list(rejected.get("negative_transfer")))
        for node in _list_of_mappings(decision.get("accepted_nodes")):
            if node.get("negative_transfer_domains"):
                transfer_accepted += 1
                transfer_seen += 1
    return {
        "negative_transfer_rate": _rate(transfer_accepted, transfer_seen),
        "stale_evidence_rejection_rate": _rate(stale_rejected, stale_seen),
        "control_counts": {
            "stale_candidates_seen": stale_seen,
            "stale_candidates_rejected": stale_rejected,
            "negative_transfer_candidates_seen": transfer_seen,
            "negative_transfer_candidates_accepted": transfer_accepted,
        },
    }


def _rejected_by_reason(rejected: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_reason: dict[str, list[str]] = {}
    for row in rejected:
        by_reason.setdefault(str(row["rejection_reason"]), []).append(str(row["node_id"]))
    return {reason: sorted(ids) for reason, ids in sorted(by_reason.items())}


def _empty_next_decision() -> JsonDict:
    return {
        "task_id": None,
        "ranked_node_ids": [],
        "accepted_node_ids": [],
        "rejected_node_ids_by_reason": {},
        "selected_node_id": None,
        "selected_action": None,
        "utility_scores": {},
        "accepted_nodes": [],
        "rejected_nodes": [],
    }


def _normalise_tests_run(tests_run: Sequence[str | Mapping[str, Any]]) -> list[JsonDict]:
    rows = []
    for row in tests_run:
        if isinstance(row, str):
            rows.append({"command": row, "outcome": "passed"})
        else:
            rows.append(dict(row))
    return rows


def _resolve_output_path(root: Path, path: Path | str) -> Path:
    output = Path(path)
    if output.is_absolute():
        return output
    return root / output


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        _json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _list_of_mappings(value: Any) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _rate(numerator: int, denominator: int) -> float:
    return _round(numerator / denominator) if denominator else 0.0


def _round(value: float) -> float:
    return round(float(value), 6)


def _honest_verdict(ready: bool, heldout_delta: float) -> str:
    if ready:
        return f"complete: experience_graph_replay_ready_delta_{heldout_delta:+.6f}"
    return "blocked: experience_graph_replay_not_ready"
