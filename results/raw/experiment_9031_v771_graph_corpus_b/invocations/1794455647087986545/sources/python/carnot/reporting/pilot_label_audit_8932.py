"""REQ-REPORT-8932: distinguish output format failures from graph reasoning errors."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from carnot.reporting import bfs_error_pilot_8918 as pilot
from carnot.reporting import graph_source_methods_8931 as methods
from carnot.reporting.graph_source_8931 import category
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file

Json = dict[str, Any]
ROOT = pilot.ROOT
NAME = "experiment_8932_v764_pilot_label_audit"
TASK = "exp8932-pilot-label-audit"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/pilot_label_audit_8932.py",
    "python/carnot/reporting/pilot_label_audit_runner_8932.py",
    CLI,
]
TEST = "tests/python/test_pilot_label_audit_8932.py"
NOTE = "docs/research-notes/label-provenance-audit-step1-2026-11-07.md"
CATEGORIES = ("malformed", "out_of_graph", "valid_wrong_distance", "correct")


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries so supervision sees completed and outstanding work."""
    print(f"[exp8932] phase={phase} completed={completed} pending={pending}", flush=True)


def distances(graph: Json) -> dict[str, int | None]:
    """Relax every undirected edge independently of the original queue-based BFS."""
    bound = len(graph["nodes"])
    lengths = dict.fromkeys(graph["nodes"], bound)
    lengths[graph["start"]] = 0
    for _ in range(bound - 1):
        previous = dict(lengths)
        for a, b in graph["edges"]:
            lengths[a] = min(lengths[a], previous[b] + 1)
            lengths[b] = min(lengths[b], previous[a] + 1)
    return {node: length if length < bound else None for node, length in lengths.items()}


def rate(numerator: int, denominator: int) -> Json:
    """Keep an empty semantic denominator undefined rather than claiming zero errors."""
    if not denominator:
        return dict(numerator=numerator, denominator=0, value=None, wilson_95=None)
    p = numerator / denominator
    z = 1.959963984540054
    scale = 1 + z * z / denominator
    center = (p + z * z / (2 * denominator)) / scale
    radius = z * (p * (1 - p) / denominator + z * z / (4 * denominator**2)) ** 0.5 / scale
    return dict(
        numerator=numerator,
        denominator=denominator,
        value=p,
        wilson_95=dict(
            method="Wilson",
            confidence=0.95,
            denominator=denominator,
            low=0.0 if numerator == 0 else max(0, center - radius),
            high=1.0 if numerator == denominator else min(1, center + radius),
        ),
    )


def authenticate(root: Path) -> Json:
    """Pin producer bytes, all raw operands, task authority and historical custody before use."""
    evidence: Json = methods.authenticate(root)
    evidence.update(pilot={}, runtime={})
    path = root / methods.PILOT
    if not evidence["failures"]:
        try:
            pilot.replay(path)
            value = json.loads(path.read_text())
            refs = value["raw_shard_hashes"] + value["source_artifact_hashes"]
            runtime_path = Path(
                next(r["path"] for r in refs if Path(r["path"]).name == "runtime.json")
            )
            runtime = json.loads(runtime_path.read_text())
            parent = runtime_path.parent
            ledger = {r["call_id"]: r for r in runtime["ledger"]}
            for response in runtime["responses"]:
                unit = response["unit_id"]
                request = json.loads((parent / "requests" / (unit + ".json")).read_text())
                raw = json.loads((parent / "responses" / (unit + ".json")).read_text())
                if (
                    raw != response
                    or canonical_hash(request) != response["request_sha256"]
                    or ledger[unit]["response_sha256"] != canonical_hash(raw["raw_response"])
                    or ledger[unit]["request_sha256"] != canonical_hash(request)
                    or ledger[unit]["status"] != "completed"
                ):
                    raise ValueError("raw_response_custody")
            if (
                runtime["counts"]["generation_calls_completed"] != 50
                or value["failed_count"] != 0
                or value["censored_count"] != 0
                or not runtime["model_receipt"]["owner"]["owned_by_task"]
            ):
                raise ValueError("historical_custody")
            evidence.update(pilot=value, runtime=runtime)
            evidence["references"].extend(refs)
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            evidence["failures"].append(
                dict(
                    check="retained_pilot_custody",
                    upstream=pilot.TASK,
                    artifact_field="retained_operands_and_model_offload",
                    expected="exact authenticated raw bytes and completed calls",
                    observed="missing_file" if isinstance(error, OSError) else "rejected_producer",
                    path=str(path),
                    hash=sha256_file(path),
                    detail=str(error),
                )
            )
    return evidence


def reduce(value: Json, runtime: Json) -> Json:
    """Preserve all originals while independently recomputing graph-level labels and rates."""
    graphs, labels = value["graph_manifest"], value["answer_rows"]
    responses = {r["unit_id"]: r for r in runtime["responses"]}
    old = {r["unit_id"]: r for r in labels}
    ids = {g["unit_id"] for g in graphs}
    if (
        len(graphs) != 50
        or len(ids) != 50
        or len({g["seed"] for g in graphs}) != 50
        or len(old) != len(labels)
        or set(old) != ids
        or len(responses) != len(runtime["responses"])
        or not set(responses) <= ids
    ):
        raise ValueError("unit_roster")
    rows, disagreements = [], []
    for graph in graphs:
        unit = graph["unit_id"]
        response = responses.get(unit, dict(status="censored", missing_reason="not_started"))
        status = response["status"]
        if status not in {"completed", "runtime_failed", "censored"}:
            raise ValueError("response_status")
        lengths = distances(graph)
        expected = sorted(n for n, d in lengths.items() if d == graph["k"])
        answer = response.get("answer", "")
        kind = category(answer, graph["nodes"], expected) if status == "completed" else None
        error = int(kind != "correct") if kind is not None else None
        parsed, reason = pilot.parse_answer(answer, graph["nodes"])
        comparisons: dict[str, tuple[Any, Any]] = dict(expected=(graph["expected"], expected))
        if status == "completed":
            comparisons.update(
                error=(old[unit]["error"], error),
                parsed_node=(old[unit]["parsed_node"], parsed),
                parser_reason=(old[unit]["parser_reason"], reason),
                raw_answer=(response["raw_response"]["choices"][0]["message"]["content"], answer),
            )
        for field, (original, recomputed) in comparisons.items():
            if original != recomputed:
                disagreements.append(
                    dict(unit_id=unit, field=field, original=original, recomputed=recomputed)
                )
        rows.append(
            dict(
                unit_id=unit,
                cluster_id=str(graph["seed"]),
                seed=graph["seed"],
                arm="retained_qwen_reasoning_off_k2",
                metric="final_answer_error",
                value=error,
                numerator=error,
                denominator=1,
                status=status,
                missing_reason=None
                if status == "completed"
                else response.get("missing_reason", "runtime_failure"),
                error_type=kind,
                answer=answer,
                parsed_node=parsed,
                shortest_distance=lengths.get(parsed),
                distances=lengths,
                expected=expected,
                output_characters=len(answer) if kind else None,
                completion_tokens=response.get("usage", {}).get("completion_tokens")
                if kind
                else None,
                original_graph=graph,
                original_response=response,
                original_label=old[unit],
            )
        )
    completed = sum(r["status"] == "completed" for r in rows)
    counts = {kind: sum(r["error_type"] == kind for r in rows) for kind in CATEGORIES}
    semantic_n = counts["valid_wrong_distance"] + counts["correct"]

    def histogram(key: str) -> dict[str, int]:
        return dict(
            sorted(Counter(str(r[key]) for r in rows if r["status"] == "completed").items())
        )

    return dict(
        rows=rows,
        error_type_counts=counts,
        error_type_rows=[
            dict(error_type=kind, **rate(counts[kind], completed)) for kind in CATEGORIES
        ],
        total_error_rate=rate(completed - counts["correct"], completed),
        semantic_error_rate=dict(
            **rate(counts["valid_wrong_distance"], semantic_n),
            definition="valid in-graph answers only; malformed and absent nodes remain in total rate",
        ),
        output_length_distribution=dict(
            characters=histogram("output_characters"),
            completion_tokens=histogram("completion_tokens"),
            denominator=completed,
            scope="exact unedited output characters and retained server completion-token usage",
        ),
        label_disagreements=disagreements,
        intended_count=50,
        completed_count=completed,
        failed_count=sum(r["status"] == "runtime_failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        independent_count=completed,
    )
