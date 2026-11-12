"""REQ-REPORT-9055-CONTINUATION: extend the fixed proposal sequence without new science."""

from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

import networkx as nx

from carnot.reporting import fresh_seal_methods_9042 as old
from carnot.reporting import v773_branch_evidence as history
from carnot.reporting import relational_methods_8959 as g
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.invocation_provenance_9000 import reference, retain

Json = dict[str, Any]
ROOT, SEED, TARGET = old.ROOT, old.SEED, 1024
NAME, TASK = "experiment_9055_v773_prospective_roster", "exp9055-prospective-roster"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_prospective_roster_9055.py"
OWNED = [
    "python/carnot/reporting/prospective_roster_9055.py",
    "python/carnot/reporting/prospective_roster_runner_9055.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
ORIGINAL_CUSTODY = "results/raw/experiment_9042_v772_fresh_seal_methods/invocations/1794476323104834541/production_custody/checkpoint.json"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual counts expose stalled work without extending its deadline."""
    print(f"[exp9055] phase={phase} completed={completed} pending={pending}", flush=True)


def recount(prefix: Json) -> Json:
    """Recount primitive rows rather than importing the old headline counts."""
    rows = prefix["proposal_rows"]
    return dict(
        accepted=sum(r["status"] == "accepted" for r in rows),
        rejection_counts=dict(Counter(r["reason"] for r in rows if r["reason"])),
        block_yields=[
            sum(r["status"] == "accepted" for r in rows[i : i + 5000])
            for i in range(0, len(rows), 5000)
        ],
    )


def fresh(graph: Json, seed: int, index: int) -> Json:
    """Keep the original presentation, namespace and seeded role assignment."""
    with patch.object(g, "SEED", SEED):
        result = g.fresh_graph(graph, seed, index)
    uid = f"v772-graph-{index:04d}-k2"
    result.update(graph_id=uid.removesuffix("-k2"), unit_id=uid)
    result["questions"][0]["unit_id"] = uid
    return dict(result)


def proposal(
    counter: int, index: int, buckets: dict[Any, list[Any]], seeds: set[int]
) -> tuple[Json, Json | None]:
    """The original predicate checks every prior proposal, including rejected graphs."""
    seed = 2**52 + SEED * 10000000 + counter
    graph = g.source.candidate(seed)
    topology = g.nx_graph(graph)
    key = g.topology_key(topology)
    collision = seed in seeds
    for previous in buckets.get(key, []):
        prior = g.source.candidate(previous) if isinstance(previous, int) else previous
        if nx.is_isomorphic(topology, g.nx_graph(prior)):
            collision = True
            break
    answers = [
        g.source.distance_nodes(graph["nodes"], graph["edges"], graph["start"], k) for k in (2, 3)
    ]
    reason = (
        "isomorphic_relabeling"
        if collision
        else "unmatched_cardinality"
        if not answers[0] or len(answers[0]) != len(answers[1])
        else None
    )
    row = dict(
        proposal_counter=counter,
        seed=seed,
        graph_hash=g.source.graph_hash(graph),
        reason=reason,
        roster_index=index,
        status="rejected" if reason else "accepted",
    )
    seeds.add(seed)
    buckets.setdefault(key, []).append(seed)
    return row, None if reason else fresh(graph, seed, index)


def construct(
    prefix: Json,
    inventory: Json,
    raw: Path,
    *,
    proposal_limit: int = 100000,
    seconds: float = 300,
    interrupt_after: int | None = None,
) -> Json:
    """Replay durable proposals before continuing; a killed process never changes its prefix."""
    started = heartbeat = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    checkpoint = raw / "checkpoint.json"
    binding = dict(prefix_sha256=canonical_hash(prefix), inventory_sha256=canonical_hash(inventory))
    saved_checkpoint = json.loads(checkpoint.read_bytes()) if checkpoint.is_file() else {}
    if saved_checkpoint and any(saved_checkpoint.get(k) != v for k, v in binding.items()):
        raise ValueError("checkpoint_authority")
    elapsed = (
        saved_checkpoint["construction_duration_s"]
        if saved_checkpoint
        else prefix["construction_duration_s"]
    )
    buckets: dict[Any, list[Any]] = {}
    seeds = set(inventory["old_seeds"])
    originals = [*inventory["old_graphs"], *sorted(seeds)]
    progress("history_index_before", 0, len(originals))
    for index, original in enumerate(originals):
        if time.monotonic() - started + elapsed >= seconds:
            raise ValueError("incomplete_history")
        graph = g.source.candidate(original) if isinstance(original, int) else original
        buckets.setdefault(g.topology_key(g.nx_graph(graph)), []).append(original)
        if time.monotonic() - heartbeat >= 30:
            progress("history_index", index + 1, len(originals) - index - 1)
            heartbeat = time.monotonic()
    progress("history_index_after", len(originals), 0)
    rows: list[Json] = []
    roster: list[Json] = []
    for expected in prefix["proposal_rows"]:
        if time.monotonic() - heartbeat >= 30:
            progress("original_ledger_replay", len(rows), len(prefix["proposal_rows"]) - len(rows))
            heartbeat = time.monotonic()
        row, graph = proposal(len(rows), len(roster), buckets, seeds)
        if row != expected:
            raise ValueError(
                "changed_prefix_exposure"
                if row["reason"] == "isomorphic_relabeling"
                else "original_ledger"
            )
        rows.append(row)
        if graph is not None:
            roster.append(graph)
    if roster != prefix["graph_manifest"] or len(rows) != prefix["proposal_count"]:
        raise ValueError("original_prefix")
    later = [*inventory.get("additional_graphs", []), *inventory.get("additional_seeds", [])]
    accepted_buckets: dict[Any, list[Any]] = {}
    for accepted in roster:
        topology = g.nx_graph(accepted)
        accepted_buckets.setdefault(g.topology_key(topology), []).append(topology)
    for later_index, original in enumerate(later):
        graph = g.source.candidate(original) if isinstance(original, int) else original
        topology = g.nx_graph(graph)
        key = g.topology_key(topology)
        for previous in accepted_buckets.get(key, []):
            if nx.is_isomorphic(topology, previous):
                raise ValueError("changed_prefix_exposure")
        buckets.setdefault(key, []).append(original)
        if isinstance(original, int):
            seeds.add(original)
        if time.monotonic() - heartbeat >= 30:
            progress("later_history", later_index + 1, len(later) - later_index - 1)
            heartbeat = time.monotonic()
    originals.extend(later)
    ledger = raw / "continuation.jsonl"
    saved = ledger.read_bytes().splitlines() if ledger.exists() else []
    for line in saved:
        if time.monotonic() - heartbeat >= 30:
            progress(
                "continuation_replay",
                len(rows) - len(prefix["proposal_rows"]),
                len(saved) - (len(rows) - len(prefix["proposal_rows"])),
            )
            heartbeat = time.monotonic()
        record = json.loads(line)
        row, graph = proposal(len(rows), len(roster), buckets, seeds)
        if record != dict(row, graph=graph):
            raise ValueError("continuation_ledger")
        rows.append(row)
        if graph is not None:
            roster.append(graph)
    progress("prefix_replay_after", len(rows), max(0, proposal_limit - len(rows)))
    reason = None
    with ledger.open("a", buffering=1) as stream:
        while len(roster) < TARGET and len(rows) < min(proposal_limit, 100000):
            if time.monotonic() - started + elapsed >= seconds:
                reason = "construction_time_exhausted"
                break
            row, graph = proposal(len(rows), len(roster), buckets, seeds)
            stream.write(json.dumps(dict(row, graph=graph), sort_keys=True) + "\n")
            rows.append(row)
            if graph is not None:
                roster.append(graph)
            if len(rows) % 1000 == 0 or interrupt_after == len(rows):
                os.fsync(stream.fileno())
                atomic_json(
                    checkpoint,
                    dict(
                        proposal_count=len(rows),
                        **binding,
                        construction_duration_s=elapsed + time.monotonic() - started,
                    ),
                )
            if interrupt_after == len(rows):
                os._exit(73)
            if time.monotonic() - heartbeat >= 30:
                progress("graph_proposals", len(rows), 100000 - len(rows))
                heartbeat = time.monotonic()
    complete = len(roster) == TARGET
    duration = elapsed + time.monotonic() - started
    result = dict(
        graph_manifest=roster,
        proposal_rows=rows,
        proposal_count=len(rows),
        complete=complete,
        terminal_reason=None if complete else reason or "proposal_budget_exhausted",
        history_indexed=len(originals),
        history_pending=0,
        construction_duration_s=duration,
    )
    atomic_json(
        checkpoint,
        dict(
            proposal_count=len(rows),
            **binding,
            construction_duration_s=duration,
        ),
    )
    atomic_json(raw / "construction.json", result)
    progress("construction_after", len(roster), TARGET - len(roster))
    return result


def authenticate(root: Path, raw: Path) -> Json:
    """Independent primary pins bind original ledger bytes and every frozen scientific input."""
    evidence = old.authenticate(root, raw)
    pins = {
        ORIGINAL_CUSTODY: "sha256:ebd3524db0d662059a2f3ac288d991d5e6a07a32aa69fdc95fe1a7a00eb40d05",
        history.ROSTER: history.PINS[history.ROSTER],
        history.CAPSTONE: history.PINS[history.CAPSTONE],
        "results/experiment_9054_v773_branch_contract.json": "sha256:9d2757fa9ba48ed23457d667cb92bc88187b1466fbdb948caab4d84e8155fe74",
    }
    values: Json = {}
    for name, pin in pins.items():
        path = root / name
        observed = reference(path)["sha256"] or "missing_file"
        evidence["checks"].append(
            old.gate(path, "sha256", pin, observed, "continuation_input_bytes")
        )
        if path.is_file():
            evidence["references"].append(retain(path, raw))
        if observed == pin:
            values[name] = json.loads(path.read_bytes())
    collections = [
        reference(path)
        for number in (9045, 9046, 9058, 9059)
        for folder in sorted((root / "results/raw").glob(f"experiment_{number}_*"))
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    ]
    evidence["checks"].append(
        old.gate(
            root / "results/raw",
            "later_collection_files",
            [],
            collections,
            "no_collection_since_prefix",
        )
    )
    evidence["failures"] = [r for r in evidence["checks"] if not r["passed"]]
    if evidence["failures"]:
        return evidence
    custody = values[ORIGINAL_CUSTODY]
    evidence["checks"].append(
        old.gate(
            root / ORIGINAL_CUSTODY,
            "original_transport_state",
            dict(starts=[], transport={}, status={"pending": 414}),
            dict(
                starts=custody["starts"],
                transport=custody["transport"],
                status=dict(Counter(custody["status"].values())),
            ),
            "original_no_responses",
        )
    )
    primary = values[history.ROSTER]
    for ref in primary["raw_shard_hashes"] + primary["code_config_hashes"]:
        path = root / Path(ref["path"]).relative_to(ROOT)
        observed = reference(path)["sha256"] or "missing_file"
        evidence["checks"].append(
            old.gate(path, "sha256", ref["sha256"], observed, "original_primitive_bytes")
        )
        if path.is_file():
            evidence["references"].append(retain(path, raw))
    branch = values["results/experiment_9054_v773_branch_contract.json"]
    for field, expected in [("frozen_methods_ready_score", 1), ("required_checks_passed", True)]:
        evidence["checks"].append(
            old.gate(
                root / "results/experiment_9054_v773_branch_contract.json",
                field,
                expected,
                branch.get(field, "missing_field"),
                "continuation_gate",
            )
        )
    evidence["checks"].append(
        old.gate(
            root / history.ROSTER,
            "model_invocation_counts",
            ZERO_INVOCATION_COUNTS,
            primary.get("model_invocation_counts", "missing_field"),
            "original_no_model",
        )
    )
    evidence["failures"] = [r for r in evidence["checks"] if not r["passed"]]
    if not evidence["failures"]:
        refs = {Path(r["path"]).name: r for r in primary["raw_shard_hashes"]}
        evidence["prefix"] = json.loads(
            (root / Path(refs["construction.json"]["path"]).relative_to(ROOT)).read_bytes()
        )
        evidence["original_manifest"] = json.loads(
            (root / Path(refs["fresh_role_manifest.json"]["path"]).relative_to(ROOT)).read_bytes()
        )
        evidence["original_inventory"] = json.loads(
            (root / Path(refs["historical_inventory.json"]["path"]).relative_to(ROOT)).read_bytes()
        )
        expected_recount = dict(
            accepted=414,
            rejection_counts=dict(unmatched_cardinality=19586),
            block_yields=[112, 102, 107, 93],
        )
        evidence["checks"].append(
            old.gate(
                root / history.ROSTER,
                "primitive_recount",
                expected_recount,
                recount(evidence["prefix"]),
                "prefix_recount",
            )
        )
    evidence["failures"] = [r for r in evidence["checks"] if not r["passed"]]
    return evidence


def exposure(root: Path, evidence: Json, raw: Path) -> Json:
    """All later public source identities enter authority; response-bearing prefix reuse blocks."""
    inventory = deepcopy(evidence["original_inventory"])
    prefix_hashes = {g["graph_hash"] for g in evidence["prefix"]["graph_manifest"]}
    prefix_ids = {g["unit_id"] for g in evidence["prefix"]["graph_manifest"]}
    rows: list[Json] = []
    exposed: set[str] = set()
    graphs: dict[str, Json] = {}
    seeds = set(inventory["old_seeds"])

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            digest = value.get("graph_hash")
            if {"nodes", "edges", "start"} <= value.keys():
                digest = g.source.graph_hash(value)
                if digest not in prefix_hashes:
                    graphs[digest] = {k: value[k] for k in ("nodes", "edges", "start")}
            if digest and isinstance(value.get("seed"), int) and digest not in prefix_hashes:
                seeds.add(value["seed"])
            if (digest in prefix_hashes or value.get("unit_id") in prefix_ids) and any(
                k in value
                for k in (
                    "response",
                    "response_sha256",
                    "response_reference",
                    "response_text",
                    "correct",
                    "error",
                )
            ):
                exposed.add(str(digest or value["unit_id"]))
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    paths = [
        p
        for p in sorted((root / "results").glob("experiment_*.json"))
        if p.stem.split("_")[1].isdigit()
        and 8988 <= int(p.stem.split("_")[1]) <= 9054
        and int(p.stem.split("_")[1]) != 9042
    ]
    for index, path in enumerate(paths):
        value = json.loads(path.read_bytes())
        visit(value)
        refs = [
            r
            for key, r in value.items()
            if "manifest" in key and isinstance(r, dict) and "path" in r and "sha256" in r
        ]
        for ref in refs:
            target = root / Path(ref["path"]).relative_to(ROOT)
            if reference(target)["sha256"] != ref["sha256"]:
                raise ValueError("later_source_bytes")
            visit(json.loads(target.read_bytes()))
            evidence["references"].append(retain(target, raw))
        evidence["references"].append(retain(path, raw))
        rows.append(dict(source=path.name, **reference(path), eligible_as_new_test=False))
        if index % 10 == 0:
            progress("later_authority", index + 1, len(paths) - index - 1)
    original_hashes = {g.source.graph_hash(graph) for graph in inventory["old_graphs"]}
    inventory["additional_graphs"] = [
        graph for digest, graph in graphs.items() if digest not in original_hashes
    ]
    inventory["additional_seeds"] = sorted(seeds - set(inventory["old_seeds"]))
    return dict(inventory=inventory, rows=rows, exposed=sorted(exposed))
