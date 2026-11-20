"""REQ-REPORT-9139-V779: schedule unchanged obligations with explicit custody.

A complete cost graph can prove infeasibility without running dependent science.
Partition receipts belong to one invocation and never replace historical commands.
"""

from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import signal
import time
from typing import Any

from carnot.reporting import receipt_qualification_9125 as prior
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.natural_label_eligibility_8975 import ref
from carnot.reporting.spline_basis_methods_8986 import gate

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_9139_v779_validation_schedule"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_validation_schedule_9139.py"
CONTRACT = "results/experiment_9138_v779_contract.json"
PREVIOUS = "results/experiment_9125_v778_receipt_qualification.json"
PINS = {
    CONTRACT: "sha256:65528ba505f6ab790c80ef7bba257c1bbe30e2a63502cad0be0d247f88125169",
    PREVIOUS: "sha256:77b9e1c0d194e269c5c3c6617a68f61e75624e6e0376ec2717a06291c857309d",
}
HISTORICAL_READER = ROOT / (
    "results/raw/experiment_9137_v778_capstone/invocations/1794824820951431038/retained/"
    "8e3c488160956714a0a77136c619b298cc210835382d4853faddbaf120920041/"
    "forward_evidence_runner_9111.py"
)
MODEL_SPECS: list[Json] = []
SEED = 7799139


def progress(phase: str) -> None:
    """Flush phase changes while the shared supervisor reports child heartbeats."""
    print("[exp9139] phase=" + phase, flush=True)


def authenticate(root: Path, raw: Path) -> Json:
    """Check immutable upstream gates before reading dependent historical work."""
    work: Json = dict(checks=[], references=[], receipts=[], code_config_hashes=[])
    for name in [CONTRACT, PREVIOUS]:
        path = root / name
        observed = ref(path)["sha256"] or "missing_file"
        work["checks"].append(gate(path, "sha256", PINS[name], observed, "source_bytes", name))
        if not work["checks"][-1]["passed"]:
            return work
        target = raw / "inputs" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        value = json.loads(prior.old.bound(dict(path=str(target), sha256=PINS[name])).read_bytes())
        work["references"].append(
            dict(ref(target), producer_path=str(path), imported_fields=list(value))
        )
        if name == CONTRACT:
            observed = value.get("contract_ready_score", "missing_field")
            work["checks"].append(
                gate(path, "contract_ready_score", 1, observed, "contract", "exp9138-contract")
            )
            if not work["checks"][-1]["passed"]:
                return work
        else:
            work["historical"] = value
    h = work["historical"]["replay_input"]["work"]
    references = h["original_assertion_references"] + h["previous"]["code_config_hashes"]
    references += h.get("event_references", [])
    for receipt in h["receipts"] + h["original_validation_receipts"]:
        references += [
            dict(path=receipt[s + "_path"], sha256=receipt[s + "_sha256"])
            for s in ["stdout", "stderr"]
        ]
    references += [h["historical_work"]["diagnostic_input_reference"]]
    work["original_source_pins"] = references
    for reference in references:
        source = Path(reference["path"])
        if source.name == "forward_evidence_runner_9111.py":
            source = HISTORICAL_READER
        elif (
            source == ROOT / "tests/python/test_adversarial_verify_arc_no_llm_floor_2026_07_30.py"
            and reference["sha256"]
            == "sha256:30dde5c4fd63097d24010d781f45441f9fca9db4cb6d3c126882a18d94d5d845"
        ):
            source = (
                ROOT
                / "results/raw/experiment_9256_v788_corpus_prefilter/baseline/"
                "test_adversarial_verify_arc_no_llm_floor_2026_07_30.py"
            )
        observed = ref(source)["sha256"] or "missing_file"
        work["checks"].append(
            gate(source, "sha256", reference["sha256"], observed, "original_bytes", "exp9125")
        )
        if not work["checks"][-1]["passed"]:
            continue
        target = raw / "inputs" / reference["sha256"].split(":")[1] / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            prior.old.bound(dict(path=str(source), sha256=reference["sha256"])).read_bytes()
        )
        work["references"].append(
            dict(ref(target), producer_path=reference["path"], imported_fields=["exact_bytes"])
        )
    if all(c["passed"] for c in work["checks"]):
        events = [
            json.loads(line)
            for r in h["event_references"]
            for line in prior.old.bound(r).read_text().splitlines()
        ]
        if events != h["events"] or len(h["nodes"]) != 621:
            raise ValueError("historical_node_evidence")
        diagnostic = json.loads(
            prior.old.bound(h["historical_work"]["diagnostic_input_reference"]).read_bytes()
        )
        semantic = prior.old.semantic_control_rows()
        structural = prior.old.structural_controls(diagnostic)
        if (
            semantic != h["original_semantic_controls"]
            or structural != h["adversarial_control_rows"]
        ):
            raise ValueError("historical_control_drift")
        work["semantic_controls"] = semantic
        work["count_controls"] = structural
        work["schedule"] = schedule(h)
        work["historical_reader_reference"] = ref(HISTORICAL_READER)
    return work


def schedule(historical: Json) -> Json:
    """Charge every node and required check once, with unknown costs kept as bounds."""
    nodes = historical["nodes"]
    groups = prior.partitions(nodes)
    rows = prior.node_rows(nodes, historical["events"])
    measured = [
        dict(
            node_id=r["unit_id"],
            measured_seconds=r["duration_s"],
            bound_seconds=max(30.0, 2 * r["duration_s"]),
        )
        for r in rows
        if r["status"] == "completed"
    ]
    known = {r["node_id"]: r["bound_seconds"] for r in measured}
    unknown = [
        dict(node_id=n, bound_seconds=30, reason="no_complete_phase_measurement")
        for n in nodes
        if n not in known
    ]
    graph = [
        dict(
            partition_id=str(i),
            node_ids=g,
            fixture_scope=g[0].split("::")[0],
            node_seconds={n: known.get(n, 30) for n in g},
            startup_bound_s=90,
            budget_s=90 + sum(known.get(n, 30) for n in g),
        )
        for i, g in enumerate(groups)
    ]
    graph.sort(key=lambda p: (any(n not in known for n in p["node_ids"]), p["budget_s"]))
    commands = historical["required_command_union"]
    remaining = [c for c in commands if c["name"] not in {"owned_tests", "direct_consumers"}]
    checks_s = sum(c.get("deadline", 240) for c in remaining)
    projected = sum(p["budget_s"] for p in graph) + checks_s
    return dict(
        authority="carnot.validation_schedule.v2",
        original_node_manifest=nodes,
        original_assertion_hashes=historical["original_assertion_references"],
        original_command_contract=commands,
        verification_graph=remaining,
        execution_graph=graph,
        measured_cost_rows=measured,
        unknown_cost_rows=unknown,
        exclusions=[],
        units="wall_seconds",
        ordering="measured cost across whole module lifetimes",
        budget_equation=dict(
            measurement_s=3300,
            verification_s=1200,
            publication_margin_s=300,
            hard_cap_s=4800,
            verification_reservations=1,
            node_and_startup_bound_s=sum(p["budget_s"] for p in graph),
            remaining_command_bound_s=checks_s,
            projected_total_s=projected,
            feasible=projected <= 3300,
        ),
        historical_budget_control=dict(measurement_s=2100, reserve_s=1800, available_s=2100 - 1800),
    )


def reduce_nodes(nodes: list[str], events: list[Json]) -> list[Json]:
    """Reject repeated phase evidence before reducing the original assertion outcomes."""
    identities = [(e["nodeid"], e["when"]) for e in events]
    if len(set(identities)) != len(identities) or any(e["nodeid"] not in nodes for e in events):
        raise ValueError("phase_evidence")
    return prior.node_rows(nodes, events)


def expire(signum: int, frame: Any) -> None:
    """A timed-out node terminates its module; unfinished fixtures cannot qualify."""
    os._exit(124)


def pytest_runtest_logstart(nodeid: str, location: Any) -> None:
    """Keep the original event schema and bound each node within its module lifetime."""
    prior.pytest_runtest_logstart(nodeid, location)
    signal.signal(signal.SIGALRM, expire)
    limits = json.loads(os.environ.get("CARNOT_9139_NODE_LIMITS", "{}"))
    signal.setitimer(
        signal.ITIMER_REAL,
        float(limits.get(nodeid, os.environ.get("CARNOT_9139_NODE_SECONDS", "30"))),
    )


def pytest_runtest_logreport(report: Any) -> None:
    """Record actual setup, assertion and teardown outcomes without inferring success."""
    prior.pytest_runtest_logreport(report)


def pytest_runtest_logfinish(nodeid: str, location: Any) -> None:
    """Cancel the node timer only after teardown has finished."""
    signal.setitimer(signal.ITIMER_REAL, 0)


def partition_argv(part: Json, raw: Path, node_seconds: int | None = None) -> list[str]:
    """Bind the exact command and node limits so receipt edits cannot widen scope."""
    return [
        "env",
        "CARNOT_9125_NODE_LOG=" + str(raw / "events.jsonl"),
        "CARNOT_9139_NODE_LIMITS="
        + json.dumps({} if node_seconds else part["node_seconds"], sort_keys=True),
        "CARNOT_9139_NODE_SECONDS=" + str(node_seconds or 30),
        str(ROOT / ".venv/bin/python"),
        "-m",
        "pytest",
        "-n0",
        "-o",
        "addopts=",
        "--no-cov",
        "-p",
        "carnot.reporting.validation_schedule_9139",
        "-q",
        *part["node_ids"],
    ]


def execute(
    plan: Json, raw: Path, invocation: str, hashes: list[Json], *, node_seconds: int | None = None
) -> Json:
    """Execute a qualification subset once; a failed module is retained, never retried."""
    if not plan["budget_equation"]["feasible"]:
        raise ValueError("schedule_infeasible")
    raw.mkdir(parents=True, exist_ok=True)
    bundle: Json = dict(
        invocation_id=invocation,
        started_monotonic_ns=time.monotonic_ns(),
        schedule=plan,
        code_config_hashes=hashes,
        partitions=[],
        checks=[],
        root=str(raw),
        node_seconds=node_seconds,
    )
    stop = time.monotonic() + 3300
    for part in plan["execution_graph"]:
        progress("partition_" + part["partition_id"] + "_before")
        if time.monotonic() >= stop:
            break
        directory = raw / part["partition_id"]
        argv = partition_argv(part, directory, node_seconds)
        receipt = prior.run_check(
            "partition_" + part["partition_id"],
            argv,
            directory,
            min(part["budget_s"], stop - time.monotonic()),
        )
        bundle["partitions"].append(
            dict(
                partition_id=part["partition_id"],
                node_ids=part["node_ids"],
                receipt=receipt,
                receipt_reference=ref(
                    directory / receipt["name"] / (receipt["name"] + ".receipt.json")
                ),
                events_reference=ref(directory / "events.jsonl"),
            )
        )
        atomic_json(raw / "bundle.json", bundle)
        progress("partition_" + part["partition_id"] + "_after")
    for spec in plan["verification_graph"]:
        if time.monotonic() >= stop:
            break
        receipt = prior.run_check(
            spec["name"],
            spec["argv"],
            raw / "checks",
            min(spec.get("deadline", 240), stop - time.monotonic()),
            expected=spec.get("expected", 0),
        )
        bundle["checks"].append(
            dict(
                receipt=receipt,
                receipt_reference=ref(
                    raw / "checks" / spec["name"] / (spec["name"] + ".receipt.json")
                ),
            )
        )
    bundle["ended_monotonic_ns"] = time.monotonic_ns()
    atomic_json(raw / "bundle.json", bundle)
    return bundle


def consume(bundle: Json, invocation: str) -> Json:
    """Authenticate same-invocation results without manufacturing an old command receipt."""
    if bundle["invocation_id"] != invocation:
        raise ValueError("foreign_invocation")
    plan = bundle["schedule"]
    for reference in plan["original_assertion_hashes"] + bundle["code_config_hashes"]:
        prior.old.bound(reference)
    expected = plan["execution_graph"]
    parts = bundle["partitions"]
    if [p["partition_id"] for p in parts] != [p["partition_id"] for p in expected]:
        raise ValueError("partition_union")
    all_events = []
    nodes = []
    for part, spec in zip(parts, expected, strict=True):
        receipt = part["receipt"]
        actual = json.loads(prior.old.bound(part["receipt_reference"]).read_bytes())
        if (
            receipt != actual
            or receipt["exit_code"] != 0
            or not receipt["normal_exit"]
            or receipt["timed_out"]
        ):
            raise ValueError("partition_exit")
        directory = Path(bundle["root"]) / spec["partition_id"]
        if (
            part["node_ids"] != spec["node_ids"]
            or receipt["argv"] != partition_argv(spec, directory, bundle["node_seconds"])
            or not bundle["started_monotonic_ns"]
            <= receipt["started_monotonic_ns"]
            <= bundle["ended_monotonic_ns"]
        ):
            raise ValueError("partition_identity")
        for stream in ["stdout", "stderr"]:
            prior.old.bound(
                dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
            )
        events = [
            json.loads(line)
            for line in prior.old.bound(part["events_reference"]).read_text().splitlines()
        ]
        rows = reduce_nodes(spec["node_ids"], events)
        if not all(r["status"] == "completed" for r in rows):
            raise ValueError("unfinished_partition")
        nodes.extend(spec["node_ids"])
        all_events.extend(events)
    if Counter(nodes) != Counter(plan["original_node_manifest"]) or len(set(nodes)) != len(nodes):
        raise ValueError("obligation_union")
    checks = bundle["checks"]
    if len(checks) != len(plan["verification_graph"]):
        raise ValueError("command_union")
    for check, spec in zip(checks, plan["verification_graph"], strict=True):
        receipt = check["receipt"]
        if (
            receipt != json.loads(prior.old.bound(check["receipt_reference"]).read_bytes())
            or receipt["argv"] != spec["argv"]
            or receipt["name"] != spec["name"]
            or receipt["exit_code"] != spec.get("expected", 0)
            or not receipt["normal_exit"]
            or receipt["timed_out"]
            or not bundle["started_monotonic_ns"]
            <= receipt["started_monotonic_ns"]
            <= bundle["ended_monotonic_ns"]
        ):
            raise ValueError("command_evidence")
        for stream in ["stdout", "stderr"]:
            prior.old.bound(
                dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
            )
    rows = reduce_nodes(plan["original_node_manifest"], all_events)
    return dict(
        authority=plan["authority"],
        invocation_id=invocation,
        rows=rows,
        completed_count=len(rows),
        duplicate_execution_count=0,
        rerun_commands=[],
        forward_evidence_ready_score=0,
        model_work_authorized=False,
    )


def orchestration(path: Path, invocation: str) -> Json:
    """The current runner consumes a bound bundle while historical replay stays version 1."""
    bundle = json.loads(path.read_bytes())
    result = consume(bundle, invocation)
    atomic_json(path.parent / "consumed.json", result)
    return result
