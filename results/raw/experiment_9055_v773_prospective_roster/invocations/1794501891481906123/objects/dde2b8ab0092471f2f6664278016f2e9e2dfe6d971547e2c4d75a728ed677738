"""REQ-REPORT-9042-V772: scoped validation and atomic publication without a model."""

import argparse
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
from unittest.mock import patch

import networkx as nx
import pytest

from carnot.reporting import fresh_seal_methods_9042 as d
from carnot.reporting import fresh_seal_custody_9042 as c
from carnot.reporting import invocation_provenance_9000 as provenance
from carnot.reporting import scoped_provenance_9014 as checks
from carnot.reporting import retained_shard_runner_9030 as terminal
from carnot.reporting import relational_methods_8959 as g
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v756_branch_qualification import measured, verify_measured

Json = dict[str, Any]
DIRECT = [
    *checks.DIRECT,
    "tests/python/test_retained_shard_a_9030.py",
    "tests/python/test_graph_corpus_b_9031.py",
]


@pytest.fixture(autouse=True)
def preserve_coverage_environment() -> Iterator[None]:
    """Historical in-process CLI tests must not pass a deleted coverage config to the next test."""
    before = os.environ.get("COVERAGE_RCFILE")
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("COVERAGE_RCFILE", None)
        else:
            os.environ["COVERAGE_RCFILE"] = before


def refused(fn: Any) -> bool:
    """Negative controls must observe an actual custody refusal, not a headline boolean."""
    try:
        fn()
    except (ValueError, PermissionError):
        return True
    return False


def qualify(raw: Path) -> list[Json]:
    """Real killed and resumed children establish mechanics only in private scratch."""
    private = Path(tempfile.mkdtemp(prefix="carnot-9042-custody-"))
    manifest: Json = dict(
        schema="v772.fresh-seal.v1",
        fixture_only=True,
        graph_manifest=[dict(unit_id=f"private-{i}", role="test") for i in range(3)],
        state_hashes=dict(spline="private"),
    )
    pin = canonical_hash(manifest)
    request = private / "request.json"
    atomic_json(request, dict(manifest=manifest, manifest_sha256=pin, lost_unit="private-1"))
    argv = [
        "env",
        "-u",
        "PYTHONPATH",
        "-C",
        str(private),
        sys.executable,
        "-u",
        str(d.ROOT / d.CLI),
        "--custody-worker",
        str(request),
    ]
    receipts = []
    for name, folder, extra, expected in [
        ("interrupted", "resumed", ["--interrupt"], -9),
        ("resumed", "resumed", [], 0),
        ("baseline", "baseline", [], 0),
    ]:
        receipts.append(
            child(
                name,
                [*argv, "--worker-raw", str(private / folder), *extra],
                raw / name,
                deadline=60,
                expected=expected,
            )
        )
    state = c.Checkpoint(manifest, pin, private / "resumed")
    baseline = json.loads((private / "baseline/checkpoint.json").read_bytes())
    observed = dict(
        real_interruption=receipts[0]["exit_code"] == -9 and not receipts[0]["timed_out"],
        exact_continuation=all(r["passed"] for r in receipts)
        and state.state == baseline
        and len(set(state.state["starts"])) == 3,
    )
    changed = deepcopy(manifest)
    changed["graph_manifest"][0]["unit_id"] = "altered"
    observed["altered_roster"] = refused(lambda: c.Checkpoint(changed, pin, private / "altered"))
    changed["state_hashes"]["spline"] = "altered"
    observed["changed_model_fit_hash"] = refused(
        lambda: c.Checkpoint(changed, canonical_hash(changed), private / "resumed")
    )
    observed["duplicate_first_response"] = refused(lambda: state.start("private-0"))
    observed["nested_label_logs"] = refused(
        lambda: state.finish("private-0", dict(log=dict(correct=True)))
    )
    observed["unauthorized_test_access"] = refused(
        lambda: state.labels(private / "absent-truth.json", "unavailable", "test", "collection")
    )
    observed["missing_prediction_seals"] = refused(
        lambda: state.labels(
            private / "absent-truth.json", "unavailable", "test", "final_evaluation"
        )
    )
    truth = private / "truth.json"
    atomic_json(truth, dict(labels=[0, 1]))
    truth_pin = provenance.reference(truth)["sha256"]
    atomic_json(truth, dict(labels=[1, 0]))
    observed["swapped_labels"] = refused(
        lambda: state.labels(truth, truth_pin, "calibration", "calibration")
    )
    graph = dict(nodes=["A", "B", "C"], edges=[["A", "B"], ["B", "C"]], start="A")
    renamed = dict(nodes=["X", "Y", "Z"], edges=[["X", "Y"], ["Y", "Z"]], start="X")
    observed["earlier_topology"] = nx.is_isomorphic(g.nx_graph(graph), g.nx_graph(renamed))
    rows = [
        dict(name=name, passed=bool(ok), fixture_only=True, claim_scope="private_custody_mechanics")
        for name, ok in observed.items()
    ]
    rows[0]["receipt"] = receipts[0]
    atomic_json(raw / "checkpoint_comparison.json", dict(resumed=state.state, baseline=baseline))
    atomic_json(raw / "mutation_rows.json", dict(rows=rows, receipts=receipts))
    return rows


def freeze(root: Path, raw: Path, fixture: bool) -> Json:
    """Authenticate resources and states before bounded construction or custody qualification."""
    start = time.monotonic()
    d.progress("preconditions_before", 0, len(d.PINS))
    environment = provenance.environment(raw)
    evidence = d.authenticate(root, raw)
    d.progress("preconditions_after", len(evidence["checks"]), len(evidence["failures"]))
    result: Json = dict(graph_manifest=[], complete=False, proposal_rows=[])
    inventory: Json = dict(old_graphs=[], old_seeds=[])
    if not evidence["failures"] and all(r["passed"] for r in environment["checks"]):
        inventory = d.history(evidence)
        atomic_json(raw / "state_hashes.json", evidence["state_hashes"])
        result = d.construct(inventory, raw)
    atomic_json(raw / "historical_inventory.json", inventory)
    reference = provenance.reference(raw / "historical_inventory.json")
    if evidence["failures"]:
        manifest = dict(
            schema="v772.fresh-seal.v1",
            fixture_only=fixture,
            graph_manifest=[],
            complete=False,
            state_hashes={},
        )
    else:
        manifest = d.manifest(evidence, result, reference)
    atomic_json(raw / "fresh_role_manifest.json", manifest)
    state = c.Checkpoint(manifest, canonical_hash(manifest), raw / "production_custody")
    denial = refused(
        lambda: state.labels(
            raw / "protected/test_truth.json", "not_collected", "test", "final_evaluation"
        )
    )
    controls = qualify(raw / "custody_controls")
    controls.append(
        dict(
            name="production_test_denied_before_both_seals",
            passed=denial,
            fixture_only=False,
            access_log_reference=provenance.reference(
                raw / "production_custody/label_access.jsonl"
            ),
        )
    )
    work = dict(
        raw=str(raw),
        input_root=str(root),
        fixture_only=fixture,
        evidence=evidence,
        environment=environment,
        result=result,
        controls=controls,
        manifest_reference=provenance.reference(raw / "fresh_role_manifest.json"),
        overlap_rows=[
            dict(
                source="retained_development_and_V764_V771",
                inventory_reference=reference,
                graph_count=len(inventory["old_graphs"]),
                seed_count=len(inventory["old_seeds"]),
                comparator="networkx.is_isomorphic, unmarked topology; stricter than start/target identity",
                pending=result.get("history_pending"),
                matches=sum(
                    r["reason"] == "isomorphic_relabeling" for r in result["proposal_rows"]
                ),
            )
        ],
        duration_s=time.monotonic() - start,
        phase_spans=[
            dict(phase="authentication_construction_custody", duration_s=time.monotonic() - start)
        ],
        raw_references=[],
    )
    atomic_json(raw / "work.json", work)
    return work


def replay(path: Path) -> Json:
    """Rebuild aggregate fields and graph proposals from unchanged primitive source bytes."""
    try:
        value = json.loads(path.read_bytes())
        for ref in (
            value["source_artifact_hashes"]
            + value["raw_shard_hashes"]
            + value["code_config_hashes"]
        ):
            if provenance.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("primitive_bytes")
        work = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_bytes())
        validation = json.loads((Path(work["raw"]) / "validation.json").read_bytes())
        if work["accepted"] and not (
            checks.receipts_pass(
                dict(
                    validation_scope_manifest=validation["scope"],
                    validation=validation,
                    frozen_ns=validation["scope"]["frozen_ns"],
                )
            )
            and verify_measured(validation["coverage"], d.OWNED)
        ):
            raise ValueError("validation_drift")
        work["raw_references"] = value["raw_shard_hashes"]
        if d.build(work, validation, work["accepted"]) != value:
            raise ValueError("aggregate_drift")
        for graph in work["result"].get("graph_manifest", []):
            original = g.source.candidate(graph["seed"])
            with patch.object(g, "SEED", d.SEED):
                fresh = g.fresh_graph(
                    original, graph["seed"], int(graph["graph_id"].rsplit("-", 1)[1])
                )
            fresh.update(graph_id=graph["graph_id"], unit_id=graph["unit_id"])
            fresh["questions"][0]["unit_id"] = graph["unit_id"]
            if fresh != graph:
                raise ValueError("roster_drift")
        if work["result"].get("proposal_count") is not None:
            inventory = json.loads((Path(work["raw"]) / "historical_inventory.json").read_bytes())
            rebuilt = d.construct(inventory, Path(tempfile.mkdtemp(prefix="carnot-9042-replay-")))
            for field in [
                "graph_manifest",
                "proposal_rows",
                "complete",
                "terminal_reason",
                "history_indexed",
                "history_pending",
            ]:
                if rebuilt[field] != work["result"][field]:
                    raise ValueError("proposal_replay_drift")
        return dict(passed=True, reason="primitive_replay_passed")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        return dict(passed=False, reason=str(error))


def plan(private: Path) -> tuple[list[Any], Json]:
    """Reuse the complete scoped check vocabulary and freeze exact commands before work."""
    with patch.multiple(checks, OWNED=d.OWNED, TEST=d.TEST, DIRECT=DIRECT):
        commands = checks.validation_plan(private)
        scope = checks.scope_manifest(private)
    isolated = []
    for command in commands:
        argv = command.argv
        if "pytest" in argv:
            index = argv.index("pytest") + 1
            argv = (*argv[:index], "-p", "carnot.reporting.fresh_seal_runner_9042", *argv[index:])
        isolated.append(
            replace(
                command,
                argv=argv,
                timeout_s=900 if command.name == "direct_consumers" else command.timeout_s,
            )
        )
    scope["commands"] = [
        dict(name=c.name, argv=list(c.argv), timeout_s=c.timeout_s) for c in isolated
    ]
    scope["frozen_ns"] = time.monotonic_ns()
    return isolated, scope


def main(argv: list[str] | None = None) -> int:
    """One invocation retains every failed check and publishes only a checked candidate."""
    os.environ.update(PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261112"], default="20261112")
    parser.add_argument("--input-root", type=Path, default=d.ROOT)
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--custody-worker", type=Path)
    parser.add_argument("--worker-raw", type=Path)
    parser.add_argument("--interrupt", action="store_true")
    args = parser.parse_args(argv)
    if args.cold_replay:
        report = replay(args.cold_replay)
        print(json.dumps(report), flush=True)
        return int(not report["passed"])
    if args.custody_worker:
        if args.worker_raw is None:
            parser.error("--worker-raw is required")
        c.worker(args.custody_worker, args.worker_raw, args.interrupt)
        return 0
    fixture = args.fixture_e2e is not None
    output = args.fixture_e2e or d.ROOT / "results" / (d.NAME + ".json")
    if fixture and output.resolve().is_relative_to(d.ROOT):
        parser.error("fixture writes require private scratch")
    raw = output.absolute().parent / "raw" / d.NAME / "invocations" / str(time.time_ns())
    private = Path(tempfile.mkdtemp(prefix="carnot-9042-validation-"))
    commands, scope = plan(private)
    atomic_json(raw / "validation_scope_manifest.json", scope)
    started = time.monotonic()
    work = freeze(args.input_root, raw, fixture)
    validation: Json = dict(receipts=[], coverage={}, scope=scope)
    accepted = False
    if not fixture:
        os.environ["COVERAGE_RCFILE"] = str(private / "coverage.ini")
        for index, command in enumerate(commands):
            d.progress("validation", index, len(commands) - index)
            validation["receipts"].append(checks.run_check(command, raw / "validation", scope))
        validation["coverage"] = measured([private / ".coverage"], d.OWNED, raw / "measured")
        validation["coverage"].pop("rows", None)
        accepted = checks.receipts_pass(
            dict(
                validation_scope_manifest=scope, validation=validation, frozen_ns=scope["frozen_ns"]
            )
        ) and verify_measured(validation["coverage"], d.OWNED)
    work.update(accepted=accepted, duration_s=time.monotonic() - started)
    work["phase_spans"].append(
        dict(
            phase="owned_validation",
            duration_s=work["duration_s"] - sum(p["duration_s"] for p in work["phase_spans"]),
        )
    )
    atomic_json(raw / "validation.json", validation)
    atomic_json(raw / "work.json", work)
    work["raw_references"] = [
        provenance.reference(raw / "work.json"),
        provenance.reference(raw / "validation.json"),
        work["manifest_reference"],
    ]
    work["raw_references"] += [
        provenance.reference(p)
        for p in [
            raw / "historical_inventory.json",
            raw / "construction.json",
            raw / "state_hashes.json",
        ]
        if p.is_file()
    ]
    value = d.build(work, validation, accepted)
    atomic_json(raw / "candidate.json", value)
    if not fixture and not value["required_checks_passed"]:
        d.progress("owned_checks_failed_primary_unchanged", 0, 1)
        return 1
    if fixture:
        publish_primary(output, value, replay)
    else:
        with patch.object(terminal, "d", d):
            publish_primary(output, value, lambda p: terminal.scanners(p, raw / "terminal", scope))
    d.progress("terminal_published", value["completed_count"], value["censored_count"])
    print(
        json.dumps(
            dict(
                path=str(output),
                honest_verdict=value["honest_verdict"],
                fresh_source_ready_score=value["fresh_source_ready_score"],
            )
        ),
        flush=True,
    )
    return 0
