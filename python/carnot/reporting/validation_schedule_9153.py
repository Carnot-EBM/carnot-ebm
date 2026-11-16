"""REQ-REPORT-9153-V780: measure process costs without treating watchdogs as work.

Complete module lifetimes include shared fixtures and interpreter startup. Partial
node timings remain observations, while unmeasured lifetimes remain unknown.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time
from typing import Any

from coverage import Coverage

from carnot.reporting import validation_schedule_9139 as old
from carnot.reporting import validation_schedule_runner_9139 as r
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    build_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_9153_v780_validation_schedule"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_validation_schedule_9153.py"
OWNED = ["python/carnot/reporting/validation_schedule_9153.py", CLI]
MODEL_SPECS: list[Json] = []
SEED = 7809153
INPUTS = {
    "results/experiment_9152_v780_contract.json": (
        "sha256:147516eaae8f20be2c165884037131e3e827717ed09f307c625acc377faedf3a",
        "contract_ready_score",
        1,
    ),
    "results/experiment_9139_v779_validation_schedule.json": (
        "sha256:e1a8ca891dc49a977354a53ca1746de7de8c510b6e8bfc2bb3feba77f1c30a33",
        "schedule_ready_score",
        1,
    ),
    "results/experiment_9140_v779_forward_qualification.json": (
        "sha256:761aea6f77740bed81737d663ed3667d7b3af8ab54aa9b91b3ec84e05a3b8093",
        "forward_validation_ready_score",
        0,
    ),
}


def schedule(legacy: Json, additions: list[str], measured: Json | None = None) -> Json:
    """Keep expected process costs nullable; watchdogs only constrain execution."""
    plan = deepcopy(legacy)
    nodes = legacy["original_node_manifest"] + additions
    groups = old.prior.partitions(nodes)
    ceilings = {n: p["node_seconds"][n] for p in legacy["execution_graph"] for n in p["node_ids"]}
    graph = [
        dict(
            partition_id=str(i),
            node_ids=g,
            fixture_scope=g[0].split("::")[0],
            node_seconds={n: ceilings.get(n, 30) for n in g},
            startup_bound_s=90,
            budget_s=90 + sum(ceilings.get(n, 30) for n in g),
        )
        for i, g in enumerate(groups)
    ]
    costs = [
        dict(
            unit_id="module:" + p["fixture_scope"],
            command="pytest",
            fixture_lifetime=p["fixture_scope"],
            kill_timeout_s=p["budget_s"],
        )
        for p in graph
    ]
    costs += [
        dict(
            unit_id="command:" + c["name"],
            command=c["argv"],
            fixture_lifetime="complete_process",
            kill_timeout_s=c.get("deadline", 240),
        )
        for c in legacy["verification_graph"]
    ]
    for row in costs:
        elapsed = (measured or {}).get(row["unit_id"])
        row.update(
            measured_elapsed_s=elapsed,
            estimated_remaining_s=elapsed,
            unknown_cost=elapsed is None,
            uncertainty="unmeasured" if elapsed is None else "single_process_observation",
            prediction_scope="next_qualification_invocation",
        )
    known = sum(c["estimated_remaining_s"] or 0 for c in costs)
    unknown = [c for c in costs if c["unknown_cost"]]
    historical = sum(p["budget_s"] for p in legacy["execution_graph"]) + sum(
        c.get("deadline", 240) for c in legacy["verification_graph"]
    )
    if historical != legacy["budget_equation"]["projected_total_s"]:
        raise ValueError("historical_ceiling_drift")
    plan.update(
        authority="carnot.validation_schedule.v3",
        original_node_manifest=nodes,
        historical_original_node_manifest=legacy["original_node_manifest"],
        v779_additions=additions,
        execution_graph=graph,
        historical_ceiling_sum_s=historical,
        cost_model_rows=costs,
        measured_cost_rows=[
            dict(
                x,
                measured_elapsed_s=x["measured_seconds"],
                fixture_lifetime="incomplete_node_phases",
            )
            for x in legacy["measured_cost_rows"]
        ],
        unknown_cost_rows=unknown,
        ordering="whole module lifetimes; no fixture splitting",
    )
    plan["budget_equation"] = dict(
        measurement_s=3300,
        verification_s=1200,
        publication_margin_s=300,
        hard_cap_s=4800,
        verification_reservations=1,
        measured_known_s=known,
        estimated_remaining_s=None if unknown else known,
        feasible=False if known > 3300 else None if unknown else True,
        formula="sum measured complete process predictions once; unknown terms remain null; watchdogs excluded",
    )
    return plan


def execute(plan: Json, raw: Path, invocation: str, hashes: list[Json], stop: float) -> Json:
    """Spend one global deadline and retain every unfinished obligation without retries."""
    old.prior.partitions(plan["original_node_manifest"])
    graph = plan["execution_graph"]
    identities = [p["partition_id"] for p in graph]
    if sorted(p["node_ids"] for p in graph) != sorted(
        old.prior.partitions(plan["original_node_manifest"])
    ) or len(identities) != len(set(identities)):
        raise ValueError("partition_union")
    names = [c["name"] for c in plan["verification_graph"]]
    if len(names) != len(set(names)):
        raise ValueError("duplicate_command")
    raw.mkdir(parents=True, exist_ok=True)
    bundle: Json = dict(
        invocation_id=invocation,
        started_monotonic_ns=time.monotonic_ns(),
        schedule=plan,
        code_config_hashes=hashes,
        partitions=[],
        checks=[],
        root=str(raw),
        node_seconds=None,
    )
    for part in plan["execution_graph"]:
        remaining = stop - time.monotonic()
        if remaining <= 0:
            break
        directory = raw / part["partition_id"]
        old.progress("v780_module_before_" + part["fixture_scope"])
        receipt = old.prior.run_check(
            "partition_" + part["partition_id"],
            old.partition_argv(part, directory),
            directory,
            min(part["budget_s"], remaining),
        )
        bundle["partitions"].append(
            dict(
                partition_id=part["partition_id"],
                node_ids=part["node_ids"],
                receipt=receipt,
                receipt_reference=old.ref(
                    directory / receipt["name"] / (receipt["name"] + ".receipt.json")
                ),
                events_reference=old.ref(directory / "events.jsonl"),
            )
        )
        atomic_json(raw / "bundle.json", bundle)
        old.progress("v780_module_after_" + part["fixture_scope"])
    for spec in plan["verification_graph"]:
        remaining = stop - time.monotonic()
        if remaining <= 0:
            break
        receipt = old.prior.run_check(
            spec["name"],
            spec["argv"],
            raw / "checks",
            min(spec.get("deadline", 240), remaining),
            expected=spec.get("expected", 0),
        )
        bundle["checks"].append(
            dict(
                receipt=receipt,
                receipt_reference=old.ref(
                    raw / "checks" / spec["name"] / (spec["name"] + ".receipt.json")
                ),
            )
        )
    bundle["ended_monotonic_ns"] = time.monotonic_ns()
    events = [
        json.loads(line)
        for p in bundle["partitions"]
        if p["events_reference"]["sha256"]
        for line in old.prior.old.bound(p["events_reference"]).read_text().splitlines()
    ]
    bundle["rows"] = old.reduce_nodes(plan["original_node_manifest"], events)
    bundle["remaining_nodes"] = [x["unit_id"] for x in bundle["rows"] if x["status"] != "completed"]
    atomic_json(raw / "bundle.json", bundle)
    return bundle


def authenticate(root: Path, raw: Path) -> Json:
    """Authenticate exact gates before importing historical schedules or assertions."""
    work: Json = dict(checks=[], references=[], receipts=[], code_config_hashes=[])
    inputs = {}
    work["authority_references"] = {}
    for name, (pin, field, expected) in INPUTS.items():
        path = root / name
        observed = old.ref(path)["sha256"] or "missing_file"
        work["checks"].append(old.gate(path, "sha256", pin, observed, "source_bytes", name))
        if not work["checks"][-1]["passed"]:
            return work
        value = json.loads(old.prior.old.bound(dict(path=str(path), sha256=pin)).read_bytes())
        work["checks"].append(
            old.gate(path, field, expected, value.get(field, "missing_field"), field, name)
        )
        if not work["checks"][-1]["passed"]:
            return work
        target = raw / "inputs" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        work["references"].append(
            dict(old.ref(target), producer_path=str(path), imported_fields=list(value))
        )
        work["authority_references"][name] = old.ref(target)
        inputs[name] = value
    prior = old.authenticate(root, raw / "historical")
    for key in ["checks", "references"]:
        work[key].extend(prior[key])
    if not all(c["passed"] for c in work["checks"]):
        return work
    archived = inputs[list(INPUTS)[1]]
    forward = inputs[list(INPUTS)[2]]
    additions = [
        n for nodes in forward["replay_input"]["work"]["additions"].values() for n in nodes
    ]
    work.update(
        historical_reader_reference=prior["historical_reader_reference"],
        semantic_controls=prior["semantic_controls"],
        count_controls=prior["count_controls"],
        computational_root_cause=archived["computational_root_cause"],
        legacy_schedule=archived["schedule_reference"],
        additions=additions,
    )
    if prior["schedule"] != work["legacy_schedule"] or len(additions) != 53:
        raise ValueError("historical_scope_drift")
    work["addition_assertion_hashes"] = [
        reference
        for value in [archived, forward]
        for reference in value["code_config_hashes"]
        if reference["path"].endswith(tuple(forward["replay_input"]["work"]["additions"]))
    ]
    for reference in work["addition_assertion_hashes"]:
        observed = old.ref(Path(reference["path"]))["sha256"] or "missing_file"
        work["checks"].append(
            old.gate(
                Path(reference["path"]),
                "sha256",
                reference["sha256"],
                observed,
                "addition_assertion_bytes",
                "v779",
            )
        )
        work["references"].append(reference)
    work["schedule"] = schedule(work["legacy_schedule"], additions)
    return work


def commands(raw: Path) -> list[tuple[str, list[str]]]:
    """Freeze affected checks once; complete scientific validation belongs to Exp9154."""
    py = str(ROOT / ".venv/bin/python")
    pytest = [py, "-m", "pytest", "-n0", "-o", "addopts=", "--no-cov", "-q"]
    specs = [
        (
            "owned_tests",
            [
                py,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(raw / "coverage.ini"),
                "-m",
                "pytest",
                "-n0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                TEST,
            ],
        ),
        ("original_9139_controls", [*pytest, old.TEST]),
        ("original_9140_controls", [*pytest, "tests/python/test_forward_qualification_9140.py"]),
        (
            "e2e018_021",
            [
                *pytest,
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
                "tests/python/test_restricted_decision_audit_8210.py",
            ],
        ),
        ("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", *OWNED, TEST]),
        ("ruff_format", [str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, TEST]),
        (
            "strict_mypy",
            [str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED],
        ),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", TEST]),
    ]
    return [
        (
            name,
            [
                "env",
                "CARNOT_9125_NODE_LOG=" + str(raw / "verification" / name / "events.jsonl"),
                *argv,
                "-p",
                "carnot.reporting.receipt_qualification_9125",
            ]
            if name
            in {"owned_tests", "original_9139_controls", "original_9140_controls", "e2e018_021"}
            else argv,
        )
        for name, argv in specs
    ]


def measure(work: Json, raw: Path, stop: float) -> None:
    """Time a complete module and consume its logs in the real normal runner."""
    plan = work["schedule"]
    module = plan["original_node_manifest"][0].split("::")[0]
    nodes = [n for n in plan["original_node_manifest"] if n.split("::")[0] == module]
    probe = old.schedule(
        dict(
            nodes=nodes,
            events=[],
            required_command_union=[],
            original_assertion_references=plan["original_assertion_hashes"],
        )
    )
    bundle = execute(
        schedule(probe, []), raw / "probe", raw.parent.name, work["code_config_hashes"], stop
    )
    work.update(
        bundle=bundle, rows=bundle["rows"], bundle_reference=old.ref(raw / "probe/bundle.json")
    )
    work["receipts"].extend(p["receipt"] for p in bundle["partitions"])
    path = raw / "probe/bundle.json"
    if time.monotonic() >= stop:
        work["measurement_error"] = "measurement_deadline_exhausted"
        return
    receipt = old.prior.run_check(
        "normal_orchestration",
        r.foreign_argv(
            old.prior.old.CLI,
            "--schedule-v779",
            str(path),
            "--schedule-invocation",
            raw.parent.name,
        ),
        raw,
        min(180, max(0.001, stop - time.monotonic())),
    )
    work["receipts"].append(receipt)
    work["isolated_child_receipts"] = [receipt]
    work["consumed_reference"] = old.ref(path.parent / "consumed.json")
    work["references"].append(work["consumed_reference"])
    measured = (
        {"module:" + module: bundle["partitions"][0]["receipt"]["duration_s"]}
        if not bundle["remaining_nodes"]
        else {}
    )
    work["schedule"] = schedule(work["legacy_schedule"], work["additions"], measured)
    if time.monotonic() >= stop:
        work["measurement_error"] = "measurement_deadline_exhausted"
        return
    slow_node = "tests/python/test_forward_evidence_runner_9111.py::test_rehashed_primitive_tampering[derivation]"
    trace = old.prior.run_check(
        "slow_trace",
        [
            str(ROOT / ".venv/bin/python"),
            "-c",
            "import faulthandler,sys; faulthandler.dump_traceback_later(5, repeat=True); import pytest; raise SystemExit(pytest.main(sys.argv[1:]))",
            "-n0",
            "-o",
            "addopts=",
            "--no-cov",
            "-o",
            "faulthandler_timeout=5",
            "-q",
            slow_node,
        ],
        raw,
        min(20, max(0.001, stop - time.monotonic())),
    )
    work["slow_trace"] = trace
    work["receipts"].append(trace)
    work["checks"].append(
        dict(
            check="slow_child_causal_trace",
            upstream="exp9125-retained-slow-child",
            path=trace["stderr_path"],
            sha256=trace["stderr_sha256"],
            artifact_field="computational_root_cause",
            op="==",
            expected="established_by_trace",
            observed="unknown",
            passed=False,
        )
    )
    work["blocked_prerequisite"] = (
        "Identify the cause of the retained slow subprocess using descendant commands and wait stacks; a bounded probe cannot establish a cause."
    )


def coverage_rows(tracer: Coverage) -> list[Json]:
    """Count only newly owned statements, with no excluded lines."""
    rows = []
    for name in OWNED:
        _, statements, excluded, missing, _ = tracer.analysis2(str(ROOT / name))
        rows.append(dict(path=name, statements=statements, excluded=excluded, missing=missing))
    return rows


def verify(work: Json, raw: Path, stop: float) -> None:
    """Run each fixed check once, then read coverage only after all children exit."""
    work["verification_root"] = str(raw)
    work["verification_receipt_references"] = {}
    for name, argv in commands(raw):
        if time.monotonic() >= stop:
            break
        old.progress("v780_" + name + "_before")
        receipt = old.prior.run_check(
            name, argv, raw / "verification", min(600, stop - time.monotonic())
        )
        work["receipts"].append(receipt)
        work["verification_receipt_references"][name] = old.ref(
            raw / "verification" / name / (name + ".receipt.json")
        )
        old.progress("v780_" + name + "_after")
    tracer = Coverage(config_file=str(raw / "coverage.ini"))
    tracer.combine(strict=True, keep=True)
    tracer.save()
    rows = coverage_rows(tracer)
    total = sum(len(x["statements"]) for x in rows)
    work["coverage"] = dict(
        files=rows,
        statement_percent=100 * (total - sum(len(x["missing"]) for x in rows)) / total,
        excluded_statements=sum(len(x["excluded"]) for x in rows),
        data_reference=old.ref(raw / ".coverage"),
    )


def qualified(work: Json) -> bool:
    """Require exact commands, bound logs, post-exit coverage and same-invocation evidence."""
    try:
        expected = commands(Path(work["verification_root"]))
        receipts = work["receipts"][-len(expected) :]
        if len(receipts) != len(expected):
            return False
        for receipt, (name, argv) in zip(receipts, expected, strict=True):
            if receipt != json.loads(
                old.prior.old.bound(work["verification_receipt_references"][name]).read_bytes()
            ):
                return False
            if (
                receipt["name"] != name
                or receipt["argv"] != argv
                or not receipt["passed"]
                or not receipt["normal_exit"]
                or receipt["timed_out"]
            ):
                return False
            for stream in ["stdout", "stderr"]:
                old.prior.old.bound(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                )
        tracer = Coverage(
            config_file=False,
            data_file=str(old.prior.old.bound(work["coverage"]["data_reference"])),
        )
        tracer.set_option("report:exclude_lines", [])
        tracer.load()
        rows = coverage_rows(tracer)
        if rows != work["coverage"]["files"] or any(x["missing"] or x["excluded"] for x in rows):
            return False
        result = old.consume(work["bundle"], work["bundle"]["invocation_id"])
        return result == json.loads(
            old.prior.old.bound(work["consumed_reference"]).read_bytes()
        ) and all(x["passed"] for x in work["isolated_child_receipts"])
    except (KeyError, OSError, ValueError):
        return False


def build(work: Json, date: str, start: int, end: int) -> Json:
    """Reuse the established row schema while keeping V780 authority and counters separate."""
    value = r.build(work, date, start, end)
    blocked = any(not c["passed"] for c in work["checks"])
    checks_passed = qualified(work)
    affected_failed = "verification_root" in work and not checks_passed
    ready = not blocked and checks_passed
    value.update(
        experiment_id=9153,
        task_id="exp9153-measured-validation",
        milestone="2026.11.780",
        schema="carnot.validation_schedule.v3",
        random_seed=SEED,
        honest_verdict="complete_disqualified_schedule_checks"
        if affected_failed
        else "complete_blocked_validation_prerequisite"
        if blocked
        else "complete_null_schedule_qualified"
        if ready
        else "complete_disqualified_schedule_checks",
        verdict_class="disqualified"
        if affected_failed
        else "blocked"
        if blocked
        else "null"
        if ready
        else "disqualified",
        schedule_ready_score=int(ready),
        required_checks_passed=checks_passed,
        cost_model_rows=list(work.get("schedule", {}).get("cost_model_rows", [])),
        computational_root_cause=work.get("computational_root_cause", "unknown"),
        blocked_prerequisite=work.get("blocked_prerequisite"),
        projected_execution_verdict="unknown_cost_requires_measurement",
        full_674_node_qualification_score=0,
        acceptance_gates=dict(
            authenticated_inputs=all(
                c["passed"] for c in work["checks"] if c["check"] != "slow_child_causal_trace"
            ),
            owned_validation=checks_passed,
            causal_prerequisite=not blocked,
            full_qualification_deferred=True,
        ),
    )
    for row in value["rows"]:
        if row["condition"] == "deferred_to_exp9140":
            row.update(
                condition="deferred_to_exp9154", censor_reason="full_qualification_owned_by_exp9154"
            )
    receipt = build_current_work_receipt(
        run_id="exp9153",
        owner_pid=work.get("owner_pid", os.getpid()),
        events=[],
        inference_substrate="deterministic_no_model_load",
        inference_substrate_details=dict(
            work="CPU validation scheduling and actual subprocess timing"
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
    )
    value["current_work_receipts"] = [receipt]
    value["source_references"] = [
        dict(
            x,
            producer_path=x.get("producer_path", x["path"]),
            imported_fields=x.get("imported_fields", ["exact_bytes"]),
        )
        for x in work["references"]
    ]
    value["cost_model_rows"] += [
        dict(
            unit_id="current_command:" + x["name"],
            command=x["argv"],
            measured_elapsed_s=x["duration_s"],
            estimated_remaining_s=None,
            unknown_cost=x["timed_out"],
            uncertainty="right_censored" if x["timed_out"] else "observed_current_process_only",
            kill_timeout_s=x["budget_s"],
            fixture_lifetime="complete_process" if x["normal_exit"] else "unfinished_process",
        )
        for x in work["receipts"]
        if not x["name"].startswith("partition_")
    ]
    value["current_revision_reference"]["authority"] = "carnot.validation_schedule.v3"
    value["current_revision_reference"]["v779_addition_assertion_hashes"] = work.get(
        "addition_assertion_hashes", []
    )
    value["field_principles"].update(
        cost_model_rows="Measured process costs and watchdog ceilings have separate units of evidence.",
        required_checks_passed="Only actual affected normal-exit checks and complete new-statement coverage qualify.",
        computational_root_cause="A wait stack identifies a wait, not the cause of slow work.",
    )
    value["replay_input"] = dict(work=work, date=date, start=start, end=end)
    value["reproducibility_checksum"] = canonical_hash(value["replay_input"])
    return value


def replay(path: Path) -> Json:
    """Recompute the plan and rows from bound sources before trusting any headline."""
    value = json.loads(path.read_bytes())
    saved = value["replay_input"]
    if value["reproducibility_checksum"] != canonical_hash(saved):
        raise ValueError("replay_hash")
    work = saved["work"]
    for reference in value["source_references"] + value["code_config_hashes"]:
        old.prior.old.bound(reference)
    if "legacy_schedule" in work:
        if work.get("authority_references"):
            archived = json.loads(
                old.prior.old.bound(work["authority_references"][list(INPUTS)[1]]).read_bytes()
            )
            forward = json.loads(
                old.prior.old.bound(work["authority_references"][list(INPUTS)[2]]).read_bytes()
            )
            additions = [
                n for nodes in forward["replay_input"]["work"]["additions"].values() for n in nodes
            ]
            if (
                work["legacy_schedule"] != archived["schedule_reference"]
                or work["additions"] != additions
            ):
                raise ValueError("historical_scope_drift")
        measured = {}
        if "bundle_reference" in work:
            bundle = json.loads(old.prior.old.bound(work["bundle_reference"]).read_bytes())
            if bundle != work["bundle"] or r.probe_rows(bundle) != work["rows"]:
                raise ValueError("bundle_drift")
            if not bundle["remaining_nodes"]:
                old.consume(bundle, bundle["invocation_id"])
                for part in bundle["partitions"]:
                    module = part["node_ids"][0].split("::")[0]
                    measured["module:" + module] = part["receipt"]["duration_s"]
        if work["schedule"] != schedule(work["legacy_schedule"], work["additions"], measured):
            raise ValueError("schedule_drift")
    if value != build(work, saved["date"], saved["start"], saved["end"]):
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True)


def terminal(path: Path, raw: Path) -> Json:
    """Both unchanged scanners must accept the exact candidate bytes before publication."""
    replay(path)
    checks = [
        old.prior.run_check(
            name, [str(ROOT / ".venv/bin/python"), str(ROOT / script), flag, str(path)], raw
        )
        for name, script, flag in [
            ("adversarial", "scripts/adversarial_verify.py", "--json"),
            ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
        ]
    ]
    return dict(passed=all(c["passed"] for c in checks), checks=checks)


def repository_health(reference: Json) -> Json:
    """Retained unrelated health cannot substitute for any current affected check."""
    receipt: Json = json.loads(old.prior.old.bound(reference).read_bytes())
    if receipt["name"] != "repository_health":
        raise ValueError("foreign_repository_health")
    for stream in ["stdout", "stderr"]:
        old.prior.old.bound(
            dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
        )
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Bound measurement, reserve verification once, and publish only checked terminal bytes."""
    from carnot.reporting.forward_qualification_9140 import host_checks

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261116")
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.replay:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        if args.date != "20261116":
            raise ValueError("run_date")
        start = time.monotonic_ns()
        stop = start / 1e9 + 3300
        raw = args.output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        old.progress("v780_authentication_before")
        resources = host_checks(raw)
        work = (
            authenticate(args.input_root, raw)
            if all(c["passed"] for c in resources)
            else dict(checks=[], references=[], receipts=[], code_config_hashes=[])
        )
        work["checks"].extend(resources)
        work["owner_pid"] = os.getpid()
        work["code_config_hashes"] = [old.ref(ROOT / p) for p in [*OWNED, TEST, "pyproject.toml"]]
        old.progress("v780_authentication_after")
        if all(c["passed"] for c in work["checks"]):
            config = raw / "coverage.ini"
            config.write_text(
                "[run]\nparallel=true\npatch=subprocess,_exit\ndata_file="
                + str(raw / ".coverage")
                + "\ninclude=\n"
                + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
                + "[report]\nexclude_lines=\nomit=\n"
            )
            work["references"].append(old.ref(config))
            atomic_json(raw / "commands.json", dict(commands=commands(raw)))
            work["references"].append(old.ref(raw / "commands.json"))
            tracer = Coverage(config_file=str(config), data_suffix=True)
            tracer.start()
            measure(work, raw / "measurement", stop)
            retained_health = os.environ.get("CARNOT_9153_REPOSITORY_HEALTH_REFERENCE")
            if retained_health:
                reference = json.loads(retained_health)
                work["repository_health"] = repository_health(reference)
                work["repository_health_scope"] = "retained_unrelated_repository_health"
                work["references"].append(reference)
            elif time.monotonic() < stop:
                work["repository_health"] = old.prior.run_check(
                    "repository_health",
                    [str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
                    raw / "health",
                    min(120, max(0.001, stop - time.monotonic())),
                )
            tracer.stop()
            tracer.save()
            verify(work, raw, min(start / 1e9 + 4500, time.monotonic() + 1200))
        value = build(work, args.date, start, time.monotonic_ns())
        old.progress("v780_publication_before")
        publish_primary(args.output, value, lambda p: terminal(p, raw / "terminal"))
        old.progress("v780_publication_after")
        print(
            json.dumps(
                dict(
                    path=str(args.output),
                    honest_verdict=value["honest_verdict"],
                    schedule_ready_score=value["schedule_ready_score"],
                )
            ),
            flush=True,
        )
        return 0
    except (OSError, ValueError, KeyError) as error:
        print("[exp9153] refused=" + str(error), flush=True)
        return 1
