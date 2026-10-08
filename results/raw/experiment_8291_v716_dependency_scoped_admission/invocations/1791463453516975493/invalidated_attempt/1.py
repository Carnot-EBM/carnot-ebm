"""REQ-REPORT-8291: publish exact CPU fixture mechanics with no natural benefit.

The frozen graph roster is a software oracle, not independent scientific truth.
GRACE motivates locality only; this study supplies its own explicit equations.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from statistics import median
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import coverage_custody_8262 as custody
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.primary_publication import (
    publish_primary,
    read_bound_sidecar,
    validate_primary,
)
from carnot.reporting.v709_execution import child, progress
from carnot.verify import dependency_scoped_admission_8291 as d
from carnot.verify import typed_admission_8263 as typed

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8291_v716_dependency_scoped_admission"
TASK = "exp8291-dependency-scoped-admission"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_dependency_scoped_admission_8291.py"
OWNED = [
    "python/carnot/verify/dependency_scoped_admission_8291.py",
    "python/carnot/reporting/dependency_admission_execution_8291.py",
    CLI,
]
MODEL_SPECS: list[Json] = []


def ref(path: Path) -> Json:
    """Bind actual bytes, so a path never substitutes for evidence."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def authenticate(root: Path) -> tuple[list[Json], list[Json]]:
    """Authenticate typed and durable-coverage primitives without any CUDA gate."""
    checks: list[Json] = []
    refs: list[Json] = []

    def gate(path: Path, field: str, expected: Any, observed: Any) -> None:
        checks.append(
            dict(
                upstream=path.stem,
                path=str(path),
                hash=sha256_file(path) if path.is_file() else None,
                artifact_field=field,
                op="==",
                expected=expected,
                observed=observed,
                passed=expected == observed,
            )
        )

    for name in ["8263_v714_protocol_conformance", "8262_v714_coverage_custody"]:
        path = root / f"results/experiment_{name}.json"
        gate(path, "exists", True, True if path.is_file() else None)
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_bytes())
            refs.append(
                dict(
                    ref(path),
                    fields_imported=[
                        "required_checks_passed",
                        "flagged_adversarial",
                        "raw_shard_hashes",
                        "coverage_command_receipt",
                        "terminal_validation_sidecar_path",
                    ],
                )
            )
            for field, expected in [
                ("required_checks_passed", True),
                ("flagged_adversarial", False),
            ]:
                gate(path, field, expected, value.get(field))
            terminal = Path(value["terminal_validation_sidecar_path"])
            report = json.loads(terminal.read_bytes())
            gate(
                terminal,
                "publication.primary_sha256",
                sha256_file(path),
                report["publication"]["primary_sha256"],
            )
            sidecar = Path(report["publication"]["sidecar_path"])
            gate(
                sidecar,
                "report.passed",
                True,
                read_bound_sidecar(path, sidecar)["report"]["passed"],
            )
            refs.extend(
                [
                    dict(ref(terminal), fields_imported=["publication"]),
                    dict(ref(sidecar), fields_imported=["report.passed"]),
                ]
            )
            for binding in value["raw_shard_hashes"]:
                operand = Path(binding["path"])
                gate(
                    operand,
                    "sha256",
                    binding["sha256"],
                    sha256_file(operand) if operand.is_file() else None,
                )
                refs.append(dict(binding, fields_imported=["primitive_bytes"]))
                if operand.name == "primitive_evidence.json" and operand.is_file():
                    evidence = json.loads(operand.read_bytes())
                    for key, state in evidence["states"].items():
                        rebuilt = typed.initial()
                        for event in state["events"]:
                            typed.transition(rebuilt, event)
                        gate(
                            operand,
                            f"states.{key}.causal_replay",
                            canonical_hash(state),
                            canonical_hash(rebuilt),
                        )
            measured = custody.replay(value["coverage_command_receipt"])
            gate(path, "coverage_command_receipt.measured_complete", True, bool(measured))
            refs.extend(
                dict(
                    ref(Path(value["coverage_command_receipt"][k])),
                    fields_imported=["coverage_primitive"],
                )
                for k in ["report_path", "receipt_path"]
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            gate(path, "authenticated_terminal_and_primitives", True, str(error))
    return checks, refs


def commands(private: Path) -> Json:
    """Freeze exact argv and bounded deadlines before measurement, including children."""
    py, cov, pytest, ruff, mypy = [
        str(ROOT / ".venv/bin" / n) for n in ["python", "coverage", "pytest", "ruff", "mypy"]
    ]
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel = true\npatch = subprocess, _exit\ndata_file = "
        + str(private / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
    )
    rc = "--rcfile=" + str(config)
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    rows = [
        (
            "unit",
            [
                "/usr/bin/env",
                "COVERAGE_RCFILE=" + str(config),
                cov,
                "run",
                rc,
                "-m",
                "pytest",
                *common,
                "-s",
                TEST,
            ],
            300,
        ),
        (
            "consumers_E2E015_019",
            [
                pytest,
                *common,
                "tests/python/test_protocol_conformance_8263.py",
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_source_boundary_7852.py",
                "tests/python/test_experiment_7942_v689_sentence_labels.py",
            ],
            180,
        ),
        ("coverage_combine", [cov, "combine", rc], 30),
        ("coverage_report", [cov, "report", rc, "--show-missing", "--fail-under=100"], 30),
        ("coverage_json", [cov, "json", rc, "-o", str(private / "coverage.json")], 30),
        ("ruff_check", [ruff, "check", *OWNED, TEST], 30),
        ("ruff_format", [ruff, "format", "--check", *OWNED, TEST], 30),
        ("strict_mypy", [mypy, "--strict", "--follow-imports=silent", *OWNED], 60),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", "--files", TEST], 30),
    ]
    return dict(
        commands=[dict(name=n, argv=a, deadline_s=t, expected_exit=0) for n, a, t in rows],
        repository_health=dict(
            name="repository_full_suite",
            argv=[pytest, "tests/python", "-q"],
            deadline_s=120,
            expected_exit=0,
        ),
    )


def measure(work: Json, raw: Path, *, fixture: bool = False) -> None:
    """Measure the frozen stream without changing graph sizes or thresholds."""
    started = time.monotonic()
    roster = json.loads(Path(work["manifest_path"]).read_bytes())
    graphs = roster["graphs"][:1] if fixture else roster["graphs"]
    for index, graph in enumerate(graphs):
        progress("before_benchmark_graph", index, len(graphs) - index)
        for rep in range(-1, 5):
            order = d.ARMS if rep % 2 == 0 else list(reversed(d.ARMS))
            for arm in order:
                if time.monotonic() - started >= 600:
                    raise TimeoutError("cpu_measurement_deadline")
                path = raw / "journals" / f"{index}-{rep}-{arm}.jsonl"
                progress("before_benchmark_" + arm, rep + 1, 5 - rep)
                d.execute(graph, arm, path)
                progress("after_benchmark_" + arm, rep + 2, 4 - rep)
                if rep >= 0:
                    work["runs"].append(
                        dict(
                            graph=index,
                            repetition=rep,
                            arm=arm,
                            journal=ref(path),
                            costs=ref(path.with_suffix(".jsonl.costs")),
                            prefix=ref(path.with_suffix(".jsonl.prefix.costs")),
                        )
                    )
        for arm in d.ARMS:
            path = raw / "crashes" / f"{index}-{arm}.jsonl"
            base = [
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / CLI),
                "--worker",
                work["manifest_path"],
                "--journal",
                str(path),
                "--graph",
                str(index),
                "--arm",
                arm,
            ]
            receipts = []
            for event in [24, 48, -1]:
                remaining = 600 - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("cpu_measurement_deadline")
                receipt = child(
                    f"crash-{index}-{arm}-{event}",
                    base + ["--crash", str(event)],
                    raw / "crash_logs",
                    expected=-9 if event >= 0 else 0,
                    deadline=min(30, remaining),
                    heartbeat=20,
                )
                state = d.load(graph, arm, path)
                receipts.append(
                    dict(
                        receipt,
                        crash_event=event,
                        issue_count=len(state["issues"]),
                        release_count=len(state["releases"]),
                    )
                )
            work["crashes"].append(dict(graph=index, arm=arm, journal=ref(path), receipts=receipts))
        atomic_json(raw / "work.json", work)
        progress("after_benchmark_graph", index + 1, len(graphs) - index - 1)
    work["measurement_duration_s"] = time.monotonic() - started


def reduce_work(work: Json) -> Json:
    """Derive all aggregates from authenticated journals in a fresh process."""
    path = Path(work["manifest_path"])
    if (
        sha256_file(path) != work["manifest_sha256"]
        or json.loads(path.read_bytes()) != d.manifest()
    ):
        raise ValueError("manifest_drift")
    graphs = d.manifest()["graphs"]
    indices = range(1 if work["fixture_mode"] else 24)
    expected_runs = {(i, rep, arm) for i in indices for rep in range(5) for arm in d.ARMS}
    actual_runs = [(r["graph"], r["repetition"], r["arm"]) for r in work["runs"]]
    if set(actual_runs) != expected_runs or len(actual_runs) != len(expected_runs):
        raise ValueError("run_roster")
    if {(r["graph"], r["arm"]) for r in work["crashes"]} != {
        (i, arm) for i in indices for arm in d.ARMS
    }:
        raise ValueError("crash_roster")
    costs, event_rows, later, states = [], [], [], {}
    disagreements = violations = 0
    for run in work["runs"]:
        graph, arm, rep = graphs[run["graph"]], run["arm"], run["repetition"]
        for key in ["journal", "costs", "prefix"]:
            if sha256_file(Path(run[key]["path"])) != run[key]["sha256"]:
                raise ValueError("primitive_hash")
        state = d.load(graph, arm, Path(run["journal"]["path"]))
        if len(state["issues"]) != 72 or len(state["releases"]) != 64:
            raise ValueError("incomplete_stream")
        states[run["graph"], rep, arm] = state
        prefix = [
            json.loads(line)["row"] for line in Path(run["prefix"]["path"]).read_text().splitlines()
        ]
        if (
            len(prefix) != 8
            or [r["id"] for r in prefix] != list(range(8))
            or any(
                r["total_ns"] != r["ended_monotonic_ns"] - r["started_monotonic_ns"]
                or r["total_ns"] < r["persistence_ns"]
                for r in prefix
            )
        ):
            raise ValueError("prefix_clock_drift")
        rows = [
            json.loads(line)["row"] for line in Path(run["costs"]["path"]).read_text().splitlines()
        ]
        if len(rows) != 64 or [r["id"] for r in rows] != list(range(64)):
            raise ValueError("cost_roster")
        for recorded, expected in zip(rows, state["releases"], strict=True):
            if any(recorded[k] != expected[k] for k in expected):
                raise ValueError("cost_decision_drift")
            if (
                recorded["total_ns"]
                != recorded["ended_monotonic_ns"] - recorded["started_monotonic_ns"]
            ):
                raise ValueError("clock_drift")
            if (
                recorded["total_ns"]
                < recorded["closure_ns"] + recorded["replay_ns"] + recorded["persistence_ns"]
            ):
                raise ValueError("span_drift")
            if arm != "one_hop" and recorded["accepted"]:
                # Committed values must satisfy the independent hard equations.
                violations += len(d.reference(graph, recorded["values"])[1])
        costs.append(
            dict(
                graph_id=graph["id"],
                topology=graph["topology"],
                arm=arm,
                repetition=rep,
                total_ns=sum(r["total_ns"] for r in rows + prefix),
                prefix_issue_ns=sum(r["total_ns"] for r in prefix),
                **{
                    k: sum(r[k] for r in rows)
                    for k in ["closure_ns", "replay_ns", "persistence_ns", "scan_count"]
                },
                evaluated_constraint_fraction=median(
                    r["evaluated_constraint_count"] / graph["size"] for r in rows
                ),
                fallback_count=sum(r["fallback_reason"] is not None for r in rows),
            )
        )
        if rep == 0:
            event_rows.extend(
                dict(
                    r,
                    graph_id=graph["id"],
                    arm=arm,
                    status="completed",
                    unit_id=f"{graph['id']}:{arm}:{r['id']}",
                    independent_source_count=0,
                    row_sha256=canonical_hash(r),
                    source_sha256=canonical_hash(r["source"]),
                    read_dependencies=graph["events"][r["id"]]["reads"],
                    write_dependencies=graph["events"][r["id"]]["writes"],
                    feedback_label=graph["events"][r["id"]]["label"],
                )
                for r in rows
            )
            later.extend(dict(r, graph_id=graph["id"], arm=arm) for r in state["issues"])
    graph_rows = []
    for index in sorted({r["graph"] for r in work["runs"]}):
        graph = graphs[index]
        for rep in range(5):
            a, b = d.semantic(states[index, rep, "full"]), d.semantic(states[index, rep, "scoped"])
            disagreements += sum(
                x != y
                for key in ["issues", "releases"]
                for x, y in zip(a[key], b[key], strict=True)
            )
            disagreements += int(a["soft"] != b["soft"] or a["values"] != b["values"])
        selected = [r for r in costs if r["graph_id"] == graph["id"]]
        scoped, full = [[r for r in selected if r["arm"] == arm] for arm in ["scoped", "full"]]
        graph_rows.append(
            dict(
                unit_id=graph["id"],
                graph_id=graph["id"],
                topology=graph["topology"],
                size=graph["size"],
                status="completed",
                denominator=1,
                numerator=1,
                independent_source_count=0,
                hard_constraint_manifest_sha256=canonical_hash(graph["constraints"]),
                unique_feedback_sources=len(states[index, 0, "scoped"]["seen"]),
                paired_median_time_difference_ns=median(
                    a["total_ns"] - b["total_ns"] for a, b in zip(scoped, full, strict=True)
                ),
                arm_median_transaction_ns={
                    arm: median(r["total_ns"] for r in selected if r["arm"] == arm)
                    for arm in d.ARMS
                },
                paired_median_time_ratio=median(
                    a["total_ns"] / b["total_ns"] for a, b in zip(scoped, full, strict=True)
                ),
                evaluated_constraint_fraction=median(
                    r["evaluated_constraint_fraction"] for r in scoped
                ),
            )
        )
    crash_parity = True
    for crash in work["crashes"]:
        for receipt in crash["receipts"]:
            event = receipt["crash_event"]
            crash_parity &= receipt["passed"] and (
                event < 0
                or (
                    receipt["actual_exit"] == -9
                    and receipt["issue_count"] == event + 1
                    and receipt["release_count"] == event - 8
                )
            )
        if sha256_file(Path(crash["journal"]["path"])) != crash["journal"]["sha256"]:
            raise ValueError("crash_hash")
        recovered = d.load(graphs[crash["graph"]], crash["arm"], Path(crash["journal"]["path"]))
        crash_parity &= d.semantic(recovered) == d.semantic(states[crash["graph"], 0, crash["arm"]])
    planted = [r for r in event_rows if r["id"] == 0]
    negative = bool(planted) and all(
        r["accepted"] and not r["conflicts"] for r in planted if r["arm"] == "one_hop"
    )
    detected = bool(planted) and all(
        not r["accepted"] and "goal" in r["conflicts"] for r in planted if r["arm"] != "one_hop"
    )
    sparse = [r for r in graph_rows if r["topology"] in ["chain", "sparse_dag"]]
    return dict(
        per_graph_rows=graph_rows,
        per_event_rows=event_rows,
        later_decision_rows=later,
        operation_cost_rows=costs,
        full_scan_disagreements=disagreements,
        full_scan_event_pair_denominator=len(states) // 3 * (72 + 64),
        hard_constraint_violations=dict(
            count=violations,
            checked_transactions=sum(
                r["accepted"]
                for key, run in states.items()
                if key[2] != "one_hop"
                for r in run["releases"]
            ),
            scope="Full/scoped accepted transactions; unsafe negative control excluded",
        ),
        negative_control_detected=negative,
        crash_parity=bool(crash_parity),
        planted_conflict_detected=detected,
        h3_sound=bool(graph_rows)
        and disagreements == violations == 0
        and crash_parity
        and negative
        and detected,
        efficiency_signal=bool(sparse)
        and median(r["paired_median_time_ratio"] for r in sparse) <= 1
        and median(r["evaluated_constraint_fraction"] for r in sparse) <= 0.5,
        fallback_counts={
            arm: sum(r["fallback_reason"] is not None for r in event_rows if r["arm"] == arm)
            for arm in d.ARMS
        },
    )


def build(work: Json, reduction: Json, receipts: list[Json], raw: Path, fixture: bool) -> Json:
    """Soundness readiness qualifies durable mechanics; efficiency is a separate outcome."""
    failed = [r for r in work["checks"] if not r["passed"]]
    owned = all(r["passed"] for r in receipts) and not work.get("owned_failure")
    if not fixture:
        owned &= (raw / "coverage_binding.json").is_file()
    sound = owned and not failed and reduction.get("h3_sound", False)
    efficiency = sound and reduction.get("efficiency_signal", False)
    verdict = (
        "disqualified"
        if not owned
        else "blocked"
        if failed
        else "circular_positive"
        if efficiency
        else "null"
    )
    if not failed and not reduction.get("h3_sound", False):
        verdict = "disqualified"
        owned = False
    rows = reduction.get("per_graph_rows", [])
    suffix = failed[0]["upstream"] if failed else "dependency_scoped_admission"
    value = dict(
        experiment_id=8291,
        task_id=TASK,
        milestone=".716",
        run_date="20261008",
        honest_verdict=f"complete_{verdict}_{suffix}",
        verdict_class=verdict,
        inference_substrate="deterministic_exact_verifier_and_versioned_external_state_no_llm",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=deepcopy(ZERO_INVOCATION_COUNTS),
        verifier_is_oracle=True,
        exposure_scope="exposed_development_deterministic_software_fixtures",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        generator_weight_updates=0,
        scientific_benefit_measured=False,
        fixture_mode=fixture,
        gate_check_summary=work["checks"],
        rows=rows,
        intended_count=1 if fixture else 24,
        completed_count=len(rows),
        failed_count=0,
        owned_failed_check_count=sum(not r["passed"] for r in receipts)
        + int(bool(work.get("owned_failure"))),
        censored_count=(1 if fixture else 24) - len(rows),
        excluded_count=0,
        independent_count=len(rows),
        required_checks_passed=bool(owned),
        flagged_adversarial=False,
        acceptance_gates=dict(
            owned_checks=bool(owned),
            external_inputs=not failed,
            H3=bool(sound),
            sparse_efficiency=bool(efficiency),
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["checks"],
        duration_s=(time.monotonic_ns() - work["started_monotonic_ns"]) / 1e9,
        random_seed=7161,
        soundness_ready_score=int(sound),
        dependency_efficiency_signal_score=int(efficiency),
        actual_crash_receipts=work["crashes"],
        manifest_path=work["manifest_path"],
        manifest_sha256=work["manifest_sha256"],
        source_artifact_hashes=work["refs"],
        cited_upstream_artifacts=work["refs"],
        code_config_hashes=[ref(ROOT / p) for p in OWNED + [TEST]],
        raw_shard_hashes=[
            ref(raw / p)
            for p in [
                "manifest.json",
                "work.json",
                "reduction.json",
                "validation_commands.json",
                "validation_receipts.json",
            ]
        ],
        phase_spans=work["phase_spans"],
        repository_health=work.get("repository_health", {}),
        invocation_argv=work["invocation_argv"],
        **reduction,
    )
    value["methodology_note"] = (
        "24 explicit deterministic graph fixtures, 64 proposals and eight retention clock issues each; "
        "delay eight, fsynced issue before release, immutable hard constraints, five paired alternating "
        "repetitions after one uncounted warm-up. Graph is the unit. No confidence or natural-learning "
        "claim. Unsafe one-hop is only a sensitivity control and never a live implementation. "
        "GRACE motivates local typed checking but does not prove Carnot soundness. Generator frozen. "
        "Cost includes issue and release fsync; timing-receipt instrumentation and crash replay are "
        "reported separately. Scoped checks are not a repository-wide pass. Future natural adoption "
        "requires a separate preregistration."
    )
    if (raw / "coverage_binding.json").is_file():
        binding = json.loads((raw / "coverage_binding.json").read_bytes())
        value["coverage_command_receipt"] = binding
        value["owned_statement_counts"] = custody.replay(binding)
        value["raw_shard_hashes"].append(ref(raw / "coverage_binding.json"))
    value["field_principles"] = {
        k: "Bind this current invocation to authenticated software-fixture evidence; "
        "separate readiness, measured cost and absent natural generalization."
        for k in value
    }
    value["field_principles"].update(
        soundness_ready_score="Zero event disagreements, exact crash parity and planted-conflict sensitivity qualify fixture mechanics only.",
        dependency_efficiency_signal_score="Sparse graph-unit median constraint fraction <=0.5 and paired median whole-transaction ratio <=1; repetitions add no independent sources.",
        operation_cost_rows="Measured monotonic spans include conservative fallback, dense and cyclic work even when slower.",
        actual_crash_receipts="Actual SIGKILL exits after durable issue and before feedback certify the recovery boundary.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Rehashed headline edits cannot replace independent causal reconstruction."""
    try:
        value = json.loads(path.read_bytes())
        validate_primary(value, Path(NAME + ".json"))
        if (
            canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"})
            != value["reproducibility_checksum"]
        ):
            return False
        for binding in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
        ):
            if sha256_file(Path(binding["path"])) != binding["sha256"]:
                return False
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        work = json.loads((raw / "work.json").read_bytes())
        expected = reduce_work(work) if work["runs"] else empty_reduction()
        if any(value[k] != v for k, v in expected.items()):
            return False
        if value["fixture_mode"] != work["fixture_mode"]:
            return False
        for receipt in value["validation_receipts"] + [
            r for c in work["crashes"] for r in c["receipts"]
        ]:
            for prefix in ["stdout", "stderr"]:
                if sha256_file(Path(receipt[prefix + "_path"])) != receipt[prefix + "_sha256"]:
                    return False
        if "coverage_command_receipt" in value:
            custody.replay(value["coverage_command_receipt"])
        return value["soundness_ready_score"] == int(
            value["required_checks_passed"]
            and not any(not c["passed"] for c in work["checks"])
            and expected["h3_sound"]
        ) and value["dependency_efficiency_signal_score"] == int(
            value["soundness_ready_score"] and expected["efficiency_signal"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def empty_reduction() -> Json:
    """Missing operands have no measured zero or fabricated event decisions."""
    return dict(
        per_graph_rows=[],
        per_event_rows=[],
        later_decision_rows=[],
        operation_cost_rows=[],
        full_scan_disagreements=None,
        full_scan_event_pair_denominator=0,
        hard_constraint_violations=None,
        negative_control_detected=False,
        crash_parity=False,
        planted_conflict_detected=False,
        h3_sound=False,
        efficiency_signal=False,
        fallback_counts={},
    )


def main(argv: list[str] | None = None) -> int:
    """Bound CPU measurement and validate private bytes before atomic publication."""
    progress("start_dependency_admission")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261008"], default="20261008")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--graph", type=int, default=0)
    parser.add_argument("--arm", choices=d.ARMS, default="scoped")
    parser.add_argument("--crash", type=int, default=-1)
    parser.add_argument("--reduce", type=Path)
    parser.add_argument("--reduction", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        progress("replay_passed" if passed else "reduction_drift")
        return 0 if passed else 1
    if args.worker:
        if args.journal is None:
            parser.error("worker requires --journal")
        graph = json.loads(args.worker.read_bytes())["graphs"][args.graph]
        d.execute(graph, args.arm, args.journal, args.crash)
        return 0
    if args.reduce:
        if args.reduction is None:
            parser.error("reduce requires --reduction")
        atomic_json(args.reduction, reduce_work(json.loads(args.reduce.read_bytes())))
        return 0
    fixture = args.fixture_output is not None
    output = (args.fixture_output or args.output).absolute()
    if fixture and output.resolve().is_relative_to(ROOT / "results"):
        parser.error("fixtures must remain outside results")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="carnot-8291-") as directory:
        private = Path(directory)
        plan = commands(private)
        candidate = output.parent / "raw" / output.stem / "terminal_candidate.json"
        terminal = [
            dict(
                name="cold_replay",
                argv=[
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(ROOT / CLI),
                    "--cold-replay",
                    str(candidate),
                ],
                deadline_s=90,
                expected_exit=0,
            ),
            dict(
                name="adversarial",
                argv=[
                    str(ROOT / ".venv/bin/python"),
                    "scripts/adversarial_verify.py",
                    "--json",
                    str(candidate),
                ],
                deadline_s=60,
                expected_exit=0,
            ),
            dict(
                name="strict_rows",
                argv=[
                    str(ROOT / ".venv/bin/python"),
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ],
                deadline_s=60,
                expected_exit=0,
            ),
        ]
        plan["terminal"] = terminal
        plan["measurement_children"] = [
            dict(
                name=f"crash-{index}-{arm}-{event}",
                expected_exit=-9 if event >= 0 else 0,
                deadline_s=30,
                argv=[
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(ROOT / CLI),
                    "--worker",
                    str(raw / "manifest.json"),
                    "--journal",
                    str(raw / "crashes" / f"{index}-{arm}.jsonl"),
                    "--graph",
                    str(index),
                    "--arm",
                    arm,
                    "--crash",
                    str(event),
                ],
            )
            for index in range(1 if fixture else 24)
            for arm in d.ARMS
            for event in [24, 48, -1]
        ]
        plan["reduction"] = dict(
            name="fresh_reduction",
            argv=[
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / CLI),
                "--reduce",
                str(raw / "work.json"),
                "--reduction",
                str(raw / "reduction.json"),
            ],
            deadline_s=90,
        )
        atomic_json(raw / "validation_commands.json", plan)
        atomic_json(raw / "manifest.json", d.manifest())
        work: Json = dict(
            checks=[],
            refs=[],
            runs=[],
            crashes=[],
            fixture_mode=fixture,
            manifest_path=str(raw / "manifest.json"),
            manifest_sha256=sha256_file(raw / "manifest.json"),
            started_monotonic_ns=time.monotonic_ns(),
            invocation_argv=[str(ROOT / CLI), *(argv or sys.argv[1:])],
            phase_spans=[],
        )
        if not fixture and output.resolve().is_relative_to(ROOT / "results"):
            note = ROOT / "docs/research-notes/v716-dependency-scoped-admission.md"
            note.write_text(
                "# V716 dependency-scoped admission preregistration\n\nFrozen before measurement on 2026-10-08.\n\n"
                + f"Manifest: `{work['manifest_path']}`\n\nSHA-256: `{work['manifest_sha256']}`\n\n"
                + "Seeds 7161/7162/7163; 32/128 hard constraints; chain, sparse DAG, cyclic and dense; "
                "24 graph units, 64 proposed updates and eight retention clock issues each. Boolean "
                "value checks, equality propagation and any-antecedent implication have explicit "
                "operands and read/write metadata. Hard constraints are immutable. Feedback delay=8. "
                "Retention nodes never receive updates. Five paired repetitions alternate arm order "
                "after one uncounted warm-up. Children are killed at issues 24 and 48 before release.\n\n"
                + "H3: zero event decision/state/admissibility disagreements with independent full rescan; "
                "exact crash parity; planted transitive conflict detected. One-hop must miss a conflict. "
                "Sparse median constraint fraction <=0.5 and paired median transaction ratio <=1 are "
                "required for efficiency. Dense/cyclic/fallback costs remain visible. No threshold tuning. "
                "Whole transactions include issue/release fsync; separate cost-receipt instrumentation "
                "is outside the committed transaction. CPU cap=600s; validation cap=900s. Frozen validation "
                "argv is adjacent validation_commands.json. Generator frozen; no_model_load; MODEL_SPECS=[]; "
                "zero LLM calls. Software fixtures alone; both generalization scores zero. Natural adoption "
                "requires another preregistration. [GRACE](https://arxiv.org/html/2607.09175v2) motivates "
                "typed locality but supplies no Carnot soundness theorem.\n"
            )
        if not fixture:
            progress("before_authentication")
            work["checks"], work["refs"] = authenticate(args.root)
            progress("after_authentication", len(work["checks"]))
        atomic_json(raw / "work.json", work)
        reduction = empty_reduction()
        if all(r["passed"] for r in work["checks"]):
            started = time.monotonic_ns()
            try:
                measure(work, raw, fixture=fixture)
                atomic_json(raw / "work.json", work)
                receipt = child(
                    "fresh_reduction",
                    plan["reduction"]["argv"],
                    raw / "reduction_logs",
                    deadline=90,
                )
                if not receipt["passed"]:
                    raise ValueError("fresh_reduction_failed")
                reduction = json.loads((raw / "reduction.json").read_bytes())
                work["reduction_receipt"] = receipt
            except (OSError, ValueError, TimeoutError) as error:
                work["owned_failure"] = str(error)
            work["phase_spans"].append(
                dict(
                    phase="measurement",
                    started_monotonic_ns=started,
                    ended_monotonic_ns=time.monotonic_ns(),
                )
            )
        atomic_json(raw / "reduction.json", reduction)
        receipts = []
        validation_started = time.monotonic()
        if not fixture:
            for spec in plan["commands"] + [plan["repository_health"]]:
                remaining = 900 - (time.monotonic() - validation_started)
                receipt = child(
                    spec["name"],
                    spec["argv"],
                    raw / "validation_logs",
                    deadline=max(0.001, min(spec["deadline_s"], remaining)),
                    scope="diagnostic" if spec["name"] == "repository_full_suite" else "owned",
                )
                if spec["name"] == "repository_full_suite":
                    work["repository_health"] = receipt
                else:
                    receipts.append(receipt)
                if spec["name"] == "coverage_json" and receipt["passed"]:
                    try:
                        atomic_json(
                            raw / "coverage_binding.json",
                            custody.preserve(ROOT, spec, receipt, OWNED, raw / "coverage"),
                        )
                    except ValueError as error:
                        receipt.update(passed=False, coverage_error=str(error))
        atomic_json(raw / "validation_receipts.json", receipts)
        atomic_json(raw / "work.json", work)
        value = build(work, reduction, receipts, raw, fixture)
        reports = []

        def validator(path: Path) -> Json:
            batch = []
            for spec in terminal:
                receipt = child(
                    spec["name"],
                    spec["argv"],
                    raw / "terminal_logs" / str(len(reports)),
                    deadline=spec["deadline_s"],
                )
                reports.append(receipt)
                batch.append(receipt)
            return dict(passed=all(r["passed"] for r in batch), checks=batch)

        try:
            publication = publish_primary(output, value, validator)
        except ValueError as error:
            if str(error) != "candidate_rejected":
                raise
            atomic_json(raw / "failed_terminal_candidate.json", value)
            atomic_json(raw / "failed_terminal_report.json", reports)
            value.update(
                honest_verdict="complete_disqualified_terminal_validation",
                verdict_class="disqualified",
                required_checks_passed=False,
                soundness_ready_score=0,
                dependency_efficiency_signal_score=0,
                flagged_adversarial=any(
                    r["name"] == "adversarial" and not r["passed"] for r in reports
                ),
            )
            value["validation_receipts"] = receipts + reports
            value["acceptance_gates"].update(owned_checks=False, H3=False, sparse_efficiency=False)
            value.pop("reproducibility_checksum")
            value["reproducibility_checksum"] = canonical_hash(value)
            publication = publish_primary(output, value, validator)
        atomic_json(
            raw / "terminal_validation.json",
            dict(
                publication=publication,
                owned_checks_passed=value["required_checks_passed"],
                normal_process_exit=True,
            ),
        )
        progress("publication_complete_dependency_admission", 1)
    return 0
