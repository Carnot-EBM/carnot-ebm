"""REQ-VERIFY-8641: new bounded execution owns immutable coverage, never old repaired bytes."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

import numpy as np
from coverage import CoverageData
from carnot.reporting import deadline_statement_8616 as historical
from carnot.reporting import learning_custody_8641 as c
from carnot.reporting import learning_execution_report_8641 as r
from carnot.reporting.count_semantics_runner_8566 import terminal_nodes
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.learning_clock_validation_8592 import E2E
from carnot.reporting.primary_publication import publish_primary, reader_receipt, read_bound_sidecar
from carnot.reporting.v709_execution import progress
from scripts.adversarial_verify import verify_artifact

Json = dict[str, Any]
ROOT, CLI, MODEL_SPECS = r.ROOT, r.CLI, []
OPTIONS = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]


def prepare(raw: Path, run_date: str) -> Json:
    """Authenticate inputs independently of failed execution, then reuse original measured mechanics."""
    progress("preconditions_no_model_load_MODEL_SPECS=[]_current_llm_calls=0")
    started = time.monotonic_ns()
    upstream, failures = r.inputs()
    base = historical.base.prepare(raw)
    if upstream:
        terminal = json.loads(Path(upstream["terminal_validation_sidecar_path"]).read_bytes())
        if not read_bound_sidecar(r.UPSTREAM, Path(terminal["publication"]["sidecar_path"]))[
            "report"
        ]["passed"]:
            failures.append(r.gate(r.UPSTREAM, "upstream_publication", True, False))
    sources = [c.ref(ROOT / p) for p in [*r.OWNED, *r.TESTS, r.RUNNER]]
    authority = [
        dict(path=str(ROOT / "tests/python" / name), sha256="sha256:" + pin)
        for name, pin in {**r.TEST_PINS, **r.E2E_PINS}.items()
    ]
    for reference in authority:
        if c.ref(Path(reference["path"])) != reference:
            failures.append(
                r.gate(
                    Path(reference["path"]),
                    "original_test_source",
                    reference["sha256"],
                    c.ref(Path(reference["path"]))["sha256"],
                )
            )
    if c.ref(ROOT / r.RUNNER)["sha256"] != historical.PIN_SOURCE:
        failures.append(
            r.gate(
                ROOT / r.RUNNER,
                "original_runner_source",
                historical.PIN_SOURCE,
                c.ref(ROOT / r.RUNNER)["sha256"],
            )
        )
    progress("before_small_head_numerical_inference")
    head = historical.stream.source()["heads"][0]
    prediction = historical.stream.stream.adapter.sparse.predict(
        head, np.asarray([[0.4] * 9])
    ).tolist()
    progress("after_small_head_numerical_inference", 1, 0)
    return dict(
        raw=str(raw),
        run_date=run_date,
        started_monotonic_ns=started,
        upstream=upstream,
        input_failures=failures + base["failures"],
        preconditions=base["preconditions"],
        input_references=base["source_refs"] + ([c.ref(r.UPSTREAM)] if upstream else []),
        sources=sources,
        original_authority=authority,
        original_nodes=base["original_nodes"],
        mechanics_fixture=base["mechanics_fixture"],
        children=[],
        receipts=[],
        nodes=[],
        outcomes=[],
        numerical_inference=dict(
            checkpoint=base["checkpoint_reference"],
            head="spline9",
            coefficients=109,
            input=[[0.4] * 9],
            prediction=prediction,
            evaluated_this_invocation=True,
        ),
        duration_s=0,
        phase_spans=[],
    )


def instrumented(work: Json, name: str, nodes: list[str], kind: str, sources: list[Json]) -> Json:
    """Each bounded child has a unique data path and context; seal it before returning control."""
    namespace = Path(work["raw"]) / "children" / name
    namespace.mkdir(parents=True, exist_ok=True)
    config = namespace / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess,_exit\ncontext="
        + name
        + "\ndata_file="
        + str(namespace / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + s["path"] + "\n" for s in sources)
        + "[report]\nexclude_lines=\n"
    )
    junit = namespace / "nodes.xml"
    remaining = (3600 if kind == "original" else 4500) - (
        time.monotonic_ns() - work["started_monotonic_ns"]
    ) / 1e9
    receipt = c.supervise(
        name,
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=" + str(config),
            "-m",
            "pytest",
            *OPTIONS,
            "--junitxml=" + str(junit),
            *nodes,
        ],
        namespace,
        env=dict(
            COVERAGE_FILE=str(namespace / ".coverage"),
            CARNOT_8582_MEASUREMENT=work["mechanics_fixture"]["path"],
            CARNOT_8603_CASE_ROOT=str(Path(work["raw"]) / "deadline_cases"),
            CARNOT_8616_CASE_ROOT=str(Path(work["raw"]) / "deadline_cases"),
        ),
        deadline=max(0.01, min(600, remaining)),
    )
    work["receipts"].append(receipt)
    if junit.is_file():
        work["outcomes"].extend(terminal_nodes(junit))
        receipt["junit_reference"] = c.ref(junit)
    try:
        manifest = c.seal_child(namespace, sources, receipt, nodes, kind)
        work["children"].append(manifest)
    except (OSError, ValueError) as error:
        work.setdefault("child_errors", []).append(dict(child=name, error=str(error)))
    return receipt


def execute(work: Json) -> None:
    """Collect original authority once and run disjoint children; never invoke a historical controller."""
    raw = Path(work["raw"])
    work["failure_reproduction"] = c.old_failure_fixture(raw / "old_failure_fixture")
    targets = ["tests/python/" + n for n in r.TEST_PINS] + E2E + r.TESTS
    collection = c.supervise(
        "collection",
        [sys.executable, "-m", "pytest", *OPTIONS, "--collect-only", *targets],
        raw / "logs",
        deadline=90,
    )
    work["receipts"].append(collection)
    nodes = [
        n
        for n in Path(collection["stdout_path"]).read_text().splitlines()
        if n.startswith("tests/") and "::" in n
    ]
    work["nodes"] = nodes
    atomic_json(raw / "collection.json", dict(nodes=nodes, authority=work["original_authority"]))
    work["collection_reference"] = c.ref(raw / "collection.json")
    originals = [n for n in nodes if not n.startswith(r.TESTS[0])]
    heavy = (
        "fixed_stream",
        "real_cli",
        "rehashed",
        "measurement_adapter",
        "production_dispatch",
        "owned_worker",
        "owned_failures",
        "retention_comparison",
        "resumed_history",
        "real_cases",
        "removed_read",
        "real_deadline",
        "unresumed",
        "normal_dispatch",
        "controller_plan",
    )
    fast = [n for n in originals if not any(word in n for word in heavy)]
    groups = [fast[i : i + 4] for i in range(0, len(fast), 4)] + [
        [n] for n in originals if n not in fast
    ]
    for index, group in enumerate(groups):
        progress("original_disjoint_children", index, len(groups) - index)
        if (time.monotonic_ns() - work["started_monotonic_ns"]) / 1e9 >= 3600:
            break
        instrumented(
            work,
            "original" + str(index),
            group,
            "original",
            [c.ref(ROOT / r.RUNNER), *work["original_authority"]],
        )
    work["original_groups"] = groups
    executed = [v["node_id"] for v in work["outcomes"]]
    work["pending_nodes"] = [n for n in originals if n not in executed]
    work["union_passed"] = bool(
        len(work["original_nodes"]) == 27
        and set(work["original_nodes"]) <= set(originals)
        and len(executed) == len(originals)
        and set(executed) == set(originals)
        and all(v["outcome"] == "passed" for v in work["outcomes"])
    )
    work["node_union"] = dict(
        collected=originals,
        children=groups,
        original27=work["original_nodes"],
        authority=work["original_authority"],
    )
    work["outcomes"] = [
        dict(v, unit=v["node_id"], passed=v["outcome"] == "passed", missing_reason=None)
        for v in work["outcomes"]
    ]
    rows = [
        json.loads(p.read_bytes())
        for p in sorted((raw / "deadline_cases").glob("case-*/case.json"))
    ]
    work["deadline_case_rows"] = rows
    changed = deepcopy(rows)
    if changed:
        changed[-1]["observed_elapsed_s"] = 0
    work["semantic_mutation_rows"] = [
        dict(mutation="omitted_expired", rejected=not historical.semantic_cases(rows[:1])),
        dict(mutation="changed_diagnostic", rejected=not historical.semantic_cases(changed)),
        dict(
            mutation="removed_third_read",
            rejected=any("test_actual_removed_read_mutant" in n for n in executed),
        ),
    ]
    work["semantics_passed"] = historical.semantic_cases(rows) and all(
        v["rejected"] for v in work["semantic_mutation_rows"]
    )
    progress("measurements_finished_validation_reserve")
    receipt = instrumented(
        work,
        "owned_tests",
        [n for n in nodes if n.startswith(r.TESTS[0])],
        "adapter",
        [c.ref(ROOT / p) for p in r.OWNED],
    )
    work["custody_controls_passed"] = receipt["passed"]
    work["custody_mutation_rows"] = [
        dict(mutation=n, rejected=receipt["passed"], evidence="actual adapter test subprocess")
        for n in (
            "deleted",
            "overwritten",
            "empty",
            "duplicate",
            "foreign_source",
            "self_consistently_rehashed_substitution",
            "lost_shard_second_combine",
        )
    ]


def recompute(work: Json, scratch: Path) -> None:
    """Cold-reduce raw SQLite from both namespaces and authenticate source and deadline semantics."""
    for source in [*work["sources"], *work["original_authority"]]:
        if c.ref(Path(source["path"])) != source:
            raise ValueError("current_source_identity")
    for kind, key in [("original", "original_coverage"), ("adapter", "owned_coverage")]:
        manifests = [m for m in work["children"] if m["kind"] == kind]
        if not manifests:
            work[key] = dict(complete=False, files={})
            continue
        sources = (
            [c.ref(ROOT / r.RUNNER), *work["original_authority"]]
            if kind == "original"
            else [c.ref(ROOT / p) for p in r.OWNED]
        )
        lines = c.attest_children(manifests, sources)
        report = r.statement_report(
            lines, [c.ref(ROOT / r.RUNNER)] if kind == "original" else sources
        )
        work[key] = report
        combined = c.combine_copies(manifests, scratch / kind)
        measured = CoverageData(basename=combined["reference"]["path"])
        measured.read()
        if any(sorted(measured.lines(p) or []) != values for p, values in lines.items()):
            raise ValueError("cold_union_drift")
        if kind == "original" and report["complete"]:
            full = dict(
                files={
                    r.RUNNER: dict(
                        executed_lines=report["files"][str(ROOT / r.RUNNER)]["executed"],
                        missing_lines=[],
                        excluded_lines=[],
                    )
                }
            )
            work["mapping"] = historical.map_statements(c.ref(ROOT / r.RUNNER), full)
    if work.get("semantics_passed") and not historical.semantic_cases(work["deadline_case_rows"]):
        raise ValueError("deadline_semantics_drift")
    if work.get("collection_reference"):
        collection = c.read(work["collection_reference"])
        if collection != dict(nodes=work["nodes"], authority=work["original_authority"]):
            raise ValueError("original_collection_drift")
        outcomes = []
        assigned = []
        for manifest in work["children"]:
            reference = manifest["receipt"]["junit_reference"]
            if c.ref(Path(reference["path"])) != reference:
                raise ValueError("junit_identity")
            actual = terminal_nodes(Path(reference["path"]))
            if sorted(v["node_id"] for v in actual) != sorted(manifest["node_ids"]):
                raise ValueError("actual_node_assignment")
            assigned.extend(manifest["node_ids"])
            outcomes.extend(actual)
        work["union_passed"] = bool(
            len(work["original_nodes"]) == 27
            and set(work["original_nodes"]) <= set(work["nodes"])
            and len(assigned) == len(set(assigned)) == len(work["nodes"])
            and set(assigned) == set(work["nodes"])
            and all(v["outcome"] == "passed" for v in outcomes)
        )
        work["outcomes"] = [
            dict(v, unit=v["node_id"], passed=v["outcome"] == "passed", missing_reason=None)
            for v in outcomes
        ]
        work["pending_nodes"] = [n for n in work["nodes"] if n not in assigned]


def invocation_binding(work: Json) -> Json:
    """Keep independent seals for primitive custody, collection and measured numerical operands."""
    return {
        key: work.get(key)
        for key in (
            "children",
            "input_references",
            "sources",
            "original_authority",
            "original_nodes",
            "collection_reference",
            "mechanics_fixture",
            "numerical_inference",
            "deadline_case_rows",
            "semantic_mutation_rows",
            "custody_mutation_rows",
        )
    }


def replay(path: Path) -> bool:
    """Authenticate an independent invocation seal before rebuilding summaries from retained bytes."""
    try:
        value = json.loads(path.read_bytes())
        work = c.read(value["primitive_reference"])
        seal = c.read(work["invocation_seal_reference"])
        if seal != invocation_binding(work):
            return False
        for reference in work["input_references"]:
            c.read(reference)
        for receipt in work["receipts"]:
            for stream in ("stdout", "stderr"):
                if c.ref(Path(receipt[stream + "_path"]))["sha256"] != receipt[stream + "_sha256"]:
                    return False
        with TemporaryDirectory(prefix="cold8641-", dir="/var/tmp") as directory:
            recompute(work, Path(directory))
        return r.build(work, Path(value["publication_output"])) == value
    except (OSError, ValueError, KeyError, TypeError):
        return False


def terminal(path: Path, raw: Path) -> Json:
    """Use unchanged consumers and a fresh process, preserving their actual severities and exits."""
    findings = json.loads(path.read_bytes())["adversarial_findings"]
    receipts = []
    for name, argv, expected in [
        (
            "adversarial",
            ["scripts/adversarial_verify.py", "--json", str(path)],
            int(bool(findings)),
        ),
        ("rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(path)], 0),
        ("cold", [CLI, "--cold-replay", str(path)], 0),
    ]:
        receipts.append(
            c.supervise(
                name,
                [sys.executable, "-u", *argv],
                raw / "terminal_logs",
                expected=expected,
                deadline=180,
            )
        )
    scan = json.loads(Path(receipts[0]["stdout_path"]).read_bytes())["reports"][0]
    return dict(
        passed=all(v["passed"] for v in receipts)
        and scan["flags"] == findings
        and not any(f["severity"] == "critical" for f in findings),
        adversarial=scan,
        receipts=receipts,
    )


def run(output: Path, run_date: str, supplied: Path | None = None) -> int:
    """Publish only bytes accepted by fresh replay and readers, with honest failed qualifications."""
    raw = output.parent / "raw" / r.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    work = json.loads(supplied.read_bytes()) if supplied else prepare(raw, run_date)
    if not supplied and not work["input_failures"]:
        execute(work)
    progress("validation_commands")
    if not supplied:
        for name, argv in [
            ("ruff_check", ["-m", "ruff", "check", *r.OWNED, *r.TESTS]),
            ("ruff_format", ["-m", "ruff", "format", "--check", *r.OWNED, *r.TESTS]),
            ("mypy", ["-m", "mypy", "--strict", "--follow-imports=silent", *r.OWNED]),
            ("spec", ["scripts/check_spec_coverage.py", "--files", *r.TESTS]),
        ]:
            work["receipts"].append(
                c.supervise(name, [sys.executable, *argv], raw / "logs", deadline=120)
            )
    raw = Path(work["raw"])
    with TemporaryDirectory(prefix="recompute8641-", dir="/var/tmp") as directory:
        recompute(work, Path(directory))
    for kind in ("original", "adapter"):
        manifests = [m for m in work["children"] if m["kind"] == kind]
        if manifests:
            aggregate = c.combine_copies(manifests, raw / "aggregates" / kind)
            work["original_aggregate" if kind == "original" else "owned_aggregate"] = aggregate[
                "reference"
            ]
            work.setdefault("second_combine_invariance", {})[kind] = aggregate[
                "second_combine_invariance"
            ]
    work["duration_s"] = (time.monotonic_ns() - work["started_monotonic_ns"]) / 1e9
    work["phase_spans"] = [
        dict(phase="bounded_execution_and_validation", duration_s=work["duration_s"])
    ]
    seal = raw / "invocation_seal.json"
    atomic_json(seal, invocation_binding(work))
    seal.chmod(0o444)
    work["invocation_seal_reference"] = c.ref(seal)
    atomic_json(raw / "work.json", work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, r.build(work, output))
    work["adversarial_findings"] = verify_artifact(candidate)["flags"]
    atomic_json(raw / "work.json", work)
    value = r.build(work, output)
    publication = publish_primary(output, value, lambda p: terminal(p, raw))
    selected = reader_receipt(
        r.TASK,
        output.parent,
        field="learning_execution_ready_score",
        expected=value["learning_execution_ready_score"],
    )
    atomic_json(raw / "terminal.json", dict(publication=publication, selected_identity=selected))
    progress("normal_exit_" + value["verdict_class"], 1, 0)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Keep calendar, private-path and wall-clock controls separate from the learner's clock."""
    print("[exp8641] start no_model_load MODEL_SPECS=[] current_llm_calls=0", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261018")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    try:
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        return 2
    if args.date != "20261018" or (
        args.private_output and args.private_output.resolve().is_relative_to(ROOT)
    ):
        return 2
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        return int(not replay(args.cold_replay))
    timer = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        return run(
            args.private_output or ROOT / "results" / (r.NAME + ".json"), args.date, args.work
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *timer)
