"""REQ-REPORT-8654: execute the sealed core partition without changing old assertions."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import learning_custody_8641 as c
from carnot.reporting import learning_execution_report_8641 as old_report
from carnot.reporting import v743_contract_methods as methods
from carnot.reporting.count_semantics_runner_8566 import terminal_nodes
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import read_bound_sidecar

Json = dict[str, Any]
ROOT = c.ROOT
NAME = "experiment_8654_v743_learning_core_execution"
TASK = "exp8654-learning-core-execution"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = [f"tests/python/test_learning_core{s}_8654.py" for s in ("", "_runner")]
OWNED = [
    f"python/carnot/reporting/learning_core{s}_8654.py" for s in ("", "_report", "_runner")
] + [CLI]
UPSTREAM = ROOT / "results/experiment_8653_v743_contract_methods.json"
UPSTREAM_PIN = "sha256:c1176a28073046f66bd1eb8ba412ec025b4695e3bc14a212d829a5c5818e04f9"
MODEL_SPECS: list[Json] = []
OPTIONS = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush boundaries and counts so a supervising parent can detect progress."""
    print(f"[exp8654] phase={phase} completed={completed} pending={pending}", flush=True)


def groups(partition: Json, collected: list[str]) -> list[Json]:
    """Keep module fixtures together while rejecting every lost or repeated obligation."""
    cohorts = partition["cohorts"]
    flat = [n for cohort in cohorts for n in cohort["node_ids"]]
    assigned = [n for cohort in cohorts for group in cohort["groups"] for n in group["node_ids"]]
    if (
        len(collected) != len(set(collected))
        or len(flat) != len(set(flat))
        or sorted(collected) != sorted(flat)
        or sorted(flat) != sorted(partition["original_node_ids"])
        or sorted(assigned) != sorted(flat)
        or any(
            n.split("::")[0] != g["module"]
            for cohort in cohorts
            for g in cohort["groups"]
            for n in g["node_ids"]
        )
    ):
        raise ValueError("original_partition")
    return list(next(cohort for cohort in cohorts if cohort["cohort"] == "A")["groups"])


def admission(group: Json, started: int, now: int) -> Json:
    """Reserve a complete registered child budget rather than killing it at the boundary."""
    remaining = 3600 - (now - started) / 1e9
    return dict(
        module=group["module"],
        estimated_s=group["child_budget_s"],
        remaining_measurement_s=remaining,
        validation_reserve_s=900,
        task_cap_s=4800,
        admitted=remaining >= group["child_budget_s"],
    )


def attest(manifests: list[Json], sources: list[Json]) -> dict[str, list[int]]:
    """The existing sealed SQLite reader checks sources, contexts, logs and raw hashes."""
    if not manifests:
        raise ValueError("zero_child_coverage")
    result: dict[str, set[int]] = {}
    seen: set[str] = set()
    for manifest in manifests:
        if any(source not in sources for source in manifest["sources"]):
            raise ValueError("source_identity")
        lines = c.attest_children([manifest], manifest["sources"])
        for row in manifest["files"]:
            path = row["reference"]["path"]
            if path in seen:
                raise ValueError("duplicate_child_shard")
            seen.add(path)
        for path, executed in lines.items():
            result.setdefault(path, set()).update(executed)
    return {path: sorted(executed) for path, executed in result.items()}


def diagnosis(partition: Json) -> Json:
    """Read the original timeout receipt so a budget failure cannot become a scientific null."""
    work = c.read(partition["historical_primitive_reference"])
    receipt = next(r for r in work["receipts"] if r["name"] == "original25")
    for stream in ("stdout", "stderr"):
        if c.ref(Path(receipt[stream + "_path"]))["sha256"] != receipt[stream + "_sha256"]:
            raise ValueError("historical_timeout_log")
    if (
        receipt["exit_code"] != -9
        or receipt["duration_s"] != 62.074893612
        or not receipt["timed_out"]
    ):
        raise ValueError("historical_timeout_diagnosis")
    return dict(
        cause="measurement_budget_exhaustion",
        exit_code=receipt["exit_code"],
        duration_s=receipt["duration_s"],
        receipt=receipt,
        nominal_deadline_rows_passed=sum(r["passed"] for r in work["deadline_case_rows"]),
        missing_mutations=[n for n in work["pending_nodes"] if "mutant" in n],
        historical_reference=partition["historical_primitive_reference"],
    )


def source_closure(paths: list[Path]) -> list[Json]:
    """Freeze local imports and direct script operands so later reuse detects source drift."""
    pending, seen = list(paths), set()
    while pending:
        path = pending.pop().resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_bytes())
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module, *[node.module + "." + alias.name for alias in node.names]]
            for module in modules:
                relative = Path(*module.split("."))
                for root in (ROOT / "python", ROOT):
                    pending.extend(
                        [root / relative.with_suffix(".py"), root / relative / "__init__.py"]
                    )
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith("experiment_")
                and "/" not in node.value
            ):
                pending.append(
                    ROOT / "scripts/experiments" / (node.value.removesuffix(".json") + ".py")
                )
    return [c.ref(path) for path in sorted(seen)]


def prepare(raw: Path, run_date: str, upstream: Path = UPSTREAM) -> Json:
    """Check independent operands and real scratch before admitting any CPU measurement."""
    progress("preconditions_no_model_load_MODEL_SPECS=[]_current_llm_calls=0")
    started = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    value = json.loads(upstream.read_bytes()) if upstream.is_file() else {}
    failures = [
        methods.gate(upstream, k, 1, value.get(k), "exp8653-contract-methods")
        for k in ("methods_ready_score", "learning_operands_ready_score")
        if value.get(k) != 1
    ]
    with TemporaryDirectory(prefix="core8654-", dir="/var/tmp") as directory:
        scratch = Path(directory) / "writable"
        with scratch.open("wb") as stream:
            stream.write(b"disk-backed")
            stream.flush()
            os.fsync(stream.fileno())
        memory = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) * 1024
        filesystem = c.supervise(
            "scratch_filesystem",
            ["stat", "-f", "-c", "%T", directory],
            raw / "preconditions",
            deadline=10,
        )
        fs = Path(filesystem["stdout_path"]).read_text().strip()
        preconditions = dict(
            available_memory_bytes=memory,
            free_disk_bytes=shutil.disk_usage(directory).free,
            scratch_path=directory,
            filesystem=fs,
            private_disk_backed_scratch=fs not in ("tmpfs", "ramfs")
            and scratch.read_bytes() == b"disk-backed",
            tools={
                n: c.ref(ROOT / ".venv/bin" / n) if (ROOT / ".venv/bin" / n).is_file() else None
                for n in ("python", "pytest", "coverage", "ruff", "mypy")
            },
            current_llm_calls=0,
            optional_hardware_required=False,
            measurement_cap_s=3600,
            validation_reserve_s=900,
            task_cap_s=4800,
        )
    if (
        not preconditions["private_disk_backed_scratch"]
        or memory < 536870912
        or preconditions["free_disk_bytes"] < 1073741824
        or not all(preconditions["tools"].values())
    ):
        failures.append(methods.gate(raw, "cpu_resources", True, False, TASK))
    work = dict(
        raw=str(raw),
        run_date=run_date,
        started_monotonic_ns=started,
        upstream=value,
        input_failures=failures,
        preconditions=preconditions,
        receipts=[],
        children=[],
        grouping_rows=[],
        launch_budget_rows=[],
        sources=[],
        input_references=[],
        partition={},
        collected=[],
        numerical_inference=None,
        prior_failure={},
        outcomes=[],
        phase_spans=[],
        duration_s=0.0,
    )
    registered = methods.read(methods.protocol()["cohort_partition_reference"])
    work["partition"] = registered
    work["grouping_rows"] = groups(registered, registered["original_node_ids"])
    if failures:
        return work
    if c.ref(upstream) != dict(path=str(UPSTREAM), sha256=UPSTREAM_PIN):
        raise ValueError("upstream_identity")
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    if not read_bound_sidecar(upstream, Path(terminal["publication"]["sidecar_path"]))["report"][
        "passed"
    ]:
        raise ValueError("upstream_publication")
    p = methods.protocol()
    partition = methods.read(p["cohort_partition_reference"])
    work.update(
        partition=partition,
        prior_failure=diagnosis(partition),
        grouping_rows=groups(partition, partition["original_node_ids"]),
    )
    historical = c.read(partition["historical_primitive_reference"])
    work["mechanics_fixture"] = historical["mechanics_fixture"]
    refs = [
        c.ref(upstream),
        p["cohort_partition_reference"],
        partition["historical_primitive_reference"],
        historical["mechanics_fixture"],
        p["head_checkpoint_reference"],
        p["learning_data_reference"],
        c.ref(ROOT / methods.PROTOCOL),
    ]
    refs += p["operand_references"]["learning"]
    refs += [
        c.ref(Path(historical["mechanics_fixture"]["path"]).parent / name)
        for name in (
            "stream_rows.json",
            "released_feedback.json",
            "retention_rows.json",
            "coefficient_transitions.json",
        )
    ]
    for reference in refs:
        if c.ref(Path(reference["path"])) != reference:
            raise ValueError("input_identity")
    work["input_references"] = refs + operand_references(
        [c.read(historical["mechanics_fixture"])["cells"]]
    )
    work["sources"] = source_closure(
        [
            ROOT / s
            for s in [
                *OWNED,
                *TESTS,
                old_report.RUNNER,
                "tests/python/conftest.py",
                "pyproject.toml",
            ]
        ]
        + [Path(r["path"]) for r in partition["source_references"]]
    )
    progress("before_numerical_checkpoint_inference")
    work["numerical_inference"] = methods.numerical(p)
    progress("after_numerical_checkpoint_inference", 1, 0)
    return work


def instrument(
    work: Json, name: str, nodes: list[str], sources: list[Json], budget: float, kind: str
) -> Json:
    """Give each child unique data, then seal every SQLite before another combine runs."""
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
    environment = dict(
        COVERAGE_FILE=str(namespace / ".coverage"),
        PYTHONUNBUFFERED="1",
        JAX_PLATFORMS="cpu",
        TMPDIR="/var/tmp",
        CARNOT_EXPERIMENT_ARTIFACT_ROOT="/var/tmp/core8654-artifacts-" + str(time.time_ns()),
    )
    Path(environment["CARNOT_EXPERIMENT_ARTIFACT_ROOT"]).mkdir()
    if work.get("mechanics_fixture"):
        environment["CARNOT_8582_MEASUREMENT"] = work["mechanics_fixture"]["path"]
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
        env={k: v for k, v in environment.items() if k != "PYTHONUNBUFFERED"},
        deadline=budget,
    )
    receipt.update(
        transitive_source_hashes=work["sources"],
        environment=dict(environment, PYTHONUNBUFFERED="1"),
        coverage_config_reference=c.ref(config),
        source_references=sources,
    )
    work["receipts"].append(receipt)
    if junit.is_file():
        receipt["junit_reference"] = c.ref(junit)
    try:
        sealed_nodes = (
            [r["node_id"] for r in terminal_nodes(junit)]
            if kind == "owned" and junit.is_file()
            else nodes
        )
        manifest = c.seal_child(namespace, sources, receipt, sealed_nodes, kind)
        work["children"].append(manifest)
    except (OSError, ValueError) as error:
        work.setdefault("child_errors", []).append(dict(child=name, error=str(error)))
    atomic_json(namespace / "execution.json", receipt)
    return receipt


def execute(work: Json) -> None:
    """Collect once, then launch only admitted disjoint cohort A modules."""
    raw = Path(work["raw"])
    partition = work["partition"]
    targets = partition["original_node_ids"]
    receipt = c.supervise(
        "collection",
        [sys.executable, "-m", "pytest", *OPTIONS, "--collect-only", *targets],
        raw / "collection",
        deadline=90,
    )
    work["receipts"].append(receipt)
    work["collected"] = [
        n
        for n in Path(receipt["stdout_path"]).read_text().splitlines()
        if n.startswith("tests/") and "::" in n
    ]
    plan = groups(partition, work["collected"])
    atomic_json(raw / "collection.json", dict(nodes=work["collected"], receipt=receipt))
    work["collection_reference"] = c.ref(raw / "collection.json")
    for index, group in enumerate(plan):
        progress("cohort_A_module", index, len(plan) - index)
        row = admission(group, work["started_monotonic_ns"], time.monotonic_ns())
        work["launch_budget_rows"].append(row)
        if not row["admitted"]:
            break
        paths = [ROOT / group["module"], *map(Path, group["coverage_sources"])]
        if "objective_checkpoint" in group["module"]:
            paths += [
                ROOT / "python/carnot/verify" / name
                for name in ("objective_checkpoint_8461.py", "objective_derivatives_8461.py")
            ]
        instrument(
            work,
            "core" + str(index),
            group["node_ids"],
            [c.ref(p) for p in paths],
            group["child_budget_s"],
            "A",
        )
        progress("cohort_A_module_finished", index + 1, len(plan) - index - 1)


def recompute(work: Json, scratch: Path) -> None:
    """Rebuild outcomes and source coverage from actual immutable child primitives."""
    for reference in work["sources"] + work["input_references"]:
        if c.ref(Path(reference["path"])) != reference:
            raise ValueError("source_or_operand_identity")
    work["outcomes"] = []
    for manifest in work["children"]:
        receipt = manifest["receipt"]
        if (
            c.ref(Path(receipt["coverage_config_reference"]["path"]))
            != receipt["coverage_config_reference"]
        ):
            raise ValueError("child_config_identity")
        if c.ref(Path(receipt["junit_reference"]["path"])) != receipt["junit_reference"]:
            raise ValueError("junit_identity")
        actual = terminal_nodes(Path(receipt["junit_reference"]["path"]))
        if sorted(r["node_id"] for r in actual) != sorted(manifest["node_ids"]):
            raise ValueError("actual_node_assignment")
        if manifest["kind"] == "A":
            work["outcomes"].extend(actual)
    for kind, key in (("A", "original_coverage"), ("owned", "owned_coverage")):
        manifests = [m for m in work["children"] if m["kind"] == kind]
        if not manifests:
            work[key] = dict(files={}, complete=False)
            continue
        sources = list({r["path"]: r for m in manifests for r in m["sources"]}.values())
        lines = attest(manifests, sources)
        selected = [
            r
            for r in sources
            if r["path"]
            in (
                [str(ROOT / p) for p in OWNED]
                if kind == "owned"
                else [str(ROOT / old_report.RUNNER)]
            )
        ]
        work[key] = old_report.statement_report(lines, selected)
        work[key]["observed_statements"] = sum(
            len(row["executed"]) for row in work[key]["files"].values()
        )
        combined = c.combine_copies(manifests, scratch / kind)
        if not combined["second_combine_invariance"]["passed"]:
            raise ValueError("combine_changed_originals")
    if work.get("collection_reference"):
        collection = c.read(work["collection_reference"])
        if collection["nodes"] != work["collected"]:
            raise ValueError("collection_identity")
        groups(work["partition"], collection["nodes"])


def operand_references(values: list[Any]) -> list[Json]:
    """Follow numerical cell references so a surviving envelope cannot hide a missing operand."""
    pending, result = list(values), {}
    while pending:
        value = pending.pop()
        if isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, dict):
            if set(value) == {"path", "sha256"}:
                if value["path"] in result:
                    if result[value["path"]] != value:
                        raise ValueError("operand_identity")
                    continue
                path = Path(value["path"])
                if c.ref(path) != value:
                    raise ValueError("operand_identity")
                result[value["path"]] = value
                if path.suffix == ".json":
                    pending.append(json.loads(path.read_bytes()))
            else:
                pending.extend(value.values())
    return list(result.values())
