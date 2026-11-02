"""REQ-VERIFY-8624: bound work and publish only independently checked bytes."""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

import coverage
from coverage.parser import PythonParser
import yaml
from carnot.reporting import arc_observer_8624 as e
from carnot.reporting import arc_redirect_runner_8509 as validation
from carnot.reporting.arc_observer_fixtures_8624 import qualify
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from scripts.conductor_gates import evaluate_gates

Json = dict[str, Any]


def progress(phase: str) -> None:
    """Every bounded phase has an immediate visible start and end."""
    print(f"[exp8624] phase={phase}", flush=True)


def authority(root: Path) -> Json:
    """Current YAML and design must bind the same twelve registered tasks."""
    plan = yaml.safe_load((root / "research-roadmap.yaml").read_bytes())
    text = (root / "openspec/change-proposals/research-roadmap-vNEXT.md").read_text()
    contract = json.loads(
        text.split("<!-- V740_TASK_CONTRACT_START -->", 1)[1]
        .split("```json", 1)[1]
        .split("```", 1)[0]
    )
    expected = [f"exp{i}-" for i in range(8615, 8627)]
    if (
        plan["milestone"] != "2026.10.740"
        or len(plan["tasks"]) != 12
        or any(
            not t["id"].startswith(prefix)
            for t, prefix in zip(plan["tasks"], expected, strict=True)
        )
        or canonical_hash(plan["tasks"]) != canonical_hash(contract["tasks"])
    ):
        raise ValueError("current_design_authority")
    task = next(t for t in plan["tasks"] if t["id"] == e.TASK)
    gates = evaluate_gates(task, root / "results")
    return dict(
        task_sha256=canonical_hash(task),
        canonical_tasks_sha256=canonical_hash(plan["tasks"]),
        passed=gates.passed,
        gates=[asdict(g) for g in gates.gates_evaluated],
    )


def validate_checks(raw: Path, private: Path) -> list[Json]:
    """Preserve the exact collected E2E union and measure only newly owned code."""
    with patch.multiple(validation, e=e, TESTS=e.TESTS):
        receipts = list(validation.validate_checks(raw, private))
    measured, owned = live_statements(raw)
    receipts.append(
        child(
            "live_hook_tests",
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--data-file=" + str(measured.config.data_file),
                "--include=" + str(e.ROOT / e.LIVE),
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                e.TESTS[1] + "::test_actual_live_wrapper_preserves_all_actions_and_receipts",
            ],
            raw / "validation",
            deadline=180,
        )
    )
    spec = raw / "live_statement_scope.json"
    atomic_json(
        spec,
        dict(
            path=str(e.ROOT / e.LIVE),
            sha256=sha256_file(e.ROOT / e.LIVE),
            executable_new_statements=owned,
            data_file=str(measured.config.data_file),
        ),
    )
    code = (
        "import coverage,json,sys; from pathlib import Path; "
        "s=json.loads(Path(sys.argv[1]).read_text()); c=coverage.Coverage(data_file=s['data_file']); "
        "c.load(); executed=set(c.get_data().lines(s['path']) or []); "
        "missing=sorted(set(s['executable_new_statements'])-executed); "
        "print(json.dumps(dict(statements=s['executable_new_statements'],missing=missing,"
        "percent=100 if not missing and s['executable_new_statements'] else 0))); "
        "raise SystemExit(int(bool(missing) or not s['executable_new_statements']))"
    )
    receipts.append(
        child(
            "live_statement_coverage",
            [sys.executable, "-c", code, str(spec)],
            raw / "validation",
            deadline=30,
        )
    )
    return receipts


def live_statements(raw: Path) -> tuple[coverage.Coverage, list[int]]:
    """Coverage.py executable statements exclude multiline continuation anchors."""
    path = e.ROOT / e.LIVE
    tree = ast.parse(path.read_text())
    added: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "carnot.agentic.arc_outcome_observer"
        ):
            added.add(node.lineno)
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "notify"
        ):
            added.add(node.lineno)
        if isinstance(node, ast.If) and "_arc_outcome_observer" in ast.unparse(node.test):
            added.add(node.lineno)
        if isinstance(node, ast.FunctionDef) and node.name == "install_arc_outcome_observer":
            added.update(n.lineno for n in ast.walk(node) if isinstance(n, ast.stmt))
        if isinstance(node, ast.FunctionDef) and node.name == "_maybe_supervise_trajectory":
            added.update(
                n.lineno
                for n in node.body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "applied" for t in n.targets)
            )
    parser = PythonParser(filename=str(path))
    parser.parse_source()
    owned = sorted(parser.statements & {parser.first_line(n) for n in added})
    measured = coverage.Coverage(
        data_file=str(raw / "live.coverage"), config_file=False, include=[str(path)]
    )
    return measured, owned


def publish(value: Json, output: Path, raw: Path) -> Json:
    """Check replay, original validators and both live readers before replacement."""

    def validate(candidate: Path) -> Json:
        checks = [
            child(name, argv, raw / "terminal", deadline=180)
            for name, argv in [
                (
                    "terminal_replay",
                    [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(candidate)],
                ),
                (
                    "adversarial",
                    [sys.executable, "scripts/adversarial_verify.py", "--json", str(candidate)],
                ),
                (
                    "strict_rows",
                    [
                        sys.executable,
                        "scripts/verdict_row_consistency_lint.py",
                        "--strict",
                        str(candidate),
                    ],
                ),
            ]
        ]
        with TemporaryDirectory(prefix="exp8624-readers-", dir="/var/tmp") as directory:
            private = Path(directory) / output.name
            private.write_bytes(candidate.read_bytes())
            reader = reader_receipt(
                e.TASK,
                private.parent,
                field="arc_observer_ready_score",
                expected=value["arc_observer_ready_score"],
            )
        report = dict(
            passed=all(r["passed"] for r in checks) and reader["passed"],
            checks=checks,
            consumer=reader,
        )
        atomic_json(raw / "publication_attempt.json", report)
        return report

    publication = publish_primary(output, value, validate)
    reader = reader_receipt(
        e.TASK,
        output.parent,
        field="arc_observer_ready_score",
        expected=value["arc_observer_ready_score"],
    )
    side = dict(publication=publication, consumer=reader, passed=reader["passed"])
    atomic_json(Path(value["terminal_validation_sidecar_path"]), side)
    return side


def main(argv: list[str] | None = None) -> int:
    """Private controls run real children but never invoke validation controllers."""
    progress("start_no_model_load_MODEL_SPECS_empty_current_model_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--owned-failure", action="store_true")
    args = parser.parse_args(argv)
    try:
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        parser.error("--date requires YYYYMMDD")
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        try:
            accepted = e.replay(json.loads(args.cold_replay.read_bytes()))
        except (ValueError, OSError, KeyError, TypeError):
            accepted = False
        progress("cold_replay_complete")
        return 0 if accepted else 1
    if args.private_control and args.root.exists():
        parser.error("private controls require an absent input root")
    started = time.monotonic_ns()
    raw = args.output.parent / "raw" / args.output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    with TemporaryDirectory(prefix="carnot8624-", dir="/var/tmp") as directory:
        private = Path(directory)
        with patch.multiple(validation, e=e, TESTS=e.TESTS):
            failures, resources, hashes = validation.preconditions(args.root, private)
            if (args.root / "research-roadmap.yaml").is_file():
                try:
                    current = authority(args.root)
                    resources["current_authority"] = current
                    for gate in current["gates"]:
                        if not gate["passed"]:
                            failures.append(
                                dict(
                                    check="current_methods",
                                    upstream=gate["upstream"],
                                    path=gate["artifact_path"],
                                    hash=gate["artifact_sha256"],
                                    field=gate["artifact_field"],
                                    operator=gate["op"],
                                    expected=gate["expected"],
                                    observed=gate["actual"],
                                    passed=False,
                                )
                            )
                except (OSError, ValueError, KeyError, IndexError) as exc:
                    failures.append(
                        validation.gate(
                            "current_design_authority",
                            args.root / "openspec/change-proposals/research-roadmap-vNEXT.md",
                            "registered_twelve_task_contract",
                            e.TASK_PIN,
                            str(exc),
                        )
                    )
        for name in [
            "research-roadmap.yaml",
            "openspec/change-proposals/research-roadmap-vNEXT.md",
        ]:
            source = args.root / name
            if source.is_file():
                snapshot = raw / source.name
                shutil.copyfile(source, snapshot)
                hashes.pop(str(source), None)
                hashes[str(snapshot)] = sha256_file(snapshot)
        progress("measurement_before")
        history: Json = {}
        fixtures = []
        for name, digest in e.UPSTREAM.items():
            source = args.root / name
            observed = sha256_file(source) if source.is_file() else None
            if observed != digest:
                failures.append(
                    dict(
                        check="required_original_input",
                        upstream=name.split("_")[1],
                        path=str(source),
                        hash=observed,
                        field="sha256",
                        operator="==",
                        expected=digest,
                        observed=observed,
                        passed=False,
                    )
                )
        try:
            history = e.historical(args.root)
        except (OSError, ValueError) as exc:
            failure = validation.gate(
                "original_history",
                args.root / next(iter(e.UPSTREAM)),
                "original_bytes_and_authority",
                list(e.UPSTREAM.values())[0],
                str(exc),
            )
            failure["upstream"] = "exp8588-arc-capability-panel"
            failures.append(failure)
        if not args.private_control:
            fixtures = qualify()
        primitive_path = raw / "primitives.json"
        atomic_json(primitive_path, dict(fixtures=fixtures, history=history))
        elapsed = (time.monotonic_ns() - started) / 1e9
        if elapsed > 3600:
            raise TimeoutError("measurement_cap")
        progress("measurement_after_validation_before")
        registry = e.ROOT / "ops/arc_solve_registry.yaml"
        work = dict(
            run_date=args.date,
            input_root=str(args.root),
            failures=failures,
            primitive_reference=dict(path=str(primitive_path), sha256=sha256_file(primitive_path)),
            preconditions_checked=resources,
            duration_s=elapsed,
            phase_spans=[
                dict(
                    phase="preconditions_and_measurement",
                    started_monotonic_ns=started,
                    ended_monotonic_ns=time.monotonic_ns(),
                    duration_s=elapsed,
                )
            ],
            source_artifact_hashes=dict(
                hashes,
                **{
                    str(args.root / p): sha256_file(args.root / p)
                    for p, h in e.UPSTREAM.items()
                    if (args.root / p).is_file()
                },
            ),
            code_config_hashes={
                str(e.ROOT / p): sha256_file(e.ROOT / p) for p in [*e.OWNED, *e.TESTS, e.LIVE]
            },
            raw_shard_hashes={
                str(primitive_path): sha256_file(primitive_path),
                **history.get("historical_hashes", {}),
            },
            registry_precheck=dict(
                path=str(registry),
                sha256=sha256_file(registry),
                checked_before_credit=True,
                new_solve_credit=0,
                reason="scripted_fixtures_never_supply_solve_credit",
            ),
        )
        work_path = raw / "work.json"
        atomic_json(work_path, work)
        if args.private_control:
            receipts = [
                child(
                    "private_owned_check",
                    [
                        sys.executable,
                        "-c",
                        "raise SystemExit(" + str(int(args.owned_failure)) + ")",
                    ],
                    raw / "validation",
                    deadline=30,
                )
            ]
        else:
            receipts = validate_checks(raw, private)
            value = e.build(work_path, receipts, args.output)
            changed = deepcopy(value)
            changed["completed_count"] = 991
            changed.pop("reproducibility_checksum")
            changed["reproducibility_checksum"] = canonical_hash(changed)
            tamper = raw / "rehashed_candidate.json"
            atomic_json(tamper, changed)
            for name, arguments, expected in [
                ("missing_input", ["--cold-replay", str(raw / "missing.json")], 1),
                ("owned_error", ["--deliberate-error"], 7),
                ("rehashed_tamper", ["--cold-replay", str(tamper)], 1),
            ]:
                receipts.append(
                    child(
                        name,
                        [sys.executable, "-u", str(e.ROOT / e.CLI), *arguments],
                        raw / "controls",
                        deadline=180,
                        expected=expected,
                    )
                )
        progress("validation_after_publication_before")
        side = publish(e.build(work_path, receipts, args.output), args.output, raw)
        progress("publication_after")
        return 0 if side["passed"] else 1
