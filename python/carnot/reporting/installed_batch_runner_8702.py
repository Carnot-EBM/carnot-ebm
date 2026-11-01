"""REQ-REPORT-8702: bounded execution retains evidence before checked publication."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
import json
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from coverage import Coverage

from carnot.reporting import installed_batch_8702 as e
from carnot.reporting import installed_batch_measure_8702 as measurement
from carnot.reporting import installed_checkout_8660 as historical
from carnot.reporting.installed_validation_8660 import historical_command
from carnot.reporting import v746_contract_runner as shared
from carnot.reporting import v746_retained_evidence as retained
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child, progress
from scripts.adversarial_verify import verify_artifact

Json = dict[str, Any]
OPTIONS = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]


def private_namespace(prefix: list[str], scratch: Path) -> list[str]:
    """Keep historical bind-target creation and incidental writes off the live checkout."""
    if prefix[:4] != ["bwrap", "--bind", "/", "/"]:
        raise ValueError("historical_namespace")
    upper, work = scratch / "namespace-upper", scratch / "namespace-work"
    upper.mkdir()
    work.mkdir()
    return (
        prefix[:4]
        + ["--overlay-src", str(e.ROOT), "--overlay", str(upper), str(work), str(e.ROOT)]
        + prefix[4:]
    )


def validate(raw: Path, scratch: Path, sources: list[Json], original: list[str]) -> Json:
    """Collect once and keep original compatibility assertions in historical namespaces."""
    suites = list(
        dict.fromkeys(
            [n.split("::")[0] for n in original]
            + e.TESTS
            + [
                "tests/python/test_installed_lean_8633.py",
                "tests/python/test_primary_publication_7928.py",
            ]
        )
    )
    receipts = [
        child(
            "collect_once",
            [sys.executable, "-m", "pytest", *OPTIONS, "--collect-only", *suites],
            raw / "checks",
            deadline=120,
        )
    ]
    nodes = [
        n
        for n in Path(receipts[0]["stdout_path"]).read_text().splitlines()
        if n.startswith("tests/") and "::" in n
    ]
    e.atomic_json(raw / "inventory.json", dict(nodes=nodes))
    checks = [
        dict(
            name="exact_collection",
            passed=receipts[0]["passed"]
            and set(original) <= set(nodes)
            and len(nodes) == len(set(nodes)),
        )
    ]
    manifest = historical.overlay(raw / "historical-overlay")
    children, owned = [], []
    for i, suite in enumerate(suites):
        selected = [n for n in nodes if n.startswith(suite + "::")]
        private = scratch / str(i)
        private.mkdir()
        config = raw / f"coverage-{i}.ini"
        config.write_text(
            f"[run]\nparallel=true\npatch=subprocess\ndata_file={private / '.coverage'}\n[report]\nexclude_lines=\n"
        )
        prefix = (
            historical_command(manifest, raw, suite) if any(n in original for n in selected) else []
        )
        if prefix:
            prefix = private_namespace(prefix, private)
        plugin = (
            ["-p", "carnot.reporting.installed_checkout_8660"]
            if "lean_runner_8609" in suite
            else []
        )
        receipt = child(
            f"tests-{i}",
            [
                *prefix,
                "env",
                f"CARNOT_8660_MATERIALIZATION_LOG={raw / 'materialized'}",
                f"COVERAGE_RCFILE={config}",
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                *plugin,
                *OPTIONS,
                *selected,
                "--basetemp=" + str(private / "tmp"),
                "--junitxml=" + str(private / "junit.xml"),
            ],
            raw / "checks",
            deadline=420,
        )
        receipts.append(receipt)
        role = "owned" if suite in e.TESTS else "original" if prefix else "consumer"
        ref = retained.retain(receipt, selected, private, raw / "children" / str(i), sources, role)
        children.append(ref)
        if role == "owned":
            owned.append(ref)
    custody = retained.aggregate(children, nodes)
    checks += [
        dict(name="exact_disjoint_children", passed=custody["passed"]),
        dict(
            name="original122",
            passed=not original
            or len(original) == 122
            and set(original) <= set(custody["node_ids"]),
        ),
    ]
    combined = retained.combine(owned, raw / "combined")
    cov = Coverage(data_file=combined["combined_reference"]["path"], config_file=False)
    cov.load()
    files = []
    for name in e.OWNED:
        _, statements, _, missing, _ = cov.analysis2(str(e.ROOT / name))
        files.append(dict(path=name, statements=len(statements), missing=missing))
    name = "python/carnot/inference/resident_protocol.py"
    _, statements, _, missing, _ = cov.analysis2(str(e.ROOT / name))
    added = e.protocol_added_lines()
    files.append(
        dict(
            path=name, statements=len(set(statements) & added), missing=sorted(set(missing) & added)
        )
    )
    coverage = dict(
        complete=all(r["statements"] and not r["missing"] for r in files),
        files=files,
        combined=combined,
    )
    checks.append(dict(name="owned_coverage", passed=coverage["complete"]))
    owned_paths = e.OWNED + ["python/carnot/inference/resident_protocol.py"]
    for name, args in [
        ("ruff", ["-m", "ruff", "check", *owned_paths, *e.TESTS]),
        ("format", ["-m", "ruff", "format", "--check", *owned_paths, *e.TESTS]),
        ("mypy", ["-m", "mypy", "--strict", "--follow-imports=silent", *owned_paths]),
        ("spec", ["scripts/check_spec_coverage.py", "--files", *e.TESTS]),
    ]:
        receipt = child(name, [sys.executable, *args], raw / "checks", deadline=120)
        receipts.append(receipt)
        checks.append(dict(name=name, passed=receipt["passed"]))
    checks += [dict(name=r["name"], passed=r["passed"]) for r in receipts]
    return dict(
        checks=checks,
        receipts=receipts,
        coverage=coverage,
        children=children,
        original_node_union=dict(
            intended_count=len(original),
            completed_count=len(set(original) & set(custody["node_ids"])),
            passed=checks[2]["passed"],
        ),
    )


def replay(path: Path, authority: Json | None = None) -> bool:
    """Rebuild the report from separately sealed primitives and authenticate all operands."""
    try:
        value = json.loads(path.read_bytes())
        if authority is None or value["primitive_reference"] != authority:
            return False
        work = e.load(value["primitive_reference"])
        seal = e.load(work["seal_reference"])
        if seal != {k: v for k, v in work.items() if k != "seal_reference"}:
            return False
        refs = (
            work["sources"] + work["inputs"]["references"] + work["measurement"].get("traces", [])
        )
        refs += [e.load(ref)["stderr_reference"] for ref in work["measurement"].get("traces", [])]
        refs += [
            r
            for r in [
                work["measurement"].get("installed_wheel_reference"),
                work["measurement"].get("loaded_extension_reference"),
                work["measurement"].get("config_reference"),
                work["measurement"].get("installed_identity_reference"),
            ]
            if r
        ]
        if any(
            not Path(r["path"]).is_file() or e.reference(Path(r["path"]))["sha256"] != r["sha256"]
            for r in refs
            if r.get("sha256")
        ):
            return False
        validation = work["validation"]
        if validation["children"]:
            nodes = [n for ref in validation["children"] for n in e.load(ref)["node_ids"]]
            if not retained.aggregate(validation["children"], nodes)["passed"]:
                return False
        return e.build(work, Path(work["output"])) == value
    except (OSError, ValueError, KeyError, TypeError):
        return False


def terminal(path: Path, raw: Path) -> Json:
    """Run unchanged validators and preserve their actual findings and severities."""
    value = json.loads(path.read_bytes())
    bad = dict(value, completed_count=value["completed_count"] + 1)
    bad["reproducibility_checksum"] = e.canonical_hash(bad)
    e.atomic_json(raw / "tampered.json", bad)
    receipts = []
    authority = value["primitive_reference"]
    replay_args = ["--authority", authority["path"], authority["sha256"]]
    for name, args, expected in [
        ("cold", [e.CLI, "--cold-replay", str(path), *replay_args], 0),
        ("missing", [e.CLI, "--cold-replay", str(raw / "missing.json"), *replay_args], 1),
        ("tampered", [e.CLI, "--cold-replay", str(raw / "tampered.json"), *replay_args], 1),
        (
            "adversarial",
            ["scripts/adversarial_verify.py", "--json", str(path)],
            int(bool(value["adversarial_findings"])),
        ),
        ("rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(path)], 0),
    ]:
        receipts.append(
            child(
                name,
                [sys.executable, "-u", *args],
                raw / "terminal",
                expected=expected,
                deadline=120,
            )
        )
    findings = json.loads(Path(receipts[3]["stdout_path"]).read_bytes())
    return dict(
        passed=all(r["passed"] for r in receipts)
        and findings["reports"][0]["flags"] == value["adversarial_findings"],
        receipts=receipts,
        actual_findings=findings,
    )


def run(output: Path, *, root: Path = e.ROOT, private: bool = False) -> int:
    """Reserve validation time and publish only a stable reduction of immutable evidence."""
    started, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    with TemporaryDirectory(prefix="installed8702-", dir="/var/tmp") as directory:
        scratch = Path(directory)
        pre = shared.preconditions(scratch)
        inputs = e.inputs(root)
        sources = shared.source_hashes() + [e.reference(e.ROOT / p) for p in e.TESTS]
        suites = list(
            dict.fromkeys(
                [n.split("::")[0] for n in inputs.get("original_nodes", [])]
                + e.TESTS
                + [
                    "tests/python/test_installed_lean_8633.py",
                    "tests/python/test_primary_publication_7928.py",
                ]
            )
        )
        for i in range(len(suites)):
            config = raw / f"coverage-{i}.ini"
            config.write_text(
                f"[run]\nparallel=true\npatch=subprocess\ndata_file={scratch / str(i) / '.coverage'}\n[report]\nexclude_lines=\n"
            )
            sources.append(e.reference(config))
        e.atomic_json(
            raw / "premeasurement.json", dict(sources=sources, inputs=inputs, preconditions=pre)
        )
        measured: Json = {}
        error = None
        try:
            if inputs["passed"]:
                if (
                    not pre["disk_backed"]
                    or not pre["writable"]
                    or pre["free_memory_kib"] < 262144
                    or pre["free_disk_bytes"] < 268435456
                ):
                    raise ValueError("cpu_scratch_memory")
                measured = measurement.measure(
                    raw / "measurement", scratch, inputs, deadline_ns=started + 3600 * 10**9
                )
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            TimeoutError,
            subprocess.TimeoutExpired,
        ) as failure:
            error = str(failure)
        boundary = time.monotonic_ns()
        if boundary - started > 3600 * 10**9:
            error = "measurement_cap_exceeded"
        validation = dict(
            checks=[dict(name="private_control", passed=True)],
            receipts=[],
            coverage=dict(complete=False, files=[]),
            children=[],
        )
        if not private:
            validation = validate(
                raw,
                scratch,
                [
                    e.reference(e.ROOT / p)
                    for p in e.OWNED + ["python/carnot/inference/resident_protocol.py"] + e.TESTS
                ],
                inputs.get("original_nodes", []),
            )
        unchanged = all(e.reference(Path(r["path"])) == r for r in sources)
        validation["checks"] += [
            dict(name="source_seal", passed=unchanged),
            dict(
                name="installed_parity",
                passed=measured.get("passed", False) or not inputs["passed"],
            ),
        ]
        ended = time.monotonic_ns()
        work = dict(
            raw=str(raw),
            output=str(output.absolute()),
            invocation_id=raw.name,
            sources=sources,
            inputs=inputs,
            preconditions=pre,
            measurement=measured,
            validation=validation,
            owned_error=error,
            duration_s=(ended - started) / 1e9,
            started_at_utc=wall,
            ended_at_utc=datetime.now(UTC).isoformat(),
            phase_spans=[
                dict(phase=n, start_monotonic_ns=a, end_monotonic_ns=b, duration_s=(b - a) / 1e9)
                for n, a, b in [
                    ("preconditions_and_measurement", started, boundary),
                    ("validation", boundary, ended),
                ]
            ],
        )
        e.atomic_json(raw / "work.json", work)
        candidate = raw / "preterminal.json"
        e.atomic_json(candidate, e.build(work, output))
        work["adversarial_findings"] = verify_artifact(candidate)["flags"]
        e.atomic_json(raw / "invocation_seal.json", work)
        (raw / "invocation_seal.json").chmod(0o444)
        work["seal_reference"] = e.reference(raw / "invocation_seal.json")
        e.atomic_json(raw / "work.json", work)
        value = e.build(work, output)
        publication = publish_primary(output, value, lambda path: terminal(path, raw))
        e.atomic_json(
            Path(value["terminal_validation_sidecar_path"]), dict(publication=publication)
        )
    progress("8702_published_" + value["verdict_class"], 1, 0)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Reject private-output aliases before creating any evidence or subprocess."""
    progress("8702_start_no_model_load_MODEL_SPECS=[]_language_model_calls=0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261101")
    parser.add_argument("--private-output", type=Path)
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--authority", nargs=2)
    args = parser.parse_args(argv)
    if args.date != "20261101":
        return 2
    if args.private_output and (
        args.private_output.absolute().is_relative_to(e.ROOT)
        or args.private_output.resolve().is_relative_to(e.ROOT)
    ):
        return 2
    if args.cold_replay:
        authority = dict(zip(["path", "sha256"], args.authority)) if args.authority else None
        return int(not replay(args.cold_replay, authority))
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        return run(
            args.private_output or e.ROOT / "results" / (e.NAME + ".json"),
            root=args.root,
            private=bool(args.private_output),
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
