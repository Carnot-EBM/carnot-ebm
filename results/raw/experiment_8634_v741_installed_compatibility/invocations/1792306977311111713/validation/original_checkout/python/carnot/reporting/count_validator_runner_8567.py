"""REQ-VERIFY-8567: bound real validation children and publish exact checked bytes.

Existing receipt and publication helpers own process supervision and atomic
replacement. This adapter adds only the correction study's validation manifest.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import count_validator_8567 as e
from carnot.reporting.count_semantics_evidence_8566 import reference
from carnot.reporting.count_semantics_runner_8566 import (
    coverage_complete,
    exact_union,
    terminal_nodes,
)
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from scripts.adversarial_verify import verify_artifact

Json = dict[str, Any]
CLI = e.ROOT / "scripts/experiments" / (e.NAME + ".py")
TESTS = [
    "tests/python/test_count_validator_correction_8567.py",
    "tests/python/test_count_validator_runner_8567.py",
]
OWNED = [
    "python/carnot/reporting/discrete_count_proof.py",
    "python/carnot/reporting/count_validator_8567.py",
    "python/carnot/reporting/count_validator_runner_8567.py",
    str(CLI.relative_to(e.ROOT)),
]
AFFECTED = [
    "tests/python/test_adversarial_verify_baseline_identity_tautology.py",
    "tests/python/test_adversarial_verify_nested_wall_clock_tautology.py",
    "tests/python/test_count_semantics_8566.py",
    "tests/python/test_count_semantics_runner_8566.py",
]
E2E = [
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py::test_cli_private_success_failure_and_replay",
    "tests/python/test_restricted_decision_audit_8210.py::test_private_cli_and_rehashed_mutations",
]


def progress(phase: str) -> None:
    """Flush each phase so CPU-only work has the same heartbeat as long children."""
    print("[exp8567] " + phase, flush=True)


def prepare(raw: Path) -> Json:
    """Authenticate tools, disk scratch and original immutable controls before work."""
    start = time.monotonic()
    progress("start date=20261016 no_model_load MODEL_SPECS=[] current_llm_calls=0")
    raw.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir="/var/tmp", prefix="carnot-8567-probe-") as temporary:
        probe = Path(temporary) / "probe"
        with probe.open("wb") as stream:
            stream.write(b"disk-backed")
            stream.flush()
            os.fsync(stream.fileno())
        tools = {
            name: (e.ROOT / ".venv/bin" / name).is_file()
            for name in ("python", "pytest", "ruff", "mypy")
        }
        preconditions = dict(
            passed=probe.read_bytes() == b"disk-backed"
            and all(tools.values())
            and shutil.disk_usage(temporary).free > 100000000,
            tools=tools,
            scratch_parent="/var/tmp",
            free_disk_bytes=shutil.disk_usage(temporary).free,
            memory_available=Path("/proc/meminfo").read_text(),
            cuda="optional_not_required",
            board="optional_not_required",
            measurement_deadline_s=3600,
            validation_reserve_s=900,
            task_cap_s=4800,
        )
    work: Json = dict(
        raw=str(raw.absolute()),
        receipts=[],
        controls=[],
        preserved=[],
        sources=[],
        failures=[],
        preconditions=preconditions,
        duration_s=0,
        phases=[],
    )
    try:
        upstream, original, corpus = e.authenticate()
        progress("authenticated_original_authority freeze_controls")
        work.update(
            upstream=upstream,
            evidence=original["evidence"],
            sources=[reference(e.UPSTREAM), *original["evidence"]["sources"]],
            original_count_proof=e.original_proof(original),
            controls=e.classifications(corpus),
            preserved=e.preserved(upstream),
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        work["failures"].append(
            dict(
                check="sealed_authority",
                upstream="exp8566-count-semantics-controls",
                path=str(e.UPSTREAM),
                sha256=e.PIN,
                field="authenticated_controls",
                operator="==",
                expected=True,
                observed=None if isinstance(error, FileNotFoundError) else str(error),
                passed=False,
            )
        )
    work["code"] = [
        reference(e.ROOT / p) for p in [*OWNED, *TESTS, "scripts/adversarial_verify.py"]
    ]
    old = (e.BASE / "old_adversarial_verify.py").read_text()
    new = (e.ROOT / "scripts/adversarial_verify.py").read_text()
    (raw / "correction.diff").write_text(
        "".join(
            difflib.unified_diff(
                old.splitlines(True),
                new.splitlines(True),
                fromfile="old/adversarial_verify.py",
                tofile="new/adversarial_verify.py",
            )
        )
    )
    atomic_json(raw / "controls.json", dict(rows=work["controls"], preserved=work["preserved"]))
    work["duration_s"] = time.monotonic() - start
    work["phases"] = [dict(phase="authenticate_and_measure", duration_s=work["duration_s"])]
    atomic_json(raw / "work.json", work)
    return work


def terminal(path: Path, raw: Path) -> Json:
    """Keep final scanner findings even when publication must reject the candidate."""
    progress("final_bytes_validation")
    scan = verify_artifact(path)
    receipt = child(
        "final_row_lint",
        [sys.executable, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)],
        raw / "terminal_logs",
        deadline=60,
    )
    atomic_json(raw / "final_adversarial.json", scan)
    return dict(
        passed=e.replay(path)
        and not any(f["severity"] == "critical" for f in scan["flags"])
        and receipt["passed"],
        adversarial=scan,
        row_lint=receipt,
    )


def run(output: Path, *, fixture: bool = False) -> int:
    """Private fixtures exercise publication without recursively starting qualification."""
    start = time.monotonic()
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = prepare(raw)
    with TemporaryDirectory(dir="/var/tmp", prefix="carnot-8567-") as temporary:
        if not fixture:
            validate(work, Path(temporary))
        work["duration_s"] = time.monotonic() - start
        work["phases"].append(
            dict(
                phase="validation", duration_s=work["duration_s"] - work["phases"][0]["duration_s"]
            )
        )
        atomic_json(raw / "work.json", work)
        value = e.build(work, output)
        publication = publish_primary(output, value, lambda p: terminal(p, raw))
        consumer = reader_receipt(
            e.TASK,
            output.parent,
            field="count_validator_ready_score",
            expected=int(value["required_checks_passed"]),
        )
        atomic_json(
            raw / "terminal_validation.json", dict(publication=publication, consumer=consumer)
        )
    progress("complete verdict=" + value["verdict_class"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Reject date drift and protect production authority from private test fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261016")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--fixture-output", type=Path)
    args = parser.parse_args(argv)
    if args.date != "20261016" or (
        args.fixture_output and args.fixture_output.resolve().is_relative_to(e.ROOT / "results")
    ):
        return 2
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        progress("cold_replay=" + str(passed))
        return 0 if passed else 1
    return run(
        args.fixture_output or e.ROOT / "results" / (e.NAME + ".json"),
        fixture=args.fixture_output is not None,
    )


def validate(work: Json, scratch: Path) -> None:
    """Collect original IDs once, then validate disjoint children with durable logs."""
    raw, py = Path(work["raw"]), sys.executable
    receipts = work["receipts"]

    def command(name: str, argv: list[str], expected: int = 0, deadline: int = 180) -> Json:
        receipt = child(name, argv, raw / "logs", expected=expected, deadline=deadline)
        receipts.append(receipt)
        return receipt

    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess\ndata_file="
        + str(raw / ".coverage")
        + "\ninclude=\n"
        + "".join(
            "    " + str(e.ROOT / p) + "\n" for p in [*OWNED, "scripts/adversarial_verify.py"]
        )
    )
    pytest = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    collection = command(
        "collection", [py, "-m", "pytest", *pytest, "--collect-only", *TESTS, *AFFECTED, *E2E]
    )
    nodes = [
        s
        for s in Path(collection["stdout_path"]).read_text().splitlines()
        if s.startswith("tests/") and "::" in s
    ]
    groups = [[n for n in nodes if n.split("::")[0] in paths] for paths in (TESTS, AFFECTED)]
    groups += [[n for n in nodes if n == spec] for spec in E2E]
    work["node_union"] = dict(original_node_ids=nodes, child_node_ids=groups)
    outcomes = []
    for name, group in zip(("owned", "affected", "e2e018", "e2e021"), groups, strict=True):
        junit = raw / (name + ".xml")
        command(
            name,
            [
                py,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                *pytest,
                "--junitxml=" + str(junit),
                *group,
            ],
            deadline=600,
        )
        if junit.is_file():
            outcomes.extend(terminal_nodes(junit))
    work["node_union"]["terminal_node_outcomes"] = outcomes
    work["union_passed"] = (
        bool(nodes)
        and exact_union(nodes, groups)
        and exact_union(nodes, [[r["node_id"] for r in outcomes]])
        and all(r["outcome"] == "passed" for r in outcomes)
    )
    command(
        "ruff_check", [py, "-m", "ruff", "check", *OWNED, "scripts/adversarial_verify.py", *TESTS]
    )
    command(
        "ruff_format",
        [py, "-m", "ruff", "format", "--check", *OWNED, "scripts/adversarial_verify.py", *TESTS],
    )
    command(
        "mypy",
        [
            py,
            "-m",
            "mypy",
            "--strict",
            "--follow-imports=silent",
            *OWNED,
            "scripts/adversarial_verify.py",
        ],
    )
    command("spec", [py, "scripts/check_spec_coverage.py", "--files", *TESTS])
    upstream, _, corpus = e.authenticate()
    for index, case in enumerate(corpus["cases"]):
        progress(f"cli_controls completed={index} pending={len(corpus['cases']) - index}")
        path = raw / ("control_" + case["id"] + ".json")
        atomic_json(path, e.candidate(corpus, case))
        command(
            "cli_" + case["id"],
            [py, "scripts/adversarial_verify.py", "--json", str(path)],
            expected=0 if case["id"] == "unequal" else 1,
            deadline=60,
        )
    erasure = (
        "import runpy; from carnot.reporting import count_validator_8567 as e; "
        'u,w,c=e.authenticate(); v=e.candidate(c,c["cases"][0]); '
        'old=runpy.run_path(str(e.BASE/"old_adversarial_verify.py"))["check_tautology"]; '
        'assert any(f["severity"]=="critical" for f in e.findings(v,old)); '
        'v["discrete_count_proof"]["claims"]["observed_alpha"]["value"]=2399; '
        'assert any(f["severity"]=="critical" for f in e.findings(v,e.av.check_tautology)); '
        'assert any(f["severity"]=="critical" for f in e.findings(v,old))'
    )
    command("erasure", [py, "-u", "-c", erasure])
    for name, mutation in [
        ("cold", "none"),
        ("missing", "absent"),
        ("owned_failure", "owned"),
        ("rehashed", "rows"),
    ]:
        atomic_json(raw / "work.json", work)
        path = scratch / (name + ".json")
        value = e.build(work, path)
        if mutation == "absent":
            value["primitive_reference"]["path"] = str(scratch / "absent")
        elif mutation == "owned":
            value["required_checks_passed"] = True
        elif mutation == "rows":
            value["old_new_classification_rows"] = []
        value["reproducibility_checksum"] = e.checksum(value)
        atomic_json(path, value)
        command(
            name,
            [py, "-u", str(CLI), "--cold-replay", str(path)],
            expected=0 if name == "cold" else 1,
        )
    atomic_json(raw / "work.json", work)
    candidate = scratch / "candidate.json"
    atomic_json(candidate, e.build(work, candidate))
    command(
        "adversarial", [py, "scripts/adversarial_verify.py", "--json", str(candidate)], deadline=60
    )
    command(
        "row_lint",
        [py, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
        deadline=60,
    )
    command("combine", [py, "-m", "coverage", "combine", "--rcfile=" + str(config), str(raw)])
    command(
        "coverage",
        [py, "-m", "coverage", "json", "--rcfile=" + str(config), "-o", str(raw / "coverage.json")],
    )
    measured = json.loads((raw / "coverage.json").read_bytes())
    files = {k: v for k, v in measured["files"].items() if k in OWNED}
    old = (e.BASE / "old_adversarial_verify.py").read_text().splitlines()
    new = (e.ROOT / "scripts/adversarial_verify.py").read_text().splitlines()
    added = {
        n + 1
        for tag, _, _, a, b in difflib.SequenceMatcher(a=old, b=new).get_opcodes()
        if tag in {"insert", "replace"}
        for n in range(a, b)
    }
    missing = set(measured["files"]["scripts/adversarial_verify.py"]["missing_lines"])
    work["new_validator_coverage"] = dict(
        added_lines=sorted(added), missing_added_lines=sorted(added & missing)
    )
    work["coverage_complete"] = coverage_complete(dict(files=files), OWNED) and not (
        added & missing
    )
