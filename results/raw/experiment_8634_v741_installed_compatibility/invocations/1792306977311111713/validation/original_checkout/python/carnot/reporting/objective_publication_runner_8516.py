"""REQ-VERIFY-8516: bounded children bind a new primary without rewriting history."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import objective_checkpoint_runner_8489 as old_runner
from carnot.reporting import objective_checkpoint_runner_8461 as shipped_runner
from carnot.reporting import objective_publication_8516 as e
from carnot.reporting import v717_contract_runner as scaffold
from carnot.reporting import v726_feature_panel_runner as validators
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.historical_consumer_roots_8402 import assertion_sources, parse_outcomes
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, execute
from carnot.reporting.v721_capstone_evidence import frozen_inputs

Json = dict[str, Any]
COMMON = ["-n", "0", "-o", "addopts=", "--no-cov", "-vv"]
E2E_TESTS = [
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_hard_exit_learning_qualification_8206.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]


def manifest(private: Path) -> list[Json]:
    """Collect the exact original union first; coverage includes only newly owned statements."""
    private.mkdir(parents=True, exist_ok=True)
    with patch.object(scaffold, "m", e):
        plan = list(scaffold.manifest(private))
    config = private / "coverage.ini"
    config.write_text(config.read_text().replace("patch=subprocess", "patch=subprocess,_exit"))
    groups: list[list[str]] = [[] for _ in range(6)]
    receipts = []
    suites = [*e.history.ORIGINAL_TESTS, e.history.TEST, *old_runner.STORE_TESTS, *E2E_TESTS]
    for suite in suites:
        nodes, receipt = old_runner.collection.collect(private, suite)
        receipts.append(receipt)
        if suite == old_runner.STORE_TESTS[1]:
            groups[3].extend(nodes[:12])
            groups[4].extend(nodes[12:])
        else:
            index = (
                0
                if suite in e.history.ORIGINAL_TESTS
                else 1
                if suite == e.history.TEST
                else 2
                if suite == old_runner.STORE_TESTS[0]
                else 5
            )
            groups[index].extend(nodes)
    union = [n for group in groups for n in group]
    atomic_json(
        private / "historical_collection.json",
        dict(
            node_ids=union,
            shards=groups,
            union_equal=True,
            disjoint=len(union) == len(set(union)),
            total_deadline_s=1800,
            shard_deadline_s=600,
            heartbeat_s=30,
            assertion_sources=assertion_sources(union),
            collection_receipts=receipts,
            baseline_owned=e.history.OWNED,
            added_owned=e.OWNED,
        ),
    )
    plan = [p for p in plan if p["name"] != "private_E2E018_consumers"]
    shards = [
        dict(
            name=f"historical_shard_{i}",
            expected=0,
            deadline=600,
            scope="owned",
            collected_node_ids=group,
            argv=[
                str(e.ROOT / ".venv/bin/python"),
                "-u",
                str(e.ROOT / e.CLI),
                "--historical-tests",
                *COMMON,
                *group,
                "--basetemp=" + str(private / f"historical-{i}"),
            ],
        )
        for i, group in enumerate(groups)
    ]
    plan[1:1] = shards + [
        dict(
            name="historical_union",
            expected=0,
            deadline=30,
            scope="owned",
            argv=[
                str(e.ROOT / ".venv/bin/python"),
                "-u",
                str(e.ROOT / e.CLI),
                "--historical-summary",
                str(private / "historical_collection.json"),
                "--receipts",
                str(private / "historical_receipts"),
            ],
        )
    ]
    next(p for p in plan if p["name"] == "owned_tests")["deadline"] = 900
    return plan


def historical(argv: list[str], *, script: bool = False) -> int:
    """Use original sealed authority for old tests and preserve their real CLI descendants."""
    import pytest

    with TemporaryDirectory(prefix="exp8516-history-", dir="/var/tmp") as directory:
        scratch = Path(directory)
        refs = []
        for ref in e.methods.protocol()["v730_authority_references"]:
            refs.append(
                dict(
                    ref,
                    exists=True,
                    snapshot_path=ref["path"],
                    path=ref["source_path"],
                    authority_alias=str(e.ROOT / e.methods.DESIGN)
                    if ref["path"].endswith(".md")
                    else ref["source_path"],
                )
            )
        popen = subprocess.Popen

        def start(args: Any, *pos: Any, **kw: Any) -> Any:
            original_cli = str(e.ROOT / e.history.CLI)
            if isinstance(args, (list, tuple)) and original_cli in args:
                index = args.index(original_cli)
                args = [*args[:index], str(e.ROOT / e.CLI), "--historical-exec", *args[index:]]
            return popen(args, *pos, **kw)

        with frozen_inputs(refs, scratch), patch.object(subprocess, "Popen", start):
            if script:
                previous = sys.argv
                sys.argv = argv
                try:
                    runpy.run_path(argv[0], run_name="__main__")
                except SystemExit as error:
                    return int(error.code) if isinstance(error.code, int) else 0
                finally:
                    sys.argv = previous
                return 0
            if any(n.startswith(e.history.ORIGINAL_TESTS[0] + "::") for n in argv):
                return int(old_runner.original_worker(argv))
            if any(n.startswith(s + "::") for n in argv for s in old_runner.STORE_TESTS):
                stored = [
                    n for n in argv if any(n.startswith(s + "::") for s in old_runner.STORE_TESTS)
                ]
                rest = [n for n in argv if n not in stored]
                status = old_runner.original_worker([*COMMON, *stored], store=True)
                return int(
                    status
                    or (pytest.main(rest) if any(n.startswith("tests/") for n in rest) else 0)
                )
            return int(pytest.main(argv))


def summarize(ledger: Path, receipts: Path) -> int:
    """Passed node identities, real exits and the total budget independently qualify the union."""
    try:
        value = json.loads(ledger.read_bytes())
        rows = [
            json.loads((receipts / f"historical_shard_{i}.receipt.json").read_bytes())
            for i in range(len(value["shards"]))
        ]
        union = [n for group in value["shards"] for n in group]
        passed = union == value["node_ids"] and len(union) == len(set(union)) and bool(union)
        passed = passed and sum(r["duration_s"] for r in rows) <= 1800
        for row, group in zip(rows, value["shards"], strict=True):
            outcomes = parse_outcomes(Path(row["stdout_path"]))
            passed = (
                passed
                and row["passed"]
                and not row["timed_out"]
                and outcomes["passed"] == group
                and not any(outcomes[k] for k in ("failed", "error", "skipped"))
            )
        return int(not passed)
    except (OSError, ValueError, KeyError, TypeError):
        return 1


def run(root: Path, output: Path, private: Path, *, validate: bool = True) -> int:
    """Validate a private seal and actual learning consumer before exposing final primary bytes."""
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    plan = manifest(private) if validate else []
    atomic_json(
        raw / "execution_manifest.json",
        dict(commands=plan, owned=e.OWNED, full_repository_suite="not_run"),
    )
    if (private / "collection").exists():
        shutil.copytree(private / "collection", raw / "collection")
        atomic_json(
            raw / "collection_archive.json",
            [
                dict(original_path=str(private / "collection" / p.name), **e.reference(p))
                for p in sorted((raw / "collection").iterdir())
            ],
        )
    work = e.measure(root, raw, private / "checkpoints")
    work["collection_archive_reference"] = (
        e.reference(raw / "collection_archive.json")
        if (raw / "collection_archive.json").exists()
        else None
    )
    receipts = []
    historical_elapsed = 0.0
    for i, spec in enumerate(plan):
        e.progress("validation", i, len(plan) - i)
        receipt = child(
            spec["name"],
            spec["argv"],
            raw / "validation",
            deadline=min(spec["deadline"], max(0.001, 1800 - historical_elapsed))
            if spec["name"].startswith("historical_shard_")
            else spec["deadline"],
            heartbeat=30,
        )
        receipts.append(receipt)
        if spec["name"].startswith("historical_shard_"):
            historical_elapsed += receipt["duration_s"]
            atomic_json(private / "historical_receipts" / (spec["name"] + ".receipt.json"), receipt)
    if (private / "coverage.json").exists():
        shards = raw / "coverage_shards"
        shards.mkdir()
        for path in private.glob(".coverage*"):
            shutil.copyfile(path, shards / path.name)
        shutil.copyfile(private / "coverage.json", raw / "coverage.json")
        work["coverage_shards"] = [e.reference(p) for p in sorted(shards.iterdir())]
    atomic_json(raw / "measurement.json", work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, e.build(work, receipts, raw, output))
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI)]
    for name, args, expected in [
        ("cold", ["--cold-replay", str(candidate)], 0),
        ("missing", ["--cold-replay", str(raw / "absent")], 1),
        ("error", ["--deliberate-error"], 7),
    ]:
        receipts.append(child(name, argv + args, raw / "controls", expected=expected, deadline=180))
    changed = deepcopy(json.loads(candidate.read_bytes()))
    changed["learning_qualification_ready_score"] = 999
    changed["reproducibility_checksum"] = e.checksum(changed)
    atomic_json(raw / "rehashed_tamper.json", changed)
    receipts.append(
        child(
            "rehashed_tamper",
            argv + ["--cold-replay", str(raw / "rehashed_tamper.json")],
            raw / "controls",
            expected=1,
            deadline=180,
        )
    )
    atomic_json(candidate, e.build(work, receipts, raw, output))
    checks, findings = validators.scan(candidate, raw / "preterminal_validators")
    receipts.extend(checks)
    work["adversarial_findings"] = findings
    work["ended_monotonic_ns"] = time.monotonic_ns()
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, receipts, raw, output)
    reports: list[Json] = []

    def validator(path: Path) -> Json:
        """A private publication exercises the unchanged consumer against exact candidate bytes."""
        checks, flags = validators.scan(path, raw / "terminal_validators")
        probe = raw / "publication_probe" / "results" / output.name
        publication = publish_primary(
            probe, json.loads(path.read_bytes()), lambda p: dict(passed=e.replay(p))
        )
        side = Path(publication["sidecar_path"])
        qualification = e.consumer(probe, side)
        original_bytes = probe.read_bytes()
        probe.write_bytes(original_bytes + b"\n")
        mutation = e.consumer(probe, side, replay=False)
        probe.write_bytes(original_bytes)
        honest_failure = (
            value["verdict_class"] in ("disqualified", "blocked")
            and value["objective_kernel_ready_score"] == 0
        )
        report = dict(
            passed=e.replay(path)
            and (all(r["passed"] for r in checks) or honest_failure)
            and (
                (
                    qualification["passed"]
                    and qualification["primary_sha256"]
                    == publication["primary_sha256"]
                    == e.reference(path)["sha256"]
                )
                or honest_failure
            )
            and mutation.get("error") == "stale_primary_hash",
            findings=flags,
            receipts=checks,
            qualification_consumer=qualification,
            post_seal_mutation_rejected=mutation.get("error") == "stale_primary_hash",
            mutated_consumer_receipt=mutation,
            private_publication=publication,
        )
        reports.append(report)
        return report

    publication = publish_primary(output, value, validator)
    qualification = e.consumer(output, Path(publication["sidecar_path"]))
    reader = reader_receipt(
        e.TASK,
        output.parent,
        field="learning_qualification_ready_score",
        expected=value["learning_qualification_ready_score"],
    )
    atomic_json(Path(value["qualification_consumer_receipt"]["path"]), qualification)
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            publication=publication,
            consumer=reader,
            qualification_consumer=qualification,
            primary_sha256=publication["primary_sha256"],
            validation_receipts=receipts,
            terminal_reports=reports,
            adversarial_findings=findings,
        ),
    )
    if value["learning_qualification_ready_score"] and not (
        qualification["passed"]
        and reader["passed"]
        and qualification.get("primary_sha256") == publication["primary_sha256"]
        and reader.get("gate_sha256")
        == reader.get("document_sha256")
        == publication["primary_sha256"]
    ):
        atomic_json(raw / "failed_owned_publication.json", value)
        failure = raw / "post_publication_failure.json"
        atomic_json(
            failure, dict(qualification=qualification, reader=reader, publication=publication)
        )
        receipts.append(
            child(
                "post_publication_consumer",
                [
                    str(e.ROOT / ".venv/bin/python"),
                    "-c",
                    "import json,sys; v=json.load(open(sys.argv[1])); assert v['qualification']['passed'] and v['reader']['passed'] and v['qualification'].get('primary_sha256') == v['publication']['primary_sha256'] == v['reader'].get('gate_sha256') == v['reader'].get('document_sha256')",
                    str(failure),
                ],
                raw / "controls",
                deadline=30,
            )
        )
        work["ended_monotonic_ns"] = time.monotonic_ns()
        atomic_json(raw / "measurement.json", work)
        value = e.build(work, receipts, raw, output)
        superseded = publication
        publication = publish_primary(output, value, validator)
        atomic_json(
            Path(value["terminal_validation_sidecar_path"]),
            dict(
                publication=publication,
                superseded_owned_publication=superseded,
                owned_failure_reference=e.reference(failure),
                qualification_consumer=qualification,
                terminal_reports=reports,
                validation_receipts=receipts,
            ),
        )
        raise ValueError("post_publication_consumer_failed")
    e.progress("finished", value["completed_count"], value["censored_count"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Keep dated publication and historical child modes separate to preserve original authority."""
    e.progress("start_no_model_load_current_LLM_calls_0")
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] in ("--historical-tests", "--historical-exec"):
        return historical(arguments[1:], script=arguments[0] == "--historical-exec")
    if any(a in arguments for a in ("--worker", "--state-replay")):
        return int(shipped_runner.main(arguments))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261014"], default="20261014")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--historical-summary", type=Path)
    parser.add_argument("--receipts", type=Path)
    args = parser.parse_args(arguments)
    if args.historical_summary:
        if not args.receipts:
            parser.error("historical receipts required")
        return summarize(args.historical_summary, args.receipts)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT):
        parser.error("private control outputs must be outside the repository")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8516-owned-", dir="/var/tmp") as directory:
            return run(args.root, output, Path(directory), validate=not args.private_control)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
