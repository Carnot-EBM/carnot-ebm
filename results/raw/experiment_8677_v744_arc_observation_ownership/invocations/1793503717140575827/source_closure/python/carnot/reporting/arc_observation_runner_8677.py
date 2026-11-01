"""REQ-VERIFY-8677: supervise finite CPU children and publish checked final bytes."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import arc_observation_8677 as e
from carnot.reporting import arc_observation_live_8677 as live
from carnot.reporting import v744_contract_runner as validation
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def measure(root: Path, raw: Path, private: Path, date: str) -> Json:
    """Authenticate CPU operands and freeze sources before any numerical head load."""
    started = time.monotonic_ns()
    work = e.empty_work(date, [r for r in e.gates(root) if not r["passed"]])
    tools = {
        n: (e.ROOT / ".venv/bin" / n).is_file()
        for n in ("python", "pytest", "ruff", "mypy", "coverage")
    }
    memory = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    mounts = [line.split() for line in Path("/proc/mounts").read_text().splitlines()]
    filesystem = max(
        (m for m in mounts if str(private).startswith(m[1].rstrip("/") + "/")),
        key=lambda m: len(m[1]),
    )[2]
    probe = private / "writable"
    probe.write_text("disk-backed scratch")
    resources: Json = dict(
        tools=tools,
        available_memory_bytes=memory,
        free_disk_bytes=shutil.disk_usage(private).free,
        filesystem=filesystem,
        writable=probe.read_text() == "disk-backed scratch",
        optional_hardware_required=False,
    )
    work["preconditions_checked"] = resources
    if not (
        all(tools.values())
        and memory >= 2**30
        and resources["free_disk_bytes"] >= 2**30
        and filesystem not in ("tmpfs", "ramfs")
    ):
        work["failures"].append(
            dict(
                upstream="host",
                check="cpu_resources",
                path=str(private),
                hash=None,
                field="resources",
                operator="sufficient",
                expected=True,
                observed=resources,
            )
        )
    head = e.ROOT / "results/experiment_4629_live_frame_change_cnn.pt"
    head_pin = "sha256:ee308cc8f2a1c9182cdf3bdbef8e25f0561b63ba2bca115fa7b51f17f006a78a"
    work["numerical_head_reference"] = dict(
        path=str(head), sha256=head_pin, language_model=False, memory_withheld=True
    )
    observed_head = sha256_file(head) if head.is_file() else None
    if observed_head != head_pin:
        work["failures"].append(
            dict(
                upstream="exp4629",
                check="numerical_checkpoint",
                path=str(head),
                hash=observed_head,
                field="sha256",
                operator="==",
                expected=head_pin,
                observed=observed_head,
            )
        )
    sources = (
        e.OWNED
        + e.TESTS
        + [
            "python/carnot/agentic/arc_competition_agent.py",
            "python/carnot/agentic/arc_outcome_observer.py",
            "python/carnot/agentic/arc_reset_provenance.py",
            "python/carnot/agentic/arc_live_panel_runtime_8384.py",
            "python/carnot/reporting/arc_observer_live_8637.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/v709_execution.py",
            "ops/exclusion_manifest.yaml",
            "ops/e2e-test-plan.md",
            "AGENTS.md",
            "CODEX.md",
            "CLAUDE.md",
            "scripts/experiment_template.py",
            e.UPSTREAM,
            e.PROTOCOL,
            e.HISTORY,
            "openspec/change-proposals/research-roadmap-vNEXT.md",
            "openspec/change-proposals/research-roadmap-v743-preserved-20261101.md",
            "openspec/change-proposals/v743-methods-protocol.json",
            "research-references.md",
            "ops/north-star.md",
            "ops/arc_solve_registry.yaml",
            "openspec/capabilities/research-reporting/spec.md",
            "openspec/capabilities/verification/spec.md",
            "docs/research-notes/v744-arc-observation-ownership.md",
        ]
    )
    for name in sources:
        source = e.ROOT / name
        snapshot = raw / "source_closure" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, snapshot)
        work["source_artifact_hashes"][str(snapshot)] = sha256_file(snapshot)
        if name in e.OWNED + e.TESTS or name.endswith("arc_competition_agent.py"):
            work["code_config_hashes"][str(source)] = sha256_file(source)
    registry = e.ROOT / "ops/arc_solve_registry.yaml"
    roster = yaml.safe_load(registry.read_bytes())["games"]
    work["registry_precheck"] = dict(
        path=str(registry),
        sha256=sha256_file(registry),
        checked_before_execution=True,
        policy_received_registry=False,
        rows=[
            dict(game=g, reproduced=next((r for r in roster if r["game"] == g.split("-")[0]), None))
            for g in e.GAMES
        ],
    )
    e.progress("preconditions_complete")
    if not work["failures"]:
        try:
            p = e.authenticate()
        except (OSError, ValueError, KeyError) as exc:
            work["failures"].append(
                dict(
                    upstream="exp8664",
                    check="historical_authentication",
                    path=str(e.ROOT / e.HISTORY),
                    hash=e.HISTORY_PIN,
                    field="sealed_original_authority",
                    operator="==",
                    expected=True,
                    observed=str(exc),
                )
            )
    if not work["failures"]:
        work["historical_model_provenance"] = p["historical_model_provenance"]
        work["source_artifact_hashes"].update(p["historical_source_hashes"])
        for index in range(16):
            spec: Json = dict(
                kind="diagnosis", index=index, directory=str(raw / "diagnosis" / str(index))
            )
            run_child(spec, raw, work)
            target = Path(spec["directory"]) / "diagnosis.json"
            row = (
                json.loads(target.read_bytes())
                if target.is_file()
                else dict(
                    passed=False,
                    duplicate_observation_count=None,
                    missing_reason="diagnosis_child_failed",
                )
            )
            work["diagnosis_rows"].append(row)
            if not row["passed"]:
                e.progress("diagnosis_gap_original_assertion_preserved")
                break
    if (
        not work["failures"]
        and len(work["diagnosis_rows"]) == 16
        and all(r["passed"] for r in work["diagnosis_rows"])
    ):
        work["controls"] = [
            live.control(case)
            for case in ("horizon", "reset", "conflict", "terminal", "repeated_is_done")
        ]
        for game in e.GAMES:
            for seed in p["seeds"]:
                for mode in ("off", "on"):
                    if (time.monotonic_ns() - started) / 1e9 > 3450:
                        raise TimeoutError("measurement_cap")
                    spec = dict(
                        kind="episode",
                        game=game,
                        seed=seed,
                        observer=mode,
                        arm="on",
                        episode_id=f"{game}:{seed}:{mode}",
                        max_actions=512,
                        max_seconds=120,
                        directory=str(raw / "episodes" / f"{game}-{seed}-{mode}"),
                    )
                    run_child(spec, raw, work)
                    target = Path(spec["directory"]) / "ownership_stream.json"
                    if target.is_file():
                        work["references"].append(e.reference(target))
    ended = time.monotonic_ns()
    work["duration_s"] = (ended - started) / 1e9
    work["phase_spans"] = [
        dict(
            phase="measurement",
            started_monotonic_ns=started,
            ended_monotonic_ns=ended,
            duration_s=work["duration_s"],
        )
    ]
    work["raw_file_hashes"] = {
        str(p): sha256_file(p)
        for p in raw.rglob("*")
        if p.is_file() and "source_closure" not in p.parts
    }
    return work


def run_child(spec: Json, raw: Path, work: Json) -> None:
    """Each child has a finite deadline and its own retained command and output."""
    path = Path(spec["directory"]) / "unit.json"
    atomic_json(path, spec)
    index = len(work["measurement_receipts"])
    e.progress("measurement_child_before", index, 32 - index)
    work["measurement_receipts"].append(
        child(
            f"measure_{index}",
            [sys.executable, "-u", str(e.ROOT / e.CLI), "--episode", str(path)],
            raw / "children",
            deadline=150,
        )
    )
    e.progress("measurement_child_after", index + 1, 31 - index)


def main(argv: list[str] | None = None) -> int:
    """Private CLI controls use the same reducer, cold reader and atomic publisher."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_language_model_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261101")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--episode", type=Path)
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--owned-failure", action="store_true")
    args = parser.parse_args(argv)
    try:
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        parser.error("--date requires YYYYMMDD")
    if args.cold_replay:
        try:
            accepted = e.replay(json.loads(args.cold_replay.read_bytes()))
        except (OSError, ValueError, KeyError, TypeError):
            accepted = False
        e.progress("cold_replay_complete", int(accepted), 0)
        return int(not accepted)
    if args.episode:
        spec = json.loads(args.episode.read_bytes())
        if spec["kind"] == "diagnosis":
            return int(
                not live.diagnose(e.authenticate(), spec["index"], Path(spec["directory"]))[
                    "passed"
                ]
            )
        from carnot.reporting.arc_supervisor_live_panel_8384 import sdk

        arcade = live.trace_sdk(spec["fixture_reference"]) if "fixture_reference" in spec else sdk()
        result = live.collect(spec, arcade, Path(spec["directory"]))
        return int(
            bool(
                result["termination"]["error"]
                or result["observer_issues"]
                or result["model_tripwire_attempts"]
            )
        )
    if args.private_control and args.root.exists():
        return 2
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        raw = args.output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True)
        with TemporaryDirectory(prefix="carnot8677-", dir="/var/tmp") as directory:
            private = Path(directory)
            work = measure(args.root, raw, private, args.date)
            path = raw / "work.json"
            atomic_json(path, work)
            atomic_json(raw / "measurement_seal.json", dict(work_sha256=sha256_file(path)))
            e.progress("measurement_after_validation_before")
            if args.private_control:
                receipts = [
                    child(
                        "private_owned",
                        [
                            sys.executable,
                            "-c",
                            "raise SystemExit(" + str(int(args.owned_failure)) + ")",
                        ],
                        raw / "validation",
                    )
                ]
            else:
                plan = validation_plan(private / "validation")
                atomic_json(raw / "execution_manifest.json", dict(commands=plan))
                with patch.object(validation.previous, "e", e):
                    receipts = validation.execute(plan, raw / "validation")
                receipts.append(validation.previous.previous.exact_union(plan))
                for source in (private / "validation").glob("*.xml"):
                    shutil.copyfile(source, raw / source.name)
                shutil.copyfile(private / "validation/coverage.json", raw / "coverage.json")
                shutil.copytree(private / "validation/collection", raw / "collection")
            atomic_json(
                raw / "validation_seal.json",
                dict(receipts_sha256=canonical_hash(work["measurement_receipts"] + receipts)),
            )
            value = e.build(path, work["measurement_receipts"] + receipts, args.output)
            side = publish(value, args.output, raw)
            e.progress(
                "publication_complete", value["completed_count"], 16 - value["completed_count"]
            )
            return int(not side["passed"])
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)


def reader(task: str, results: Path, **kwargs: Any) -> Json:
    """Keep unchanged consumers while selecting the exact downstream ownership field."""
    from carnot.reporting.primary_publication import reader_receipt

    kwargs["field"] = "observer_delivery_ready_score"
    return dict(reader_receipt(task, results, **kwargs))


def publish(value: Json, output: Path, raw: Path) -> Json:
    """Only bytes accepted by cold replay and unchanged readers become the primary."""
    from carnot.reporting.primary_publication import publish_primary

    def validate(candidate: Path) -> Json:
        commands = [
            ("cold_replay", [e.CLI, "--cold-replay", str(candidate)]),
            ("adversarial", ["scripts/adversarial_verify.py", "--json", str(candidate)]),
            (
                "strict_rows",
                ["scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
            ),
        ]
        checks = [
            child(name, [sys.executable, "-u", *args], raw / "terminal", deadline=120)
            for name, args in commands
        ]
        with TemporaryDirectory(prefix="exp8677-readers-", dir="/var/tmp") as directory:
            candidate_copy = Path(directory) / output.name
            candidate_copy.write_bytes(candidate.read_bytes())
            consumer = reader(
                e.TASK, candidate_copy.parent, expected=value["observer_delivery_ready_score"]
            )
        report = dict(
            passed=all(r["passed"] for r in checks) and consumer["passed"],
            checks=checks,
            consumer=consumer,
            adversarial_report=json.loads(Path(checks[1]["stdout_path"]).read_bytes()),
        )
        atomic_json(raw / "publication_attempt.json", report)
        return report

    publication_receipt = publish_primary(output, value, validate)
    consumer = reader(e.TASK, output.parent, expected=value["observer_delivery_ready_score"])
    side = dict(publication=publication_receipt, consumer=consumer, passed=consumer["passed"])
    atomic_json(Path(value["terminal_validation_sidecar_path"]), side)
    return side


def validation_plan(private: Path) -> list[Json]:
    """Collect original assertion families once, alongside the existing private E2E nodes."""
    with patch.multiple(validation, e=e, report=e):
        plan = list(validation.manifest(private))
    original_nodes = []
    for suite in (
        "tests/python/test_arc_outcome_observer_8624.py",
        "tests/python/test_arc_reset_transfer_8664.py",
    ):
        nodes, receipt = validation.collection.collect(private, suite)
        original_nodes.extend(nodes)
        path = private / "collection" / (receipt["name"] + ".receipt.json")
        plan.insert(
            0,
            dict(
                name=receipt["name"],
                argv=[
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import sys; from carnot.reporting.v743_contract_runner import verify_collection; raise SystemExit(not verify_collection(Path(sys.argv[1]), sys.argv[2]))",
                    str(path),
                    sha256_file(path),
                ],
                deadline=60,
                expected=0,
                scope="owned",
                collected_node_ids=nodes,
            ),
        )
    junit = private / "original_observer_families.xml"
    spec = dict(
        name="original_observer_families",
        argv=[
            str(e.ROOT / ".venv/bin/coverage"),
            "run",
            "--rcfile=" + str(private / "consumer-coverage.ini"),
            "-m",
            "pytest",
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "-q",
            *original_nodes,
            "--junitxml=" + str(junit),
            "--basetemp=" + str(private / "original_families"),
        ],
        deadline=180,
        expected=0,
        scope="owned",
        collected_node_ids=original_nodes,
        junit_path=str(junit),
    )
    plan.insert(next(i for i, row in enumerate(plan) if row["name"] == "coverage_combine"), spec)
    return plan
