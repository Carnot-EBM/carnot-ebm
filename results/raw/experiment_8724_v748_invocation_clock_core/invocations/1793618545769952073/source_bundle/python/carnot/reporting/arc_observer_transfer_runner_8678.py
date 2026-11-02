"""REQ-VERIFY-8678: freeze fresh transfer work and reuse bounded terminal readers."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any
from unittest.mock import patch

import yaml

from carnot.reporting import arc_observation_runner_8677 as previous
from carnot.reporting import arc_observation_8677 as qualification
from carnot.reporting import arc_observer_transfer_8678 as e
from carnot.reporting import arc_observer_transfer_live_8678 as live
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def measure(root: Path, raw: Path, private: Path, date: str) -> Json:
    """Authenticate branch operands and CPU resources before fresh environment work."""
    started = time.monotonic_ns()
    work: Json = dict(e.empty_work(date, [r for r in e.gates(root) if not r["passed"]]))
    work["upstream_gate_checks"] = e.gates(root)
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
        work["failures"].append(failure("cpu_resources", private, "resources", True, resources))
    head = e.ROOT / "results/experiment_4629_live_frame_change_cnn.pt"
    pin = "sha256:ee308cc8f2a1c9182cdf3bdbef8e25f0561b63ba2bca115fa7b51f17f006a78a"
    work["numerical_head_reference"] = dict(
        path=str(head), sha256=pin, language_model=False, memory_withheld=True
    )
    observed = sha256_file(head) if head.is_file() else None
    if observed != pin:
        work["failures"].append(failure("numerical_checkpoint", head, "sha256", pin, observed))
    if not work["failures"]:
        try:
            p = e.authenticate()
            work.update(
                diagnosis_rows=p["diagnosis_rows"],
                controls=p["controls"],
                historical_model_provenance=p["historical_model_provenance"],
            )
        except (OSError, ValueError, KeyError) as exc:
            work["failures"].append(
                failure(
                    "upstream_authentication",
                    root / next(iter(e.PINS)),
                    "qualified_source_closure",
                    True,
                    str(exc),
                )
            )
    registry = e.ROOT / "ops/arc_solve_registry.yaml"
    roster = yaml.safe_load(registry.read_bytes())["games"]
    work["registry_precheck"] = dict(
        path=str(registry),
        sha256=sha256_file(registry),
        checked_before_execution=True,
        policy_received_registry=False,
        targeted_registered_solve=False,
        headline_solve_credit=0,
        rows=[
            dict(
                game=g, public_solve=next((r for r in roster if r["game"] == g.split("-")[0]), None)
            )
            for g in e.GAMES
        ],
    )
    dependencies = [
        "python/carnot/agentic/arc_competition_agent.py",
        "python/carnot/agentic/arc_observation_delivery.py",
        "python/carnot/agentic/arc_outcome_observer.py",
        "python/carnot/agentic/arc_reset_provenance.py",
        "python/carnot/agentic/arc_live_panel_runtime_8384.py",
        "python/carnot/agentic/arc_trajectory_supervisor.py",
        "python/carnot/reporting/arc_observer_live_8637.py",
        "python/carnot/reporting/arc_observation_8677.py",
        "python/carnot/reporting/arc_observation_live_8677.py",
        "python/carnot/reporting/arc_observation_runner_8677.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/v709_execution.py",
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "ops/exclusion_manifest.yaml",
        "ops/e2e-test-plan.md",
        "ops/arc_solve_registry.yaml",
        "scripts/experiment_template.py",
        "research-references.md",
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/verification/spec.md",
        "openspec/change-proposals/research-roadmap-vNEXT.md",
        "openspec/change-proposals/research-roadmap-v743-preserved-20261101.md",
        "openspec/change-proposals/v743-methods-protocol.json",
        qualification.PROTOCOL,
        qualification.HISTORY,
        "docs/research-notes/v744-arc-observer-transfer.md",
        *e.PINS,
    ]
    for name in e.OWNED + e.TESTS + dependencies:
        source, target = e.ROOT / name, raw / "source_closure" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        work["source_artifact_hashes"][str(target)] = sha256_file(target)
        if name.startswith(("python/", "scripts/", "tests/")):
            work["code_config_hashes"][str(source)] = sha256_file(source)
    if observed == pin:
        work["source_artifact_hashes"][str(head)] = pin
    e.progress("preconditions_complete", 0, 16)
    if not work["failures"]:
        for index, unit in enumerate(e.units()):
            if (time.monotonic_ns() - started) / 1e9 + 150 > 2400:
                e.progress("cumulative_measurement_budget_exhausted", index, 16 - index)
                break
            spec = dict(unit, kind="episode", directory=str(raw / "episodes" / f"{index:02}"))
            run_child(spec, raw, work)
            target = Path(spec["directory"]) / "transfer_stream.json"
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


def failure(check: str, path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Keep external failure operands explicit so blocked never means an observed zero."""
    return dict(
        upstream="host" if check == "cpu_resources" else "exp8677-arc-observation-ownership",
        check=check,
        path=str(path),
        hash=sha256_file(path) if path.is_file() else None,
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
    )


def run_child(spec: Json, raw: Path, work: Json) -> None:
    """Retain each child command and exit; a killed child remains failed evidence."""
    path = Path(spec["directory"]) / "unit.json"
    atomic_json(path, spec)
    index = len(work["measurement_receipts"])
    e.progress("episode_before", index, 16 - index)
    work["measurement_receipts"].append(
        child(
            f"measure_{index}",
            [sys.executable, "-u", str(e.ROOT / e.CLI), "--episode", str(path)],
            raw / "children",
            deadline=150,
        )
    )
    e.progress("episode_after", index + 1, 15 - index)


def reader(task: str, results: Path, **kwargs: Any) -> Json:
    """Select this producer's exact downstream field through the unchanged consumers."""
    from carnot.reporting.primary_publication import reader_receipt

    kwargs["field"] = "arc_rows_ready_score"
    return dict(reader_receipt(task, results, **kwargs))


def main(argv: list[str] | None = None) -> int:
    """Reuse finite validation and atomic publication without historical controllers."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--episode" in arguments:
        return live.episode(Path(arguments[arguments.index("--episode") + 1]))
    with patch.multiple(previous, e=e, measure=measure, reader=reader):
        return int(previous.main(arguments))
