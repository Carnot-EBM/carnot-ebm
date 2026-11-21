"""REQ-VERIFY-8389: bounded acquisition and unchanged validators check custody."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import human_label_custody_8389 as e
from carnot.reporting import v723_contract_methods as authority
from carnot.reporting import external_evidence_runner_8375 as base
from carnot.reporting import v717_contract_runner as commands
from carnot.reporting.current_work_receipt import atomic_json, ZERO_INVOCATION_COUNTS
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
SCRATCH = Path.home() / ".cache/carnot-exp8389-private"
INPUTS = [
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "ops/e2e-test-plan.md",
    "ops/exclusion_manifest.yaml",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/verification/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/primary_publication.py",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    "research-references.md",
    "python/carnot/reporting/external_evidence_8375.py",
    "python/carnot/reporting/current_work_receipt.py",
    "results/experiment_8375_v722_external_evidence_readiness.json",
    "research-roadmap.yaml",
    "openspec/change-proposals/v723-service-and-label-protocol.json",
]


def plan(private: Path) -> list[Json]:
    """Reuse shipped scoped checks and invocation-only subprocess coverage."""
    with patch.object(commands, "m", e):
        specs = commands.manifest(private)
    os.environ["COVERAGE_RCFILE"] = str(private / "coverage.ini")
    os.environ["COVERAGE_FILE"] = str(private / ".coverage")
    specs[0]["deadline"] = 360
    return list(specs)


def measure(root: Path, raw: Path, supplied: Path | None, private: Path) -> Json:
    """Authenticate actual task and resource operands before acquiring target bytes."""
    started = time.monotonic()
    e.progress("preconditions_before")
    mounts = [s.split() for s in Path("/proc/self/mountinfo").read_text().splitlines()]
    mount = max(
        (m for m in mounts if private.resolve().is_relative_to(Path(m[4]))), key=lambda m: len(m[4])
    )
    filesystem = mount[mount.index("-") + 1]
    memory = (
        int(
            next(
                s.split()[1]
                for s in Path("/proc/meminfo").read_text().splitlines()
                if s.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    pre: Json = dict(
        scratch_path=str(private),
        scratch_mode=oct(private.stat().st_mode & 0o777),
        disk_backed=filesystem not in ("tmpfs", "ramfs"),
        filesystem=filesystem,
        free_disk_bytes=shutil.disk_usage(private).free,
        available_memory_bytes=memory,
        minimum_memory_bytes=536870912,
        task_cap_s=4800,
        acquisition_cap_s=600,
        acquisition_cap_bytes=52428800,
        per_call_deadline_s=30,
        no_model_load=True,
        tools={
            n: (root / ".venv/bin" / n).is_file()
            for n in ("python", "pytest", "coverage", "ruff", "mypy")
        },
    )
    failures = []
    resources = (
        pre["disk_backed"]
        and memory >= pre["minimum_memory_bytes"]
        and pre["free_disk_bytes"] >= 52428800
        and all(pre["tools"].values())
    )
    if not resources:
        failures.append(e.gate(str(private), "resource_preconditions", True, pre))
    checked = authority.authority(root, raw / "authority")
    tasks = [t for t in checked["tasks"] if t.get("id") == e.TASK]
    if not checked["activated"] or len(tasks) != 1 or tasks[0].get("MODEL_SPECS") != []:
        failures.append(
            e.gate(
                str(root / authority.DESIGN),
                "current_task_authority",
                e.TASK,
                dict(activated=checked["activated"], matching_tasks=len(tasks)),
                e.reference(root / authority.DESIGN)["sha256"]
                if (root / authority.DESIGN).is_file()
                else None,
            )
        )
    refs = []
    for index, name in enumerate(INPUTS):
        path = root / name
        if path.is_file():
            copy = raw / "inputs" / f"{index}.bin"
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(path.read_bytes())
            refs.append(dict(e.reference(copy), source_path=str(path)))
        else:
            failures.append(e.gate(str(path), "input_available", True, None))
    e.progress("preconditions_after", len(refs), len(INPUTS) - len(refs))
    atomic_json(raw / "frozen_target_mapping.json", e.TARGET)
    if supplied is None and resources:
        manifest = e.acquire(raw)
    elif supplied is not None and supplied.is_file():
        manifest = json.loads(supplied.read_bytes())
    else:
        manifest = dict(
            release_commit=e.COMMIT,
            release_files={},
            target_mapping=e.TARGET,
            overlap_inventory=[],
            license_permitted=False,
            acquisition_gates=[],
        )
        failures.append(
            e.gate(str(supplied or raw / "release"), "external_release_available", True, None)
        )
    if "overlap_inventory" not in manifest:
        manifest["overlap_inventory"] = e.overlap_inventory(root, raw)
    path = raw / "source_manifest.json"
    reduction = e.validate_manifest(manifest)
    manifest["validated_rows"] = reduction["rows"]
    atomic_json(path, manifest)
    # The stable task-level manifest contains hashes and private paths, not answer text.
    if raw.is_relative_to(root / "results/raw" / e.NAME):
        atomic_json(root / "results/raw" / e.NAME / "source_manifest.json", manifest)
    primitive = raw / "primitive_rows.json"
    atomic_json(primitive, reduction)
    return dict(
        manifest_reference=e.reference(path),
        primitive_reference=e.reference(primitive),
        source_artifact_hashes=refs + list(manifest["release_files"].values()),
        code_config_hashes=[e.reference(root / n) for n in e.OWNED]
        + [e.reference(raw / "frozen_target_mapping.json")],
        authority=checked,
        gate_check_summary=failures,
        preconditions_checked=pre,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        duration_s=time.monotonic() - started,
        phase_spans=[
            dict(phase="preconditions_and_acquisition", duration_s=time.monotonic() - started)
        ],
        invocation_id=raw.name,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        repository_health=dict(
            scope="global",
            current_run=False,
            observation="V722 capstone records prior global-suite timeout; scoped task does not rerun the repository suite",
        ),
    )


def controls(candidate: Path, raw: Path) -> list[Json]:
    """The shipped control adapter executes positive and three real failure children."""
    with patch.object(base, "e", e):
        return list(base.controls(candidate, raw))


def validate(candidate: Path, raw: Path) -> Json:
    """Use the existing cold, typed adversarial and strict row consumers unchanged."""
    with patch.object(base, "e", e):
        return dict(base.validate(candidate, raw))


def main(argv: list[str] | None = None) -> int:
    """Check terminal bytes before publication under one bounded task deadline."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261010"], default="20261010")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        e.progress("deliberate_error_rejected")
        return 1
    if args.cold_replay:
        return int(not e.replay(args.cold_replay))
    began = time.monotonic()
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        SCRATCH.mkdir(parents=True, exist_ok=True, mode=0o700)
        SCRATCH.chmod(0o700)
        with TemporaryDirectory(dir=SCRATCH, prefix="invocation-") as directory:
            private = Path(directory)
            os.environ["TMPDIR"] = str(private)
            raw = args.output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
            raw.mkdir(parents=True, mode=0o700)
            frozen = plan(private)
            atomic_json(raw / "validation_plan.json", dict(commands=frozen))
            work = measure(e.ROOT, raw, args.manifest, private)
            work["work_path"] = str(raw / "work.json")
            work["code_config_hashes"].append(e.reference(raw / "validation_plan.json"))
            atomic_json(Path(work["work_path"]), work)
            receipts = []
            for index, spec in enumerate(frozen):
                e.progress("validation", index, len(frozen) - index)
                receipts.append(
                    child(
                        spec["name"],
                        spec["argv"],
                        raw / "validation",
                        deadline=spec["deadline"],
                        expected=spec["expected"],
                        scope=spec["scope"],
                    )
                )
            coverage = private / "coverage.json"
            if coverage.is_file():
                (raw / "owned_coverage.json").write_bytes(coverage.read_bytes())
            candidate = private / (e.NAME + ".json")
            atomic_json(candidate, e.build(work, receipts))
            receipts.extend(controls(candidate, raw))
            initial = validate(candidate, raw / "initial_terminal")
            work["adversarial_findings"] = initial["adversarial"]["findings"]
            receipts.extend(initial["checks"])
            work["duration_s"] = time.monotonic() - began
            work["phase_spans"].extend(
                dict(phase=r["name"], duration_s=r["duration_s"]) for r in receipts
            )
            atomic_json(Path(work["work_path"]), work)
            value = e.build(work, receipts)
            publication = publish_primary(
                args.output, value, lambda p: validate(p, raw / "terminal")
            )
            atomic_json(Path(work["terminal_validation_sidecar_path"]), publication)
            consumer = reader_receipt(
                e.TASK,
                args.output.parent,
                field="released_label_panel_ready_score",
                expected=value["released_label_panel_ready_score"],
            )
            atomic_json(raw / "live_consumer_receipt.json", consumer)
            if not consumer["passed"]:
                raise ValueError("live_consumer_identity")
            e.progress("published_" + value["verdict_class"], 1, 0)
            return 0 if value["required_checks_passed"] else 1
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        e.progress("owned_failure_" + str(error))
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
