"""REQ-VERIFY-8403: bounded qualification reuses unchanged receipt consumers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import released_question_identity_8403 as e
from carnot.reporting import historical_consumer_roots_8402 as authority
from carnot.reporting import human_label_custody_runner_8389 as shipped
from carnot.reporting import external_evidence_runner_8375 as base
from carnot.reporting.current_work_receipt import atomic_json, ZERO_INVOCATION_COUNTS
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import execute

Json = dict[str, Any]
SCRATCH = Path.home() / ".cache/carnot-exp8403-private"
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
    authority.DESIGN,
    authority.PROTOCOL,
    "research-references.md",
    "docs/research-notes/v724-label-custody.md",
]


def plan(private: Path) -> list[Json]:
    """Shipped commands record genuine child coverage before any measurement starts."""
    with patch.object(shipped, "e", e):
        specs = shipped.plan(private)
    next(s for s in specs if s["name"] == "coverage_combine")["argv"].append("--keep")
    specs.insert(
        2,
        dict(
            name="private_E2E019",
            argv=[
                str(e.ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_experiment_7942_v689_sentence_labels.py",
                "--basetemp=" + str(private / "e2e019"),
            ],
            expected=0,
            deadline=240,
            scope="owned",
        ),
    )
    return list(specs)


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Record resources and full task authority before opening pinned upstream fields."""
    began = time.monotonic()
    e.progress("preconditions_before")
    mounts = [line.split() for line in Path("/proc/self/mountinfo").read_text().splitlines()]
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
        filesystem=filesystem,
        disk_backed=filesystem not in ("tmpfs", "ramfs"),
        free_disk_bytes=shutil.disk_usage(private).free,
        available_memory_bytes=memory,
        minimum_memory_bytes=536870912,
        task_cap_s=4800,
        child_heartbeat_s=30,
        no_model_load=True,
        tools={
            n: os.access(e.ROOT / ".venv/bin" / n, os.X_OK)
            for n in ("python", "pytest", "coverage", "ruff", "mypy")
        },
    )
    gates = []
    if (
        not pre["disk_backed"]
        or memory < pre["minimum_memory_bytes"]
        or pre["free_disk_bytes"] < 52428800
        or not all(pre["tools"].values())
    ):
        gates.append(e.old.gate(str(private), "resource_preconditions", True, pre))
    checked = authority.authority(root, raw / "authority")
    tasks = [t for t in checked["tasks"] if t.get("id") == e.TASK]
    if (
        not checked["activated"]
        or len(tasks) != 1
        or tasks[0].get("MODEL_SPECS") != []
        or tasks[0].get("inference_substrate_class") != "no_model_load"
    ):
        gates.append(
            e.old.gate(
                str(root / authority.DESIGN),
                "complete_current_task_authority",
                e.TASK,
                dict(activated=checked["activated"], matching_tasks=len(tasks)),
                e.reference(root / authority.DESIGN)["sha256"]
                if (root / authority.DESIGN).is_file()
                else None,
            )
        )
    refs = []
    for ordinal, name in enumerate(INPUTS):
        path = root / name
        if path.is_file():
            copy = raw / "inputs" / f"{ordinal}.bin"
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(path.read_bytes())
            refs.append(dict(e.reference(copy), source_path=str(path)))
        else:
            gates.append(e.old.gate(str(path), "input_available", True, None))
    manifest = e.input_manifest(root if not gates else raw / "preconditions-stopped")
    for key, ref in manifest["primaries"].items():
        if Path(ref["path"]).is_file():
            data = e.read_reference(ref)
            copy = raw / "inputs" / (key + ".json")
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(data)
            manifest["primaries"][key] = e.reference(copy)
            refs.append(dict(e.reference(copy), source_path=ref["path"]))
    e.progress("preconditions_after", len(refs), len(gates))
    path = raw / "source_manifest.json"
    atomic_json(path, manifest)
    e.progress("measurement_before")
    reduction = e.reduce_manifest(manifest)
    primitive = raw / "primitive_rows.json"
    atomic_json(primitive, reduction)
    e.progress("measurement_after", reduction["completed_count"], reduction["censored_count"])
    return dict(
        manifest_reference=e.reference(path),
        primitive_reference=e.reference(primitive),
        source_artifact_hashes=refs,
        code_config_hashes=[e.reference(e.ROOT / p) for p in [*e.OWNED, e.TEST]],
        authority=checked,
        gate_check_summary=gates,
        preconditions_checked=pre,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        duration_s=time.monotonic() - began,
        phase_spans=[
            dict(
                phase="preconditions_and_identity_measurement", duration_s=time.monotonic() - began
            )
        ],
        invocation_id=raw.name,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        repository_health=dict(
            scope="global",
            current_run=False,
            observation="unknown; scoped qualification only; prior V722 global timeout is not a current pass",
        ),
    )


def controls(candidate: Path, raw: Path) -> list[Json]:
    """The shipped adapter checks valid, missing, deliberate-error and rehashed primitive children."""
    with patch.object(base, "e", e):
        return list(base.controls(candidate, raw))


def validate(candidate: Path, raw: Path) -> Json:
    """Unchanged cold replay, adversarial and strict-row consumers check exact candidate bytes."""
    with patch.object(base, "e", e):
        return dict(base.validate(candidate, raw))


def main(argv: list[str] | None = None) -> int:
    """One capped invocation freezes checks, measures, cold replays and atomically publishes."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261011"], default="20261011")
    parser.add_argument("--input-root", type=Path, default=e.ROOT)
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
            work = measure(args.input_root, raw, private)
            work["work_path"] = str(raw / "work.json")
            work["code_config_hashes"].append(e.reference(raw / "validation_plan.json"))
            atomic_json(Path(work["work_path"]), work)
            receipts = execute(frozen, raw / "validation")
            for shard in private.glob(".coverage*"):
                if shard.is_file():
                    target = raw / "coverage_shards" / shard.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(shard, target)
            coverage = private / "coverage.json"
            if coverage.is_file():
                shutil.copyfile(coverage, raw / "owned_coverage.json")
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
                field="release_criterion_ready_score",
                expected=value["release_criterion_ready_score"],
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
