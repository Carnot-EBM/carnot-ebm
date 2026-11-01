"""REQ-VERIFY-8375: bounded execution preserves missing evidence and checked bytes."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import external_evidence_8375 as e
from carnot.reporting import v722_contract_methods as authority
from carnot.reporting.current_work_receipt import atomic_json, ZERO_INVOCATION_COUNTS
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v718_replay_runner import audit

Json = dict[str, Any]
SCRATCH = Path.home() / ".cache/carnot-exp8375-private"
OWNED = [
    "python/carnot/reporting/external_evidence_8375.py",
    "python/carnot/reporting/external_evidence_runner_8375.py",
    e.CLI,
]
INPUTS = [
    "AGENTS.md",
    "CLAUDE.md",
    "CODEX.md",
    "ops/e2e-test-plan.md",
    "ops/exclusion_manifest.yaml",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/verification/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/primary_publication.py",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    "openspec/change-proposals/research-roadmap-v721-preserved-20261010.md",
    "research-references.md",
    "ops/verifier_gaps.md",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/pipeline/extract.py",
    "python/carnot/verify/sentence_spline_fit_8334.py",
    "results/experiment_8361_v721_utility_audit_qualification.json",
]


def measure(root: Path, raw: Path, manifest_path: Path, private: Path) -> Json:
    """Authenticate task and inputs before reducing any example availability."""
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
    tools = {
        name: (root / ".venv/bin" / name).is_file() for name in ("python", "pytest", "ruff", "mypy")
    }
    pre = dict(
        scratch_path=str(private),
        scratch_mode=oct(private.stat().st_mode & 0o777),
        filesystem=filesystem,
        disk_backed=filesystem not in {"tmpfs", "ramfs"},
        free_disk_bytes=shutil.disk_usage(private).free,
        available_memory_bytes=memory,
        minimum_memory_bytes=536870912,
        tools=tools,
        task_cap_s=4800,
        retrieval_cap_s=900,
        retrieval_cap_bytes=209715200,
        no_model_load=True,
    )
    failures = []
    if not pre["disk_backed"] or memory < pre["minimum_memory_bytes"] or not all(tools.values()):
        failures.append(e.gate(private, "resource_preconditions", True, pre))
    e.progress("authority_before")
    checked = authority.authority(root, raw / "authority")
    for failure in checked["gate_check_summary"]:
        failures.append(
            dict(
                failure,
                upstream=failure.get("upstream_id", e.TASK),
                path=failure.get("artifact_path"),
                sha256=failure.get("artifact_hash"),
                field=failure.get("artifact_field"),
                operator=failure.get("op", "=="),
                expected_value=failure.get("expected"),
                observed_value=failure.get("observed"),
            )
        )
    tasks = [t for t in checked["tasks"] if t.get("id") == e.TASK]
    if len(tasks) != 1 or tasks[0].get("MODEL_SPECS") != []:
        failures.append(e.gate(root / "research-roadmap.yaml", "exact_task", e.TASK, tasks))
    e.progress("authority_after", len(tasks), 1 - len(tasks))
    refs = []
    snapshots = raw / "inputs"
    snapshots.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(INPUTS):
        source = root / name
        if source.is_file():
            target = snapshots / (str(index) + ".bin")
            target.write_bytes(source.read_bytes())
            refs.append(dict(e.reference(target), source_path=str(source)))
        else:
            failures.append(e.gate(source, "input_available", True, None))
    e.progress("preconditions_after", len(refs), len(INPUTS) - len(refs))
    if not manifest_path.is_file():
        failures.append(e.gate(manifest_path, "external_release_available", True, None))
        manifest_path = raw / "absent_release_observation.json"
        atomic_json(manifest_path, dict(units=[], release_manifest=[], unavailable_release=True))
    manifest_ref = e.reference(manifest_path)
    manifest = json.loads(e.read_reference(manifest_ref))
    e.progress("structural_inspection_before")
    reduction = e.reduce_manifest(e.hydrate(manifest))
    primitive = raw / "primitive_rows.json"
    atomic_json(primitive, reduction)
    e.progress(
        "structural_inspection_after", reduction["completed_count"], reduction["censored_count"]
    )
    return dict(
        manifest_reference=manifest_ref,
        primitive_reference=e.reference(primitive),
        source_artifact_hashes=refs + manifest.get("operands", []),
        code_config_hashes=[e.reference(e.ROOT / name) for name in OWNED],
        authority=checked,
        gate_check_summary=failures,
        preconditions_checked=pre,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        duration_s=time.monotonic() - began,
        phase_spans=[
            dict(
                phase="preconditions_and_structural_inspection", duration_s=time.monotonic() - began
            )
        ],
        invocation_id=str(time.time_ns()),
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
    )


def validate(candidate: Path, raw: Path) -> Json:
    """Unchanged validators decide whether these exact candidate bytes can publish."""
    cold = child(
        "cold",
        [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(candidate)],
        raw,
        deadline=60,
    )
    found = audit(candidate, raw / "adversarial", {})
    rows = child(
        "strict_rows",
        [
            sys.executable,
            "-u",
            "scripts/verdict_row_consistency_lint.py",
            "--strict",
            str(candidate),
        ],
        raw,
        deadline=60,
    )
    return dict(
        passed=all(r["passed"] for r in (cold, found["receipt"], rows)),
        checks=[cold, found["receipt"], rows],
        adversarial=found,
    )


def controls(candidate: Path, raw: Path) -> list[Json]:
    """Real child failures prove that changed aggregates cannot hide behind new hashes."""
    original = json.loads(candidate.read_bytes())
    rehashed = deepcopy(original)
    primitive = json.loads(e.read_reference(original["primitive_reference"]))
    primitive["completed_count"] += 1
    changed = raw / "changed_rows.json"
    atomic_json(changed, primitive)
    work = json.loads(e.read_reference(original["work_reference"]))
    work["primitive_reference"] = e.reference(changed)
    work_path = raw / "changed_work.json"
    work["work_path"] = str(work_path)
    atomic_json(work_path, work)
    rehashed = e.build(work, original["validation_receipts"])
    rehashed["work_reference"] = e.reference(work_path)
    tamper = raw / "rehashed_tamper.json"
    atomic_json(tamper, rehashed)
    prefix = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    specs = [
        ("valid", ["--cold-replay", str(candidate)], 0),
        ("missing_input", ["--cold-replay", str(raw / "absent.json")], 1),
        ("deliberate_error", ["--deliberate-error"], 1),
        ("rehashed_tamper", ["--cold-replay", str(tamper)], 1),
    ]
    return [
        child(name, prefix + args, raw / "controls", expected=exit_code, deadline=60)
        for name, args, exit_code in specs
    ]


def plan(private: Path) -> Json:
    """Freeze exact commands and subprocess coverage before structural measurement."""
    path = e.ROOT / "results/raw" / e.NAME / "frozen_validation_commands.json"
    value: Json = json.loads(path.read_bytes())
    config = private / "coverage.ini"
    # File filters avoid importing JAX through package discovery while its own
    # extension is still loading. Subprocess coverage still records the real CLI.
    config.write_text(
        "[run]\nparallel = true\npatch = subprocess\ninclude =\n"
        + "\n".join("    " + str(e.ROOT / name) for name in OWNED)
        + "\n"
    )
    os.environ["COVERAGE_RCFILE"] = str(config)
    os.environ["COVERAGE_FILE"] = str(private / ".coverage")
    for command in value["commands"]:
        command["argv"][0] = str(e.ROOT / command["argv"][0])
        if "pytest" in command["argv"][0]:
            command["argv"].append("--basetemp=" + str(private / command["name"]))
    return value


def main(argv: list[str] | None = None) -> int:
    """Private validation precedes atomic publication; the real CLI owns its cap."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_LLM_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261010"], default="20261010")
    parser.add_argument(
        "--manifest", type=Path, default=e.ROOT / "results/raw" / e.NAME / "release_manifest.json"
    )
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--inspect-manifest", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        e.progress("deliberate_error_rejected")
        return 1
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_failed")
        return int(not passed)
    if args.inspect_manifest:
        print(
            json.dumps(e.reduce_manifest(e.hydrate(json.loads(args.manifest.read_bytes())))),
            flush=True,
        )
        return 0
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
            atomic_json(raw / "validation_plan.json", frozen)
            work = measure(e.ROOT, raw, args.manifest, private)
            work["work_path"] = str(raw / "work.json")
            work["code_config_hashes"].append(e.reference(raw / "validation_plan.json"))
            atomic_json(Path(work["work_path"]), work)
            receipts = []
            for index, spec in enumerate(frozen["commands"]):
                e.progress("validation", index, len(frozen["commands"]) - index)
                receipts.append(
                    child(
                        spec["name"],
                        spec["argv"],
                        raw / "validation",
                        deadline=spec["deadline_s"],
                        scope=spec["scope"],
                    )
                )
            candidate = private / (e.NAME + ".json")
            atomic_json(candidate, e.build(work, receipts))
            receipts.extend(controls(candidate, raw))
            initial = validate(candidate, raw / "initial_terminal")
            work["adversarial_findings"] = initial["adversarial"]["findings"]
            work["duration_s"] = time.monotonic() - began
            work["phase_spans"].extend(
                dict(phase=r["name"], duration_s=r["duration_s"])
                for r in receipts + initial["checks"]
            )
            atomic_json(Path(work["work_path"]), work)
            receipts.extend(initial["checks"])
            value = e.build(work, receipts)
            publication = publish_primary(
                args.output, value, lambda p: validate(p, raw / "terminal")
            )
            atomic_json(Path(work["terminal_validation_sidecar_path"]), publication)
            e.progress("published_" + value["verdict_class"], 1, 0)
            return 0 if value["required_checks_passed"] else 1
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        e.progress("owned_failure_" + type(error).__name__)
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
