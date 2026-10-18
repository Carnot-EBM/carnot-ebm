"""REQ-REPORT-8565: bind full authority before admitting independent CPU methods.

This registration consumes sealed evidence; it does not run a model or measure
learning. A stale visible design remains an explicit external failed operand.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import (
    atomic_json as atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.roadmap_contract import compare_contract, parse_design

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK, MILESTONE = (
    "experiment_8565_v736_contract_methods",
    "exp8565-contract-methods",
    "2026.10.736",
)
DESIGN, ACTIVE, STAGED = (
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    "research-roadmap.yaml",
    "research-roadmap-next.yaml",
)
PROTOCOL = "openspec/change-proposals/v736-methods-protocol.json"
PROTOCOL_PIN = "sha256:88bbf528a7206e67bbee730ef1fb71d7d7ccb27c7ef3b442430c4ff51026f434"
TASKS_PIN = "a2502b102c28cb7c71a8f74bf0b202e2f205f49fbcdd389d178058a3da8145aa"
INHERITED = [
    "objective_updates",
    "objective_audit",
    "evidence_canary",
    "arc_capability_mask",
    "nine_input_native_inference",
    "closed_source_comparison",
    "partition",
    "features",
    "decision",
]
CLI = f"scripts/experiments/{NAME}.py"
TESTS = [
    "tests/python/test_v736_contract_methods_8565.py",
    "tests/python/test_v736_contract_runner_8565.py",
]
TEST = TESTS[0]
OWNED = [
    f"python/carnot/reporting/v736_{s}.py"
    for s in ["contract_methods", "historical_inputs", "contract_report", "contract_runner"]
] + [CLI]
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so a supervising conductor can distinguish work from a stalled child."""
    print(f"[exp8565] phase={phase} completed={completed} pending={pending}", flush=True)


def reference(path: Path) -> Json:
    """Bind bytes rather than trusting a filename that another producer could replace."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def read(ref: Json) -> Json:
    """Reject missing or changed evidence before decoding its scientific operands."""
    path = Path(ref["path"])
    if sha256_file(path) != ref["sha256"]:
        raise ValueError("evidence_hash")
    return dict(json.loads(path.read_bytes()))


def tasks_digest(tasks: list[Json]) -> str:
    """The full dictionaries include routes and prompts omitted from the short table."""
    return hashlib.sha256(
        json.dumps(tasks, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def checksum(value: Json) -> str:
    """Exclude only the checksum itself so rehashed summary tampering still needs replay."""
    return str(canonical_hash({k: v for k, v in value.items() if k != "reproducibility_checksum"}))


def gate(path: Path, field: str, expected: Any, observed: Any, upstream: str = TASK) -> Json:
    """Missing is represented by null; it cannot masquerade as an observed zero."""
    return dict(
        check=field,
        upstream=upstream,
        upstream_id=upstream,
        path=str(path),
        sha256=sha256_file(path) if path.is_file() else None,
        artifact_field=field,
        field=field,
        op="==",
        operator="==",
        expected=expected,
        observed=observed,
        passed=False,
    )


def protocol() -> Json:
    """An independent pin prevents rehashed envelopes from changing registered rules."""
    p = read(dict(path=str(ROOT / PROTOCOL), sha256=PROTOCOL_PIN))
    old = read(p["inherited_v735_protocol_reference"])
    if any(p[k] != old[k] for k in INHERITED) or tasks_digest(p["task_map_v736"]) != TASKS_PIN:
        raise ValueError("scientific_or_task_operand")
    ref = p["preserved_v735_design_reference"]
    if (
        ref["sha256"] != "sha256:f198170ac62aeb1074482f9b35dd71995f9c8f6ada4235b831958124a7dbfd15"
        or sha256_file(Path(ref["path"])) != ref["sha256"]
    ):
        raise ValueError("original_design")
    return p


def authority(root: Path) -> Json:
    """Compare independently printed and machine authorities without inventing staging."""
    p = protocol()
    failures: list[Json] = []
    paths = [root / DESIGN, root / ACTIVE]
    refs = [reference(path) for path in paths if path.is_file()]
    try:
        text = paths[0].read_text()
        active = yaml.safe_load(paths[1].read_bytes())
        staged = (
            yaml.safe_load((root / STAGED).read_bytes()) if (root / STAGED).is_file() else active
        )
        result = compare_contract(
            text, staged, active, milestone=MILESTONE, first_id=8565, count=14
        )
        expected = p["task_map_v736"]
        machine = parse_design(text, milestone=MILESTONE)[1]
        for i, row in enumerate(result["rows"]):
            row["checks"]["full_dictionary"] = (
                i < len(active["tasks"])
                and active["tasks"][i] == expected[i] == staged["tasks"][i] == machine[i]
            )
            row["matched"] = all(row["checks"].values())
        ready = bool(
            result["passed"]
            and all(r["matched"] for r in result["rows"])
            and re.search(
                r"(?:canonical_tasks_sha256:|Canonical task SHA256:)\s*`?" + TASKS_PIN, text
            )
            and "<!-- V736_TASK_CONTRACT_END -->" in text
            and tasks_digest(active["tasks"]) == TASKS_PIN
        )
        rows = result["rows"]
        if not ready:
            failures.append(gate(paths[0], "fourteen_complete_objects", True, False))
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        ready = False
        failures.append(gate(paths[0], "v736_design_and_activation", MILESTONE, str(error)))
        rows = [
            dict(
                unit_id=t["id"],
                order=i + 1,
                matched=False,
                status="unstarted",
                checks={},
                censored=True,
                excluded=False,
            )
            for i, t in enumerate(p["task_map_v736"])
        ]
    for row in rows:
        row.update(
            absolute_metric=int(row["matched"]),
            effective_independent_groups=0,
            missing_reason=None if row["matched"] else "authority_absent_or_mismatched",
        )
    return dict(
        ready=ready,
        rows=rows,
        gate_check_summary=failures,
        references=refs,
        staging_present=(root / STAGED).is_file(),
    )


def measure(root: Path, raw: Path) -> Json:
    """Freeze real resource and input observations before any validation child starts."""
    from carnot.reporting import v736_historical_inputs as history

    start = time.monotonic_ns()
    progress("preconditions_before", 0, 14)
    raw.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="exp8565-", dir="/var/tmp") as directory:
        scratch = Path(directory) / "probe"
        scratch.write_bytes(b"disk-backed-exp8565")
        resources: Json = dict(
            scratch_write_read_passed=scratch.read_bytes() == b"disk-backed-exp8565",
            scratch_mount=max(
                (
                    line.split()
                    for line in Path("/proc/self/mountinfo").read_text().splitlines()
                    if Path(directory).is_relative_to(line.split()[4])
                ),
                key=lambda cells: len(cells[4]),
            ),
            scratch_device=scratch.stat().st_dev,
            free_disk_bytes=shutil.disk_usage(directory).free,
            memory_available_kb=next(
                int(s.split()[1])
                for s in Path("/proc/meminfo").read_text().splitlines()
                if s.startswith("MemAvailable:")
            ),
            tools={
                n: shutil.which(str(ROOT / ".venv/bin" / n))
                for n in ["python", "pytest", "coverage", "ruff", "mypy"]
            },
            measurement_cap_s=3600,
            validation_reserve_s=900,
            task_cap_s=4800,
        )
    precondition_failures = [
        gate(Path(str(ROOT / ".venv/bin" / n)), "executable_available", True, None)
        for n, available in resources["tools"].items()
        if available is None
    ]
    resources["tool_hashes"] = {
        name: reference(Path(path)) for name, path in resources["tools"].items() if path is not None
    }
    for field, observed, expected in [
        ("scratch_write_read_passed", resources["scratch_write_read_passed"], True),
        (
            "disk_backed_scratch",
            resources["scratch_mount"][resources["scratch_mount"].index("-") + 1]
            not in {"tmpfs", "ramfs"},
            True,
        ),
        ("disk_capacity_at_least_128MiB", resources["free_disk_bytes"] >= 128 * 1024 * 1024, True),
        ("memory_at_least_128MiB", resources["memory_available_kb"] >= 128 * 1024, True),
    ]:
        if observed != expected:
            precondition_failures.append(gate(Path("/var/tmp"), field, expected, observed))
    p = protocol()
    progress("authority_before", 0, 14)
    contract = authority(root)
    progress(
        "authority_after",
        sum(r["matched"] for r in contract["rows"]),
        14 - sum(r["matched"] for r in contract["rows"]),
    )
    historical = history.historical(p)
    runtime = history.runtime(root, raw / "runtime")
    sources = (
        [
            p[k]
            for k in [
                "inherited_v735_protocol_reference",
                "preserved_v735_design_reference",
                "literature_reference",
                "methods_reference",
                "conductor_reference",
            ]
        ]
        + list(p["original_producers"].values())
        + contract["references"]
    )
    work = dict(
        root=str(root.absolute()),
        protocol_reference=reference(ROOT / PROTOCOL),
        protocol=p,
        contract=contract,
        history=historical,
        runtime=runtime,
        preconditions=resources,
        precondition_failures=precondition_failures,
        source_refs=sources,
        code_refs=[reference(ROOT / n) for n in [*OWNED, *TESTS]],
        start_ns=start,
        end_ns=time.monotonic_ns(),
        wall_ns=time.time_ns(),
    )
    work["duration_s"] = (work["end_ns"] - start) / 1e9
    work["phase_spans"] = [
        dict(
            phase="sealed_cpu_aggregation",
            start_monotonic_ns=start,
            end_monotonic_ns=work["end_ns"],
            duration_s=work["duration_s"],
        )
    ]
    atomic_json(raw / "measurement.json", work)
    progress("measurement_frozen", 14, 0)
    return work
