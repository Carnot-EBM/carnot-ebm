"""REQ-VERIFY-8382: bounded children keep runtime evidence private and replayable."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.evidence_features_custody_7980 import reference
from carnot.reporting.v709_execution import execute
from carnot.reporting import sentence_spline_execution_8334 as publisher
from carnot.verify import cuda_primitive_8290 as primitive
from carnot.verify import runtime_evidence_delta_8382 as q

Json = dict[str, Any]
PY = str(q.ROOT / ".venv/bin/python")
SCRATCH = Path.home() / ".cache/carnot-exp8382-private"


def manifest(private: Path) -> list[Json]:
    """Freeze scoped statement coverage, private consumers and one global diagnostic."""
    config = private / "coverage.ini"
    config.write_text(
        f"[run]\nparallel=True\npatch=subprocess\ndata_file={private / '.coverage'}\n[report]\nexclude_lines=\n"
    )
    tests = [str(q.ROOT / ".venv/bin/pytest"), "-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    specs = [
        (
            "tools",
            [PY, "-c", "import coverage,pytest,ruff,mypy; print('required tools importable')"],
            20,
            "owned",
        ),
        (
            "owned_coverage",
            [
                PY,
                "-m",
                "coverage",
                "run",
                "--rcfile",
                str(config),
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                q.TEST,
                "--basetemp=" + str(private / "owned"),
            ],
            600,
            "owned",
        ),
        (
            "coverage_combine",
            [PY, "-m", "coverage", "combine", "--rcfile", str(config), "--keep"],
            20,
            "owned",
        ),
        (
            "coverage_json",
            [
                PY,
                "-m",
                "coverage",
                "json",
                "--rcfile",
                str(config),
                "--include=" + ",".join(str(q.ROOT / p) for p in q.OWNED),
                "--fail-under=100",
                "-o",
                str(private / "coverage.json"),
            ],
            20,
            "owned",
        ),
        (
            "typed_runtime_consumers",
            tests
            + [
                "tests/python/test_typed_runtime_8368.py",
                "tests/python/test_primary_publication_7928.py",
                "--basetemp=" + str(private / "consumers"),
            ],
            600,
            "owned",
        ),
        (
            "private_E2E018",
            tests
            + [
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
                "tests/python/test_v722_contract_methods_8374.py::test_private_e2e018",
                "--basetemp=" + str(private / "e2e018"),
            ],
            300,
            "owned",
        ),
        ("ruff_check", [PY, "-m", "ruff", "check", *q.OWNED, q.TEST], 30, "owned"),
        ("ruff_format", [PY, "-m", "ruff", "format", "--check", *q.OWNED, q.TEST], 30, "owned"),
        (
            "strict_mypy",
            [PY, "-m", "mypy", "--strict", "--follow-imports=silent", *q.OWNED[:2]],
            90,
            "owned",
        ),
        ("spec_coverage", [PY, "scripts/check_spec_coverage.py", "--files", q.TEST], 30, "owned"),
        ("private_disk_scratch", ["/usr/bin/df", "-T", str(private)], 10, "owned"),
        (
            "repository_health_once",
            [
                str(q.ROOT / ".venv/bin/pytest"),
                "tests/python",
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--basetemp=" + str(private / "global"),
            ],
            900,
            "global",
        ),
    ]
    return [dict(name=n, argv=a, deadline=d, expected=0, scope=s) for n, a, d, s in specs]


def resources(private: Path) -> Json:
    """Inspect the actual mount and memory before admitting disk-backed child work."""
    mounts = [line.split() for line in Path("/proc/self/mountinfo").read_text().splitlines()]
    mounted = max(
        (m for m in mounts if private.resolve().is_relative_to(Path(m[4]))), key=lambda m: len(m[4])
    )
    filesystem = mounted[mounted.index("-") + 1]
    memory = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        )
        * 1024
    )
    return dict(
        scratch_path=str(private),
        private_disk_backed_scratch=filesystem not in {"tmpfs", "ramfs"},
        filesystem=filesystem,
        available_memory_bytes=memory,
        minimum_memory_bytes=536870912,
        time_budget_s=4800,
        heartbeat_s=20,
        no_model_load=True,
        current_llm_calls=0,
    )


def main(argv: list[str] | None = None) -> int:
    """Expose replay and measurement through the same bounded no-model invocation."""
    args = list(sys.argv[1:] if argv is None else argv)
    q.progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_calls_zero")
    if "--cuda-probe" in args:
        return int(primitive.main(args))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261010"], default="20261010")
    parser.add_argument("--root", type=Path, default=q.ROOT)
    parser.add_argument("--output", type=Path, default=q.ROOT / "results" / (q.NAME + ".json"))
    parser.add_argument("--environment-receipt", type=Path)
    parser.add_argument("--receipt-sha256")
    parser.add_argument("--private-run", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parsed = parser.parse_args(args)
    if parsed.cold_replay:
        passed = q.replay(parsed.cold_replay)
        q.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = parsed.output.absolute()
    if parsed.private_run and output.is_relative_to(q.ROOT / "results"):
        parser.error("private run requires private output")
    began, wall = time.monotonic_ns(), time.time_ns()
    raw = output.parent / "raw" / q.NAME / "invocations" / str(wall)
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        SCRATCH.mkdir(parents=True, exist_ok=True, mode=0o700)
        SCRATCH.chmod(0o700)
        with TemporaryDirectory(prefix="invocation-", dir=SCRATCH) as directory:
            private = Path(directory)
            observed = resources(private)
            plan = manifest(private)
            frozen = raw / "execution_manifest.json"
            atomic_json(
                frozen,
                dict(
                    commands=plan,
                    frozen_before_measurement=True,
                    terminal_argv=[
                        [
                            PY,
                            "-u",
                            str(q.ROOT / q.CLI),
                            "--cold-replay",
                            str(output.parent / "raw" / q.NAME / "terminal_candidate.json"),
                        ],
                        [
                            PY,
                            "scripts/adversarial_verify.py",
                            "--json",
                            str(output.parent / "raw" / q.NAME / "terminal_candidate.json"),
                        ],
                        [
                            PY,
                            "scripts/verdict_row_consistency_lint.py",
                            "--strict",
                            str(output.parent / "raw" / q.NAME / "terminal_candidate.json"),
                        ],
                    ],
                    heartbeat_s=20,
                    control_argv=[
                        [PY, "-u", str(q.ROOT / q.CLI), "--cold-replay", str(private / name)]
                        for name in ("valid.json", "absent", "tamper.json")
                    ]
                    + [[PY, "-c", "raise SystemExit(1)"]],
                    task_cap_s=4800,
                ),
            )
            q.progress("validation_before", 0, len(plan))
            with patch.dict(
                os.environ,
                {"TMPDIR": str(private), "PYTHONUNBUFFERED": "1", "JAX_PLATFORMS": "cpu"},
            ):
                receipts = execute(plan[:1] if parsed.private_run else plan, raw / "validation")
            q.progress("validation_after", len(receipts), 0)
            work = q.measure(
                parsed.root,
                raw,
                parsed.environment_receipt,
                parsed.receipt_sha256,
                fixture=parsed.private_run,
            )
            work["resources"] = observed
            work["refs"].append(reference(frozen))
            if (
                not observed["private_disk_backed_scratch"]
                or observed["available_memory_bytes"] < observed["minimum_memory_bytes"]
            ):
                work["failures"].append(
                    q.old.gate(private, "private_disk_memory_budget", True, observed)
                )
            if (private / "coverage.json").is_file():
                coverage = json.loads((private / "coverage.json").read_bytes())
                atomic_json(raw / "owned_coverage.json", coverage)
                work["owned_statement_coverage"] = coverage["totals"]
                work["refs"].append(reference(raw / "owned_coverage.json"))
            ended = time.monotonic_ns()
            work.update(
                duration_s=(ended - began) / 1e9,
                phase_spans=[
                    dict(
                        phase="validation_and_evidence",
                        started_wall_ns=wall,
                        started_monotonic_ns=began,
                        ended_monotonic_ns=ended,
                        duration_s=(ended - began) / 1e9,
                    )
                ],
            )
            atomic_json(raw / "measurement.json", work)
            value = q.build(work, raw, receipts)
            valid = private / "valid.json"
            atomic_json(valid, value)
            bad = deepcopy(value)
            bad["runtime_changed_score"] = 1 - bad["runtime_changed_score"]
            bad["reproducibility_checksum"] = q.checksum(bad)
            tamper = private / "tamper.json"
            atomic_json(tamper, bad)
            controls = [
                dict(
                    name=name,
                    argv=[PY, "-u", str(q.ROOT / q.CLI), "--cold-replay", str(path)],
                    deadline=60,
                    expected=expected,
                    scope="owned",
                )
                for name, path, expected in [
                    ("valid_replay", valid, 0),
                    ("missing_input_replay", private / "absent", 1),
                    ("rehashed_tamper", tamper, 1),
                ]
            ]
            controls.append(
                dict(
                    name="deliberate_error",
                    argv=[PY, "-c", "raise SystemExit(1)"],
                    deadline=15,
                    expected=1,
                    scope="owned",
                )
            )
            receipts.extend(execute(controls, raw / "cold_controls"))
            with patch.object(publisher, "e", q):
                publisher.publish(q.build(work, raw, receipts), output, raw)
        return 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
