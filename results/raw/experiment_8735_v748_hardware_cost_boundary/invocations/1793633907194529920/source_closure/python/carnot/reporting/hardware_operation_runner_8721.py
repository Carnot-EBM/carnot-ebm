"""REQ-VERIFY-8721: reuse bounded validation and immutable publication unchanged."""

from __future__ import annotations

import argparse
from datetime import datetime, UTC
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from threading import Event, Thread
import time
from typing import Any, cast
from unittest.mock import patch

from carnot.reporting import hardware_operation_boundary_8721 as e
from carnot.reporting import hardware_operation_inputs_8721 as i
from carnot.reporting import hardware_operation_runner_8707 as retained
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.gatemate_continuity_8400 import budget

Json = dict[str, Any]
BASE_VALIDATE, BASE_REPLAY, BASE_TERMINAL, BASE_RUN = (
    retained.validate,
    retained.replay,
    retained.terminal,
    retained.run,
)


def heartbeat(stop: Event) -> None:
    """The parent stays visible while validation processes authenticate large operands."""
    while not stop.wait(30):
        e.progress("parent_heartbeat")


def measure(raw: Path, date: str, root: Path, private: bool) -> Json:
    """Freeze exact operands and source snapshots before any cost reduction or tests."""
    start, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    raw.mkdir(parents=True, exist_ok=True)
    e.progress("preconditions_before")
    with TemporaryDirectory(prefix="hardware8721-", dir="/var/tmp") as directory:
        scratch = Path(directory)
        pre = resources(scratch)
        pre["write_probe_passed"] = (
            scratch / "write_probe"
        ).read_bytes() == b"private disk scratch"
    pre.update(optional_devices_required=False, language_model_calls=0)
    failures = []
    if not (
        pre["disk_backed"]
        and pre["tools_available"]
        and pre["write_probe_passed"]
        and min(pre["free_disk_bytes"], pre["memory_available_bytes"]) >= 2**30
    ):
        failures.append(
            dict(
                upstream="host",
                path="/var/tmp",
                sha256=None,
                check="cpu_resources",
                field="disk_memory_tools",
                operator="==",
                expected=True,
                observed=pre,
                operand_state="resource_unavailable",
                passed=False,
            )
        )
    e.progress("authenticate_independent_inputs_before", 0, 3)
    evidence = i.authenticate(root)
    e.progress("authenticate_independent_inputs_after", 3, 0)
    closure = set(
        e.OWNED
        + e.TESTS
        + [
            i.DESIGN,
            i.contract.ACTIVE,
            "AGENTS.md",
            "CODEX.md",
            "CLAUDE.md",
            "pyproject.toml",
            "tests/python/conftest.py",
            "openspec/capabilities/hardware-operation-boundary/spec.md",
            "openspec/capabilities/research-reporting/spec.md",
            "openspec/capabilities/verification/spec.md",
            "openspec/change-proposals/research-roadmap-v746-preserved-20261102.md",
            "ops/exclusion_manifest.yaml",
            "ops/e2e-test-plan.md",
            "ops/north-star.md",
            "research-references.md",
            "research-hardware-wishlist.md",
            "scripts/experiment_template.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "scripts/check_spec_coverage.py",
            "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            "tests/python/test_restricted_decision_audit_8210.py",
            "tests/python/test_primary_publication_7928.py",
            "docs/research-notes/v747-hardware-operation-boundary.md",
        ]
    )
    closure.update(
        str(Path(cast(str, m.__file__)).resolve().relative_to(e.ROOT))
        for m in list(sys.modules.values())
        if getattr(m, "__file__", None)
        and Path(cast(str, m.__file__)).suffix == ".py"
        and Path(cast(str, m.__file__)).is_file()
        and Path(cast(str, m.__file__)).resolve().is_relative_to(e.ROOT)
        and ".venv" not in Path(cast(str, m.__file__)).parts
    )
    source_refs = evidence["refs"].copy()
    for name in sorted(closure):
        source, target = e.ROOT / name, raw / "source_closure" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        source_refs += [i.reference(source), i.reference(target)]
    code_refs = [i.reference(e.ROOT / p) for p in e.OWNED + e.TESTS]
    atomic_json(
        raw / "premeasurement_seal.json", dict(source_refs=source_refs, code_refs=code_refs)
    )
    end = time.monotonic_ns()
    if (end - start) / 1e9 > 3600:
        raise TimeoutError("measurement_deadline")
    work = dict(
        raw=str(raw.absolute()),
        run_date=date,
        root=str(root.absolute()),
        private_control=private,
        evidence=evidence,
        source_refs=source_refs,
        code_refs=code_refs,
        resource_failures=failures,
        preconditions=pre,
        started_at_utc=wall,
        ended_at_utc=datetime.now(UTC).isoformat(),
        start_ns=start,
        end_ns=end,
        duration_s=(end - start) / 1e9,
        phase_spans=[
            dict(
                phase="authenticate_and_freeze",
                start_ns=start,
                end_ns=end,
                duration_s=(end - start) / 1e9,
            )
        ],
    )
    atomic_json(raw / "measurement.json", work)
    atomic_json(
        raw.parents[1] / "seals" / (raw.name + ".json"),
        dict(measurement=i.reference(raw / "measurement.json"), code_refs=code_refs),
    )
    e.progress("measurement_sealed", 3, 0)
    return work


def validate(raw: Path) -> list[Json]:
    """Original disjoint children retain SQLite and JUnit before combining owned copies."""
    with patch.multiple(retained, e=e, i=i):
        return list(BASE_VALIDATE(raw))


def replay(path: Path) -> None:
    """Independent seals bind primitive identity even when a candidate is rehashed."""
    with patch.multiple(retained, e=e, i=i):
        BASE_REPLAY(path)


def terminal(path: Path, raw: Path) -> Json:
    """Unchanged scanners and a fresh bounded replay inspect the final candidate bytes."""
    with patch.multiple(retained, e=e, i=i):
        return dict(BASE_TERMINAL(path, raw))


def main(argv: list[str] | None = None) -> int:
    """Private CLI controls preserve checkout guards and cannot grant production readiness."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_language_model_calls_zero")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261102")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    stop = Event()
    thread = Thread(target=heartbeat, args=(stop,), daemon=True)
    thread.start()
    try:
        if args.cold_replay:
            replay(args.cold_replay)
            return 0
        if len(args.date) != 8:
            raise ValueError("calendar_date")
        datetime.strptime(args.date, "%Y%m%d")
        with (
            budget(4800),
            patch.multiple(
                retained, e=e, i=i, measure=measure, validate=validate, terminal=terminal
            ),
        ):
            return int(
                BASE_RUN(
                    args.output.absolute(), args.date, args.root.absolute(), args.private_control
                )
            )
    except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
        print(f"[exp8721] error={error}", file=sys.stderr, flush=True)
        return 1
    finally:
        stop.set()
        thread.join(timeout=1)
