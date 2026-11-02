"""REQ-VERIFY-8599: bounded checks qualify bytes before atomic publication."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import arc_outcome_identifiability_8599 as e
from carnot.reporting import arc_redirect_runner_8509 as validation
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def validate_checks(raw: Path, private: Path) -> list[Json]:
    """Reuse exact node collection and disjoint E2E children without historical controllers."""
    with patch.multiple(validation, e=e, TESTS=e.TESTS):
        return list(validation.validate_checks(raw, private))


def controls(value: Json, raw: Path) -> list[Json]:
    """Fresh children retain real rejection exits, including rehashed aggregate tampering."""
    changed = deepcopy(value)
    changed["completed_count"] = 999
    changed.pop("reproducibility_checksum")
    changed["reproducibility_checksum"] = canonical_hash(changed)
    tamper = raw / "controls" / "rehashed_candidate.json"
    atomic_json(tamper, changed)
    return [
        child(
            name,
            [sys.executable, "-u", str(e.ROOT / e.CLI), *arguments],
            raw / "controls",
            deadline=180,
            expected=expected,
        )
        for name, arguments, expected in [
            ("missing_input", ["--cold-replay", str(raw / "missing.json")], 1),
            ("owned_error", ["--deliberate-error"], 7),
            ("rehashed_tamper", ["--cold-replay", str(tamper)], 1),
        ]
    ]


def publish(value: Json, output: Path, raw: Path) -> Json:
    """All consumers see only the exact bytes checked by the original validators."""

    def validate(candidate: Path) -> Json:
        checks = []
        for name, argv in [
            (
                "terminal_replay",
                [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(candidate)],
            ),
            (
                "adversarial",
                [sys.executable, "scripts/adversarial_verify.py", "--json", str(candidate)],
            ),
            (
                "strict_rows",
                [
                    sys.executable,
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ],
            ),
        ]:
            checks.append(child(name, argv, raw / "terminal", deadline=180))
        report = dict(passed=all(r["passed"] for r in checks), checks=checks)
        atomic_json(raw / "publication_attempt.json", report)
        return report

    publication = publish_primary(output, value, validate)
    bound = read_bound_sidecar(output, Path(publication["sidecar_path"]))
    consumer = reader_receipt(
        e.TASK,
        output.parent,
        field="arc_outcome_audit_ready_score",
        expected=value["arc_outcome_audit_ready_score"],
    )
    side = dict(
        publication=publication,
        report=bound["report"],
        consumer=consumer,
        passed=bound["report"]["passed"] and consumer["passed"],
    )
    atomic_json(Path(value["terminal_validation_sidecar_path"]), side)
    return side


def main(argv: list[str] | None = None) -> int:
    """One invocation owns scratch, measurement clocks and its validation deadline."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--audit-only", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--owned-failure", action="store_true")
    args = parser.parse_args(argv)
    try:
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        parser.error("--date requires YYYYMMDD")
    if args.deliberate_error:
        return 7
    if args.audit_only:
        atomic_json(args.audit_only, e.audit(args.root, args.audit_only.parent / "primitives"))
        return 0
    if args.cold_replay:
        try:
            accepted = e.replay(json.loads(args.cold_replay.read_bytes()))
        except (ValueError, OSError, KeyError, TypeError):
            accepted = False
        e.progress("cold_replay_complete")
        return 0 if accepted else 1
    if args.private_control and args.root.exists():
        parser.error("private controls require an absent input root")
    start = time.monotonic_ns()
    raw = args.output.parent / "raw" / args.output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    with TemporaryDirectory(prefix="carnot8599-", dir="/var/tmp") as temp:
        private = Path(temp)
        with patch.multiple(validation, e=e, TESTS=e.TESTS):
            failures, resources, hashes = validation.preconditions(args.root, private)
        active = args.root / "research-roadmap.yaml"
        if active.is_file():
            shutil.copyfile(active, raw / "authority.yaml")
            hashes.pop(str(active), None)
            hashes[str(raw / "authority.yaml")] = sha256_file(raw / "authority.yaml")
        data: Json = {}
        e.progress("authenticate_and_measure_before")
        try:
            data = e.audit(args.root, raw / "primitives")
        except (OSError, ValueError, TimeoutError) as exc:
            failed_gate = validation.gate(
                "qualified_upstream",
                args.root / e.PANEL_PATH,
                "authenticated_original_panel",
                e.PANEL_PIN,
                str(exc),
            )
            failed_gate["upstream"] = "exp8588-arc-capability-panel"
            failures.append(failed_gate)
        duration = (time.monotonic_ns() - start) / 1e9
        e.progress("authenticate_and_measure_after", len(data.get("rows", [])), 0)
        work = dict(
            data=data,
            failures=failures,
            run_date=args.date,
            input_root=str(args.root),
            replay_scratch=str(raw / "cold_replay_primitives"),
            preconditions_checked=resources,
            duration_s=duration,
            phase_spans=[
                dict(
                    phase="authenticate_and_reduce",
                    started_monotonic_ns=start,
                    ended_monotonic_ns=time.monotonic_ns(),
                    duration_s=duration,
                )
            ],
            code_config_hashes={
                str(e.ROOT / p): sha256_file(e.ROOT / p) for p in [*e.OWNED, *e.TESTS]
            },
            source_artifact_hashes=hashes,
            raw_shard_hashes={str(p): sha256_file(p) for p in (raw / "primitives").glob("*.json")},
        )
        for name in [e.PANEL_PATH, e.PUBLICATION_PATH]:
            path = args.root / name
            if path.is_file():
                work["source_artifact_hashes"][str(path)] = sha256_file(path)
        work_path = raw / "work.json"
        atomic_json(work_path, work)
        if args.private_control:
            receipts = [
                child(
                    "private_owned_check",
                    [
                        sys.executable,
                        "-c",
                        "raise SystemExit(" + str(1 if args.owned_failure else 0) + ")",
                    ],
                    raw / "validation",
                    deadline=30,
                )
            ]
        else:
            receipts = validate_checks(raw, private)
            receipts.extend(controls(e.build(work_path, receipts, args.output), raw))
        value = e.build(work_path, receipts, args.output)
        e.progress("publication_before")
        side = publish(value, args.output, raw)
        e.progress("publication_after", 1, 0)
        return 0 if side["passed"] else 1
