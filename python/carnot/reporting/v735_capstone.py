"""REQ-VERIFY-8564: reuse bounded validation and unchanged atomic publication."""

import argparse
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import v733_capstone as previous
from carnot.reporting import v735_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.gatemate_continuity_8400 import budget
from carnot.reporting.v735_capstone_report import build

Json = dict[str, Any]
SUITES = previous.SUITES
BASE = previous.shipped.old


def manifest(private: Path) -> list[Json]:
    """Retain exact original consumer unions and instrument only newly owned statements."""
    with patch.object(previous, "e", e):
        plan = list(previous.manifest(private))
    for spec in plan:
        if spec["name"] == "owned_tests" or "shard_node_ids" in spec:
            spec["argv"] = ["-vv" if a == "-q" else a for a in spec["argv"]]
    return plan


def controls(value: Json, raw: Path) -> list[Json]:
    """An independently rebuilt thirteenth row must reject a repaired fourteen-row claim."""
    rows = []
    for name, changed, expected in [
        ("valid", deepcopy(value), 0),
        ("rehashed", dict(value, task_count=14), 1),
    ]:
        changed["reproducibility_checksum"] = e.checksum(changed)
        path = raw / (name + ".json")
        atomic_json(path, changed)
        rows.append(
            BASE.child(
                "cold_" + name,
                [
                    str(e.ROOT / ".venv/bin/python"),
                    "-u",
                    str(e.ROOT / e.CLI),
                    "--cold-replay",
                    str(path),
                ],
                raw / "logs",
                expected=expected,
                deadline=180,
            )
        )
    for name, args, expected in [
        ("missing", ["--cold-replay", str(raw / "absent.json")], 1),
        ("deliberate_error", ["--deliberate-error"], 7),
    ]:
        rows.append(
            BASE.child(
                "cold_" + name,
                [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *args],
                raw / "logs",
                expected=expected,
                deadline=60,
            )
        )
    return rows


def run(
    root: Path, output: Path, private: Path, *, control: bool = False, failed: bool = False
) -> int:
    """The shipped pipeline retains failed checks and publishes only validated final bytes."""
    with patch.multiple(
        BASE,
        e=e,
        build=build,
        manifest=previous.shipped._MANIFEST if control else manifest,
        controls=controls,
        child=previous.shipped.complete_child,
        terminal_child=previous.shipped.complete_child,
    ):
        return int(previous.shipped._RUN(root, output, private, control=control, failed=failed))


def main(argv: list[str] | None = None) -> int:
    """Use disk scratch and an invocation deadline; private controls never replace history."""
    e.progress("start_no_model_load_current_LLM_invocations_0", 0, 13)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261016"], default="20261016")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--failed-child", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--worker-request", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args(argv)
    if args.worker_request:
        if not args.worker_output:
            parser.error("worker output required")
        return e.worker_process(args.worker_request, args.worker_output)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT):
        parser.error("private controls must publish outside the repository")
    with budget(4800), TemporaryDirectory(prefix="exp8564-owned-", dir="/var/tmp") as directory:
        return run(
            args.root,
            output,
            Path(directory),
            control=args.private_control,
            failed=args.failed_child,
        )
