"""REQ-REPORT-8211: freeze validation and reuse the checked normal-exit publisher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import methods_stream_execution_8111 as previous
from carnot.verify import calibrated_memory_trajectory_8211 as e

Json = dict[str, Any]
OWNED = [e.MODULE, e.RUNNER, e.CLI]


def restart_specs(raw: Path, labels: Path, features: Json) -> list[Json]:
    """Use the qualified Coverage.py exit patch before the causal child imports."""
    directory = raw / "child_coverage"
    directory.mkdir(parents=True, exist_ok=True)
    config = directory / "coverage.ini"
    config.write_text(
        "[run]\npatch = _exit\nparallel = true\ndata_file = "
        + str(directory / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in OWNED)
        + "[report]\nexclude_lines =\n"
    )
    cli = [
        "/usr/bin/env",
        "-u",
        "PYTHONPATH",
        "-u",
        "COVERAGE_PROCESS_START",
        "-u",
        "COVERAGE_RCFILE",
        str(e.ROOT / ".venv/bin/python"),
        "-u",
        "-m",
        "coverage",
        "run",
        "--rcfile=" + str(config),
        str(e.ROOT / e.CLI),
        "--seed-input",
        features["path"],
        "--label-path",
        str(labels),
    ]
    return [
        dict(
            name=name,
            argv=cli + ["--seed-output", str(raw / folder)] + args,
            expected_exit=expected,
            deadline_s=120,
            classification="required",
        )
        for name, folder, args, expected in [
            ("uninterrupted", "uninterrupted", [], 0),
            ("hard_exit", "restart", ["--crash-slot", "90"], 73),
            ("resume", "restart", ["--resume-state", str(raw / "restart/crash.json")], 0),
        ]
    ]


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze this experiment's owned files and one separate full-suite health run."""
    with patch.object(previous, "e", e), patch.object(previous, "OWNED", OWNED):
        specs = previous.manifest(private, candidate)
    config = private / "coverage.ini"
    config.write_text(
        config.read_text().replace("[run]\n", "[run]\npatch = _exit\n")
        + "[report]\nexclude_lines =\n"
    )
    for spec in specs["commands"]:
        if spec["name"] == "owned_unit_and_private_CLI":
            spec["argv"].insert(1, "CARNOT_8211_COVERAGE_PARENT=" + str(private))
            spec["deadline_s"] = 600
        if spec["name"].startswith("coverage_"):
            spec["argv"].append("--data-file=" + str(private / ".coverage"))
        if spec["name"] == "spec_coverage":
            spec["argv"].insert(-1, "--files")
    specs["repository_health"]["deadline_s"] = 1200
    return specs


def main(argv: list[str] | None = None) -> int:
    """Child mode shares the natural causal runner; publication uses qualified code."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--seed-input" in arguments:
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed-input", type=Path, required=True)
        parser.add_argument("--label-path", type=Path, required=True)
        parser.add_argument("--seed-output", type=Path, required=True)
        parser.add_argument("--crash-slot", type=int, default=0)
        parser.add_argument("--resume-state", type=Path)
        args = parser.parse_args(arguments)
        rows = json.loads(args.seed_input.read_text())["rows"]
        state = json.loads(args.resume_state.read_text()) if args.resume_state else None
        e.run_seed(
            rows, args.label_path, 101, args.seed_output, state=state, crash_slot=args.crash_slot
        )
        return 0
    with (
        patch.object(previous, "e", e),
        patch.object(previous, "OWNED", OWNED),
        patch.object(previous, "manifest", manifest_adapter),
        patch.object(previous, "run_check", e.run_check),
    ):
        return previous.main(arguments)


BASE_MANIFEST = previous.manifest


def manifest_adapter(private: Path, candidate: Path) -> Json:
    """Restore the original builder temporarily to avoid recursive scoped adapters."""
    with patch.object(previous, "manifest", BASE_MANIFEST):
        return manifest(private, candidate)
