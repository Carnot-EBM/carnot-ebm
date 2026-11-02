"""REQ-REPORT-8180: reuse immutable validation logs and atomic primary publication."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import methods_stream_execution_8111 as previous
from carnot.reporting.current_work_receipt import sha256_file
from carnot.verify import calibrated_memory_methods_8180 as e

Json = dict[str, Any]
OWNED = [e.MODULE, e.CLI, e.RUNNER]
BASE_MANIFEST = previous.manifest
BASE_CHECK = previous.run_check


def manifest(private: Path, candidate: Path) -> Json:
    """Static tools get file paths; coverage measures only this experiment's code."""
    with patch.object(previous, "e", e), patch.object(previous, "OWNED", OWNED):
        specs = BASE_MANIFEST(private, candidate)
    specs["commands"] = [s for s in specs["commands"] if s["name"] != "consumer_and_E2E015_019"]
    for s in specs["commands"]:
        s["deadline_s"] = 900 if s["name"] == "owned_unit_and_private_CLI" else 180
        if s["name"].startswith("coverage_"):
            s["argv"].append("--data-file=" + str(private / ".coverage"))
    for s in specs["terminal_commands"]:
        s["deadline_s"] = 300
        if s["name"] == "cold_replay":
            s["argv"][:0] = ["/usr/bin/env", "-u", "PYTHONPATH"]
    specs["repository_health"]["deadline_s"] = 1200
    return specs


def check(root: Path, spec: Json, private: Path, durable: Path, **kwargs: Any) -> Json:
    """Reuse only a completed diagnostic whose exact argv and sealed log match.

    A recovery can finish owned validation without starting the large repository
    suite twice. Waiting reports the real outstanding diagnostic, and preserves
    its original failure or timeout rather than relabeling that run as success.
    """
    source = os.environ.get("CARNOT_8180_HEALTH_WORK")
    if spec["name"] != "repository_full_suite" or not source:
        return BASE_CHECK(root, spec, private, durable, **kwargs)
    path = Path(source)
    while "global_health" not in (work := json.loads(path.read_text())):
        e.progress("prior_repository_health_pending", 0, 1)
        time.sleep(30)
    receipt: Json = work["global_health"]
    if (
        receipt["argv"] != spec["argv"]
        or sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
    ):
        raise ValueError("health_custody")
    e.progress("prior_repository_health_reused", 1, 0)
    return dict(receipt, reused=True, source_receipt=dict(path=str(path), sha256=sha256_file(path)))


def main(argv: list[str] | None = None) -> int:
    """Private restart children use the same calibrated engine as qualification."""
    args = sys.argv[1:] if argv is None else argv
    if "--seed-input" in args:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--seed-input", type=Path, required=True)
        parser.add_argument("--seed-output", type=Path, required=True)
        parser.add_argument("--resume-state", type=Path)
        parser.add_argument("--crash-slot", type=int, default=0)
        parsed = parser.parse_args(args)
        e.seed_child(parsed.seed_input, parsed.seed_output, parsed.resume_state, parsed.crash_slot)
        return 0
    with (
        patch.object(previous, "e", e),
        patch.object(previous, "OWNED", OWNED),
        patch.object(previous, "manifest", manifest),
        patch.object(previous, "run_check", check),
    ):
        return previous.main(args)
