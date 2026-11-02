"""REQ-REPORT-8193: reuse checked publication while freezing exact owned tools.

Static tools receive file paths. The historical tests and real covered child
together measure the old failure route without altering its scientific method.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import calibrated_memory_execution_8180 as old_runner
from carnot.reporting import methods_stream_execution_8111 as previous
from carnot.reporting import experiment_7303_validation_scope as supervisor
from carnot.reporting.current_work_receipt import sha256_file
from carnot.verify import learning_qualification_8193 as e

Json = dict[str, Any]
OWNED = [e.MODULE, e.RUNNER, e.CLI]
BASE_PROGRESS = supervisor._progress
BASE_CHECK = previous.run_check


def child_progress(name: str, phase: str, started: float, detail: str = "") -> None:
    """An outstanding real child has zero completed and one pending operation."""
    BASE_PROGRESS(name, phase, started, detail)
    done = int(phase == "after_subprocess")
    e.progress(phase + "_" + name, done, 1 - done)


def check(root: Path, spec: Json, private: Path, durable: Path, **kwargs: Any) -> Json:
    """Reuse one completed full-suite diagnostic without retrying unrelated health."""
    source = os.environ.get("CARNOT_8193_HEALTH_WORK")
    if spec["name"] != "repository_full_suite" or not source:
        return BASE_CHECK(root, spec, private, durable, **kwargs)
    while "global_health" not in (work := json.loads(Path(source).read_text())):
        e.progress("repository_health_pending", 0, 1)
        time.sleep(30)
    receipt: Json = work["global_health"]
    if (
        receipt["argv"] != spec["argv"]
        or sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
    ):
        raise ValueError("health_custody")
    return dict(
        receipt, reused=True, source_receipt=dict(path=source, sha256=sha256_file(Path(source)))
    )


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze complete statement includes and private E2E commands before fitting."""
    with patch.object(previous, "e", e), patch.object(previous, "OWNED", OWNED):
        specs = old_runner.BASE_MANIFEST(private, candidate)
    config = private / "coverage.ini"
    config.write_text(
        config.read_text()
        + "[report]\ninclude =\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in OWNED)
    )
    specs["commands"] = [s for s in specs["commands"] if s["name"] != "consumer_and_E2E015_019"]
    for spec in specs["commands"]:
        spec["deadline_s"] = 900 if spec["name"] == "owned_unit_and_private_CLI" else 180
        if spec["name"] == "owned_unit_and_private_CLI":
            spec["argv"].extend(
                e.legacy.TEST + "::" + name
                for name in [
                    "test_convex_scalar_reference_and_bounds",
                    "test_schedule_admission_and_overflow",
                    "test_no_signal_and_ineligible",
                ]
            )
        if spec["name"].startswith("coverage_"):
            spec["argv"].append("--data-file=" + str(private / ".coverage"))
    py = str(e.ROOT / ".venv/bin/python")
    cli = str(e.ROOT / "scripts/experiments/experiment_7868_v683_intervention_protocol.py")
    output = str(private / "E2E016.json")
    for name, args in [
        ("success", ["--date", "20260929", "--fixture-e2e", output]),
        ("replay", ["--date", "20260929", "--cold-replay", output]),
    ]:
        specs["commands"].append(
            dict(
                name="E2E016_" + name,
                argv=["/usr/bin/env", "-u", "PYTHONPATH", py, "-u", cli, *args],
                expected_exit=0,
                deadline_s=180,
                classification="required",
            )
        )
    for spec in specs["terminal_commands"]:
        spec["deadline_s"] = 300
        if spec["name"] == "cold_replay":
            spec["argv"][:0] = ["/usr/bin/env", "-u", "PYTHONPATH"]
    specs["repository_health"]["deadline_s"] = 1200
    return specs


def main(argv: list[str] | None = None) -> int:
    """Use the actual calibrated seed child and existing normal-exit publisher."""
    args = sys.argv[1:] if argv is None else argv
    if "--seed-input" in args:
        with patch.object(e.legacy, "seed_child", e.seed_child):
            return old_runner.main(args)
    with (
        patch.object(previous, "e", e),
        patch.object(previous, "OWNED", OWNED),
        patch.object(previous, "manifest", manifest),
        patch.object(previous, "run_check", check),
        patch.object(supervisor, "_progress", child_progress),
    ):
        return previous.main(args)
