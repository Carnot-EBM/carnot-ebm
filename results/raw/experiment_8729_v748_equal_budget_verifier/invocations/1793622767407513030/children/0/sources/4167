"""REQ-VERIFY-8664: run fresh SDK children through a corrected panel adapter."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import arc_reset_runner_8650 as prior
from carnot.reporting import arc_reset_transfer_8664 as e
from carnot.reporting.arc_reset_live_8649 import collect as base_collect
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

Json = dict[str, Any]
BASE_EPISODE, BASE_MEASURE = prior.episode, prior.measure
BASE_VALIDATE = prior.inherited.validate_checks


def collect(unit: Json, sdk: Any, raw: Path) -> Json:
    """Retain bounded error details and every callback without changing policy actions."""
    from carnot.agentic import arc_reset_provenance as reset

    notify, recorder = reset.notify, reset.ResetProvenance
    policies: list[Any] = []

    def notified(policy: Any, method: str, *args: Any) -> None:
        notify(policy, method, *args)
        if not any(p is policy for p in policies):
            policies.append(policy)

    with (
        patch.object(reset, "notify", notified),
        patch.object(reset, "ResetProvenance", lambda: recorder(limit=unit["max_actions"] * 4)),
    ):
        result: Json = base_collect(unit, sdk, raw)
    result["recorder_error_metadata"] = [
        r for p in policies for r in getattr(p, "_arc_reset_provenance_diagnostics", [])
    ]
    result["adapter_binding"] = dict(
        producer="exp8663-arc-reset-provenance",
        qualified_primary_sha256=list(e.PINS.values())[1],
        recording_capacity=unit["max_actions"] * 4,
    )
    return result


def episode(path: Path) -> int:
    """Keep original public frames and observer replay while using the new adapter."""
    from carnot.reporting import arc_reset_live_8649 as live

    with patch.object(live, "collect", collect), patch.object(prior, "e", e):
        code = BASE_EPISODE(path)
    raw = Path(json.loads(path.read_bytes())["directory"])
    target = raw / "transfer_stream.json"
    value = json.loads(target.read_bytes())
    value["sdk_identity"]["official"] = value["sdk_identity"]["class_name"].startswith("arc_agi.")
    atomic_json(target, value)
    return code


def measure(root: Path, raw: Path, private: Path, date: str) -> Json:
    """Reuse the original budgets and replace outdated upstream failure identities."""
    with patch.object(prior, "e", e):
        work = BASE_MEASURE(root, raw, private, date)
    gates = e.gate_summary(root)
    work["upstream_gate_checks"] = gates
    if any(not r["passed"] for r in gates):
        work["failures"] = [
            r
            for r in work["failures"]
            if r["check"] not in {"required_upstream", "upstream_authentication"}
        ] + [r for r in gates if not r["passed"]]
    work["adapter_binding"] = dict(
        qualified_producer="exp8663-arc-reset-provenance",
        current_adapter=str(e.ROOT / e.OWNED[1]),
        sha256=sha256_file(e.ROOT / e.OWNED[1]),
        original_observer_sha256=e.OBSERVER_PIN,
    )
    for name in [
        "python/carnot/reporting/arc_reset_transfer_8650.py",
        "python/carnot/reporting/arc_reset_runner_8650.py",
        "python/carnot/reporting/arc_observer_transfer_8637.py",
        "python/carnot/reporting/arc_observer_runner_8637.py",
        "python/carnot/reporting/arc_observer_8624.py",
        "python/carnot/reporting/v709_execution.py",
        "python/carnot/reporting/primary_publication.py",
    ]:
        snapshot = raw / "current_source_closure" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(e.ROOT / name, snapshot)
        work["source_artifact_hashes"][str(snapshot)] = sha256_file(snapshot)
    return work


def validate_checks(raw: Path, private: Path) -> list[Json]:
    """Keep consumer coverage separate and retain raw SQLite before combining copies."""
    child = prior.inherited.child

    def retained(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
        if name.startswith("historical_"):
            role = private / name
            role.mkdir(parents=True, exist_ok=True)
            config = role / "coverage.ini"
            config.write_text(
                "[run]\nparallel=True\npatch=subprocess\ndata_file="
                + str(role / ".coverage")
                + "\ninclude="
                + str(e.ROOT / "python/carnot/reporting/v685_authority_lifecycle.py")
                + ","
                + str(e.ROOT / "python/carnot/verify/*8210*.py")
                + "\n"
            )
            argv = [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                *argv[1:],
                "--junitxml=" + str(raw / "validation" / (name + ".xml")),
            ]
        elif name == "owned_tests":
            argv += ["--junitxml=" + str(raw / "validation/owned.xml")]
        receipt = child(name, argv, logs, **kwargs)
        if name.startswith("historical_"):
            destination = raw / "consumer_coverage" / name
            destination.mkdir(parents=True, exist_ok=True)
            for source in (private / name).glob(".coverage*"):
                shutil.copyfile(source, destination / source.name)
            receipt["artifact_hashes"] = {str(p): sha256_file(p) for p in destination.iterdir()}
        return receipt

    with patch.multiple(prior.inherited, e=e, child=retained):
        return list(BASE_VALIDATE(raw, private))


def main(argv: list[str] | None = None) -> int:
    """The thin CLI uses existing deadlines and publication without historical controllers."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--date" not in arguments:
        arguments += ["--date", "20261019"]
    with (
        patch.multiple(prior, e=e, episode=episode, measure=measure),
        patch.object(prior.inherited, "validate_checks", validate_checks),
    ):
        return int(prior.main(arguments))
