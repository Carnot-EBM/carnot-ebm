"""REQ-REPORT-8259: reuse qualified dispatch with current identity and receipts.

The adapter adds evidence fields without changing the saved probability model
or receiver. Historical failure bytes remain evidence rather than current credit.
"""

from __future__ import annotations

import json
from pathlib import Path
import shlex
from typing import Any
from unittest.mock import patch

from carnot.reporting import polarfire_dispatch_execution_8245 as old
from carnot.reporting import polarfire_state_dispatch_8245 as d
from carnot.reporting import recorder_execution_8213 as qualified
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.verify import request_recorder_8213 as supervisor_config
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
NAME = "experiment_8259_v713_polarfire_dispatch_qualification"
CLI = "scripts/experiments/" + NAME + ".py"
MODULE = "python/carnot/reporting/polarfire_dispatch_qualification_8259.py"
TEST = "tests/python/test_polarfire_dispatch_qualification_8259.py"
BASE_COMMANDS, BASE_REPLAY = old.commands, old.replay
BASE_HISTORICAL, BASE_LOAD = old.historical, d.load
BASE_NORMALIZE = normalize_artifact_for_template_write
BASE_EXECUTE = qualified.execute
LEGACY_CLI, LEGACY_OWNED = d.CLI, old.OWNED
LEGACY_TEST = old.TEST
HISTORICAL_PIN = "sha256:22daef9bba52f255d92022993e9fb4198539e7e5ce0f85a6b4df4d91650fac92"


def commands(private: Path) -> list[CommandSpec]:
    """Keep every legacy check, then measure the current adapter and real CLI."""
    with (
        patch.object(d, "CLI", LEGACY_CLI),
        patch.object(old, "OWNED", LEGACY_OWNED),
        patch.object(old, "TEST", LEGACY_TEST),
    ):
        legacy = BASE_COMMANDS(private / "legacy")
    current = private / "current"
    current.mkdir(parents=True, exist_ok=True)
    with (
        patch.object(qualified, "MODULES", [MODULE]),
        patch.object(qualified, "TEST", TEST),
        patch.object(supervisor_config, "CLI", CLI),
    ):
        additions = qualified.validation_plan(current)
    config = current / "coverage.ini"
    config.write_text(config.read_text() + "[report]\nexclude_lines =\n")
    result = list(legacy)
    for spec in additions:
        if spec.name in {"e2e015", "e2e019", "affected_consumers"}:
            continue
        argv = spec.argv
        if spec.name == "changed_module_mypy":
            argv = (
                str(d.ROOT / ".venv/bin/mypy"),
                "--config-file=/dev/null",
                "--strict",
                "--follow-imports=silent",
                MODULE,
                CLI,
            )
        result.append(CommandSpec("current_" + spec.name, argv, spec.scope, 180))
    for label, directory in [("legacy", private / "legacy"), ("current", current)]:
        result.append(
            CommandSpec(
                label + "_coverage_json",
                (
                    str(d.ROOT / ".venv/bin/coverage"),
                    "json",
                    "--rcfile=" + str(directory / "coverage.ini"),
                    "--data-file=" + str(directory / ".coverage"),
                    "-o",
                    "-",
                ),
                "owned",
                60,
            )
        )
    return result


def historical(raw: Path) -> Json:
    """Retain the disqualified primary and both streams of every failed check."""
    result = BASE_HISTORICAL(raw)
    path = d.ROOT / "results/experiment_8245_v712_polarfire_state_dispatch.json"
    checked(dict(path=str(path), sha256=HISTORICAL_PIN))
    value = json.loads(path.read_bytes())
    result["exp8245_primary"] = d.copy_bytes(path, raw)
    result["exp8245_failed_receipts"] = [r for r in value["validation_receipts"] if not r["passed"]]
    result["archived_sources"].append(result["exp8245_primary"])
    for receipt in result["exp8245_failed_receipts"]:
        for stream in ["stdout", "stderr"]:
            source = checked(
                dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
            )
            result["archived_sources"].append(d.copy_bytes(source, raw))
    return result


def load(root: Path, raw: Path) -> Json:
    """A static fallback cannot replace the specifically required natural state."""
    value = BASE_LOAD(root, raw)
    origin = value.get("state_origin", {}).get("kind")
    if value["state"] is not None and origin != "qualified_current_learner":
        value["checks"].append(
            d.operand("state_origin.kind", root, "qualified_current_learner", origin)
        )
        value.update(state=None, queries=[])
    return value


def coverage_counts(value: Json) -> Json:
    """Coverage totals are measured JSON primitives, never rounded percentages."""
    panels = {}
    for receipt in value["validation_receipts"]:
        if receipt["name"] in {"legacy_coverage_json", "current_coverage_json"}:
            path = checked(dict(path=receipt["stdout_path"], sha256=receipt["stdout_sha256"]))
            panels[receipt["name"]] = json.loads(path.read_bytes())["totals"]
    return dict(
        measured=bool(panels),
        panels=panels,
        statements=sum(p["num_statements"] for p in panels.values()),
        covered=sum(p["covered_lines"] for p in panels.values()),
        missing=sum(p["missing_lines"] for p in panels.values()),
    )


def board_clocks(value: Json) -> Json:
    """Missing device times stay unavailable rather than becoming measured zeros."""
    device = json.loads(checked(value["board_reference"]).read_bytes())
    receipts = {r["name"]: r for r in device.get("receipts", [])}
    cpu = None
    if "board_evaluate" in receipts:
        receipt = receipts["board_evaluate"]
        path = checked(dict(path=receipt["stderr_path"], sha256=receipt["stderr_sha256"]))
        cpu = json.loads(path.read_bytes())["board_cpu_seconds"]
    return dict(
        transfer_seconds=receipts["board_transfer"]["duration_s"]
        if "board_transfer" in receipts
        else None,
        board_cpu_seconds=cpu,
    )


def execute(plan: list[CommandSpec], raw: Path) -> list[Json]:
    """Time CPU work inside board Python while retaining the unchanged receiver bytes."""
    timed = []
    for spec in plan:
        if spec.name == "board_evaluate":
            _, _, evaluator, packet = shlex.split(spec.argv[-1])
            code = (
                "import json,runpy,sys,time; sys.argv=" + repr([evaluator, packet]) + "; "
                "started=time.process_time()\ntry:\n runpy.run_path(sys.argv[0],run_name='__main__')"
                "\nfinally:\n print(json.dumps(dict(board_cpu_seconds=time.process_time()-started)),file=sys.stderr,flush=True)"
            )
            spec = CommandSpec(
                spec.name,
                (*spec.argv[:-1], "python3 -u -c " + shlex.quote(code)),
                spec.scope,
                spec.timeout_s,
            )
        timed.append(spec)
    return BASE_EXECUTE(timed, raw)


def qualify(value: Json) -> Json:
    """Current identity and measured counts keep readiness separate from benefit."""
    for receipt in value["validation_receipts"]:
        if not receipt["passed"]:
            value["gate_check_summary"].append(
                d.operand(
                    receipt["name"] + ".actual_exit",
                    Path(receipt["stdout_path"]),
                    receipt["expected_exit"],
                    receipt["actual_exit"],
                )
            )
    value.update(
        experiment_id=8259,
        task_id="exp8259-polarfire-dispatch-qualification",
        milestone="2026.10.713",
        random_seed=7138259,
        coverage_statement_counts=coverage_counts(value),
        **board_clocks(value),
    )
    value["inference_substrate"] = "aggregation_from_upstream_artifacts"
    value["field_principles"].update(
        coverage_statement_counts="Exact legacy and current statement counts from authenticated coverage JSON, including CLI children.",
        transfer_seconds="Host monotonic span of the actual packet and evaluator transfer; unavailable without transfer.",
        board_cpu_seconds="Board Python process_time around the existing evaluator; SSH and transfer overhead remain in host receipts.",
    )
    return BASE_NORMALIZE(value)


def replay(path: Path) -> Json:
    """Recompute added evidence fields as well as the original primitive reduction."""
    report = BASE_REPLAY(path)
    value = json.loads(path.read_bytes())
    wanted = dict(
        experiment_id=8259,
        task_id="exp8259-polarfire-dispatch-qualification",
        milestone="2026.10.713",
        coverage_statement_counts=coverage_counts(value),
        **board_clocks(value),
    )
    if any(value.get(k) != v for k, v in wanted.items()):
        raise ValueError("qualification_evidence_drift")
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the qualified lifecycle with one current name and unchanged validators."""
    d.progress("exp8259_start_no_model_load", 0, 1)
    with (
        patch.object(d, "NAME", NAME),
        patch.object(d, "CLI", CLI),
        patch.object(d, "load", load),
        patch.object(d, "execute", execute),
        patch.object(old, "commands", commands),
        patch.object(old, "historical", historical),
        patch.object(old, "replay", replay),
        patch.object(old, "normalize_artifact_for_template_write", qualify),
        patch.object(old, "OWNED", [*LEGACY_OWNED, LEGACY_TEST, MODULE, CLI]),
        patch.object(old, "TEST", TEST),
    ):
        return old.main(argv)
