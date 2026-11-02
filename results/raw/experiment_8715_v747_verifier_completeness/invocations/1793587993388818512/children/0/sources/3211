"""Qualify the two missing intervention failures without changing transport.

REQ-REPORT-7917-V687. Fixture agreement measures protocol mechanics only.
Historical disqualification remains immutable even when current checks pass.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
from typing import Any

from carnot import experiment_7905_v686_intervention_qualification as previous
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

Json = dict[str, Any]
ROOT = previous.ROOT
NAME = "experiment_7917_v687_intervention_qualification"
OUTPUT = ROOT / "results" / f"{NAME}.json"
RAW = ROOT / "results/raw" / NAME
PRIOR = previous.OUTPUT
PRIOR_SHA256 = "sha256:d1d3033a3ecd22ce6eec808c94d4ea2552703a87092a87cd95ca42bc8fc69ea2"
CLI = f"scripts/experiments/{NAME}.py"
TEST = f"tests/python/test_{NAME}.py"
MODULE = f"python/carnot/{NAME}.py"
MODEL_SPECS: list[Json] = []


def command_manifest(scratch: Path) -> Json:
    """Extend the existing roster so all child obligations remain reviewable."""
    plan = previous.command_manifest(scratch)
    plan["schema"] = "carnot.exp7917.validation_manifest.v1"
    plan["affected_tests"].append(TEST)
    plan["affected_sources"].extend((MODULE, CLI))
    for path in (TEST, MODULE, CLI):
        plan["source_hashes"][path] = sha256_file(ROOT / path)
    include = plan["coverage_settings"]["include"] + f",*/{NAME}.py"
    plan["coverage_settings"]["include"] = include
    for command in plan["commands"]:
        name, argv = command["name"], command["argv"]
        if name in {"affected_pytest", "unit_coverage", "ruff_check", "ruff_format", "scoped_spec"}:
            argv.append(TEST)
        if name in {"ruff_check", "ruff_format", "mypy"}:
            argv.extend((MODULE, CLI))
        for index, argument in enumerate(argv):
            if argument.startswith("--include="):
                argv[index] = (
                    argument + f",*/{NAME}.py"
                    if name.endswith("report")
                    else "--include=" + include
                )
        if name in {"e2e_016_fixture", "e2e_016_replay"}:
            argv[2] = "scripts/experiments/experiment_7868_v683_intervention_protocol.py"
            argv[argv.index("--date") + 1] = "20260929"
    additions = []
    for name, date, route, path, error in (
        ("fixture", "20260930", "--fixture-e2e", scratch / "v687-fixture.json", None),
        ("replay", "20260930", "--cold-replay", scratch / "v687-fixture.json", None),
        ("failure", "20260930", "--cold-replay", scratch / "absent-v687.json", "FileNotFoundError"),
        (
            "wrong_date",
            "20260929",
            "--fixture-e2e",
            scratch / "wrong-date.json",
            "run_date_mismatch",
        ),
    ):
        data = scratch / "coverage" / f".coverage.v687.{name}"
        plan["coverage_files"].append(str(data))
        additions.append(
            {
                "name": "v687_cli_" + name,
                "argv": [
                    str(ROOT / ".venv/bin/coverage"),
                    "run",
                    f"--data-file={data}",
                    "--include=" + include,
                    CLI,
                    "--date",
                    date,
                    route,
                    str(path),
                ],
                "classification": "required",
                "timeout_s": 60,
                **({"expected_exit_code": 1, "expected_error_token": error} if error else {}),
            }
        )
    combine = next(row for row in plan["commands"] if row["name"] == "coverage_combine")
    plan["commands"][plan["commands"].index(combine) : plan["commands"].index(combine)] = additions
    combine["argv"].extend(plan["coverage_files"][-4:])
    health = plan["repository_health_command"]
    health["argv"] = [
        str(ROOT / ".venv/bin/pytest"),
        "tests/python",
        "-q",
        f"--basetemp={scratch / 'repository-pytest'}",
    ]
    plan["commands"].append(health)
    plan["closure_rationale"] += "; exact historical date and repeated terminal rejection"
    return plan


def preflight(plan: Json, *, input_path: Path | None = None) -> tuple[list[Json], Json]:
    """Authenticate upstream failures without replacing missing prerequisites."""
    checks, hashes = previous.preflight(plan)
    path = PRIOR if input_path is None else input_path
    observed = sha256_file(path) if path.is_file() else None
    for field, expected, actual in (
        ("exists", True, path.is_file()),
        ("sha256", PRIOR_SHA256, observed),
    ):
        checks.append(previous.prior_code.operand("prior_exp7905", path, field, expected, actual))
    hashes["prior_exp7905"] = {
        "path": str(path),
        "sha256": observed,
        "date": "20260930",
        "role": "historical",
        "exposure_status": "historical",
    }
    if observed == PRIOR_SHA256:
        old = json.loads(path.read_text())
        for field, expected in (
            ("verdict_class", "disqualified"),
            ("intervention_protocol_ready_score", 0),
        ):
            checks.append(
                previous.prior_code.operand("prior_exp7905", path, field, expected, old[field])
            )
        counts = old["coverage_statement_counts"].values()
        total = sum(row["statements"] for row in counts)
        covered = sum(row["covered"] for row in old["coverage_statement_counts"].values())
        checks.append(
            previous.prior_code.operand(
                "prior_exp7905", path, "coverage_statement_counts", [697, 699], [covered, total]
            )
        )
    return checks, hashes


def result_row(raw: Path, *args: Any) -> Json:
    """Reuse the primitive reducer and assign current evidence to its producer."""
    result = previous.result_row(*args)
    result.update(
        schema="carnot.exp7917.intervention_result.v1",
        experiment_id=7917,
        task_id="exp7917-intervention-qualification",
        milestone="2026.09.687",
        inference_substrate="aggregation_from_upstream_artifacts",
        validation_command_manifest_path=str(raw / "validation_command_manifest.json"),
    )
    authority = result["source_artifact_hashes"]["prior_exp7905"]["path"]
    old = (
        json.loads(Path(authority).read_text())
        if result["source_artifact_hashes"]["prior_exp7905"]["sha256"] == PRIOR_SHA256
        else {}
    )
    historical = [
        dict(row, historical_producer="exp7905-intervention-qualification")
        for row in old.get("validation_receipts", [])
        if row.get("classification") == "required" and not row.get("passed")
    ]
    result["historical_required_failures"].extend(historical)
    counts = old.get("coverage_statement_counts", {})
    result["historical_coverage"] = {
        "covered": sum(row["covered"] for row in counts.values()) if counts else None,
        "statements": sum(row["statements"] for row in counts.values()) if counts else None,
        "missing": sum(row["missing"] for row in counts.values()) if counts else None,
        "verdict_class": old.get("verdict_class"),
        "receipts": historical,
    }
    result["repository_health"]["current_receipts"] = [
        row for row in result["validation_receipts"] if row["classification"] == "diagnostic"
    ]
    result["resolved_imports"][f"carnot.{NAME}"] = str(Path(__file__).resolve())
    result["rows"] = [
        dict(
            row,
            intended=True,
            eligible=not row["excluded"],
            failed=bool(row["started"] and not row["completed"] and not row["censored"]),
        )
        for row in result["rows"]
    ]
    result["fixture_rows"] = result["rows"]
    result["field_principles"]["historical_coverage"] = (
        "Current qualification cannot erase 697/699."
    )
    return result


def run(
    date: str, scratch: Path, output: Path = OUTPUT, raw: Path = RAW, input_path: Path | None = None
) -> Json:
    """Pass explicit producer hooks to the already qualified bounded runner."""
    print("[exp7917] phase=start event=begin model_loads=0 execution_date=" + date, flush=True)
    return previous.run(
        date,
        scratch,
        output,
        raw_root=raw,
        producer_id=7917,
        build_plan=command_manifest,
        check_inputs=partial(preflight, input_path=input_path),
        build_result=partial(result_row, raw),
    )


def main(argv: list[str] | None = None) -> int:
    """Keep fixture outputs and validation scratch separate from final evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--input", type=Path, default=PRIOR)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--raw-root", type=Path, default=RAW)
    parser.add_argument("--scratch", type=Path, default=Path("/tmp/carnot-7917-v687-20260930"))
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.date != "20260930":
        raise ValueError("run_date_mismatch")
    if args.fixture_e2e:
        measured = previous.write_fixtures(
            args.fixture_e2e, args.fixture_e2e.parent / "checkpoints"
        )
        print(json.dumps({"independent_families": measured["independent_families"]}), flush=True)
        return 0
    if args.cold_replay:
        print(json.dumps(previous.cold_replay(args.cold_replay)), flush=True)
        return 0
    return int(
        run(args.date, args.scratch, args.output, args.raw_root, args.input)["verdict_class"]
        == "disqualified"
    )
