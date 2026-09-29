"""Qualify current source-view mechanics with dated, model-free fixtures.

REQ-REPORT-7893-V685. Scripted replies establish transport behavior only;
they cannot establish whether an edited source supports an answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_7881_v684_intervention_protocol as prior_code
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.verify import context_sufficiency_7854 as context
from carnot.verify import source_interventions as source

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7893_v685_intervention_protocol"
OUTPUT = ROOT / "results" / f"{NAME}.json"
RAW = ROOT / "results/raw" / NAME
PRIOR = ROOT / "results/experiment_7881_v684_intervention_protocol.json"
PRIOR_SHA256 = "sha256:3768699bb06920a4b293d70f35970fb77cfd65ae9fd5cea141af4225eb677cf8"
CLI = f"scripts/experiments/{NAME}.py"
TEST = f"tests/python/test_{NAME}.py"
MODULE = f"python/carnot/{NAME}.py"
write_fixtures = prior_code.write_fixtures
cold_replay = prior_code.cold_replay


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Make each measured boundary visible to the supervising process."""
    print(
        f"[exp7893] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} "
        f"completed_units={units}",
        flush=True,
    )


def command_manifest(scratch: Path) -> Json:
    """Freeze the consumer closure and explicit current CLI paths before work."""
    plan = prior_code.command_manifest(scratch)
    plan["schema"] = "carnot.exp7893.validation_manifest.v1"
    private_pytest = Path("/tmp") / ("carnot-7893-pytest-" + canonical_hash(str(scratch))[7:19])
    for item in [*plan["commands"], plan["repository_health_command"]]:
        item["argv"] = [
            "--basetemp=" + str(private_pytest / item["name"])
            if argument.startswith("--basetemp=")
            else argument
            for argument in item["argv"]
        ]
    plan["affected_tests"].append(TEST)
    plan["affected_sources"].extend((MODULE, CLI))
    for path in (TEST, MODULE, CLI):
        plan["source_hashes"][path] = sha256_file(ROOT / path)
    include = ",".join(
        "*/" + p.removeprefix("python/carnot/") for p in (*prior_code.MODULES, MODULE, CLI)
    )
    plan["coverage_settings"]["include"] = include
    for command in plan["commands"]:
        argv = command["argv"]
        if command["name"] in {
            "affected_pytest",
            "unit_coverage",
            "ruff_check",
            "ruff_format",
            "scoped_spec",
        }:
            argv.append(TEST)
        if command["name"] in {"ruff_check", "ruff_format", "mypy"}:
            argv.extend((MODULE, CLI))
        for index, arg in enumerate(argv):
            if arg.startswith("--include=") and command["name"].endswith("coverage"):
                argv[index] = "--include=" + include
        if command["name"] == "coverage_report":
            argv[argv.index(next(a for a in argv if a.startswith("--include=")))] = (
                "--include=*/" + NAME + ".py,*/" + prior_code.NAME + ".py"
            )
    py = str(ROOT / ".venv/bin/python")
    coverage = str(ROOT / ".venv/bin/coverage")
    covs = [
        scratch / "coverage" / f".coverage.v685.{route}"
        for route in ("fixture", "replay", "failure")
    ]
    fixture = scratch / "v685-fixture.json"
    routes = (
        ("v685_cli_fixture", "--fixture-e2e", fixture),
        ("v685_cli_replay", "--cold-replay", fixture),
        ("v685_cli_failure", "--cold-replay", scratch / "absent-v685.json"),
    )
    additions = []
    for (name, flag, path), data in zip(routes, covs, strict=True):
        additions.append(
            {
                "name": name,
                "argv": [
                    coverage,
                    "run",
                    f"--data-file={data}",
                    f"--include={include}",
                    CLI,
                    "--date",
                    "20260929",
                    flag,
                    str(path),
                ],
                "classification": "required",
                "timeout_s": 60,
                **(
                    {"expected_exit_code": 1, "expected_error_token": "FileNotFoundError"}
                    if name == "v685_cli_failure"
                    else {}
                ),
            }
        )
    combine = next(i for i, row in enumerate(plan["commands"]) if row["name"] == "coverage_combine")
    plan["commands"][combine:combine] = additions
    plan["commands"][combine + len(additions)]["argv"].extend(str(p) for p in covs)
    plan["coverage_files"].extend(str(p) for p in covs)
    plan["inapplicable_e2e"] = ["E2E-001 through E2E-015 and E2E-017: other producers"]
    plan["closure_rationale"] += "; current V685 CLI and terminal regression"
    return plan


def preflight(plan: Json) -> tuple[list[Json], Json]:
    """Separate missing historical evidence from a changed gate value."""
    checks, hashes = prior_code.preflight(plan)
    checks.append(prior_code.operand("prior_exp7881", PRIOR, "exists", True, PRIOR.is_file()))
    observed = sha256_file(PRIOR) if PRIOR.is_file() else None
    checks.append(prior_code.operand("prior_exp7881", PRIOR, "sha256", PRIOR_SHA256, observed))
    hashes["prior_exp7881"] = {
        "path": str(PRIOR),
        "sha256": observed,
        "date": "20260929",
        "role": "historical",
        "exposure_status": "historical",
    }
    if observed == PRIOR_SHA256:
        old = json.loads(PRIOR.read_text())
        checks.append(
            prior_code.operand(
                "prior_exp7881", PRIOR, "verdict_class", "disqualified", old.get("verdict_class")
            )
        )
        failures = [
            row["name"]
            for row in old.get("validation_receipts", [])
            if row.get("classification") == "required" and not row.get("passed")
        ]
        checks.append(
            prior_code.operand(
                "prior_exp7881",
                PRIOR,
                "historical_required_failures",
                ["coverage_report"],
                failures,
            )
        )
    return checks, hashes


def result_row(
    checks: list[Json],
    hashes: Json,
    rows: list[Json],
    plan: Json,
    receipts: list[Json],
    fixture: Path | None,
    protocol_path: Path | None,
    started_ns: int,
    spans: list[Json],
    flagged: bool,
) -> Json:
    """Reuse the primitive reducer while assigning custody to this producer."""
    result = prior_code.result_row(
        checks, hashes, rows, plan, receipts, fixture, protocol_path, started_ns, spans, flagged
    )
    result.update(
        schema="carnot.exp7893.intervention_result.v1",
        experiment_id=7893,
        task_id="exp7893-intervention-protocol",
        milestone="2026.09.685",
        inference_substrate="deterministic_cpu",
        validation_command_manifest_path=str(RAW / "validation_command_manifest.json"),
        fixture_rows=rows,
        terminal_validation_chain=[],
    )
    result["sample_size_budget"]["failed"] = sum(
        row["started"] and not row["completed"] and not row["censored"] for row in rows
    )
    old = json.loads(PRIOR.read_text()) if PRIOR.is_file() else {}
    result["historical_required_failures"].extend(
        {**row, "historical_producer": "exp7881-intervention-protocol"}
        for row in old.get("validation_receipts", [])
        if row.get("classification") == "required" and not row.get("passed")
    )
    result["repository_health"] = {
        "status": "historical_diagnostic_retained",
        "historical_receipt": old.get("repository_health", {}).get("diagnostic_receipt"),
        "affects_required_checks": False,
        "inapplicable_e2e": plan["inapplicable_e2e"],
    }
    result["reproducibility_checksum"] = canonical_hash(
        {
            "source_artifact_hashes": hashes,
            "source_hashes": plan["source_hashes"],
            "configuration": context.freeze_protocol(seed=prior_code.SEED),
            "command_manifest": canonical_hash(plan),
        }
    )
    result["field_principles"].update(
        {
            "fixture_rows": "Primitive rows retain all planned arms without semantic truth labels.",
            "terminal_validation_chain": "Each report is bound to exact candidate bytes.",
            "sample_size_budget": "Failed and excluded work remains visible.",
        }
    )
    return result


def validate_terminal(
    candidate: Path, scratch: Path, started: float, attempt: int
) -> tuple[bool, bool, Json]:
    """Read both real validators and bind their sealed logs to the candidate."""
    specs = [
        CommandSpec(
            "terminal_adversarial",
            (
                str(ROOT / ".venv/bin/python"),
                "-u",
                "scripts/adversarial_verify.py",
                "--json",
                str(candidate),
            ),
            "required",
            60,
        ),
        CommandSpec(
            "terminal_strict_rows",
            (
                str(ROOT / ".venv/bin/python"),
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "required",
            60,
        ),
    ]
    progress(started, "terminal", "before_subprocess", attempt)
    reports = [
        prior_code.seal(item, attempt * 2 + index, scratch)
        for index, item in enumerate(
            run_commands(ROOT, specs, log_dir=scratch / f"terminal_logs_{attempt}", heartbeat_s=30)
        )
    ]
    progress(started, "terminal", "after_subprocess", attempt + 1)
    try:
        parsed = json.loads(Path(reports[0]["log_path"]).read_text())
        flagged = bool(parsed["flagged_count"])
    except (KeyError, ValueError, OSError, TypeError):
        flagged = True
    binding = {
        "candidate_sha256": sha256_file(candidate),
        "reports": reports,
        "flagged_adversarial": flagged,
    }
    atomic_json(scratch / f"terminal_report_{attempt}.json", binding)
    return not flagged and all(row["passed"] for row in reports), flagged, binding


def run(date: str, scratch: Path, output: Path = OUTPUT) -> Json:
    """Measure fixtures, run frozen checks, and publish only checked bytes."""
    started, started_ns = time.monotonic(), time.monotonic_ns()
    progress(started, "start", "begin")
    if date != "20260929":
        raise ValueError("run_date_mismatch")
    plan = command_manifest(scratch)
    manifest_path = RAW / "validation_command_manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()) != plan:
        raise ValueError("validation_manifest_drift")
    atomic_json(manifest_path, plan)
    progress(started, "preconditions", "begin")
    phase = time.monotonic()
    checks, hashes = preflight(plan)
    spans = [
        {
            "phase": "preconditions",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(checks),
        }
    ]
    progress(started, "preconditions", "complete", len(checks))
    if any(not check["passed"] for check in checks):
        blocked = result_row(checks, hashes, [], plan, [], None, None, started_ns, spans, False)
        candidate = scratch / "blocked_candidate.json"
        atomic_json(candidate, blocked)
        prior_code.publish_exact(candidate, output)
        progress(started, "publish", "blocked")
        return blocked
    progress(started, "prepare", "begin")
    phase = time.monotonic()
    frozen = context.freeze_protocol(seed=prior_code.SEED)
    public = [json.loads(line) for line in prior_code.PUBLIC.read_text().splitlines()]
    selected = sorted(
        public, key=lambda row: source.digest(f"{prior_code.SEED}:{row['source_sha256']}".encode())
    )[:48]
    protocol_path = RAW / "four_call_protocol.json"
    atomic_json(
        protocol_path,
        {
            "schema": "carnot.exp7893.four_call_protocol.v1",
            "fixture_family_budget": 24,
            "future_family_budget": 48,
            "future_call_budget": 192,
            "model_visible_settings": frozen,
            "arms": list(context.ARMS),
            "exposed_development_families": [
                {
                    "family_id": row["family_id"],
                    "source_sha256": row["source_sha256"],
                    "answer_sha256": row["response_sha256"],
                }
                for row in selected
            ],
            "model_calls": 0,
            "semantic_sensitivity": None,
        },
    )
    fixture_path = scratch / "fixture.json"
    fixture = write_fixtures(fixture_path, scratch / "checkpoints")
    cold_replay(fixture_path)
    rows = fixture["rows"]
    spans.append(
        {"phase": "prepare", "duration_s": time.monotonic() - phase, "completed_units": len(rows)}
    )
    progress(started, "prepare", "complete", len(rows))
    progress(started, "validation", "begin")
    phase = time.monotonic()
    receipts = prior_code.execute(plan, scratch, started)
    if {row["name"] for row in receipts} != {row["name"] for row in plan["commands"]}:
        raise ValueError("missing_required_command")
    spans.append(
        {
            "phase": "validation",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(receipts),
        }
    )
    progress(started, "validation", "complete", len(receipts))
    result = result_row(
        checks, hashes, rows, plan, receipts, fixture_path, protocol_path, started_ns, spans, False
    )
    candidate = scratch / "terminal_candidate.json"
    atomic_json(candidate, result)
    passed, flagged, first = validate_terminal(candidate, scratch, started, 0)
    chain = [first]
    if not passed:
        result = result_row(
            checks,
            hashes,
            rows,
            plan,
            receipts,
            fixture_path,
            protocol_path,
            started_ns,
            spans,
            flagged,
        )
        result["honest_verdict"] = "complete_disqualified_required_checks"
        result["verdict_class"] = "disqualified"
        result["intervention_protocol_ready_score"] = 0
        result["acceptance_gate_results"]["validity"] = False
        result["acceptance_gate_results"]["readiness"] = False
        result["gate_check_summary"].extend(
            prior_code.operand(
                "terminal_validation",
                Path(item["log_path"]),
                item["name"] + ".passed",
                True,
                item["passed"],
            )
            for item in first["reports"]
            if not item["passed"]
        )
        atomic_json(candidate, result)
        passed, _retry_flagged, second = validate_terminal(candidate, scratch, started, 1)
        chain.append(second)
        if not passed:
            raise ValueError("terminal_revalidation_failed")
    final_hash = sha256_file(candidate)
    if final_hash != chain[-1]["candidate_sha256"]:
        raise ValueError("terminal_candidate_drift")
    sidecar = scratch / "terminal_validation_chain.json"
    atomic_json(sidecar, {"candidate_sha256": final_hash, "chain": chain})
    prior_code.publish_exact(candidate, output)
    progress(started, "publish", "complete", len(rows))
    return result


def main(argv: list[str] | None = None) -> int:
    """Expose dated fixture, replay, and measured run routes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--scratch", type=Path, default=RAW / "attempts/20260929")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.date != "20260929":
        raise ValueError("run_date_mismatch")
    if args.fixture_e2e:
        measured = write_fixtures(args.fixture_e2e, args.fixture_e2e.parent / "checkpoints")
        print(json.dumps({"independent_families": measured["independent_families"]}), flush=True)
        return 0
    if args.cold_replay:
        print(json.dumps(cold_replay(args.cold_replay), sort_keys=True), flush=True)
        return 0
    return int(run(args.date, args.scratch, args.output)["verdict_class"] == "disqualified")
