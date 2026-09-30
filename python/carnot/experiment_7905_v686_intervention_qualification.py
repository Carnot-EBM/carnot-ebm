"""Qualify dated intervention mechanics with a measured CPU fixture.

REQ-REPORT-7905-V686. Scripted source replies test transport and custody. They
cannot establish whether a changed source supports an answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
from typing import Any

from carnot import experiment_7881_v684_intervention_protocol as prior_code
from carnot import experiment_7893_v685_intervention_protocol as previous
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import context_sufficiency_7854 as context
from carnot.verify import source_interventions as source

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7905_v686_intervention_qualification"
OUTPUT = ROOT / "results" / f"{NAME}.json"
RAW = ROOT / "results/raw" / NAME
PRIOR = ROOT / "results/experiment_7893_v685_intervention_protocol.json"
PRIOR_SHA256 = "sha256:b166ff59400288338f379ad20ad3663b93bb5ff2e1cc82138d95523da0ca4ebf"
CLI = f"scripts/experiments/{NAME}.py"
TEST = f"tests/python/test_{NAME}.py"
MODULE = f"python/carnot/{NAME}.py"
write_fixtures = prior_code.write_fixtures
cold_replay = prior_code.cold_replay
validate_terminal = previous.validate_terminal


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Print completed work so a supervisor can detect a stalled child."""
    print(
        f"[exp7905] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} "
        f"completed_units={units}",
        flush=True,
    )


def command_manifest(scratch: Path) -> Json:
    """Freeze every affected command before fixture work or subprocesses."""
    plan = previous.command_manifest(scratch)
    plan["schema"] = "carnot.exp7905.validation_manifest.v1"
    plan["affected_tests"].append(TEST)
    plan["affected_sources"].extend((MODULE, CLI))
    for path in (TEST, MODULE, CLI):
        plan["source_hashes"][path] = sha256_file(ROOT / path)
    include = ",".join(
        "*/" + path.removeprefix("python/carnot/")
        for path in (*prior_code.MODULES, previous.MODULE, previous.CLI, MODULE, CLI)
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
        if command["name"] == "worktree_imports":
            argv[-1] = argv[-1].replace(
                "; print(json.dumps", f"; names += ['carnot.{NAME}']; print(json.dumps"
            )
        if command["name"] in {"e2e_016_fixture", "e2e_016_replay"}:
            argv[argv.index("--date") + 1] = "20260930"
            argv[2] = CLI
        for index, argument in enumerate(argv):
            if argument.startswith("--include=") and command["name"].endswith("coverage"):
                argv[index] = "--include=" + include
        if command["name"] == "coverage_report":
            argv[next(i for i, a in enumerate(argv) if a.startswith("--include="))] = (
                "--include=*/"
                + NAME
                + ".py,*/"
                + previous.NAME
                + ".py,*/"
                + prior_code.NAME
                + ".py"
            )
    coverage = str(ROOT / ".venv/bin/coverage")
    fixture = scratch / "v686-fixture.json"
    additions = []
    for name, route, path in (
        ("v686_cli_fixture", "--fixture-e2e", fixture),
        ("v686_cli_replay", "--cold-replay", fixture),
        ("v686_cli_failure", "--cold-replay", scratch / "absent-v686.json"),
    ):
        data = scratch / "coverage" / f".coverage.{name}"
        plan["coverage_files"].append(str(data))
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
                    "20260930",
                    route,
                    str(path),
                ],
                "classification": "required",
                "timeout_s": 60,
                **(
                    {"expected_exit_code": 1, "expected_error_token": "FileNotFoundError"}
                    if name == "v686_cli_failure"
                    else {}
                ),
            }
        )
    combine = next(i for i, row in enumerate(plan["commands"]) if row["name"] == "coverage_combine")
    plan["commands"][combine:combine] = additions
    plan["commands"][combine + len(additions)]["argv"].extend(plan["coverage_files"][-3:])
    plan["closure_rationale"] += "; current V686 bounded CLI and terminal validation"
    return plan


def preflight(plan: Json) -> tuple[list[Json], Json]:
    """Keep missing history distinct from a changed historical verdict."""
    checks, hashes = previous.preflight(plan)
    checks.append(prior_code.operand("prior_exp7893", PRIOR, "exists", True, PRIOR.is_file()))
    observed = sha256_file(PRIOR) if PRIOR.is_file() else None
    checks.append(prior_code.operand("prior_exp7893", PRIOR, "sha256", PRIOR_SHA256, observed))
    hashes["prior_exp7893"] = {
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
                "prior_exp7893", PRIOR, "verdict_class", "disqualified", old.get("verdict_class")
            )
        )
        failures = [
            row["name"]
            for row in old.get("validation_receipts", [])
            if row.get("classification") == "required" and not row.get("passed")
        ]
        checks.append(
            prior_code.operand(
                "prior_exp7893",
                PRIOR,
                "historical_required_failures",
                ["coverage_report", "ruff_format"],
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
    """Reduce owned rows while retaining every prior failed obligation."""
    result = previous.result_row(
        checks, hashes, rows, plan, receipts, fixture, protocol_path, started_ns, spans, flagged
    )
    result.update(
        schema="carnot.exp7905.intervention_result.v1",
        experiment_id=7905,
        task_id="exp7905-intervention-qualification",
        milestone="2026.09.686",
        run_date="20260930",
        inference_substrate="deterministic_cpu",
        validation_command_manifest_path=str(RAW / "validation_command_manifest.json"),
    )
    result["sample_size_budget"]["failed"] = sum(
        bool(row["started"]) and not row["completed"] and not row["censored"] for row in rows
    )
    old = json.loads(PRIOR.read_text()) if PRIOR.is_file() else {}
    result["historical_required_failures"].extend(
        {**row, "historical_producer": "exp7893-intervention-protocol"}
        for row in old.get("validation_receipts", [])
        if row.get("classification") == "required" and not row.get("passed")
    )
    result["resolved_imports"][f"carnot.{NAME}"] = str(Path(__file__).resolve())
    result["source_artifact_hashes"] = hashes
    result["repository_health"] = {
        "status": "historical_diagnostic_retained",
        "historical_receipt": old.get("repository_health"),
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
    counts: Json = {}
    for receipt in receipts:
        if receipt["name"] == "coverage_report":
            for line in receipt.get("output_tail", "").splitlines():
                parts = line.split()
                if (
                    len(parts) >= 4
                    and parts[0].endswith(".py")
                    and parts[1].isdigit()
                    and parts[2].isdigit()
                ):
                    counts[parts[0]] = {
                        "statements": int(parts[1]),
                        "covered": int(parts[1]) - int(parts[2]),
                        "missing": int(parts[2]),
                    }
    result["coverage_statement_counts"] = counts
    result["validation_receipts"] = receipts
    result["field_principles"].update(
        {
            "coverage_statement_counts": "Each changed file needs nonempty complete statement coverage.",
            "validation_command_manifest_path": "Exact child arguments are fixed before measurement.",
            "historical_required_failures": "A current pass cannot erase previous disqualification.",
            "repository_health": "Repository debt is separate from the current required scope.",
            "sample_size_budget": "Families, arms, and failed units keep separate denominators.",
        }
    )
    return result


def run(date: str, scratch: Path, output: Path = OUTPUT) -> Json:
    """Measure fixtures, run bounded checks, and publish checked bytes only."""
    started, started_ns = time.monotonic(), time.monotonic_ns()
    progress(started, "start", "begin")
    if date != "20260930":
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
    public = [json.loads(line) for line in prior_code.PUBLIC.read_text().splitlines()]
    selected = sorted(
        public, key=lambda row: source.digest(f"{prior_code.SEED}:{row['source_sha256']}".encode())
    )[:48]
    protocol_path = RAW / "four_call_protocol.json"
    atomic_json(
        protocol_path,
        {
            "schema": "carnot.exp7905.four_call_protocol.v1",
            "fixture_family_budget": 24,
            "future_family_budget": 48,
            "future_call_budget": 192,
            "model_visible_settings": context.freeze_protocol(seed=prior_code.SEED),
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
    durable_fixture = RAW / f"fixture_{sha256_file(fixture_path)[7:]}.json"
    durable_fixture.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fixture_path, durable_fixture)
    rows = fixture["rows"]
    spans.append(
        {"phase": "prepare", "duration_s": time.monotonic() - phase, "completed_units": len(rows)}
    )
    progress(started, "prepare", "complete", len(rows))
    progress(started, "validation", "begin")
    phase = time.monotonic()
    receipts = prior_code.execute(plan, scratch, started)
    if {item["name"] for item in receipts} != {item["name"] for item in plan["commands"]}:
        raise ValueError("missing_required_command")
    receipts = [prior_code.seal(item, index, RAW) for index, item in enumerate(receipts)]
    spans.append(
        {
            "phase": "validation",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(receipts),
        }
    )
    progress(started, "validation", "complete", len(receipts))
    result = result_row(
        checks,
        hashes,
        rows,
        plan,
        receipts,
        durable_fixture,
        protocol_path,
        started_ns,
        spans,
        False,
    )
    candidate = scratch / "terminal_candidate.json"
    atomic_json(candidate, result)
    chain: list[Json] = []
    reclassified = False
    for attempt in range(2):
        passed, flagged, binding = validate_terminal(candidate, scratch, started, attempt)
        binding["reports"] = [
            prior_code.seal(item, len(receipts) + attempt * 2 + index, RAW)
            for index, item in enumerate(binding["reports"])
        ]
        chain.append(binding)
        if not passed:
            if attempt:
                raise ValueError("terminal_revalidation_failed")
            result = result_row(
                checks,
                hashes,
                rows,
                plan,
                receipts,
                durable_fixture,
                protocol_path,
                started_ns,
                spans,
                flagged,
            )
            result.update(
                honest_verdict="complete_disqualified_required_checks",
                verdict_class="disqualified",
                intervention_protocol_ready_score=0,
            )
            result["acceptance_gate_results"].update(validity=False, readiness=False)
            result["gate_check_summary"].extend(
                prior_code.operand(
                    "terminal_validation",
                    Path(item["log_path"]),
                    item["name"] + ".passed",
                    True,
                    item["passed"],
                )
                for item in binding["reports"]
                if not item["passed"]
            )
            atomic_json(candidate, result)
            reclassified = True
    if reclassified:
        passed, _flagged, binding = validate_terminal(candidate, scratch, started, 2)
        binding["reports"] = [
            prior_code.seal(item, len(receipts) + 4 + index, RAW)
            for index, item in enumerate(binding["reports"])
        ]
        chain.append(binding)
        if not passed:
            raise ValueError("terminal_revalidation_failed")
    final_hash = sha256_file(candidate)
    if final_hash != chain[-1]["candidate_sha256"] or chain[-2]["candidate_sha256"] != final_hash:
        raise ValueError("terminal_candidate_drift")
    atomic_json(
        RAW / f"terminal_validation_{final_hash[7:]}.json",
        {"candidate_sha256": final_hash, "chain": chain},
    )
    prior_code.publish_exact(candidate, output)
    progress(started, "publish", "complete", len(rows))
    return result


def main(argv: list[str] | None = None) -> int:
    """Expose only dated model-free fixture, replay, and measured routes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--scratch", type=Path, default=Path("/tmp/carnot-7905-v686-20260930"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.date != "20260930":
        raise ValueError("run_date_mismatch")
    if args.fixture_e2e:
        measured = write_fixtures(args.fixture_e2e, args.fixture_e2e.parent / "checkpoints")
        print(json.dumps({"independent_families": measured["independent_families"]}), flush=True)
        return 0
    if args.cold_replay:
        print(json.dumps(cold_replay(args.cold_replay), sort_keys=True), flush=True)
        return 0
    return int(run(args.date, args.scratch, args.output)["verdict_class"] == "disqualified")
