"""Seal the current CPU intervention fixture under a prospective command plan.

REQ-REPORT-7881-V684. The scripted verifier exercises transport mechanics;
its replies are not new human judgments about the source.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.verify import context_sufficiency_7854 as context
from carnot.verify import intervention_protocol_7868 as protocol
from carnot.verify import source_interventions as source

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7881_v684_intervention_protocol"
OUTPUT = ROOT / "results" / f"{NAME}.json"
RAW = ROOT / "results/raw" / NAME
PRIOR = ROOT / "results/experiment_7868_v683_intervention_protocol.json"
PUBLIC = ROOT / "results/raw/experiment_7727_v673_development_corpus/evaluation_public.jsonl"
SEED = 67801
MODEL_SPECS: list[Json] = []
TESTS = (
    "tests/python/test_experiment_7881_v684_intervention_protocol.py",
    "tests/python/test_experiment_7868_v683_intervention_protocol.py",
    "tests/python/test_experiment_7854_v682_intervention_protocol.py",
    "tests/python/test_experiment_7839_v681_intervention_protocol.py",
    "tests/python/test_experiment_7303_v642_validation_scope.py",
    "tests/python/test_current_work_receipt.py",
)
MODULES = (
    "python/carnot/experiment_7881_v684_intervention_protocol.py",
    "python/carnot/verify/intervention_protocol_7868.py",
    "python/carnot/verify/context_sufficiency_7854.py",
    "python/carnot/verify/source_interventions.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "python/carnot/reporting/current_work_receipt.py",
)
CLI = f"scripts/experiments/{NAME}.py"


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Expose elapsed work so a silent child cannot hide a stalled run."""
    print(
        f"[exp7881] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} "
        f"completed_units={units}",
        flush=True,
    )


def operand(upstream: str, path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Keep a missing file distinct from a wrong value in an existing file."""
    return {
        "upstream_id": upstream,
        "artifact_path": str(path),
        "artifact_sha256": sha256_file(path) if path.is_file() else None,
        "artifact_field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def preflight(plan: Json | None = None) -> tuple[list[Json], Json]:
    """Check historical authority and exposed source bytes before any fixture work."""
    inputs = {
        "prior_exp7868": (PRIOR, "historical"),
        "exposed_public": (PUBLIC, "exposed_development"),
    }
    checks = [
        operand(key, path, "exists", True, path.is_file()) for key, (path, _) in inputs.items()
    ]
    hashes = {
        key: {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
            "date": "20260929",
            "role": role,
            "exposure_status": role,
        }
        for key, (path, role) in inputs.items()
    }
    if plan is not None:
        for command in [*plan["commands"], plan["repository_health_command"]]:
            program = Path(command["argv"][0])
            checks.append(operand("current_program", program, "exists", True, program.is_file()))
        for label, expected in plan["source_hashes"].items():
            path = ROOT / label
            observed = sha256_file(path) if path.is_file() else None
            checks.append(operand("current_program", path, "exists", True, path.is_file()))
            checks.append(operand("current_program", path, "sha256", expected, observed))
            hashes[label] = {
                "path": str(path),
                "sha256": observed,
                "date": "20260929",
                "role": "current_code",
                "exposure_status": "current_code",
            }
    if not all(check["passed"] for check in checks):
        return checks, hashes
    prior = json.loads(PRIOR.read_text())
    checks.append(
        operand(
            "prior_exp7868",
            PRIOR,
            "schema",
            "carnot.exp7868.intervention_result.v1",
            prior.get("schema"),
        )
    )
    checks.append(
        operand("prior_exp7868", PRIOR, "verdict_class", "disqualified", prior.get("verdict_class"))
    )
    checks.append(
        operand(
            "prior_exp7868",
            PRIOR,
            "intervention_protocol_ready_score",
            0,
            prior.get("intervention_protocol_ready_score"),
        )
    )
    old_failures = [
        row["name"]
        for row in prior.get("validation_receipts", [])
        if row.get("classification") == "required" and not row.get("passed")
    ]
    checks.append(
        operand(
            "prior_exp7868", PRIOR, "historical_required_failures", ["full_pytest"], old_failures
        )
    )
    public_rows = [json.loads(line) for line in PUBLIC.read_text().splitlines()]
    checks.append(operand("exposed_public", PUBLIC, "row_count", 64, len(public_rows)))
    for row in public_rows:
        for text_field, digest_field in (
            ("complete_source", "source_sha256"),
            ("complete_response", "response_sha256"),
        ):
            checks.append(
                operand(
                    "exposed_public",
                    PUBLIC,
                    f"{row['family_id']}.{digest_field}",
                    source.digest(row[text_field].encode()),
                    row.get(digest_field),
                )
            )
    return checks, hashes


def command_manifest(scratch: Path) -> Json:
    """Name the entire affected dependency closure before any child is started."""
    py, pytest = str(ROOT / ".venv/bin/python"), str(ROOT / ".venv/bin/pytest")
    coverage, ruff, mypy = (str(ROOT / ".venv/bin" / tool) for tool in ("coverage", "ruff", "mypy"))
    common = ["-n", "0", "-o", "addopts=", "--no-cov"]
    include = ",".join("*/" + path.removeprefix("python/carnot/") for path in MODULES)
    new_include = f"*/{NAME}.py"
    reused_include = ",".join("*/" + path.removeprefix("python/carnot/") for path in MODULES[1:])
    base = scratch / "pytest"
    unit, cli_cov = scratch / "coverage/.coverage.unit", scratch / "coverage/.coverage.cli"
    replay_cov = scratch / "coverage/.coverage.replay"
    failure_cov = scratch / "coverage/.coverage.failure"
    combined = scratch / "coverage/.coverage.combined"
    fixture = scratch / "fixture-cli.json"
    replay = scratch / "e2e-016-fixture.json"
    commands = [
        (
            "worktree_imports",
            [
                py,
                "-u",
                "-c",
                "import importlib,json; names="
                + repr(
                    [
                        "carnot."
                        + p.removeprefix("python/carnot/").removesuffix(".py").replace("/", ".")
                        for p in MODULES
                    ]
                )
                + "; print(json.dumps({'resolved_imports':{n:importlib.import_module(n).__file__ for n in names}}))",
            ],
            60,
        ),
        (
            "affected_pytest",
            [pytest, *common, f"--basetemp={base / 'affected'}", *TESTS, "-q"],
            300,
        ),
        (
            "unit_coverage",
            [
                coverage,
                "run",
                f"--data-file={unit}",
                f"--include={include}",
                "-m",
                "pytest",
                *common,
                f"--basetemp={base / 'coverage'}",
                *TESTS,
                "-q",
            ],
            300,
        ),
        (
            "cli_coverage",
            [
                coverage,
                "run",
                f"--data-file={cli_cov}",
                f"--include={include}",
                CLI,
                "--date",
                "20260929",
                "--fixture-e2e",
                str(fixture),
            ],
            60,
        ),
        (
            "cli_replay_coverage",
            [coverage, "run", f"--data-file={replay_cov}", f"--include={include}",
             CLI, "--date", "20260929", "--cold-replay", str(fixture)],
            60,
        ),
        (
            "cli_failure_coverage",
            [coverage, "run", f"--data-file={failure_cov}", f"--include={include}",
             CLI, "--date", "20260929", "--cold-replay", str(scratch / "missing-fixture.json")],
            60,
        ),
        (
            "coverage_combine",
            [coverage, "combine", "--keep", f"--data-file={combined}",
             str(unit), str(cli_cov), str(replay_cov), str(failure_cov)],
            60,
        ),
        (
            "coverage_report",
            [
                coverage,
                "report",
                f"--data-file={combined}",
                f"--include={new_include}",
                "--show-missing",
                "--fail-under=100",
            ],
            60,
        ),
        (
            "reused_coverage_report",
            [coverage, "report", f"--data-file={combined}", f"--include={reused_include}"],
            60,
        ),
        ("ruff_check", [ruff, "check", *MODULES, CLI, *TESTS], 60),
        ("ruff_format", [ruff, "format", "--check", *MODULES, CLI, *TESTS], 60),
        ("mypy", [mypy, "--strict", *MODULES, CLI], 120),
        ("scoped_spec", [py, "-u", "scripts/check_spec_coverage.py", *TESTS], 60),
        (
            "e2e_016_fixture",
            [
                py,
                "-u",
                "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                "--date",
                "20260929",
                "--fixture-e2e",
                str(replay),
            ],
            60,
        ),
        (
            "e2e_016_replay",
            [
                py,
                "-u",
                "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                "--date",
                "20260929",
                "--cold-replay",
                str(replay),
            ],
            60,
        ),
        (
            "current_cli_replay",
            [py, "-u", CLI, "--date", "20260929", "--cold-replay", str(fixture)],
            60,
        ),
    ]
    return {
        "schema": "carnot.exp7881.validation_manifest.v3",
        "affected_sources": [*MODULES, CLI],
        "affected_tests": list(TESTS),
        "source_hashes": {
            path: sha256_file(ROOT / path)
            for path in (
                *MODULES,
                CLI,
                *TESTS,
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
                "scripts/check_spec_coverage.py",
            )
        },
        "closure_rationale": "Direct fixture, source-view, receipt, and scope dependencies plus existing consumers",
        "commands": [
            {"name": name, "argv": argv, "classification": "required", "timeout_s": timeout,
             **({"expected_exit_code": 1, "expected_error_token": "FileNotFoundError"}
                if name == "cli_failure_coverage" else {})}
            for name, argv, timeout in commands
        ],
        "repository_health_command": {
            "name": "repository_full_pytest",
            "classification": "diagnostic",
            "argv": [pytest, *common, f"--basetemp={base / 'repository'}", "tests/python", "-q"],
            "timeout_s": 180,
        },
        "historical_required_failures": [
            {
                "name": "full_pytest",
                "path": str(PRIOR),
                "classification": "historical_required_failure",
            }
        ],
        "prior_owned_attempt": {
            "path": str(RAW / "attempts/attempt1_candidate.json"),
            "sha256": sha256_file(RAW / "attempts/attempt1_candidate.json")
            if (RAW / "attempts/attempt1_candidate.json").is_file()
            else None,
            "disposition": "disqualified_coverage_and_publication_failure",
        },
        "second_owned_attempt": {
            "path": str(RAW / "attempts/attempt2_candidate.json"),
            "sha256": sha256_file(RAW / "attempts/attempt2_candidate.json")
            if (RAW / "attempts/attempt2_candidate.json").is_file() else None,
            "disposition": "disqualified_cold_replay_cli_coverage",
        },
        "coverage_files": [str(unit), str(cli_cov), str(replay_cov), str(failure_cov)],
        "coverage_settings": {"branch": False, "include": include},
        "inapplicable_e2e": [
            "E2E-001 through E2E-015: other model, hardware or producer owners",
            "E2E-017: supervisor delta owner",
        ],
    }


def family(index: int) -> tuple[Json, str, str]:
    """Vary original bytes while keeping each synthetic family independent."""
    case = index % 8
    if case == 4:
        original = ""
    elif case == 5:
        original = f"First {index}. Middle {index}. Last {index}."
    elif case == 7:
        original = (
            f"Repeat {index}. Repeat {index}. Center {index}. Repeat {index}. Repeat {index}."
        )
    else:
        original = f"Café {index}. Second {index}. Third {index}. Fourth {index}. Fifth {index}."
    answer = "No complete sentence" if case == 6 else f"Café {index}. Never sent."
    row = {
        "family_id": f"fixture-{index:02d}",
        "complete_source": original,
        "complete_response": answer,
        "source_sha256": source.digest(original.encode()),
        "response_sha256": source.digest(answer.encode()),
    }
    witness = 1 if case == 5 else 99 if case == 3 else 2
    reply = json.dumps({"unsupported_probability": 0.3, "source_sentence_id": witness})
    return row, "{" if case == 1 else reply, "length" if case == 2 else "stop"


def primitive_rows(cases: list[Json]) -> list[Json]:
    """Keep all planned arms visible, including arms stopped by a bad witness."""
    arms = ("full_source", "witness_only", "witness_neighbors", "matched_control")
    rows: list[Json] = []
    for case in cases:
        for index, arm in enumerate(arms):
            request = case["requests"][index] if index < len(case["requests"]) else None
            if index == 0:
                status = (
                    "completed"
                    if case["status"] == "excluded_no_matched_context"
                    else case["status"]
                )
            elif request is not None:
                status = "constructed"
            elif case["status"].startswith("excluded_"):
                status = case["status"]
            else:
                status = "unstarted_invalid_witness"
            body = json.loads(request["messages"][1]["content"]) if request else None
            rows.append(
                {
                    "family_id": case["family_id"],
                    "family": case["family_id"],
                    "arm": arm,
                    "seed": case["seed"],
                    "status": status,
                    "started": request is not None,
                    "completed": status in {"completed", "constructed"},
                    "censored": index == 0
                    and request is not None
                    and case["finish_reason"] == "length",
                    "excluded": status.startswith("excluded_"),
                    "independent": index == 0,
                    "source_sha256": case["source_sha256"],
                    "answer_sha256": case["answer_sha256"],
                    "request_sha256": source.digest(
                        json.dumps(
                            request, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                        ).encode()
                    )
                    if request
                    else None,
                    "visible_source_sentence_ids": body["visible_source_sentence_ids"]
                    if body
                    else [],
                    "source_sentence_offsets": body["source_sentence_offsets"] if body else [],
                    "syntax_valid": case["syntax_valid"] if index == 0 else None,
                    "source_byte_fidelity": case["source_byte_fidelity"] if request else None,
                    "semantic_sensitivity": None,
                    "matching_error": case["matching_error"] if arm == "matched_control" else None,
                    "nominated_risk": case.get("nominated_risk") if index == 0 else None,
                }
            )
    return rows


def write_fixtures(path: Path, checkpoints: Path) -> Json:
    """Resume only byte- and code-identical cases; never discard bad rows."""
    started = time.monotonic()
    progress(started, "fixtures", "begin")
    code = canonical_hash(
        {"protocol": sha256_file(Path(protocol.__file__)), "driver": sha256_file(Path(__file__))}
    )
    cases: list[Json] = []
    hits = 0
    for index in range(24):
        row, reply, finish = family(index)
        fixture_identity = canonical_hash(
            {"row": row, "reply": reply, "finish": finish, "seed": SEED}
        )
        identity = canonical_hash({"fixture": fixture_identity, "code": code})
        checkpoint = checkpoints / f"{fixture_identity[7:]}.json"
        if checkpoint.is_file():
            saved = json.loads(checkpoint.read_text())
            if saved["identity"] != identity:
                raise ValueError("checkpoint_drift")
            case = saved["row"]
            hits += 1
        else:
            case = protocol.fixture_case(row, reply, finish, SEED)
            atomic_json(checkpoint, {"identity": identity, "row": case})
        cases.append(case)
        if index % 6 == 5:
            progress(started, "fixtures", "batch", index + 1)
    payload = {
        "schema": "carnot.exp7881.fixture.v1",
        "cases": cases,
        "rows": primitive_rows(cases),
        "independent_families": 24,
        "checkpoint_hits": hits,
        "model_calls": 0,
        "model_loads": 0,
    }
    atomic_json(path, payload)
    progress(started, "fixtures", "complete", 24)
    return payload


def cold_replay(path: Path) -> Json:
    """Rebuild the rows from fixed primitive inputs in a fresh process."""
    payload = json.loads(path.read_text())
    if len(payload["cases"]) != 24:
        raise ValueError("fixture_count_drift")
    for index, observed in enumerate(payload["cases"]):
        row, reply, finish = family(index)
        if observed != protocol.fixture_case(row, reply, finish, SEED):
            raise ValueError("fixture_row_drift")
    if payload["rows"] != primitive_rows(payload["cases"]):
        raise ValueError("fixture_row_drift")
    return {"families": 24, "rows": len(payload["rows"])}


def seal(receipt: Json, index: int, scratch: Path) -> Json:
    """Name a closed child log by exact bytes after its process has exited."""
    original = ROOT / receipt["log_path"]
    digest = sha256_file(original)
    sealed = scratch / "sealed_logs" / f"{index:02d}_{receipt['name']}_{digest[7:]}.log"
    sealed.parent.mkdir(parents=True, exist_ok=True)
    if sealed.is_file():
        if sha256_file(sealed) != digest:
            raise ValueError("sealed_log_drift")
    else:
        shutil.copyfile(original, sealed)
    return {**receipt, "log_path": str(sealed), "log_sha256": digest}


def execute(plan: Json, scratch: Path, started: float) -> list[Json]:
    """Run declared children with heartbeats and keep every real exit."""
    receipts: list[Json] = []
    for index, command in enumerate(plan["commands"]):
        progress(started, command["name"], "before_subprocess", index)
        for argument in command["argv"]:
            if argument.startswith(("--basetemp=", "--data-file=")):
                Path(argument.partition("=")[2]).parent.mkdir(parents=True, exist_ok=True)
        spec = CommandSpec(
            command["name"], tuple(command["argv"]), "required", command["timeout_s"]
        )
        observed = run_commands(
            ROOT,
            [spec],
            log_dir=scratch / "child_logs" / command["name"],
            extra_env={"CARNOT_FORCE_LIVE": "1", "JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )[0]
        if observed["command_argv"] != command["argv"]:
            raise ValueError("child_command_drift")
        if "expected_exit_code" in command:
            observed["raw_passed"] = observed["passed"]
            observed["passed"] = (
                observed["exit_code"] == command["expected_exit_code"]
                and not observed["timed_out"]
                and command["expected_error_token"] in observed["output_tail"]
            )
            observed["expected_exit_code"] = command["expected_exit_code"]
            observed["expected_error_token"] = command["expected_error_token"]
        receipts.append({**seal(observed, index, scratch), "classification": "required"})
        progress(started, command["name"], "after_subprocess", index + 1)
    return receipts


def verdict(checks: list[Json], receipts: list[Json], flagged: bool) -> tuple[str, str, int]:
    """External blocks terminate; current failed work disqualifies readiness."""
    if any(not row["passed"] for row in checks):
        return "complete_blocked_required_source", "blocked", 0
    if flagged or any(
        row["classification"] == "required" and not row["passed"] for row in receipts
    ):
        return "complete_disqualified_required_checks", "disqualified", 0
    return "complete_circular_positive_protocol_qualification", "circular_positive", 1


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
    """Reduce primitive rows and actual exits into one closed-class artifact."""
    honest, klass, ready = verdict(checks, receipts, flagged)
    failed = [row for row in checks if not row["passed"]]
    failed.extend(
        operand("current_validation", Path(row["log_path"]), row["name"] + ".passed", True, False)
        for row in receipts
        if row["classification"] == "required" and not row["passed"]
    )
    budget = {
        "intended": 96,
        "eligible": sum(not row["excluded"] for row in rows),
        "started": sum(bool(row["started"]) for row in rows),
        "completed": sum(bool(row["completed"]) for row in rows),
        "censored": sum(bool(row["censored"]) for row in rows),
        "excluded": sum(bool(row["excluded"]) for row in rows),
        "independent": len({row["family_id"] for row in rows}),
    }
    code = {str(ROOT / path): sha256_file(ROOT / path) for path in plan["affected_sources"]}
    artifact: Json = {
        "schema": "carnot.exp7881.intervention_result.v1",
        "experiment_id": 7881,
        "task_id": "exp7881-intervention-protocol",
        "milestone": "2026.09.684",
        "run_date": "20260929",
        "honest_verdict": honest,
        "verdict_class": klass,
        "flagged_adversarial": flagged,
        "gate_check_summary": failed,
        "rows": rows,
        "sample_size_budget": budget,
        "acceptance_gate_results": {
            "validity": bool(ready),
            "readiness": bool(ready),
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": (time.monotonic_ns() - started_ns) / 1e9,
        "phase_spans": spans,
        "random_seed": SEED,
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "resolved_imports": {
            name: str(Path(importlib.import_module(name).__file__).resolve())
            for name in (
                "carnot.verify.intervention_protocol_7868",
                "carnot.verify.context_sufficiency_7854",
                "carnot.verify.source_interventions",
                "carnot.reporting.experiment_7303_validation_scope",
                "carnot.reporting.current_work_receipt",
            )
        },
        "validation_receipts": receipts,
        "validation_command_manifest_path": str(RAW / "validation_command_manifest_attempt3.json"),
        "observed_child_commands": [
            {
                "name": row["name"],
                "argv": row["command_argv"],
                "classification": row["classification"],
            }
            for row in receipts
        ],
        "historical_required_failures": [
            row
            for row in json.loads(PRIOR.read_text()).get("validation_receipts", [])
            if row.get("classification") == "required" and not row.get("passed")
        ]
        if PRIOR.is_file()
        else [],
        "prior_owned_attempts": [plan["prior_owned_attempt"], plan["second_owned_attempt"]],
        "repository_health": {
            "status": "diagnostic_not_run",
            "prior_full_suite": "timed_out",
            "command": plan["repository_health_command"],
            "inapplicable_e2e": plan["inapplicable_e2e"],
        },
        "verifier_is_oracle": True,
        "claim_scope": "CPU fixture mechanics; exposed_development sources; no independent verifier claim",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "MODEL_SPECS": MODEL_SPECS,
        "model_specs": [],
        "target_model": "none (no pretrained model)",
        "model_invocation_counts": {
            "loads": 0,
            "calls": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "model_file_hashes": [],
        },
        "trained_head_specs": [],
        "intervention_protocol_ready_score": ready,
        "protocol_manifest_path": str(protocol_path) if protocol_path else None,
        "protocol_manifest_sha256": sha256_file(protocol_path) if protocol_path else None,
        "fixture_rows_path": str(fixture) if fixture else None,
        "fixture_rows_sha256": sha256_file(fixture) if fixture else None,
        "syntax_valid": [row["syntax_valid"] for row in rows],
        "source_byte_fidelity": [row["source_byte_fidelity"] for row in rows],
        "semantic_sensitivity": None,
        "methodology": {
            "current_work": "Measured CPU fixture construction and validation",
            "model_loads": 0,
            "model_calls": 0,
        },
    }
    artifact["reproducibility_checksum"] = canonical_hash(
        {"code": code, "inputs": hashes, "seed": SEED, "plan": canonical_hash(plan)}
    )
    artifact["field_principles"] = {
        key: "Bind current work, source custody, or the exact gate for replay." for key in artifact
    }
    artifact["field_principles"].update(
        {
            "syntax_valid": "A parser result does not establish factuality.",
            "source_byte_fidelity": "Every copied span must match original UTF-8 bytes.",
            "semantic_sensitivity": "No human truth labels exist for edited CPU fixtures.",
            "intervention_protocol_ready_score": "All current affected and terminal checks must pass.",
            "acceptance_gate_results": "Mechanics and scientific benefit are separate claims.",
            "historical_required_failures": "A current scope cannot erase the earlier required timeout.",
        }
    )
    return artifact


def publish_exact(candidate: Path, output: Path) -> None:
    """Copy checked bytes beside the result before an atomic same-device rename."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    with candidate.open("rb") as source_stream, temporary.open("wb") as target_stream:
        shutil.copyfileobj(source_stream, target_stream)
        target_stream.flush()
        os.fsync(target_stream.fileno())
    if sha256_file(candidate) != sha256_file(temporary):
        raise ValueError("candidate_copy_drift")
    os.replace(temporary, output)
    candidate.unlink()


def prior_repository_health(plan: Json) -> Json | None:
    """Reuse a sealed diagnostic so a retry does not rerun the full suite."""
    previous = RAW / "attempts/attempt1_candidate.json"
    if not previous.is_file():
        return None
    receipt = json.loads(previous.read_text())["repository_health"]["diagnostic_receipt"]
    if receipt["command_argv"] != plan["repository_health_command"]["argv"]:
        raise ValueError("diagnostic_command_drift")
    if sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]:
        raise ValueError("diagnostic_log_drift")
    return receipt


def run(date: str, scratch: Path) -> Json:
    """Freeze inputs and commands, measure fixtures, then check terminal bytes."""
    started, started_ns = time.monotonic(), time.monotonic_ns()
    progress(started, "start", "begin")
    if date != "20260929":
        raise ValueError("run_date_mismatch")
    plan = command_manifest(scratch)
    plan_path = RAW / "validation_command_manifest_attempt3.json"
    if plan_path.is_file() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("validation_manifest_drift")
    atomic_json(plan_path, plan)
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
        atomic_json(OUTPUT, blocked)
        progress(started, "publish", "blocked")
        return blocked
    progress(started, "prepare", "begin")
    phase = time.monotonic()
    public = [json.loads(line) for line in PUBLIC.read_text().splitlines()]
    selected = sorted(
        public, key=lambda row: source.digest(f"{SEED}:{row['source_sha256']}".encode())
    )[:48]
    frozen = context.freeze_protocol(seed=SEED)
    protocol_manifest = {
        "schema": "carnot.exp7881.four_call_protocol.v1",
        "seed": SEED,
        "future_family_budget": 48,
        "future_call_budget": 192,
        "first_answer_sentence_unchanged": True,
        "arms": ["full_source", "witness_only", "witness_neighbors", "matched_control"],
        "model_visible_settings": frozen,
        "families": [
            {
                "family_id": row["family_id"],
                "source_sha256": row["source_sha256"],
                "answer_sha256": row["response_sha256"],
            }
            for row in selected
        ],
        "semantic_sensitivity": None,
    }
    protocol_path = RAW / "four_call_protocol.json"
    atomic_json(protocol_path, protocol_manifest)
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
    receipts = execute(plan, scratch, started)
    spans.append(
        {
            "phase": "validation",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(receipts),
        }
    )
    progress(started, "validation", "complete", len(receipts))
    expected = {item["name"] for item in plan["commands"]}
    observed = {item["name"] for item in receipts}
    if expected != observed:
        raise ValueError("missing_required_command")
    diagnostic = plan["repository_health_command"]
    progress(started, "repository_health", "before_subprocess", len(receipts))
    for arg in diagnostic["argv"]:
        if arg.startswith("--basetemp="):
            Path(arg.partition("=")[2]).parent.mkdir(parents=True, exist_ok=True)
    health_spec = CommandSpec(
        diagnostic["name"], tuple(diagnostic["argv"]), "diagnostic", diagnostic["timeout_s"]
    )
    health = prior_repository_health(plan)
    if health is None:
        health = seal(
            run_commands(
                ROOT,
                [health_spec],
                log_dir=scratch / "child_logs/repository_health",
                heartbeat_s=30,
            )[0],
            len(receipts),
            scratch,
        )
    progress(started, "repository_health", "after_subprocess", len(receipts) + 1)
    candidate = result_row(
        checks, hashes, rows, plan, receipts, fixture_path, protocol_path, started_ns, spans, False
    )
    candidate["repository_health"] = {
        "status": "passed" if health["passed"] else "degraded_open",
        "diagnostic_receipt": health,
        "affects_required_checks": False,
        "inapplicable_e2e": plan["inapplicable_e2e"],
    }
    candidate_path = scratch / "terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    terminal_specs = [
        CommandSpec(
            "terminal_adversarial",
            (
                str(ROOT / ".venv/bin/python"),
                "-u",
                "scripts/adversarial_verify.py",
                "--json",
                str(candidate_path),
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
                str(candidate_path),
            ),
            "required",
            60,
        ),
    ]
    progress(started, "terminal", "before_subprocess", len(receipts))
    terminal = [
        seal(item, len(receipts) + 1 + index, scratch)
        for index, item in enumerate(
            run_commands(ROOT, terminal_specs, log_dir=scratch / "terminal_logs", heartbeat_s=30)
        )
    ]
    progress(started, "terminal", "after_subprocess", len(receipts) + len(terminal))
    try:
        report = json.loads(Path(terminal[0]["log_path"]).read_text())
        flagged = bool(report["flagged_count"])
    except (KeyError, ValueError, OSError, TypeError):
        flagged = True
    if flagged or any(not item["passed"] for item in terminal):
        candidate = result_row(
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
        if any(not item["passed"] for item in terminal):
            candidate["honest_verdict"] = "complete_disqualified_required_checks"
            candidate["verdict_class"] = "disqualified"
            candidate["intervention_protocol_ready_score"] = 0
            candidate["acceptance_gate_results"]["validity"] = False
            candidate["acceptance_gate_results"]["readiness"] = False
        candidate["repository_health"] = {
            "status": "passed" if health["passed"] else "degraded_open",
            "diagnostic_receipt": health,
            "affects_required_checks": False,
            "inapplicable_e2e": plan["inapplicable_e2e"],
        }
        candidate["gate_check_summary"].extend(
            operand(
                "terminal_validation",
                Path(item["log_path"]),
                item["name"] + ".passed",
                True,
                item["passed"],
            )
            for item in terminal
            if not item["passed"]
        )
        atomic_json(candidate_path, candidate)
        terminal = [
            seal(item, len(receipts) + 3 + index, scratch)
            for index, item in enumerate(
                run_commands(
                    ROOT, terminal_specs, log_dir=scratch / "terminal_retry_logs", heartbeat_s=30
                )
            )
        ]
    sidecar = scratch / "terminal_validation_reports.json"
    atomic_json(sidecar, {"candidate_sha256": sha256_file(candidate_path), "reports": terminal})
    publish_exact(candidate_path, OUTPUT)
    progress(started, "publish", "complete", len(rows))
    return candidate


def main(argv: list[str] | None = None) -> int:
    """Provide current fixture and replay routes without loading a model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--scratch", type=Path, default=Path("/tmp/carnot-7881-v684-20260929"))
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
    return int(run(args.date, args.scratch)["verdict_class"] == "disqualified")
