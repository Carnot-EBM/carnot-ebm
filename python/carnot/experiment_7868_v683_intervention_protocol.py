"""Qualify a CPU source-view protocol with exact terminal validation.

REQ-REPORT-7868-V683. Fixture agreement tests only request construction.
The original human labels stay development evidence, not fixture truth.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.verify import intervention_protocol_7868 as protocol
from carnot.verify import source_interventions as source

ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_7868_v683_intervention_protocol"
OUTPUT = ROOT / "results" / f"{NAME}.json"
RAW = ROOT / "results/raw" / NAME
PRIVATE = Path("/tmp/carnot-7868-v683-20260929")
SEED = 68301
MODEL_SPECS: list[dict[str, Any]] = []
OLD = ROOT / "results/experiment_7854_v682_intervention_protocol.json"
PUBLIC = ROOT / "results/raw/experiment_7727_v673_development_corpus/evaluation_public.jsonl"
Json = dict[str, Any]


def progress(start: float, phase: str, event: str, units: int = 0) -> None:
    """Keep the elapsed work and completed unit count visible to the supervisor."""
    print(
        f"[exp7868] phase={phase} event={event} elapsed_s={time.monotonic() - start:.3f} "
        f"completed_units={units}",
        flush=True,
    )


def methodology_fixture() -> Json:
    """Give the unchanged verifier typed evidence of this model-free method."""
    return {
        "schema": "carnot.exp7868.intervention_result.v1",
        "experiment_id": 7868,
        "task_id": "exp7868-intervention-protocol",
        "milestone": "2026.09.683",
        "run_date": "20260929",
        "honest_verdict": "complete_circular_positive_protocol_qualification",
        "verdict_class": "circular_positive",
        "verifier_is_oracle": True,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "inference_substrate_class": "no_model_load",
        "execution_venue": "host",
        "MODEL_SPECS": [],
        "model_specs": [],
        "target_model": "none (no pretrained model)",
        "model_invocation_counts": {
            "loads": 0,
            "calls": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "model_file_hashes": [],
        },
        "random_seed": SEED,
        "reproducibility_checksum": canonical_hash({"fixture": SEED}),
        "duration_s": 1.0,
        "preconditions_checked": [],
        "methodology": {
            "current_work": "Measured CPU fixture construction and verifier checks",
            "model_loads": 0,
            "model_calls": 0,
        },
    }


def operand(upstream: str, path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Distinguish a missing resource from a present but wrong field."""
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


def preflight() -> tuple[list[Json], Json, list[Json]]:
    """Authenticate the prior failure and current exposed development source."""
    inputs = {"prior_exp7854": OLD, "exposed_public": PUBLIC}
    checks = [operand(key, path, "exists", True, path.is_file()) for key, path in inputs.items()]
    hashes = {
        key: {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
            "date": "20260929",
            "role": key,
            "exposure_status": "historical" if key == "prior_exp7854" else "exposed_development",
        }
        for key, path in inputs.items()
    }
    if any(not check["passed"] for check in checks):
        return checks, hashes, []
    old = json.loads(OLD.read_text())
    checks.extend(
        [
            operand(
                "prior_exp7854", OLD, "verdict_class", "disqualified", old.get("verdict_class")
            ),
            operand(
                "prior_exp7854",
                OLD,
                "intervention_protocol_ready_score",
                0,
                old.get("intervention_protocol_ready_score"),
            ),
        ]
    )
    bad = [
        r
        for r in old.get("validation_receipts", [])
        if r.get("classification") == "required" and not r.get("passed")
    ]
    checks.append(
        operand(
            "prior_exp7854",
            OLD,
            "failed_required_checks",
            ["adversarial_verify"],
            [r["name"] for r in bad],
        )
    )
    rows = [json.loads(line) for line in PUBLIC.read_text().splitlines()]
    checks.append(operand("exposed_public", PUBLIC, "row_count", 64, len(rows)))
    for row in rows:
        checks.append(
            operand(
                "exposed_public",
                PUBLIC,
                row["family_id"] + ".source_sha256",
                source.digest(row["complete_source"].encode()),
                row["source_sha256"],
            )
        )
        checks.append(
            operand(
                "exposed_public",
                PUBLIC,
                row["family_id"] + ".response_sha256",
                source.digest(row["complete_response"].encode()),
                row["response_sha256"],
            )
        )
    return checks, hashes, rows if all(check["passed"] for check in checks) else []


def validation_plan() -> Json:
    """Freeze every current child before measuring a fixture or checking it."""
    py = str(ROOT / ".venv/bin/python")
    pytest = str(ROOT / ".venv/bin/pytest")
    coverage = str(ROOT / ".venv/bin/coverage")
    ruff = str(ROOT / ".venv/bin/ruff")
    mypy = str(ROOT / ".venv/bin/mypy")
    test = "tests/python/test_experiment_7868_v683_intervention_protocol.py"
    modules = [
        "python/carnot/verify/intervention_protocol_7868.py",
        "python/carnot/experiment_7868_v683_intervention_protocol.py",
    ]
    cli = "scripts/experiments/experiment_7868_v683_intervention_protocol.py"
    include = "*/intervention_protocol_7868.py,*/experiment_7868_v683_intervention_protocol.py"
    base = str(PRIVATE / "pytest")
    cov_unit = str(PRIVATE / "coverage/.coverage.unit")
    cov_cli = str(PRIVATE / "coverage/.coverage.cli")
    cov_all = str(PRIVATE / "coverage/.coverage.combined")
    fixture = str(PRIVATE / "cli-fixture.json")
    commands = [
        (
            "worktree_imports",
            [
                py,
                "-u",
                "-c",
                "import importlib,json; names=['carnot.verify.intervention_protocol_7868',"
                "'carnot.experiment_7868_v683_intervention_protocol']; "
                "print(json.dumps({'resolved_imports':{n:importlib.import_module(n).__file__ for n in names}}))",
            ],
            60,
        ),
        (
            "affected_pytest",
            [
                pytest,
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={base}/affected",
                test,
                "-q",
            ],
            180,
        ),
        (
            "unit_coverage",
            [
                coverage,
                "run",
                f"--data-file={cov_unit}",
                f"--include={include}",
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={base}/coverage",
                test,
                "-q",
            ],
            180,
        ),
        (
            "cli_coverage",
            [
                coverage,
                "run",
                f"--data-file={cov_cli}",
                f"--include={include}",
                cli,
                "--date",
                "20260929",
                "--fixture-e2e",
                fixture,
            ],
            60,
        ),
        (
            "coverage_combine",
            [coverage, "combine", "--keep", f"--data-file={cov_all}", cov_unit, cov_cli],
            60,
        ),
        (
            "coverage_report",
            [
                coverage,
                "report",
                f"--data-file={cov_all}",
                f"--include={include}",
                "--show-missing",
                "--fail-under=100",
            ],
            60,
        ),
        ("ruff_check", [ruff, "check", *modules, cli, test], 60),
        ("ruff_format", [ruff, "format", "--check", *modules, cli, test], 60),
        ("mypy", [mypy, "--strict", *modules, cli], 90),
        ("scoped_spec", [py, "-u", "scripts/check_spec_coverage.py", test], 60),
        (
            "cli_e2e",
            [
                py,
                "-u",
                cli,
                "--date",
                "20260929",
                "--fixture-e2e",
                str(PRIVATE / "e2e-fixture.json"),
            ],
            60,
        ),
        (
            "cold_replay",
            [
                py,
                "-u",
                cli,
                "--date",
                "20260929",
                "--cold-replay",
                str(PRIVATE / "pending_candidate.json"),
            ],
            60,
        ),
        (
            "full_pytest",
            [
                pytest,
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={base}/full",
                "tests/python",
                "-q",
            ],
            1200,
        ),
        (
            "adversarial_verify",
            [
                py,
                "-u",
                "scripts/adversarial_verify.py",
                "--json",
                str(PRIVATE / "pending_candidate.json"),
            ],
            60,
        ),
        (
            "strict_rows",
            [
                py,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(PRIVATE / "pending_candidate.json"),
            ],
            60,
        ),
    ]
    return {
        "schema": "carnot.exp7868.validation_manifest.v1",
        "affected_sources": [*modules, cli],
        "affected_tests": [test],
        "historical_manifest_path": str(
            ROOT
            / "results/raw/experiment_7854_v682_intervention_protocol/validation_command_manifest.json"
        ),
        "historical_manifest_sha256": sha256_file(
            ROOT
            / "results/raw/experiment_7854_v682_intervention_protocol/validation_command_manifest.json"
        ),
        "required_checks": [name for name, _, _ in commands],
        "commands": [
            {"name": name, "argv": argv, "classification": "required", "timeout_s": timeout}
            for name, argv, timeout in commands
        ],
        "coverage_files": [cov_unit, cov_cli],
        "coverage_settings": {"branch": False, "include": include},
        "inapplicable_e2e": [
            "E2E-001 through E2E-014 concern model, hardware or another producer",
            "E2E-015 belongs to Exp7852",
        ],
    }


def fixture_family(index: int) -> tuple[Json, str, str]:
    """Make distinct byte families while rotating transport and span failures."""
    case = index % 8
    if case == 4:
        text = ""
    elif case == 5:
        text = f"First {index}. Middle {index}. Last {index}."
    elif case == 7:
        text = f"Repeat {index}. Repeat {index}. Center {index}. Repeat {index}. Repeat {index}."
    else:
        text = f"Café {index}. Second {index}. Third {index}. Fourth {index}. Fifth {index}."
    answer = "No complete sentence" if case == 6 else f"Café {index}. Never sent."
    row = {
        "family_id": f"fixture-{index:02d}",
        "complete_source": text,
        "complete_response": answer,
        "source_sha256": source.digest(text.encode()),
        "response_sha256": source.digest(answer.encode()),
    }
    witness = 1 if case == 5 else 99 if case == 3 else 2
    reply = json.dumps({"unsupported_probability": 0.3, "source_sentence_id": witness})
    return row, "{" if case == 1 else reply, "length" if case == 2 else "stop"


def fixture_e2e(path: Path, checkpoints: Path) -> Json:
    """Checkpoint each actual CPU case by all bytes that determine its result."""
    started = time.monotonic()
    progress(started, "fixture", "begin")
    rows: list[Json] = []
    hits = 0
    code_identity = canonical_hash(
        {
            "protocol": sha256_file(Path(protocol.__file__)),
            "driver": sha256_file(Path(__file__)),
        }
    )
    for index in range(24):
        row, reply, finish = fixture_family(index)
        fixture_identity = canonical_hash(
            {"row": row, "reply": reply, "finish": finish, "seed": SEED}
        )
        identity = canonical_hash({"fixture": fixture_identity, "code": code_identity})
        checkpoint = checkpoints / f"{fixture_identity[7:]}.json"
        if checkpoint.is_file():
            saved = json.loads(checkpoint.read_text())
            if saved["identity"] != identity:
                raise ValueError("checkpoint_drift")
            result = saved["row"]
            hits += 1
        else:
            result = protocol.fixture_case(row, reply, finish, SEED)
            atomic_json(checkpoint, {"identity": identity, "row": result})
        rows.append(result)
        if index % 6 == 5:
            progress(started, "fixture", "batch", index + 1)
    payload = {
        "schema": "carnot.exp7868.fixture.v1",
        "rows": rows,
        "independent_families": 24,
        "checkpoint_hits": hits,
        "model_calls": 0,
        "model_loads": 0,
    }
    atomic_json(path, payload)
    progress(started, "fixture", "complete", 24)
    return payload


def primitive_rows(cases: list[Json]) -> list[Json]:
    """Report each planned view as a unit, including those never constructed."""
    arms = ("full_source", "witness_only", "witness_neighbors", "matched_control")
    rows: list[Json] = []
    for case in cases:
        requests = case["requests"]
        for index, arm in enumerate(arms):
            request = requests[index] if index < len(requests) else None
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


def cold_replay(path: Path) -> Json:
    """Recompute every scripted result from frozen primitive inputs."""
    payload = json.loads(path.read_text())
    if payload["schema"] == "carnot.exp7868.intervention_result.v1":
        fixture_path = Path(payload["fixture_rows_path"])
        if sha256_file(fixture_path) != payload["fixture_rows_sha256"]:
            raise ValueError("fixture_hash_drift")
        fixture = json.loads(fixture_path.read_text())
        cold_replay(fixture_path)
        if payload["rows"] != primitive_rows(fixture["rows"]):
            raise ValueError("fixture_row_drift")
        return {"families": 24, "rows": len(payload["rows"])}
    rows = payload["rows"]
    if len(rows) != 24:
        raise ValueError("fixture_count_drift")
    for index, observed in enumerate(rows):
        row, reply, finish = fixture_family(index)
        expected = protocol.fixture_case(row, reply, finish, SEED)
        if observed != expected:
            raise ValueError("fixture_row_drift")
    return {"families": 24, "rows": len(rows)}


def seal_receipt(receipt: Json, index: int) -> Json:
    """Name a closed child log by its bytes after the child has exited."""
    original = ROOT / receipt["log_path"]
    checksum = sha256_file(original)
    sealed = PRIVATE / "sealed_logs" / f"{index:02d}_{receipt['name']}_{checksum[7:]}.log"
    sealed.parent.mkdir(parents=True, exist_ok=True)
    if sealed.is_file():
        if sha256_file(sealed) != checksum:
            raise ValueError("sealed_log_drift")
    else:
        shutil.copyfile(original, sealed)
    return {**receipt, "log_path": str(sealed), "log_sha256": checksum}


def validate(plan: Json, started: float) -> list[Json]:
    """Execute the frozen commands and retain every real exit in order."""
    receipts = []
    for index, command in enumerate(plan["commands"]):
        progress(started, command["name"], "before_subprocess", index)
        for argument in command["argv"]:
            if argument.startswith(("--basetemp=", "--data-file=")):
                Path(argument.partition("=")[2]).parent.mkdir(parents=True, exist_ok=True)
        spec = CommandSpec(
            command["name"], tuple(command["argv"]), command["classification"], command["timeout_s"]
        )
        receipt = run_commands(
            ROOT,
            [spec],
            log_dir=PRIVATE / "child_logs" / command["name"],
            extra_env={"CARNOT_FORCE_LIVE": "1", "JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )[0]
        if receipt["command_argv"] != command["argv"]:
            raise ValueError("child_command_drift")
        receipts.append(
            {**seal_receipt(receipt, index), "classification": command["classification"]}
        )
        progress(started, command["name"], "after_subprocess", index + 1)
    return receipts


def build_artifact(
    checks: list[Json],
    hashes: Json,
    rows: list[Json],
    fixture_path: Path | None,
    protocol_path: Path | None,
    plan: Json,
    receipts: list[Json],
    started_ns: int,
    spans: list[Json],
    flagged: bool,
) -> Json:
    """Make one honest status from primitive rows and all required exits."""
    failures = [check for check in checks if not check["passed"]]
    failed_owned = [r for r in receipts if r["classification"] == "required" and not r["passed"]]
    failures.extend(
        operand("current_validation", Path(r["log_path"]), r["name"] + ".passed", True, False)
        for r in failed_owned
    )
    missing = set(plan["required_checks"]) - {r["name"] for r in receipts}
    blocked = any(not check["passed"] for check in checks)
    ready = not blocked and not failed_owned and not missing and not flagged
    verdict_class = (
        "blocked"
        if blocked
        else "disqualified"
        if failed_owned or flagged
        else "partial"
        if missing
        else "circular_positive"
    )
    verdict = {
        "blocked": "complete_blocked_required_source",
        "disqualified": "complete_disqualified_required_checks",
        "partial": "partial_pending_owned_validation",
        "circular_positive": "complete_circular_positive_protocol_qualification",
    }[verdict_class]
    now_ns = time.monotonic_ns()
    current = build_current_work_receipt(
        run_id="exp7868-20260929",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_details={"method": "CPU fixture construction and replay"},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=now_ns,
        phase_spans=spans,
    )
    code = {str(ROOT / path): sha256_file(ROOT / path) for path in plan["affected_sources"]}
    result = methodology_fixture()
    result.update(
        {
            "honest_verdict": verdict,
            "verdict_class": verdict_class,
            "flagged_adversarial": flagged,
            "gate_check_summary": failures,
            "rows": rows,
            "sample_size_budget": {
                "intended": 96,
                "eligible": len(rows),
                "started": sum(bool(r["started"]) for r in rows),
                "completed": sum(bool(r["completed"]) for r in rows),
                "censored": sum(bool(r["censored"]) for r in rows),
                "excluded": sum(bool(r["excluded"]) for r in rows),
                "independent": len({r["family_id"] for r in rows}),
            },
            "duration_s": current["duration_s"],
            "phase_spans": spans,
            "reproducibility_checksum": canonical_hash(
                {
                    "code": code,
                    "inputs": hashes,
                    "seed": SEED,
                    "plan": canonical_hash(plan),
                }
            ),
            "source_artifact_hashes": hashes,
            "preconditions_checked": checks,
            "validation_receipts": receipts,
            "validation_command_manifest_path": str(RAW / "validation_command_manifest.json"),
            "observed_child_commands": [
                {
                    "name": r["name"],
                    "argv": r["command_argv"],
                    "classification": r["classification"],
                }
                for r in receipts
            ],
            "repository_health": {
                "prior_required_failure": {
                    "path": str(OLD),
                    "sha256": sha256_file(OLD) if OLD.is_file() else None,
                    "failed_checks": ["adversarial_verify"],
                    "flag_kind": "METHODOLOGY_MISSING",
                },
                "inapplicable_e2e": plan["inapplicable_e2e"],
            },
            "acceptance_gate_results": {
                "validity": ready,
                "readiness": ready,
                "probability_quality": None,
                "decision_benefit": None,
                "retention": None,
                "efficiency": None,
            },
            "claim_scope": "CPU scripted fixture conformance; exposed development evidence only",
            "planned_inference_substrate_class": "no_model_load",
            "current_work_receipt": current,
            "intervention_protocol_ready_score": int(ready),
            "protocol_manifest_path": str(protocol_path) if protocol_path else None,
            "protocol_manifest_sha256": sha256_file(protocol_path) if protocol_path else None,
            "fixture_rows_path": str(fixture_path) if fixture_path else None,
            "fixture_rows_sha256": sha256_file(fixture_path) if fixture_path else None,
            "syntax_valid": [r["syntax_valid"] for r in rows],
            "source_byte_fidelity": [r["source_byte_fidelity"] for r in rows],
            "semantic_sensitivity": None,
        }
    )
    result["field_principles"] = {
        key: "Retain exact identity, observed work, source custody or failed gate for replay."
        for key in result
    }
    result["field_principles"].update(
        {
            "syntax_valid": "Parsing a response does not prove its claim.",
            "source_byte_fidelity": "Copied source bytes must match original UTF-8 offsets.",
            "semantic_sensitivity": "CPU fixtures do not measure Qwen sensitivity.",
            "intervention_protocol_ready_score": "Every required check must pass on terminal bytes.",
            "acceptance_gate_results": "Validity and readiness differ from unmeasured benefit.",
            "model_invocation_counts": "Cited model names are not current model calls.",
        }
    )
    return result


def run_experiment(date: str) -> Json:
    """Preflight, execute owned CPU work, validate, then publish one result."""
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    progress(started, "start", "begin")
    if date != "20260929":
        raise ValueError("run_date_mismatch")
    plan = validation_plan()
    plan_path = RAW / "validation_command_manifest.json"
    if plan_path.is_file() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("validation_manifest_drift")
    atomic_json(plan_path, plan)
    phase = time.monotonic()
    progress(started, "preconditions", "begin")
    checks, hashes, public = preflight()
    spans = [
        {
            "phase": "preconditions",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(checks),
        }
    ]
    progress(started, "preconditions", "complete", len(checks))
    if any(not check["passed"] for check in checks):
        result = build_artifact(checks, hashes, [], None, None, plan, [], started_ns, spans, False)
        atomic_json(OUTPUT, result)
        progress(started, "publish", "blocked")
        return result
    phase = time.monotonic()
    progress(started, "prepare", "begin")
    selected = sorted(
        public, key=lambda row: source.digest(f"{SEED}:{row['source_sha256']}".encode())
    )[:48]
    protocol_manifest = {
        "schema": "carnot.exp7868.four_call_protocol.v1",
        "seed": SEED,
        "future_family_budget": 48,
        "future_call_budget": 192,
        "first_answer_sentence_unchanged": True,
        "arms": ["full_source", "witness_only", "witness_neighbors", "matched_control"],
        "model_visible_settings": __import__(
            "carnot.verify.context_sufficiency_7854", fromlist=["freeze_protocol"]
        ).freeze_protocol(seed=SEED),
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
    fixture_path = PRIVATE / "fixture.json"
    fixture = fixture_e2e(fixture_path, PRIVATE / "checkpoints")
    cold_replay(fixture_path)
    rows = primitive_rows(fixture["rows"])
    spans.append(
        {"phase": "prepare", "duration_s": time.monotonic() - phase, "completed_units": len(rows)}
    )
    progress(started, "prepare", "complete", len(rows))
    pending = build_artifact(
        checks, hashes, rows, fixture_path, protocol_path, plan, [], started_ns, spans, False
    )
    atomic_json(PRIVATE / "pending_candidate.json", pending)
    phase = time.monotonic()
    progress(started, "validation", "begin")
    receipts = validate(plan, started)
    spans.append(
        {
            "phase": "validation",
            "duration_s": time.monotonic() - phase,
            "completed_units": len(receipts),
        }
    )
    adverse = next((r for r in receipts if r["name"] == "adversarial_verify"), None)
    try:
        flagged = bool(json.loads(Path(adverse["log_path"]).read_text())["flagged_count"])
    except (KeyError, TypeError, ValueError, OSError):
        flagged = True
    result = build_artifact(
        checks,
        hashes,
        rows,
        fixture_path,
        protocol_path,
        plan,
        receipts,
        started_ns,
        spans,
        flagged,
    )
    atomic_json(OUTPUT, result)
    progress(started, "terminal", "begin", len(receipts))
    terminal = [
        CommandSpec(
            "terminal_adversarial",
            (
                str(ROOT / ".venv/bin/python"),
                "-u",
                "scripts/adversarial_verify.py",
                "--json",
                str(OUTPUT),
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
                str(OUTPUT),
            ),
            "required",
            60,
        ),
    ]
    final_receipts = [
        seal_receipt(r, len(receipts) + i)
        for i, r in enumerate(
            run_commands(ROOT, terminal, log_dir=PRIVATE / "terminal_logs", heartbeat_s=30)
        )
    ]
    try:
        final_flagged = bool(
            json.loads(Path(final_receipts[0]["log_path"]).read_text())["flagged_count"]
        )
    except (KeyError, TypeError, ValueError, OSError):
        final_flagged = True
    if final_flagged or any(not r["passed"] for r in final_receipts):
        result = build_artifact(
            checks,
            hashes,
            rows,
            fixture_path,
            protocol_path,
            plan,
            receipts,
            started_ns,
            spans,
            True,
        )
        result["gate_check_summary"].extend(
            operand("terminal_validation", Path(r["log_path"]), r["name"] + ".passed", True, False)
            for r in final_receipts
            if not r["passed"]
        )
        atomic_json(OUTPUT, result)
    progress(started, "publish", "complete", len(rows))
    return result


def main(argv: list[str] | None = None) -> int:
    """Expose bounded fixture and cold-replay routes without model loading."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.date != "20260929":
        raise ValueError("run_date_mismatch")
    if args.fixture_e2e:
        result = fixture_e2e(args.fixture_e2e, args.fixture_e2e.parent / "checkpoints")
        print(json.dumps({"independent_families": result["independent_families"]}), flush=True)
        return 0
    if args.cold_replay:
        print(json.dumps(cold_replay(args.cold_replay), sort_keys=True), flush=True)
        return 0
    return int(run_experiment(args.date)["verdict_class"] == "disqualified")
