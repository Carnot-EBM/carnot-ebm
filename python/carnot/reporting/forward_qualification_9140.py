"""REQ-REPORT-9140-V779: refuse an infeasible complete forward qualification.

The pinned schedule exceeds its budget before execution. Preserve every owed
node and original statement rather than relabeling a prototype as completion.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import time
from typing import Any

from coverage import Coverage

from carnot.reporting import validation_schedule_9139 as s
from carnot.reporting import validation_schedule_runner_9139 as r
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary

Json = dict[str, Any]
ROOT = s.ROOT
NAME = "experiment_9140_v779_forward_qualification"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_forward_qualification_9140.py"
OWNED = ["python/carnot/reporting/forward_qualification_9140.py", CLI]
UPSTREAM = "results/experiment_9139_v779_validation_schedule.json"
PIN = "sha256:e1a8ca891dc49a977354a53ca1746de7de8c510b6e8bfc2bb3feba77f1c30a33"
MODEL_SPECS: list[Json] = []
SEED = 7799140


def authenticate(root: Path, raw: Path) -> Json:
    """Authenticate the schedule gate before replaying any dependent authority."""
    path = root / UPSTREAM
    work: Json = dict(checks=[], references=[], receipts=[], code_config_hashes=[])
    for field, expected, observed in [
        ("sha256", PIN, s.ref(path)["sha256"] or "missing_file"),
    ]:
        work["checks"].append(s.gate(path, field, expected, observed, field, "exp9139"))
    if not work["checks"][-1]["passed"]:
        return work
    value = json.loads(s.prior.old.bound(dict(path=str(path), sha256=PIN)).read_bytes())
    work["checks"].append(
        s.gate(
            path,
            "schedule_ready_score",
            1,
            value.get("schedule_ready_score", "missing_field"),
            "schedule_ready_score",
            "exp9139",
        )
    )
    if not work["checks"][-1]["passed"]:
        return work
    r.replay(path)
    work = s.authenticate(root, raw)
    if not all(c["passed"] for c in work["checks"]):
        return work
    if work["schedule"] != value["schedule_reference"]:
        raise ValueError("schedule_drift")
    snapshots = work["historical"]["replay_input"]["work"]["historical_work"]["evidence"][
        "snapshots"
    ]
    for reference in [*snapshots.values(), *value["code_config_hashes"]]:
        source = Path(reference["path"])
        work["checks"].append(
            s.gate(
                source,
                "sha256",
                reference["sha256"],
                s.ref(source)["sha256"] or "missing_file",
                "authority_bytes",
                "original_authority",
            )
        )
        if not work["checks"][-1]["passed"]:
            return work
        work["references"].append(reference)
    target = raw / "inputs" / path.name
    target.write_bytes(path.read_bytes())
    work["schedule_primary_reference"] = dict(
        s.ref(target), imported_fields=["schedule_ready_score", "schedule_reference"]
    )
    work["references"].append(work["schedule_primary_reference"])
    work["checks"][:0] = [
        s.gate(target, "sha256", PIN, s.ref(target)["sha256"], "schedule_bytes", "exp9139"),
        s.gate(
            target,
            "schedule_ready_score",
            1,
            value["schedule_ready_score"],
            "schedule_gate",
            "exp9139",
        ),
    ]
    budget = work["schedule"]["budget_equation"]
    work["checks"].append(
        dict(
            s.ref(target),
            check="complete_schedule_projection",
            upstream="exp9139-validation-schedule",
            artifact_field="schedule_reference.budget_equation.projected_total_s",
            op="<=",
            expected=3300,
            observed=budget["projected_total_s"],
            passed=budget["projected_total_s"] <= 3300,
        )
    )
    # This producer is pinned to the infeasible revision; another schedule needs new authority.
    if budget["feasible"] or work["checks"][-1]["passed"]:
        raise ValueError("unexpected_feasible_revision")
    return work


def commands(raw: Path) -> list[tuple[str, list[str]]]:
    """Freeze new-code verification separately from the unexecuted original union."""
    py = str(ROOT / ".venv/bin/python")
    base = [py, "-m", "pytest", "-n0", "-o", "addopts=", "--no-cov", "-q"]
    return [
        (
            "owned_tests",
            [
                "env",
                "CARNOT_9125_NODE_LOG=" + str(raw / "owned_events.jsonl"),
                py,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(raw / "coverage.ini"),
                "-m",
                "pytest",
                "-n0",
                "-o",
                "addopts=",
                "--no-cov",
                "-p",
                "carnot.reporting.receipt_qualification_9125",
                "-q",
                TEST,
            ],
        ),
        (
            "e2e018_021",
            [
                *base,
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
                "tests/python/test_restricted_decision_audit_8210.py",
            ],
        ),
        ("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", *OWNED, TEST]),
        ("ruff_format", [str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, TEST]),
        (
            "strict_mypy",
            [str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED],
        ),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", TEST]),
    ]


def host_checks(raw: Path) -> list[Json]:
    """Treat unavailable imports and private storage as explicit precondition failures."""
    resources = dict(
        disk_free_bytes=r.shutil.disk_usage(raw).free,
        open_files_limit=r.resource.getrlimit(r.resource.RLIMIT_NOFILE)[0],
        imports_available=all(
            r.importlib.util.find_spec(n) is not None for n in ["pytest", "coverage"]
        ),
    )
    return [
        dict(
            s.ref(raw),
            check=key,
            upstream="current_host",
            artifact_field=key,
            op=">=",
            expected=minimum,
            observed=resources[key],
            passed=resources[key] >= minimum,
        )
        for key, minimum in [
            ("disk_free_bytes", 100_000_000),
            ("open_files_limit", 64),
            ("imports_available", True),
        ]
    ]


def verify(work: Json, raw: Path) -> None:
    """Reserve verification once and combine real child coverage only after exit."""
    raw.mkdir(parents=True, exist_ok=True)
    config = raw / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess,_exit\ndata_file="
        + str(raw / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
        + "[report]\nexclude_lines=\nomit=\n"
    )
    work["references"].append(s.ref(config))
    stop = time.monotonic() + 1200
    for name, argv in commands(raw):
        s.progress("9140_" + name + "_before")
        work["receipts"].append(
            s.prior.run_check(name, argv, raw / "checks", max(1, min(240, stop - time.monotonic())))
        )
        s.progress("9140_" + name + "_after")
    tracer = Coverage(config_file=str(config))
    tracer.combine(strict=True, keep=True)
    tracer.save()
    events = raw / "owned_events.jsonl"
    work["revision_events"] = (
        [json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
    )
    if events.exists():
        work["references"].append(s.ref(events))
    files = []
    for name in OWNED:
        _, statements, excluded, missing, _ = tracer.analysis2(str(ROOT / name))
        files.append(dict(path=name, statements=statements, excluded=excluded, missing=missing))
    total = sum(len(f["statements"]) for f in files)
    work["coverage"] = dict(
        files=files,
        statement_percent=100 * (total - sum(len(f["missing"]) for f in files)) / total,
        excluded_statements=sum(len(f["excluded"]) for f in files),
        data_reference=s.ref(raw / ".coverage"),
    )


def build(work: Json, date: str, start: int, end: int) -> Json:
    """Reduce honest blocked evidence; new-code checks never clear original debt."""
    value = r.build(work, date, start, end)
    value["current_work_receipts"][0]["current_run_id"] = "exp9140"
    value["gate_check_summary"] = [*work["checks"], *work.get("resource_checks", [])]
    value["preconditions_checked"] = work.get("resource_checks", [])
    plan = work.get("schedule", {})
    original = work.get("historical", {}).get("replay_input", {}).get("work", {})
    pins = {
        Path(p["path"]).name: p["sha256"]
        for p in original.get("previous", {}).get("code_config_hashes", [])
    }
    missing = [
        dict(row, source_sha256=pins[Path(row["path"]).name], line_numbers_are_diagnostic=True)
        for row in original.get("original_coverage", {}).get("files", [])
    ]
    for row in value["rows"]:
        row.update(
            seed=SEED,
            condition="complete_schedule_projection",
            censor_reason="authenticated_schedule_exceeds_3300_seconds",
        )
    added = [node for test in sorted(work.get("additions", {})) for node in work["additions"][test]]
    value["rows"].extend(s.reduce_nodes(added, work.get("revision_events", [])))
    counts = dict(
        intended=len(value["rows"]),
        attempted=sum(bool(row.get("phases")) for row in value["rows"]),
        completed=sum(row["status"] == "completed" for row in value["rows"]),
        failed=sum(row["status"] == "failed" for row in value["rows"]),
        censored=sum(row["status"] == "censored" for row in value["rows"]),
        effective_independent_groups=0,
    )
    value["sample_size_budget"] = counts
    for key in ["intended", "attempted", "completed", "failed", "censored"]:
        value[key + "_count"] = counts[key]
    value.update(
        experiment_id=9140,
        task_id="exp9140-forward-qualification",
        schema="carnot.forward_qualification.v779.1",
        honest_verdict="complete_blocked_infeasible_forward_schedule"
        if plan
        else "complete_blocked_upstream",
        verdict_class="blocked",
        forward_validation_ready_score=0,
        required_checks_passed=False,
        inference_substrate_class="blocked_no_run",
        random_seed=SEED,
        model_work_authorized=False,
        acceptance_gates=dict(
            authenticated_schedule=bool(plan),
            complete_original_execution=False,
            normal_children=False,
            complete_original_and_new_coverage=False,
            unchanged_checkers=True,
            complete_schedule_projection=False,
        ),
        qualified_revision_reference=dict(
            authority="carnot.forward_qualification.v779.1",
            qualified=False,
            schedule_primary_reference=work.get("schedule_primary_reference"),
            scope_hashes=dict(
                original_nodes=s.canonical_hash(plan.get("original_node_manifest", [])),
                original_assertions=s.canonical_hash(plan.get("original_assertion_hashes", [])),
            ),
            full_node_rows=value["rows"],
            semantic_control_rows=work.get("semantic_controls", []),
            count_control_rows=work.get("count_controls", []),
            complete_command_receipts=work["receipts"],
            child_coverage=work.get("coverage", {}),
            original_quarantine_preserved=bool(original),
            additions=work.get("additions", {}),
            original_command_obligations=plan.get("original_command_contract", []),
        ),
        remaining_obligations=dict(
            censored_nodes=[r["unit_id"] for r in value["rows"] if r["status"] == "censored"],
            missing_statements=missing,
            failed_controls=[
                r
                for r in work.get("semantic_controls", []) + work.get("count_controls", [])
                if not r["passed"]
            ],
            normal_children=["non_diagnostic_success", "refusal", "tamper", "cold_replay"],
            reason="projection_failure_before_original_execution"
            if plan
            else "upstream_gate_failure",
        ),
    )
    value["field_principles"].update(
        forward_validation_ready_score="Only complete actual original and new obligations authorize dependent model work.",
        qualified_revision_reference="A blocked revision records debt and cannot grant qualification.",
        remaining_obligations="Original hashes define obligations; old line numbers are diagnostic.",
    )
    saved = dict(work=work, date=date, start=start, end=end)
    value["replay_input"] = saved
    value["reproducibility_checksum"] = s.canonical_hash(saved)
    return value


def replay(path: Path) -> Json:
    """Rebuild totals and reopen pinned authority so rehashing cannot clear debt."""
    value = json.loads(path.read_bytes())
    saved = value["replay_input"]
    work = saved["work"]
    if value["reproducibility_checksum"] != s.canonical_hash(saved):
        raise ValueError("replay_hash")
    for reference in value["source_references"] + value["code_config_hashes"]:
        s.prior.old.bound(reference)
    if "schedule_primary_reference" in work:
        primary = s.prior.old.bound(work["schedule_primary_reference"])
        if s.ref(primary)["sha256"] != PIN:
            raise ValueError("schedule_authority")
        upstream = json.loads(primary.read_bytes())
        original = upstream["replay_input"]["work"]
        if any(
            work[k] != original[k]
            for k in ["schedule", "historical", "semantic_controls", "count_controls"]
        ):
            raise ValueError("scope_drift")
        projection = work["checks"][-1]
        if (
            projection["check"] != "complete_schedule_projection"
            or projection["passed"] is not False
            or projection["observed"] != work["schedule"]["budget_equation"]["projected_total_s"]
        ):
            raise ValueError("projection_drift")
    for receipt in work["receipts"]:
        for stream in ["stdout", "stderr"]:
            s.prior.old.bound(
                dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
            )
    if value != build(work, saved["date"], saved["start"], saved["end"]):
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True)


def terminal(path: Path, raw: Path) -> Json:
    """Use unchanged checkers on the exact candidate bytes before atomic exposure."""
    replay(path)
    receipts = []
    for name, script, flag in [
        ("adversarial", "scripts/adversarial_verify.py", "--json"),
        ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
    ]:
        receipts.append(
            s.prior.run_check(
                name, [str(ROOT / ".venv/bin/python"), str(ROOT / script), flag, str(path)], raw
            )
        )
    return dict(passed=all(r["passed"] for r in receipts), checks=receipts)


def main(argv: list[str] | None = None) -> int:
    """Publish terminal projection evidence and verify only the newly written code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261116")
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.replay:
            print(json.dumps(replay(args.replay)), flush=True)
            return 0
        if args.date != "20261116":
            raise ValueError("run_date")
        start = time.monotonic_ns()
        raw = args.output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        s.progress("9140_authentication_before")
        resources = host_checks(raw)
        work = (
            authenticate(args.input_root, raw)
            if all(c["passed"] for c in resources)
            else dict(checks=[], references=[], receipts=[], code_config_hashes=[])
        )
        work["resource_checks"] = resources
        work["owner_pid"] = os.getpid()
        s.progress("9140_authentication_after")
        work["code_config_hashes"] = [
            s.ref(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                "pyproject.toml",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ]
        if "schedule" in work and all(c["passed"] for c in work["resource_checks"]):
            atomic_json(
                raw / "frozen_run_plan.json",
                dict(schedule=work["schedule"], commands=commands(raw), seed=SEED, exclusions=[]),
            )
            work["references"].append(s.ref(raw / "frozen_run_plan.json"))
            collect = s.prior.run_check(
                "collect_additions",
                [
                    str(ROOT / ".venv/bin/python"),
                    "-m",
                    "pytest",
                    "-n0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "--collect-only",
                    "-q",
                    s.TEST,
                    TEST,
                ],
                raw,
                120,
            )
            work["receipts"].append(collect)
            names = Path(collect["stdout_path"]).read_text().splitlines()
            work["additions"] = {
                test: [n for n in names if n.startswith(test + "::")] for test in [s.TEST, TEST]
            }
            atomic_json(
                raw / "frozen_revision_nodes.json",
                dict(
                    additions=work["additions"],
                    original_node_manifest=work["schedule"]["original_node_manifest"],
                ),
            )
            work["references"].append(s.ref(raw / "frozen_revision_nodes.json"))
            s.progress("9140_projection_blocked_no_original_nodes_launched")
            work["repository_health"] = s.prior.run_check(
                "repository_health",
                [str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"],
                raw / "health",
                120,
            )
            verify(work, raw)
        value = build(work, args.date, start, time.monotonic_ns())
        s.progress("9140_publication_before")
        publish_primary(args.output, value, lambda p: terminal(p, raw / "terminal"))
        s.progress("9140_publication_after")
        print(
            json.dumps(
                dict(
                    path=str(args.output),
                    honest_verdict=value["honest_verdict"],
                    forward_validation_ready_score=0,
                )
            ),
            flush=True,
        )
        return 0
    except (OSError, ValueError, KeyError, StopIteration) as error:
        print("[exp9140] refused=" + str(error), flush=True)
        return 1
