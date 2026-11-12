"""REQ-REPORT-9014-SCOPED: qualify custody without rewriting failed broader history."""

from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

from carnot.reporting import invocation_provenance_9000 as original
from carnot.reporting import v770_branch_evidence as history
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
)
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured, verify_measured
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
ROOT = original.ROOT
NAME, TASK = (
    "experiment_9014_v770_scoped_provenance_qualification",
    "exp9014-scoped-provenance-qualification",
)
CLI = f"scripts/experiments/{NAME}.py"
OWNED = ["python/carnot/reporting/scoped_provenance_9014.py", CLI]
TEST = "tests/python/test_scoped_provenance_9014.py"
DIRECT = [
    original.TEST,
    "tests/python/test_graph_corpus_a_8961.py",
    "tests/python/test_graph_corpus_b_8962.py",
    "tests/python/test_graph_corpus_a_8990.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_v767_capstone_8984.py",
]
E2E = [
    ("e2e018", "tests/python/test_experiment_7891_v685_authority_lifecycle.py"),
    ("e2e021", "tests/python/test_restricted_decision_audit_8210.py"),
    (
        "e2e020",
        "tests/python/test_hard_exit_learning_qualification_8206.py::test_real_child_two_crashes_and_no_crash",
    ),
]
CHECKS = [
    "owned_tests",
    "direct_consumers",
    "e2e018",
    "e2e021",
    "e2e020",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
    "coverage_combine",
    "coverage_report",
]
RECEIPT_CONTROLS = ("missing", "stale", "failed", "timeout")
HEALTH_PINS = {p: history.PINS[p] for p in list(history.PINS)[-3:]}
REUSED = [
    *original.OWNED,
    *original.PRODUCERS,
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    "python/carnot/reporting/v767_capstone.py",
    "python/carnot/reporting/v756_branch_qualification.py",
    "python/carnot/reporting/experiment_7303_validation_scope.py",
    "python/carnot/reporting/v709_execution.py",
    "python/carnot/reporting/v770_branch_evidence.py",
    "scripts/adversarial_verify.py",
    "scripts/verdict_row_consistency_lint.py",
]
reference, retain, checksum = original.reference, original.retain, original.checksum


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Report actual work boundaries so silent children cannot hide unfinished checks."""
    print(f"[exp9014] phase={phase} completed={completed} pending={pending}", flush=True)


def foreign_argv(*args: str) -> tuple[str, ...]:
    """The real consumer must work outside the checkout without ambient imports."""
    return (
        "env",
        "-u",
        "PYTHONPATH",
        "-C",
        tempfile.gettempdir(),
        str(ROOT / ".venv/bin/python"),
        "-u",
        str(ROOT / CLI),
        *args,
    )


def validation_plan(private: Path) -> list[CommandSpec]:
    """Explicit normal-exit checks own only this consumer's new statements."""
    private.mkdir(parents=True, exist_ok=True)
    rc = private / "coverage.ini"
    rc.write_text(
        "[run]\nparallel=true\ndata_file="
        + str(private / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
        + "[report]\nexclude_lines=\n"
    )
    py = str(ROOT / ".venv/bin/python")
    common = ("--noconftest", "-s", "-n0", "-o", "addopts=", "--no-cov", "-q")
    scoped = [
        CommandSpec(
            name,
            (
                py,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(rc),
                "-m",
                "pytest",
                *common,
                "--basetemp=" + str(private / name),
                *tests,
            ),
            "current_scoped",
            300 if name == "owned_tests" else 600,
        )
        for name, tests in [("owned_tests", [TEST]), ("direct_consumers", DIRECT)]
    ]
    for name, selector in E2E:
        file, *node = selector.split("::")
        absolute = str(ROOT / file) + ("::" + node[0] if node else "")
        scoped.append(
            CommandSpec(
                name,
                (
                    "env",
                    "-u",
                    "PYTHONPATH",
                    "-C",
                    tempfile.gettempdir(),
                    py,
                    "-m",
                    "pytest",
                    *common,
                    "-o",
                    "pythonpath=" + str(ROOT / "python") + " " + str(ROOT),
                    "--basetemp=" + str(private / name),
                    absolute,
                ),
                "current_scoped",
                180,
            )
        )
    return [
        *scoped,
        CommandSpec(
            "ruff_check",
            (str(ROOT / ".venv/bin/ruff"), "check", *OWNED, TEST),
            "current_scoped",
            60,
        ),
        CommandSpec(
            "ruff_format",
            (str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, TEST),
            "current_scoped",
            60,
        ),
        CommandSpec(
            "strict_mypy",
            (
                str(ROOT / ".venv/bin/mypy"),
                "--config-file=/dev/null",
                "--strict",
                "--follow-imports=silent",
                "--ignore-missing-imports",
                *OWNED,
            ),
            "current_scoped",
            120,
        ),
        CommandSpec(
            "spec_coverage",
            (
                py,
                "scripts/check_spec_coverage.py",
                TEST,
                *DIRECT,
                *(p.split("::")[0] for _, p in E2E),
            ),
            "current_scoped",
            60,
        ),
        CommandSpec(
            "coverage_combine",
            (py, "-m", "coverage", "combine", "--rcfile=" + str(rc)),
            "current_scoped",
            60,
        ),
        CommandSpec(
            "coverage_report",
            (py, "-m", "coverage", "report", "--rcfile=" + str(rc), "--fail-under=100"),
            "current_scoped",
            60,
        ),
    ]


def scope_manifest(private: Path) -> Json:
    """Freeze original assertions and their selectors before running any test."""
    nodes = [
        dict(
            node_id=path + "::" + node.name,
            sha256=canonical_hash(ast.dump(node, include_attributes=False)),
        )
        for path in DIRECT
        for node in ast.parse((ROOT / path).read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    files = list(
        dict.fromkeys([*OWNED, TEST, *DIRECT, *REUSED, *(p.split("::")[0] for _, p in E2E)])
    )
    return dict(
        owned_paths=OWNED,
        direct_consumer_tests=DIRECT,
        original_test_node_hashes=nodes,
        code_test_hashes=[reference(ROOT / p) for p in files],
        commands=[
            dict(name=c.name, argv=list(c.argv), timeout_s=c.timeout_s)
            for c in validation_plan(private)
        ],
        required_check_names=CHECKS,
        receipt_controls=["valid", *RECEIPT_CONTROLS],
        statement_percent=100,
        excluded_statements=0,
        required_normal_exit=True,
        receipt_policy="Exact frozen commands, current monotonic interval, scope hash, code hashes and retained log hashes; no failed or timed-out receipt.",
        repository_health_separate=True,
        measurement_limit_s=3300,
        validation_reserve_s=1200,
        hard_cap_s=4800,
        claim_scope="Existing producer serialization and scoped custody only; no repository health or verifier benefit claim.",
    )


def authenticate(root: Path, raw: Path) -> tuple[Json, Json]:
    """Retain rejected operands too; a new consumer cannot repair a historical verdict."""
    rejected = [retain(root / name, raw) for name, _ in original.INPUTS if (root / name).is_file()]
    evidence = original.authenticate(root, raw)
    evidence["references"].extend(rejected)
    checks, refs = evidence["checks"], evidence["references"]
    for name, pin in HEALTH_PINS.items():
        path = root / name
        if path.is_file():
            refs.append(retain(path, raw))
        checks.append(
            original.gate(
                path,
                "sha256",
                pin,
                reference(path)["sha256"] or "missing_file",
                "historical_primary_bytes",
            )
        )
    health = history.health(root)
    checks.extend(health["checks"])
    operands = list(health["references"])
    primary = history.load(root, next(iter(HEALTH_PINS)))
    operands += primary.get("raw_shard_hashes", []) + primary.get("code_config_hashes", [])
    for index, ref in enumerate(operands):
        source = Path(ref["path"])
        path = root / (source.relative_to(ROOT) if source.is_absolute() else source)
        checks.append(
            original.gate(
                path,
                "sha256",
                ref["sha256"],
                reference(path)["sha256"] or "missing_file",
                "historical_evidence_bytes",
            )
        )
        if path.is_file():
            refs.append(retain(path, raw))
        if index % 100 == 0:
            progress("historical_custody", index + 1, len(operands) - index - 1)
    evidence["failures"] = [r for r in checks if not r["passed"]]
    return evidence, health


def receipts_pass(work: Json) -> bool:
    """Recheck current scope and exact evidence instead of trusting passing headline fields."""
    try:
        manifest, receipts = work["validation_scope_manifest"], work["validation"]["receipts"]
        expected = {c["name"]: c["argv"] for c in manifest["commands"]}
        return bool(
            sorted(expected) == sorted(CHECKS)
            and sorted(r["name"] for r in receipts) == sorted(CHECKS)
            and all(reference(Path(r["path"])) == r for r in manifest["code_test_hashes"])
            and all(
                r["passed"] is True
                and r["normal_exit"] is True
                and r["timed_out"] is False
                and type(r["exit_code"]) is int
                and r["exit_code"] == 0
                and r["ended_monotonic_ns"] >= r["started_monotonic_ns"] >= work["frozen_ns"]
                and r["scope_sha256"] == canonical_hash(manifest)
                and r["command_argv"] == expected[r["name"]]
                and r["log_sha256"] is not None
                and reference(Path(r["log_path"]))["sha256"] == r["log_sha256"]
                for r in receipts
            )
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def mutate_receipt(work: Json, kind: str) -> None:
    """Private rejection controls alter one operand without changing original evidence."""
    receipts = work["validation"]["receipts"]
    if kind == "missing":
        receipts.pop()
    elif kind == "stale":
        receipts[0]["scope_sha256"] = "sha256:stale"
    elif kind == "failed":
        receipts[0].update(passed=False, exit_code=1)
    elif kind == "timeout":
        receipts[0].update(timed_out=True, normal_exit=False)
    elif kind == "log":
        receipts[0]["log_sha256"] = "sha256:changed"
    elif kind == "command":
        receipts[0]["command_argv"] = ["unmeasured"]
    elif kind == "duplicate":
        receipts.append(deepcopy(receipts[0]))
    elif kind == "code":
        work["validation_scope_manifest"]["code_test_hashes"][0]["sha256"] = "sha256:changed"


def run_check(spec: CommandSpec, raw: Path, manifest: Json) -> Json:
    """Existing subprocess supervision supplies flushed boundaries and bounded waits."""
    started = time.monotonic_ns()
    progress("before_" + spec.name, 0, 1)
    receipt = child(
        spec.name,
        list(spec.argv),
        raw / spec.name,
        deadline=spec.timeout_s,
        heartbeat=30,
        scope=spec.scope,
    )
    receipt["command_argv"] = list(spec.argv)
    receipt.update(
        started_monotonic_ns=started,
        ended_monotonic_ns=time.monotonic_ns(),
        normal_exit=receipt["exit_code"] >= 0 and not receipt["timed_out"],
        scope_sha256=canonical_hash(manifest),
    )
    receipt["log_path"] = str(ROOT / receipt["log_path"])
    atomic_json(raw / (spec.name + ".receipt.json"), receipt)
    progress("after_" + spec.name, 1, 0)
    return receipt


def receipt_controls(work: Json, private: Path) -> list[Json]:
    """Actual CLI children accept current receipts and reject four independent mutations."""
    rows = []
    raw = Path(work["raw"])
    for kind in ("valid", *RECEIPT_CONTROLS):
        changed = deepcopy(work)
        mutate_receipt(changed, kind)
        path = private / (kind + ".json")
        atomic_json(path, changed)
        spec = CommandSpec(
            "receipt_" + kind,
            foreign_argv("--check-receipts", str(path)),
            "private_receipt_control",
            60,
        )
        receipt = run_check(spec, raw / "receipt_controls", work["validation_scope_manifest"])
        expected = 0 if kind == "valid" else 1
        rows.append(
            dict(
                name=kind,
                expected_exit=expected,
                observed_exit=receipt["exit_code"],
                passed=receipt["normal_exit"] and receipt["exit_code"] == expected,
                receipt=receipt,
                input_reference=retain(path, raw),
                fixture_only=True,
            )
        )
    return rows


def control_passed(row: Json) -> bool:
    """A passing flag cannot replace the primitive child exit and complete log bytes."""
    try:
        receipt = row["receipt"]
        expected = int(
            row["name"] in {"missing_ledger", "contradictory_current", *RECEIPT_CONTROLS}
        )
        return bool(
            row["passed"] is True
            and row["expected_exit"] == expected
            and row["observed_exit"] == receipt["exit_code"] == expected
            and receipt["normal_exit"] is True
            and receipt["timed_out"] is False
            and receipt["log_sha256"] is not None
            and reference(Path(receipt["log_path"]))["sha256"] == receipt["log_sha256"]
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


def build(work: Json) -> Json:
    """Reduce only current administrative controls; failed global health remains failed."""
    evidence, validation = work["evidence"], work["validation"]
    scoped = receipts_pass(work) and verify_measured(validation["coverage"], OWNED)
    controls = work["controls"]
    mutations = work["current_receipt_mutation_rows"]
    control_ok = (
        len(controls) == len(original.CONTROLS)
        and {r["name"] for r in controls} == set(original.CONTROLS)
        and all(control_passed(r) for r in controls)
    )
    receipt_ok = (
        len(mutations) == 1 + len(RECEIPT_CONTROLS)
        and {r["name"] for r in mutations} == {"valid", *RECEIPT_CONTROLS}
        and all(control_passed(r) for r in mutations)
    )
    terminal = work.get("terminal_validation", {}).get("passed") is True
    ready = int(scoped and control_ok and receipt_ok and terminal and not evidence["failures"])
    cls = "blocked" if evidence["failures"] else "positive" if ready else "disqualified"
    by_name = {r["name"]: r for r in [*controls, *mutations]}
    rows = []
    for name in [*original.CONTROLS, "valid", *RECEIPT_CONTROLS]:
        row = by_name.get(name)
        passed = bool(row and control_passed(row))
        rows.append(
            dict(
                unit_id=name,
                group_id=name,
                arm="serializer" if name in original.CONTROLS else "receipt_policy",
                condition=name,
                metric="expected_exit_observed",
                value=int(passed),
                numerator=int(passed),
                denominator=1,
                status="censored" if row is None else "completed" if passed else "failed",
                exclusion_reason=None
                if passed
                else "not_run_external_block"
                if row is None
                else "control_failed",
            )
        )
    value: Json = dict(
        experiment_id=9014,
        task_id=TASK,
        title="Scoped producer provenance qualification",
        milestone="2026.11.770",
        run_date="20261111",
        honest_verdict="complete_blocked_" + evidence["failures"][0]["check"]
        if cls == "blocked"
        else "complete_" + cls + "_scoped_provenance",
        verdict_class=cls,
        status="success" if ready else "blocked" if cls == "blocked" else "error",
        verifier_is_oracle=False,
        claim_scope="Producer provenance serialization and prospective scoped normal-exit validation only; no verifier benefit or repository-wide health claim.",
        exposure_scope="authenticated_historical_artifacts_and_private_schema_controls",
        independent_generalization_score=0,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        MODEL_SPECS=[],
        model_specs=[],
        trained_head_specs=[],
        model_invoked=False,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        cited_upstream_artifacts=[
            dict(
                experiment_id=int(Path(p).name.split("_")[1]),
                sha256=pin,
                fields_imported=[
                    "honest_verdict",
                    "verdict_class",
                    "repository_health",
                    "validation_receipts",
                ],
            )
            for p, pin in HEALTH_PINS.items()
        ]
        + [
            dict(
                experiment_id=int(Path(p).name.split("_")[1]),
                sha256=pin,
                fields_imported=[
                    "model_invocation_counts",
                    "invocation_ledger",
                    "preconditions_checked",
                ],
            )
            for p, pin in original.INPUTS
        ],
        required_checks_passed=bool(scoped and control_ok and receipt_ok and terminal),
        flagged_adversarial=bool(work.get("terminal_rejected")),
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        validation_scope_manifest=work["validation_scope_manifest"],
        preconditions_checked=dict(
            environment=work["environment"], checks=evidence["checks"], required_network_requests=0
        ),
        gate_check_summary=evidence["checks"],
        repository_health=work["repository_health"],
        repository_health_passed=False,
        historical_health_references=[
            r
            for r in evidence["references"]
            if r.get("source_path") in {str(ROOT / p) for p in HEALTH_PINS}
        ],
        original_verdict_preserved=all(
            r["original_verdict_class"] == "disqualified" for r in work["repository_health"]["rows"]
        ),
        original_test_node_hashes=work["validation_scope_manifest"]["original_test_node_hashes"],
        current_receipt_mutation_rows=mutations,
        provenance_ready_score=ready,
        validation_scope_ready_score=ready,
        original_flag_rows=evidence["original_flag_rows"],
        invocation_path_rows=evidence["invocation_path_rows"],
        mutation_rows=controls,
        producer_code_hashes=[
            r for r in work["code_references"] if r.get("source_path") in original.PRODUCERS
        ],
        rows=rows,
        intended_count=len(rows),
        independent_count=len(rows),
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=0,
        duration_s=work["duration_s"],
        random_seed=7709014,
        phase_spans=work["phase_spans"],
        source_artifact_hashes=evidence["references"],
        raw_shard_hashes=[reference(Path(work["raw"]) / "work.json"), *work["raw_references"]],
        code_config_hashes=work["code_references"],
        terminal_validation=work.get("terminal_validation", {}),
        fixture_only=False,
        methodology_note="No model was loaded. Existing serializers and real private CLI controls qualify custody only. Fixture shapes earn no live science credit. Three original failed full-suite observations remain authenticated and failed; the new acceptance scope was frozen before measurement.",
        acceptance_gates={
            k: dict(
                passed=v,
                principle="Authentic inputs and all current normal-exit controls qualify only this frozen consumer scope.",
            )
            for k, v in dict(
                authenticated_inputs=not evidence["failures"],
                scoped_validation=scoped,
                serializer_controls=control_ok,
                current_receipt_controls=receipt_ok,
                cold_replay_and_scanners=terminal,
            ).items()
        },
        field_principles={},
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Current counters and normal-exit receipts are separate from complete cited historical bytes; scoped custody grants no scientific benefit or repository health.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild all summary fields and reject altered bytes even after aggregate rehashing."""
    try:
        value = json.loads(path.read_bytes())
        if checksum(value) != value["reproducibility_checksum"]:
            raise ValueError("checksum_drift")
        for ref in (
            value["source_artifact_hashes"]
            + value["raw_shard_hashes"]
            + value["code_config_hashes"]
        ):
            if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("operand_drift")
        work = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_bytes())
        return dict(passed=value == build(work), replay_passed=value == build(work))
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, replay_passed=False)


def terminal(candidate: Path, raw: Path, manifest: Json) -> Json:
    """Unchanged CLI validators and a foreign-directory replay gate visible publication."""
    py = str(ROOT / ".venv/bin/python")
    specs = [
        CommandSpec("cold_replay", foreign_argv("--cold-replay", str(candidate)), "terminal", 60),
        CommandSpec(
            "adversarial",
            (py, "scripts/adversarial_verify.py", "--json", str(candidate)),
            "terminal",
            60,
        ),
        CommandSpec(
            "strict_rows",
            (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "terminal",
            60,
        ),
    ]
    receipts = [run_check(c, raw, manifest) for c in specs]
    return dict(passed=all(r["passed"] and r["normal_exit"] for r in receipts), receipts=receipts)


def execute(raw: Path, input_root: Path) -> Json:
    """Freeze scope, authenticate operands, then run bounded administrative qualification."""
    started = time.monotonic()
    private = Path(tempfile.mkdtemp(prefix="carnot-9014-"))
    manifest = scope_manifest(private)
    atomic_json(raw / "validation_scope_manifest.json", manifest)
    frozen_ns = time.monotonic_ns()
    progress("preconditions_before")
    environment = original.environment(raw)
    evidence, health = authenticate(input_root, raw)
    evidence["checks"].extend(environment["checks"])
    evidence["failures"].extend(r for r in environment["checks"] if not r["passed"])
    code = [
        dict(retain(Path(r["path"]), raw), source_path=str(Path(r["path"]).relative_to(ROOT)))
        for r in manifest["code_test_hashes"]
    ]
    progress("preconditions_after", len(evidence["checks"]), 0)
    controls = []
    if not evidence["failures"]:
        for index, kind in enumerate(original.CONTROLS):
            output = private / kind / (original.NAME + ".json")
            spec = CommandSpec(
                kind,
                original.foreign_argv(
                    "--date", "20261111", "--fixture-e2e", str(output), "--control", kind
                ),
                "private_serializer_control",
                60,
            )
            receipt = run_check(spec, raw / "serializer_controls", manifest)
            expected = 1 if kind in {"missing_ledger", "contradictory_current"} else 0
            controls.append(
                dict(
                    name=kind,
                    expected_exit=expected,
                    observed_exit=receipt["exit_code"],
                    passed=receipt["normal_exit"] and receipt["exit_code"] == expected,
                    receipt=receipt,
                    fixture_only=True,
                )
            )
            progress("serializer_controls", index + 1, len(original.CONTROLS) - index - 1)
    measured_at = time.monotonic()
    if measured_at - started >= 3300:
        raise TimeoutError("measurement_deadline")
    progress("owned_validation_before", 0, len(CHECKS))
    old_rc = os.environ.get("COVERAGE_RCFILE")
    os.environ["COVERAGE_RCFILE"] = str(private / "coverage.ini")
    try:
        receipts = []
        for index, spec in enumerate(validation_plan(private)):
            receipts.append(run_check(spec, raw / "validation", manifest))
            progress("owned_validation", index + 1, len(CHECKS) - index - 1)
    finally:
        if old_rc is None:
            os.environ.pop("COVERAGE_RCFILE", None)
        else:
            os.environ["COVERAGE_RCFILE"] = old_rc
    coverage = (
        measured([private / ".coverage"], OWNED, raw / "coverage")
        if (private / ".coverage").is_file()
        else dict(complete=False, percent=0, files=[])
    )
    coverage.pop("rows", None)
    work: Json = dict(
        raw=str(raw),
        private=str(private),
        frozen_ns=frozen_ns,
        validation_scope_manifest=manifest,
        evidence=evidence,
        environment=environment,
        repository_health=health,
        code_references=code,
        raw_references=[],
        controls=controls,
        current_receipt_mutation_rows=[],
        validation=dict(receipts=receipts, coverage=coverage),
        duration_s=time.monotonic() - started,
        phase_spans=[
            dict(phase="authentication_and_serializer_controls", duration_s=measured_at - started),
            dict(phase="validation", duration_s=time.monotonic() - measured_at),
        ],
    )
    work["current_receipt_mutation_rows"] = receipt_controls(work, private / "receipt_inputs")
    control_files = [
        p
        for name in [*original.CONTROLS, "receipt_inputs"]
        for p in sorted((private / name).rglob("*"))
        if p.is_file()
    ]
    work["raw_references"] = [retain(p, raw) for p in [private / "coverage.ini", *control_files]]
    work["duration_s"] = time.monotonic() - started
    work["raw_references"].extend(
        reference(p) for p in sorted(raw.rglob("*")) if p.is_file() and "objects" not in p.parts
    )
    atomic_json(raw / "work.json", work)
    progress("owned_validation_after", len(receipts), 0)
    return work


def main(argv: list[str] | None = None) -> int:
    """Publish checked terminal bytes; private receipt commands never modify history."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--from-work", type=Path)
    parser.add_argument("--check-receipts", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.check_receipts or args.cold_replay:
        if args.cold_replay:
            report = replay(args.cold_replay)
        else:
            report = dict(passed=receipts_pass(json.loads(args.check_receipts.read_bytes())))
        print(json.dumps(report), flush=True)
        return 0 if report["passed"] else 1
    if args.date != "20261111":
        parser.error("this consumer requires --date 20261111")
    output = (args.output or ROOT / "results" / (NAME + ".json")).absolute()
    if args.from_work and output.is_relative_to(ROOT):
        parser.error("private replay publication requires output outside the checkout")
    raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    progress("qualification_before")
    if args.from_work:
        work = json.loads(args.from_work.read_bytes())
        work["raw"] = str(raw)
        atomic_json(raw / "work.json", work)
    else:
        work = execute(raw, args.input_root)
    candidate = raw / "preflight.json"
    atomic_json(candidate, build(work))
    work["terminal_validation"] = terminal(
        candidate, raw / "preflight_checks", work["validation_scope_manifest"]
    )
    work["terminal_rejected"] = not work["terminal_validation"]["passed"]
    atomic_json(raw / "work.json", work)
    value = build(work)
    published = publish_primary(
        output,
        value,
        lambda p: terminal(p, raw / "publication_checks", work["validation_scope_manifest"]),
    )
    progress("publication_complete", 1, 0)
    print(
        json.dumps(
            dict(
                **published,
                honest_verdict=value["honest_verdict"],
                provenance_ready_score=value["provenance_ready_score"],
            )
        ),
        flush=True,
    )
    return 0
