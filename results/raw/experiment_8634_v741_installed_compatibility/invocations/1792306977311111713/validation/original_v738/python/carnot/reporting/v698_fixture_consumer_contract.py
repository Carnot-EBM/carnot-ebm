"""REQ-REPORT-8057: qualify measurement machinery without granting science credit.

The authority reader checks complete task bytes. Private consumer controls use
known outcomes, so their success remains an oracle fixture result.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone, UTC
import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any

import yaml

from carnot import experiment_8045_v697_scorer_workspace as scorer
from carnot.reporting import v685_authority_lifecycle as authority
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar, reader_receipt
from carnot.reporting.roadmap_contract import parse_design
from scripts.conductor_gates import evaluate_gates

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8057_v698_fixture_consumer_contract"
TASK = "exp8057-fixture-consumer-contract"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = "python/carnot/reporting/v698_fixture_consumer_contract.py"
TEST = "tests/python/test_fixture_consumer_contract_8057.py"
MILESTONE = "2026.10.698"
CLASSES = ("positive", "circular_positive", "null", "blocked", "disqualified", "partial")
INPUTS = [
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "ops/e2e-test-plan.md",
    "openspec/capabilities/research-reporting/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    "python/carnot/reporting/roadmap_contract.py",
    "python/carnot/reporting/v685_authority_lifecycle.py",
    "scripts/conductor_gates.py",
    "ops/exclusion_manifest.yaml",
    scorer.MODULE,
    "results/experiment_8044_v697_contract_methods.json",
    "results/experiment_8045_v697_scorer_workspace.json",
    "results/experiment_8047_fit_score_capture.json",
    "results/experiment_8056_v697_capstone.json",
    "openspec/change-proposals/research-roadmap-v697-preserved-20261003.md",
    "tests/fixtures/v697/design.md.gz",
    "tests/fixtures/v697/active.yaml.gz",
    "tests/fixtures/v698/design.md.gz",
    "tests/fixtures/v698/active.yaml.gz",
]
START = time.monotonic()


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose real work boundaries so silence cannot be mistaken for a running model."""
    print(
        f"[exp8057] phase={phase} elapsed_s={time.monotonic() - START:.3f} completed={completed} pending={pending}",
        flush=True,
    )


def failure(path: Path, field: str, expected: Any, observed: Any, upstream: str = TASK) -> Json:
    """Record the exact operand because absence and a failed scientific gate differ."""
    digest = sha256_file(path) if path.is_file() else None
    return dict(
        check=field,
        upstream=upstream,
        path=str(path),
        hash=digest,
        field=field,
        op="==",
        expected=expected,
        observed=observed,
        artifact_field=field,
        passed=False,
    )


def assess(design: Path, staged: Path, active: Path, snapshots: Path) -> Json:
    """Use the shipped reader and also bind the embedded complete task digest."""
    try:
        value = authority.assess_authorities(
            design, staged, active, snapshots, milestone=MILESTONE, first_id=8057, count=13
        )
        _, tasks = parse_design(design.read_text(), milestone=MILESTONE)
        actual = authority.tasks_digest(tasks)
        if actual != value["canonical_tasks_sha256"]:
            value["gate_check_summary"].append(
                failure(design, "design_tasks_sha256", value["canonical_tasks_sha256"], actual)
            )
        value["gate_check_summary"] = [
            {
                **r,
                **failure(
                    Path(r.get("artifact_path", r.get("path", str(design)))),
                    r.get("artifact_field", r.get("field", "authority")),
                    r["expected"],
                    r["observed"],
                    "V698_authority",
                ),
            }
            for r in value["gate_check_summary"]
        ]
        value["activated"] = not value["gate_check_summary"]
        return value
    except (OSError, ValueError, IndexError, KeyError, TypeError, yaml.YAMLError) as exc:
        return dict(
            activated=False,
            contract_rows=[],
            canonical_tasks_sha256=None,
            authority_snapshots={},
            gate_check_summary=[failure(design, "authority_readable", True, str(exc))],
        )


def resolve(root: Path, task: Json) -> Path:
    """Require the declared location and authenticated ID before invoking lax readers."""
    path = root / task["deliverable"]
    siblings = list(path.parent.glob(f"experiment_{task['id'][3:].split('-')[0]}_*.json"))
    if len(siblings) > 1:
        raise ValueError("ambiguous_fallback_paths")
    if not path.is_file():
        raise ValueError("missing_declared_path")
    if json.loads(path.read_text()).get("task_id") != task["id"]:
        raise ValueError("task_identity")
    return path


def clean_terminal(primary: Path) -> Json:
    """Authenticate both terminal evidence and the validator bound to primary bytes."""
    value = json.loads(primary.read_text())
    terminal = (
        Path(value["terminal_validation_sidecar_path"])
        if value.get("terminal_validation_sidecar_path")
        else primary.parent / "absent_terminal"
    )
    if not terminal.is_file():
        raise ValueError("missing_terminal")
    result = json.loads(terminal.read_text())
    binding = result.get("publication", result)
    if binding.get("primary_sha256") != sha256_file(primary):
        raise ValueError("terminal_hash")
    report = read_bound_sidecar(primary, Path(binding["sidecar_path"]))
    if report["report"].get("passed") is not True or value.get("flagged_adversarial") is not False:
        raise ValueError("unclean_terminal")
    return dict(
        passed=True,
        primary_path=str(primary),
        primary_sha256=sha256_file(primary),
        terminal_path=str(terminal),
        terminal_sha256=sha256_file(terminal),
        validator_path=binding["sidecar_path"],
        validator_sha256=sha256_file(Path(binding["sidecar_path"])),
    )


def consumer(task_id: str, science: bool) -> Json:
    """Fixture admission names its own class exception; scientific policy stays narrow."""
    pairs: list[tuple[str, str, Any]] = [
        ("scorer_fixture_ready_score", "==", 1),
        (
            "verdict_class",
            "in",
            ["positive", "null"] if science else ["positive", "null", "circular_positive"],
        ),
        ("flagged_adversarial", "==", False),
        ("terminal_hash_clean", "==", True),
    ]
    if science:
        pairs.append(("verifier_is_oracle", "==", False))
    return dict(
        gated_on=[dict(upstream=task_id, artifact_field=f, op=op, value=v) for f, op, v in pairs]
    )


def fixture_consumer(primary: Path, private: Path) -> Json:
    """Ask both actual consumers using the producer's authenticated fixture fields."""
    terminal = clean_terminal(primary)
    producer = json.loads(primary.read_text())
    value = {
        k: producer[k]
        for k in (
            "task_id",
            "honest_verdict",
            "verdict_class",
            "scorer_fixture_ready_score",
            "flagged_adversarial",
            "verifier_is_oracle",
        )
    }
    value["terminal_hash_clean"] = terminal["passed"]
    private.mkdir(parents=True, exist_ok=True)
    atomic_json(private / primary.name, value)
    fixture = evaluate_gates(consumer(value["task_id"], False), private)
    science = evaluate_gates(consumer(value["task_id"], True), private)
    return dict(
        fixture_passed=fixture.passed,
        science_passed=science.passed,
        terminal_receipt=terminal,
        fixture_gates=[asdict(r) for r in fixture.gates_evaluated],
        science_gates=[asdict(r) for r in science.gates_evaluated],
    )


def gate_matrix(private: Path) -> list[Json]:
    """Run every producer class and bad-input control through the real gate reader."""
    rows = []
    for verdict in CLASSES:
        for science in (False, True):
            for control in (
                "base",
                "ready_zero",
                "missing_field",
                "tampered_hash",
                "quarantined",
                "oracle",
            ):
                directory = private / f"{verdict}-{science}-{control}"
                directory.mkdir(parents=True, exist_ok=True)
                value = dict(
                    task_id="exp9000-control",
                    honest_verdict="complete_control",
                    verdict_class=verdict,
                    scorer_fixture_ready_score=1,
                    flagged_adversarial=False,
                    terminal_hash_clean=True,
                    verifier_is_oracle=False,
                )
                if control == "ready_zero":
                    value["scorer_fixture_ready_score"] = 0
                if control == "missing_field":
                    del value["scorer_fixture_ready_score"]
                if control == "tampered_hash":
                    value["terminal_hash_clean"] = False
                if control == "quarantined":
                    value["flagged_adversarial"] = True
                if control == "oracle":
                    value["verifier_is_oracle"] = True
                path = directory / "experiment_9000_control.json"
                atomic_json(path, value)
                result = evaluate_gates(consumer(value["task_id"], science), directory)
                expected = verdict in (
                    ["positive", "null"] if science else ["positive", "null", "circular_positive"]
                ) and control in (["base"] if science else ["base", "oracle"])
                rows.append(
                    dict(
                        unit_id=f"{verdict}-{science}-{control}",
                        source=str(path),
                        arm="science" if science else "fixture",
                        seed=69857,
                        producer_class=verdict,
                        control=control,
                        expected_passed=expected,
                        observed_passed=result.passed,
                        matched=result.passed == expected,
                        checks=[asdict(g) for g in result.gates_evaluated],
                        raw_numerator=int(result.passed == expected),
                        raw_denominator=1,
                        status="completed",
                        exclusion_reason=None,
                        independent_count=0,
                    )
                )
    return rows


def historical(root: Path, durable: Path) -> Json:
    """Retain actual skip bytes and prior authority instead of creating missing primaries."""
    prior = json.loads((root / "results/experiment_8044_v697_contract_methods.json").read_text())
    snapshots = prior["authority_snapshots"]
    for role, ref in snapshots.items():
        if ref["exists"]:
            path = Path(ref["snapshot_path"])
            if sha256_file(path) != ref["sha256"]:
                raise ValueError("prior_authority_hash")
            authority._snapshot(path, path.read_bytes(), durable, "previous_" + role)
    tasks = yaml.safe_load(Path(snapshots["active"]["snapshot_path"]).read_bytes())["tasks"]
    original = next(t for t in tasks if t["id"].startswith("exp8047-"))
    original_gate = evaluate_gates(original, root / "results")
    old_gate = dict(
        passed=original_gate.passed,
        summary=original_gate.summary,
        gates_evaluated=[
            asdict(g) for g in original_gate.gates_evaluated if g.upstream == scorer.TASK
        ],
    )
    skip = root / "results/experiment_8047_fit_score_capture.json"
    authority._snapshot(skip, skip.read_bytes(), durable, "actual_8047_skip")
    capstone_path = root / "results/experiment_8056_v697_capstone.json"
    capstone = json.loads(capstone_path.read_text())
    rows = []
    for n in range(8047, 8051):
        row = next(r for r in capstone["task_dispositions"] if r["task_id"].startswith(f"exp{n}-"))
        rows.append(
            dict(
                experiment_id=n,
                task_id=row["task_id"],
                evidence_kind="actual_skip_receipt" if n == 8047 else "log_only",
                path=str(skip) if n == 8047 else row["path"],
                sha256=sha256_file(skip) if n == 8047 else None,
                primary_present=n == 8047,
                measured=False,
                honest_verdict=json.loads(skip.read_text())["honest_verdict"]
                if n == 8047
                else None,
                disposition_source=str(capstone_path),
                disposition_source_sha256=sha256_file(capstone_path),
            )
        )
    return dict(rows=rows, original_gate=old_gate, previous_authority=snapshots)


def preconditions(root: Path) -> tuple[list[Json], list[Json]]:
    """Check named inputs and real terminal hashes before using fixture readiness."""
    failures, refs = [], []
    for name in INPUTS + [
        ".venv/bin/" + n for n in ("python", "pytest", "coverage", "ruff", "mypy")
    ]:
        path = root / name
        if not path.is_file():
            failures.append(failure(path, "resource_exists", True, "missing_resource"))
        else:
            refs.append(dict(path=str(path), sha256=sha256_file(path)))
    for name in ("experiment_8045_v697_scorer_workspace", "experiment_8056_v697_capstone"):
        path = root / "results" / (name + ".json")
        if path.is_file():
            try:
                receipt = clean_terminal(path)
                refs += [
                    dict(path=receipt[k], sha256=receipt[k.replace("path", "sha256")])
                    for k in ("terminal_path", "validator_path")
                ]
            except (ValueError, KeyError, OSError) as exc:
                failures.append(failure(path, "clean_terminal", True, str(exc)))
    return failures, refs


def measure(
    root: Path, design: Path, staged: Path, active: Path, durable: Path, private: Path, mutate: bool
) -> Json:
    """Freeze dependencies, then observe authority, gates and known scorer probabilities."""
    progress("preconditions")
    failures, refs = preconditions(root)
    for path in (design, active):
        if not path.is_file():
            failures.append(failure(path, "resource_exists", True, "missing_resource"))
    value: Json = dict(
        failures=failures,
        refs=refs,
        contract={},
        gate_matrix_rows=[],
        fixture_rows=[],
        historical={},
        scorer_code_hashes=[],
        workspace={},
    )
    if failures:
        return value
    for ref in refs:
        path = Path(ref["path"])
        if not path.is_relative_to(root / ".venv"):
            frozen = authority._snapshot(
                path, path.read_bytes(), durable / "inputs", path.name.replace(".", "_")
            )
            ref.update(snapshot_path=frozen["snapshot_path"])
    progress("authority")
    value["contract"] = assess(design, staged, active, durable / "authority_snapshots")
    value["failures"] += value["contract"]["gate_check_summary"]
    value["historical"] = historical(root, durable / "historical")
    producer = root / "results/experiment_8045_v697_scorer_workspace.json"
    upstream = json.loads(producer.read_text())
    for ref in upstream["scorer_code_hashes"]:
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            value["failures"].append(
                failure(
                    Path(ref["path"]),
                    "scorer_dependency_hash",
                    ref["sha256"],
                    sha256_file(Path(ref["path"])),
                )
            )
    if upstream.get("scorer_fixture_ready_score") != 1:
        value["failures"].append(
            failure(
                producer,
                "scorer_fixture_ready_score",
                1,
                upstream.get("scorer_fixture_ready_score"),
            )
        )
    value["scorer_code_hashes"] = upstream["scorer_code_hashes"]
    if not value["failures"]:
        value["fixture_consumer_results"] = fixture_consumer(
            producer, private / "real_fixture_consumer"
        )
    progress("consumer_matrix")
    value["gate_matrix_rows"] = gate_matrix(private / "consumers")
    for row in value["gate_matrix_rows"]:
        path = Path(row["source"])
        snapshot = authority._snapshot(
            path, path.read_bytes(), durable / "gate_inputs", row["unit_id"]
        )
        row["source_snapshot"] = snapshot["snapshot_path"]
        row["source_sha256"] = snapshot["sha256"]
        refs.append(dict(path=snapshot["snapshot_path"], sha256=snapshot["sha256"]))
    progress("before_workspace_subprocess")
    value["workspace"] = scorer.reproduce(private / "workspace")
    for role, receipt in value["workspace"].items():
        source = ROOT / receipt["log_path"]
        target = durable / "workspace_logs" / (role + ".log")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        receipt["log_path"] = str(target)
    capstone = json.loads((root / "results/experiment_8056_v697_capstone.json").read_text())
    value["repository_health"] = capstone["repository_health"]
    for receipt in value["repository_health"]:
        source = Path(receipt["log_path"])
        if source.is_file():
            refs.append(dict(path=str(source), sha256=sha256_file(source)))
    progress("after_workspace_subprocess")
    progress("before_scorer_fixture")
    value["fixture_rows"] = scorer.fixture_rows()
    if mutate:
        value["fixture_rows"][0]["conditional_logit_positions"][0] += 1
    progress("after_scorer_fixture", 4)
    return value


def qualify_fixture(rows: list[Json], workspace: Json) -> bool:
    """Require exact alignment and normalized probabilities plus the unchanged drift bound."""
    if not scorer.reduce_rows(rows)["passed"]:
        return False
    means = [r["numerator"] / r["denominator"] for r in rows]
    return (
        workspace.get("before", {}).get("actual_exit_code") == 1
        and workspace.get("before", {}).get("missing_parent_detected") is True
        and workspace.get("after", {}).get("actual_exit_code") == 0
        and workspace.get("after", {}).get("parent_exists_before_launch") is True
        and abs(means[0] - means[2]) <= 1e-6
        and abs(means[1] - means[3]) <= 1e-6
    )


def build(work: Json, raw: Path, started_ns: int, ended_ns: int, receipts: list[Json]) -> Json:
    """Reduce readiness from raw operands; known controls supply no independent science."""
    rows = work["contract"].get("contract_rows", [])
    for row in rows:
        row.update(
            source="active_authority",
            exclusion_reason=None if row["matched"] else "contract_mismatch",
        )
    fixture_ok = qualify_fixture(work["fixture_rows"], work["workspace"])
    matrix_ok = bool(work["gate_matrix_rows"]) and all(
        r["matched"] for r in work["gate_matrix_rows"]
    )
    checks_ok = bool(receipts) and all(r["passed"] for r in receipts)
    ready = (
        not work["failures"]
        and fixture_ok
        and matrix_ok
        and checks_ok
        and work["contract"].get("activated", False)
        and work.get("fixture_consumer_results", {}).get("fixture_passed", False)
    )
    blocked = any(r["observed"] == "missing_resource" for r in work["failures"])
    classification = "blocked" if blocked else "circular_positive" if ready else "disqualified"
    verdict = (
        "complete_blocked_" + Path(work["failures"][0]["path"]).name.replace(".", "_")
        if blocked
        else "complete_fixture_consumer_contract"
        if ready
        else "complete_disqualified_fixture_consumer_contract"
    )
    receipt = build_current_work_receipt(
        run_id=str(started_ns),
        owner_pid=work["owner_pid"],
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details={"mode": "no_model_load"},
        inference_substrate_class="aggregation",
        execution_venue="host",
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=ended_ns,
    )
    hashes = [
        dict(path=str(raw / name), sha256=sha256_file(raw / name))
        for name in ("work.json", "validation_commands.json")
    ]
    value = dict(
        experiment_id=8057,
        experiment=8057,
        title="Qualify fixture consumers and bind the complete V698 contract",
        status="complete",
        started_at=datetime.fromtimestamp(work["started_unix"], UTC).isoformat(),
        finished_at=datetime.fromtimestamp(
            work["started_unix"] + receipt["duration_s"], UTC
        ).isoformat(),
        task_id=TASK,
        milestone=MILESTONE,
        schema="carnot.experiment.v1",
        run_date="20261003",
        honest_verdict=verdict,
        verdict_class=classification,
        verifier_is_oracle=True,
        claim_scope="exposed fixture consumers and administrative authority only; no source usefulness or current model repeatability",
        flagged_adversarial=False,
        required_checks_passed=checks_ok,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=receipt["invocation_counts"],
        current_work_receipt=receipt,
        duration_s=receipt["duration_s"],
        rows=rows,
        intended_count=13,
        eligible_count=len(rows),
        independent_count=0,
        completed_count=len(rows),
        censored_count=0,
        excluded_count=sum(not r["matched"] for r in rows),
        failed_count=sum(not r["matched"] for r in rows),
        sample_size_budget=dict(
            administrative_tasks=13,
            gate_controls=72,
            scorer_calls=4,
            independent_scientific_units=0,
        ),
        gate_check_summary=work["failures"]
        + [
            failure(Path(r["log_path"]), r["name"], 0, r.get("exit_code"))
            for r in receipts
            if not r["passed"]
        ],
        random_seed=69857,
        reproducibility_checksum=canonical_hash(work),
        source_artifact_hashes=work["refs"],
        raw_shard_hashes=hashes,
        code_config_hashes=work["code_config_hashes"],
        phase_spans=[dict(phase="fixture_and_authority", start_s=0, end_s=receipt["duration_s"])],
        generalized_learning_benefit_score=0,
        fixture_consumer_ready_score=int(ready),
        contract_ready_score=int(checks_ok and work["contract"].get("activated", False)),
        canonical_tasks_sha256=work["contract"].get("canonical_tasks_sha256"),
        authority_snapshots=work["contract"].get("authority_snapshots", {}),
        gate_matrix_rows=work["gate_matrix_rows"],
        scorer_code_hashes=work["scorer_code_hashes"],
        historical_disposition_rows=work["historical"].get("rows", []),
        original_gate_failure=work["historical"].get("original_gate", {}),
        token_alignment_control_rows=work["fixture_rows"],
        substrate_declaration=dict(
            substrate="aggregation_from_upstream_artifacts", mode="no_model_load", MODEL_SPECS=[]
        ),
        methods=scorer.METHODS,
        methodology_note="Known fixture probabilities qualify new measurement machinery only; seeds and controls add zero independent scientific observations.",
        genuine_headroom="consumer class mismatch and authentication controls",
        positive_control_results=dict(fixture=fixture_ok, matrix=matrix_ok),
        acceptance_gate_results=dict(checks=checks_ok, fixture=fixture_ok, matrix=matrix_ok),
        repository_health=work.get("repository_health", []),
        fixture_consumer_results=work.get("fixture_consumer_results", {}),
        trained_head_specs=[],
    )
    value["coverage_statement_counts"] = work.get("coverage_statement_counts", {})
    value["checkpoint_hashes"] = []
    value["model_hashes"] = []
    value["staging_custody_status"] = (
        "matching_observed"
        if work["contract"].get("authority_snapshots", {}).get("staged", {}).get("exists")
        else "unknown_consumed"
    )
    value["field_principles"] = {
        k: f"Preserve {k} as an attributable operand; prevent fixture mechanics from becoming scientific credit."
        for k in value
    }
    value["field_principles"]["fixture_consumer_ready_score"] = (
        "Allow new measurements only; prevent fixture success from claiming source usefulness or current model repeatability."
    )
    value["field_principles"]["contract_ready_score"] = (
        "Bind complete ordered prompts and metadata; prevent administrative readiness from depending on scientific benefit."
    )
    value["field_principles"]["independent_count"] = (
        "Count zero independent scientific observations; repeated calls, seeds and oracle controls do not add scientific support."
    )
    return value


def replay(path: Path) -> bool:
    """Cold-reduce stored observations and reject changed claims, inputs or validation logs."""
    try:
        value = json.loads(path.read_text())
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        work = json.loads((raw / "work.json").read_text())
        for ref in (
            value["source_artifact_hashes"]
            + value["raw_shard_hashes"]
            + value["scorer_code_hashes"]
        ):
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        for ref in value["source_artifact_hashes"]:
            if (
                ref.get("snapshot_path")
                and sha256_file(Path(ref["snapshot_path"])) != ref["sha256"]
            ):
                return False
        for receipt in value["validation_receipts"]:
            if sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]:
                return False
        for name, digest in value["code_config_hashes"].items():
            if sha256_file(ROOT / name) != digest:
                return False
        times = value["current_work_receipt"]
        rebuilt = build(
            work,
            raw,
            times["started_monotonic_ns"],
            times["ended_monotonic_ns"],
            value["validation_receipts"],
        )
        return rebuilt == value
    except (OSError, KeyError, ValueError, TypeError):
        return False


def manifest(private: Path) -> list[Json]:
    """Freeze bounded checks before observations; private coverage includes only added files."""
    from carnot.reporting.v686_contract_validation import CONSUMERS

    py, cov, pytest, ruff, mypy = [
        str(ROOT / ".venv/bin" / name) for name in ("python", "coverage", "pytest", "ruff", "mypy")
    ]
    config = private / "coverage.ini"
    includes = [MODULE, CLI]
    config.write_text(
        "[run]\nparallel = true\ndata_file = "
        + str(private / ".coverage")
        + "\ninclude =\n"
        + "".join(
            "    " + str(ROOT / p) + "\n"
            for p in [*includes, "python/carnot/reporting/v685_authority_lifecycle.py"]
        )
    )
    include_arg = "--include=" + ",".join(str(ROOT / p) for p in includes)
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    tests = [TEST, scorer.TEST, "tests/python/test_experiment_7891_v685_authority_lifecycle.py"]
    commands = [
        (
            "focused_unit_scorer_E2E-018",
            [cov, "run", "--rcfile=" + str(config), "-m", "pytest", *common, *tests],
            180,
        ),
        ("consumer_tests", [pytest, *common, *CONSUMERS], 180),
        ("coverage_combine", [cov, "combine", "--rcfile=" + str(config)], 30),
        (
            "coverage_report",
            [
                cov,
                "report",
                "--rcfile=" + str(config),
                include_arg,
                "--show-missing",
                "--fail-under=100",
            ],
            30,
        ),
        (
            "coverage_json",
            [
                cov,
                "json",
                "--rcfile=" + str(config),
                include_arg,
                "-o",
                str(private / "coverage.json"),
            ],
            30,
        ),
        (
            "ruff_check",
            [ruff, "check", *includes, "python/carnot/reporting/v685_authority_lifecycle.py", TEST],
            30,
        ),
        (
            "ruff_format",
            [
                ruff,
                "format",
                "--check",
                *includes,
                "python/carnot/reporting/v685_authority_lifecycle.py",
                TEST,
            ],
            30,
        ),
        (
            "strict_mypy",
            [
                mypy,
                "--strict",
                "--follow-imports=silent",
                *includes,
                "python/carnot/reporting/v685_authority_lifecycle.py",
            ],
            60,
        ),
        ("scoped_spec_coverage", [py, "scripts/check_spec_coverage.py", TEST], 30),
    ]
    return [
        dict(
            name=n,
            argv=a,
            deadline_s=t,
            expected_exit=0,
            classification="required",
            coverage_config=str(config),
        )
        for n, a, t in commands
    ]


def validate(private: Path, raw: Path, specs: list[Json]) -> list[Json]:
    """Use the existing supervisor so commands have real exits, heartbeats and durable logs."""
    from carnot.reporting.v686_contract_validation import run_check

    os.environ["CARNOT_8057_COVERAGE_CONFIG"] = specs[0]["coverage_config"]
    receipts = []
    for index, spec in enumerate(specs):
        progress("before_" + spec["name"], index, len(specs) - index)
        receipts.append(run_check(ROOT, spec, private, raw / "validation_logs", heartbeat_s=20))
        progress("after_" + spec["name"], index + 1, len(specs) - index - 1)
    del os.environ["CARNOT_8057_COVERAGE_CONFIG"]
    return receipts


def publish(value: Json, output: Path, private: Path, raw: Path) -> None:
    """Expose a primary only after normal cold-replay and strict validator child exits."""
    from carnot.reporting.v686_contract_validation import run_check

    def validator(candidate: Path) -> Json:
        specs = [
            (
                "cold_replay",
                [str(ROOT / ".venv/bin/python"), str(ROOT / CLI), "--cold-replay", str(candidate)],
            ),
            (
                "adversarial",
                [
                    str(ROOT / ".venv/bin/python"),
                    str(ROOT / "scripts/adversarial_verify.py"),
                    "--json",
                    str(candidate),
                ],
            ),
            (
                "strict_rows",
                [
                    str(ROOT / ".venv/bin/python"),
                    str(ROOT / "scripts/verdict_row_consistency_lint.py"),
                    "--strict",
                    str(candidate),
                ],
            ),
        ]
        checks = []
        for name, argv in specs:
            progress("before_" + name)
            checks.append(
                run_check(
                    ROOT,
                    dict(name=name, argv=argv, expected_exit=0, deadline_s=60),
                    private,
                    raw / "terminal_logs",
                    heartbeat_s=20,
                )
            )
            progress("after_" + name)
        return dict(passed=all(r["passed"] for r in checks), checks=checks)

    publication = publish_primary(output, value, validator)
    readers = reader_receipt(
        TASK,
        output.parent,
        field="fixture_consumer_ready_score",
        expected=value["fixture_consumer_ready_score"],
    )
    atomic_json(
        raw / "terminal_validation.json",
        dict(publication=publication, normal_process_exit=True, readers=readers),
    )


def main(argv: list[str] | None = None) -> int:
    """Run current checks or replay existing evidence; never activate a roadmap."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261003"], default="20261003")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--design", type=Path, default=ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md"
    )
    parser.add_argument("--staged", type=Path, default=ROOT / "research-roadmap-next.yaml")
    parser.add_argument("--active", type=Path, default=ROOT / "research-roadmap.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--mutate", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        progress("replay_passed" if passed else "reduction_drift")
        return 0 if passed else 1
    started = time.monotonic_ns()
    output = (args.fixture_output or args.output).absolute()
    raw = output.parent / "raw" / output.stem / "invocations" / str(started)
    raw.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="carnot-8057-") as directory:
        private = Path(directory)
        from carnot.reporting.v686_contract_validation import dependency_hashes

        code_hashes = dependency_hashes(
            ROOT, paths=[MODULE, CLI, TEST, "python/carnot/reporting/v685_authority_lifecycle.py"]
        )
        specs = manifest(private)
        candidate = output.parent / "raw" / output.stem / "terminal_candidate.json"
        terminal_commands = [
            dict(
                name="cold_replay",
                argv=[
                    str(ROOT / ".venv/bin/python"),
                    str(ROOT / CLI),
                    "--cold-replay",
                    str(candidate),
                ],
            ),
            dict(
                name="adversarial",
                argv=[
                    str(ROOT / ".venv/bin/python"),
                    str(ROOT / "scripts/adversarial_verify.py"),
                    "--json",
                    str(candidate),
                ],
            ),
            dict(
                name="strict_rows",
                argv=[
                    str(ROOT / ".venv/bin/python"),
                    str(ROOT / "scripts/verdict_row_consistency_lint.py"),
                    "--strict",
                    str(candidate),
                ],
            ),
        ]
        atomic_json(
            raw / "validation_commands.json",
            dict(
                commands=specs,
                dependency_hashes=code_hashes,
                terminal_commands=terminal_commands,
                terminal_expected_exit=0,
                terminal_deadline_s=60,
            ),
        )
        work = measure(args.root, args.design, args.staged, args.active, raw, private, args.mutate)
        work["code_config_hashes"] = code_hashes
        work["owner_pid"] = os.getpid()
        work["started_unix"] = time.time() - (time.monotonic_ns() - started) / 1e9
        receipts = []
        if args.fixture_output:
            log = raw / "fixture_exit.log"
            log.write_text(
                "Private fixture route completed; production checks require the frozen manifest.\n"
            )
            receipts = [
                dict(
                    name="private_fixture",
                    passed=True,
                    log_path=str(log),
                    log_sha256=sha256_file(log),
                )
            ]
        elif not work["failures"]:
            receipts = validate(private, raw, specs)
        report = private / "coverage.json"
        if report.is_file():
            target = raw / "coverage.json"
            shutil.copyfile(report, target)
            work["coverage_statement_counts"] = json.loads(report.read_text())["totals"]
            work["refs"].append(dict(path=str(target), sha256=sha256_file(target)))
        atomic_json(raw / "work.json", work)
        value = build(work, raw, started, time.monotonic_ns(), receipts)
        if args.fixture_output:
            atomic_json(output, value)
        else:
            publish(value, output, private, raw)
    progress("complete", len(value["rows"]))
    return 0
