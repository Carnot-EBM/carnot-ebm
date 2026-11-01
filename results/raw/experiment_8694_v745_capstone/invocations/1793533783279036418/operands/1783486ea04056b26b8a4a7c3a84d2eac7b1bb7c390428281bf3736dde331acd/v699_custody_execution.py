"""REQ-REPORT-8070: bounded validation and cold reconstruction protect publication."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import v699_contract_custody as e
from carnot.reporting.current_work_receipt import atomic_json, build_current_work_receipt
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v686_contract_validation import (
    CONSUMERS,
    coverage_complete,
    dependency_hashes,
    run_check,
)

Json = dict[str, Any]
START = time.monotonic()
OWNED = [e.MODULE, e.RUNNER, e.CLI]
SCOPED = [*OWNED, "python/carnot/reporting/v685_authority_lifecycle.py"]
INPUTS = [
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "ops/e2e-test-plan.md",
    "openspec/capabilities/research-reporting/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/primary_publication.py",
    "ops/exclusion_manifest.yaml",
    e.DESIGN,
    "python/carnot/reporting/roadmap_contract.py",
    "python/carnot/reporting/v685_authority_lifecycle.py",
    "python/carnot/reporting/v698_fixture_consumer_contract.py",
    "scripts/conductor_gates.py",
    "openspec/change-proposals/research-roadmap-v698-preserved-20261003.md",
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Show real phase boundaries so quiet code cannot imply active inference."""
    print(
        f"[exp8070] phase={phase} elapsed_s={time.monotonic() - START:.3f} completed={completed} pending={pending}",
        flush=True,
    )


def measure(root: Path, design: Path, staged: Path, active: Path, raw: Path) -> Json:
    """Freeze inputs before reading them; independent branches survive external errors."""
    binder = e.Binder(raw / "inputs")
    progress("preconditions")
    for name in INPUTS:
        try:
            binder.bind(root / name)
        except e.InputFailure:
            pass
    for tool in ("python", "pytest", "coverage", "ruff", "mypy"):
        try:
            binder.bind(e.ROOT / ".venv/bin" / tool)
        except e.InputFailure:
            pass
    progress("authority")
    contract = e.assess(design, staged, active, raw / "authority")
    progress("historical_inputs")
    inputs = e.qualify(root, raw / "qualified_inputs")
    try:
        history = e.historical(root, raw / "historical")
    except (e.InputFailure, ValueError, OSError, KeyError) as error:
        history = dict(
            rows=[],
            refs=[],
            failures=[
                e.failure(
                    root / "results/experiment_8069_v698_capstone.json",
                    "historical_authentication",
                    True,
                    str(error),
                )
            ],
        )
    progress("gate_matrix")
    matrix = e.gate_matrix(raw / "private_gate_inputs")
    return dict(
        contract=contract,
        inputs=inputs,
        history=history,
        matrix=matrix,
        refs=binder.refs + inputs["refs"] + history["refs"],
        failures=binder.failures
        + contract["gate_check_summary"]
        + inputs["failures"]
        + history["failures"],
        environment=dict(
            python=os.sys.version,
            executable=os.sys.executable,
            JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS"),
        ),
        named_input_resolution=dict(
            requested="python/carnot/experiment_8057_v698_fixture_consumer_contract.py",
            exists=False,
            shipped_helper="python/carnot/reporting/v698_fixture_consumer_contract.py",
        ),
    )


def manifest(private: Path) -> list[Json]:
    """Freeze scoped coverage and actual commands before any measurement is opened."""
    paths = {
        n: str(e.ROOT / ".venv/bin" / n) for n in ["python", "pytest", "coverage", "ruff", "mypy"]
    }
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel = true\ndata_file = "
        + str(private / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in OWNED)
    )
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    include = "--include=" + ",".join(str(e.ROOT / p) for p in OWNED)
    commands = [
        (
            "focused_E2E018",
            [
                paths["coverage"],
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                *common,
                e.TEST,
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            ],
            180,
        ),
        ("consumers", [paths["pytest"], *common, *CONSUMERS], 120),
        ("coverage_combine", [paths["coverage"], "combine", "--rcfile=" + str(config)], 30),
        (
            "coverage_report",
            [
                paths["coverage"],
                "report",
                "--rcfile=" + str(config),
                include,
                "--show-missing",
                "--fail-under=100",
            ],
            30,
        ),
        (
            "coverage_json",
            [
                paths["coverage"],
                "json",
                "--rcfile=" + str(config),
                include,
                "-o",
                str(private / "coverage.json"),
            ],
            30,
        ),
        ("ruff_check", [paths["ruff"], "check", *SCOPED, e.TEST], 30),
        ("ruff_format", [paths["ruff"], "format", "--check", *SCOPED, e.TEST], 30),
        ("strict_mypy", [paths["mypy"], "--strict", "--follow-imports=silent", *SCOPED], 60),
        ("scoped_spec", [paths["python"], "scripts/check_spec_coverage.py", e.TEST], 30),
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


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Recompute readiness independently; fixture success creates no science credit."""
    checks = bool(receipts) and all(r["passed"] for r in receipts)
    matrix_ok = all(r["matched"] for r in work["matrix"])
    owned_ok = checks and matrix_ok
    contract = work["contract"]
    ready = int(owned_ok and contract.get("activated", False))
    source = int(owned_ok and work["inputs"]["source_ready"])
    learning = int(owned_ok and work["inputs"]["learning_ready"])
    classification = (
        "disqualified" if not owned_ok else "blocked" if work["failures"] else "circular_positive"
    )
    verdict = (
        "complete_contract_custody"
        if classification == "circular_positive"
        else "complete_"
        + classification
        + "_"
        + (
            Path(work["failures"][0]["path"]).name.replace(".", "_")
            if work["failures"]
            else "owned_checks"
        )
    )
    current = build_current_work_receipt(
        run_id=str(work["start_ns"]),
        owner_pid=work["pid"],
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details={"mode": "no_model_load"},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=work["start_ns"],
        ended_monotonic_ns=work["end_ns"],
    )
    rows = []
    for r in contract.get("contract_rows", []):
        rows.append(
            dict(
                r,
                unit=r["unit_id"],
                source="V699_active_authority",
                condition="full_contract",
                numerator=r["raw_numerator"],
                denominator=r["raw_denominator"],
                exclusion_reason=None if r["matched"] else "authority_mismatch",
            )
        )
    value: Json = dict(
        experiment_id=8070,
        experiment=8070,
        task_id=e.TASK,
        milestone=e.MILESTONE,
        schema="carnot.experiment.v1",
        run_date="20261003",
        status="complete",
        title="V699 contract custody",
        honest_verdict=verdict,
        verdict_class=classification,
        verifier_is_oracle=True,
        claim_scope="administrative authority and exposed historical input qualification only; no independent scientific benefit",
        flagged_adversarial=False,
        required_checks_passed=owned_ok,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=current["invocation_counts"],
        current_work_receipt=current,
        duration_s=current["duration_s"],
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
            historical_tasks=13,
            gate_controls=72,
            independent_scientific_units=0,
        ),
        gate_check_summary=work["failures"]
        + [
            e.failure(
                Path(r.get("log_path", raw / "work.json")),
                r["name"],
                r.get("expected_exit", True),
                r.get("exit_code", False),
            )
            for r in receipts
            if not r["passed"]
        ],
        random_seed=69970,
        reproducibility_checksum=e.canonical_hash(work),
        source_artifact_hashes=work["refs"],
        raw_shard_hashes=[
            dict(path=str(raw / n), sha256=e.sha256_file(raw / n))
            for n in ["work.json", "validation_commands.json"]
        ],
        code_config_hashes=work["code_hashes"],
        checkpoint_hashes=[],
        phase_spans=work["phase_spans"],
        generalized_learning_benefit_score=0,
        contract_ready_score=ready,
        cached_source_inputs_ready_score=source,
        historical_learning_inputs_ready_score=learning,
        canonical_tasks_sha256=contract.get("canonical_tasks_sha256"),
        authority_snapshots=contract.get("authority_snapshots", {}),
        staging_custody_status="matching_observed"
        if contract.get("planning_matched")
        else "absent_consumed_or_other_milestone",
        gate_matrix_rows=work["matrix"],
        historical_disposition_rows=work["history"]["rows"],
        preserved_v698_authority=work["history"].get("authority_snapshots", {}),
        authenticated_input_manifest=work["refs"],
        qualified_feature_rows=work["inputs"]["feature_rows"],
        qualified_initial_head=work["inputs"]["qualified_head"],
        qualified_initial_head_sha256=work["inputs"]["head_sha256"],
        historical_model_receipts=work["inputs"]["historical_model_receipts"],
        trained_head_specs=[
            dict(
                scope="historical_initial_head_only",
                current_training_operations=0,
                sha256=work["inputs"]["head_sha256"],
            )
        ],
        substrate_declaration="aggregation_from_upstream_artifacts",
        environment=work["environment"],
        named_input_resolution=work["named_input_resolution"],
        repository_health=work.get("repository_health", []),
        methodology_note="Authenticate historical bytes, reconstruct administrative tasks and test real consumer class policies; repeated fixtures supply zero independent scientific units.",
        positive_control_results=dict(gate_matrix=matrix_ok),
        acceptance_gate_results=dict(owned_checks=owned_ok),
        coverage_statement_counts=work.get("coverage_statement_counts", {}),
        fixture_validation_scope=work["fixture"],
    )
    value["field_principles"] = {
        k: f"Retain the exact {k} operand so custody cannot be confused with scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        cached_source_inputs_ready_score="Source features qualify independently; a learning failure cannot erase them.",
        historical_learning_inputs_ready_score="Learning inputs qualify independently; failed source likelihoods cannot erase them.",
        historical_model_receipts="Original model/build/run hashes stay upstream; they are never current model operations.",
        historical_disposition_rows="Nulls, skips, absent primaries and censored cache work remain distinct completed outcomes.",
    )
    return value


def replay(path: Path) -> bool:
    """Rebuild from primitives and saved bytes; changing any reduction or log fails."""
    try:
        value = json.loads(path.read_text())
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        work = json.loads((raw / "work.json").read_text())
        for ref in value["source_artifact_hashes"] + value["raw_shard_hashes"]:
            if e.sha256_file(Path(ref["path"])) != ref["sha256"] or (
                ref.get("snapshot_path")
                and e.sha256_file(Path(ref["snapshot_path"])) != ref["sha256"]
            ):
                return False
        for ref in value["authority_snapshots"].values():
            if ref["exists"] and e.sha256_file(Path(ref["snapshot_path"])) != ref["sha256"]:
                return False
        for r in value["validation_receipts"]:
            if r.get("log_path") and e.sha256_file(Path(r["log_path"])) != r["log_sha256"]:
                return False
        for name, digest in value["code_config_hashes"].items():
            if e.sha256_file(e.ROOT / name) != digest:
                return False
        return build(work, raw, value["validation_receipts"]) == value
    except (OSError, ValueError, KeyError, TypeError):
        return False


def publish(value: Json, output: Path, private: Path, raw: Path) -> None:
    """Publish only after real cold-replay and validator children exit normally."""

    def validator(candidate: Path) -> Json:
        commands = json.loads((raw / "validation_commands.json").read_text())["terminal_commands"]
        checks = []
        for spec in commands:
            progress("before_" + spec["name"])
            checks.append(run_check(e.ROOT, spec, private, raw / "terminal_logs", heartbeat_s=20))
            progress("after_" + spec["name"])
        return dict(passed=all(r["passed"] for r in checks), checks=checks)

    publication = publish_primary(output, value, validator)
    atomic_json(
        raw / "terminal_validation.json",
        dict(
            publication=publication,
            normal_process_exit=True,
            readers=reader_receipt(
                e.TASK,
                output.parent,
                field="contract_ready_score",
                expected=value["contract_ready_score"],
            ),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Run owned checks or replay existing evidence without editing any roadmap."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261003"], default="20261003")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--design", type=Path, default=e.ROOT / e.DESIGN)
    parser.add_argument("--staged", type=Path, default=e.ROOT / "research-roadmap-next.yaml")
    parser.add_argument("--active", type=Path, default=e.ROOT / "research-roadmap.yaml")
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--mutate", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = replay(args.cold_replay)
        progress("replay_passed" if passed else "reduction_drift")
        return 0 if passed else 1
    start = time.monotonic_ns()
    output = (args.fixture_output or args.output).absolute()
    raw = output.parent / "raw" / output.stem / "invocations" / str(start)
    raw.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="carnot-8070-") as directory:
        private = Path(directory)
        specs = manifest(private)
        atomic_json(raw / "validation_commands.json", dict(commands=specs))
        candidate = output.parent / "raw" / output.stem / "terminal_candidate.json"
        terminal_specs = [
            dict(
                name=name,
                argv=[str(e.ROOT / ".venv/bin/python"), *args],
                deadline_s=60,
                expected_exit=0,
            )
            for name, args in [
                ("cold_replay", [str(e.ROOT / e.CLI), "--cold-replay", str(candidate)]),
                (
                    "adversarial",
                    [str(e.ROOT / "scripts/adversarial_verify.py"), "--json", str(candidate)],
                ),
                (
                    "strict_rows",
                    [
                        str(e.ROOT / "scripts/verdict_row_consistency_lint.py"),
                        "--strict",
                        str(candidate),
                    ],
                ),
            ]
        ]
        atomic_json(
            raw / "validation_commands.json", dict(commands=specs, terminal_commands=terminal_specs)
        )
        work = measure(args.root, args.design, args.staged, args.active, raw)
        work.update(
            start_ns=start,
            pid=os.getpid(),
            fixture=bool(args.fixture_output),
            code_hashes=dependency_hashes(e.ROOT, paths=[*OWNED, e.TEST]),
        )
        if args.mutate:
            work["matrix"][0]["matched"] = False
        if args.fixture_output:
            receipts = [
                dict(
                    name="in_process_fixture_controls",
                    passed=True,
                    scope="private_oracle_only",
                    normal_process_exit=None,
                    completed_units=len(work["matrix"]),
                )
            ]
        else:
            receipts = []
            os.environ["CARNOT_8070_COVERAGE_CONFIG"] = specs[0]["coverage_config"]
            for index, spec in enumerate(specs):
                progress("before_" + spec["name"], index, len(specs) - index)
                receipts.append(
                    run_check(e.ROOT, spec, private, raw / "validation_logs", heartbeat_s=20)
                )
                progress("after_" + spec["name"], index + 1, len(specs) - index - 1)
            del os.environ["CARNOT_8070_COVERAGE_CONFIG"]
            proof = private / "coverage.json"
            complete = coverage_complete(proof, includes=OWNED)
            receipts.append(
                dict(name="coverage_statement_counts", passed=complete, scope="added_files_only")
            )
            if proof.is_file():
                counts = json.loads(proof.read_text())
                atomic_json(raw / "coverage.json", counts)
                work["coverage_statement_counts"] = {
                    p: counts["files"].get(p, {}).get("summary", {}) for p in OWNED
                }
        health = (
            e.ROOT
            / "results/raw/experiment_8070_v699_contract_custody/repository_health/receipt.json"
        )
        if health.is_file():
            report = json.loads(health.read_text())
            work["repository_health"] = [report]
            custody = e.Binder(raw / "repository_health")
            custody.bind(health)
            custody.bind(Path(report["log_path"]), report["log_sha256"])
            work["refs"] += custody.refs
        work["end_ns"] = time.monotonic_ns()
        work["phase_spans"] = [
            dict(phase="custody_and_owned_checks", start_s=0, end_s=(work["end_ns"] - start) / 1e9)
        ]
        atomic_json(raw / "work.json", work)
        value = build(work, raw, receipts)
        progress("before_publication")
        publish(value, output, private, raw)
        progress("after_publication", 13)
    return 0
