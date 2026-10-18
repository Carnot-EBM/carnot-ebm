"""Freeze checks and publish a replayable producer-bound delta. REQ-REPORT-8215.

Validation uses the qualified process-group supervisor and primary publisher.
Private fixture results are kept separate from the live outcome denominator.
"""

from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import time
from typing import Any

from carnot.reporting import arc_authoritative_frontier_8215 as reader
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v686_contract_validation import coverage_complete, dependency_hashes
from carnot.reporting.v709_execution import child, progress
from scripts.experiment_template import normalize_artifact_for_template_write

ROOT = reader.ROOT
CLI = "scripts/experiments/experiment_8215_v709_arc_authoritative_frontier.py"
OUTPUT = ROOT / "results/experiment_8215_v709_arc_authoritative_frontier.json"
TEST = "tests/python/test_arc_authoritative_frontier_8215.py"
OWNED = [
    "python/carnot/reporting/arc_authoritative_frontier_8215.py",
    "python/carnot/reporting/arc_authoritative_execution_8215.py",
    CLI,
]
CONSUMERS = [
    "tests/python/test_arc_supervisor_frontier_8189.py",
    "tests/python/test_arc_supervisor_frontier_8202.py",
    "tests/python/test_arc_supervisor_refinement.py",
    "tests/python/test_primary_publication_7928.py",
]
NAMED_INPUTS = [
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "ops/e2e-test-plan.md",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/verification/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/primary_publication.py",
    "ops/exclusion_manifest.yaml",
    "openspec/change-proposals/research-roadmap-vNEXT.md",
    "python/carnot/reporting/arc_supervisor_v708_frontier.py",
    "python/carnot/reporting/arc_supervisor_v707_frontier.py",
    "scripts/arc_bench.py",
    "scripts/arc_supervisor_refine.py",
    "python/carnot/agentic/arc_competition_agent.py",
    "ops/arc_solve_registry.yaml",
    "ops/arc_bench_latest.json",
    "results/experiment_8189_v707_arc_supervisor_frontier.json",
    "results/experiment_8202_v708_arc_supervisor_frontier.json",
]


def preconditions(private: Path) -> dict[str, Any]:
    """Authenticate branch prerequisites while preserving optional sibling dispositions."""
    checks = []
    hashes = {}
    for label in NAMED_INPUTS:
        path = ROOT / label
        required = label not in {
            "ops/arc_bench_latest.json",
            "results/experiment_8202_v708_arc_supervisor_frontier.json",
        }
        digest = sha256_file(path) if path.is_file() else None
        row = dict(
            reader.operand(path, "is_file", True, path.is_file(), digest),
            required=required,
            passed=path.is_file() or not required,
            disposition="present"
            if path.is_file()
            else "absent_not_required"
            if not required
            else "missing_required",
        )
        checks.append(row)
        if digest:
            hashes[str(path)] = digest
    probe = private / "writable_precondition"
    probe.write_text("private writable storage")
    checks.append(
        dict(
            reader.operand(
                private,
                "private_writable_storage",
                True,
                probe.read_text() == "private writable storage"
                and not private.resolve().is_relative_to(ROOT.resolve()),
                None,
            ),
            required=True,
            passed=not private.resolve().is_relative_to(ROOT.resolve()),
        )
    )
    return dict(checks=checks, failures=[r for r in checks if not r["passed"]], hashes=hashes)


def commands(private: Path) -> list[dict[str, Any]]:
    """Seal exact private unit/child coverage commands before the delta read."""
    private.mkdir(parents=True, exist_ok=True)
    cov = private / "coverage"
    cov.mkdir(exist_ok=True)
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel = True\npatch = subprocess\ndata_file = "
        + str(cov / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
    )
    py, pytest, ruff, mypy = (
        str(ROOT / ".venv/bin" / p) for p in ("python", "pytest", "ruff", "mypy")
    )
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    routes = [
        (
            "python_environment",
            [
                py,
                "-c",
                "import sys,pytest,coverage,ruff,mypy; print(sys.version); print(sys.executable)",
            ],
        ),
        (
            "unit_child_coverage",
            [
                py,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                *common,
                TEST,
                "--basetemp=" + str(private / "unit"),
            ],
        ),
        (
            "combine_coverage",
            [py, "-m", "coverage", "combine", "--rcfile=" + str(config), str(cov)],
        ),
        (
            "coverage_100",
            [
                py,
                "-m",
                "coverage",
                "json",
                "--rcfile=" + str(config),
                "-o",
                str(private / "coverage.json"),
                "--fail-under=100",
            ],
        ),
        ("ruff_check", [ruff, "check", *OWNED, TEST]),
        ("ruff_format", [ruff, "format", "--check", *OWNED, TEST]),
        ("strict_mypy", [mypy, "--strict", *OWNED]),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", "--files", TEST]),
        (
            "affected_consumers",
            [pytest, *common, *CONSUMERS, "--basetemp=" + str(private / "consumers")],
        ),
        (
            "e2e_017",
            [
                pytest,
                *common,
                "tests/python/test_arc_supervisor_delta_7874.py",
                "--basetemp=" + str(private / "e2e017"),
            ],
        ),
        ("full_python_suite", [pytest, "tests/python", "-q"]),
    ]
    return [
        dict(
            name=n,
            argv=a,
            deadline_s=180,
            expected_exit=0,
            classification="repository_health" if n == "full_python_suite" else "required",
        )
        for n, a in routes
    ]


def replay(value: dict[str, Any]) -> list[str]:
    """Reopen immutable primitives and recompute fields in a fresh process."""
    errors = []
    for group in ("source_artifact_hashes", "raw_shard_hashes", "code_config_hashes"):
        for label, digest in value.get(group, {}).items():
            path = Path(label)
            path = path if path.is_absolute() else ROOT / path
            if not path.is_file() or sha256_file(path) != digest:
                errors.append("sha256:" + label)
    primitive = Path(value["primitive_path"])
    if not primitive.is_file():
        return errors + ["missing_primitive"]
    stored = json.loads(primitive.read_text())
    reduced = reader.read_delta(
        Path(value["locator_path"]),
        precondition_failures=[
            r for r in value["preconditions_checked"] if r.get("required") and not r["passed"]
        ],
    )
    if reduced != stored:
        errors.append("primitive_reduction_drift")
    expected = (
        reader.reduction([], stored["prior_frontier"]) if value["fixture_claim_scope"] else stored
    )
    for key in reader.reduction([], {}):
        if value.get(key) != expected.get(key):
            errors.append("reduction_drift:" + key)
    if value.get("experiment_id") != 8215 or value.get("task_id") != reader.TASK_ID:
        errors.append("task_identity")
    if value.get("reproducibility_checksum") != canonical_hash(stored):
        errors.append("checksum")
    for row in value.get("validation_receipts", []):
        for stream in ("stdout", "stderr"):
            path = Path(row[stream + "_path"])
            if not path.is_file() or sha256_file(path) != row[stream + "_sha256"]:
                errors.append("validation_stream:" + row["name"] + ":" + stream)
        receipt = Path(row["stdout_path"]).with_name(row["name"] + ".receipt.json")
        original = json.loads(receipt.read_text())
        if any(row.get(k) != v for k, v in original.items()):
            errors.append("validation_receipt:" + row["name"])
    for key, expected_value in dict(
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        defaults_changed=False,
        new_level_solves_claimed=0,
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
    ).items():
        if value.get(key) != expected_value:
            errors.append("execution_claim:" + key)
    owned = [r for r in value.get("validation_receipts", []) if r["classification"] == "required"]
    covered = value["acceptance_gates"]["owned_coverage_100"]
    state = (
        "disqualified"
        if not covered or any(not r["passed"] for r in owned)
        else "blocked"
        if stored["failures"]
        else "null"
    )
    for key, expected_value in dict(
        verdict_class=state,
        required_checks_passed=covered and all(r["passed"] for r in owned),
        supervisor_reader_ready_score=int(state == "null" and not value["fixture_claim_scope"]),
        arc_delta_ready_score=int(state == "null"),
        new_outcome_ready_score=int(state == "null" and expected["new_outcome_count"] > 0),
    ).items():
        if value.get(key) != expected_value:
            errors.append("terminal_state:" + key)
    return errors


def terminal(candidate: Path, private: Path, raw: Path) -> dict[str, Any]:
    """Keep unchanged public validators and cold replay on identical candidate bytes."""
    reports = [
        child(name, argv, raw / "terminal_logs", deadline=60)
        for name, argv in [
            (
                "terminal_adversarial",
                [
                    str(ROOT / ".venv/bin/python"),
                    "scripts/adversarial_verify.py",
                    "--json",
                    str(candidate),
                ],
            ),
            (
                "terminal_rows",
                [
                    str(ROOT / ".venv/bin/python"),
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ],
            ),
            (
                "cold_replay",
                [str(ROOT / ".venv/bin/python"), str(ROOT / CLI), "--cold-replay", str(candidate)],
            ),
        ]
    ]
    return dict(passed=all(r["passed"] for r in reports), reports=reports)


def execute(locator: Path, output: Path, private: Path, *, fixture: bool = False) -> int:
    """Freeze inputs, perform one bounded read, and publish only validated bytes."""
    start = time.monotonic_ns()
    started_at = datetime.now(UTC).isoformat()
    private.mkdir(parents=True, exist_ok=True)
    raw = output.parent / "raw" / output.stem / "invocations" / str(start)
    raw.mkdir(parents=True, exist_ok=True)
    progress("exp8215_preconditions", 0, 1)
    available = preconditions(private)
    specs = [] if fixture else commands(private)
    manifest = raw / "validation_command_manifest.json"
    code = dependency_hashes(ROOT, paths=[*OWNED, TEST])
    atomic_json(
        manifest,
        dict(
            commands=specs,
            code_config_hashes=code,
            owned_files=OWNED,
            applicable_e2e=["E2E-017", "E2E-023"],
            max_sources=64,
            max_bytes=33554432,
            deadline_s=120,
            heartbeat_s=30,
            fixture=fixture,
            preconditions=available["checks"],
        ),
    )
    progress("exp8215_bounded_delta", 0, 1)
    reduced = reader.read_delta(locator, precondition_failures=available["failures"])
    primitive = raw / "primitive_rows.json"
    atomic_json(primitive, reduced)
    source_hashes = dict(reduced["source_artifact_hashes"], **available["hashes"])
    snapshots = {}
    for label, digest in source_hashes.items():
        original = Path(label)
        copy = raw / "inputs" / digest[7:] / original.name
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, copy)
        snapshots[str(copy)] = sha256_file(copy)
    source_hashes.update(snapshots)
    source_hashes[str(manifest)] = sha256_file(manifest)
    progress("exp8215_validation", 0, len(specs))
    receipts = []
    for index, spec in enumerate(specs):
        progress("exp8215_check", index, len(specs) - index)
        receipts.append(
            dict(
                child(
                    spec["name"],
                    spec["argv"],
                    raw / "logs",
                    deadline=spec["deadline_s"],
                    expected=spec["expected_exit"],
                    scope=spec["classification"],
                ),
                classification=spec["classification"],
            )
        )
        receipt_file = raw / "logs" / (spec["name"] + ".receipt.json")
        snapshots[str(receipt_file)] = sha256_file(receipt_file)
    owned = [r for r in receipts if r["classification"] == "required"]
    covered = fixture or coverage_complete(private / "coverage.json", includes=OWNED)
    gates = list(reduced["failures"]) + available["failures"]
    for row in owned:
        if not row["passed"]:
            gates.append(
                reader.operand(
                    Path(row["stdout_path"]), "exit_code", 0, row["exit_code"], row["stdout_sha256"]
                )
            )
    if not covered:
        gates.append(
            reader.operand(
                private / "coverage.json", "owned_statement_coverage", 100, "incomplete", None
            )
        )
    state = (
        "disqualified"
        if not covered or any(not r["passed"] for r in owned)
        else "blocked"
        if reduced["failures"] or available["failures"]
        else "null"
    )
    honest = (
        "complete_"
        + state
        + (
            "_authority_prerequisites"
            if state != "null"
            else "_descriptive_arm_selection"
            if reduced["new_outcome_count"] and not fixture
            else "_no_new_outcomes"
        )
    )
    live = reader.reduction([], reduced["prior_frontier"]) if fixture else reduced
    coverage = {}
    if (private / "coverage.json").is_file():
        measured = json.loads((private / "coverage.json").read_text())
        coverage = {p: v["summary"] for p, v in measured["files"].items()}
        copy = raw / "coverage.json"
        shutil.copyfile(private / "coverage.json", copy)
        snapshots[str(copy)] = sha256_file(copy)
    value = dict(
        live,
        experiment_id=8215,
        experiment=8215,
        task_id=reader.TASK_ID,
        milestone="2026.10.709",
        run_date="20261006",
        schema="arc-authoritative-frontier-v709",
        title="Producer-bound ARC authoritative outcome frontier",
        status="complete",
        honest_verdict=honest,
        verdict_class=state,
        gate_check_summary=gates,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
        duration_s=(time.monotonic_ns() - start) / 1e9,
        random_seed=0,
        reproducibility_checksum=canonical_hash(reduced),
        source_artifact_hashes=source_hashes,
        raw_shard_hashes={str(primitive): sha256_file(primitive), **snapshots},
        code_config_hashes=code,
        preconditions_checked=[*available["checks"], *reduced["authority_locator_rows"]],
        primitive_path=str(primitive),
        locator_path=str(locator),
        authority_locator_rows=reduced["authority_locator_rows"],
        historical_receipt_count=reduced["historical_receipt_count"],
        cited_upstream_artifacts=[
            dict(path=p, sha256=h, role="producer_named_input_or_contract")
            for p, h in reduced["source_artifact_hashes"].items()
        ],
        supervisor_reader_ready_score=int(state == "null" and not fixture),
        arc_delta_ready_score=int(state == "null"),
        arc_evidence_ready_score=int(state == "null" and not fixture),
        new_outcome_ready_score=int(state == "null" and live["new_outcome_count"] > 0),
        verifier_is_oracle=False,
        exposure_scope="exposed_development_cached_supervisor_receipts",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_model_invocation_count=0,
        current_game_execution_count=0,
        solve_provenance="development_proxy" if fixture else "live_agent_self_discovery",
        new_level_solves_claimed=0,
        defaults_changed=False,
        acceptance_gates=dict(
            authority_authenticated=not reduced["failures"],
            owned_checks=all(r["passed"] for r in owned),
            owned_coverage_100=covered,
            separate_live_comparison_required=True,
        ),
        required_checks_passed=covered and all(r["passed"] for r in owned),
        validation_receipts=receipts,
        coverage_statement_counts=coverage,
        repository_health=[r for r in receipts if r["classification"] == "repository_health"],
        flagged_adversarial=False,
        fixture_claim_scope="development_proxy" if fixture else None,
        fixture_result=reduced if fixture else None,
        methodology="Authenticate producer-named paths and content identities, then reduce only post-frontier redirects; descriptive leave-one-game-out selection preserves curated priorities.",
        optional_input_dispositions=[
            *reduced["optional_dispositions"],
            dict(
                path=str(ROOT / "ops/arc_bench_latest.json"),
                required=False,
                disposition="explore_engine_not_a_supervisor_receipt",
            ),
        ],
    )
    value["field_principles"] = {
        k: "Bind this invocation to exact producer bytes; exposed observations establish no independent benefit."
        for k in value
    }
    value["field_principles"].update(
        arc_delta_ready_score="A qualified completed delta read includes authenticated empty outcomes.",
        new_outcome_ready_score="Only current authenticated post-frontier outcomes make this gate one.",
        supervisor_reader_ready_score="Current owned checks qualify the producer locator and reader.",
        historical_receipt_count="Producer ledger history is real; absent chronology cannot make it new.",
        rows="Each redirect retains its source, denominator, chronology and exclusion or censoring.",
        model_invocation_counts="Copied generator provenance is not a current model invocation.",
    )
    value = normalize_artifact_for_template_write(value)
    candidate = private / output.name
    atomic_json(candidate, value)
    report = terminal(candidate, private, raw)
    if not report["passed"]:
        raise ValueError("terminal_candidate_rejected")
    published = publish_primary(
        output, value, lambda path: dict(report, passed=sha256_file(path) == sha256_file(candidate))
    )
    atomic_json(raw / "terminal_reports.json", dict(published, report=report))
    progress("exp8215_published", live["new_outcome_count"], 0)
    print(honest, flush=True)
    return int(state != "null")
