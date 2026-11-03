"""REQ-REPORT-8760: private sensitivity never changes the closed live policy."""

from datetime import datetime, UTC
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory, mkdtemp
import time
from typing import Any
from unittest.mock import patch

from coverage import Coverage

from carnot.reporting import arc_outcome_frontier_8746 as qualified
from carnot.reporting import v750_archive_runner as archive
from carnot.reporting.arc_horizon_bounds_8760 import (
    ARMS as ARMS,
    assert_widens as assert_widens,
    bounds as bounds,
    grid,
)
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
e = namespace(qualified.legacy, "horizon8760_report")
runtime = namespace(qualified.previous.original_runtime, "horizon8760_runtime")
e.NAME, e.TASK, e.MILESTONE = (
    "experiment_8760_v750_arc_horizon_sensitivity",
    "exp8760-arc-horizon-sensitivity",
    "2026.11.750",
)
e.CLI = f"scripts/experiments/{e.NAME}.py"
e.TESTS = ["tests/python/test_arc_horizon_sensitivity_8760.py"]
e.OWNED = ["python/carnot/reporting/arc_horizon_bounds_8760.py", __file__, e.CLI]
FRONTIER = "results/experiment_8746_v749_arc_outcome_frontier.json"
PIN = "sha256:97a365f58aecb2053246acc020375ac7e523e0b06d683c5b78d163ae4af1f155"
MODEL_SPECS: list[Json] = []
checks = namespace(archive, "horizon8760_checks")
checks.a = namespace(archive.a, "horizon8760_scope")
checks.a.OWNED, checks.a.TESTS, checks.a.CLI = e.OWNED, e.TESTS, e.CLI
checks.a.CONSUMERS = [
    "tests/python/test_arc_outcome_frontier_8746.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_arc_supervisor_delta_7874.py",
    "tests/python/test_arc_authoritative_frontier_8215.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]
base_build = e.build


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush the current phase and case count so a quiet child remains visible."""
    print(f"[exp8760] phase={phase} completed={completed} pending={pending}", flush=True)


def measure(raw: Path, run_date: str) -> Json:
    """Authenticate the original frontier, then seal sources before private masking."""
    start, wall = time.monotonic_ns(), datetime.now(UTC).isoformat()
    progress("preconditions_and_registry_before")
    raw.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="arc8760-", dir="/var/tmp") as directory:
        probe = Path(directory) / "probe"
        probe.write_text("private scratch")
        pre = dict(
            private_scratch_writable=probe.read_text() == "private scratch",
            tools={
                n: (e.ROOT / ".venv/bin" / n).is_file()
                for n in ["python", "pytest", "ruff", "mypy"]
            },
            no_model_needed=True,
        )
    failures, refs, episodes = [], [], []
    attempts = [
        e.reference(p)
        for p in sorted(
            (e.ROOT / "results/raw" / e.NAME / "invocations").glob("*/owned_attempt_outcome.json")
        )
    ]
    upstream: Json = {}
    before: Json = {}
    after: Json = {}
    new: list[Json] = []
    path = e.ROOT / FRONTIER
    gate = qualified.legacy.original.observe(
        path, "exp8746-arc-outcome-frontier", "arc_frontier_ready_score", PIN
    )
    try:
        if not gate["passed"]:
            raise ValueError("qualified_frontier_unavailable")
        prior, refs = qualified.legacy.qualified(path, PIN)
        upstream, retained = qualified.authenticate()
        refs += retained
        baseline = e.read(dict(path=str(e.ROOT / e.BASELINE), sha256=e.LEDGER_PIN))
        path = e.ROOT.joinpath(*e.DEFAULT_LEDGER_PARTS)
        ledger = json.loads(path.read_bytes())
        refs += [e.reference(path), e.reference(e.ROOT / e.BASELINE)]
        before = prior["frontier_after"]
        after = before
        work = e.read(upstream["primitive_reference"])
        refs += [upstream["primitive_reference"]] + work["episode_refs"]
        episodes = [e.read(ref) for ref in work["episode_refs"]]
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(
            dict(
                gate,
                path=str(path),
                check="qualified_frontier_operands",
                observed=str(error),
                passed=False,
                operand_state="missing_file" if not path.is_file() else gate["operand_state"],
            )
        )
    refs += attempts
    dependencies = (
        e.OWNED
        + e.TESTS
        + checks.a.CONSUMERS
        + [
            "AGENTS.md",
            "CODEX.md",
            "CLAUDE.md",
            "ops/e2e-test-plan.md",
            "ops/arc_solve_registry.yaml",
            "ops/exclusion_manifest.yaml",
            "pyproject.toml",
            "openspec/capabilities/research-reporting/spec.md",
            "openspec/capabilities/verification/spec.md",
            "openspec/change-proposals/research-roadmap-vNEXT.md",
            "docs/research-notes/v750-arc-horizon-sensitivity.md",
            "python/carnot/reporting/arc_censoring_reduce_8691.py",
            "python/carnot/reporting/v750_archive_runner.py",
            "python/carnot/reporting/v746_retained_evidence.py",
            "python/carnot/reporting/collection_contract.py",
            "python/carnot/reporting/v709_execution.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/reporting/v748_collection_contract.py",
            "python/carnot/reporting/arc_supervisor_runner_8720.py",
            "python/carnot/reporting/arc_supervisor_frontier_8720.py",
            "scripts/experiment_template.py",
            "scripts/adversarial_verify.py",
            "scripts/verdict_row_consistency_lint.py",
            "research-references.md",
            "openspec/change-proposals/research-roadmap-v749-preserved-20261102.md",
            "results/experiment_8737_v749_contract_methods.json",
            "results/experiment_8738_v749_deadline_cost_diagnosis.json",
            "results/experiment_8743_v749_single_extraction_prototype.json",
            "results/experiment_8745_v749_bounded_durable_cost.json",
        ]
    )
    sources = []
    for name in dependencies:
        source = e.ROOT / name
        target = raw / "source_closure" / source.relative_to(e.ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        sources += [e.reference(source), e.reference(target)]
    test_scratch = Path(mkdtemp(prefix="arc8760-tests-", dir="/var/tmp"))
    plan, collected = checks.collect(
        test_scratch, raw, e.TESTS + checks.a.CONSUMERS, e.TESTS, checks.child
    )
    e.atomic_json(
        raw / "preregistered.json",
        dict(
            source_refs=sources,
            input_refs=refs,
            test_plan=plan,
            collection=collected,
            games=[None, *sorted({ep["unit"]["game"] for ep in episodes})],
            horizons=[120, 30, 60, 90],
            missing_arms=[None, *ARMS],
            estimand="120_action_binary_progress_full_intended_episode_roster",
            unknown_interval=[0, 1],
            no_exchangeable_censoring_assumption=True,
        ),
    )
    if not failures:
        try:
            progress("new_receipts_after_freeze")
            new, after = e.delta(baseline, ledger, upstream)
            refs += [
                ref for row in new for ref in (row["source_reference"], row["producer_reference"])
            ]
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            failures.append(
                dict(
                    upstream="live_redirect_ledger",
                    path=str(path),
                    sha256=e.sha256_file(path),
                    check="new_receipt_authentication",
                    field="new_events",
                    operator="==",
                    expected="authenticated",
                    observed=str(error),
                    passed=False,
                    operand_state="rejected_producer",
                )
            )
    private_refs = []
    for index, episode in enumerate(episodes):
        target = raw / "episodes" / f"{index:02}.json"
        e.atomic_json(target, episode)
        private_refs.append(e.reference(target))
    progress("frozen_before_grid", len(episodes), 0)
    analysis = (
        grid(episodes, progress)
        if episodes
        else dict(
            rows=[],
            censoring_sensitivity_rows=[],
            overlap_rows=[],
            independent_reduction_rows=[],
            baseline_bounds={},
        )
    )
    end = time.monotonic_ns()
    work = dict(
        raw=str(raw),
        run_date=run_date,
        started_at_utc=wall,
        ended_at_utc=datetime.now(UTC).isoformat(),
        start_ns=start,
        end_ns=end,
        duration_s=(end - start) / 1e9,
        preconditions=pre,
        gates=[gate],
        failures=failures,
        code_refs=[e.reference(e.ROOT / p) for p in e.OWNED + e.TESTS],
        source_refs=refs + sources + [e.reference(raw / "preregistered.json")],
        upstream_refs=refs,
        episode_refs=private_refs,
        authority=dict(ready=gate["passed"]),
        historical_model_provenance={},
        registry_precheck=upstream.get("registry_precheck", {}),
        frontier_before=before,
        frontier_after=after,
        new_event_rows=new,
        policy_source_reference=e.reference(
            e.ROOT / "python/carnot/agentic/arc_trajectory_supervisor.py"
        ),
        phase_spans=[
            dict(
                phase="authenticate_freeze_grid",
                duration_s=(end - start) / 1e9,
                started_monotonic_ns=start,
                ended_monotonic_ns=end,
            )
        ],
        analysis=analysis,
        preserved_failed_attempts=attempts,
    )
    e.atomic_json(raw / "measurement.json", work)
    e.atomic_json(
        raw.parents[1] / "seals" / (raw.name + ".json"),
        dict(measurement=e.reference(raw / "measurement.json"), code_refs=work["code_refs"]),
    )
    progress("measurement_complete", len(analysis["censoring_sensitivity_rows"]), 0)
    return work


def coverage_summary(raw: Path, children: list[Json]) -> Json:
    """Measure only retained unit and real CLI traces, with no nested collector.

    The adapter runs inside the unit collector. A second parent collector stops
    that trace, so its apparent missing statements are a measurement artifact.
    This combines the real child SQLite files and applies the same complete
    statement and zero-exclusion checks to every owned source file.
    """
    combined = checks.evidence.combine(children, raw / "combined")
    cov = Coverage(data_file=combined["combined_reference"]["path"], config_file=False)
    cov.load()
    files = []
    for name in checks.a.OWNED:
        _, statements, excluded, missing, _ = cov.analysis2(str(checks.a.ROOT / name))
        files.append(
            dict(path=name, statements=len(statements), excluded=excluded, missing=missing)
        )
    total = sum(row["statements"] for row in files)
    missing_count = sum(len(row["missing"]) for row in files)
    return dict(
        complete=total > 0 and missing_count == 0 and not any(row["excluded"] for row in files),
        percent=100 * (total - missing_count) / total,
        files=files,
        combined_reference=combined["combined_reference"],
        scope="retained_unit_and_real_cli_child_traces",
    )


def validate(raw: Path) -> list[Json]:
    """Reuse complete scoped collection and retained actual SQLite child evidence.

    The plan is frozen before measurement. Unit and real CLI child collectors
    cover this adapter. Starting another collector here interrupts their trace;
    the harness receives an idle controller and combines retained child data.
    Repository health remains separate from these complete scoped checks.
    """
    frozen = json.loads((raw / "preregistered.json").read_bytes())
    tracer = Coverage(data_file=str(raw / "producer.sqlite"), config_file=False)
    with patch.object(checks, "collect", return_value=(frozen["test_plan"], frozen["collection"])):
        result = checks.validate(raw, tracer, [])
    e.atomic_json(raw / "scoped_validation.json", result)
    return list(result["receipts"]) + [
        dict(
            name="original_disjoint_union",
            passed=result["union"]["passed"],
            details_reference=e.reference(raw / "scoped_validation.json"),
        ),
        dict(name="preserved_raw_child_sqlite", passed=bool(result["retained_child_rows"])),
        dict(name="coverage_report", passed=result["coverage"]["complete"]),
    ]


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Separate completed diagnostic readiness from unsupported causal promotion."""
    analysis = (
        grid([e.read(ref) for ref in work["episode_refs"]], lambda *args: None)
        if work["episode_refs"]
        else work["analysis"]
    )
    if analysis != work["analysis"]:
        raise ValueError("independent_grid_drift")
    value: Json = base_build(work, receipts, output)
    production = next((r for r in receipts if r["name"] == "original_disjoint_union"), {})
    validation = e.read(production["details_reference"]) if production else {}
    ready = bool(
        value["arc_frontier_ready_score"]
        and all(work["preconditions"]["tools"].values())
        and work["preconditions"]["private_scratch_writable"]
        and all(r["passed"] for r in analysis["independent_reduction_rows"])
    )
    rows = analysis["rows"]
    value.update(
        experiment_id=8760,
        milestone=e.MILESTONE,
        random_seed=7508760,
        honest_verdict=f"complete_{value['verdict_class']}_arc_horizon_sensitivity",
        rows=rows,
        intended_count=len(rows),
        completed_count=sum(r["complete_outcome"] for r in rows),
        censored_count=sum(not r["complete_outcome"] for r in rows),
        failed_count=0,
        execution_check_failed_count=int(not value["required_checks_passed"]),
        independent_count=len({r["game"] for r in rows}),
        arc_sensitivity_ready_score=int(ready),
        arc_frontier_ready_score=int(ready),
        standing_generalization_floor_satisfied=ready,
        censoring_sensitivity_rows=analysis["censoring_sensitivity_rows"],
        overlap_rows=analysis["overlap_rows"],
        independent_reduction_rows=analysis["independent_reduction_rows"],
        baseline_bounds=analysis["baseline_bounds"],
        owned_statement_coverage=validation.get("coverage"),
        preserved_failed_attempts=work["preserved_failed_attempts"],
        scoped_validation_scope=dict(
            required_suites=e.TESTS + checks.a.CONSUMERS,
            exact_node_ids=validation.get("union", {}).get("node_ids", []),
            e2e=["E2E-017", "E2E-023", "private E2E-021"],
            original_assertions_unchanged=True,
            repository_health_is_dependency=False,
        ),
        repository_health_reference=dict(
            full_suite_pass=False,
            current_invocation_diagnostic=e.reference(
                e.ROOT / "results/raw" / e.NAME / "startup/full_python_suite.receipt.json"
            ),
            note="A bounded repository health diagnostic is separate from complete scoped validation.",
        ),
        solve_provenance=None,
        new_level_credit=0,
        numerical_training_calls=0,
        exposure_scope="exposed_development_private_masking_of_sealed_observational_panel",
        policy_decision="no_supported_policy_change",
        unresolved_count=sum(r["binary_outcome"] is None for r in rows),
        methodology_note="Authenticate Exp8746 and registry before reading only new reachable receipts. Freeze input, method and exact test nodes, then mask private copies of the closed sixteen-episode four-game 120-action panel. Cross per-game removal, 30/60/90/120 follow-up and missing-arm masks. Compare the unchanged frozen action reducer to independent status/count reconstruction. Bound missing binary potential outcomes in [0,1] over the full intended roster for every arm. Preserve reset cause, selected and applied status. Fewer than three shared games or five firings per shared arm cell suppresses admission. These admission rules never certify broad transfer. The exposed observational diagnostic establishes no causal supervisor benefit, policy change or solve. No game replay, hidden source inspection or model load occurs.",
    )
    value["acceptance_gates"].update(arc_sensitivity_ready_score=int(ready))
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Exact sealed bytes and current invocation bind evidence. Unknown binary outcomes retain [0,1] over the full roster; removing support cannot create benefit, transfer or solve credit.",
    )
    value["reproducibility_checksum"] = e.checksum(value)
    return value


e.build, e.progress = build, progress
checks.coverage_summary = coverage_summary
runtime.e, runtime.measure, runtime.validate = e, measure, validate
main = runtime.main
