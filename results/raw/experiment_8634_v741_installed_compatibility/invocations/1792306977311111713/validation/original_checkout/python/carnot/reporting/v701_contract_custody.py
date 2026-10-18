"""REQ-REPORT-8097: administrative custody preserves science without repeating it.

Authority, public numerical fixtures and historical controls have separate gates
because a failed scientific predecessor cannot invalidate an executable schedule.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import v685_authority_lifecycle as authority
from carnot.reporting import v700_contract_custody as previous
from carnot.reporting.current_work_receipt import (
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.v698_fixture_consumer_contract import failure
from carnot.reporting.v699_contract_custody import InputFailure

Json = dict[str, Any]
ROOT = previous.ROOT
NAME = "experiment_8097_v701_contract_custody"
TASK = "exp8097-contract-custody"
MILESTONE = "2026.10.701"
DESIGN = previous.DESIGN
MODULE = "python/carnot/reporting/v701_contract_custody.py"
RUNNER = previous.RUNNER
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_contract_custody_8097.py"
COVERAGE_PATHS = [MODULE, CLI]
CHECK_PATHS = [*COVERAGE_PATHS, previous.MODULE, RUNNER]
Binder = previous.Binder
HISTORY_NAMES = {
    8083: "experiment_8083_v700_contract_custody.json",
    8084: "experiment_8084_v700_fresh_cohort_methods.json",
    8085: "experiment_8085_v700_radial_memory_kernel.json",
}
HISTORY_HASHES = {
    8083: "sha256:e7072ec1370e112ca0bd85fed88a4af446fd621db812939eecb2d254466d3e90",
    8084: "sha256:ebc47b3e4072f34ea49ebcaf9ff7c5bebd168e3910ea15f7d428d3bcef12703a",
    8085: "sha256:68997833e5d8b5ec78a0d22435ed2ebf12dec74968d0f55b434deea9f9e50988",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual completed units so custody work never implies model execution."""
    print(f"[exp8097] phase={phase} completed={completed} pending={pending}", flush=True)


def assess(design: Path, staged: Path, active: Path, raw: Path) -> Json:
    """Reuse the qualified reader, including activation after staging is consumed."""
    return previous.assess(
        design, staged, active, raw, milestone=MILESTONE, first_id=8097, count=13, task=TASK
    )


def dispositions(primary: Json, path: Path, report: Json) -> Json:
    """Preserve the producer's verdict and validation result without upgrading either."""
    return dict(
        experiment_id=primary["experiment_id"],
        task_id=primary["task_id"],
        path=str(path),
        sha256=sha256_file(path),
        verdict_class=primary["verdict_class"],
        honest_verdict=primary["honest_verdict"],
        required_checks_passed=primary.get("required_checks_passed"),
        historical_validation_passed=report.get("passed"),
        terminal_validation_sidecar_path=primary["terminal_validation_sidecar_path"],
        no_retry_unchanged_outcome=True,
    )


def historical(root: Path, binder: Binder, *, fixture: bool) -> Json:
    """Authenticate three actual outcomes; controls read immutable historical copies."""
    result: Json = dict(
        historical_dispositions=[],
        kernel_ready=False,
        controls_ready=False,
        public_kernel={},
        historical_controls=[],
    )
    for index, (n, name) in enumerate(HISTORY_NAMES.items()):
        path = root / "results" / name
        progress("historical_before", index, 3 - index)
        try:
            value = binder.read(path, None if fixture else HISTORY_HASHES[n])
            report = binder.terminal_evidence(path, value)
            result["historical_dispositions"].append(dispositions(value, path, report))
            if n == 8085:
                for field, expected in [
                    ("required_checks_passed", True),
                    ("flagged_adversarial", False),
                    ("kernel_ready_score", 1),
                ]:
                    binder.require(path, field, expected, value.get(field))
                binder.require(path, "terminal.report.passed", True, report.get("passed"))
                result["public_kernel"] = dict(
                    path=str(path),
                    sha256=sha256_file(path),
                    verdict_class=value["verdict_class"],
                    kernel_ready_score=1,
                )
                for ref in value.get("source_artifact_hashes", []) + value.get(
                    "raw_shard_hashes", []
                ):
                    saved = Path(ref.get("snapshot_path", ref["path"]))
                    result["historical_controls"].append(binder.bind(saved, ref["sha256"]))
                result["kernel_ready"] = True
            if n == 8083:
                for ref in value.get("checkpoint_hashes", []):
                    result["historical_controls"].append(
                        binder.bind(Path(ref["snapshot_path"]), ref["sha256"])
                    )
                for role, ref in value.get("authority_snapshots", {}).items():
                    if ref["exists"]:
                        result["historical_controls"].append(
                            binder.bind(Path(ref["snapshot_path"]), ref["sha256"])
                        )
        except (InputFailure, OSError, ValueError, KeyError, TypeError) as error:
            binder.failures.append(
                failure(path, "historical_evidence_readable", True, str(error), TASK)
            )
        progress("historical_after", index + 1, 2 - index)
    result["controls_ready"] = len(result["historical_dispositions"]) == 3 and not binder.failures
    return result


def measure(
    root: Path, design: Path, staged: Path, active: Path, raw: Path, *, fixture: bool = False
) -> Json:
    """Freeze prerequisites and primitive authority observations without loading models."""
    started = time.monotonic_ns()
    binder = Binder(raw / "inputs", task=TASK)
    progress("preconditions_before")
    names = [
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "ops/e2e-test-plan.md",
        "openspec/capabilities/research-reporting/spec.md",
        "openspec/capabilities/verification/spec.md",
        "scripts/experiment_template.py",
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "ops/exclusion_manifest.yaml",
        "python/carnot/reporting/v685_authority_lifecycle.py",
        previous.MODULE,
    ]
    for path in ([] if fixture else [root / n for n in names]) + [
        ROOT / ".venv/bin" / n for n in ["python", "pytest", "coverage", "ruff", "mypy"]
    ]:
        try:
            binder.bind(path)
        except InputFailure:
            pass
    progress("preconditions_after", len(binder.observations))
    contract = assess(design, staged, active, raw / "authority")
    progress("authority_after", len(contract["contract_rows"]), 13 - len(contract["contract_rows"]))
    history = historical(root, binder, fixture=fixture)
    ended = time.monotonic_ns()
    return dict(
        contract=contract,
        history=history,
        refs=binder.refs,
        preconditions_checked=binder.observations + contract["gate_check_summary"],
        failures=binder.failures + contract["gate_check_summary"],
        started_ns=started,
        ended_ns=ended,
        owner_pid=os.getpid(),
        fixture=fixture,
        phase_spans=[dict(phase="custody", start_s=0, end_s=(ended - started) / 1e9)],
    )


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Reduce primitive checks; historical failure and owned failure have different effects."""
    owned = bool(receipts) and all(r["passed"] for r in receipts)
    contract, history = work["contract"], work["history"]
    failures = work["failures"] + [
        failure(
            Path(r.get("log_path", raw)),
            r["name"],
            r.get("expected_exit", 0),
            r.get("exit_code"),
            TASK,
        )
        for r in receipts
        if not r["passed"]
    ]
    verdict = "disqualified" if not owned else "blocked" if failures else "circular_positive"
    rows = []
    for i in range(13):
        original = contract["contract_rows"][i] if i < len(contract["contract_rows"]) else {}
        checks = original.get("checks", dict(executable_authority_available=False))
        matched = all(checks.values())
        rows.append(
            dict(
                source_id="V701_authority",
                source="V701_authority",
                unit_id=f"exp{8097 + i}",
                arm="contract_custody",
                condition="full_executable_contract",
                issued_state=contract["canonical_tasks_sha256"],
                metric="authority_checks_matched",
                checks=checks,
                absolute_metric=int(matched),
                numerator=sum(checks.values()),
                denominator=len(checks),
                raw_numerator=sum(checks.values()),
                raw_denominator=len(checks),
                status="completed",
                excluded=not matched,
                censored=False,
                exclusion_reason=None if matched else "external_authority_mismatch",
                effective_independent_groups=0,
            )
        )
    current = build_current_work_receipt(
        run_id=str(work["started_ns"]),
        owner_pid=work["owner_pid"],
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details=dict(mode="no_model_load"),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=work["started_ns"],
        ended_monotonic_ns=work["ended_ns"],
    )
    value: Json = dict(
        experiment_id=8097,
        experiment=8097,
        task_id=TASK,
        milestone=MILESTONE,
        schema="carnot.experiment.v1",
        run_date="20261004",
        status="complete",
        title="V701 contract custody",
        honest_verdict="complete_circular_positive_contract_custody"
        if verdict == "circular_positive"
        else "complete_"
        + verdict
        + "_"
        + (Path(failures[0]["path"]).name.replace(".", "_") if failures else "owned_validation"),
        verdict_class=verdict,
        verifier_is_oracle=0,
        claim_scope=0,
        exposure_scope=0,
        generalized_learning_benefit_score=0,
        flagged_adversarial=False,
        required_checks_passed=owned,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        preconditions_checked=work["preconditions_checked"],
        gate_check_summary=failures,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=current["invocation_counts"],
        trained_head_specs=[],
        current_work_receipt=current,
        duration_s=current["duration_s"],
        rows=rows,
        intended_count=13,
        eligible_count=sum(not r["excluded"] for r in rows),
        independent_count=0,
        completed_count=13,
        excluded_count=sum(r["excluded"] for r in rows),
        censored_count=0,
        failed_count=0,
        sample_size_budget=dict(
            administrative_tasks=13,
            historical_executed_tasks=3,
            unscheduled_promises=11,
            independent_scientific_units=0,
        ),
        random_seed=70197,
        reproducibility_checksum=canonical_hash(work),
        source_artifact_hashes=work["refs"],
        raw_shard_hashes=[
            dict(path=str(raw / n), sha256=sha256_file(raw / n))
            for n in ["work.json", "validation_commands.json"]
        ]
        if (raw / "work.json").is_file()
        else [],
        code_config_hashes=work.get("code_hashes", {}),
        phase_spans=work["phase_spans"],
        contract_ready_score=int(owned and contract["activated"]),
        radial_kernel_ready_score=int(owned and history["kernel_ready"]),
        historical_controls_ready_score=int(owned and history["controls_ready"]),
        canonical_tasks_sha256=contract["canonical_tasks_sha256"],
        authority_snapshots=contract["authority_snapshots"],
        task_contract=contract["tasks"],
        historical_dispositions=history["historical_dispositions"],
        promised_but_unscheduled_ids=list(range(8086, 8097)),
        public_kernel=history["public_kernel"],
        historical_controls=history["historical_controls"],
        repository_health=work.get("repository_health", []),
        coverage_statement_counts=work.get("coverage_statement_counts", {}),
        fixture_validation_scope=work["fixture"],
        substrate_declaration="aggregation_from_upstream_artifacts",
        methodology_note="Hash immutable full authority and three terminal V700 receipts; independently reduce fixture custody, not natural-data or model benefit.",
    )
    value["acceptance_gates"] = dict(
        contract="Owned checks and matching full activated authority prevent invented scheduled work.",
        kernel="Clean public kernel receipts and exact immutable controls prevent fixture science-class confusion.",
        history="Authenticity preserves predecessor negatives without laundering failed checks.",
        publication="Normal child exit and bound terminal validators prevent unchecked primary exposure.",
    )
    value["field_principles"] = {
        k: f"Record {k} so custody cannot imply independent scientific benefit." for k in value
    }
    value["field_principles"].update(
        verdict_class="Missing upstream evidence is terminal blocked; unfinished owned work alone is partial.",
        honest_verdict="Completed negative science never requests retries.",
        contract_ready_score="Scheduling authority is independent of historical scientific validity.",
        radial_kernel_ready_score="Circular numerical fixtures can be ready without positive science.",
        promised_but_unscheduled_ids="Eleven promised IDs were never scheduled or completed.",
    )
    return value


def replay(path: Path) -> bool:
    """Recompute sealed authority and predecessor rows so local rehashes cannot hide drift."""
    try:
        value = json.loads(path.read_text())
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        work = json.loads((raw / "work.json").read_text())
        refs = (
            value["source_artifact_hashes"]
            + value["raw_shard_hashes"]
            + [r for r in value["authority_snapshots"].values() if r["exists"]]
        )
        for ref in refs:
            if sha256_file(Path(ref.get("snapshot_path", ref.get("path")))) != ref["sha256"]:
                return False
        for ref in value["validation_receipts"]:
            if ref.get("log_path") and sha256_file(Path(ref["log_path"])) != ref["log_sha256"]:
                return False
        for label, ref in work["code_snapshots"].items():
            if sha256_file(Path(ref["snapshot_path"])) != value["code_config_hashes"][label]:
                return False
        snapshots = value["authority_snapshots"]
        with TemporaryDirectory(prefix="carnot-8097-replay-") as directory:
            paths = [
                Path(snapshots[k].get("snapshot_path", Path(directory) / k))
                for k in ["design", "staged", "active"]
            ]
            rebuilt = assess(*paths, Path(directory) / "assessment")
            for field in ["activated", "contract_rows", "tasks", "canonical_tasks_sha256"]:
                if rebuilt[field] != work["contract"][field]:
                    return False
        saved = {r["path"]: r for r in work["refs"]}
        controls = []
        kernel_ready = False
        for row in work["history"]["historical_dispositions"]:
            ref = saved[row["path"]]
            primary = json.loads(Path(ref["snapshot_path"]).read_text())
            terminal = json.loads(
                Path(
                    saved[primary["terminal_validation_sidecar_path"]]["snapshot_path"]
                ).read_text()
            )
            report_path = terminal.get("publication", terminal)["sidecar_path"]
            report = json.loads(Path(saved[report_path]["snapshot_path"]).read_text())["report"]
            expected = dispositions(primary, Path(ref["snapshot_path"]), report)
            expected.update(path=row["path"], sha256=ref["sha256"])
            if expected != row:
                return False
            candidates = []
            if primary["experiment_id"] == 8083:
                candidates = primary.get("checkpoint_hashes", []) + [
                    r for r in primary.get("authority_snapshots", {}).values() if r["exists"]
                ]
            if primary["experiment_id"] == 8085:
                qualified = (
                    primary.get("kernel_ready_score") == 1
                    and primary.get("required_checks_passed") is True
                    and primary.get("flagged_adversarial") is False
                    and report.get("passed") is True
                )
                kernel = (
                    dict(
                        path=row["path"],
                        sha256=ref["sha256"],
                        verdict_class=primary["verdict_class"],
                        kernel_ready_score=primary["kernel_ready_score"],
                    )
                    if qualified
                    else {}
                )
                if kernel != work["history"]["public_kernel"]:
                    return False
                candidates = (
                    primary.get("source_artifact_hashes", []) + primary.get("raw_shard_hashes", [])
                    if qualified
                    else []
                )
            for candidate in candidates:
                source = candidate.get("snapshot_path", candidate.get("path"))
                if source not in saved or saved[source]["sha256"] != candidate["sha256"]:
                    break
                controls.append(saved[source])
            else:
                if primary["experiment_id"] == 8085:
                    kernel_ready = qualified
        history = work["history"]
        controls_ready = len(history["historical_dispositions"]) == 3 and all(
            r.get("passed")
            for r in work["preconditions_checked"]
            if r["field"]
            in {
                "resource_exists",
                "sha256",
                "terminal.primary_sha256",
                "kernel_ready_score",
                "required_checks_passed",
                "flagged_adversarial",
                "terminal.report.passed",
            }
        )
        return (
            controls == history["historical_controls"]
            and kernel_ready == history["kernel_ready"]
            and controls_ready == history["controls_ready"]
            and build(work, raw, value["validation_receipts"]) == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
