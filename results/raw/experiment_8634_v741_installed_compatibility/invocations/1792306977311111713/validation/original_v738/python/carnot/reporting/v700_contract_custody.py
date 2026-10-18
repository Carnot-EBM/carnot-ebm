"""REQ-REPORT-8083: preserve negative science while qualifying separate custody inputs.

Historical model activity belongs to its original producers. This administrative
run only hashes and reduces saved evidence; it never retries their experiments.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.reporting import v685_authority_lifecycle as authority
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v698_fixture_consumer_contract import failure
from carnot.reporting.v699_contract_custody import Binder as HistoricalBinder, InputFailure

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8083_v700_contract_custody"
TASK = "exp8083-contract-custody"
MILESTONE = "2026.10.700"
DESIGN = "openspec/change-proposals/research-roadmap-vNEXT.md"
MODULE = "python/carnot/reporting/v700_contract_custody.py"
RUNNER = "python/carnot/reporting/v700_custody_execution.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_contract_custody_8083.py"
CAPSTONE_HASH = "sha256:5039003a263ffe69de443c9e9ad7c3e1dee7d875677d01d63e9ae5475cee8828"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Report actual phase work so a quiet child cannot look like model progress."""
    print(f"[exp8083] phase={phase} completed={completed} pending={pending}", flush=True)


class Binder(HistoricalBinder):
    """Record successful checks as well as failures; availability is not science."""

    def __init__(self, raw: Path, *, task: str = TASK):
        super().__init__(raw)
        self.task = task
        self.observations: list[Json] = []

    def require(self, path: Path, field: str, expected: Any, observed: Any) -> None:
        row = failure(path, field, expected, observed, self.task)
        row["passed"] = expected == observed
        self.observations.append(row)
        try:
            super().require(path, field, expected, observed)
        except InputFailure:
            self.failures[-1]["upstream"] = self.task
            raise

    def terminal_evidence(self, path: Path, value: Json) -> Json:
        """Authenticate failed receipts too, without changing their original validity."""
        side = self.read(Path(value["terminal_validation_sidecar_path"]))
        publication = side.get("publication", side)
        self.require(
            path, "terminal.primary_sha256", sha256_file(path), publication.get("primary_sha256")
        )
        validator = Path(publication["sidecar_path"])
        report = read_bound_sidecar(path, validator)
        self.bind(validator)
        self.logs(report)
        self.logs(side)
        return report["report"]

    def logs(self, value: Any) -> None:
        """Seal actual historical validator logs; failed diagnostics remain auditable."""
        if isinstance(value, dict):
            if value.get("log_path") and value.get("log_sha256"):
                path = Path(value["log_path"])
                self.bind(path if path.is_absolute() else ROOT / path, value["log_sha256"])
            for child in value.values():
                self.logs(child)
        elif isinstance(value, list):
            for child in value:
                self.logs(child)


def assess(
    design: Path,
    staged: Path,
    active: Path,
    raw: Path,
    *,
    milestone: str = MILESTONE,
    first_id: int = 8083,
    count: int = 14,
    task: str = TASK,
) -> Json:
    """Freeze every observed authority even when the external design is incomplete."""
    snapshots = {
        role: authority._snapshot(path, path.read_bytes() if path.is_file() else None, raw, role)
        for role, path in [("design", design), ("staged", staged), ("active", active)]
    }
    try:
        frozen = [
            Path(snapshots[role].get("snapshot_path", raw / ("absent-" + role)))
            for role in ["design", "staged", "active"]
        ]
        value = authority.assess_authorities(
            *frozen, raw / "assessment", milestone=milestone, first_id=first_id, count=count
        )
        _, tasks = parse_design(frozen[0].read_text(), milestone=milestone)
        digest = authority.tasks_digest(tasks)
        if digest != value["canonical_tasks_sha256"]:
            value["gate_check_summary"].append(
                failure(
                    design, "design_tasks_sha256", value["canonical_tasks_sha256"], digest, task
                )
            )
        if (
            snapshots["staged"]["exists"]
            and value["planning_matched"]
            and frozen[1].read_bytes() != frozen[2].read_bytes()
        ):
            value["gate_check_summary"].append(
                failure(
                    active,
                    "active_snapshot_bytes",
                    snapshots["staged"]["sha256"],
                    snapshots["active"]["sha256"],
                    task,
                )
            )
        value["tasks"] = tasks
        value["activated"] = not value["gate_check_summary"]
    except (OSError, ValueError, KeyError, TypeError, IndexError, yaml.YAMLError) as error:
        value = dict(
            activated=False,
            contract_rows=[],
            tasks=[],
            canonical_tasks_sha256=None,
            gate_check_summary=[failure(design, "authority_readable", True, str(error), task)],
        )
    value["authority_snapshots"] = snapshots
    value["gate_check_summary"] = [
        dict(
            check=r.get("field", r.get("artifact_field")),
            upstream=task,
            path=snapshots.get(
                next(
                    (k for k in snapshots if k in Path(r.get("artifact_path", r.get("path"))).name),
                    "design",
                ),
                snapshots["design"],
            )["source_path"],
            hash=r.get("hash", r.get("artifact_hash")),
            field=r.get("field", r.get("artifact_field")),
            op=r["op"],
            expected=r["expected"],
            observed=r["observed"],
            passed=False,
        )
        for r in value["gate_check_summary"]
    ]
    return value


def historical(root: Path, binder: Binder, *, expected_hash: str | None = CAPSTONE_HASH) -> Json:
    """Preserve thirteen original dispositions; only input qualification needs clean checks."""
    capstone_path = root / "results/experiment_8082_v699_capstone.json"
    capstone = binder.read(capstone_path, expected_hash)
    binder.terminal_evidence(capstone_path, capstone)
    tasks = capstone["task_contract"]
    binder.require(
        capstone_path,
        "historical_task_sequence",
        [f"exp{n}" for n in range(8070, 8083)],
        [t["id"].split("-")[0] for t in tasks],
    )
    previous = {r["task_id"]: r for r in capstone["task_dispositions"]}
    rows, qualified, cached = [], {}, {}
    for index, task in enumerate(tasks):
        progress("historical_primary_before", index, len(tasks) - index)
        path = root / task["deliverable"]
        old = previous[task["id"]]
        value = binder.read(path, old.get("sha256") if index < 12 else expected_hash)
        report = binder.terminal_evidence(path, value)
        rows.append(
            dict(
                task_id=task["id"],
                path=str(path),
                sha256=sha256_file(path),
                verdict_class=value["verdict_class"],
                honest_verdict=value["honest_verdict"],
                original_disposition=old,
                no_retry_unchanged_outcome=True,
                historical_validation_passed=report.get("passed"),
                terminal_validation_sidecar_path=value["terminal_validation_sidecar_path"],
            )
        )
        if index in [1, 2, 3, 11]:
            cached[8070 + index] = value
        progress("historical_primary_after", index + 1, len(tasks) - index - 1)
    ready = True
    try:
        methods, fitted, board = cached[8072], cached[8073], cached[8081]
        for n in [8072, 8073, 8081]:
            binder.require(
                root / tasks[n - 8070]["deliverable"],
                "historical_owned_checks",
                True,
                cached[n].get("required_checks_passed") is True,
            )
        binder.require(
            root / tasks[2]["deliverable"],
            "qualified_head_sha256",
            methods["qualified_head_sha256"],
            canonical_hash(methods["qualified_head"]),
        )
        binder.require(
            root / tasks[2]["deliverable"],
            "source_manifest_roles",
            ["evaluation", "fit", "retention", "stream", "tune"],
            sorted(methods["role_manifests"]),
        )
        qualified["heads"] = [
            binder.bind(Path(r["path"]), r["sha256"]) for r in fitted["head_checkpoints"]
        ]
        qualified["source_manifests"] = {
            role: binder.bind(Path(ref["path"]), ref["sha256"])
            for role, ref in methods["role_manifests"].items()
        }
        qualified["exposure_inventory"] = methods["exposure_rows"]
        qualified["boards"] = [
            binder.bind(root / row["source_path"], row["source_hash"])
            for row in board["board_rows"]
        ]
        qualified["initial_head"] = methods["qualified_head"]
        qualified["initial_head_sha256"] = methods["qualified_head_sha256"]
    except (InputFailure, OSError, ValueError, KeyError, TypeError) as error:
        ready = False
        binder.failures.append(
            failure(capstone_path, "qualified_inputs_readable", True, str(error), TASK)
        )
    diagnostic = {
        k: cached[8071].get(k)
        for k in [
            "forward_pass_counts",
            "arm_passed",
            "duplicate_drift_rows",
            "model_invocation_counts",
            "MODEL_SPECS",
            "model_build_hashes",
        ]
    }
    return dict(
        historical_dispositions=rows,
        historical_inputs_ready=ready,
        qualified=qualified,
        diagnostic_8071=diagnostic,
    )


def measure(
    root: Path, design: Path, staged: Path, active: Path, raw: Path, *, fixture: bool = False
) -> Json:
    """Observe external prerequisites before reducing saved history; no model is loaded."""
    started = time.monotonic_ns()
    binder = Binder(raw / "inputs")
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
        "scripts/experiments/experiment_8070_v699_contract_custody.py",
    ]
    for path in ([] if fixture else [root / n for n in names]) + [
        ROOT / ".venv/bin" / n for n in ["python", "pytest", "coverage", "ruff", "mypy"]
    ]:
        try:
            binder.bind(path)
        except InputFailure:
            pass
    binder.observations.append(
        dict(
            check="environment",
            upstream=TASK,
            path=str(root),
            hash=None,
            field="environment",
            op="observed",
            expected="record actual environment",
            observed=dict(
                python=os.sys.version,
                executable=os.sys.executable,
                PYTHONUNBUFFERED=os.environ.get("PYTHONUNBUFFERED"),
                JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS"),
            ),
            passed=True,
        )
    )
    progress("preconditions_after", len(binder.observations))
    contract = assess(design, staged, active, raw / "authority")
    binder.observations.extend(contract["gate_check_summary"])
    progress(
        "authority_after",
        sum(r["matched"] for r in contract["contract_rows"]),
        14 - len(contract["contract_rows"]),
    )
    try:
        history = historical(root, binder, expected_hash=None if fixture else CAPSTONE_HASH)
    except (InputFailure, OSError, ValueError, KeyError, TypeError) as error:
        if not binder.failures:
            binder.failures.append(
                failure(
                    root / "results/experiment_8082_v699_capstone.json",
                    "historical_readable",
                    True,
                    str(error),
                    TASK,
                )
            )
        history = dict(
            historical_dispositions=[],
            historical_inputs_ready=False,
            qualified={},
            diagnostic_8071={},
        )
    progress(
        "historical_after",
        len(history["historical_dispositions"]),
        13 - len(history["historical_dispositions"]),
    )
    missing = "python/carnot/experiment_8067_v698_arc_supervisor_frontier.py"
    actual = "python/carnot/reporting/arc_supervisor_v698_frontier.py"
    ended = time.monotonic_ns()
    return dict(
        contract=contract,
        history=history,
        refs=binder.refs,
        preconditions_checked=binder.observations,
        failures=binder.failures + contract["gate_check_summary"],
        started_ns=started,
        ended_ns=ended,
        owner_pid=os.getpid(),
        fixture=fixture,
        phase_spans=[dict(phase="custody", start_s=0, end_s=(ended - started) / 1e9)],
        named_input_resolution=[
            dict(
                requested=missing,
                requested_exists=(root / missing).is_file(),
                shipped_helper=actual,
                shipped_helper_exists=(root / actual).is_file(),
                shipped_cli="scripts/experiments/experiment_8067_v698_arc_supervisor_frontier.py",
                evidence_scope="path availability only; historical Exp8080 remains blocked",
            )
        ],
    )


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Rebuild readiness from owned checks and separate historical availability."""
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
    for i in range(14):
        original = contract["contract_rows"][i] if i < len(contract["contract_rows"]) else {}
        checks = original.get("checks", dict(executable_authority_available=False))
        matched = all(checks.values())
        rows.append(
            dict(
                source="V700_authority",
                unit=f"exp{8083 + i}",
                unit_id=f"exp{8083 + i}",
                arm="contract_custody",
                condition="full_executable_contract",
                seed=None,
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
        experiment_id=8083,
        experiment=8083,
        task_id=TASK,
        milestone=MILESTONE,
        schema="carnot.experiment.v1",
        run_date="20261004",
        status="complete",
        title="V700 contract custody",
        honest_verdict="complete_contract_custody"
        if verdict == "circular_positive"
        else "complete_"
        + verdict
        + "_"
        + (Path(failures[0]["path"]).name.replace(".", "_") if failures else "owned_validation"),
        verdict_class=verdict,
        verifier_is_oracle=True,
        claim_scope="administrative authority and historical controls only; zero fresh scientific outcomes",
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
        current_work_receipt=current,
        duration_s=current["duration_s"],
        trained_head_specs=[
            dict(
                scope="historical_control_only",
                current_training_operations=0,
                sha256=history["qualified"].get("initial_head_sha256"),
            )
        ],
        rows=rows,
        intended_count=14,
        eligible_count=sum(not r["excluded"] for r in rows),
        independent_count=0,
        completed_count=14,
        excluded_count=sum(r["excluded"] for r in rows),
        censored_count=0,
        failed_count=0,
        sample_size_budget=dict(
            administrative_tasks=14, historical_tasks=13, independent_scientific_units=0
        ),
        random_seed=70083,
        reproducibility_checksum=canonical_hash(work),
        source_artifact_hashes=work["refs"],
        raw_shard_hashes=[
            dict(path=str(raw / n), sha256=sha256_file(raw / n))
            for n in ["work.json", "validation_commands.json"]
        ],
        code_config_hashes=work["code_hashes"],
        checkpoint_hashes=history["qualified"].get("heads", []),
        phase_spans=work["phase_spans"],
        exposure_scope="recorded historical inventory only; unknown prior access and pretraining exposure remain unknown",
        generalized_learning_benefit_score=0,
        contract_ready_score=int(owned and contract["activated"]),
        historical_inputs_ready_score=int(owned and history["historical_inputs_ready"]),
        canonical_tasks_sha256=contract["canonical_tasks_sha256"],
        authority_snapshots=contract["authority_snapshots"],
        task_contract=contract["tasks"],
        historical_dispositions=history["historical_dispositions"],
        diagnostic_8071=history["diagnostic_8071"],
        authenticated_historical_inputs=history["qualified"],
        named_input_resolution=work["named_input_resolution"],
        substrate_declaration="aggregation_from_upstream_artifacts",
        repository_health=work.get("repository_health", []),
        coverage_statement_counts=work.get("coverage_statement_counts", {}),
        fixture_validation_scope=work["fixture"],
        methodology_note="Independently hash sealed authority and historical controls; fixture matching is not independent model benefit.",
    )
    value["field_principles"] = {
        k: f"Preserve {k} so administrative custody cannot imply fresh scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        honest_verdict="Finished negative science does not trigger retries.",
        verdict_class="External missing inputs are terminal blocked; owned validation failure disqualifies.",
        historical_inputs_ready_score="Historical availability is separate from current authority validity.",
        diagnostic_8071="Historical 64 forwards and failed provenance stay historical, never current invocations.",
        generalized_learning_benefit_score="Recorded-history separation does not prove absence of unknown exposure or general lifelong benefit.",
    )
    return value


def historical_replay(work: Json) -> bool:
    """Reduce dispositions and exposure directly from preserved primary bytes."""
    history = work["history"]
    rows = history["historical_dispositions"]
    if not rows:
        return not history["historical_inputs_ready"] and not history["qualified"]
    refs = {r["path"]: r for r in work["refs"]}

    def saved(path: str) -> Json:
        return dict(json.loads(Path(refs[path]["snapshot_path"]).read_text()))

    capstone_ref = next(
        r for r in work["refs"] if Path(r["path"]).name == "experiment_8082_v699_capstone.json"
    )
    capstone = saved(capstone_ref["path"])
    root = Path(capstone_ref["path"]).parents[1]
    old = {r["task_id"]: r for r in capstone["task_dispositions"]}
    cached = {}
    expected = []
    for task in capstone["task_contract"]:
        path = str(root / task["deliverable"])
        primary = saved(path)
        terminal = saved(primary["terminal_validation_sidecar_path"])
        report = saved(terminal.get("publication", terminal)["sidecar_path"])["report"]
        expected.append(
            dict(
                task_id=task["id"],
                path=path,
                sha256=refs[path]["sha256"],
                verdict_class=primary["verdict_class"],
                honest_verdict=primary["honest_verdict"],
                original_disposition=old[task["id"]],
                no_retry_unchanged_outcome=True,
                historical_validation_passed=report.get("passed"),
                terminal_validation_sidecar_path=primary["terminal_validation_sidecar_path"],
            )
        )
        if primary["experiment_id"] in [8071, 8072]:
            cached[primary["experiment_id"]] = primary
    qualified = history["qualified"]
    return (
        expected == rows
        and history["diagnostic_8071"]
        == {
            k: cached[8071].get(k)
            for k in [
                "forward_pass_counts",
                "arm_passed",
                "duplicate_drift_rows",
                "model_invocation_counts",
                "MODEL_SPECS",
                "model_build_hashes",
            ]
        }
        and (
            not history["historical_inputs_ready"]
            or (
                qualified["initial_head"] == cached[8072]["qualified_head"]
                and qualified["initial_head_sha256"]
                == canonical_hash(cached[8072]["qualified_head"])
                and qualified["exposure_inventory"] == cached[8072]["exposure_rows"]
            )
        )
    )


def replay(path: Path) -> bool:
    """Recompute from sealed bytes without requiring mutable live source paths."""
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
        with __import__("tempfile").TemporaryDirectory(prefix="carnot-8083-replay-") as directory:
            paths = [
                Path(snapshots[k].get("snapshot_path", Path(directory) / k))
                for k in ["design", "staged", "active"]
            ]
            rebuilt = assess(*paths, Path(directory) / "assessment")
            for field in ["activated", "contract_rows", "tasks", "canonical_tasks_sha256"]:
                if rebuilt[field] != work["contract"][field]:
                    return False
        return historical_replay(work) and build(work, raw, value["validation_receipts"]) == value
    except (OSError, ValueError, KeyError, TypeError):
        return False
