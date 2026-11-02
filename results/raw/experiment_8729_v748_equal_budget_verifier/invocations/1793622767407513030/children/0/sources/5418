"""REQ-REPORT-8192: bind current scheduling while preserving failed old evidence.

Administrative matching establishes a readable contract. It cannot qualify
historical science, independent learning or a disqualified numerical fixture.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
import sys
import time
from typing import Any

import yaml

from carnot.reporting import v700_contract_custody as base
from carnot.reporting import v702_contract_custody as frozen
from carnot.reporting import v706_contract_context as context
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v700_custody_execution import main as execute
from carnot.reporting.v698_fixture_consumer_contract import failure
from carnot.reporting.v708_contract_history import (
    ADMIN,
    PRESERVED,
    SKIPS,
    capture,
    historical,
    literature,
)

Json = dict[str, Any]
ROOT = base.ROOT
Binder = base.Binder
authority = base.authority
NAME = "experiment_8192_v708_contract_custody"
TASK = "exp8192-contract-custody"
MILESTONE = "2026.10.708"
RUN_DATE = "20261006"
DESIGN = base.DESIGN
MODULE = "python/carnot/reporting/v708_contract_custody.py"
HISTORY = "python/carnot/reporting/v708_contract_history.py"
RUNNER = base.RUNNER
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_contract_custody_8192.py"
COVERAGE_PATHS = [MODULE, HISTORY, CLI]
CHECK_PATHS = COVERAGE_PATHS
MODEL_SPECS: list[Json] = []
PHASE_STATE = ("start", 0, 1)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush measured work counts so a quiet child never implies model activity."""
    global PHASE_STATE
    pending = pending or int(phase.startswith("before_"))
    PHASE_STATE = (phase, completed, pending)
    print(f"[exp8192] phase={phase} completed={completed} pending={pending}", flush=True)


def run(argv: list[str] | None = None, *, heartbeat_s: float = 30) -> int:
    """Keep actual pending validation visible while the qualified runner waits."""
    stop = Event()

    def heartbeat() -> None:
        while not stop.wait(heartbeat_s):
            phase, completed, pending = PHASE_STATE
            print(
                f"[exp8192] phase={phase}_child_wait completed={completed} pending={pending}",
                flush=True,
            )

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        return execute(argv, experiment=sys.modules[__name__])
    finally:
        stop.set()
        thread.join()


def assess(design: Path, staged: Path, active: Path, raw: Path) -> Json:
    """Use the qualified reader and validate staging privately before activation."""
    observed_staged = staged if active.is_file() else raw / "absent-staging"
    result = base.assess(
        design,
        observed_staged,
        active,
        raw,
        milestone=MILESTONE,
        first_id=8192,
        count=13,
        task=TASK,
    )
    result["authority_snapshots"]["staged"] = authority._snapshot(
        staged, staged.read_bytes() if staged.is_file() else None, raw, "staged"
    )
    if not result["tasks"] and active.is_file():
        observed = yaml.safe_load(active.read_bytes())
        if isinstance(observed, dict) and isinstance(observed.get("tasks"), list):
            result["tasks"] = observed["tasks"]
            result["canonical_tasks_sha256"] = authority.tasks_digest(observed["tasks"])
    if design.is_file() and "## Exact task contract" not in design.read_text():
        result["gate_check_summary"] = [
            failure(design, "design_exact_task_contract", True, False, TASK)
        ]
    result["private_activation_validated"] = False
    if result["tasks"] and result["authority_snapshots"]["staged"]["exists"]:
        saved = Path(result["authority_snapshots"]["staged"]["snapshot_path"])
        private = base.assess(
            design,
            saved,
            saved,
            raw / "private_activation",
            milestone=MILESTONE,
            first_id=8192,
            count=13,
            task=TASK,
        )
        result["private_activation_validated"] = private["activated"]
        result["planning_matched"] = private.get("planning_matched", False)
        result["staged_tasks_sha256"] = private.get("staged_tasks_sha256")
    return result


def measure(
    root: Path, design: Path, staged: Path, active: Path, raw: Path, *, fixture: bool = False
) -> Json:
    """Check runtime, writable private storage and frozen inputs before reduction."""
    start = time.monotonic_ns()
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
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/v685_authority_lifecycle.py",
        "python/carnot/reporting/roadmap_contract.py",
        "python/carnot/reporting/v707_contract_custody.py",
        "ops/exclusion_manifest.yaml",
        MODULE,
        HISTORY,
        CLI,
        TEST,
        RUNNER,
    ]
    paths = ([] if fixture else [root / name for name in names]) + [
        ROOT / ".venv/bin" / name for name in ["python", "pytest", "coverage", "ruff", "mypy"]
    ]
    for path in paths:
        capture(binder, path)
    with TemporaryDirectory(prefix="carnot-8192-storage-") as directory:
        private = Path(directory)
        probe = private / "write-read-check"
        probe.write_bytes(b"actual writable private storage")
        binder.require(
            private,
            "private_writable_storage",
            True,
            probe.read_bytes() == b"actual writable private storage"
            and private.stat().st_mode & 0o077 == 0,
        )
    binder.observations.append(
        dict(
            check="runtime",
            upstream=TASK,
            path=str(root),
            hash=None,
            artifact_field="runtime",
            op="observed",
            expected="record actual runtime",
            passed=True,
            observed=dict(
                python=os.sys.version,
                executable=os.sys.executable,
                PYTHONUNBUFFERED=os.environ.get("PYTHONUNBUFFERED"),
                JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS"),
            ),
        )
    )
    boundary = time.monotonic_ns()
    progress("preconditions_after", len(paths), 0)
    contract = assess(design, staged, active, raw / "authority")
    history_start = time.monotonic_ns()
    progress("authority_after", len(contract["contract_rows"]), 13 - len(contract["contract_rows"]))
    past = historical(root, binder)
    context.validate_producers(contract["tasks"], active, binder)
    ledger = dict(
        past,
        historical_dispositions=past["historical_dispositions"] + past["older_scope_dispositions"],
    )
    past["prior_scope_ledger"] += context.scope_ledger(root, contract["tasks"], ledger, binder)
    past["literature_mapping"] = literature(root, binder, contract)
    atomic_json(
        raw / "primitive_rows.json",
        dict(
            authority_rows=contract["contract_rows"],
            historical_dispositions=past["historical_dispositions"],
        ),
    )
    atomic_json(
        raw / "scope_literature.json",
        dict(
            prior_scope_ledger=past["prior_scope_ledger"],
            literature_mapping=past["literature_mapping"],
        ),
    )
    end = time.monotonic_ns()
    progress("measurement_after", 13, 0)
    return dict(
        contract=contract,
        history=past,
        refs=binder.refs,
        preconditions_checked=binder.observations + contract["gate_check_summary"],
        failures=past["historical_hash_failures"]
        + binder.failures
        + contract["gate_check_summary"],
        started_ns=start,
        ended_ns=end,
        owner_pid=os.getpid(),
        fixture=fixture,
        root=str(root),
        named_input_resolution=[],
        phase_spans=[
            dict(phase=name, start_s=(a - start) / 1e9, end_s=(b - start) / 1e9)
            for name, a, b in [
                ("preconditions", start, boundary),
                ("authority", boundary, history_start),
                ("historical_custody", history_start, end),
            ]
        ],
    )


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Independently reduce current rows while leaving failed history disqualified."""
    receipts = [
        dict(
            r,
            passed=bool(
                r["passed"] and r.get("exit_code", 0) >= 0 and not r.get("timed_out", False)
            ),
        )
        for r in receipts
    ]
    value = base.build(work, raw, receipts)
    owned = value["required_checks_passed"]
    verdict = (
        "disqualified"
        if not owned
        else "blocked"
        if value["gate_check_summary"]
        else "circular_positive"
        if work["fixture"]
        else "null"
    )
    for field in [
        "diagnostic_8071",
        "authenticated_historical_inputs",
        "checkpoint_hashes",
        "historical_inputs_ready_score",
    ]:
        value.pop(field)
    value["rows"] = value["rows"][:13]
    for index, row in enumerate(value["rows"]):
        row.update(
            unit_id=f"exp{8192 + index}",
            unit=f"exp{8192 + index}",
            source="V708_authority",
            source_id="V708_authority",
            source_cluster_id="V708_authority",
            metric="authority_checks",
        )
    history = work["history"]
    value.update(
        experiment_id=8192,
        experiment=8192,
        task_id=TASK,
        milestone=MILESTONE,
        run_date=RUN_DATE,
        title="V708 immutable contract custody",
        random_seed=70892,
        honest_verdict="complete_"
        + verdict
        + "_"
        + (
            value["gate_check_summary"][0]["check"]
            if value["gate_check_summary"]
            else "contract_custody"
        ),
        verdict_class=verdict,
        verifier_is_oracle=0,
        trained_head_specs=[],
        call_ledger=[],
        intended_count=13,
        completed_count=13,
        eligible_count=sum(not r["excluded"] for r in value["rows"]),
        excluded_count=sum(r["excluded"] for r in value["rows"]),
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        exposure_scope="exposed development; no independent benefit",
        historical_evidence_ready_score=int(
            owned
            and not history["historical_hash_failures"]
            and not work["failures"]
            and len(history["historical_dispositions"]) == 14
            and all(
                r["required_checks_passed"]
                for r in history["historical_dispositions"]
                if r["primary_present"]
            )
        ),
        task_dispositions=history["historical_dispositions"],
        historical_hash_failures=history["historical_hash_failures"],
        historical_authority_snapshots=history["authorities"],
        immutable_authority_paths={
            role: r.get("snapshot_path") for role, r in value["authority_snapshots"].items()
        },
        prior_scope_ledger=history["prior_scope_ledger"],
        literature_mapping=history["literature_mapping"],
        sample_size_budget=dict(
            administrative_tasks=13,
            historical_primaries=12,
            historical_gate_skips=2,
            independent_scientific_units=0,
        ),
        cited_upstream_artifacts=[
            dict(
                experiment_id=r["experiment_id"],
                path=r["path"],
                sha256=r["sha256"],
                fields_imported=[
                    "honest_verdict",
                    "verdict_class",
                    "failed_receipts",
                    "historical_MODEL_SPECS",
                    "historical_trained_head_specs",
                    "historical_model_invocation_counts",
                ],
            )
            for r in history["historical_dispositions"]
        ],
        methodology_note="Bind thirteen executable tasks. Preserve twelve V707 primaries, two untested gate skips, Exp8180 disqualification and historical hash failures. This is administrative custody with no model, training or scientific benefit.",
    )
    value["gate_check_summary"] = [
        dict(r, artifact_field=r.get("artifact_field", r.get("field")))
        for r in value["gate_check_summary"]
    ]
    value["raw_shard_hashes"].extend(
        dict(path=str(raw / n), sha256=sha256_file(raw / n))
        for n in ["primitive_rows.json", "scope_literature.json"]
    )
    value["acceptance_gates"] = dict(
        contract="Full task agreement and normal owned checks qualify scheduling.",
        history="New snapshots cannot repair original failed provenance.",
        science="All old cohorts are exposed development; benefit scores remain zero.",
        nfr01="PRD complete Rust/Python throughput threshold remains 10x.",
        publication="Normal unchanged terminal validators precede atomic publication.",
    )
    value["field_principles"] = {
        k: f"Record {k} so current custody cannot qualify failed historical science." for k in value
    }
    return value


def replay(path: Path) -> bool:
    """Rebuild from authenticated copies so later live spec edits remain harmless."""
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
        for receipt in value["validation_receipts"]:
            if (
                receipt.get("log_path")
                and sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
            ):
                return False
        for label, ref in work["code_snapshots"].items():
            if sha256_file(Path(ref["snapshot_path"])) != value["code_config_hashes"][label]:
                return False
        with TemporaryDirectory(prefix="carnot-8192-replay-") as directory:
            private = Path(directory)
            snaps = value["authority_snapshots"]
            paths = [
                Path(snaps[k].get("snapshot_path", private / k))
                for k in ["design", "staged", "active"]
            ]
            contract = assess(*paths, private / "authority")
            if any(
                contract[k] != work["contract"][k]
                for k in [
                    "activated",
                    "contract_rows",
                    "tasks",
                    "canonical_tasks_sha256",
                    "private_activation_validated",
                ]
            ):
                return False
            binder = frozen.FrozenBinder(private / "inputs", work["refs"])
            history = historical(Path(work["root"]), binder)
            ledger = dict(
                history,
                historical_dispositions=history["historical_dispositions"]
                + history["older_scope_dispositions"],
            )
            history["prior_scope_ledger"] += context.scope_ledger(
                Path(work["root"]), contract["tasks"], ledger, binder
            )
            history["literature_mapping"] = literature(Path(work["root"]), binder, work["contract"])
            if history != work["history"]:
                return False
        primitives = json.loads((raw / "primitive_rows.json").read_text())
        return (
            primitives
            == dict(
                authority_rows=work["contract"]["contract_rows"],
                historical_dispositions=work["history"]["historical_dispositions"],
            )
            and build(work, raw, value["validation_receipts"]) == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
