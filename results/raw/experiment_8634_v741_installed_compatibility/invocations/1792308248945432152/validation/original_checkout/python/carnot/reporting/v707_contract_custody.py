"""REQ-REPORT-8178: bind current scheduling without laundering old evidence.

The current contract can be ready while V706's evidence remains blocked. Both
scores bind to saved bytes and normal owned checks, with zero model execution.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any

import yaml

from carnot.reporting import v700_contract_custody as base
from carnot.reporting import v702_contract_custody as frozen
from carnot.reporting import v706_contract_context as context
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v707_contract_history import SKIPS, PRESERVED, capture, historical, literature

Json = dict[str, Any]
ROOT = base.ROOT
Binder = base.Binder
authority = base.authority
NAME = "experiment_8178_v707_contract_custody"
TASK = "exp8178-contract-custody"
MILESTONE = "2026.10.707"
RUN_DATE = "20261006"
DESIGN = base.DESIGN
MODULE = "python/carnot/reporting/v707_contract_custody.py"
HISTORY = "python/carnot/reporting/v707_contract_history.py"
RUNNER = base.RUNNER
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_contract_custody_8178.py"
COVERAGE_PATHS = [MODULE, HISTORY, CLI]
CHECK_PATHS = COVERAGE_PATHS
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose actual completed work so a quiet child never implies model activity."""
    print(f"[exp8178] phase={phase} completed={completed} pending={pending}", flush=True)


def assess(design: Path, staged: Path, active: Path, raw: Path) -> Json:
    """Delegate complete task and table checks to the shipped immutable reader."""
    return base.assess(
        design, staged, active, raw, milestone=MILESTONE, first_id=8178, count=14, task=TASK
    )


def mutation_controls(contract: Json) -> list[Json]:
    """Change real executable operands privately so short titles cannot mask drift."""
    if not contract["activated"]:
        return []
    names = [
        "id",
        "title",
        "phase",
        "deliverable",
        "MODEL_SPECS",
        "inference_substrate_class",
        "prompt",
        "gated_on",
        "prior_failures",
        "omitted",
        "extra",
        "reordered",
        "stale_milestone",
    ]
    rows = []
    with TemporaryDirectory(prefix="carnot-8178-mutations-") as directory:
        private = Path(directory)
        design = Path(contract["authority_snapshots"]["design"]["snapshot_path"])
        for index, name in enumerate(names):
            progress("mutation_before", index, len(names) - index)
            tasks = deepcopy(contract["tasks"])
            if name == "omitted":
                tasks.pop()
            elif name == "extra":
                tasks.append(deepcopy(tasks[0]))
            elif name == "reordered":
                tasks.reverse()
            elif name != "stale_milestone":
                tasks[0][name] = "changed"
            active = private / "active.yaml"
            active.write_text(
                yaml.safe_dump(
                    dict(milestone="old" if name == "stale_milestone" else MILESTONE, tasks=tasks)
                )
            )
            observed = assess(design, private / "absent.yaml", active, private / "snapshots")
            rows.append(
                dict(
                    unit_id=name,
                    arm="authority_mutation",
                    status="completed",
                    expected_activation=False,
                    observed_activation=observed["activated"],
                    passed=not observed["activated"],
                )
            )
            progress("mutation_after", index + 1, len(names) - index - 1)
    return rows


def measure(
    root: Path, design: Path, staged: Path, active: Path, raw: Path, *, fixture: bool = False
) -> Json:
    """Freeze authority, history and current prerequisites before any validation child."""
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
        "ops/exclusion_manifest.yaml",
        MODULE,
        HISTORY,
        CLI,
        TEST,
        RUNNER,
    ]
    paths = ([] if fixture else [root / name for name in names]) + [
        ROOT / ".venv/bin" / n for n in ["python", "pytest", "coverage", "ruff", "mypy"]
    ]
    for path in paths:
        capture(binder, path)
    boundary = time.monotonic_ns()
    progress("preconditions_after", len(paths), 0)
    contract = assess(design, staged, active, raw / "authority")
    history_start = time.monotonic_ns()
    progress("authority_after", len(contract["contract_rows"]), 14 - len(contract["contract_rows"]))
    past = historical(root, binder)
    context.validate_producers(contract["tasks"], active, binder)
    ledger_history = dict(
        past,
        historical_dispositions=past["historical_dispositions"] + past["older_scope_dispositions"],
    )
    past["prior_scope_ledger"] = context.scope_ledger(
        root, contract["tasks"], ledger_history, binder
    )
    past["literature_mapping"] = literature(root, binder, contract)
    mutations = mutation_controls(contract)
    atomic_json(
        raw / "primitive_rows.json",
        dict(
            authority_rows=contract["contract_rows"],
            historical_dispositions=past["historical_dispositions"],
            mutation_rows=mutations,
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
    return dict(
        contract=contract,
        history=past,
        mutation_rows=mutations,
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
    """Use the qualified schema and independently reduce current and old readiness."""
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
    for index, row in enumerate(value["rows"]):
        row.update(
            unit_id=f"exp{8178 + index}",
            unit=f"exp{8178 + index}",
            source="V707_authority",
            source_id="V707_authority",
            source_cluster_id="V707_authority",
            metric="authority_checks",
        )
    history = work["history"]
    value.update(
        experiment_id=8178,
        experiment=8178,
        task_id=TASK,
        milestone=MILESTONE,
        run_date=RUN_DATE,
        title="V707 immutable contract custody",
        random_seed=70778,
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
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        historical_evidence_ready_score=int(
            owned
            and not history["historical_hash_failures"]
            and not work["failures"]
            and len(history["historical_dispositions"]) == 14
        ),
        task_dispositions=history["historical_dispositions"],
        historical_hash_failures=history["historical_hash_failures"],
        historical_authority_snapshots=history["authorities"],
        mutation_rows=work["mutation_rows"],
        immutable_authority_paths={
            role: r.get("snapshot_path") for role, r in value["authority_snapshots"].items()
        },
        prior_scope_ledger=history["prior_scope_ledger"],
        literature_mapping=history["literature_mapping"],
        sample_size_budget=dict(
            administrative_tasks=14,
            historical_primaries=11,
            historical_gate_skips=3,
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
                    "historical_model_invocation_counts",
                ],
            )
            for r in history["historical_dispositions"]
        ],
        methodology_note="Bind full V707 executable authority and reduce fourteen administrative rows. Preserve eleven V706 primaries, three gate skips and original historical hash failures. No model, training, scientific benefit or service benchmark is measured.",
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
        contract="Full task agreement and normal owned checks qualify current scheduling.",
        history="An original historical failure remains blocked even if a new capture succeeds.",
        science="No scientific or independent generalization credit.",
        publication="Normal terminal validators precede atomic publication.",
    )
    value["field_principles"] = {
        k: f"Record {k} so current custody cannot repair historical provenance or imply scientific benefit."
        for k in value
    }
    return value


def replay(path: Path) -> bool:
    """Rebuild current authority and historical operands from authenticated copies.

    A mutable live spec may change after capture. A frozen source, log or headline
    may not change: replay checks its hash and redoes the administrative reduction.
    """
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
        with TemporaryDirectory(prefix="carnot-8178-replay-") as directory:
            private = Path(directory)
            snaps = value["authority_snapshots"]
            paths = [
                Path(snaps[k].get("snapshot_path", private / k))
                for k in ["design", "staged", "active"]
            ]
            contract = assess(*paths, private / "authority")
            for field in ["activated", "contract_rows", "tasks", "canonical_tasks_sha256"]:
                if contract[field] != work["contract"][field]:
                    return False
            if mutation_controls(contract) != work["mutation_rows"]:
                return False
            binder = frozen.FrozenBinder(private / "inputs", work["refs"])
            history = historical(Path(work["root"]), binder)
            ledger_history = dict(
                history,
                historical_dispositions=history["historical_dispositions"]
                + history["older_scope_dispositions"],
            )
            history["prior_scope_ledger"] = context.scope_ledger(
                Path(work["root"]), contract["tasks"], ledger_history, binder
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
                mutation_rows=work["mutation_rows"],
            )
            and build(work, raw, value["validation_receipts"]) == value
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
