"""REQ-REPORT-8164: preserve scheduling evidence without inventing science.

A conductor skip records an administrative outcome. Only a published producer
can supply its scientific verdict. These two records must remain distinct.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any

import yaml

from carnot.reporting import v703_contract_custody as previous
from carnot.reporting import v706_contract_context as context
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v698_fixture_consumer_contract import failure

Json = dict[str, Any]
ROOT = previous.ROOT
Binder = previous.Binder
authority = previous.authority
NAME = "experiment_8164_v706_contract_custody"
TASK = "exp8164-contract-custody"
MILESTONE = "2026.10.706"
RUN_DATE = "20261005"
DESIGN = previous.DESIGN
MODULE = "python/carnot/reporting/v706_contract_custody.py"
RUNNER = previous.RUNNER
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_contract_custody_8164.py"
COVERAGE_PATHS = [MODULE, CLI, "python/carnot/reporting/v706_contract_context.py"]
CHECK_PATHS = COVERAGE_PATHS
MODEL_SPECS: list[Json] = []
PRESERVED = "openspec/change-proposals/research-roadmap-v705-preserved-20261005.md"
PRIMARIES = set(range(8150, 8164)) - {8157, 8158}
SKIPS = {8157, 8158}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts so quiet custody work cannot imply model execution."""
    print(f"[exp8164] phase={phase} completed={completed} pending={pending}", flush=True)


def assess(design: Path, staged: Path, active: Path, raw: Path) -> Json:
    """Use the shipped reader so all executable task fields bind to activation."""
    return previous.base.assess(
        design, staged, active, raw, milestone=MILESTONE, first_id=8164, count=14, task=TASK
    )


def mutation_controls(contract: Json) -> list[Json]:
    """Challenge twelve executable operands using private planning copies only."""
    if not contract["activated"]:
        return []
    rows = []
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
    with TemporaryDirectory(prefix="carnot-8164-mutations-") as directory:
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
                    dict(
                        milestone="2026.10.705" if name == "stale_milestone" else MILESTONE,
                        tasks=tasks,
                    )
                )
            )
            observed = assess(design, private / "consumed.yaml", active, private / "snapshots")
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


def historical(root: Path, binder: Any, *, fixture: bool) -> Json:
    """Freeze terminal evidence without turning historical failure into today's failure.

    A gate skip has no scientific verdict. Saved primaries and their validator
    bytes remain available even when the original science was disqualified.
    """
    result: Json = dict(
        historical_dispositions=[],
        qualified_inputs={},
        authorities={},
        kernel_ready=False,
        controls_ready=False,
        public_kernel={},
        historical_controls=[],
    )
    admin = root / (
        "results/experiment_8150_fixture.json"
        if fixture
        else "results/experiment_8150_v705_contract_custody.json"
    )
    try:
        receipt = binder.read(admin)
        for role in ["active", "design"]:
            ref = receipt["authority_snapshots"][role]
            result["authorities"][role] = binder.bind(Path(ref["snapshot_path"]), ref["sha256"])
        result["authorities"]["preserved_design"] = binder.bind(root / PRESERVED)
        active = yaml.safe_load(Path(result["authorities"]["active"]["snapshot_path"]).read_bytes())
        tasks = active["tasks"]
        binder.require(admin, "historical_milestone", "2026.10.705", active["milestone"])
        binder.require(
            admin,
            "historical_tasks_digest",
            receipt["canonical_tasks_sha256"],
            authority.tasks_digest(tasks),
        )
        binder.require(
            admin,
            "historical_sequence",
            list(range(8150, 8164)),
            [int(t["id"].split("-")[0][3:]) for t in tasks],
        )
        log_path = root / "ops/conductor-log.md"
        log = Path(binder.bind(log_path)["snapshot_path"]).read_text()
        changelog = Path(binder.bind(root / "ops/changelog.md")["snapshot_path"]).read_text()
        values = {
            t["id"]: binder.read(root / t["deliverable"])
            for n, t in zip(range(8150, 8164), tasks)
            if n in PRIMARIES
        }
        for index, t in enumerate(tasks):
            n = 8150 + index
            progress("historical_before", index, 14 - index)
            path = root / t["deliverable"]
            lines = [s for s in log.splitlines() if t["title"][:48] in s]
            statuses = [
                s.split("|")[3].strip() if len(s.split("|")) > 5 else s.split("|")[2].strip()
                for s in lines
            ]
            expected = "GATE_BLOCK" if n in SKIPS else "OK"
            binder.require(log_path, f"conductor_disposition_{n}", True, expected in statuses)
            value = values.get(t["id"], {})
            ref = next((r for r in binder.refs if r["path"] == str(path)), None)
            side_ref = None
            validator_ref = None
            terminal = {}
            if n in PRIMARIES:
                binder.require(path, "task_id", t["id"], value.get("task_id"))
                side_path = Path(value["terminal_validation_sidecar_path"])
                terminal = binder.read(side_path)
                side_ref = next(r for r in binder.refs if r["path"] == str(side_path))
                publication = terminal.get("publication", terminal)
                binder.require(
                    side_path, "primary_sha256", ref["sha256"], publication.get("primary_sha256")
                )
                validator_path = Path(publication["sidecar_path"])
                validator_ref = binder.bind(validator_path)
                report = binder.read(validator_path)
                context.bind_logs(binder, report)
                context.bind_logs(binder, terminal)
            context.bind_logs(binder, value)
            shards = context.hash_refs(value)
            for completed, shard in enumerate(shards):
                if completed % 64 == 0:
                    progress("historical_shards", completed, len(shards) - completed)
                source = Path(shard.get("snapshot_path", shard.get("path", "")))
                try:
                    binder.bind(source if source.is_absolute() else root / source, shard["sha256"])
                except (OSError, ValueError, KeyError):
                    continue
            gates = []
            for gate in t.get("gated_on", []):
                upstream = next(task for task in tasks if task["id"] == gate["upstream"])
                operand = failure(
                    root / upstream["deliverable"],
                    gate["artifact_field"],
                    gate["value"],
                    values.get(gate["upstream"], {}).get(gate["artifact_field"]),
                    gate["upstream"],
                )
                operand.update(op=gate["op"], artifact_field=gate["artifact_field"])
                gates.append(operand)
            if n not in PRIMARIES:
                gates.append(failure(path, "primary_exists", True, False, t["id"]))
            logged_rows = [s for s in changelog.splitlines() if Path(t["deliverable"]).name in s]
            logged_verdicts = [
                match
                for line in logged_rows
                for match in re.findall(r"honest_verdict=(complete_[a-zA-Z0-9_]+)", line)
            ]
            result["historical_dispositions"].append(
                dict(
                    task_id=t["id"],
                    experiment_id=n,
                    path=str(path),
                    sha256=ref["sha256"] if ref else None,
                    primary_present=n in PRIMARIES,
                    disposition="primary" if n in PRIMARIES else "gate_skipped",
                    conductor_statuses=statuses,
                    conductor_log_rows=lines,
                    honest_verdict=value.get("honest_verdict"),
                    current_primary_verdict=value.get("honest_verdict"),
                    verdict_class=value.get("verdict_class"),
                    required_checks_passed=value.get("required_checks_passed"),
                    flagged_adversarial=value.get("flagged_adversarial"),
                    gate_check_summary=value.get("gate_check_summary", gates),
                    earlier_logged_verdicts=logged_verdicts,
                    earlier_logged_rows=logged_rows,
                    validation_sidecar_snapshot=side_ref,
                    validator_snapshot=validator_ref,
                    historical_owned_checks_passed=terminal.get(
                        "owned_checks_passed", terminal.get("required_checks_passed")
                    ),
                    failed_receipts=[
                        r for r in value.get("validation_receipts", []) if not r["passed"]
                    ],
                    historical_MODEL_SPECS=value.get("MODEL_SPECS", []),
                    historical_model_invocation_counts=value.get("model_invocation_counts", {}),
                    no_retry_unchanged_outcome=True,
                )
            )
            progress("historical_after", index + 1, 13 - index)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        binder.failures.append(
            failure(admin, "historical_authority_readable", True, str(error), TASK)
        )
    return result


def measure(
    root: Path, design: Path, staged: Path, active: Path, raw: Path, *, fixture: bool = False
) -> Json:
    """Freeze prerequisites and real phase times without loading a model."""
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
        "python/carnot/reporting/current_work_receipt.py",
        "python/carnot/reporting/primary_publication.py",
        "python/carnot/reporting/roadmap_contract.py",
        "python/carnot/reporting/v685_authority_lifecycle.py",
        "ops/exclusion_manifest.yaml",
        MODULE,
        "python/carnot/reporting/v706_contract_context.py",
        CLI,
        TEST,
        RUNNER,
    ]
    paths = [] if fixture else [root / n for n in names]
    paths += [ROOT / ".venv/bin" / n for n in ["python", "pytest", "coverage", "ruff", "mypy"]]
    for path in paths:
        try:
            binder.bind(path)
        except previous.previous.InputFailure:
            pass
    boundary = time.monotonic_ns()
    progress("preconditions_after", len(paths), 0)
    progress("authority_before")
    contract = assess(design, staged, active, raw / "authority")
    history_start = time.monotonic_ns()
    progress("authority_after", len(contract["contract_rows"]), 14 - len(contract["contract_rows"]))
    history = historical(root, binder, fixture=fixture)
    context.validate_producers(contract["tasks"], active, binder)
    history["prior_scope_ledger"] = context.scope_ledger(root, contract["tasks"], history, binder)
    history["literature_mapping"] = context.literature(root, binder, contract)
    mutations = mutation_controls(contract)
    atomic_json(
        raw / "primitive_rows.json",
        dict(
            authority_rows=contract["contract_rows"],
            historical_dispositions=history["historical_dispositions"],
            mutation_rows=mutations,
        ),
    )
    end = time.monotonic_ns()
    boundaries = [start, boundary, history_start, end]
    return dict(
        contract=contract,
        history=history,
        mutation_rows=mutations,
        refs=binder.refs,
        preconditions_checked=binder.observations + contract["gate_check_summary"],
        failures=binder.failures + contract["gate_check_summary"],
        started_ns=start,
        ended_ns=end,
        owner_pid=os.getpid(),
        fixture=fixture,
        root=str(root),
        phase_spans=[
            dict(
                phase=name,
                start_s=(boundaries[i] - start) / 1e9,
                end_s=(boundaries[i + 1] - start) / 1e9,
            )
            for i, name in enumerate(["preconditions", "authority", "historical_custody"])
        ],
    )


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Reuse the qualified schema; reduce fourteen administrative units only."""
    value = previous.build(work, raw, receipts)
    template = value["rows"][0]
    rows = []
    for i in range(14):
        original = (
            work["contract"]["contract_rows"][i]
            if i < len(work["contract"]["contract_rows"])
            else {}
        )
        checks = original.get("checks", dict(executable_authority_available=False))
        matched = all(checks.values())
        rows.append(
            dict(
                template,
                unit_id=f"exp{8164 + i}",
                source="V706_authority",
                source_id="V706_authority",
                source_cluster_id="V706_authority",
                checks=checks,
                absolute_metric=int(matched),
                numerator=sum(checks.values()),
                denominator=len(checks),
                raw_numerator=sum(checks.values()),
                raw_denominator=len(checks),
                excluded=not matched,
                exclusion_reason=None if matched else "external_authority_mismatch",
            )
        )
    value.update(
        experiment_id=8164,
        experiment=8164,
        task_id=TASK,
        milestone=MILESTONE,
        title="V706 contract custody",
        mutation_rows=work["mutation_rows"],
        rows=rows,
        intended_count=14,
        completed_count=14,
        eligible_count=sum(not r["excluded"] for r in rows),
        excluded_count=sum(r["excluded"] for r in rows),
        random_seed=70650,
        run_date=RUN_DATE,
        prior_scope_ledger=work["history"]["prior_scope_ledger"],
        literature_mapping=work["history"]["literature_mapping"],
        verifier_is_oracle=0,
        task_dispositions=work["history"]["historical_dispositions"],
        sample_size_budget=dict(
            administrative_tasks=14,
            historical_primaries=12,
            historical_gate_skips=2,
            independent_scientific_units=0,
        ),
        methodology_note="Compare complete V706 tasks with strict authority parsing. Preserve twelve V705 primaries and two conductor skips. No science, model, training or service benchmark is measured.",
    )
    value["acceptance_gates"] = dict(
        contract="Full activated task agreement plus normal owned validation qualifies scheduling.",
        history="Fourteen authentic dispositions preserve absent scientific verdicts.",
        publication="Normal bound terminal validators precede primary publication.",
    )
    value["cited_upstream_artifacts"] = [
        dict(
            r,
            experiment_id=r["experiment_id"],
            fields_imported=[
                "honest_verdict",
                "verdict_class",
                "failed_receipts",
                "historical_MODEL_SPECS",
                "historical_model_invocation_counts",
            ],
        )
        for r in work["history"]["historical_dispositions"]
    ]
    value["field_principles"] = {
        k: f"Record {k} so administrative custody cannot imply scientific benefit." for k in value
    }
    value["field_principles"].update(
        contract_ready_score="Administrative authority agreement is independent of every research branch.",
        historical_dispositions="Absent producer verdicts remain absent; conductor statuses are administrative.",
        cited_upstream_artifacts="Historical Qwen identity and calls never enter current invocation evidence.",
    )
    return value


def replay(path: Path) -> bool:
    """Rebuild frozen operands independently so rehashed aggregates cannot pass."""
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
        with TemporaryDirectory(prefix="carnot-8164-replay-") as directory:
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
            binder = previous.previous.FrozenBinder(private / "inputs", work["refs"])
            history = historical(Path(work["root"]), binder, fixture=work["fixture"])
            history["prior_scope_ledger"] = context.scope_ledger(
                Path(work["root"]), contract["tasks"], history, binder
            )
            history["literature_mapping"] = context.literature(Path(work["root"]), binder, contract)
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
