"""REQ-REPORT-8205: qualify consumers without repairing historical science.

The existing parser decides exact authority agreement. These adapters document
only the saved mapping-v1 and reference-list-v1 hash/receipt representations.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

import yaml

from carnot.reporting import v700_contract_custody as base
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.roadmap_contract import parse_design
from carnot.reporting.v709_execution import progress

Json = dict[str, Any]
ROOT = base.ROOT
DESIGN = "openspec/change-proposals/research-roadmap-vNEXT.md"
MILESTONE = "2026.10.709"
TASK = "exp8205-contract-consumer-qualification"
INPUTS = [
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "ops/e2e-test-plan.md",
    "openspec/capabilities/research-reporting/spec.md",
    "openspec/capabilities/verification/spec.md",
    "scripts/experiment_template.py",
    "python/carnot/reporting/primary_publication.py",
    "ops/exclusion_manifest.yaml",
    DESIGN,
    "research-roadmap.yaml",
    "research-complete.yaml",
    "ops/conductor-log.md",
    "python/carnot/reporting/roadmap_contract.py",
    "python/carnot/reporting/v685_authority_lifecycle.py",
    "python/carnot/reporting/v708_contract_custody.py",
    "python/carnot/reporting/v708_capstone_inputs.py",
    "python/carnot/reporting/v708_capstone.py",
    "results/experiment_8192_v708_contract_custody.json",
    "results/experiment_8204_v708_capstone.json",
    "openspec/change-proposals/research-roadmap-v708-preserved-20261006.md",
]


def hash_references(value: Any) -> list[Json]:
    """Accept documented path-to-hash mappings and path/hash reference lists only."""
    if isinstance(value, dict):
        value = [dict(path=path, sha256=digest) for path, digest in value.items()]
    if not isinstance(value, list):
        raise ValueError("hash_reference_schema")
    for row in value:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("path"), str)
            or not row["path"]
            or not isinstance(row.get("sha256"), str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", row["sha256"])
        ):
            raise ValueError("hash_reference_entry")
    return [dict(row) for row in value]


def receipts(value: Any) -> list[Json]:
    """Accept named-receipt mappings and named-receipt lists without truth coercion."""
    if isinstance(value, dict):
        if not all(isinstance(v, dict) for v in value.values()):
            raise ValueError("receipt_mapping_entry")
        value = [dict(row, name=name) for name, row in value.items()]
    if not isinstance(value, list):
        raise ValueError("receipt_schema")
    for row in value:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("name"), str)
            or not row["name"]
            or type(row.get("passed")) is not bool
        ):
            raise ValueError("receipt_entry")
    return [dict(row) for row in value]


def legacy_operand(value: Json) -> Json:
    """Execute the old list indexing on saved hashes to identify the real cause."""
    operand = value.get("code_config_hashes", [])
    try:
        for ref in operand:
            Path(ref["path"])
        return dict(typeerror_reproduced=False, operand=operand, error=None)
    except TypeError as error:
        return dict(typeerror_reproduced=True, operand=operand, error=str(error))


def assess(design: Path, staged: Path, active: Path, raw: Path) -> Json:
    """Reuse qualified complete-task authority; staging is reported independently."""
    return base.assess(
        design, staged, active, raw, milestone=MILESTONE, first_id=8205, count=13, task=TASK
    )


def mutations(design: Path, active: Path, raw: Path) -> list[Json]:
    """Change independent operands, keeping every mutation in private scratch."""
    rows = []
    original = yaml.safe_load(active.read_bytes())
    for name in ("count", "order", "title", "prompt", "gate_spelling", "digest"):
        parent = raw / name
        parent.mkdir(parents=True, exist_ok=True)
        mutated, text = deepcopy(original), design.read_text()
        if name == "count":
            mutated["tasks"].pop()
        elif name == "order":
            mutated["tasks"].reverse()
        elif name in ("title", "prompt"):
            mutated["tasks"][0][name] += " changed operand"
        elif name == "gate_spelling":
            gate = next(t for t in mutated["tasks"] if t["gated_on"])["gated_on"][0]
            gate["artifact_filed"] = gate.pop("artifact_field")
        else:
            text = re.sub(r"(Canonical full-task SHA256: `)[0-9a-f]{64}", r"\g<1>" + "0" * 64, text)
        d, a = parent / "design.md", parent / "active.yaml"
        d.write_text(text)
        a.write_text(yaml.safe_dump(mutated))
        observed = assess(d, parent / "absent", a, parent / "snapshots")
        rows.append(
            dict(
                control=name,
                rejected=not observed["activated"],
                gate_check_summary=observed["gate_check_summary"],
            )
        )
    return rows


def terminal(primary: Path, value: Json, raw: Path) -> Json:
    """Call the real byte-bound terminal reader; never replace failed reports."""
    binder = base.Binder(raw, task=TASK)
    try:
        report = binder.terminal_evidence(primary, value)
        if not isinstance(report, dict) or type(report.get("passed")) is not bool:
            raise ValueError("terminal_report_schema")
        return dict(
            passed=report["passed"],
            report=report,
            failures=binder.failures,
            references=binder.refs,
            error=None,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(
            passed=False,
            report=None,
            failures=binder.failures,
            references=binder.refs,
            error=str(error),
            failed_operand=value.get("terminal_validation_sidecar_path"),
        )


def terminal_controls(primary: Path, value: Json, raw: Path) -> list[Json]:
    """Make invalid private receipt copies so historical evidence stays immutable."""
    rows = []
    original = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    for name in ("missing", "altered", "malformed"):
        parent = raw / name
        parent.mkdir(parents=True, exist_ok=True)
        copied = parent / primary.name
        copied.write_bytes(primary.read_bytes())
        side = parent / "terminal.json"
        operand = deepcopy(value)
        operand["terminal_validation_sidecar_path"] = str(side)
        if name == "altered":
            changed = deepcopy(original)
            changed.get("publication", changed)["primary_sha256"] = "sha256:" + "0" * 64
            atomic_json(side, changed)
        elif name == "malformed":
            atomic_json(side, [])
        observed = terminal(copied, operand, parent / "custody")
        rows.append(
            dict(control=name, rejected=observed["error"] is not None, observation=observed)
        )
    return rows


def measure(root: Path, raw: Path, *, initial_failures: list[Json] | None = None) -> Json:
    """Authenticate only this branch's prerequisites before independent reduction."""
    progress("preconditions_before", 0, len(INPUTS))
    raw.mkdir(parents=True, exist_ok=True)
    snapshots, failures, checked = [], list(initial_failures or []), []
    for index, name in enumerate(INPUTS):
        path = root / name
        ref = base.authority._snapshot(
            path,
            path.read_bytes() if path.is_file() else None,
            raw / "inputs",
            "input" + str(index),
        )
        snapshots.append(ref)
        row = dict(
            upstream_id=TASK,
            path=str(path),
            hash=ref["sha256"],
            artifact_field="exists",
            op="==",
            expected=True,
            observed=path.is_file(),
            passed=path.is_file(),
        )
        checked.append(row)
        if not row["passed"]:
            failures.append(row)
    work = dict(
        preconditions_checked=checked,
        precondition_failures=failures,
        source_artifact_hashes=snapshots,
        task_dispositions=[],
        historical_required_failures=[],
        legacy_operands=[],
        terminal_observations=[],
        consumer_controls=[],
        mutation_controls=[],
    )
    progress("preconditions_after", len(INPUTS), 0)
    work["contract"] = assess(
        root / DESIGN,
        root / "research-roadmap-next.yaml",
        root / "research-roadmap.yaml",
        raw / "authority",
    )
    if failures:
        return work
    progress("authority_before", 0, 13)
    work["mutation_controls"] = mutations(
        root / DESIGN, root / "research-roadmap.yaml", raw / "mutations"
    )
    progress("authority_after", 13, 0)
    archive = yaml.safe_load((root / "research-complete.yaml").read_bytes())
    finished = next(m for m in archive["milestones"] if m["id"] == "2026.10.708")
    atomic_json(raw / "finished_v708_rows.json", finished)
    old = json.loads((root / "results/experiment_8204_v708_capstone.json").read_bytes())
    work["historical_required_failures"] = deepcopy(old.get("historical_hash_failures", []))
    work["historical_required_failures"].extend(old.get("gate_check_summary", []))
    for index, task in enumerate(finished["tasks"]):
        progress("historical_before", index, 13 - index)
        path = root / task["deliverable"]
        present = path.is_file()
        value = json.loads(path.read_bytes()) if present else {}
        ref = base.authority._snapshot(
            path, path.read_bytes() if present else None, raw / "history", "primary" + str(index)
        )
        work["source_artifact_hashes"].append(ref)
        observed = terminal(path, value, raw / "terminal" / str(index)) if present else None
        skip = task["result"] == "GATE_BLOCKED"
        alternates = [
            dict(path=str(p), sha256=sha256_file(p), artifact=json.loads(p.read_bytes()))
            for p in sorted(path.parent.glob(f"experiment_{8192 + index}_*.json"))
            if p != path
        ]
        for alternate in alternates:
            p = Path(alternate["path"])
            work["source_artifact_hashes"].append(
                base.authority._snapshot(p, p.read_bytes(), raw / "history", "skip" + str(index))
            )
        row = dict(
            task_id=task["id"],
            experiment_id=8192 + index,
            path=str(path),
            sha256=ref["sha256"],
            primary_present=present,
            conductor_result=task["result"],
            disposition="conductor_skip" if skip else "producer" if present else "missing_primary",
            original_archive_row=task,
            saved_capstone_row=old["task_dispositions"][index],
            honest_verdict=value.get("honest_verdict"),
            verdict_class=value.get("verdict_class"),
            required_checks_passed=value.get("required_checks_passed"),
            skip_evidence=alternates,
        )
        work["task_dispositions"].append(row)
        if present:
            work["legacy_operands"].append(dict(task_id=task["id"], **legacy_operand(value)))
            refs = hash_references(value.get("code_config_hashes", []))
            current_hashes = [
                dict(
                    r, observed=sha256_file(Path(r["path"])) if Path(r["path"]).is_file() else None
                )
                for r in refs
            ]
            work["terminal_observations"].append(
                dict(
                    task_id=task["id"],
                    code_schema="mapping-v1"
                    if isinstance(value.get("code_config_hashes"), dict)
                    else "reference-list-v1",
                    code_hash_comparisons=current_hashes,
                    observation=observed,
                )
            )
            work["historical_required_failures"].extend(
                dict(task_id=task["id"], receipt=r)
                for r in receipts(value.get("validation_receipts", []))
                if not r["passed"]
            )
            work["historical_required_failures"].extend(
                dict(task_id=task["id"], hash_failure=r)
                for r in current_hashes
                if r["sha256"] != r["observed"]
            )
            if not observed["passed"]:
                work["historical_required_failures"].append(
                    dict(task_id=task["id"], terminal_failure=observed)
                )
            if index == 0 and observed["error"] is None:
                work["consumer_controls"] = terminal_controls(path, value, raw / "receipt_controls")
        progress("historical_after", index + 1, 12 - index)
    return work


def build(work: Json, validation: list[Json], *, fixture: bool = False) -> Json:
    """Keep owned qualification, current authority and old science as distinct facts."""
    contract = work["contract"]
    owned = bool(validation) and all(
        r["passed"] and r["exit_code"] >= 0 and not r["timed_out"] for r in validation
    )
    consumers = (
        bool(work["consumer_controls"])
        and all(r["rejected"] for r in work["consumer_controls"])
        and any(r["typeerror_reproduced"] for r in work["legacy_operands"])
    )
    controls = bool(work["mutation_controls"]) and all(
        r["rejected"] for r in work["mutation_controls"]
    )
    failures = work["precondition_failures"] + contract["gate_check_summary"]
    state = (
        "blocked"
        if work["precondition_failures"]
        else "disqualified"
        if not owned
        else "blocked"
        if failures
        else "circular_positive"
        if consumers and controls
        else "disqualified"
    )
    rows = []
    for index in range(13):
        source = contract["contract_rows"][index] if index < len(contract["contract_rows"]) else {}
        checks = source.get("checks", dict(authority_available=False))
        matched = all(checks.values())
        rows.append(
            dict(
                unit_id=f"exp{8205 + index}",
                source_id="V709_authority",
                source_cluster_id="V709_authority",
                arm="authority_qualification",
                condition="full_task_agreement",
                seed=None,
                status="completed",
                completed=True,
                checks=checks,
                absolute_metric=int(matched),
                numerator=sum(checks.values()),
                denominator=len(checks),
                raw_numerator=sum(checks.values()),
                raw_denominator=len(checks),
                failed=not matched,
                excluded=not matched,
                censored=False,
                effective_independent_groups=0,
            )
        )
    value = dict(
        experiment_id=8205,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261006",
        schema="carnot.v709.qualification.v1",
        title="V709 authority and consumer qualification",
        honest_verdict="complete_" + state + "_authority_consumer_qualification",
        verdict_class=state,
        gate_check_summary=failures,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[],
        call_ledger=[],
        random_seed=7098205,
        preconditions_checked=work["preconditions_checked"],
        rows=rows,
        intended_count=13,
        completed_count=13,
        failed_count=sum(r["failed"] for r in rows),
        censored_count=0,
        excluded_count=sum(r["excluded"] for r in rows),
        independent_count=0,
        verifier_is_oracle=True,
        exposure_scope="exposed development; administrative oracle controls only",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        validation_receipts=validation,
        contract_ready_score=int(owned and controls and contract["activated"]),
        consumer_reader_ready_score=int(owned and consumers),
        task_dispositions=work["task_dispositions"],
        authority_snapshots=contract["authority_snapshots"],
        canonical_tasks_sha256=contract["canonical_tasks_sha256"],
        staged_readiness=contract.get("planning_matched", False),
        activated_readiness=contract["activated"],
        historical_required_failures=work["historical_required_failures"],
        historical_evidence_ready_score=0,
        source_artifact_hashes=work["source_artifact_hashes"],
        cited_upstream_artifacts=[
            dict(
                experiment_id=r["experiment_id"],
                path=r["path"],
                sha256=r["sha256"],
                fields_imported=[
                    "honest_verdict",
                    "verdict_class",
                    "required_checks_passed",
                    "code_config_hashes",
                    "terminal_validation_sidecar_path",
                    "validation_receipts",
                    "historical_hash_failures",
                    "gate_check_summary",
                    "task_dispositions",
                ],
            )
            for r in work["task_dispositions"]
        ],
        consumer_controls=work["consumer_controls"],
        mutation_controls=work["mutation_controls"],
        legacy_operands=work["legacy_operands"],
        terminal_observations=work["terminal_observations"],
        fixture_validation_scope=fixture,
        flagged_adversarial=False,
        methodology_note="Exact authority agreement and real terminal/pytest controls qualify administrative execution only. Ten historical primaries and three gate skips remain original. No model, training, generalization or scientific benefit is measured.",
        acceptance_gates=dict(
            authority=dict(
                passed=contract["activated"],
                principle="Full current task authority agreement only.",
            ),
            mutations=dict(
                passed=controls, principle="Reject changed independent contract operands."
            ),
            consumers=dict(
                passed=consumers,
                principle="Schema and invalid terminal controls cannot repair old science.",
            ),
            owned_validation=dict(
                passed=owned,
                principle="Every normal required child outcome must match its frozen expectation.",
            ),
        ),
    )
    return value
