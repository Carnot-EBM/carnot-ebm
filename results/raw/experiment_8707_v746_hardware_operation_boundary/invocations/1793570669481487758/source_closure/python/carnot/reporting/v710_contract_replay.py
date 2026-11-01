"""REQ-REPORT-8218: bind current authority without upgrading historical science.

Immutable byte copies let future readers distinguish unavailable original code
from later checkout edits. Administrative agreement grants no learning benefit.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

import yaml

from carnot.reporting import v700_contract_custody as base
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, sha256_file
from carnot.reporting.roadmap_contract import parse_design

Json = dict[str, Any]
ROOT = base.ROOT
NAME = "experiment_8218_v710_contract_replay_qualification"
CLI = f"scripts/experiments/{NAME}.py"
DESIGN = "openspec/change-proposals/research-roadmap-vNEXT.md"
HISTORY = "openspec/change-proposals/research-roadmap-v709-preserved-20261006.md"
MILESTONE = "2026.10.710"
TASK = "exp8218-contract-replay-qualification"
TEST = "tests/python/test_v710_contract_replay_8218.py"
OWNED = [
    "python/carnot/reporting/v710_contract_replay.py",
    "python/carnot/reporting/v710_replay_history.py",
    "python/carnot/reporting/v710_replay_runner.py",
    CLI,
]

REUSED = [
    "python/carnot/reporting/v709_execution.py",
    "python/carnot/reporting/v709_runner.py",
    "python/carnot/reporting/primary_publication.py",
    "scripts/experiment_template.py",
    "python/carnot/reporting/current_work_receipt.py",
    "python/carnot/reporting/roadmap_contract.py",
    "python/carnot/reporting/v685_authority_lifecycle.py",
    "python/carnot/reporting/v700_contract_custody.py",
    "python/carnot/reporting/v709_qualification.py",
    "python/carnot/verify/restricted_decision_audit_8210.py",
    "python/carnot/verify/restricted_decision_rule_8210.py",
]


def snapshot(path: Path, raw: Path, role: str) -> Json:
    """Retain bytes once so later source edits do not invalidate historical custody."""
    return dict(
        base.authority._snapshot(path, path.read_bytes() if path.is_file() else None, raw, role),
        path=str(path),
    )


def verify_reference(ref: Json) -> bool:
    """Read only immutable copies; absence stays an explicitly unmeasured operand."""
    path = Path(ref.get("snapshot_path", ref["path"]))
    return not ref.get("exists", True) or (path.is_file() and sha256_file(path) == ref["sha256"])


def require_reference(ref: Json) -> None:
    """A copied hash failure cannot become a successful replay through truth coercion."""
    if not verify_reference(ref):
        raise ValueError("immutable_evidence_hash_drift:" + ref["path"])


def failure(
    path: Path, field: str, expected: Any, observed: Any, digest: str | None = None
) -> Json:
    """Name the failed operand precisely so unavailable bytes differ from measured zero."""
    return dict(
        upstream_id=TASK,
        path=str(path),
        hash=digest,
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=False,
    )


def assess(design: Path, staged: Path, active: Path, raw: Path) -> Json:
    """Existing complete-task readers own activation and the fourteen-task identity."""
    snapshots = {
        role: snapshot(path, raw, role)
        for role, path in [("design", design), ("staged", staged), ("active", active)]
    }
    frozen = [
        Path(snapshots[k].get("snapshot_path", raw / ("absent-" + k)))
        for k in ("design", "staged", "active")
    ]
    try:
        value = base.authority.assess_authorities(
            *frozen, raw / "assessment", milestone=MILESTONE, first_id=8218, count=14
        )
        _, tasks = parse_design(frozen[0].read_text(), milestone=MILESTONE)
        value["tasks"] = tasks
        value["gate_check_summary"] = [
            failure(
                Path(g["artifact_path"]),
                g["artifact_field"],
                g["expected"],
                g["observed"],
                g["artifact_hash"],
            )
            for g in value["gate_check_summary"]
        ]
    except (OSError, ValueError, KeyError, TypeError, IndexError, yaml.YAMLError) as error:
        value = dict(
            activated=False,
            planning_matched=False,
            contract_rows=[],
            tasks=[],
            canonical_tasks_sha256=None,
            gate_check_summary=[
                failure(
                    design, "authority_readable", True, str(error), snapshots["design"]["sha256"]
                )
            ],
        )
    value["authority_snapshots"] = snapshots
    return dict(value)


def mutations(design: Path, active: Path, raw: Path) -> list[Json]:
    """Change independent operands on private full authority, preserving live task bytes."""
    original = yaml.safe_load(active.read_bytes())
    rows = []
    for name in (
        "count",
        "order",
        "title",
        "prompt",
        "gate",
        "model",
        "deliverable",
        "digest",
        "table",
    ):
        parent = raw / name
        parent.mkdir(parents=True, exist_ok=True)
        changed, text = deepcopy(original), design.read_text()
        if name == "count":
            changed["tasks"].pop()
        elif name == "order":
            changed["tasks"].reverse()
        elif name in ("title", "prompt", "deliverable"):
            changed["tasks"][0][name] += " changed"
        elif name == "gate":
            gate = next(t for t in changed["tasks"] if t["gated_on"])["gated_on"][0]
            gate["artifact_filed"] = gate.pop("artifact_field")
        elif name == "model":
            changed["tasks"][0]["MODEL_SPECS"] = [{"hf_id": "changed"}]
        elif name == "digest":
            text = re.sub(r"(Canonical full-task SHA256: `)[0-9a-f]{64}", r"\g<1>" + "0" * 64, text)
        else:
            text = text.replace("| 1 |", "| 99 |", 1)
        d, a = parent / "design.md", parent / "active.yaml"
        d.write_text(text)
        a.write_text(yaml.safe_dump(changed))
        observation = assess(d, parent / "absent", a, parent / "snapshots")
        rows.append(
            dict(
                control=name,
                rejected=not observation["activated"],
                gate_check_summary=observation["gate_check_summary"],
            )
        )
    return rows


def commands(private: Path) -> list[Json]:
    """Reuse qualified bounded validation and measure real subprocess statements."""
    from unittest.mock import patch
    from carnot.reporting import v709_runner as runner

    with (
        patch.object(runner, "OWNED", OWNED),
        patch.object(runner, "TEST", TEST),
        patch.object(runner, "CLI", CLI),
    ):
        plan: list[Json] = [p for p in runner.commands(private) if p["name"] != "focused_pytest"]
    config = private / "coverage.ini"
    config.write_text(
        config.read_text().replace("parallel=true", "parallel=true\npatch=subprocess")
    )
    plan.append(
        dict(
            name="E2E021",
            argv=[
                str(ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_restricted_decision_audit_8210.py",
                "--basetemp=" + str(private / "audit"),
            ],
            deadline=180,
            expected=0,
            scope="owned",
        )
    )
    affected = [
        "test_source_boundary_7852.py",
        "test_experiment_7942_v689_sentence_labels.py",
        "test_v709_qualification_8205.py",
        "test_restricted_decision_audit_8210.py",
        "test_memory_benefit_audit_8212.py",
        "test_prospective_service_measurement_8214.py",
        "test_arc_authoritative_frontier_8215.py",
        "test_hardware_workload_obligations_8216.py",
    ]
    plan.append(
        dict(
            name="affected_reducers_E2E015_019",
            argv=[
                str(ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                *["tests/python/" + t for t in affected],
                "--basetemp=" + str(private / "affected"),
            ],
            deadline=360,
            expected=0,
            scope="owned",
        )
    )
    return plan


def reduce(work: Json, receipts: list[Json]) -> Json:
    """Separate current readiness from old science and give owned failures precedence."""
    owned = bool(receipts) and all(
        r["passed"] and r["exit_code"] >= 0 and not r["timed_out"] for r in receipts
    )
    contract = work["contract"]
    failures = work["failures"] + [
        dict(g, artifact_field=g.get("field", g.get("check")))
        for g in contract["gate_check_summary"]
    ]
    failures.extend(
        failure(
            Path(r.get("stdout_path", "unavailable_command_stream")),
            "owned_validation." + r.get("name", "unnamed") + ".exit_code",
            r.get("expected_exit", 0),
            r["exit_code"],
            r.get("stdout_sha256"),
        )
        for r in receipts
        if not r["passed"]
    )
    state = "disqualified" if not owned else "blocked" if failures else "circular_positive"
    suffix = failures[0]["artifact_field"] if failures else "administrative_replay"
    suffix = re.sub("[^a-z0-9_]", "_", suffix.lower())
    rows = []
    for index in range(14):
        source = contract["contract_rows"][index] if index < len(contract["contract_rows"]) else {}
        checks = source.get("checks", dict(authority_available=False))
        matched = all(checks.values())
        rows.append(
            dict(
                unit_id=f"exp{8218 + index}",
                source_id="V710_authority",
                source_cluster_id="V710_authority",
                arm="authority_qualification",
                condition="immutable_full_task_agreement",
                seed=None,
                status="completed",
                completed=True,
                missing_status=not bool(source),
                checks=checks,
                metric="contract_agreement",
                absolute_metric=int(matched),
                numerator=sum(checks.values()),
                denominator=len(checks),
                raw_numerator=sum(checks.values()),
                raw_denominator=len(checks),
                failed=not matched,
                censored=False,
                excluded=not matched,
                effective_independent_groups=0,
            )
        )
    ready = state == "circular_positive"
    return dict(
        experiment_id=8218,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261007",
        schema="carnot.v710.contract-replay.v1",
        title="Immutable historical replay qualification",
        honest_verdict="complete_" + state + "_" + suffix,
        verdict_class=state,
        gate_check_summary=failures,
        rows=rows,
        intended_count=14,
        completed_count=14,
        failed_count=sum(r["failed"] for r in rows),
        censored_count=0,
        excluded_count=sum(r["excluded"] for r in rows),
        independent_count=0,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[],
        call_ledger=[],
        verifier_is_oracle=True,
        exposure_scope="exposed historical development; administrative custody only",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        contract_ready_score=int(owned and contract["activated"]),
        historical_replay_ready_score=int(ready),
        canonical_tasks_sha256=contract["canonical_tasks_sha256"],
        staged_readiness=contract.get("planning_matched", False),
        activated_readiness=contract["activated"],
        authority_snapshots=contract["authority_snapshots"],
        historical_dispositions=work["historical_dispositions"],
        immutable_code_snapshots=work["immutable_code_snapshots"],
        replay_controls=work["replay_controls"],
        h1_custody_qualification=work["h1_custody"],
        preconditions_checked=work["preconditions_checked"],
        source_artifact_hashes=work["source_artifact_hashes"],
        validation_receipts=receipts,
        required_checks_passed=owned,
        flagged_adversarial=False,
        acceptance_gates=dict(
            owned_validation=dict(passed=owned),
            authority=dict(passed=contract["activated"]),
            immutable_replay=dict(passed=not work["failures"]),
        ),
        scientific_gate_membership=[],
        external_publication_authorized=False,
        methodology_note="Exact contract and primitive custody qualify administrative execution. "
        "Historical disqualifications, nulls and benefit thresholds are unchanged. "
        "No current model calls, training, deployment or independent generalization.",
    )
