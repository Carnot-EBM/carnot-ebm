"""REQ-REPORT-8004: preserve thirteen terminal outcomes without transferring science.

Current work reads cached bytes. Historical model calls and small fitted heads
remain producer evidence, rather than claims about this invocation.
"""

from datetime import UTC, datetime
import gzip
import json
import os
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.reporting import v690_authority as authority
from carnot.reporting import v693_capstone_reduction as reduction
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from scripts.conductor_gates import _eval_op

ROOT = Path(__file__).resolve().parents[3]
Json = dict[str, Any]
OWNED = [
    "python/carnot/reporting/v693_capstone.py",
    "python/carnot/reporting/v693_capstone_reduction.py",
    "python/carnot/reporting/v693_capstone_validation.py",
    "scripts/experiments/experiment_8004_v693_capstone.py",
]
TEST = "tests/python/test_experiment_8004_v693_capstone.py"


def progress(phase: str, units: int = 0) -> None:
    """Expose each audit boundary even when there is no model activity."""
    print(f"[exp8004] phase={phase} completed_units={units}", flush=True)


def prepare_fixture(directory: Path) -> None:
    """Private oracle inputs test protocol completion without supplying natural evidence."""
    directory.mkdir(parents=True, exist_ok=True)
    raw = gzip.decompress((ROOT / "tests/fixtures/v693/active.yaml.gz").read_bytes())
    (directory / "research-roadmap.yaml").write_bytes(raw)
    roadmap = yaml.safe_load(raw)
    tasks = roadmap["tasks"]
    fields: Json = {t["id"]: {} for t in tasks}
    for t in tasks:
        for gate in t.get("gated_on", []):
            if gate["artifact_field"] not in {"verdict_class", "flagged_adversarial"}:
                fields[gate["upstream"]][gate["artifact_field"]] = gate["value"]
    table = ["| Order | ID | Title | Phase | Deliverable |", "| --- | --- | --- | --- | --- |"]
    table += [
        f"| {i} | {t['id']} | {t['title']} | {t['phase']} | {t['deliverable']} |"
        for i, t in enumerate(tasks, 1)
    ]
    design = "# Private circular V693 oracle\n\n## Exact task contract\n\n" + "\n".join(table)
    design += "\nCanonical full-task SHA-256: `" + authority.lifecycle.tasks_digest(tasks) + "`\n"
    design += "\n<!-- V693_TASK_CONTRACT_START -->\n```json\n" + json.dumps(roadmap) + "\n```\n"
    (directory / "design.md").write_text(design)
    for i, t in enumerate(tasks[:-1], 7992):
        p = directory / t["deliverable"]
        s = directory / "sidecars" / f"{i}.json"
        atomic_json(
            p,
            dict(
                experiment_id=i,
                task_id=t["id"],
                verdict_class="null",
                honest_verdict="complete_null_private",
                execution_date="20261001",
                flagged_adversarial=False,
                terminal_validation_sidecar_path=str(s),
                rows=[],
                **fields[t["id"]],
            ),
        )
        atomic_json(s, dict(primary_sha256=sha256_file(p), passed=True))


def read(path: Path) -> Json:
    """Absence is distinct from malformed producer data and never triggers a glob fallback."""
    if not path.is_file():
        return {}
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("producer_object_required")
    return value


def operand(path: Path, upstream: str, field: str, expected: Any, observed: Any) -> Json:
    """Every blocked operand retains its exact byte identity and missing-field status."""
    return dict(
        upstream_id=upstream,
        path=str(path),
        hash=sha256_file(path) if path.is_file() else None,
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
    )


def reference(path: Path, role: str, fields: list[str] | None = None) -> Json:
    """Record imported fields and producer dates without checking mutable historical code."""
    return dict(
        path=str(path),
        sha256=sha256_file(path) if path.is_file() else None,
        role=role,
        imported_fields=fields or [],
    )


def collect(root: Path, tasks: list[Json]) -> tuple[list[Json], list[Json], list[Json], list[Json]]:
    """Read declared primaries, exact skip receipts and byte-bound producer sidecars."""
    dispositions, refs, failures, audits = [], [], [], []
    targets: Json = {}
    data_rows = []
    for t in tasks[:-1]:
        path = root / t["deliverable"]
        try:
            data = read(path)
        except (ValueError, OSError):
            data = dict(
                verdict_class="disqualified",
                honest_verdict="complete_disqualified_malformed_producer",
            )
        data_rows.append(data)
        labelled = (
            data.get("verdict_class") in {"positive", "null"}
            and data.get("flagged_adversarial") is False
        )
        for r in (data.get("rows", []) + data.get("retention_rows", [])) if labelled else []:
            if isinstance(r, dict) and type(r.get("y")) is int and r["y"] in (0, 1):
                targets[r["family_id"]] = r["y"]
    for index, (t, data) in enumerate(zip(tasks[:-1], data_rows, strict=True)):
        path = root / t["deliverable"]
        ref = reference(path, "declared_primary", list(data))
        ref["execution_date"] = data.get("execution_date", data.get("run_date"))
        refs.append(ref)
        skip = (
            root
            / f"results/experiment_{7992 + index}_{t['id'].split('-', 1)[1].replace('-', '_')}.json"
        )
        dispatch = read(skip) if skip != path else {}
        refs.append(reference(skip, "conductor_skip_receipt"))
        issues = []
        by_id = {
            task["id"]: (task, observed)
            for task, observed in zip(tasks[:-1], data_rows, strict=True)
        }
        for gate in t.get("gated_on", []):
            upstream, observed_data = by_id[gate["upstream"]]
            field = gate["artifact_field"]
            observed = observed_data.get(field, "contract_error_missing_field")
            if field not in observed_data or not _eval_op(observed, gate["op"], gate["value"])[0]:
                failed = operand(
                    root / upstream["deliverable"], gate["upstream"], field, gate["value"], observed
                )
                failed["op"] = gate["op"]
                issues.append(failed)
        for field in ("verdict_class", "honest_verdict", "execution_date", "flagged_adversarial"):
            if field not in data:
                issues.append(
                    operand(path, t["id"], field, "present", "contract_error_missing_field")
                )
        state = data.get("verdict_class", "blocked")
        if (
            state in {"blocked", "disqualified", "partial", "circular_positive"}
            or data.get("flagged_adversarial") is not False
        ):
            issues.append(
                operand(
                    path, t["id"], "qualified_non_circular_verdict", ["positive", "null"], state
                )
            )
        sidecar = data.get("terminal_validation_sidecar_path")
        if sidecar:
            p = Path(sidecar)
            saved = read(p)
            refs.append(reference(p, "terminal_sidecar"))
            if saved.get("primary_sha256", saved.get("candidate_sha256")) != ref["sha256"]:
                issues.append(
                    operand(
                        p,
                        t["id"],
                        "primary_sha256",
                        ref["sha256"],
                        saved.get("primary_sha256", "contract_error_missing_field"),
                    )
                )
            if saved.get("sidecar_path"):
                bound = Path(saved["sidecar_path"])
                refs.append(reference(bound, "bound_validator_output"))
                report = read(bound)
                if (
                    report.get("primary_sha256") != ref["sha256"]
                    or report.get("report", {}).get("passed") is not True
                ):
                    issues.append(operand(bound, t["id"], "bound_validation_passed", True, False))
        else:
            issues.append(
                operand(
                    path,
                    t["id"],
                    "terminal_validation_sidecar_path",
                    "present",
                    "contract_error_missing_field",
                )
            )
        shards = data.get("raw_shard_hashes", [])
        if isinstance(shards, dict):
            shards = [dict(path=p, sha256=h) for p, h in shards.items()]
        for raw in shards:
            p = Path(raw["path"])
            refs.append(reference(p, "primitive_shard"))
            if refs[-1]["sha256"] != raw["sha256"]:
                issues.append(operand(p, t["id"], "sha256", raw["sha256"], refs[-1]["sha256"]))
        try:
            audit = reduction.audit(data, targets)
        except (ValueError, KeyError, TypeError, OSError) as error:
            audit = dict(reduction_error=str(error))
            issues.append(operand(path, t["id"], "primitive_reduction", "valid", str(error)))
        qualified = not issues
        refs.extend(audit.get("causal_updates", {}).get("references", []))
        audits.append(
            dict(
                task_id=t["id"],
                producer_sha256=ref["sha256"],
                qualified=qualified,
                claim_scope="finite_development_descriptive"
                if qualified
                else "excluded_producer_diagnostic",
                **audit,
            )
        )
        dispositions.append(
            dict(
                task_id=t["id"],
                path=str(path),
                sha256=ref["sha256"],
                execution_date=ref["execution_date"],
                honest_verdict=data.get(
                    "honest_verdict", "complete_blocked_missing_declared_producer"
                ),
                verdict_class=state,
                producer_status="present" if path.is_file() else "missing",
                eligible=qualified,
                excluded=not qualified,
                failed=state == "disqualified",
                censored=False,
                started=bool(data),
                completed=True,
                raw_numerator=int(qualified),
                raw_denominator=1,
                arm="task_disposition",
                unit_id=t["id"],
                status="completed",
                conductor_skip_receipt=dict(
                    path=str(skip),
                    sha256=refs[2 * index + 1]["sha256"]
                    if index == 0
                    else sha256_file(skip)
                    if skip.is_file()
                    else None,
                    gates_evaluated=dispatch.get("gates_evaluated", []),
                ),
                gate_check_summary=issues,
                producer_gate_check_summary=data.get("gate_check_summary", []),
                producer_owned_failures=[
                    r
                    for r in data.get("validation_receipts", [])
                    if not r.get("passed") and r.get("classification") != "diagnostic"
                ],
                reported_code_config_hashes=data.get("code_config_hashes", {}),
            )
        )
        failures.extend(issues)
        progress("producer_reduced", index + 1)
    return dispositions, refs, failures, audits


def retirements(tasks: list[Json], dispositions: list[Json]) -> list[Json]:
    """Compare exact scope verdicts; the shared word null has no retirement meaning."""
    return [
        dict(
            task_id=t["id"],
            prior_experiment_id=p["experiment_id"],
            prior_verdict=p["verdict"],
            terminal_verdict=d["honest_verdict"],
            retire_if_same_verdict=p["retire_if_same_verdict"],
            repeated_exact_verdict=p["verdict"] == d["honest_verdict"],
            retire=p["retire_if_same_verdict"] and p["verdict"] == d["honest_verdict"],
            scope=t["id"] + ": unchanged " + d["honest_verdict"],
            reopen_condition=p["addressed_by"],
            producer_path=d["path"],
            producer_sha256=d["sha256"],
        )
        for t, d in zip(tasks, dispositions, strict=True)
        for p in t.get("prior_failures", [])
    ]


def authenticate_retirements(root: Path, rows: list[Json]) -> None:
    """Historical task declarations supply exact paths, never a guessed filename match."""
    archive = root / "research-complete.yaml"
    completed = (
        yaml.safe_load(archive.read_bytes()).get("milestones", []) if archive.is_file() else []
    )
    paths = {
        t["id"]: root / t["deliverable"]
        for m in completed
        for t in m.get("tasks", [])
        if t.get("deliverable")
    }
    for row in rows:
        path = paths.get(row["prior_experiment_id"])
        prior = read(path) if path is not None else {}
        row.update(
            prior_path=str(path) if path else None,
            prior_sha256=sha256_file(path) if path and path.is_file() else None,
            prior_authenticated=prior.get("honest_verdict") == row["prior_verdict"],
        )


def append_retirements(root: Path, rows: list[Json]) -> None:
    """Append only authenticated exact repeats while retaining every old manifest byte."""
    path = root / "ops/exclusion_manifest.yaml"
    original = path.read_text()
    existing = yaml.safe_load(original).get("retired_extras", [])
    ids = {r.get("id") for r in existing}
    additions = []
    for row in rows:
        identity = (
            "v693_exact_repeat_"
            + canonical_hash([row["task_id"], row["prior_experiment_id"], row["prior_verdict"]])[
                7:23
            ]
        )
        if row["retire"] and row.get("prior_authenticated") and identity not in ids:
            additions.append(
                dict(
                    id=identity,
                    experiment_scope=row["scope"],
                    reason="Exact repeated terminal verdict: " + row["prior_verdict"],
                    retired_milestone="2026.10.693",
                    retired_by_artifact="results/experiment_8004_v693_capstone.json",
                    retire_if_same_verdict=True,
                    prior_path=row["prior_path"],
                    prior_sha256=row["prior_sha256"],
                    producer_path=row["producer_path"],
                    producer_sha256=row["producer_sha256"],
                    reopening_condition=row["reopen_condition"],
                )
            )
    if additions:
        appended = original + "\n" + yaml.safe_dump(additions, sort_keys=False)
        if len(yaml.safe_load(appended)["retired_extras"]) != len(existing) + len(additions):
            raise ValueError("retirement_append_schema")
        path.write_text(appended)


def build(root: Path, active: Path, design: Path, date: str, snapshots: Path) -> Json:
    """Freeze authority before measurement and distinguish development from deployment."""
    if date != "20261002":
        raise ValueError("execution_date_changed")
    started, tick = datetime.now(UTC).isoformat(), time.monotonic_ns()
    progress("freeze_authority")
    tasks = yaml.safe_load(active.read_bytes())["tasks"]
    if [t["id"].split("-")[0] for t in tasks] != [f"exp{n}" for n in range(7992, 8005)]:
        raise ValueError("thirteen_task_roster_required")
    assessed = authority.shared.assess(
        design,
        root / "research-roadmap-next.yaml",
        active,
        snapshots,
        milestone="2026.10.693",
        first_id=7992,
        count=13,
    )
    dispositions, refs, failures, audits = collect(root, tasks)
    for f in assessed["gate_check_summary"]:
        failures.append(
            dict(
                upstream_id="V693_authority",
                path=f.get("path", f.get("artifact_path")),
                hash=f.get("hash", f.get("artifact_hash")),
                artifact_field=f["artifact_field"],
                op=f["op"],
                expected=f["expected"],
                observed=f["observed"],
            )
        )
    old = root / "results/experiment_7991_v692_capstone.json"
    history = read(old)
    refs.append(
        reference(
            old,
            "historical_v692",
            ["honest_verdict", "verdict_class", "outcome_rows", "validation_receipts"],
        )
    )
    historical = history.get("outcome_rows", [])[:-1] + [
        dict(
            task_id="exp7991-capstone",
            honest_verdict=history.get("honest_verdict"),
            verdict_class=history.get("verdict_class"),
            failed_owned_checks=[
                r for r in history.get("validation_receipts", []) if not r["passed"]
            ],
        )
    ]
    status = "blocked" if failures else "null"
    verdict = (
        "complete_blocked_v693_prerequisites" if failures else "complete_null_v693_development_only"
    )
    own = dict(
        task_id=tasks[-1]["id"],
        path=None,
        sha256=None,
        execution_date=date,
        honest_verdict=verdict,
        verdict_class=status,
        producer_status="current_capstone_pending_validation",
        eligible=False,
        excluded=False,
        failed=False,
        censored=False,
        started=True,
        completed=False,
        raw_numerator=0,
        raw_denominator=1,
        arm="task_disposition",
        unit_id=tasks[-1]["id"],
        status="pending",
        self_hash_policy="no recursive final artifact hash",
    )
    dispositions.append(own)
    code = [reference(ROOT / p, "current_code") for p in OWNED]
    gaps = {
        "useful_oracle_distinct_decisions": dict(
            closed=False,
            decision="unvalidated_development_static_decisions",
            evidence_tasks=["exp7997-typed-development-decisions"],
            next_action="Repair owned typed-decision verification; evaluate source-audited independent natural targets.",
            reopen_condition="Qualified same-information useful decisions with working controls and genuine headroom.",
        ),
        "durable_self_learning": dict(
            closed=False,
            decision="finite_replay_only",
            evidence_tasks=["exp7998-selective-feedback-learning", "exp7999-learning-causal-audit"],
            next_action="Resolve natural class support and selectable headroom before new independent evaluation.",
            reopen_condition="Durable causal updates improve future independent decisions and sealed retention without hidden-label access.",
        ),
        "validated_affordable_deployment": dict(
            closed=False,
            decision="deployment_generalization_unavailable",
            evidence_tasks=[
                "exp8001-arc-supervisor-qualification",
                "exp8002-service-cost",
                "exp8003-hardware-sparse-boundary",
            ],
            next_action="Require a useful workload and complete measured acquisition, load amortization, durable service and compatible device kernel.",
            reopen_condition="Qualified out-of-distribution or hidden-game outcomes with affordable full-service cost on an authenticated deployment path.",
        ),
    }
    value = build_current_work_receipt(
        run_id="exp8004-" + date,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        inference_substrate_details=dict(work="independent cached primitive audit"),
        started_monotonic_ns=tick,
        ended_monotonic_ns=time.monotonic_ns(),
    )
    value.update(
        experiment_id=8004,
        task_id=tasks[-1]["id"],
        milestone="2026.10.693",
        run_date=date,
        execution_date=date,
        started_at=started,
        finished_at=datetime.now(UTC).isoformat(),
        honest_verdict=verdict,
        verdict_class=status,
        gate_check_summary=failures,
        preconditions_checked=dict(
            authority=assessed["activated"],
            exact_declared_producers=True,
            pretrained_calls=0,
            private_fixture_scope="circular_positive",
        ),
        rows=dispositions,
        task_dispositions=dispositions,
        independent_reduction_rows=audits,
        historical_failure_rows=historical,
        source_artifact_hashes=refs,
        cited_upstream_artifacts=[r for r in refs if r["role"] == "declared_primary"],
        raw_shard_hashes=[r for r in refs if r["role"] == "primitive_shard"],
        code_config_hashes=code,
        authority_snapshots=assessed["authority_snapshots"],
        canonical_tasks_sha256=authority.lifecycle.tasks_digest(tasks),
        task_contract=tasks,
        input_root=str(root),
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(
            model_loads_attempted=0,
            model_loads_completed=0,
            generation_calls_attempted=0,
            generation_calls_completed=0,
            loads=0,
            calls=0,
            tokens=0,
        ),
        trained_head_specs=[],
        random_seed=6938004,
        verifier_is_oracle=False,
        claim_scope=dict(
            natural_data="source_disjoint_development",
            annotations="fallible_human_source_support",
            independent_benefit=False,
            protocol_controls="circular_positive",
            hidden_game_generalization=False,
            deployment_generalization=False,
        ),
        positive_control_results=reduction.controls(),
        acceptance_gate_results=dict(
            validity=False,
            decision_benefit=False,
            finite_replay_benefit=False,
            deployment_generalization=False,
            readiness=0,
        ),
        capstone_execution_ready_score=0,
        gap_decisions=gaps,
        science_ready=False,
        paper_ready=False,
        g1=False,
        g2=False,
        g3=False,
        g4=False,
        unmet_gates=["publication_gate_not_executed"],
        publication_gate_results={},
        generalized_learning_benefit_score=0,
        retirement_rows=retirements(tasks, dispositions),
        reopen_conditions=[g["reopen_condition"] for g in gaps.values()],
        flagged_adversarial=False,
        validation_command_manifest_path=None,
        validation_receipts=[],
        coverage_statement_counts={},
        terminal_validation_sidecar_path=None,
        sample_size_budget=dict(
            intended=13,
            eligible=sum(r["eligible"] for r in dispositions),
            started=sum(r["started"] for r in dispositions),
            completed=12,
            excluded=sum(r["excluded"] for r in dispositions),
            failed=sum(r["failed"] for r in dispositions),
            censored=0,
            independent=0,
            unit="task_disposition",
        ),
    )
    value["phase_spans"] = [
        dict(
            phase="frozen_authority_and_primitive_reduction",
            start_s=0,
            end_s=value["duration_s"],
            completed_units=12,
        )
    ]
    authenticate_retirements(root, value["retirement_rows"])
    value["reproducibility_checksum"] = canonical_hash(
        dict(
            sources=refs,
            code=code,
            authority=value["canonical_tasks_sha256"],
            seed=value["random_seed"],
        )
    )
    value["field_principles"] = {
        k: "Bind current audit to exact evidence; preserve failed history and separate deployment claims."
        for k in value
    }
    return value


def terminal_disposition(value: Json) -> None:
    """The current task becomes terminal only after observed owned validation."""
    d = value["task_dispositions"][-1]
    d.update(
        honest_verdict=value["honest_verdict"],
        verdict_class=value["verdict_class"],
        producer_status="current_capstone_validated",
        completed=True,
        status="completed",
        eligible=value["acceptance_gate_results"]["validity"],
        failed=value["verdict_class"] == "disqualified",
        raw_numerator=int(value["acceptance_gate_results"]["validity"]),
    )
    value["rows"] = value["task_dispositions"]
    value["sample_size_budget"].update(
        completed=13,
        eligible=sum(r["eligible"] for r in value["rows"]),
        failed=sum(r["failed"] for r in value["rows"]),
    )
    value["retirement_rows"] = retirements(value["task_contract"], value["rows"])
    authenticate_retirements(Path(value["input_root"]), value["retirement_rows"])


def cold_replay(value: Json) -> list[str]:
    """Independently reread source rows so an edited aggregate cannot pass a stored seal."""
    progress("cold_replay_start")
    for ref in value["source_artifact_hashes"] + value["code_config_hashes"]:
        p = Path(ref["path"])
        if (sha256_file(p) if p.is_file() else None) != ref["sha256"]:
            return ["source_bytes_changed"]
    snapshots = value["authority_snapshots"]
    root = Path(value["input_root"])
    tasks = yaml.safe_load(Path(snapshots["active"]["snapshot_path"]).read_bytes())["tasks"]
    dispositions, _, _, audits = collect(root, tasks)
    errors = []
    if (
        dispositions != value["task_dispositions"][:-1]
        or audits != value["independent_reduction_rows"]
    ):
        errors.append("independent_reduction_drift")
    if value["generalized_learning_benefit_score"] != 0 or value["science_ready"] is not False:
        errors.append("unsupported_generalization")
    if (
        value["verdict_class"] in {"blocked", "disqualified"}
        and value["capstone_execution_ready_score"] != 0
    ):
        errors.append("unsafe_readiness")
    if authority.lifecycle.tasks_digest(tasks) != value["canonical_tasks_sha256"]:
        errors.append("authority_drift")
    return errors
