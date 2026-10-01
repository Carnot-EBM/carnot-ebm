"""REQ-REPORT-7978-V691: keep audit completion separate from scientific benefit.

Frozen invocation receipts retain each producer's date across UTC midnight.
Exact primaries and checkpoint readers prevent dispatch records from becoming science.
"""

from datetime import UTC, datetime
import gzip
import json
import os
from pathlib import Path
import time
from typing import Any

import yaml

from carnot import experiment_7968_v691_response_role_targets as response
from carnot import experiment_7972_v691_qwen_energy_calibration as calibration
from carnot.reporting import service_cost_7976 as service
from carnot.reporting import v690_capstone as prior
from carnot.reporting import v691_contract_methods as methods
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)

ROOT = prior.ROOT
REPLAY_FIELDS = (*prior.REPLAY_FIELDS, "producer_date_rows", "scientific_branches", "science_ready")
BRANCHES = {"source_energy": (7970, 7971, 7973, 7974), "qwen_calibration": (7968, 7969, 7972)}


def progress(started: float, phase: str, units: int) -> None:
    """Show actual work boundaries without adding artificial runtime."""
    print(
        f"[exp7978] phase={phase} completed_units={units} elapsed_s={time.monotonic() - started:.3f}",
        flush=True,
    )


def tasks_from(active: Path) -> list[dict[str, Any]]:
    """A missing authority uses only the preserved roster for explicit dispositions."""
    try:
        tasks = yaml.safe_load(active.read_bytes())["tasks"]
        if [t["id"].split("-")[0] for t in tasks] != [f"exp{n}" for n in range(7966, 7979)]:
            raise ValueError("task_roster")
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError):
        tasks = yaml.safe_load(
            gzip.decompress((ROOT / "tests/fixtures/v691/active.yaml.gz").read_bytes())
        )["tasks"]
    return list(tasks)


def freeze_producers(root: Path, active: Path) -> list[dict[str, Any]]:
    """Freeze producer bytes before any reduction or owned validation begins."""
    return [freeze_producer(root / t["deliverable"]) for t in tasks_from(active)[:-1]]


def freeze_producer(path: Path) -> dict[str, Any]:
    """Preserve malformed primary bytes so they receive an explicit invalid disposition."""
    try:
        return methods.freeze_invocation(path)
    except (ValueError, AttributeError, TypeError):
        return dict(path=str(path), sha256=sha256_file(path), identity={}, historical_timestamps={})


def prior_evidence(root: Path, declared: dict[str, Any]) -> dict[str, Any]:
    """Retirement needs the previous primary's exact bytes and actual terminal verdict."""
    number = declared["experiment_id"].split("-")[0][3:]
    paths = sorted((root / "results").glob(f"experiment_{number}_*.json"))
    found = []
    for path in paths:
        value, digest = prior.prior.shared.read(path)
        if value.get("task_id") == declared["experiment_id"] or len(paths) == 1:
            found.append(
                dict(path=str(path), sha256=digest, honest_verdict=value.get("honest_verdict"))
            )
    if len(found) != 1:
        return dict(path=None, sha256=None, honest_verdict=None, verified=False)
    return dict(**found[0], verified=found[0]["honest_verdict"] == declared["verdict"])


def reduction(number: int, data: dict[str, Any]) -> dict[str, Any]:
    """Checkpoint readers recover substantive claims instead of averaging ready scores."""
    value = prior.reduce_primitives(data.get("rows", []))
    if number == 7968 and (
        data.get("public_role_manifests") or "response_roles_ready_score" in data
    ):
        value["checkpoint_reduction"] = response.reconstruct(data)
    if number == 7972 and (
        data.get("calibrator_checkpoints") or "qwen_calibration_ready_score" in data
    ):
        value["checkpoint_reduction"] = calibration.replay(data)
    if number == 7976 and "service_rows" in data:
        value["checkpoint_reduction"] = service.replay(data)
        value["complete_service_cost"] = service.complete_cost(
            data["service_rows"], data["sample_size_budget"]["intended"]
        )
    return value


def decision(rows: list[dict[str, Any]]) -> str:
    """External absence differs from invalid measurements and completed null results."""
    if any(
        r["status"] in {"absent", "skipped", "blocked", "partial", "retired"} or r["gate_blocked"]
        for r in rows
    ):
        return "blocked"
    if any(not r["eligible"] for r in rows):
        return "disqualified"
    return "measured-null"


def build_candidate(
    root: Path,
    design: Path,
    active: Path,
    date: str,
    publication: dict[str, Any] | None = None,
    *,
    snapshots: Path | None = None,
    invocations: list[dict[str, Any]] | None = None,
    seal_rows: bool = True,
) -> dict[str, Any]:
    """Reduce all thirteen tasks independently while keeping each source's identity."""
    started, start_ns = time.monotonic(), time.monotonic_ns()
    started_at = datetime.now(UTC).isoformat()
    progress(started, "start", 0)
    if date != "20261001":
        raise ValueError("v691_date_changed")
    durable = root / "results/raw/experiment_7978_v691_capstone"
    directory = snapshots or durable / "authority"
    try:
        authority = methods.assess(design, root / "research-roadmap-next.yaml", active, directory)
    except (OSError, ValueError, TypeError, IndexError, KeyError, yaml.YAMLError):
        authority = prior.fallback_assess(
            design,
            root / "research-roadmap-next.yaml",
            active,
            directory,
            milestone="2026.10.691",
            first_id=7966,
            count=13,
        )
        authority["activated"] = False
    tasks = tasks_from(active)
    invocations = invocations if invocations is not None else freeze_producers(root, active)
    if len(invocations) != 12:
        raise ValueError("producer_invocation_roster")
    sources = [
        dict(path=s["source_path"], sha256=s["sha256"], role=role, exposure="administrative")
        for role, s in authority["authority_snapshots"].items()
    ]
    observed = {
        t["id"]: (*prior.prior.shared.read(root / t["deliverable"]), root / t["deliverable"])
        for t in tasks[:-1]
    }
    failures = list(authority["gate_check_summary"])
    outcomes, reductions, dates, citations, history, retirements = [], [], [], [], [], []
    progress(started, "inputs_hashes_invocations_frozen", 12)
    for index, task in enumerate(tasks):
        number = 7966 + index
        data, digest, path = observed.get(task["id"], ({}, None, root / task["deliverable"]))
        state = "self_administrative" if index == 12 else str(data.get("verdict_class", "absent"))
        own, dispatch = [], dict(path=None, sha256=None, gates_evaluated=[])
        if index < 12:
            frozen = invocations[index]
            try:
                issued = methods.validate_invocation(path, frozen)
            except (ValueError, AttributeError, TypeError):
                issued = dict(
                    **frozen,
                    passed=False,
                    gate_check_summary=[
                        prior.prior.shared.operand(
                            path, frozen["sha256"], task["id"], "producer_json_object", True, False
                        )
                    ],
                )
            dates.append(issued)
            own.extend(issued["gate_check_summary"])
            if data:
                try:
                    for key in ("run_date", "execution_date"):
                        datetime.strptime(data[key], "%Y%m%d")
                    beginning = datetime.fromisoformat(data["started_at"].replace("Z", "+00:00"))
                    ending = datetime.fromisoformat(data["finished_at"].replace("Z", "+00:00"))
                    if beginning.tzinfo is None or ending.tzinfo is None or ending < beginning:
                        raise ValueError("invalid_producer_timestamps")
                except (KeyError, ValueError, TypeError, AttributeError):
                    failure = prior.prior.shared.operand(
                        path,
                        digest,
                        task["id"],
                        "producer_invocation_identity",
                        "valid dates and ordered UTC timestamps",
                        "missing_or_invalid",
                    )
                    issued["passed"] = False
                    issued["gate_check_summary"].append(failure)
                    own.append(failure)
            sources.append(
                dict(
                    path=str(path),
                    sha256=digest,
                    role="declared_producer",
                    exposure="exposed_development",
                )
            )
            if not data:
                stub_path = (
                    root
                    / f"results/experiment_{number}_{task['id'].split('-', 1)[1].replace('-', '_')}.json"
                )
                stub, stub_hash = prior.prior.shared.read(stub_path)
                dispatch = dict(
                    path=str(stub_path),
                    sha256=stub_hash,
                    gates_evaluated=stub.get("gates_evaluated", []),
                )
                state = (
                    "skipped" if stub.get("blocked_at_layer") == "conductor_pre_gate" else "absent"
                )
                sources.append(
                    dict(
                        path=str(stub_path),
                        sha256=stub_hash,
                        role="conductor_dispatch",
                        exposure="administrative",
                    )
                )
                if path.is_file():
                    state = "disqualified"
        if data:
            expected = dict(
                experiment_id=number,
                task_id=task["id"],
                milestone="2026.10.691",
                flagged_adversarial=False,
            )
            own.extend(
                prior.prior.shared.operand(
                    path, digest, task["id"], key, val, data.get(key, "missing_field")
                )
                for key, val in expected.items()
                if data.get(key, "missing_field") != val
            )
            failures.extend(data.get("gate_check_summary", []))
            history.extend(data.get("historical_required_failures", []))
            citations.append(
                dict(
                    experiment_id=number,
                    path=str(path),
                    sha256=digest,
                    fields_imported=[
                        "rows",
                        "verdict_class",
                        "gate_check_summary",
                        "sample_size_budget",
                        "sealed_checkpoints",
                    ],
                )
            )
            if own:
                state = "disqualified"
        task_gate_passed = True
        for gate in task.get("gated_on", []):
            previous, previous_hash, previous_path = observed[gate["upstream"]]
            actual = previous.get(
                gate["artifact_field"], "missing_field" if previous else "missing_source"
            )
            if not prior._eval_op(actual, gate["op"], gate["value"])[0]:
                task_gate_passed = False
                failures.append(
                    prior.prior.shared.operand(
                        previous_path,
                        previous_hash,
                        gate["upstream"],
                        gate["artifact_field"],
                        gate["value"],
                        actual,
                        gate["op"],
                    )
                )
            prerequisite = next(r for r in outcomes if r["task_id"] == gate["upstream"])
            if not prerequisite["eligible"]:
                task_gate_passed = False
                failures.append(
                    prior.prior.shared.operand(
                        previous_path,
                        previous_hash,
                        gate["upstream"],
                        "authenticated_producer_eligible",
                        True,
                        False,
                    )
                )
        eligible = (
            state in prior.prior.shared.QUALIFIED
            and not own
            and task_gate_passed
            and authority["activated"]
        )
        raw = data.get("rows", [])
        raw_path = durable / "rows" / f"{task['id']}-{canonical_hash(raw)[7:]}.json"
        if seal_rows:
            atomic_json(raw_path, dict(producer_sha256=digest, rows=raw))
        sources.append(
            dict(
                path=str(raw_path),
                sha256=sha256_file(raw_path) if raw_path.is_file() else None,
                role="sealed_primitive_rows",
                exposure="exposed_development",
            )
        )
        try:
            reduced = reduction(number, data) if eligible else prior.reduce_primitives([])
        except (ValueError, TypeError, KeyError, OSError) as error:
            state, eligible = "disqualified", False
            reduced = prior.reduce_primitives([])
            own.append(
                prior.prior.shared.operand(
                    path,
                    digest,
                    task["id"],
                    "primitive_audit",
                    "valid rows/checkpoints",
                    str(error),
                )
            )
        failures.extend(own)
        reductions.append(
            dict(
                upstream_id=task["id"],
                path=str(path),
                sha256=digest,
                scientifically_eligible=bool(eligible),
                primitive_rows_sha256=canonical_hash(raw),
                primitive_rows_path=str(raw_path),
                unit="producer_primitive",
                **reduced,
            )
        )
        outcomes.append(
            dict(
                task_id=task["id"],
                upstream_id=task["id"],
                path=str(path),
                hash=digest,
                status=state,
                eligible=int(eligible),
                gate_blocked=not task_gate_passed,
                intended=1,
                started=1,
                completed=1,
                failed=0,
                censored=0,
                excluded=0,
                independent=0,
                arm="administrative_disposition",
                seed=7978,
                producer_budget=data.get("sample_size_budget"),
                exposure="exposed_development",
                conductor_gate_receipt=dispatch,
            )
        )
        for item in task.get("prior_failures", []):
            evidence = prior_evidence(root, item)
            if evidence["path"]:
                sources.append(
                    dict(
                        path=evidence["path"],
                        sha256=evidence["sha256"],
                        role="historical_terminal",
                        exposure="historical",
                    )
                )
            retirements.append(
                dict(
                    upstream_id=task["id"],
                    declared_prior=item,
                    prior_primary=evidence,
                    new_terminal_verdict=data.get("honest_verdict")
                    if index < 12
                    else "pending_capstone",
                    current_path=str(path),
                    current_sha256=digest,
                    decision="pending_comparison",
                    forward_difference=item.get("addressed_by"),
                    continuation_rule="Require changed inputs or prerequisites; preserve exact previous determinations.",
                )
            )
        progress(started, "independent_reduction", index + 1)
    branches = {
        name: dict(
            required_producers=list(numbers),
            decision=decision([outcomes[n - 7966] for n in numbers]),
            reductions=[reductions[n - 7966] for n in numbers],
            independent_benefit=False,
        )
        for name, numbers in BRANCHES.items()
    }
    ready = bool(
        authority["activated"] and all(r["eligible"] for r in outcomes[:-1]) and not failures
    )
    verdict = (
        "complete_null_independent_benefit_unshown" if ready else "complete_blocked_missing_science"
    )
    gaps = {
        "FR-12/FR-06": dict(
            decision=branches["source_energy"]["decision"],
            source_feature_decisions=outcomes[5],
            qwen_scalar_calibration=outcomes[6],
            qwen_branch_decision=branches["qwen_calibration"]["decision"],
            independent_benefit=False,
            continue_if="Qualified Exp7970 and Exp7971 must show action benefit against same-information classical controls.",
            retire_if="Retire unchanged missing or null scopes; raw-Qwen advantage alone cannot establish learned-energy benefit.",
        ),
        "FR-11": dict(
            decision=decision(outcomes[7:9]),
            independent_benefit=False,
            continue_if="Qualified delayed-feedback acquisition and issued-state calibration must show future benefit and retention against no-write and shuffled controls.",
            retire_if="Retire unchanged unavailable acquisition scopes; durable writes alone prove no future benefit.",
        ),
        "FR-05/FR-08/NFR-01": dict(
            decision=decision([outcomes[i] for i in (1, 10, 11)]),
            measured_service_cost=reductions[10],
            historical_board_custody=reductions[11],
            independent_benefit=False,
            continue_if="Private training qualification must pass; source branch and durable acquisition costs must join measured whole requests.",
            retire_if="Retire unchanged device probes without new custody or measured whole-service benefit.",
        ),
    }
    publication = publication or {}
    value = build_current_work_receipt(
        run_id="exp7978-20261001",
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details={"work": "independent primitive and checkpoint reduction"},
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=start_ns,
        ended_monotonic_ns=time.monotonic_ns(),
    )
    value.update(
        experiment_id=7978,
        task_id="exp7978-capstone",
        milestone="2026.10.691",
        run_date=date,
        execution_date=date,
        started_at=started_at,
        finished_at=datetime.now(UTC).isoformat(),
        honest_verdict=verdict,
        verdict_class="null" if ready else "blocked",
        flagged_adversarial=False,
        gate_check_summary=failures,
        rows=outcomes,
        outcome_rows=outcomes,
        independent_reduction_rows=reductions,
        producer_date_rows=dates,
        scientific_branches=branches,
        gap_decisions=gaps,
        sample_size_budget=dict(
            zip(
                prior.prior.shared.COUNTS,
                (13, sum(r["eligible"] for r in outcomes), 13, 13, 0, 0, 0, 0),
                strict=True,
            ),
            unit="task_disposition",
            science_unit_budgets={
                r["upstream_id"]: dict(
                    primitive_observations=r["intended"],
                    families=r["independent_families"],
                    source_groups=r["source_groups"],
                    seeds=r["seed_count"],
                    scientifically_independent=0,
                )
                for r in reductions
            },
        ),
        acceptance_gate_results=dict(
            validity=False,
            readiness=0,
            calibration=None,
            decision_benefit=None,
            retention=None,
            efficiency=None,
        ),
        random_seed=7978,
        source_artifact_hashes=sources,
        cited_upstream_artifacts=citations,
        preconditions_checked=failures,
        resolved_imports={
            m.__name__: str(Path(m.__file__).resolve())
            for m in (prior, methods, response, calibration, service)
        },
        validation_receipts=[],
        validation_command_manifest_path=None,
        observed_child_commands=[],
        coverage_statement_counts={},
        historical_required_failures=history,
        repository_health=dict(status="historical_backlog_open", affects_required_checks=False),
        verifier_is_oracle=False,
        claim_scope=dict(
            natural_data="exposed_development",
            fixture_agreement="circular_positive",
            gap_oracle_distinct="open_after_20260928_corrigendum",
            independent_benefit=False,
            human_target="fallible model-independent source annotations",
            DiffusionGemma="pending",
        ),
        model_specs=[],
        target_model="none",
        model_invocation_counts=dict(loads=0, calls=0, tokens=0),
        trained_head_specs=[],
        capstone_execution_ready_score=0,
        science_ready=ready,
        retirement_decisions=retirements,
        authority_snapshots=authority["authority_snapshots"],
        canonical_tasks_sha256=authority["canonical_tasks_sha256"],
        activation_confirmed=authority["activated"],
        publication_gate_results=publication,
        primary_resolution_receipt=dict(path=str(durable / "primary_resolution_receipt.json")),
        terminal_validation_sidecar_path=None,
        scratch_root_receipt={},
        next_reopen_conditions=[item["continue_if"] for item in gaps.values()],
        planning_handoff=dict(
            answered_questions=[
                "Producer dates need not equal consumer date.",
                "Source features and Qwen scalar calibration have separate prerequisites.",
            ],
            measured_nulls=[
                dict(task_id=r["task_id"], path=r["path"], sha256=r["hash"])
                for r in outcomes
                if r["status"] == "null"
            ],
            supporting_evidence=dict(arc_inventory=outcomes[9], historical_boards=outcomes[11]),
            exact_reopen_conditions=[item["continue_if"] for item in gaps.values()],
        ),
        **prior.publication_operands(publication),
    )
    for item in retirements:
        if item["new_terminal_verdict"] == "pending_capstone":
            item["new_terminal_verdict"] = verdict
        unchanged = (
            item["prior_primary"]["verified"]
            and item["declared_prior"]["verdict"] == item["new_terminal_verdict"]
        )
        item.update(
            unchanged_prior_verdict=unchanged,
            decision="retire_unchanged_scope" if unchanged else "retain_changed_contract_scope",
        )
    value["reproducibility_checksum"] = canonical_hash(
        dict(sources=sources, seed=7978, code_sha256=sha256_file(Path(__file__)))
    )
    value["phase_spans"] = [
        dict(
            phase="authority_and_reduction",
            start_s=0,
            end_s=value["duration_s"],
            completed_units=13,
        )
    ]
    value["field_principles"] = {
        key: "Bind exact producer bytes; measurement validity does not prove scientific benefit."
        for key in value
    }
    value["field_principles"].update(
        {
            f"acceptance_gate_results.{key}": "Report this gate separately; missing measurements remain unknown."
            for key in value["acceptance_gate_results"]
        }
    )
    return value


def cold_replay(value: dict[str, Any], root: Path, design: Path, active: Path) -> list[str]:
    """Recompute claims and compare all frozen identities, checkpoint results and hashes."""
    frozen = [
        {key: row[key] for key in ("path", "sha256", "identity", "historical_timestamps")}
        for row in value["producer_date_rows"]
    ]
    expected = build_candidate(
        root,
        design,
        active,
        "20261001",
        value.get("publication_gate_results"),
        snapshots=Path(
            value["authority_snapshots"]["active"].get("snapshot_path")
            or root / "results/raw/experiment_7978_v691_capstone/authority/absent.bin"
        ).parent,
        invocations=frozen,
        seal_rows=False,
    )
    errors = [f"{key}_changed" for key in REPLAY_FIELDS if value.get(key) != expected[key]]
    for source in value["source_artifact_hashes"]:
        path = Path(source["path"])
        if (sha256_file(path) if path.is_file() else None) != source["sha256"]:
            errors.append("source_bytes_changed")
    if any(
        value.get(key) != val
        for key, val in prior.publication_operands(
            value.get("publication_gate_results", {})
        ).items()
    ):
        errors.append("publication_operands_changed")
    return sorted(set(errors))
