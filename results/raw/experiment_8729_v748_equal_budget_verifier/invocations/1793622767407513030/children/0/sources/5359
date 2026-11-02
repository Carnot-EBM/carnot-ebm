"""REQ-REPORT-7991-V692: keep terminal audit and independent science separate.

Each branch retains its own prerequisites. Historical failures and dispatch
receipts cannot replace a declared scientific primary.
"""

from datetime import UTC, datetime
import gzip
import json
import os
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.reporting import v691_capstone as prior
from carnot.reporting import v692_contract_methods as methods
from carnot.reporting import v692_capstone_reduction as reduction
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)

ROOT = prior.ROOT
BRANCHES = dict(
    fit=(7980, 7982),
    reserved_decisions=(7981, 7982, 7983),
    evidence_ablation=(7982, 7984),
    persistent_learning=(7980, 7981, 7985, 7986),
    issued_confidence=(7980, 7981, 7987),
    service=(7989,),
    acceleration=(7989, 7990),
)
REPLAY_FIELDS = (
    "rows",
    "outcome_rows",
    "independent_reduction_rows",
    "producer_date_rows",
    "scientific_branches",
    "gap_decisions",
    "sample_size_budget",
    "authority_snapshots",
    "canonical_tasks_sha256",
    "activation_confirmed",
    "retirement_decisions",
    "science_ready",
    "reproducibility_checksum",
)
Json = dict[str, Any]


def progress(phase: str, units: int = 0) -> None:
    """Show measured work boundaries without manufacturing runtime."""
    print(f"[exp7991] phase={phase} completed_units={units}", flush=True)


def tasks_from(active: Path) -> list[Json]:
    """Preserve the exact historical roster when a missing authority blocks work."""
    try:
        tasks = yaml.safe_load(active.read_bytes())["tasks"]
        if [t["id"].split("-")[0] for t in tasks] != [f"exp{n}" for n in range(7979, 7992)]:
            raise ValueError("task_roster")
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
        tasks = yaml.safe_load(
            gzip.decompress((ROOT / "tests/fixtures/v692/active.yaml.gz").read_bytes())
        )["tasks"]
    return list(tasks)


def build_candidate(
    root: Path,
    design: Path,
    active: Path,
    date: str,
    publication: Json | None = None,
    *,
    snapshots: Path | None = None,
    invocations: list[Json] | None = None,
) -> Json:
    """Freeze exact inputs before independently reading each declared disposition."""
    if date != "20261001":
        raise ValueError("capstone_date_changed")
    started, start_ns = datetime.now(UTC).isoformat(), time.monotonic_ns()
    progress("freeze_authority")
    durable = root / "results/raw/experiment_7991_v692_capstone"
    try:
        authority = methods.assess(
            design, root / "research-roadmap-next.yaml", active, snapshots or durable / "authority"
        )
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
        authority = prior.prior.fallback_assess(
            design,
            root / "research-roadmap-next.yaml",
            active,
            snapshots or durable / "authority",
            milestone="2026.10.692",
            first_id=7979,
            count=13,
        )
        authority["activated"] = False
    tasks = tasks_from(active)
    frozen = (
        invocations
        if invocations is not None
        else [prior.freeze_producer(root / t["deliverable"]) for t in tasks[:-1]]
    )
    if len(frozen) != 12:
        raise ValueError("producer_invocation_roster")
    sources = [
        dict(path=s[key], sha256=s["sha256"], role=role + "_" + key)
        for role, s in authority["authority_snapshots"].items()
        for key in ("source_path", "snapshot_path")
        if s.get(key)
    ]
    failures, outcomes, reductions, dates, history, retirements = (
        list(authority["gate_check_summary"]),
        [],
        [],
        [],
        [],
        [],
    )
    observed = {
        t["id"]: (*prior.prior.prior.shared.read(root / t["deliverable"]), root / t["deliverable"])
        for t in tasks[:-1]
    }
    for index, task in enumerate(tasks):
        number = 7979 + index
        data, digest, path = observed.get(task["id"], ({}, None, root / task["deliverable"]))
        state = str(data.get("verdict_class", "absent")) if index < 12 else "self_administrative"
        own, dispatch, checkpoint_error = [], {}, None
        if index < 12:
            sources.append(dict(path=str(path), sha256=digest, role="declared_primary"))
            stub_path = (
                root
                / f"results/experiment_{number}_{task['id'].split('-', 1)[1].replace('-', '_')}.json"
            )
            stub, stub_hash = prior.prior.prior.shared.read(stub_path)
            dispatch = dict(
                path=str(stub_path),
                sha256=stub_hash,
                gates_evaluated=stub.get("gates_evaluated", []),
            )
            sources.append(dict(path=str(stub_path), sha256=stub_hash, role="conductor_dispatch"))
            if not data:
                state = "disqualified" if path.is_file() else "skipped" if stub_hash else "absent"
                own.append(
                    reduction.operand(
                        path,
                        task["id"],
                        "declared_primary_exists",
                        True,
                        "malformed" if path.is_file() else "missing_source",
                    )
                )
            observed_invocation = prior.freeze_producer(path)
            issued = dict(
                frozen[index],
                passed=all(observed_invocation.get(k) == v for k, v in frozen[index].items()),
            )
            dates.append(issued)
            if not issued["passed"]:
                own.append(
                    reduction.operand(
                        path, task["id"], "frozen_sha256", frozen[index]["sha256"], digest
                    )
                )
            if data:
                for field, expected in dict(
                    experiment_id=number,
                    task_id=task["id"],
                    milestone="2026.10.692",
                    flagged_adversarial=False,
                ).items():
                    if data.get(field, "missing_field") != expected:
                        own.append(
                            reduction.operand(
                                path, task["id"], field, expected, data.get(field, "missing_field")
                            )
                        )
                try:
                    for field in ("run_date", "execution_date"):
                        datetime.strptime(data[field], "%Y%m%d")
                    if data.get("started_at") and data.get("finished_at"):
                        if datetime.fromisoformat(
                            data["finished_at"].replace("Z", "+00:00")
                        ) < datetime.fromisoformat(data["started_at"].replace("Z", "+00:00")):
                            raise ValueError("producer_time_reversed")
                except (KeyError, ValueError, TypeError):
                    own.append(
                        reduction.operand(
                            path,
                            task["id"],
                            "producer_invocation_dates",
                            "valid issued dates",
                            "missing_or_invalid",
                        )
                    )
                history.append(
                    dict(
                        path=str(path),
                        sha256=digest,
                        honest_verdict=data.get("honest_verdict"),
                        failures=data.get("gate_check_summary", []),
                        repository_health=data.get("repository_health"),
                        current_pass=False,
                    )
                )
                for failure in data.get("gate_check_summary", []):
                    label = failure.get("path", failure.get("artifact_path", str(path)))
                    failure_path = Path(label)
                    failure_path = (
                        failure_path if failure_path.is_absolute() else root / failure_path
                    )
                    row = reduction.operand(
                        failure_path,
                        failure.get("upstream_id", task["id"]),
                        failure.get("artifact_field", failure.get("field", "external_gate")),
                        failure.get("expected"),
                        failure.get("observed", failure.get("actual", "missing_field")),
                        failure.get("op", "=="),
                    )
                    row["producer_sha256"] = digest
                    failures.append(row)
                for ref in reduction.references(data):
                    p = Path(ref["path"])
                    p = p if p.is_absolute() else root / p
                    actual = sha256_file(p) if p.is_file() else None
                    sources.append(
                        dict(
                            path=str(p),
                            sha256=actual,
                            expected_sha256=ref["sha256"],
                            role=ref["role"],
                        )
                    )
                    if actual != ref["sha256"]:
                        own.append(
                            reduction.operand(p, task["id"], "sha256", ref["sha256"], actual)
                        )
        gates = []
        for gate in task.get("gated_on", []):
            upstream, up_hash, up_path = observed[gate["upstream"]]
            actual = upstream.get(
                gate["artifact_field"], "missing_field" if upstream else "missing_source"
            )
            if not prior.prior._eval_op(actual, gate["op"], gate["value"])[0]:
                gates.append(
                    reduction.operand(
                        up_path,
                        gate["upstream"],
                        gate["artifact_field"],
                        gate["value"],
                        actual,
                        gate["op"],
                    )
                )
            if not next(r for r in outcomes if r["task_id"] == gate["upstream"])["eligible"]:
                gates.append(
                    reduction.operand(
                        up_path, gate["upstream"], "authenticated_producer_eligible", True, False
                    )
                )
        try:
            reduced = reduction.audit(number, data, root)
        except (OSError, ValueError, KeyError, TypeError) as error:
            checkpoint_error = str(error)
            reduced = dict(
                primitive_rows=len(data.get("rows", [])),
                independent_sources=0,
                independent_benefit=False,
                unavailable_reason=checkpoint_error,
            )
            try:
                reduced.update(reduction.reduce_rows(data.get("rows", [])))
            except ValueError:
                pass
            own.append(
                reduction.operand(
                    path,
                    task["id"],
                    "primitive_checkpoint_replay",
                    "valid rows and source bytes",
                    checkpoint_error,
                )
            )
        eligible = int(
            state in {"positive", "circular_positive", "null"}
            and not own
            and not gates
            and authority["activated"]
        )
        failures.extend(own + gates)
        outcomes.append(
            dict(
                task_id=task["id"],
                upstream_id=task["id"],
                path=str(path),
                hash=digest,
                status=state,
                eligible=eligible,
                gate_blocked=bool(gates),
                intended=1,
                started=1,
                completed=1,
                failed=0,
                censored=0,
                excluded=0,
                independent=0,
                arm="administrative_disposition",
                seed=7991,
                conductor_gate_receipt=dispatch,
                producer_budget=data.get("sample_size_budget"),
                gate_check_summary=own + gates,
            )
        )
        reductions.append(
            dict(
                upstream_id=task["id"],
                path=str(path),
                sha256=digest,
                scientifically_eligible=bool(eligible),
                primitive_rows_sha256=canonical_hash(data.get("rows", [])),
                **reduced,
            )
        )
        for declared in task.get("prior_failures", []):
            old = prior.prior_evidence(root, declared)
            if old["path"]:
                sources.append(
                    dict(path=old["path"], sha256=old["sha256"], role="historical_terminal")
                )
            same = old["verified"] and old["honest_verdict"] == data.get("honest_verdict")
            old_data = (
                json.loads(Path(old["path"]).read_bytes())
                if old["path"] and old["verified"]
                else {}
            )
            same_scope = (
                bool(data.get("claim_scope"))
                and old_data.get("claim_scope") == data.get("claim_scope")
                and declared["experiment_id"].split("-", 1)[-1] == task["id"].split("-", 1)[-1]
            )
            if number == 7989:
                same_scope = same_scope and {r.get("arm") for r in old_data.get("rows", [])} == {
                    r.get("arm") for r in data.get("rows", [])
                }
            retirements.append(
                dict(
                    upstream_id=task["id"],
                    declared_prior=declared,
                    prior_primary=old,
                    current_sha256=digest,
                    current_path=str(path),
                    unchanged_prior_verdict=same,
                    unchanged_claim_scope=same_scope,
                    decision="retire_unchanged_scope"
                    if same and same_scope and declared.get("retire_if_same_verdict")
                    else "retain_changed_scope",
                    forward_difference=declared.get("addressed_by"),
                )
            )
        progress("reduce_disposition", index + 1)
    branches = {
        name: dict(
            required_producers=list(numbers),
            decision="measured-null"
            if all(outcomes[n - 7979]["eligible"] for n in numbers)
            else "blocked",
            independent_benefit=False,
            reductions=[reductions[n - 7979] for n in numbers],
        )
        for name, numbers in BRANCHES.items()
    }
    gaps = {
        "FR-12/FR-06": dict(
            decision=branches["reserved_decisions"]["decision"],
            fit_decision=branches["fit"]["decision"],
            ablation_decision=branches["evidence_ablation"]["decision"],
            independent_benefit=False,
            continue_if="Authenticate reserved source exposure and fresh capture; pass same-source cost and Brier gates against classical controls.",
            retire_if="Retire only an exact unchanged declared verdict and scope; feature transport and fitted heads alone prove no independent benefit.",
        ),
        "FR-11": dict(
            decision=branches["persistent_learning"]["decision"],
            issued_confidence_decision=branches["issued_confidence"]["decision"],
            independent_benefit=False,
            causal_label_reads=None,
            restart_equality=None,
            retention=None,
            continue_if="Qualified stream and acquisition primaries must prove released-label custody, restart equality, future cost benefit and retention against no-write and shuffled controls.",
            retire_if="Retire unchanged failed learning scopes; durable writes alone do not prove future benefit.",
        ),
        "FR-05/FR-08/NFR-01": dict(
            decision=branches["acceleration"]["decision"],
            service_decision=branches["service"]["decision"],
            independent_benefit=False,
            continue_if="Authenticate all service dependencies and durable learning work; measure a compatible device kernel with transfer and whole-service spans.",
            retire_if="Retire unchanged probe scopes with no compatible kernel; preserve historical fabric limits and processor-only evidence.",
        ),
    }
    science_ready = all(branch["decision"] == "measured-null" for branch in branches.values())
    publication = publication or {}
    value = build_current_work_receipt(
        run_id="exp7991-" + date,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        execution_venue="host",
        inference_substrate_details=dict(work="primitive and checkpoint audit"),
        started_monotonic_ns=start_ns,
        ended_monotonic_ns=time.monotonic_ns(),
    )
    code = [
        dict(path=str(Path(__file__)), sha256=sha256_file(Path(__file__)), role="current_code"),
        dict(
            path=str(Path(reduction.__file__)),
            sha256=sha256_file(Path(reduction.__file__)),
            role="current_code",
        ),
    ]
    value.update(
        experiment_id=7991,
        task_id="exp7991-capstone",
        milestone="2026.10.692",
        run_date=date,
        execution_date=date,
        started_at=started,
        finished_at=datetime.now(UTC).isoformat(),
        honest_verdict="complete_null_independent_benefit_unshown"
        if science_ready
        else "complete_blocked_missing_qualified_science",
        verdict_class="null" if science_ready else "blocked",
        flagged_adversarial=False,
        gate_check_summary=failures,
        preconditions_checked=failures,
        rows=outcomes,
        outcome_rows=outcomes,
        independent_reduction_rows=reductions,
        producer_date_rows=dates,
        scientific_branches=branches,
        gap_decisions=gaps,
        authority_snapshots=authority["authority_snapshots"],
        activation_confirmed=authority["activated"],
        canonical_tasks_sha256=authority["canonical_tasks_sha256"],
        retirement_decisions=retirements,
        next_reopen_conditions=[g["continue_if"] for g in gaps.values()],
        sample_size_budget=dict(
            intended=13,
            eligible=sum(r["eligible"] for r in outcomes),
            started=13,
            completed=13,
            failed=0,
            censored=0,
            excluded=0,
            independent=0,
            unit="task_disposition",
        ),
        source_artifact_hashes=sources,
        raw_shard_hashes=[r for r in sources if r["role"] == "raw_shard_hashes"],
        code_config_hashes=code,
        cited_upstream_artifacts=[r for r in sources if r["role"] == "declared_primary"],
        historical_required_failures=history,
        repository_health=dict(current_pass=False, historical_only=True),
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(loads=0, calls=0, tokens=0),
        trained_head_specs=[],
        target_model="none",
        random_seed=7991,
        verifier_is_oracle=False,
        claim_scope=dict(
            independent_benefit=False,
            natural_data="exposed_development",
            reserved_data="exposure audit required; unavailable independent evaluation",
            fixture_agreement="circular_positive",
            generator_weights_changed=False,
            production_defaults_changed=False,
        ),
        science_ready=science_ready,
        capstone_execution_ready_score=0,
        acceptance_gate_results=dict(
            validity=False, readiness=0, decision_benefit=False, retention=None, efficiency=None
        ),
        validation_command_manifest_path=None,
        validation_receipts=[],
        coverage_statement_counts={},
        terminal_validation_sidecar_path=None,
        publication_gate_results=publication,
        primary_resolution_receipt=dict(path=str(durable / "primary_resolution_receipt.json")),
        **prior.prior.publication_operands(publication),
    )
    value["reproducibility_checksum"] = canonical_hash(dict(sources=sources, code=code, seed=7991))
    value["phase_spans"] = [
        dict(
            phase="authority_and_independent_reduction",
            start_s=0,
            end_s=value["duration_s"],
            completed_units=13,
        )
    ]
    value["field_principles"] = {
        k: "Bind measured observations to source bytes; audit completion does not prove benefit."
        for k in value
    }
    return value


def cold_replay(value: Json, root: Path, design: Path, active: Path) -> list[str]:
    """Reconstruct science and original authority without accepting changed aggregates."""
    progress("cold_replay_start")
    for source in value["source_artifact_hashes"] + value["code_config_hashes"]:
        path = Path(source["path"])
        if (sha256_file(path) if path.is_file() else None) != source["sha256"]:
            return ["source_bytes_changed"]
    frozen = [
        {k: r[k] for k in ("path", "sha256", "identity", "historical_timestamps")}
        for r in value["producer_date_rows"]
    ]
    expected = build_candidate(
        root,
        design,
        active,
        "20261001",
        value["publication_gate_results"],
        snapshots=Path(
            value["authority_snapshots"]["active"].get("snapshot_path")
            or root / "results/raw/experiment_7991_v692_capstone/authority/absent.bin"
        ).parent,
        invocations=frozen,
    )
    errors = [
        f"{key}_changed"
        for key in REPLAY_FIELDS
        if value.get(key) != expected[key]
        and not (key == "science_ready" and value["verdict_class"] == "disqualified")
    ]
    if any(
        value.get(k) != v
        for k, v in prior.prior.publication_operands(value["publication_gate_results"]).items()
    ):
        errors.append("publication_operands_changed")
    if value["verdict_class"] == "disqualified":
        if not any(
            not r["passed"] and r.get("classification") != "diagnostic"
            for r in value["validation_receipts"]
        ):
            errors.append("unsubstantiated_disqualification")
    elif value["verdict_class"] != expected["verdict_class"]:
        errors.append("verdict_class_changed")
    if (
        value["verdict_class"] in {"blocked", "disqualified"}
        and value["capstone_execution_ready_score"]
    ):
        errors.append("unsafe_readiness")
    return sorted(set(errors))
