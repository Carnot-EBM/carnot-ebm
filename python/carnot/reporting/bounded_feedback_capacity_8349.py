"""REQ-REPORT-8349: readiness describes constructed mechanics, never natural benefit."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any

import yaml

from carnot.reporting import v720_frozen_input_contract as authority
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary
from carnot.verify import bounded_feedback_capacity_8349 as k

Json = dict[str, Any]
ROOT = k.kernel.ROOT
NAME, TASK = "experiment_8349_v720_bounded_feedback_capacity", "exp8349-bounded-feedback-capacity"
MILESTONE = "2026.10.720"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_bounded_feedback_capacity_8349.py"
OWNED = [
    "python/carnot/verify/bounded_feedback_capacity_8349.py",
    "python/carnot/reporting/bounded_feedback_capacity_8349.py",
    "python/carnot/reporting/bounded_feedback_execution_8349.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
UPSTREAM = "results/experiment_8347_v720_local_consumer_qualification.json"
PIN = "sha256:0297d25cbdc6d3133a5bc9a93a40d16d2e863b017969687d384e595c94ff7a37"


def reference(path: Path) -> Json:
    """Byte hashes bind readers to specific evidence rather than its mutable filename."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def require(work: Json, path: Path, field: str, expected: Any, observed: Any) -> None:
    """Missing operands remain None and name the exact failing upstream operator."""
    row = dict(
        upstream=path.stem,
        path=str(path),
        hash=sha256_file(path) if path.is_file() else None,
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=expected == observed,
    )
    work["gates"].append(row)
    if not row["passed"]:
        work["failures"].append(row)
        raise ValueError(field)


def bind(work: Json, path: Path, raw: Path, expected: str | None = None) -> Json:
    """Snapshot source bytes before reuse; absence cannot be replaced with a fixture."""
    require(
        work,
        path,
        "source_bytes",
        expected or True,
        (sha256_file(path) if expected else True) if path.is_file() else None,
    )
    saved = raw / "inputs" / (sha256_file(path)[7:] + "-" + path.name)
    saved.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, saved)
    work["refs"].append(dict(reference(saved), source_path=str(path)))
    return dict(json.loads(saved.read_bytes())) if path.suffix == ".json" else {}


def preconditions(root: Path, raw: Path) -> Json:
    """Authenticate current permission separately from historical numeric provenance."""
    k.progress("preconditions_before")
    raw.mkdir(parents=True, exist_ok=True)
    raw.chmod(0o700)
    work: Json = dict(gates=[], failures=[], refs=[], authority={}, historical=[])
    for tool in ["python", "pytest", "coverage", "ruff", "mypy"]:
        if not os.access(ROOT / ".venv/bin" / tool, os.X_OK):
            raise OSError("missing executable: " + tool)
    if shutil.disk_usage(raw).free < 1024**3:
        raise OSError("private storage below 1GiB")
    try:
        source = root / UPSTREAM
        upstream = bind(work, source, raw, PIN)
        validate_primary(upstream, source)
        terminal = bind(work, Path(upstream["terminal_validation_sidecar_path"]), raw)
        side = Path(terminal["publication"]["sidecar_path"])
        report = read_bound_sidecar(source, side)
        bind(work, side, raw)
        for field, expected, observed in [
            ("terminal_passed", True, report["report"]["passed"]),
            (
                "terminal_primary_sha256",
                sha256_file(source),
                terminal["publication"]["primary_sha256"],
            ),
            ("local_kernel_ready_score", 1, upstream.get("local_kernel_ready_score")),
            ("required_checks_passed", True, upstream.get("required_checks_passed")),
            ("flagged_adversarial", False, upstream.get("flagged_adversarial")),
        ]:
            require(work, source, field, expected, observed)
        work["historical"] = upstream["historical_model_provenance"]
        work["authority"] = authority.authority(root, raw / "authority")
        task = next(v for v in work["authority"]["tasks"] if v["id"] == TASK)
        require(
            work,
            root / authority.ACTIVE,
            "exact_task_authority",
            True,
            work["authority"]["activated"]
            and task["deliverable"] == f"results/{NAME}.json"
            and task["MODEL_SPECS"] == []
            and task["gated_on"]
            == [
                dict(
                    upstream="exp8347-local-consumer-qualification",
                    artifact_field="local_kernel_ready_score",
                    op="==",
                    value=1,
                )
            ],
        )
        for path in [
            authority.DESIGN,
            authority.ACTIVE,
            authority.PROTOCOL,
            "ops/exclusion_manifest.yaml",
            "python/carnot/verify/local_update_isolation_8306.py",
        ]:
            bind(work, root / path, raw, authority.base.PIN if path == authority.PROTOCOL else None)
        policy = yaml.safe_load((root / "ops/exclusion_manifest.yaml").read_bytes())
        retired = [
            v
            for key in ["retired_experiments", "retired_extras"]
            for v in policy.get(key, [])
            if v.get("experiment_id") in [8349, TASK] or TASK in str(v.get("experiment_scope", ""))
        ]
        require(work, root / "ops/exclusion_manifest.yaml", "not_retired", [], retired)
    except (OSError, ValueError, KeyError, StopIteration, TypeError, yaml.YAMLError) as error:
        if not work["failures"]:
            work["failures"].append(
                dict(
                    upstream="external_authentication",
                    path=str(root / UPSTREAM),
                    hash=None,
                    artifact_field="authenticated_operands",
                    op="==",
                    expected=True,
                    observed=str(error),
                    passed=False,
                )
            )
    k.progress("preconditions_after", 1, 0)
    return work


def measure(root: Path, raw: Path) -> Json:
    """Measure all frozen units, including hard exits, without any model inference."""
    from carnot.reporting import bounded_feedback_execution_8349 as runner

    began = time.monotonic()
    work = preconditions(root, raw)
    plan = k.manifest()
    atomic_json(raw / "protocol.json", plan)
    work.update(
        states=[],
        crashes=[],
        child_receipts=[],
        checks={},
        protocol_reference=reference(raw / "protocol.json"),
    )
    if not work["failures"]:
        try:
            for i, unit in enumerate(plan["units"]):
                k.progress("before_benchmark", i, len(plan["units"]) - i)
                trace = plan["traces"][unit["family"]]
                expected = k.simulate(trace, unit)
                bundle = raw / "bundles" / (unit["id"] + ".json")
                atomic_json(bundle, dict(trace=trace, unit=unit))
                dest = raw / "workers" / unit["id"]
                for slot in [128, 256, 0]:
                    argv = runner.cli() + [
                        "--worker",
                        str(bundle),
                        "--worker-dir",
                        str(dest),
                        "--crash",
                        str(slot),
                    ]
                    receipt = runner.check(
                        dict(
                            name=f"{unit['id']}-{slot}",
                            argv=argv,
                            expected_exit=73 if slot else 0,
                            deadline_s=120,
                        ),
                        raw / "worker_logs",
                    )
                    work["child_receipts"].append(receipt)
                    if not receipt["passed"]:
                        raise ValueError("owned_worker_failure")
                state = json.loads((dest / "final.json").read_bytes())
                work["states"].append(state)
                for slot in [128, 256]:
                    checkpoint = json.loads((dest / f"checkpoint-{slot}.json").read_bytes())
                    work["crashes"].append(
                        dict(
                            unit=unit["id"],
                            slot=slot,
                            crash_exit=73,
                            passed=k.semantic(state) == k.semantic(expected)
                            and k.audit(trace, checkpoint),
                            rng_state_hash=canonical_hash(checkpoint["scheduler_rng_state"]),
                            checkpoint=reference(dest / f"checkpoint-{slot}.json"),
                            uninterrupted_hash=canonical_hash(k.semantic(expected)),
                            resumed_hash=canonical_hash(k.semantic(state)),
                        )
                    )
                k.progress("after_benchmark", i + 1, len(plan["units"]) - i - 1)
            work["checks"] = dict(
                capacity_causal_order=all(
                    k.audit(plan["traces"][s["unit"]["family"]], s) for s in work["states"]
                ),
                exact_restart=all(r["passed"] for r in work["crashes"]),
                sparse_dense=all(
                    s["heads"]["sparse"] == s["heads"]["dense"] for s in work["states"]
                ),
                complete_accounting=len(work["states"]) == len(plan["units"]),
                **k.controls(),
            )
        except (OSError, ValueError, KeyError) as error:
            work["checks"]["owned_measurement"] = False
            work["owned_failure"] = str(error)
    work.update(
        duration_s=time.monotonic() - began,
        preconditions_checked=True,
        phase_spans=[
            dict(
                phase="authentication_and_constructed_measurement",
                start_s=0,
                duration_s=time.monotonic() - began,
            )
        ],
        code_config_hashes=[reference(ROOT / p) for p in OWNED],
    )
    atomic_json(raw / "measurement.json", work)
    return work


def summaries(work: Json) -> list[Json]:
    """Score every issued prediction with evaluator labels, including lost feedback."""
    plan = json.loads(Path(work["protocol_reference"]["path"]).read_bytes())
    rows = []
    for unit in plan["units"]:
        state = next((s for s in work["states"] if s["unit"] == unit), None)
        trace = plan["traces"][unit["family"]]
        for arm in ["sparse", "dense"]:
            loss, cost = [], []
            if state is not None:
                for issue in [v for v in state["events"] if v["kind"] == "issue"]:
                    p = min(1 - 1e-15, max(1e-15, issue["p"][arm]))
                    y = trace["events"][issue["slot"] - 1]["y"]
                    loss.append(-y * math.log(p) - (1 - y) * math.log(1 - p))
                    act = k.kernel.action(p)
                    cost.append(0.5 if act == "escalate" else float((act == "reject") != bool(y)))
            rows.append(
                dict(
                    unit_id=unit["id"] + "-" + arm,
                    **unit,
                    arm=arm,
                    status="completed" if state is not None else "censored",
                    label_scope="constructed",
                    predictions=len(loss),
                    log_loss=sum(loss) / len(loss) if loss else None,
                    typed_action_cost=sum(cost) / len(cost) if cost else None,
                    maximum_pending_count=state["maximum_pending_count"]
                    if state is not None
                    else None,
                    feedback_retained=state["feedback_retained"] if state is not None else None,
                    feedback_lost=state["feedback_lost"] if state is not None else None,
                    **(state["metrics"][arm] if state is not None else {}),
                )
            )
    return rows


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """All intended rows remain visible; constructed utility has no readiness threshold."""
    rows = summaries(work)
    passed = bool(receipts) and all(r["passed"] for r in receipts) and all(work["checks"].values())
    verdict = (
        "disqualified" if not passed else "blocked" if work["failures"] else "circular_positive"
    )
    completed = sum(r["status"] == "completed" for r in rows)
    evidence = [reference(raw / "measurement.json"), work["protocol_reference"]]
    for path in sorted((raw / "workers").rglob("*")):
        if path.is_file():
            evidence.append(reference(path))
    states = work["states"]
    contrasts = []
    for row in [v for v in rows if v["status"] == "completed" and v["policy"] == "random"]:
        comparator = next(
            v
            for v in rows
            if v["family"] == row["family"]
            and v["seed"] == row["seed"]
            and v["capacity"] == row["capacity"]
            and v["arm"] == row["arm"]
            and v["policy"] == "first"
        )
        contrasts.append(
            dict(
                unit_id=row["unit_id"],
                contrast="random_minus_first",
                log_loss_difference=row["log_loss"] - comparator["log_loss"],
                typed_cost_difference=row["typed_action_cost"] - comparator["typed_action_cost"],
                scope="constructed_descriptive_only",
            )
        )
    value: Json = dict(
        experiment_id=8349,
        task_id=TASK,
        milestone=MILESTONE,
        run_date="20261009",
        honest_verdict="complete_" + verdict + "_bounded_feedback_capacity",
        verdict_class=verdict,
        gate_check_summary=work["gates"] + work["failures"],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        no_model_load=True,
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=work["historical"],
        generator_weight_updates=0,
        rows=rows,
        intended_count=len(rows),
        completed_count=completed,
        failed_count=0,
        censored_count=len(rows) - completed,
        excluded_count=0,
        independent_count=len({v["family"] for v in rows if v["status"] == "completed"}),
        sample_size_budget=dict(
            trace_families=3,
            events_per_trace=512,
            admission_seeds=[11, 22, 33],
            capacities=[4, 16, 64],
            policies=["unlimited", "first", "random"],
            implementations=["sparse", "dense"],
            intended_units=126,
        ),
        verifier_is_oracle=True,
        exposure_scope="constructed_exposed_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        acceptance_gates=work["checks"],
        validation_receipts=receipts + work["child_receipts"],
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=work.get("adversarial_findings", []),
        finding_dispositions=work.get("finding_dispositions", []),
        prior_publication_attempts=work.get("prior_publication_attempts", []),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=[11, 22, 33],
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=evidence,
        cited_upstream_artifacts=work["refs"],
        capacity_ready_score=int(verdict == "circular_positive"),
        per_event_rows=[
            dict(
                unit_id=s["unit"]["id"],
                event_count=len(s["events"]),
                primitive_reference=reference(raw / "workers" / s["unit"]["id"] / "final.json"),
            )
            for s in states
        ],
        maximum_pending_count=max((s["maximum_pending_count"] for s in states), default=None),
        feedback_retained=sum(s["feedback_retained"] for s in states),
        feedback_lost=sum(s["feedback_lost"] for s in states),
        scheduler_rng_state=[
            dict(unit_id=s["unit"]["id"], state=s["scheduler_rng_state"]) for s in states
        ],
        crash_replay_rows=work["crashes"],
        trace_family_count=3,
        constructed_utility_rows=contrasts,
        dense_sparse_error_max=0 if states and k.numeric_proof(states)["recomputed"] else None,
        measurement_reference=reference(raw / "measurement.json"),
        authority=work["authority"],
        owned_coverage_reference=work.get("owned_coverage_reference"),
        execution_manifest_reference=work.get("execution_manifest"),
        methodology_note="Three preconstructed traces; seeds randomize admission only. Issue/admit precedes expiration; dropped learner labels are permanently lost. Frozen spline basis and step.01, norm cap1, slope1/intercept0/T1; no IPW. Sparse/dense share retained labels. Cost: correct0, incorrect1, escalate.5. Actual update time includes kernel, encoding, append, flush and fsync; checkpoint persistence is separate. No natural H1/H2 inference or learning benefit is claimed; this is a systems adaptation, not the paper scheduler or DW-FTRL.",
    )
    value["field_principles"] = {
        f: "Bind " + f + " to authenticated bytes, measured primitives and constructed-only scope."
        for f in value
    }
    value["field_principles"].update(
        capacity_ready_score="All causal capacity, ordering, recovery and deliberate-error invariants only; utility gain is unnecessary.",
        per_event_rows="Byte-bound references expose every issue, priority, admission membership, drop, due event and update.",
        independent_count="Only three trace families; seeds, arms and timing repeats are not independent natural datasets.",
        reproducibility_checksum="Canonical terminal claims are bound to exact evidence.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Fresh replay recomputes semantic traces and reduction, including rehashed tampering."""
    try:
        value = json.loads(path.read_bytes())
        ref = value["measurement_reference"]
        raw = Path(ref["path"]).parent
        work = json.loads(Path(ref["path"]).read_bytes())
        for operand in work["refs"] + work["code_config_hashes"] + value["raw_shard_hashes"]:
            if sha256_file(Path(operand["path"])) != operand["sha256"]:
                return False
        plan = json.loads(Path(work["protocol_reference"]["path"]).read_bytes())
        if plan != k.manifest():
            return False
        for state in work["states"]:
            if not k.audit(plan["traces"][state["unit"]["family"]], state):
                return False
            final = json.loads((raw / "workers" / state["unit"]["id"] / "final.json").read_bytes())
            if final != state:
                return False
        own = value["validation_receipts"][
            : len(value["validation_receipts"]) - len(work["child_receipts"])
        ]
        return build(work, raw, own) == value
    except (OSError, ValueError, KeyError, TypeError):
        return False
