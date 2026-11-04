"""REQ-REPORT-8772: ordinary lifecycle qualification does not imply a cost benefit."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot.reporting import bounded_durable_cost_8745 as previous
from carnot.reporting import lifecycle_archive_8771 as archive
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
__all__ = ["previous", "atomic_json", "canonical_hash", "reference", "verify", "read", "projection"]
ROOT, ARMS = previous.ROOT, previous.ARMS
NAME, TASK = "experiment_8772_v751_ordinary_durable_cost", "exp8772-ordinary-update-cost"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/ordinary_durable_cost_8772.py",
    "python/carnot/reporting/ordinary_durable_runner_8772.py",
    CLI,
]
TESTS = ["tests/python/test_ordinary_durable_cost_8772.py"]
SEED = 7518772
MODEL_SPECS: list[Json] = []
UPSTREAM = ROOT / "results/experiment_8771_v751_lifecycle_archive_qualification.json"
PIN = "sha256:b53d1d20698ca082eb2dcc439d4b3e718e22203d63710f2f3f7c074357890401"
METHOD = dict(
    previous.METHOD,
    bootstrap_seed=SEED,
    natural_renormalization_required=False,
    protocol=archive.e.METHOD,
)
reference, verify, read, projection = (
    previous.reference,
    previous.verify,
    previous.read,
    previous.projection,
)
base = namespace(previous, "ordinary8772")
base.NAME, base.TASK, base.METHOD, base.SEED = NAME, TASK, METHOD, SEED


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so disk waits and child launches remain visible."""
    print(f"[exp8772] phase={phase} completed={completed} pending={pending}", flush=True)


def roster() -> list[Json]:
    """Rotate arm order within each repeat without selecting cells by their costs."""
    rows = previous.roster()
    for row in rows:
        row["order"] = (ARMS.index(row["arm"]) - row["repeat"] - row["offset"] // 2032) % 3
    return sorted(rows, key=lambda r: (r["repeat"], r["offset"], r["order"]))


def failure(
    ref: Json,
    field: str,
    expected: Any,
    observed: Any,
    state: str = "observed_value",
    operator: str = "==",
) -> Json:
    """A failed gate names the exact upstream operand and comparison."""
    return dict(
        previous.failure(ref, field, expected, observed, state=state, operator=operator),
        upstream="exp8771-lifecycle-archive-qualification",
        check="ordinary_durable_admission",
    )


def inputs() -> Json:
    """Threshold readiness is deliberately absent from ordinary admission checks."""
    value: Json = dict(failures=[], refs=[], qualified={}, admitted=False)
    operand = dict(path=str(UPSTREAM), sha256=PIN)
    field = "ordinary_window_ready_score"
    expected: Any = 1
    try:
        qualified = read(operand)
        value["refs"].append(operand)
        if field not in qualified:
            value["failures"].append(failure(operand, field, expected, None, "missing_field"))
            return value
        if qualified[field] != 1:
            value["failures"].append(failure(operand, field, 1, qualified[field]))
            return value
        field, expected = "eligible_producer", True
        if (
            not qualified["required_checks_passed"]
            or qualified["flagged_adversarial"]
            or qualified["private_control"]
        ):
            value["failures"].append(failure(operand, field, True, False, "rejected_producer"))
            return value
        terminal = read(reference(Path(qualified["terminal_validation_sidecar_path"])))
        bound = read(reference(Path(terminal["publication"]["sidecar_path"])))
        if (
            terminal["publication"]["primary_sha256"] != PIN
            or bound["primary_sha256"] != PIN
            or not bound["report"]["passed"]
        ):
            raise ValueError("upstream_terminal_binding")
        work = read(qualified["primitive_reference"])
        protocol = read(qualified["cost_protocol_reference"])
        field, expected = "exact_cost_protocol", METHOD["protocol"]
        if protocol != expected:
            value["failures"].append(
                failure(qualified["cost_protocol_reference"], field, expected, protocol)
            )
            return value
        refs = qualified["code_config_hashes"] + work["source_refs"] + work["raw_refs"]
        refs += [
            qualified["primitive_reference"],
            qualified["validation_reference"],
            qualified["cost_protocol_reference"],
        ]
        refs += [
            reference(Path(qualified["terminal_validation_sidecar_path"])),
            reference(Path(terminal["publication"]["sidecar_path"])),
        ]
        field, expected = "source_and_binary_identity", "authenticated"
        for operand in refs:
            verify(operand)
        retained = previous.inputs()
        if retained["failures"]:
            value["failures"] += retained["failures"]
            return value
        value.update(retained)
        value["refs"] += refs + [dict(path=str(UPSTREAM), sha256=PIN)]
        operand = work["native_runtime_reference"]
        verify(operand)
        value.update(qualified=qualified, extension=operand, admitted=True)
    except (OSError, ValueError, KeyError, TypeError) as error:
        value["failures"].append(
            failure(
                operand,
                field,
                expected,
                str(error),
                "missing_file"
                if isinstance(error, FileNotFoundError)
                else "invalid_or_changed_operand",
            )
        )
    return value


def starting_states(admitted: Json, raw: Path) -> tuple[list[Json], list[Json], Json]:
    """Time arithmetic prefix reconstruction separately from complete durable windows."""
    starts, costs, _ = previous.starting_states(admitted, raw)
    archived = {r["offset"]: r for r in admitted["qualified"]["ordinary_starting_state_rows"]}
    for row in starts:
        if read(row["reference"]) != read(archived[row["offset"]]["reference"]):
            raise ValueError("ordinary_state_identity")
    return starts, costs, dict(passed=True)


def request(admitted: Json, offset: int, arm: str, updates: int = 32) -> Json:
    """Every worker imports the same authenticated state, counters and job."""
    saved = next(
        r for r in admitted["qualified"]["ordinary_starting_state_rows"] if r["offset"] == offset
    )
    return dict(
        job=admitted["job"],
        snapshot=read(saved["reference"]),
        extension=admitted["extension"],
        arm=arm,
        start=offset,
        stop=offset + updates,
    )


def check_window(summary: Json, updates: int) -> bool:
    """Two commits each require snapshot/pointer file and directory barriers."""
    if (
        summary["updates"] != updates
        or summary["durable_commits"] != 2 * updates
        or summary["fsync_calls"] < 8 * updates
    ):
        raise ValueError("durable_window")
    return True


def intervals(rows: list[Json], draws: list[list[int]]) -> Json:
    """Resample paired repeats together across all windows and arms."""
    if len(rows) != 90 or any(r["status"] != "completed" for r in rows):
        return {}
    lookup = {(r["repeat"], r["offset"], r["arm"]): r for r in rows}
    result = {}
    for comparator in ["json_native", "python"]:
        ratios, tails = [], []
        for repeat in range(10):
            ratios.append(
                [
                    lookup[repeat, offset, comparator]["complete_ns"]
                    / lookup[repeat, offset, "resident_native"]["complete_ns"]
                    for offset in METHOD["offsets"]
                ]
            )
            tails.append(
                [
                    lookup[repeat, offset, "resident_native"]["p95_ns"]
                    / lookup[repeat, offset, comparator]["p95_ns"]
                    for offset in METHOD["offsets"]
                ]
            )
        paired = np.asarray(ratios)
        samples = paired[np.asarray(draws)].mean(axis=(1, 2))
        result[comparator] = dict(
            complete_throughput_ratio=float(paired.mean()),
            lower95=float(np.quantile(samples, 0.025)),
            upper95=float(np.quantile(samples, 0.975)),
            p95_ratio=float(np.max(np.mean(tails, axis=0))),
            bootstrap_draws=len(draws),
            independent_scientific_cases=1,
        )
    return result


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Execution acceptance remains separate from the bounded scientific comparison."""
    value: Json = base.build(work, receipts, output)
    rows = work["window_rows"]
    paired = intervals(rows, read(reference(Path(work["raw"]) / "bootstrap.json"))["draws"])
    complete = sum(r["status"] == "completed" for r in rows)
    owned = all(r["passed"] for r in receipts)
    complete_updates = all(
        len(r["update_rows"]) == 32
        and all(u["status"] == "completed" and u["passed"] for u in r["update_rows"])
        and sum(s["durable_commits"] for s in r["segments"]) == 64
        and sum(s["fsync_calls"] for s in r["segments"]) >= 256
        for r in rows
    )
    ready = owned and complete == 90 and complete_updates and not work["failures"]
    native = paired.get("json_native", {})
    benefit = (
        ready and native.get("lower95", 0) > 1.05 and native.get("p95_ratio", float("inf")) <= 1.05
    )
    verdict = (
        "disqualified"
        if not owned
        else "blocked"
        if not ready
        else "positive"
        if benefit
        else "null"
    )
    updates = [u for row in rows for u in row["update_rows"]]
    value.update(
        experiment_id=8772,
        milestone="2026.11.751",
        rows=rows,
        honest_verdict="complete_" + verdict + "_ordinary_durable_cost",
        verdict_class=verdict,
        completed_count=complete,
        completed_window_count=complete,
        failed_count=sum(r["status"] == "failed" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        independent_count=int(complete > 0),
        inference_substrate="deterministic_verifier",
        inference_substrate_class="no_model_load",
        ordinary_update_cost_ready_score=int(ready),
        bounded_durable_benefit_score=int(benefit),
        bounded_update_cost_ready_score=int(ready),
        bounded_resident_benefit_score=int(benefit),
        per_window_rows=rows,
        per_update_rows=updates,
        prefix_preparation_cost_rows=work["excluded_prefix_cost_rows"],
        durability_rows=[
            dict(
                unit_id=r["unit_id"],
                observed_updates=r["observed_updates"],
                durable_commits=sum(s["durable_commits"] for s in r["segments"]),
                fsync_calls=sum(s["fsync_calls"] for s in r["segments"]),
                passed=r["status"] == "completed",
            )
            for r in rows
        ],
        paired_repeat_intervals=paired,
        natural_dense_event_count=sum(
            u.get("renormalization_writes", 0) > 0 for u in updates if u["status"] == "completed"
        ),
        numerical_training=dict(performed=False, arithmetic_prefix_replay_only=True),
        repository_health_reference=work.get("repository_health_reference", {}),
        scoped_validation_scope=dict(
            owned=OWNED,
            tests=TESTS,
            named_e2es=["E2E-003", "E2E-004", "E2E-018", "E2E-021"],
            full_suite_pass_claimed=False,
        ),
        methodology_note="Fixed 90-cell ordinary durable study; two shipped Store commits per update. Complete window includes setup, import, prescribed process restart and shutdown. Arithmetic preparation is separate. Zero natural dense events cannot be replaced by threshold controls. No uninterrupted throughput, useful learning, natural transfer or physical acceleration claim.",
    )
    value["acceptance_gates"] = dict(
        owned_validation=owned,
        qualified_ordinary_states=bool(work["starting_state_rows"]),
        complete_windows=complete == 90,
        resource_admission=work["resource_projection"].get("admitted", False),
        bounded_durable_benefit=bool(benefit),
    )
    if verdict == "blocked" and not value["gate_check_summary"]:
        value["gate_check_summary"] = [
            dict(
                failure(
                    value["primitive_reference"],
                    "complete_durable_update_evidence",
                    True,
                    complete_updates,
                ),
                upstream=TASK,
            )
        ]
    value["field_principles"].update(
        {
            k: "Bind actual ordinary durable work to sealed bytes; missing costs remain missing and no long-run claim follows."
            for k in value
        }
    )
    return value
