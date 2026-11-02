"""REQ-REPORT-8724, REQ-VERIFY-8724: bind current clocks to the original core."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from carnot.reporting import learning_core_8696 as original
from carnot.reporting import learning_core_runner_8696 as original_runner
from carnot.reporting import v748_collection_contract as contract
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
e = contract.namespace(original, "clock_core")
runtime = contract.namespace(original_runner, "clock_core_runner")
e.NAME, e.TASK = "experiment_8724_v748_invocation_clock_core", "exp8724-invocation-clock-core"
e.CLI = f"scripts/experiments/{e.NAME}.py"
e.OWNED = ["python/carnot/reporting/invocation_clock_core_8724.py", e.CLI]
e.TESTS = ["tests/python/test_invocation_clock_8724.py"]
e.REQUIRED = e.REQUIRED | {"deadline_smoke"}
e.authority = SimpleNamespace(
    TASK=original.authority.TASK,
    original=original.authority.original,
    authority=contract.m.authority,
    terminal=original.authority.terminal,
    gate=contract.gate,
)
runtime.e, runtime.RUN_DATE = e, "20261102"
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Keep current phases visible while shared supervision retains child heartbeats."""
    print(f"[exp8724] phase={phase} completed={completed} pending={pending}", flush=True)


def fixture(reference: Json, destination: Path) -> Json:
    """Retain original science and clocks; only current source identity is rebound."""
    value = e.c.read(reference)
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "stream_rows.json",
        "released_feedback.json",
        "retention_rows.json",
        "coefficient_transitions.json",
    ):
        e.shutil.copyfile(Path(value["raw"]) / name, destination / name)
    value["original_clock_provenance"] = {
        key: value[key] for key in ("started_monotonic_ns", "ended_monotonic_ns")
    }
    value["historical_code_provenance"] = value["code_refs"]
    value["code_refs"] = [e.c.ref(Path(row["path"])) for row in value["code_refs"]]
    value["raw"] = str(destination)
    path = destination / "measurement.json"
    e.atomic_json(path, value)
    return dict(e.c.ref(path))


def prepare(raw: Path, run_date: str, upstream: Path = e.UPSTREAM) -> Json:
    """Keep the reusable seal and fixture alive independently of private scratch."""
    with (
        patch.object(
            e,
            "freeze_sources",
            lambda sources, checkout: original.freeze_sources(sources, raw / "source_bundle"),
        ),
        patch.object(
            e, "fresh_fixture", lambda reference, destination: fixture(reference, raw / "fixture")
        ),
    ):
        work: Json = original_prepare(raw, run_date, upstream)
    return work


def execute(work: Json) -> None:
    """Original core execution precedes a disjoint three-node deadline smoke child."""
    original_execute(work)
    progress("before_original_deadline_smoke", 39, 3)
    nodes = [
        "tests/python/test_deadline_edge_8603.py::test_real_deadline_edges" + suffix
        for suffix in ("[3599.0-2-None]", "[3600.0-3-3600.25]", "[3601.0-3-3601.25]")
    ]
    sources = [e.c.ref(e.ROOT / e.core.old_report.RUNNER)]
    with patch.dict(e.os.environ, CARNOT_8603_CASE_ROOT=str(Path(work["raw"]) / "deadline_cases")):
        e.instrument(work, "deadline_smoke", nodes, sources, 300, "smoke")
    progress("after_original_deadline_smoke", 42, 0)


def recompute(work: Json, scratch: Path) -> None:
    """Smoke evidence has the same independent SQLite custody checks as core evidence."""
    original_recompute(work, scratch)
    for child in work["children"]:
        if child["kind"] == "smoke":
            e.core.attest([child], child["sources"])


def build(work: Json, output: Path) -> Json:
    """Report execution readiness separately from scientific learning or hardware benefit."""
    value: Json = original_build(work, output)
    historical = e.c.read(work["original_mechanics_fixture"])
    children = work["children"]
    clocks = [
        dict(
            child=row["name"],
            start_monotonic_ns=row["started_monotonic_ns"],
            end_monotonic_ns=row["ended_monotonic_ns"],
            duration_s=row["duration_s"],
            exit_code=row["exit_code"],
            passed=row["passed"],
        )
        for row in work["receipts"]
    ]
    value.update(
        experiment_id=8724,
        task_id=e.TASK,
        milestone="2026.11.748",
        random_seed=7488724,
        honest_verdict=f"complete_{value['verdict_class']}_v748_invocation_clock_core",
        source_seal=work["frozen_source_bundle"],
        core_node_rows=value["rows"],
        current_clock_rows=clocks,
        original_clock_provenance={
            key: historical[key] for key in ("started_monotonic_ns", "ended_monotonic_ns")
        },
        retained_child_rows=children,
        source_seal_reuse_owner="exp8725-learning-union",
        diagnosed_cross_epoch_case=dict(
            retained_start_monotonic_ns=497771539033492,
            diagnosed_end_monotonic_ns=232478218800900,
            invalid_duration_s=(232478218800900 - 497771539033492) / 1e9,
            status="regression_operand_not_current_measurement",
        ),
        canonical_task_digest=contract.m.PIN,
        preserved_historical_limits=value["preserved_historical_limits"]
        + [
            "V747: Exp8709 collection failure; skipped learning is unmeasured.",
            "Exp8712 constructed collisions; Exp8715 unequal probe counts.",
            "Exp8717 batching 1.138x; same-batch native/Python 1.005x, below 10x.",
            "Exp8719 exact signed32/24 parity; no speed or physical acceleration.",
            "CUDA repair absent; no new ARC outcomes.",
        ],
    )
    value["field_principles"].update(
        {
            key: "Bind current execution to original evidence; readiness grants no learning benefit."
            for key in value
        }
    )
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


original_prepare, original_execute, original_build = e.prepare, e.execute, e.build
original_recompute = e.recompute
e.prepare, e.execute, e.build, e.progress = prepare, execute, build, progress
e.recompute = recompute
main, replay = runtime.main, runtime.replay
