"""REQ-REPORT-8520: current qualification permits new clocks, not historical speed."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import append_import_qualification_8519 as qualified
from carnot.reporting import append_service_cost_8494 as shipped
from carnot.reporting import v732_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v721_capstone_evidence import frozen_inputs

Json = dict[str, Any]
ROOT = shipped.ROOT
NAME, TASK = "experiment_8520_v732_append_service_cost", "exp8520-append-service-cost"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_append_service_cost_8520.py"
OWNED = [
    "python/carnot/reporting/append_service_cost_8520.py",
    "python/carnot/reporting/append_service_runner_8520.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
UPSTREAM = "results/experiment_8519_v732_append_import_qualification.json"
UPSTREAM_PIN = "sha256:6e4233edc4865de8ada5a6775429ed04ac379306e5a25f7b4dcfad002f46a904"
REQUIRED = shipped.REQUIRED | {"historical_union"}
store, k = shipped.upstream, shipped.k
reference, read, checksum, progress, GateFailure = (
    shipped.reference,
    shipped.read,
    shipped.checksum,
    shipped.progress,
    store.GateFailure,
)


def require(path: Path, field: str, expected: Any, observed: Any) -> None:
    """Keep exact failed operands so missing evidence cannot be mistaken for a zero."""
    if observed != expected:
        error = GateFailure(field, path, field, expected, observed)
        error.row.update(
            upstream=qualified.TASK,
            expected_sha256=UPSTREAM_PIN if path.name == Path(UPSTREAM).name else None,
        )
        raise error


def preflight(root: Path, raw: Path) -> Json:
    """Reuse checked tools and current task admission while preserving historical pins."""
    with (
        patch.multiple(qualified, TASK=TASK, NAME=NAME),
        patch.multiple(
            shipped,
            TASK="exp8494-append-service-cost",
            NAME="experiment_8494_v730_append_service_cost",
        ),
        patch.multiple(
            store,
            TASK="exp8493-append-store-qualification",
            NAME="experiment_8493_v730_append_store_qualification",
        ),
    ):
        return dict(qualified.preflight(root, raw))


def qualified_replay(path: Path) -> bool:
    """Historical reconstruction writes stay private while unchanged readers verify original bytes."""
    with TemporaryDirectory(dir="/var/tmp", prefix="exp8520-upstream-") as folder:
        with frozen_inputs([], Path(folder)):
            return bool(qualified.replay(path))


def admission(root: Path, raw: Path) -> Json:
    """Authenticate V732 and original five-input operands before permitting fresh disk work."""
    path = root / UPSTREAM
    require(path, "sha256", UPSTREAM_PIN, sha256_file(path) if path.is_file() else None)
    value = read(dict(path=str(path), sha256=UPSTREAM_PIN))
    for field, expected in dict(
        experiment_id=8519,
        task_id=qualified.TASK,
        run_date="20261014",
        append_runner_ready_score=1,
        required_checks_passed=True,
        flagged_adversarial=False,
    ).items():
        require(path, field, expected, value.get(field))
    checks = preflight(root, raw)
    if checks["gate_check_summary"]:
        error = GateFailure("current_preflight", path, "preconditions", True, False)
        error.row = checks["gate_check_summary"][0]
        raise error
    authority = methods.authority(root, raw / "current_authority")
    require(root / methods.ACTIVE, "activated", True, authority["activated"])
    task = next(t for t in authority["tasks"] if t["id"] == TASK)
    require(root / methods.ACTIVE, "deliverable", "results/" + NAME + ".json", task["deliverable"])
    terminal = read(reference(Path(value["terminal_validation_sidecar_path"])))
    side = read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))
    require(path, "bound_sidecar_passed", True, side["report"]["passed"])
    require(
        path,
        "typed_consumer_passed",
        True,
        reader_receipt(qualified.TASK, path.parent, field="append_runner_ready_score")["passed"],
    )
    original_protocol = read(
        reference(ROOT / "openspec/change-proposals/v730-methods-protocol.json")
    )
    for field in ("append_protocol", "append_cost"):
        require(ROOT / methods.PROTOCOL, field, original_protocol[field], methods.protocol()[field])
    original = read(dict(path=str(root / store.UPSTREAM), sha256=store.UPSTREAM_PIN))
    require(root / store.UPSTREAM, "original_run_date", "20261013", original["run_date"])
    tapes = [read(ref) for ref in original["transition_fixture"]["tape_references"]]
    receipt = child(
        "qualified_typed_replay",
        [sys.executable, "-u", str(ROOT / CLI), "--qualified-replay", str(path)],
        raw / "upstream_logs",
        deadline=180,
    )
    require(path, "qualified_cold_replay_passed", True, receipt["passed"])
    return dict(
        authority=authority,
        tapes=tapes,
        upstream=original,
        receipt=receipt,
        refs=[
            *checks["source_artifact_hashes"],
            reference(path),
            reference(root / store.UPSTREAM),
            *original["transition_fixture"]["tape_references"],
            original["append_protocol_reference"],
            reference(Path(terminal["publication"]["sidecar_path"])),
        ],
    )


@contextmanager
def binding() -> Iterator[None]:
    """Change task identity and admission only; shipped measurement thresholds stay intact."""
    with patch.multiple(
        shipped,
        TASK=TASK,
        NAME=NAME,
        CLI=CLI,
        OWNED=OWNED,
        TEST=TEST,
        REQUIRED=REQUIRED,
        admission=admission,
    ):
        yield


def worker(frozen_path: Path, directory: Path, arm: str) -> int:
    """The shipped worker records actual complete clocks and retained synced store bytes."""
    return int(shipped.worker(frozen_path, directory, arm))


def measure(root: Path, raw: Path) -> Json:
    """Fresh children restore all24 arms, retaining censored work when admission fails."""
    with binding():
        work = dict(shipped.measure(root, raw))
    work["run_date"] = "20261014"
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Readiness needs equal durable work; only the primary five pairs can establish benefit."""
    with binding(), patch.object(shipped.legacy.prior, "OWNED", OWNED):
        value = dict(shipped.build(work, receipts, raw, output))
    shares = []
    for row in work["rows"]:
        m = row.get("metrics", {})
        wall = m.get("wall_s")
        costs = {
            key: m.get(key) for key in ("cold_reconstruction_s", "startup_s", "tail_checkpoint_s")
        }
        costs["transactions_and_response_s"] = wall - sum(costs.values()) if wall else None
        shares.append(
            dict(
                unit_id=row["unit_id"],
                wall_s=wall,
                seconds=costs,
                fractions={key: cost / wall if wall else None for key, cost in costs.items()},
                missing_reason=row["missing_reason"],
            )
        )
    value.update(
        experiment_id=8520,
        milestone="2026.10.732",
        run_date="20261014",
        honest_verdict="complete_" + value["verdict_class"] + "_append_service_cost",
        complete_cost_shares=shares,
        historical_clock_eligibility="Exp8494 remains disqualified; no historical clock is a current observation",
        restart_scope="Separate same-process store.read after clearing numerical prefix cache; includes reconstruction and disk replay, excludes interpreter startup and physical power loss",
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8521, 8526)
        ],
    )
    value["field_principles"].update(
        {
            key: "Only fresh complete clocks support local engineering costs; historical authority retains original dates."
            for key in value
            if key not in value["field_principles"]
        }
    )
    value["field_principles"]["complete_cost_shares"] = (
        "Disjoint primitive wall spans; transactions include feature, prediction, feedback, serialization, sync and response. No achieved hardware acceleration is projected."
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild from sealed tapes and disk bytes; repaired summary hashes cannot invent work."""
    try:
        value = json.loads(path.read_bytes())
        if value["experiment_id"] != 8520 or value["reproducibility_checksum"] != checksum(value):
            return False
        work = read(value["work_reference"])
        if len(work["rows"]) != 24:
            return False
        if (
            work["authority"].get("activated")
            and methods.tasks_digest(work["authority"]["tasks"]) != methods.TASKS_PIN
        ):
            return False
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        with TemporaryDirectory(dir="/var/tmp", prefix="exp8520-cold-") as folder:
            if work.get("tapes"):
                inputs = admission(Path(work["root"]), Path(folder))
                if k.base.encode(inputs["tapes"]) != k.base.encode(work["tapes"]):
                    return False
            roster = [
                (cell, repeat, arm)
                for cell in k.CELLS
                for repeat in range(6)
                for arm in (("snapshot", "append") if repeat % 2 == 0 else ("append", "snapshot"))
            ]
            for index, (row, slot) in enumerate(zip(work["rows"], roster, strict=True)):
                progress("cold_primitive", index, 24 - index)
                if (row["cell"], row["repeat"], row["arm"]) != slot:
                    return False
                if row["status"] != "completed":
                    continue
                primitive = read(row["primitive_reference"])
                frozen = read(primitive["frozen_reference"])
                sealed = work["tapes"][k.CELLS.index(row["cell"])]
                if k.base.encode(frozen) != k.base.encode(sealed) or k.base.encode(
                    row["metrics"]
                ) != k.base.encode(k.reduce(frozen, primitive)):
                    return False
                for ref in primitive["store_references"]:
                    if sha256_file(Path(ref["path"])) != ref["sha256"]:
                        return False
                target = Path(folder) / str(index)
                shutil.copytree(Path(primitive["directory"]) / "store", target)
                disk = (k.base.Snapshot if row["arm"] == "snapshot" else shipped.Store)(
                    target, frozen
                )
                if k.base.encode(disk.read()) != k.base.encode(primitive["final"]):
                    return False
        return bool(
            k.base.encode(value)
            == k.base.encode(build(work, value["validation_receipts"], Path(work["raw"]), path))
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration):
        return False
