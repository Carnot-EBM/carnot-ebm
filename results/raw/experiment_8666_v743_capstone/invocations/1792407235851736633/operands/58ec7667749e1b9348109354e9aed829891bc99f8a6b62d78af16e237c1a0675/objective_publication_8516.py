"""REQ-REPORT-8516: new qualification never authorizes altered historical bytes."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np

from carnot.reporting import brier_online_learning_8490 as downstream
from carnot.reporting import objective_checkpoint_8461 as legacy
from carnot.reporting import objective_checkpoint_8489 as history
from carnot.reporting import objective_checkpoint_evidence_8461 as evidence
from carnot.reporting import v732_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.verify import objective_checkpoint_8461 as adapter
from carnot.verify import objective_derivatives_8461 as derivatives

Json = dict[str, Any]
__all__ = ["history", "methods"]
ROOT = methods.ROOT
NAME = "experiment_8516_v732_objective_publication_qualification"
TASK = "exp8516-objective-publication-qualification"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_objective_publication_8516.py"
OWNED = [
    "python/carnot/reporting/objective_publication_8516.py",
    "python/carnot/reporting/objective_publication_runner_8516.py",
    CLI,
]
REQUIRED_CHECKS = {
    "owned_tests",
    "historical_union",
    "coverage_combine",
    "coverage_report",
    "coverage_json",
    "ruff_check",
    "ruff_format",
    "strict_mypy",
    "spec_coverage",
}
MODEL_SPECS: list[Json] = []
OLD_PRIMARY = ROOT / "results" / (history.NAME + ".json")
OLD_PIN = downstream.UPSTREAM_PIN
OLD_SEAL = "sha256:e9145dae2143a2563f7c088db80dbcbf1178fd37618789b2f346e0d0dfdb6359"
METHODS_PIN = "sha256:d8aad6fcd8662fe8abc98acaa95add49df44b3b5359700ab09bbaafee3114eec"
FIXTURE = ROOT / "tests/fixtures/objective_primitives_8516.json"
FIXTURE_PIN = "sha256:7370f53889804d389158618cca32ab3431e53f6e56d36e214225d7fe4bf48612"
read, reference, checksum = history.read, history.reference, history.checksum
_MEASURE, _BUILD, _REPLAY = legacy.measure, legacy.build, legacy.replay
_METHODS = SimpleNamespace(**dict(vars(methods), CLI=evidence.methods.CLI))


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real work counts so a supervisor can detect a stalled child."""
    print(f"[exp8516] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Change task identity while the shipped equations and state rules stay intact."""
    with (
        patch.multiple(
            legacy,
            NAME=NAME,
            TASK=TASK,
            CLI=CLI,
            TEST=TEST,
            OWNED=OWNED,
            REQUIRED_CHECKS=REQUIRED_CHECKS,
            build=build,
            measure=measure,
            replay=replay,
            progress=progress,
        ),
        patch.multiple(
            evidence,
            NAME=NAME,
            TASK=TASK,
            CLI=CLI,
            methods=_METHODS,
            authenticate=authenticate,
            progress=progress,
        ),
    ):
        yield


def consumer(primary: Path, sidecar: Path, *, replay: bool = True) -> Json:
    """Use the actual learning consumer checks on bound bytes, then replay their operands."""
    failures: list[Json] = []
    receipts: list[Json] = []
    try:
        digest = sha256_file(primary)
        value = read(reference(primary))
        failures.extend(downstream.qualification_failures(primary, digest, value))
        bound = read_bound_sidecar(primary, sidecar)
        if not bound["report"]["passed"]:
            raise ValueError("bound_publication_validation")
        if replay:
            receipt = child(
                "qualification_cold_replay",
                [
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(ROOT / CLI),
                    "--cold-replay",
                    str(primary),
                ],
                sidecar.parent / "consumer",
                deadline=180,
            )
            receipts.append(receipt)
            if not receipt["passed"]:
                raise ValueError("consumer_cold_replay")
        return dict(
            passed=not failures,
            primary_path=str(primary),
            primary_sha256=digest,
            failures=failures,
            receipts=receipts,
            consumer="brier_online_learning_8490.qualification_failures",
        )
    except (OSError, ValueError, KeyError) as error:
        return dict(
            passed=False,
            primary_path=str(primary),
            failures=failures,
            error=str(error),
            receipts=receipts,
        )


def history_control(raw: Path) -> Json:
    """Copy both original records privately and retain the consumer that rejected them."""
    old = read(dict(path=str(OLD_PRIMARY), sha256=OLD_PIN))
    terminal = read(reference(Path(old["terminal_validation_sidecar_path"])))
    side = Path(terminal["publication"]["sidecar_path"])
    copied = raw / "results" / OLD_PRIMARY.name
    copied.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OLD_PRIMARY, copied)
    copied_side = copied.parent / "raw" / copied.stem / "validators" / side.name
    copied_side.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(side, copied_side)
    side_value = read(reference(side))
    checked = consumer(copied, copied_side, replay=False)
    result = dict(
        flagged_primary_reference=dict(
            reference(OLD_PRIMARY), flagged_adversarial=old["flagged_adversarial"]
        ),
        sidecar_reference=reference(side),
        sidecar_bound_sha256=side_value["primary_sha256"],
        failed_consumer_reference=reference(ROOT / "results" / (downstream.NAME + ".json")),
        stale_sidecar_rejected=checked.get("error") == "stale_primary_hash",
        copied_flagged_envelope_rejected=bool(checked["failures"]),
        copied_primary_reference=reference(copied),
        copied_sidecar_reference=reference(copied_side),
        original_run_date=old["run_date"],
        consumer_receipt=checked,
    )
    if not (
        result["stale_sidecar_rejected"]
        and result["copied_flagged_envelope_rejected"]
        and result["sidecar_bound_sha256"] == OLD_SEAL
    ):
        raise ValueError("original_publication_mismatch")
    atomic_json(raw / "original_publication_mismatch.json", result)
    return result


def primitive_grid(head: Json) -> list[Json]:
    """Pinned high-precision scalar derivatives check the bias in both shipped gradients."""
    fixtures = read(dict(path=str(FIXTURE), sha256=FIXTURE_PIN))
    rows: list[Json] = list(derivatives.checks(head)) + list(history.log_loss_checks(head))
    for row in rows:
        fixture = next(
            f for f in fixtures if all(f[k] == row[k] for k in ("temperature", "logit", "target"))
        )
        h = deepcopy(head)
        h["temperature"] = row["temperature"]
        low, high = np.asarray(h["scaler"]["minimum"]), np.asarray(h["scaler"]["maximum"])
        x = low + {"lower": 0, "interior": 4 / 9, "upper": 1}[row["support"]] * (high - low)
        matrix, theta, offset = derivatives.operands(h, x)
        theta[-1] += row["logit"] - offset - matrix @ theta
        h.update(parameters=theta.tolist(), decay_scale=1.0)
        if row["arm"] == "brier":
            _, receipt = adapter.update(h, x, row["target"])
            ids, gradient = receipt["gradient_ids"], receipt["data_gradient"]
        else:
            ids, gradient = adapter.sparse.sparse_gradient(h, x, row["target"])
        actual = float(gradient[list(ids).index(108)])
        expected = fixture[row["arm"]]
        error = abs(actual - expected)
        relative = error / abs(expected) if abs(expected) > 1e-6 else 0
        row["pinned_scalar_error"] = error
        row["pinned_scalar_relative_error"] = relative
        row["passed"] = row["passed"] and error <= 1e-7 and relative <= 1e-5
    return rows


def authenticate(root: Path, raw: Path) -> Json:
    """Current task authority and independently sealed original producers must both qualify."""
    result = dict(history._AUTHENTICATE(root, raw))
    for path, pin in [
        (FIXTURE, FIXTURE_PIN),
        (OLD_PRIMARY, OLD_PIN),
        (history.ORIGINAL, history.ORIGINAL_PIN),
        (
            ROOT / "results" / (downstream.NAME + ".json"),
            "sha256:723b8e819301e0c28fabc5768673e503ad38e9dc13aa0c60e36e006e6446bfb3",
        ),
    ]:
        observed = None
        try:
            observed = sha256_file(path)
            if observed != pin:
                raise ValueError("original_input_sha256")
            result["refs"].append(reference(path))
        except (OSError, ValueError):
            result["failures"].append(downstream.gate(path, pin, "sha256", pin, observed, TASK))
    path = root / "results" / (methods.NAME + ".json")
    try:
        value = read(dict(path=str(path), sha256=METHODS_PIN))
        terminal = read(reference(Path(value["terminal_validation_sidecar_path"])))
        bound = read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))
        if bound["report"]["passed"] is not True:
            raise ValueError("methods_bound_validation")
        for field in ("methods_ready_score", "learning_inputs_ready_score"):
            if (
                value.get(field) != 1
                or not reader_receipt(methods.TASK, path.parent, field=field)["passed"]
            ):
                raise ValueError(field)
        result["refs"] += [
            reference(path),
            reference(Path(terminal["publication"]["sidecar_path"])),
        ]
    except (OSError, ValueError, KeyError) as error:
        result["failures"].append(
            downstream.gate(
                path,
                METHODS_PIN,
                "methods_and_learning_inputs_ready_score",
                1,
                None if isinstance(error, FileNotFoundError) else str(error),
                methods.TASK,
            )
        )
    return result


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Reuse real hard-exit workers after resource checks; no target stream enters this task."""
    with binding():
        work = dict(_MEASURE(root, raw, private))
    primitive = read(work["primitive_reference"])
    rows = primitive_grid(primitive["head"]) if primitive["derivative_rows"] else []
    atomic_json(raw / "derivative_grid.json", rows)
    work["derivative_grid_reference"] = reference(raw / "derivative_grid.json")
    control = history.changed_reference_control(raw / "changed_reference")
    work["changed_reference_reference"] = reference(
        raw / "changed_reference/changed_reference_control.json"
    )
    work["changed_reference_passed"] = all(
        control[k]
        for k in ("valid_original_passed", "changed_reference_rejected", "outer_hashes_consistent")
    )
    old = history_control(raw / "history")
    work["source_refs"] += [
        old["flagged_primary_reference"],
        old["sidecar_reference"],
        old["failed_consumer_reference"],
    ]
    work["history_reference"] = reference(raw / "history/original_publication_mismatch.json")
    work["fixture_pin_reference"] = dict(path=str(FIXTURE), sha256=FIXTURE_PIN)
    work["historical_collection_reference"] = None
    ledger = private.parent / "historical_collection.json"
    if ledger.exists():
        shutil.copyfile(ledger, raw / "historical_collection.json")
        work["historical_collection_reference"] = reference(raw / "historical_collection.json")
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """All original mechanics remain necessary; successful bound consumer bytes qualify publication."""
    with binding():
        value = dict(_BUILD(work, receipts, raw, output))
    grid = read(work["derivative_grid_reference"])
    old = read(work["history_reference"])
    for i, row in enumerate(grid[:90]):
        value["rows"][i].update(
            row,
            status="completed" if row["passed"] else "failed",
            absolute_metric=int(row["passed"]),
            raw_numerator=int(row["passed"]),
        )
    good = len(grid) == 180 and all(r["passed"] for r in grid) and work["changed_reference_passed"]
    if not good and not work["failures"]:
        value.update(
            verdict_class="disqualified",
            objective_kernel_ready_score=0,
            required_checks_passed=False,
            flagged_adversarial=True,
        )
        value["gate_check_summary"].append(
            downstream.gate(raw, None, "new_primitive_and_rejection_checks", True, good, TASK)
        )
    log_rows = [r for r in grid if r["arm"] == "log_loss"] or [
        dict(
            unit_id=f"log_loss:{s}:T{t}:z{z}:y{y}",
            family="constructed_derivatives",
            arm="log_loss",
            passed=None,
        )
        for s in ("lower", "interior", "upper")
        for t in (0.5, 1.0, 2.0)
        for z in (-20.0, -4.0, 0.0, 4.0, 20.0)
        for y in (0, 1)
    ]
    value["rows"] += [
        dict(
            r,
            status="censored" if r["passed"] is None else "completed" if r["passed"] else "failed",
            absolute_metric=None if r["passed"] is None else int(r["passed"]),
            raw_numerator=None if r["passed"] is None else int(r["passed"]),
            raw_denominator=1,
            censored=r["passed"] is None,
            excluded=False,
            missing_reason="required_cpu_operand_unavailable" if r["passed"] is None else None,
        )
        for r in log_rows
    ]
    value.update(
        experiment_id=8516,
        milestone="2026.10.732",
        run_date="20261014",
        honest_verdict=f"complete_{value['verdict_class']}_v732_objective_publication_qualification",
        intended_count=216,
        completed_count=sum(r["status"] == "completed" for r in value["rows"]),
        failed_count=sum(r["status"] == "failed" for r in value["rows"]),
        censored_count=sum(r["censored"] for r in value["rows"]),
        derivative_rows=grid,
        learning_qualification_ready_score=value["objective_kernel_ready_score"],
        original_flagged_reference=old["flagged_primary_reference"],
        original_sidecar_reference=dict(old["sidecar_reference"], primary_sha256=OLD_SEAL),
        publication_byte_history=[
            dict(stage="original_flagged", **old["flagged_primary_reference"]),
            dict(stage="original_sidecar", primary_sha256=OLD_SEAL, **old["sidecar_reference"]),
            dict(
                stage="new_atomic_publication",
                receipt_path=value["terminal_validation_sidecar_path"],
            ),
        ],
        qualification_consumer_receipt=dict(
            path=str(output.parent / "raw" / output.stem / "qualification_consumer.json"),
            required_before_publication=True,
            successful_exact_bytes_required=True,
        ),
        historical_collection_reference=work["historical_collection_reference"],
        collection_archive_reference=work.get("collection_archive_reference"),
        derivative_grid_reference=work["derivative_grid_reference"],
        history_reference=work["history_reference"],
        changed_reference_reference=work["changed_reference_reference"],
        primitive_fixture_reference=work["fixture_pin_reference"],
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation") for n in (8517, 8518)
        ],
    )
    value["sample_size_budget"]["constructed_derivative_cells"] = 180
    value["gate_check_summary"] = [
        g for g in value["gate_check_summary"] if g["check"] != "required_owned_check"
    ] + [
        dict(
            downstream.gate(
                Path(r["stdout_path"]),
                r["stdout_sha256"],
                "validation_checks." + r["name"],
                True,
                r["passed"],
                TASK,
            ),
            exit_code=r["exit_code"],
            timed_out=r["timed_out"],
        )
        for r in receipts
        if not r["passed"]
    ]
    value["gate_check_summary"] = [
        dict(
            g,
            upstream=g.get("upstream", g.get("upstream_id", TASK)),
            path=g.get("path", g.get("artifact_path", str(raw))),
            sha256=g.get("sha256", g.get("artifact_hash")),
            field=g.get("field", g.get("artifact_field", g["check"])),
            operator=g.get("operator", g.get("op", "==")),
        )
        for g in value["gate_check_summary"]
    ]
    value["acceptance_gates"].update(
        pinned_derivatives=good,
        original_reference_rejection=work["changed_reference_passed"],
        bound_consumer_required=True,
        historical_quarantine_preserved=True,
    )
    value["raw_shard_hashes"] += [
        work[k]
        for k in (
            "derivative_grid_reference",
            "history_reference",
            "changed_reference_reference",
            "fixture_pin_reference",
        )
    ]
    if work.get("collection_archive_reference"):
        value["raw_shard_hashes"].append(work["collection_archive_reference"])
    value["field_principles"].update(
        {
            k: "New invocation evidence; successful exact bound consumer bytes are necessary; no scientific benefit measured."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Reconstruct numerical evidence so rehashing an envelope cannot repair altered operands."""
    try:
        value = read(reference(path))
        work = read(value["primitive_reference"])
        primitive = read(work["primitive_reference"])
        grid = primitive_grid(primitive["head"]) if primitive["derivative_rows"] else []
        if read(value["derivative_grid_reference"]) != grid:
            return False
        old = read(value["history_reference"])
        if (
            old["flagged_primary_reference"]
            != dict(reference(OLD_PRIMARY), flagged_adversarial=True)
            or old["sidecar_bound_sha256"] != OLD_SEAL
        ):
            return False
        if (
            old["flagged_primary_reference"]["sha256"] != OLD_PIN
            or consumer(
                Path(old["copied_primary_reference"]["path"]),
                Path(old["copied_sidecar_reference"]["path"]),
                replay=False,
            ).get("error")
            != "stale_primary_hash"
        ):
            return False
        control = read(value["changed_reference_reference"])
        read(control["candidate_reference"])
        changed_file = control["changed_file_reference"]
        if sha256_file(Path(changed_file["path"])) != changed_file["sha256"]:
            return False
        if control["original_terminal_reference"] != dict(
            path=str(history.ORIGINAL), sha256=history.ORIGINAL_PIN
        ):
            return False
        read(value["primitive_fixture_reference"])
        if value["primitive_fixture_reference"]["sha256"] != FIXTURE_PIN:
            return False
        if value["historical_collection_reference"] is not None:
            ledger = read(value["historical_collection_reference"])
            union = [n for group in ledger["shards"] for n in group]
            if union != ledger["node_ids"] or len(union) != len(set(union)):
                return False
        with binding():
            return bool(_REPLAY(path))
    except (OSError, ValueError, KeyError, TypeError):
        return False
