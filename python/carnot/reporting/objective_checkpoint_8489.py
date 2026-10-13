"""REQ-REPORT-8489: narrow current authority preserves original objective assertions."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np
from scipy.special import expit

from carnot.reporting import objective_checkpoint_8461 as _legacy
from carnot.reporting import v730_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.historical_consumer_roots_8402 import inject
from carnot.reporting.human_label_custody_8389 import gate
from carnot.reporting.primary_publication import read_bound_sidecar, reader_receipt

Json = dict[str, Any]
legacy = _legacy
ROOT = methods.ROOT
NAME, TASK = (
    "experiment_8489_v730_objective_checkpoint_qualification",
    "exp8489-objective-checkpoint-qualification",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_objective_checkpoint_8489.py"
NEW_OWNED = [
    "python/carnot/reporting/objective_checkpoint_8489.py",
    "python/carnot/reporting/objective_checkpoint_runner_8489.py",
    CLI,
]
ORIGINAL_OWNED = list(legacy.OWNED)
ORIGINAL_TESTS = [legacy.TEST, legacy.EXTRA_TEST]
OWNED = ORIGINAL_OWNED + NEW_OWNED
REQUIRED_CHECKS = legacy.REQUIRED_CHECKS | {"original_objective_tests"}
_ORIGINAL_REQUIRED = set(legacy.REQUIRED_CHECKS)
_ORIGINAL_METHODS = legacy.evidence.methods
_ORIGINAL_PROGRESS = legacy.progress
ORIGINAL = ROOT / "results" / (legacy.NAME + ".json")
ORIGINAL_PIN = "sha256:e4adbdbbb533054cb6a5496fcd0761bed7dac0d0fd4f99e4a60741c40f48cf16"
METHODS_PIN = "sha256:552831473fa8df1fd272ae690082cbf23513e559fa9648e9ab0f588f603bb667"
HISTORICAL_LIMIT_REFS = [
    dict(
        path=str(ROOT / "results/experiment_8465_v728_journal_equivalence.json"),
        sha256="sha256:6becdfa9334f6f6ab5b1695eb025ce0cb09ff09f838f0e6c415edb97c580a086",
    ),
    dict(
        path=str(ROOT / "results/experiment_8466_v728_journal_service_cost.json"),
        sha256="sha256:98705463e6d4a85cdddadd122821dae0365700e97459437813b56507cfab718f",
    ),
]
MODEL_SPECS: list[Json] = []
read, reference, checksum, sha256_file = (
    legacy.read,
    legacy.reference,
    legacy.checksum,
    legacy.sha256_file,
)
_MEASURE, _BUILD, _REPLAY, _AUTHENTICATE = (
    legacy.measure,
    legacy.build,
    legacy.replay,
    legacy.evidence.authenticate,
)
_METHODS = SimpleNamespace(**dict(vars(methods), CLI=legacy.evidence.methods.CLI))


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real phase counts because a quiet child could otherwise hide a timeout."""
    print(f"[exp8489] phase={phase} completed={completed} pending={pending}", flush=True)


def log_loss_checks(original: Json) -> list[Json]:
    """Individual bases and centered differences check the unchanged reference updater.

    Effective coefficients include bias in the same L2 sum. Stable logaddexp
    arithmetic avoids replacing extreme probabilities with clipped objectives.
    """
    rows = []
    low, high = np.asarray(original["scaler"]["minimum"]), np.asarray(original["scaler"]["maximum"])
    for support, v in (("lower", 0.0), ("interior", 4 / 9), ("upper", 1.0)):
        x = low + v * (high - low)
        for temperature in (0.5, 1.0, 2.0):
            for z in (-20.0, -4.0, 0.0, 4.0, 20.0):
                head = deepcopy(original)
                head["temperature"] = temperature
                matrix, theta, offset = legacy.derivatives.operands(head, x)
                theta[-1] += z - offset - matrix @ theta
                head.update(parameters=theta.tolist(), decay_scale=1.0)
                for y in (0, 1):
                    p = float(expit(z / temperature))
                    data = matrix * (p - y) / temperature
                    dense = data + 0.002 * theta
                    ids, gradient = legacy.adapter.sparse.sparse_gradient(head, x, y)
                    sparse = 0.002 * theta
                    sparse[ids] += gradient
                    # Extended precision preserves tiny L2 differences beside extreme log loss.
                    delta = np.eye(109, dtype=np.longdouble) * np.longdouble("0.0001")
                    up, down = theta + delta, theta - delta
                    up_eta, down_eta = (
                        (offset + up @ matrix) / temperature,
                        (offset + down @ matrix) / temperature,
                    )
                    upper = np.logaddexp(0, up_eta) - y * up_eta + 0.001 * np.sum(up * up, axis=1)
                    lower = (
                        np.logaddexp(0, down_eta)
                        - y * down_eta
                        + 0.001 * np.sum(down * down, axis=1)
                    )
                    numeric = (upper - lower) / 2e-4
                    error = np.maximum(np.abs(numeric - dense), np.abs(sparse - dense))
                    mask = np.abs(dense) > 1e-6
                    absolute = float(np.max(error))
                    relative = float(np.max(error[mask] / np.abs(dense[mask]), initial=0))
                    rows.append(
                        dict(
                            unit_id=f"log_loss:{support}:T{temperature}:z{z}:y{y}",
                            family="constructed_derivatives",
                            arm="log_loss",
                            support=support,
                            temperature=temperature,
                            logit=z,
                            target=y,
                            absolute_error=absolute,
                            relative_error=relative,
                            saturated=float(np.max(np.abs(data))) <= 1e-7,
                            omit_temperature_rejected=float(
                                np.max(np.abs(data * temperature - data))
                            )
                            > 1e-7,
                            wrong_sign_rejected=float(np.max(np.abs(2 * data))) > 1e-7,
                            passed=absolute <= 1e-7 and relative <= 1e-5,
                        )
                    )
        progress("log_loss_derivatives", len(rows), 90 - len(rows))
    return rows


@contextmanager
def binding() -> Iterator[None]:
    """Only current task identity and authority change; numerical and durable rules stay shipped."""
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
            legacy.evidence,
            NAME=NAME,
            TASK=TASK,
            CLI=CLI,
            methods=_METHODS,
            authenticate=authenticate,
            progress=progress,
        ),
    ):
        yield


def authenticate(root: Path, raw: Path) -> Json:
    """Independently pinned methods gates must pass before cached heads can enter measurement."""
    result = dict(_AUTHENTICATE(root, raw))
    path = root / "results" / (methods.NAME + ".json")
    try:
        value = read(dict(path=str(path), sha256=METHODS_PIN))
        terminal = read(reference(Path(value["terminal_validation_sidecar_path"])))
        sidecar = read_bound_sidecar(path, Path(terminal["publication"]["sidecar_path"]))
        if not sidecar["report"]["passed"]:
            raise ValueError("methods_terminal_validation")
        for field in ("methods_ready_score", "learning_inputs_ready_score"):
            consumer = reader_receipt(methods.TASK, path.parent, field=field)
            if value.get(field) != 1 or not consumer["passed"]:
                raise ValueError(field)
        result["refs"] += [
            reference(path),
            reference(Path(terminal["publication"]["sidecar_path"])),
        ]
    except (OSError, ValueError, KeyError) as error:
        result["failures"].append(
            dict(
                gate(
                    str(path),
                    "methods_and_learning_inputs_ready_score",
                    1,
                    None if isinstance(error, FileNotFoundError) else str(error),
                    upstream=methods.TASK,
                ),
                artifact_hash=METHODS_PIN,
            )
        )
    return result


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Measure the original derivative and exit grid only after current prerequisites authenticate."""
    with binding():
        work = dict(_MEASURE(root, raw, private))
    primitive = read(work["primitive_reference"])
    log_rows = log_loss_checks(primitive["head"]) if primitive["derivative_rows"] else []
    atomic_json(raw / "log_loss_derivatives.json", log_rows)
    work["log_loss_reference"] = reference(raw / "log_loss_derivatives.json")
    try:
        control = changed_reference_control(raw / "changed_reference")
        work["changed_reference_control"] = reference(
            raw / "changed_reference/changed_reference_control.json"
        )
        work["changed_reference_passed"] = all(
            control[k]
            for k in (
                "valid_original_passed",
                "changed_reference_rejected",
                "outer_hashes_consistent",
            )
        )
    except (OSError, ValueError, KeyError) as error:
        work.update(changed_reference_control=None, changed_reference_passed=None)
        work["failures"].append(
            gate(
                str(ORIGINAL),
                "original_reference_fixture_input",
                True,
                None if isinstance(error, FileNotFoundError) else str(error),
                upstream="exp8461",
            )
        )
    ledger = private.parent / "historical_collection.json"
    work["historical_collection_reference"] = None
    if ledger.is_file():
        shutil.copyfile(ledger, raw / "historical_collection.json")
        work["historical_collection_reference"] = reference(raw / "historical_collection.json")
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """All original required checks remain necessary; the new rejection control adds a gate."""
    with binding():
        value = dict(_BUILD(work, receipts, raw, output))
    log_rows = read(work["log_loss_reference"])
    log_ok = len(log_rows) == 90 and all(
        r["passed"]
        and (r["saturated"] or r["wrong_sign_rejected"])
        and (r["temperature"] == 1 or abs(r["logit"]) > 4 or r["omit_temperature_rejected"])
        for r in log_rows
    )
    if not log_ok and not work["failures"]:
        value.update(
            verdict_class="disqualified",
            objective_kernel_ready_score=0,
            required_checks_passed=False,
            flagged_adversarial=True,
        )
        value["gate_check_summary"].append(
            gate(str(raw), "reference_log_loss_derivatives", True, False, upstream=TASK)
        )
    additional = log_rows or [
        dict(
            unit_id=f"log_loss:{support}:T{t}:z{z}:y{y}",
            family="constructed_derivatives",
            arm="log_loss",
            passed=None,
        )
        for support in ("lower", "interior", "upper")
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
        for r in additional
    ]
    value.update(
        intended_count=216,
        completed_count=sum(r["status"] == "completed" for r in value["rows"]),
        failed_count=sum(r["status"] == "failed" for r in value["rows"]),
        censored_count=sum(r["censored"] for r in value["rows"]),
        log_loss_reference=work["log_loss_reference"],
    )
    value["derivative_rows"] += log_rows
    value["sample_size_budget"]["constructed_derivative_cells"] = 180
    value["acceptance_gates"]["reference_log_loss_derivatives"] = log_ok
    value["gate_check_summary"] = [
        g for g in value["gate_check_summary"] if g["check"] != "required_owned_check"
    ] + [
        dict(
            gate(
                r["stdout_path"], "validation_checks." + r["name"], True, r["passed"], upstream=TASK
            ),
            artifact_hash=r["stdout_sha256"],
            exit_code=r["exit_code"],
            timed_out=r["timed_out"],
        )
        for r in receipts
        if not r["passed"]
    ]
    if work["changed_reference_passed"] is False:
        value.update(
            verdict_class="disqualified",
            objective_kernel_ready_score=0,
            required_checks_passed=False,
            flagged_adversarial=True,
        )
        value["gate_check_summary"].append(
            gate(
                str(raw),
                "original_changed_reference_control",
                True,
                False,
                upstream=TASK,
            )
        )
    value.update(
        experiment_id=8489,
        task_id=TASK,
        milestone="2026.10.730",
        run_date="20261013",
        honest_verdict=f"complete_{value['verdict_class']}_v730_objective_checkpoint_qualification",
        changed_reference_control_reference=work["changed_reference_control"],
        historical_collection_reference=work["historical_collection_reference"],
        superseded_qualification=dict(
            path=str(ORIGINAL),
            sha256=ORIGINAL_PIN,
            terminal_bytes_preserved=True,
            original_statements_required=564,
        ),
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation") for n in (8490, 8491)
        ],
    )
    value["raw_shard_hashes"] += [
        r
        for r in (
            work["changed_reference_control"],
            work["historical_collection_reference"],
            work["log_loss_reference"],
        )
        if r is not None
    ]
    value["acceptance_gates"]["changed_reference_rejection"] = work["changed_reference_passed"]
    value["historical_limits"].update(
        V728_journal_equivalence="qualified",
        V728_complete_service_speedup=1.0447833367,
        V728_complete_service_CI95=[0.9942346768, 1.1095895448],
        V729="Objective repair not attempted; methods readiness zero was not learning evidence.",
    )
    value["cited_upstream_artifacts"] += [
        dict(HISTORICAL_LIMIT_REFS[0], imported_fields=["journal_ready_score", "honest_verdict"]),
        dict(
            HISTORICAL_LIMIT_REFS[1],
            imported_fields=["pooled_speedup", "speedup_ci95", "honest_verdict"],
        ),
    ]
    for ref in value["cited_upstream_artifacts"]:
        if ref["sha256"] == METHODS_PIN:
            ref["imported_fields"] = [
                "methods_ready_score",
                "learning_inputs_ready_score",
                "terminal_validation_sidecar_path",
            ]
    value["field_principles"].update(
        {
            k: "Invocation-bound update qualification; no measured learning benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        log_loss_reference="Dense individual bases, sparse gradients and centered differences check (p-y)/T with full bias L2; no learning benefit is measured.",
        historical_collection_reference="Exact unchanged node identities and assertion hashes prove disjoint complete shards; unfinished children cannot qualify.",
        changed_reference_control_reference="Successful sealed original replay precedes changed private bytes; outer hashes do not override stored reference hashes.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Original reconstruction and immutable control references reject repaired outer hashes."""
    try:
        value = read(reference(path))
        work = read(value["primitive_reference"])
        primitive = read(work["primitive_reference"])
        expected_log = log_loss_checks(primitive["head"]) if primitive["derivative_rows"] else []
        if read(value["log_loss_reference"]) != expected_log:
            return False
        if value["changed_reference_control_reference"] is not None:
            control = read(value["changed_reference_control_reference"])
            read(control["candidate_reference"])
            changed = control["changed_file_reference"]
            if sha256_file(Path(changed["path"])) != changed["sha256"]:
                return False
            if control["original_terminal_reference"] != dict(
                path=str(ORIGINAL), sha256=ORIGINAL_PIN
            ):
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


@contextmanager
def original_binding(scratch: Path) -> Iterator[None]:
    """Original complete authority remains separate from the current activated V730 queue."""
    old = read(dict(path=str(ORIGINAL), sha256=ORIGINAL_PIN))
    work = read(old["primitive_reference"])
    refs = []
    for ref in work["authority"]["authority_snapshots"].values():
        if ref.get("exists"):
            source = Path(ref["snapshot_path"])
            if sha256_file(source) != ref["sha256"]:
                raise ValueError("original_authority_pin")
            refs.append(dict(ref, path=str(source), snapshot_path=str(source)))
    with (
        patch.multiple(
            legacy,
            NAME=ORIGINAL.stem,
            TASK="exp8461-objective-checkpoint-qualification",
            CLI=ORIGINAL_OWNED[-1],
            TEST=ORIGINAL_TESTS[0],
            OWNED=ORIGINAL_OWNED,
            REQUIRED_CHECKS=_ORIGINAL_REQUIRED,
            build=_BUILD,
            measure=_MEASURE,
            replay=_REPLAY,
            progress=_ORIGINAL_PROGRESS,
        ),
        patch.multiple(
            legacy.evidence,
            NAME=ORIGINAL.stem,
            TASK="exp8461-objective-checkpoint-qualification",
            CLI=ORIGINAL_OWNED[-1],
            methods=_ORIGINAL_METHODS,
            authenticate=_AUTHENTICATE,
            progress=_ORIGINAL_PROGRESS,
        ),
        inject(dict(refs=refs), scratch),
    ):
        yield


def changed_reference_control(raw: Path) -> Json:
    """Changed private bytes reach the original rejection before independent numerical replay.

    Both outer JSON hashes are regenerated. The stored operand digest is retained
    because an outer envelope cannot authorize different underlying evidence.
    """
    progress("changed_reference_control", 0, 2)
    raw.mkdir(parents=True, exist_ok=True)
    old = read(dict(path=str(ORIGINAL), sha256=ORIGINAL_PIN))
    work = deepcopy(read(old["primitive_reference"]))
    source = raw / "independent_source.txt"
    source.write_text("independent original reference bytes\n")
    stored = reference(source)
    work["source_refs"].append(stored)
    shutil.copyfile(ROOT / ORIGINAL_OWNED[0], raw / "update_adapter.py")
    candidate = raw / "original_candidate.json"
    atomic_json(raw / "measurement.json", work)
    with original_binding(raw / "authority"):
        baseline = legacy.build(work, old["validation_receipts"], raw, candidate)
    atomic_json(candidate, baseline)
    with original_binding(raw / "authority"):
        valid = legacy.replay(candidate)
    source.write_text("changed independent reference bytes\n")
    atomic_json(raw / "measurement.json", work)
    with original_binding(raw / "authority"):
        value = legacy.build(work, old["validation_receipts"], raw, candidate)
    atomic_json(candidate, value)
    with original_binding(raw / "authority"):
        rejected = not legacy.replay(candidate)
    result = dict(
        valid_original_passed=valid,
        changed_reference_rejected=rejected,
        outer_hashes_consistent=value["primitive_reference"] == reference(raw / "measurement.json")
        and value["reproducibility_checksum"] == checksum(value),
        stored_reference_sha256=stored["sha256"],
        changed_file_sha256=sha256_file(source),
        changed_file_reference=reference(source),
        candidate_reference=reference(candidate),
        original_terminal_reference=dict(path=str(ORIGINAL), sha256=ORIGINAL_PIN),
        original_rejection_statement="python/carnot/reporting/objective_checkpoint_8461.py:388",
    )
    atomic_json(raw / "changed_reference_control.json", result)
    progress("changed_reference_control_complete", 2, 0)
    return result
