"""REQ-REPORT-8487: fit a scalar head without changing authenticated historical heads.

Only thirteen small coefficients learn from exposed fit labels. Historical
language model outputs remain cached operands, not current model invocations.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch
import time

import numpy as np
from scipy.interpolate import BSpline
from scipy.special import expit

from carnot.reporting import v730_contract_methods as methods
from carnot.reporting import v728_source_control_panel as controls
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar, reader_receipt
from carnot.verify import v726_criterion_energy_fit as old
from carnot.verify import sparse_energy_7996 as sparse

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK = "experiment_8487_v730_source_control_fit", "exp8487-source-control-fit"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v730_source_control_fit_8487.py"
OWNED = [
    "python/carnot/verify/v730_source_control_fit.py",
    "python/carnot/reporting/v730_source_fit_runner.py",
    CLI,
]
UPSTREAM = "results/experiment_8486_v730_source_control_qualification.json"
UPSTREAM_PIN = "sha256:f513a56bc388076cb17338cd92f7a256616e2c5fbf2fb8c71dc649872fe4d007"
OLD_PIN = "sha256:38db38433d5e2803a7d4b008c88fba5a1a91138abceb42e52f3dadf5b2f10b7f"
OLD_NAME = old.NAME
ARMS, SEEDS = controls.ARMS, sparse.SEEDS
INHERITED = dict(
    raw_p2="raw_p2",
    scalar_logistic="scalar_calibrated",
    full_spline9="spline9",
    source_permuted_spline9="source_permuted_spline9",
)
CONFIG = dict(old.CONFIG, arms=list(ARMS), scalar_parameters=len(sparse.KNOTS) - 3, scalar_inputs=1)
MODEL_SPECS: list[Json] = []
read, reference, checksum = methods.read, methods.reference, methods.checksum
_METHODS = SimpleNamespace(**dict(vars(methods), legacy=old.methods.legacy))


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts so the supervisor can detect a stalled process."""
    print(f"[exp8487] phase={phase} completed={completed} pending={pending}", flush=True)


def semantic(value: Any) -> Any:
    """Exclude measured clock durations when comparing deterministic numeric states."""
    if isinstance(value, dict):
        return {
            k: semantic(v)
            for k, v in value.items()
            if k not in {"duration_s", "current_optimization_duration_s"}
        }
    if isinstance(value, list):
        return [semantic(v) for v in value]
    return value


def scalar_matrix(q: sparse.Array, scaler: Json) -> sparse.Array:
    """The shipped one-input basis uses only q and original fit bounds."""
    return np.asarray(controls.scalar_basis(q.tolist(), scaler), dtype=float)


def scalar_logit(head: Json, q: sparse.Array) -> sparse.Array:
    """Keep the inherited logit offset so zero coefficients retain raw q."""
    clipped = np.clip(q, 1e-4, 1 - 1e-4)
    return np.asarray(
        np.log(clipped / (1 - clipped))
        + scalar_matrix(q, head["scaler"]) @ np.asarray(head["parameters"])
    )


def objective(head: Json, q: sparse.Array, y: sparse.Array) -> float:
    """Temperature and every coefficient's L2 penalty belong to the same objective."""
    z = scalar_logit(head, q) / head["temperature"]
    return float(
        np.mean(np.logaddexp(0, z) - y * z) + 0.001 * np.sum(np.asarray(head["parameters"]) ** 2)
    )


def numeric_checks(head: Json, q: sparse.Array) -> Json:
    """Individual dense splines and finite differences independently check the optimizer."""
    low, high = head["scaler"]["minimum"][0], head["scaler"]["maximum"][0]
    scaled = np.clip((q - low) / (high - low if high > low else 1), 0, 1)
    count = len(sparse.KNOTS) - 4
    dense = np.column_stack(
        [BSpline(sparse.KNOTS, np.eye(count)[i], 3)(scaled) for i in range(count)]
        + [np.ones(len(q))]
    )
    basis_error = float(np.max(np.abs(dense - scalar_matrix(q, head["scaler"]))))
    theta = np.asarray(head["parameters"])
    errors, parity = [], []
    for temperature in (0.5, 1.0, 2.0):
        h = dict(head, temperature=temperature)
        for label in (0, 1):
            y: sparse.Array = np.full(len(q), label, dtype=float)
            z = scalar_logit(h, q) / temperature
            gradient = dense.T @ (expit(z) - y) / (len(q) * temperature) + 0.002 * theta
            for i in range(len(theta)):
                delta = np.eye(len(theta))[i] * 1e-6
                up, down = (
                    dict(h, parameters=(theta + delta).tolist()),
                    dict(h, parameters=(theta - delta).tolist()),
                )
                errors.append(
                    abs((objective(up, q, y) - objective(down, q, y)) / 2e-6 - gradient[i])
                )
            weights = np.exp(np.column_stack((np.zeros(len(z)), z)) - np.maximum(0, z)[:, None])
            parity.append(float(np.max(np.abs(weights[:, 1] / weights.sum(axis=1) - expit(z)))))
    error = float(max(errors))
    return dict(
        passed=basis_error < 1e-12 and error < 1e-7 and max(parity) < 1e-12,
        dense_basis_error=basis_error,
        finite_difference_error=error,
        normalization_error=max(parity),
        temperatures=[0.5, 1, 2],
        labels=[0, 1],
    )


def fit(
    seed: int, scaler: Json, q: sparse.Array, y: sparse.Array, tq: sparse.Array, ty: sparse.Array
) -> Json:
    """Use the registered zero start and fixed full-batch budget without test labels."""
    start = time.monotonic()
    matrix = scalar_matrix(q, scaler)
    theta = np.zeros(matrix.shape[1])
    head = dict(
        arm="scalar_cubic_spline",
        seed=seed,
        scaler=scaler,
        parameters=theta.tolist(),
        parameter_count=len(theta),
        temperature=1.0,
        optimizer_steps=200,
    )
    initial = objective(head, q, y)
    offset = scalar_logit(head, q)
    for step in range(200):
        theta -= 0.01 * (matrix.T @ (expit(offset + matrix @ theta) - y) / len(q) + 0.002 * theta)
        if step % 50 == 0:
            progress(f"scalar_seed{seed}_steps{step + 1}/200", step + 1, 199 - step)
    head["parameters"] = theta.tolist()
    losses = {
        str(t): old.scalar.bce(np.asarray(expit(scalar_logit(head, tq) / t)), ty)
        for t in (0.5, 1.0, 2.0)
    }
    head.update(
        initial_loss=initial,
        final_loss=objective(head, q, y),
        tune_temperature_losses=losses,
        temperature=old.choose_temperature(losses),
        duration_s=time.monotonic() - start,
    )
    return head


def operands(root: Path) -> tuple[Json, Json]:
    """Independent seals forbid substituting locally rehashed heads or unqualified controls."""
    upstream = read(dict(path=str(root / UPSTREAM), sha256=UPSTREAM_PIN))
    consumer = reader_receipt(
        "exp8486-source-control-qualification",
        root / "results",
        field="source_controls_ready_score",
    )
    terminal = json.loads(Path(upstream["terminal_validation_sidecar_path"]).read_bytes())
    bound = read_bound_sidecar(root / UPSTREAM, Path(terminal["publication"]["sidecar_path"]))
    if (
        not upstream["required_checks_passed"]
        or not consumer["passed"]
        or not bound["report"]["passed"]
    ):
        raise ValueError("source_controls_not_qualified")
    protocol = methods.protocol()
    for ref in protocol["historical_authorities"]["V726"]:
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("original_authority_pin")
    original_ref = dict(path=str(root / "results" / (OLD_NAME + ".json")), sha256=OLD_PIN)
    original = read(original_ref)
    work = read(original["primitive_reference"])
    for ref in work["code_refs"]:
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("original_method_hash")
    heads = read(dict(path=original["frozen_heads_path"], sha256=original["frozen_heads_sha256"]))
    wrapper = read(upstream["trainer_view"])
    view = read(wrapper["reference"])
    panel = read(upstream["feature_manifest_reference"])
    if (
        heads["scaler"] != panel["operators"]["scaler"]
        or panel["operators"]["knots"] != sparse.KNOTS
    ):
        raise ValueError("original_scaler_knots")
    if wrapper["reference"] != original["source_artifact_hashes"][1]:
        raise ValueError("original_fit_tune_pin")
    upstream = dict(
        upstream,
        trainer_view=wrapper["reference"],
        stratum_roster=[],
        additional_refs=[
            original_ref,
            reference(Path(original["frozen_heads_path"])),
            upstream["feature_manifest_reference"],
            *protocol["historical_authorities"]["V726"],
        ],
    )
    return upstream, dict(view, original_heads=heads, panel=panel)


def train(view: Json, strata: list[Json]) -> Json:
    """Reuse four authenticated arms verbatim and fit only the missing scalar capacity control."""
    if any(r["role"] not in {"fit", "tune"} for r in view["rows"]):
        raise ValueError("test_label_boundary")
    f, t = (
        [r for r in view["rows"] if r["role"] == role and r["x"] is not None and r["y"] is not None]
        for role in ("fit", "tune")
    )
    q, tq = (np.asarray([r["x"][0] for r in rows], dtype=float) for rows in (f, t))
    y, ty = (np.asarray([r["y"] for r in rows], dtype=float) for rows in (f, t))
    original = view["original_heads"]
    result: Json = deepcopy(original)
    result["heads"] = {arm: deepcopy(original["heads"][name]) for arm, name in INHERITED.items()}
    result["heads"]["scalar_cubic_spline"] = []
    checks = {}
    for index, seed in enumerate(SEEDS):
        progress("scalar_fit_begin", index, len(SEEDS) - index)
        head = fit(seed, original["scaler"], q, y, tq, ty)
        check = numeric_checks(head, np.asarray([0.0, 0.01, 0.14, 0.5, 0.95, 1.0]))
        if not check["passed"]:
            raise ValueError("scalar_numeric_check")
        checks[str(seed)] = check
        result["heads"]["scalar_cubic_spline"].append(head)
        progress("scalar_fit_end", index + 1, len(SEEDS) - index - 1)
    result.update(
        fit_questions=len({r["question_hash"] for r in f}),
        tune_questions=len({r["question_hash"] for r in t}),
        fit_rows=len(f),
        tune_rows=len(t),
        test_label_reads=0,
        optimizer_steps=1000,
        config=CONFIG,
        scalar_numeric_checks=checks,
    )
    result["fit_tune_losses"] = {}
    result["head_hashes"] = {}
    result["temperature_choices"] = {}
    result["coefficient_changes"] = {}
    for arm in ARMS:
        fit_vectors = {
            r["answer_id"]: r["x"]
            for r in view["panel"]["rows"]
            if r["arm"] == arm and r["role"] == "fit"
        }
        fx = np.asarray([fit_vectors[r["unit_id"]] for r in f], dtype=float)
        result["temperature_choices"][arm] = {
            str(h["seed"]): h["temperature"] for h in result["heads"][arm]
        }
        theta = np.asarray(result["heads"][arm][0]["parameters"])
        result["coefficient_changes"][arm] = dict(
            changed=bool(np.any(theta != 0)),
            changed_count=int(np.sum(theta != 0)),
            maximum_absolute_change=float(np.max(np.abs(theta))) if len(theta) else 0.0,
            initial_parameters=[0.0] * len(theta),
            fitted_parameters=theta.tolist(),
            current_fitting_performed=arm == "scalar_cubic_spline",
        )
        result["head_hashes"][arm] = [
            canonical_hash(semantic({k: v for k, v in h.items() if k != "seed"}))
            for h in result["heads"][arm]
        ]
        result["fit_tune_losses"][arm] = [
            dict(
                seed=h["seed"],
                initial_fit_objective=h["initial_loss"],
                final_fit_objective=h["final_loss"],
                fit_log_loss=old.scalar.bce(
                    np.asarray(expit(scalar_logit(h, fx[:, 0]) / h["temperature"]))
                    if arm == "scalar_cubic_spline"
                    else old.predict(h, fx),
                    y,
                ),
                tune_temperature_losses=h["tune_temperature_losses"],
                chosen_tune_log_loss=h["tune_temperature_losses"][str(h["temperature"])],
                duration_s=h.get("duration_s"),
                current_optimization_duration_s=h.get("duration_s", 0.0),
                timing_missing_reason=None
                if "duration_s" in h
                else "original_producer_did_not_retain_per_head_optimization_timing",
            )
            for h in result["heads"][arm]
        ]
    result["duplicate_seed_heads"] = {
        arm: dict(unique_heads=len(set(result["head_hashes"][arm])), independent_replications=1)
        for arm in ARMS
    }
    result["gradient_checks"] = dict(
        passed=True,
        rows={
            **{
                f"{arm}:{seed}": original["gradient_checks"]["rows"][f"{source}:{seed}"]
                for arm, source in INHERITED.items()
                if arm != "raw_p2"
                for seed in SEEDS
            },
            **{f"scalar_cubic_spline:{seed}": checks[str(seed)] for seed in SEEDS},
        },
    )
    return result


def decision(p: float) -> Json:
    """Fixed thresholds define typed actions; confidence stays visible during escalation."""
    return dict(
        decision="accept" if p <= 0.2 else "reject" if p >= 0.8 else "escalate",
        confidence=max(p, 1 - p),
        probability=p,
    )


def predictions(result: Json, panel: Json) -> list[Json]:
    """Predict only after every arm is frozen, using a view with no evaluation targets."""
    rows: list[Json] = []
    for arm in ARMS:
        units = [r for r in panel["rows"] if r["arm"] == arm and r["role"] == "test"]
        x = np.asarray([r["x"] for r in units], dtype=float)
        for head in result["heads"][arm]:
            p = (
                expit(scalar_logit(head, x[:, 0]) / head["temperature"])
                if arm == "scalar_cubic_spline"
                else old.predict(head, x)
            )
            rows.extend(
                dict(
                    unit_id=r["unit_id"] + f":{head['seed']}",
                    answer_id=r["answer_id"],
                    question_hash=r["question_hash"],
                    arm=arm,
                    seed=head["seed"],
                    **decision(float(probability)),
                )
                for r, probability in zip(units, p, strict=True)
            )
        progress("frozen_test_prediction_" + arm, len(rows), 7500 - len(rows))
    return rows


@contextmanager
def binding() -> Iterator[None]:
    """The shipped measurement and schema retain their existing resource and failure checks."""
    with patch.multiple(
        old,
        NAME=NAME,
        TASK=TASK,
        CLI=CLI,
        TEST=TEST,
        OWNED=OWNED,
        UPSTREAM=UPSTREAM,
        UPSTREAM_PIN=UPSTREAM_PIN,
        ARMS=ARMS,
        CONFIG=CONFIG,
        methods=_METHODS,
        operands=operands,
        train=train,
        progress=progress,
    ):
        yield


def measure(root: Path, raw: Path) -> Json:
    """Authenticate CPU prerequisites before fitting and seal heads before test prediction."""
    with binding():
        work = dict(old.measure(root, raw))
    for failure in work["failures"]:
        if (
            failure.get("field", failure.get("artifact_field"))
            == "feature_panel_ready_score_and_sealed_fit_tune"
        ):
            failure.update(
                check="source_controls_and_authenticated_original_heads",
                field="source_controls_ready_score_and_required_head_seals",
                artifact_field="source_controls_ready_score_and_required_head_seals",
                upstream="exp8486-source-control-qualification",
                upstream_id="exp8486-source-control-qualification",
                sha256=UPSTREAM_PIN,
                artifact_hash=UPSTREAM_PIN,
            )
    work["preconditions"].update(measurement_cap_s=3600, optional_cuda_and_boards_gate_cpu=False)
    work["decisions"] = []
    if work["result"]:
        upstream, view = operands(root)
        work["source_refs"] += upstream["additional_refs"]
        work["decisions"] = predictions(work["result"], view["panel"])
    atomic_json(raw / "typed_decisions.json", dict(rows=work["decisions"], test_label_reads=0))
    work["primitive_files"].append(reference(raw / "typed_decisions.json"))
    work["ended_monotonic_ns"] = time.monotonic_ns()
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Control readiness qualifies numerical work, never independent factual generalization."""
    with binding():
        value = dict(old.build(work, receipts, raw, output))
    result = work["result"] or {}
    value.update(
        experiment_id=8487,
        task_id=TASK,
        milestone="2026.10.730",
        run_date="20261013",
        honest_verdict=f"complete_{value['verdict_class']}_v730_source_control_fit",
        head_controls_ready_score=value["head_ready_score"],
        frozen_heads_reference=reference(raw / "frozen_heads.json"),
        typed_decisions_reference=reference(raw / "typed_decisions.json"),
        fitting_rows=result.get("fit_tune_losses", {}),
        head_parameter_counts=dict(zip(ARMS, [0, 2, len(sparse.KNOTS) - 3, 109, 109], strict=True)),
        scalar_numeric_checks=result.get("scalar_numeric_checks", {}),
        fresh_process_bitwise_agreement=bool(result)
        and any(r.get("name") == "cold" and r["passed"] for r in receipts),
        test_prediction_count=len(work["decisions"]),
        optimization_timing_scope="Only the new scalar arm has current optimization timing; inherited timings remain missing.",
        methodology_note="Five fixed capacity controls on exposed cached answers; only q-only cubic coefficients fit now. Head readiness is qualification; no scientific benefit is measured.",
    )
    value["field_principles"].update(
        {
            k: "Authenticated primitive computations and fixed typed decisions; duplicate seeds are not independent replications."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        head_controls_ready_score="Qualification of frozen controls only, never scientific benefit.",
        head_parameter_counts="One scalar input contributes twelve cubic basis coefficients plus bias; no padding.",
        fresh_process_bitwise_agreement="Fresh cold replay recomputes predictions and compares their exact float64 bytes.",
    )
    value["historical_model_provenance"][0]["original_producer_hashes"] = work["source_refs"]
    for ref in value["cited_upstream_artifacts"]:
        name = Path(ref["path"]).name
        ref["imported_fields"] = (
            [
                "source_controls_ready_score",
                "feature_manifest_reference",
                "trainer_view",
                "operator_manifest",
            ]
            if ref["sha256"] == UPSTREAM_PIN
            else ["head_ready_score", "frozen_heads_path", "frozen_heads_sha256"]
            if ref["sha256"] == OLD_PIN
            else ["heads", "scaler", "fit_tune_losses"]
            if name == "frozen_heads.json"
            else ["rows"]
            if name == "trainer_view.json"
            else ["operators", "rows"]
            if name == "feature_panel.json"
            else ["scalar_fit", "decision", "source_controls", "historical_authorities"]
            if name == "v730-methods-protocol.json"
            else ["sealed_original_authority_bytes"]
        )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Reconstruct numeric operands independently so repaired local hashes cannot hide drift."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        for ref in [*work["source_refs"], *work["code_refs"], *work["primitive_files"]]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("operand_drift")
        with TemporaryDirectory(prefix="exp8487-cold-", dir="/var/tmp") as directory:
            root = Path(directory)
            for role, name in [
                ("design", methods.DESIGN),
                ("active", methods.ACTIVE),
                ("staged", methods.STAGED),
            ]:
                snap = work["authority"]["authority_snapshots"][role]
                if "snapshot_path" in snap:
                    source = Path(snap["snapshot_path"])
                    if sha256_file(source) != snap["sha256"]:
                        raise ValueError("authority_snapshot_drift")
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(source.read_bytes())
            if (
                methods.authority(root, root / "authority")["activated"]
                != work["authority"]["activated"]
            ):
                raise ValueError("authority_drift")
        raw = Path(value["primitive_reference"]["path"]).parent
        frozen = read(reference(raw / "frozen_heads.json"))
        if work["result"]:
            if work["upstream_reference"]["sha256"] != UPSTREAM_PIN:
                raise ValueError("independent_upstream_pin")
            _, view = operands(Path(work["upstream_reference"]["path"]).parents[1])
            expected = train(view, [])
            if semantic(expected) != semantic(work["result"]) or semantic(expected) != semantic(
                frozen
            ):
                raise ValueError("frozen_fit_drift")
            decisions = predictions(expected, view["panel"])
            saved = read(reference(raw / "typed_decisions.json"))["rows"]
            if decisions != work["decisions"] or decisions != saved:
                raise ValueError("typed_decision_drift")
            if (
                np.asarray([r["probability"] for r in decisions]).tobytes()
                != np.asarray([r["probability"] for r in saved]).tobytes()
            ):
                raise ValueError("prediction_bit_drift")
            if read(reference(raw / "trainer_operands.json"))["view"] != view:
                raise ValueError("trainer_operand_drift")
        elif frozen.get("heads") or work["decisions"]:
            raise ValueError("missing_operand_fitted_head")
        for receipt in value["validation_receipts"]:
            for stream in ["stdout", "stderr"]:
                if (
                    stream + "_path" in receipt
                    and sha256_file(Path(receipt[stream + "_path"])) != receipt[stream + "_sha256"]
                ):
                    raise ValueError("receipt_drift")
        return bool(
            value
            == build(work, value["validation_receipts"], raw, Path(value["publication_output"]))
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
