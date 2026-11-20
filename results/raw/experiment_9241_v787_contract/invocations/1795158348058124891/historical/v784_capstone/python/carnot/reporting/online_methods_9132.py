"""REQ-REPORT-9132-V778: qualify delayed logit mechanics without model access.

The exposed human roster fixes a later estimand. Constructed CPU streams test
causality and recovery; their gains cannot establish human prediction benefit.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize

from carnot.autoresearch import spline_basis_8986 as basis
from carnot.memory import human_online_9099 as learner
from carnot.reporting import logit_methods_9126 as logits
from carnot.reporting import online_methods_9099 as old
from carnot.reporting.current_work_receipt import (
    atomic_json as atomic_json,
    canonical_hash as canonical_hash,
)
from carnot.reporting.logit_methods_9126 import ref as ref
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_9132_v778_online_methods", "exp9132-online-methods"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_online_methods_9132.py"
OWNED = [
    "python/carnot/memory/human_online_9099.py",
    "python/carnot/reporting/online_methods_9132.py",
    "python/carnot/reporting/online_methods_runner_9132.py",
    CLI,
]
PROTOCOL = "openspec/change-proposals/v778-online-logit-protocol.json"
UPSTREAM = "results/experiment_9099_v776_online_methods.json"
HISTORICAL_WORKER = "results/raw/experiment_9124_v778_contract/invocations/1794805159615746781/authority/results/raw/experiment_9109_v776_capstone/invocations/1794766780181981018/retained/fafe8a078530887722006d1d2bcb6b81f38aaa9c1c14d9d81c1393bf9e8093ec/human_online_9099.py"
PINS = {
    UPSTREAM: (
        "sha256:ca82730179091d4a80d9a4c1384f04d0e38e24937c22b4277932b0dce7af34c7",
        "online_methods_ready_score",
    ),
    "results/experiment_9126_v778_logit_methods.json": (
        "sha256:16cd3742f2ec4dea0d67b6503e3d9e4dd84f7a3898fdbe15087065208ed1a1fb",
        "methods_ready_score",
    ),
}
MODEL_SPECS: list[Json] = []
CONDITIONS = (
    "positive",
    "no_signal",
    "donor_mismatch",
    "constant_feature",
    "future_labels",
    "sibling_labels",
    "guard_reject",
)


def progress(phase: str) -> None:
    """Flush boundaries so numerical work and blocked inputs remain visible."""
    print(f"[exp9132] phase={phase}", flush=True)


def protocol() -> Json:
    """Read separate frozen choices without rewriting the original embedding study."""
    return dict(json.loads((ROOT / PROTOCOL).read_bytes()))


def authenticate(root: Path) -> Json:
    """Pins precede fields, transitive operands and reconstruction of original rosters."""
    checks, references, values = [], [], {}
    for name, (pin, field) in PINS.items():
        start = len(checks)
        path = root / name
        references.append(ref(path))
        checked = logits.byte_check(path, pin)
        checks.append(checked)
        if checked["passed"]:
            checks.append(logits.field_check(path, field, 1))
            source = json.loads(path.read_bytes())
            values[name] = source
            key = "online_protocol_reference" if name == UPSTREAM else "protocol_reference"
            reference = source.get(key)
            checks.append(
                logits.check(
                    path,
                    key,
                    "present",
                    "present" if isinstance(reference, dict) else "missing_field",
                    "reference_field",
                )
            )
            bound = [*source["source_references"]]
            if isinstance(reference, dict):
                bound.append(reference)
            for reference in bound:
                target = Path(reference["path"])
                references.append(ref(target))
                checks.append(logits.byte_check(target, reference["sha256"]))
        for check in checks[start:]:
            check["upstream"] = (
                "exp9099-online-methods" if name == UPSTREAM else "exp9126-logit-methods"
            )
    roster: Json = {}
    if all(c["passed"] for c in checks):
        historical = ROOT / HISTORICAL_WORKER
        declared = next(
            r["sha256"]
            for r in values[UPSTREAM]["code_config_hashes"]
            if r["path"].endswith("/human_online_9099.py")
        )
        references.append(ref(historical))
        checks.append(
            dict(logits.byte_check(historical, declared), upstream="exp9099-online-methods")
        )
    if all(c["passed"] for c in checks):
        inherited = old.authenticate(root)
        checks.extend(inherited["gate_check_summary"])
        references.extend(inherited["source_references"])
        roster = inherited["roster"]
        if not inherited["blocker"]:
            for key in [
                "initial_group_roster",
                "stream_roster",
                "guard_roster",
                "retention_roster",
                "donor_map",
                "sibling_roster",
            ]:
                checks.append(
                    dict(
                        logits.check(
                            root / UPSTREAM,
                            key,
                            roster[key],
                            values[UPSTREAM][key],
                            "original_roster",
                        ),
                        upstream="exp9099-online-methods",
                    )
                )
    return dict(
        gate_check_summary=checks,
        source_references=references,
        roster=roster,
        blocker=next((c["check"] for c in checks if not c["passed"]), None),
    )


def transform(x: Any, fitted: Json) -> Any:
    """Impute and scale using only initialization operands, even for future extremes."""
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[1] != 7 or np.isinf(a).any():
        raise ValueError("logit_transform")
    a = (np.where(np.isnan(a), fitted["median"], a) - fitted["mean"]) / fitted["scale"]
    low, high = np.asarray(fitted["minimum"]), np.asarray(fitted["maximum"])
    span = high - low
    return np.clip(np.where(span > 0, 2 * (a - low) / np.where(span > 0, span, 1) - 1, 0), -1, 1)


def design(x: Any, spline: bool) -> Any:
    """Reuse the original fixed cubic knots directly on seven logit coordinates."""
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[1] != 7 or not np.isfinite(a).all() or np.any(np.abs(a) > 1):
        raise ValueError("logit_design")
    linear = np.column_stack((np.ones(len(a)), a))
    return (
        np.column_stack((linear, BSpline(basis.KNOTS, np.eye(7), 3)(a).reshape(len(a), 49)))
        if spline
        else linear
    )


def fit_initial(x: Any, labels: Any) -> Json:
    """Fit exactly300 initialization siblings; no static150 fit is an accepted input."""
    a, y = np.asarray(x, dtype=float), np.asarray(labels)
    if (
        a.shape != (300, 7)
        or y.shape != (300,)
        or np.isinf(a).any()
        or np.isnan(a).all(axis=0).any()
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError("logit_initialization")
    median = np.nanmedian(a, axis=0)
    imputed = np.where(np.isnan(a), median, a)
    scale = imputed.std(axis=0)
    scale[scale == 0] = 1
    standardized = (imputed - imputed.mean(axis=0)) / scale
    fitted = dict(
        median=median.tolist(),
        mean=imputed.mean(axis=0).tolist(),
        scale=scale.tolist(),
        minimum=standardized.min(axis=0).tolist(),
        maximum=standardized.max(axis=0).tolist(),
    )
    fits: Json = {}
    for kind in ("spline", "linear"):
        phi = design(transform(a, fitted), kind == "spline")
        result = minimize(
            basis.objective,
            np.zeros(phi.shape[1]),
            args=(phi, y),
            method="L-BFGS-B",
            jac=True,
            options=dict(maxiter=2000, gtol=1e-7, ftol=1e-12),
        )
        loss, gradient = basis.objective(result.x, phi, y)
        residual = float(np.max(np.abs(gradient)))
        if (
            not np.isfinite(loss)
            or not np.isfinite(result.x).all()
            or not np.isfinite(gradient).all()
            or residual > 1e-6
        ):
            raise ValueError("logit_nonconvergence")
        fits[kind] = dict(
            coefficients=result.x.tolist(),
            loss=loss,
            gradient_inf=residual,
            iterations=int(result.nit),
        )
    initial = {
        a: deepcopy(fits["linear" if a == "adaptive_linear" else "spline"]["coefficients"])
        for a in learner.arms(dict(feature_count=7))
    }
    return dict(initial=initial, transform=fitted, fits=fits, fit_rows=300, temperature=1)


def operands(condition: str) -> tuple[Json, Json]:
    """Freeze synthetic signal and negative controls without reading human labels."""
    y = np.repeat([0, 1], 150)
    x = np.repeat((2 * y - 1)[:, None], 7, axis=1).astype(float)
    y[113:150], y[263:300] = 1, 0
    if condition == "constant_feature":
        x[:] = 1
    fit = fit_initial(x, y)
    online = np.ones((15, 7))
    guard, retention = np.zeros((150, 7)), np.zeros((300, 7))
    request = dict(
        feature_count=7,
        random_seed=778132,
        initial=fit["initial"],
        initial_labels={f"fixture-init-{i}": y.reshape(100, 3)[i].tolist() for i in range(100)},
        donor_qids=["fixture-init-0"] * 5,
        donor_labels=[[0, 0, 0]] * 5,
        online_qids=[f"fixture-stream-{i}" for i in range(5)],
        retention_qids=[f"fixture-retention-{i}" for i in range(100)],
        online_design={},
        guard_design={},
        retention_design={},
        guard_labels=np.tile([0, 1], 75).tolist(),
        condition="cpu_fixture_" + condition,
    )
    for kind, width in [("spline", 57), ("linear", 8)]:
        request["online_design"][kind] = (
            design(online, kind == "spline").reshape(5, 3, width).tolist()
        )
        request["guard_design"][kind] = design(guard, kind == "spline").tolist()
        request["retention_design"][kind] = (
            design(retention, kind == "spline").reshape(100, 3, width).tolist()
        )
    labels = (
        np.random.default_rng(778132).integers(0, 2, (5, 3)).tolist()
        if condition == "no_signal"
        else [[1, 1, 1]] * 5
    )
    if condition == "future_labels":
        labels[4] = [0, 0, 0]
    if condition == "sibling_labels":
        labels[0] = [0, 0, 0]
    if condition in ("guard_reject", "constant_feature"):
        request["guard_labels"] = [0] * 150
    if condition == "constant_feature":
        request["donor_qids"] = ["fixture-init-50"] * 5
        request["donor_labels"] = [request["initial_labels"][q] for q in request["donor_qids"]]
    if condition == "guard_reject":
        for kind in ("spline", "linear"):
            request["guard_design"][kind] = request["online_design"][kind][0] * 50
    return request, dict(labels=labels)


def evaluate(raw: Path) -> Json:
    """Run frozen controls and genuine process deaths before claiming recovery readiness."""
    for condition in CONDITIONS:
        progress("fixture_before_" + condition)
        request, labels = operands(condition)
        directory = raw / condition
        atomic_json(directory / "request.json", request)
        atomic_json(directory / "labels.json", labels)
        learner.drive(request, directory / "store", directory / "labels.json")
        progress("fixture_after_" + condition)
    hard, resumed = [], []
    for point in protocol()["recovery"]["hard_exits"]:
        directory = raw / point
        argv = [
            str(ROOT / ".venv/bin/python"),
            "-u",
            str(ROOT / CLI),
            "--worker",
            str(raw / "positive/request.json"),
            str(directory),
            str(raw / "positive/labels.json"),
        ]
        hard.append(
            child(
                point,
                argv + ["--crash", point],
                raw / "child_logs",
                expected=73,
                deadline=90,
                heartbeat=30,
            )
        )
        if hard[-1]["passed"]:
            atomic_json(
                directory / "interrupted.json", json.loads((directory / "state.json").read_bytes())
            )
        resumed.append(
            child(point + "_resume", argv, raw / "child_logs", deadline=90, heartbeat=30)
        )
    atomic_json(raw / "recovery_receipts.json", dict(hard=hard, resumed=resumed))
    return reduce(raw)


def temperature_reference() -> list[Json]:
    """A separate scalar logloss derivative checks sign, clipping and optimizer memory."""
    rows = []
    for u, accumulator, score, y in [(0.0, 0.0, 2.0, 1), (0.2, 0.3, -1.3, 0), (0.0, 0.0, 99.0, 0)]:
        q = math.exp(-u)
        p = 1 / (1 + math.exp(-q * score))
        raw = (y - p) * q * score + 0.01 * u
        clipped = min(1.0, max(-1.0, raw))
        total = accumulator + clipped**2
        expected = min(
            math.log(4), max(math.log(0.5), u - 0.05 * clipped / math.sqrt(total + 1e-8))
        )
        actual = learner.temperature_step(u, accumulator, score, y)
        rows.append(
            dict(
                u=u,
                accumulator=accumulator,
                score=score,
                label=y,
                raw_gradient=raw,
                clipped_gradient=clipped,
                expected_u=expected,
                expected_accumulator=total,
                observed=list(actual),
                passed=abs(actual[0] - expected) < 1e-14 and abs(actual[1] - total) < 1e-14,
            )
        )
    return rows


def reduce(raw: Path) -> Json:
    """Reconstruct every state so rehashed prediction or temperature tampering still fails."""
    states = {}
    for condition in CONDITIONS:
        request, labels = operands(condition)
        directory = raw / condition
        if (
            json.loads((directory / "request.json").read_bytes()) != request
            or json.loads((directory / "labels.json").read_bytes()) != labels
        ):
            raise ValueError("fixture_operand_drift")
        with tempfile.TemporaryDirectory(prefix="carnot-9132-replay-") as replay:
            fresh = learner.drive(request, Path(replay), directory / "labels.json")
        envelope = json.loads((directory / "store/state.json").read_bytes())
        if fresh != envelope["state"] or canonical_hash(fresh) != envelope["sha256"]:
            raise ValueError("trajectory_drift")
        states[condition] = fresh
    positive = states["positive"]
    duplicate = deepcopy(positive)
    no_second_effect = (
        not learner.feedback(duplicate, operands("positive")[0], 0, [1, 1, 1])
        and duplicate == positive
    )

    def predictions(condition: str, arm: str | None = None) -> list[Any]:
        """Compare sealed values while preserving the distinct fixture condition labels."""
        return [
            (row["unit_id"], row["value"])
            for row in states[condition]["rows"]
            if arm is None or row["arm"] == arm
        ]

    rejected = states["guard_reject"]
    checks = dict(
        future_label=predictions("positive") == predictions("future_labels"),
        sibling_leak=predictions("positive")[:36] == predictions("sibling_labels")[:36],
        duplicate_ack=no_second_effect,
        donor_causality=predictions("positive", "donor_label_spline")
        == predictions("sibling_labels", "donor_label_spline"),
        rollback=all(
            not row["passed"] and row["before_hash"] == row["after_hash"]
            for row in rejected["rollback_checks"]
            if row["arm"] != "donor_label_spline"
        )
        and rejected["acknowledgments"] == list(range(5)),
    )
    rows, metrics = [], {}
    for condition in CONDITIONS[:4]:
        labels = operands(condition)[1]["labels"]
        scored = [
            dict(
                row,
                qid=row["original_qid"],
                metric="Brier",
                p_error=row["value"],
                label=labels[row["group"]][row["sibling"]],
                value=(row["value"] - labels[row["group"]][row["sibling"]]) ** 2,
                absolute_metric=(row["value"] - labels[row["group"]][row["sibling"]]) ** 2,
                numerator=(row["value"] - labels[row["group"]][row["sibling"]]) ** 2,
            )
            for row in states[condition]["rows"]
        ]
        for row in scored:
            row["unit_id"] = condition + ":" + row["unit_id"]
        rows.extend(scored)
        metrics[condition] = {
            arm: float(
                np.mean([row["value"] for row in scored if row["arm"] == arm and row["group"] >= 1])
            )
            for arm in learner.arms(dict(feature_count=7))
        }
    checks.update(
        positive_learning=metrics["positive"]["adaptive_spline"]
        < metrics["positive"]["frozen_spline"],
        temperature_learning=metrics["positive"]["temperature_only_spline"]
        < metrics["positive"]["frozen_spline"],
        no_signal_fixture=len(states["no_signal"]["acknowledgments"]) == 5,
        donor_mismatch=metrics["donor_mismatch"]["adaptive_spline"]
        < metrics["donor_mismatch"]["donor_label_spline"],
        constant_feature=np.allclose(
            [row["value"] for row in states["constant_feature"]["rows"]], 0.5
        ),
    )
    receipts = json.loads((raw / "recovery_receipts.json").read_bytes())
    exact_rows = []
    for point, hard, resumed in zip(
        protocol()["recovery"]["hard_exits"], receipts["hard"], receipts["resumed"], strict=True
    ):
        for receipt in (hard, resumed):
            for stream in ("stdout", "stderr"):
                if ref(Path(receipt[stream + "_path"]))["sha256"] != receipt[stream + "_sha256"]:
                    raise ValueError("hard_exit_log_drift")
        interrupted = json.loads((raw / point / "interrupted.json").read_bytes())
        checkpoint = interrupted["state"]
        final = json.loads((raw / point / "state.json").read_bytes())
        exact = (
            hard["passed"]
            and hard["normal_exit"]
            and hard["exit_code"] == 73
            and not hard["timed_out"]
            and resumed["passed"]
            and resumed["exit_code"] == 0
            and checkpoint["query"] in (2, 3)
            and bool(checkpoint["pending"])
            and checkpoint["temperature"]["accumulator"] > 0
            and checkpoint["temperature"]["u"] != 0
            and interrupted["sha256"] == canonical_hash(checkpoint)
            and final["sha256"] == canonical_hash(final["state"])
            and final["state"] == positive
        )
        exact_rows.append(
            dict(
                check=point,
                passed=bool(exact),
                uninterrupted_hash=canonical_hash(positive),
                resumed_hash=canonical_hash(final["state"]),
                prediction_equal=final["state"]["rows"] == positive["rows"],
                temperature_equal=final["state"]["temperature"] == positive["temperature"],
                interrupted_temperature=checkpoint["temperature"],
            )
        )
    reference = temperature_reference()
    checks["temperature_gradient"] = all(row["passed"] for row in reference)
    return dict(
        rows=rows,
        fixture_metrics=metrics,
        causal_controls=[dict(check=k, passed=bool(v)) for k, v in checks.items()],
        hard_exit_receipts=receipts["hard"],
        recovery_receipts=receipts["resumed"],
        exact_recovery_rows=exact_rows,
        exact_resume=len(exact_rows) == 3 and all(row["passed"] for row in exact_rows),
        rollback_checks=positive["rollback_checks"] + rejected["rollback_checks"],
        temperature_gradient_reference_rows=reference,
    )
