"""REQ-REPORT-9099-V776: single-vector human updates with durable delayed feedback.

This learner owns no future or retention targets. The evaluator releases one
already predicted question at a time; a complete snapshot commits its effect.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from carnot.autoresearch import spline_basis_8986 as basis
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify.direct_atomic_state_8376 import sync_directory

Json = dict[str, Any]
ARMS = (
    "frozen_spline",
    "adaptive_spline",
    "bias_only_spline",
    "adaptive_linear",
    "donor_label_spline",
)


def arms(request: Json) -> tuple[str, ...]:
    """The logit protocol adds a control without changing the embedding study."""
    return ARMS + ("temperature_only_spline",) if request.get("feature_count") == 7 else ARMS


def temperature_step(u: float, accumulator: float, score: float, label: int) -> tuple[float, float]:
    """Learn log temperature while keeping all initial spline coefficients fixed."""
    if (
        not np.isfinite([u, accumulator, score]).all()
        or accumulator < 0
        or not np.log(0.5) <= u <= np.log(4)
        or type(label) is not int
        or label not in (0, 1)
    ):
        raise ValueError("temperature_operands")
    scaled = float(np.exp(-u) * score)
    gradient = float(np.clip((label - expit(scaled)) * scaled + 0.01 * u, -1, 1))
    total = accumulator + gradient**2
    return float(
        np.clip(u - 0.05 * gradient / np.sqrt(total + 1e-8), np.log(0.5), np.log(4))
    ), total


def transform(x: Any, fitted: Json) -> Any:
    """Apply only initialization statistics so future observations cannot move the basis."""
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[1] != len(fitted["mean"]) or not np.isfinite(a).all():
        raise ValueError("transform_operands")
    projected = ((a - fitted["mean"]) / fitted["scale"]) @ np.asarray(fitted["projection"])
    return basis.scale(projected, fitted["ranges"])[0]


def fit_initial(x: Any, labels: Any) -> Json:
    """Refit a proper-loss head and all transforms on exactly300 initialization siblings."""
    a, y = np.asarray(x, dtype=float), np.asarray(labels, dtype=float)
    if (
        a.ndim != 2
        or len(a) != 300
        or y.shape != (300,)
        or not np.isfinite(a).all()
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError("initialization_operands")
    scale = a.std(axis=0)
    scale[scale == 0] = 1
    projection = np.random.default_rng(775084).normal(0, 1 / np.sqrt(27), (a.shape[1], 27))
    fitted = dict(
        mean=a.mean(axis=0).tolist(), scale=scale.tolist(), projection=projection.tolist()
    )
    fitted["ranges"] = basis.fit_scaling(((a - fitted["mean"]) / scale) @ projection)
    scaled = transform(a, fitted)
    fits, initial = {}, {}
    for kind in ("spline", "linear"):
        phi = basis.design(scaled, kind == "spline")
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
            raise ValueError("initialization_nonconvergence")
        fits[kind] = dict(
            coefficients=result.x.tolist(),
            gradient_inf=residual,
            loss=loss,
            iterations=int(result.nit),
        )
    for arm in ARMS:
        initial[arm] = deepcopy(
            fits["linear" if arm == "adaptive_linear" else "spline"]["coefficients"]
        )
    return dict(initial=initial, transform=fitted, fits=fits, fit_rows=300, temperature=1)


def step(
    theta: Any, accumulator: Any, initial: Any, phi: Any, label: int, bias: bool
) -> tuple[Any, Any]:
    """Anchor nonintercept coefficients, clip once, then accumulate the applied gradient."""
    t, acc, start, x = (np.asarray(v, dtype=float) for v in (theta, accumulator, initial, phi))
    if (
        t.ndim != 1
        or not len(t)
        or any(v.shape != t.shape for v in (acc, start, x))
        or type(label) is not int
        or label not in (0, 1)
        or np.any(acc < 0)
        or not all(np.isfinite(v).all() for v in (t, acc, start, x))
    ):
        raise ValueError("update_operands")
    anchor = 0.01 * (t - start)
    anchor[0] = 0
    gradient = (expit(t @ x) - label) * x + anchor
    if bias:
        gradient[1:] = 0
    gradient /= max(1.0, float(np.linalg.norm(gradient)))
    total = acc + gradient**2
    return t - 0.05 * gradient / np.sqrt(total + 1e-8), total


def guard(candidate: Any, initial: Any, design: Any, labels: Any) -> Json:
    """All150 guard rows compare to the arm's initial state, never a moving baseline."""
    phi, y = np.asarray(design, dtype=float), np.asarray(labels)
    if (
        phi.shape != (150, len(initial))
        or y.shape != (150,)
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(phi).all()
    ):
        raise ValueError("guard_roster")
    before, after = expit(phi @ initial), expit(phi @ candidate)
    brier = float(np.mean((after - y) ** 2 - (before - y) ** 2))
    error = int(np.sum((after >= 0.5) != y) - np.sum((before >= 0.5) != y))
    return dict(
        brier_increase=brier, error_increase=error / 150, passed=brier <= 0.005 and error <= 0
    )


def initial(request: Json) -> Json:
    """A new stream receives identical coefficients and fresh zero accumulators."""
    state = dict(
        request_hash=canonical_hash(request),
        query=0,
        phase="predict",
        coefficients=deepcopy(request["initial"]),
        accumulators={a: np.zeros_like(request["initial"][a]).tolist() for a in arms(request)},
        rows=[],
        retention_rows=[],
        pending=[],
        acknowledgments=[],
        feedback_rows=[],
        rollback_checks=[],
    )
    if request.get("feature_count") == 7:
        state["temperature"] = dict(u=0.0, accumulator=0.0)
    return state


def predictions(state: Json, request: Json, role: str, group: int) -> list[Json]:
    """Predict all siblings and all arms from one state before the evaluator sees a label."""
    rows = []
    for arm in arms(request):
        kind = "linear" if arm == "adaptive_linear" else "spline"
        phi = np.asarray(request[role + "_design"][kind])[group]
        q = np.exp(-state["temperature"]["u"]) if arm == "temperature_only_spline" else 1
        for sibling, p in enumerate(expit(q * (phi @ state["coefficients"][arm]))):
            rows.append(
                dict(
                    unit_id=f"{request[role + '_qids'][group]}:{sibling}:{arm}",
                    original_qid=request[role + "_qids"][group],
                    group_id=request[role + "_qids"][group],
                    sibling=sibling,
                    group=group,
                    arm=arm,
                    condition=request["condition"],
                    seed=request.get("random_seed", 776099),
                    metric="p_error",
                    absolute_metric=float(p),
                    value=float(p),
                    numerator=float(p),
                    denominator=1,
                    decision="accept" if p <= 0.1 else "reject" if p >= 0.9 else "escalate",
                    status="completed",
                    exclusion_reason=None,
                    censor_reason=None,
                )
            )
    return rows


def release(
    state: Json, request: Json, labels: Path, group: int, *, role: str = "online"
) -> list[int]:
    """The evaluator permits only sealed past siblings; retention has no update interface."""
    if role != "online":
        raise PermissionError("retention_label_access_refused")
    if (
        group not in state["pending"]
        or group != min(state["pending"])
        or len(state["rows"]) != state["query"] * 3 * len(arms(request))
        or not (
            group == state["query"] - 2
            or group == 4
            and state["query"] == 5
            and state["phase"] == "drain"
        )
        or state["request_hash"] != canonical_hash(request)
    ):
        raise ValueError("prediction_seal")
    values = json.loads(labels.read_bytes())["labels"]
    if len(values) != 5 or any(
        len(row) != 3 or any(type(y) is not int or y not in (0, 1) for y in row) for row in values
    ):
        raise ValueError("feedback_labels")
    return list(values[group])


def feedback(state: Json, request: Json, group: int, labels: list[int]) -> bool:
    """One atomic effect includes coefficients, accumulators and acknowledgment, even on rejection."""
    if group in state["acknowledgments"]:
        if next(r["labels"] for r in state["feedback_rows"] if r["group"] == group) != labels:
            raise ValueError("duplicate_feedback_drift")
        return False
    if (
        group not in state["pending"]
        or group != min(state["pending"])
        or len(labels) != 3
        or any(type(y) is not int or y not in (0, 1) for y in labels)
    ):
        raise ValueError("feedback_order")
    for arm in ARMS[1:]:
        kind = "linear" if arm == "adaptive_linear" else "spline"
        before = [state["coefficients"][arm], state["accumulators"][arm]]
        theta, acc = (np.asarray(v) for v in before)
        ys = request["donor_labels"][group] if arm == "donor_label_spline" else labels
        for phi, y in zip(request["online_design"][kind][group], ys, strict=True):
            theta, acc = step(
                theta, acc, request["initial"][arm], phi, y, arm == "bias_only_spline"
            )
        checked = guard(
            theta, request["initial"][arm], request["guard_design"][kind], request["guard_labels"]
        )
        if checked["passed"]:
            state["coefficients"][arm], state["accumulators"][arm] = theta.tolist(), acc.tolist()
        state["rollback_checks"].append(
            dict(
                arm=arm,
                group=group,
                before_hash=canonical_hash(before),
                candidate_hash=canonical_hash([theta.tolist(), acc.tolist()]),
                after_hash=canonical_hash([state["coefficients"][arm], state["accumulators"][arm]]),
                **checked,
            )
        )
    if request.get("feature_count") == 7:
        arm = "temperature_only_spline"
        before_temperature = deepcopy(state["temperature"])
        u, total = before_temperature["u"], before_temperature["accumulator"]
        theta = np.asarray(request["initial"][arm])
        for phi, y in zip(request["online_design"]["spline"][group], labels, strict=True):
            u, total = temperature_step(u, total, float(theta @ phi), y)
        candidate = dict(u=u, accumulator=total)
        checked = guard(
            np.exp(-u) * theta, theta, request["guard_design"]["spline"], request["guard_labels"]
        )
        if checked["passed"]:
            state["temperature"] = candidate
        state["rollback_checks"].append(
            dict(
                arm=arm,
                group=group,
                before_hash=canonical_hash(before_temperature),
                candidate_hash=canonical_hash(candidate),
                after_hash=canonical_hash(state["temperature"]),
                **checked,
            )
        )
    state["pending"].remove(group)
    state["acknowledgments"].append(group)
    state["feedback_rows"].append(
        dict(group=group, labels=labels, prediction_hash=canonical_hash(state["rows"]))
    )
    return True


def checked_request(request: Json) -> None:
    """Reject the old graph shape and malformed features before creating durable state."""
    known = request["initial_labels"]
    if (
        len(known) != 100
        or set(known) & (set(request["online_qids"]) | set(request["retention_qids"]))
        or len(request["donor_qids"]) != 5
        or any(q not in known for q in request["donor_qids"])
        or request["donor_labels"] != [known[q] for q in request["donor_qids"]]
    ):
        raise ValueError("donor_causality")
    if (
        request.get("feature_count", 27) not in (7, 27)
        or set(request["initial"]) != set(arms(request))
        or len(set(request["online_qids"])) != 5
        or len(set(request["retention_qids"])) != 100
        or set(request["online_qids"]) & set(request["retention_qids"])
    ):
        raise ValueError("human_stream_roster")
    count = request.get("feature_count", 27)
    for kind, width in [("spline", 1 + count * 8), ("linear", 1 + count)]:
        for role, shape in [
            ("online", (5, 3, width)),
            ("retention", (100, 3, width)),
            ("guard", (150, width)),
        ]:
            phi = np.asarray(request[role + "_design"][kind], dtype=float)
            if phi.shape != shape or not np.isfinite(phi).all() or not np.all(phi[..., 0] == 1):
                raise ValueError("human_design")
    for arm in arms(request):
        theta = np.asarray(request["initial"][arm])
        if (
            theta.shape != (1 + count if arm == "adaptive_linear" else 1 + count * 8,)
            or not np.isfinite(theta).all()
        ):
            raise ValueError("human_head")
    if count == 7 and any(
        request["initial"][a] != request["initial"]["frozen_spline"]
        for a in arms(request)
        if a != "adaptive_linear"
    ):
        raise ValueError("initial_spline_parity")
    donor = np.asarray(request["donor_labels"])
    if donor.shape != (5, 3) or not np.isin(donor, [0, 1]).all():
        raise ValueError("donor_labels")
    guard(
        request["initial"]["adaptive_spline"],
        request["initial"]["adaptive_spline"],
        request["guard_design"]["spline"],
        request["guard_labels"],
    )


def commit(
    path: Path, state: Json, hook: Callable[[str], None], *, prediction: bool = False
) -> None:
    """Fsync complete state before atomic replacement; acknowledge only after directory sync."""
    temporary = path.with_name(".state.tmp")
    with temporary.open("w") as stream:
        json.dump(dict(state=state, sha256=canonical_hash(state)), stream, sort_keys=True)
        if prediction:
            hook("before_prediction_fsync")
        stream.flush()
        os.fsync(stream.fileno())
    hook("during_atomic_replace")
    temporary.replace(path)
    sync_directory(path.parent)


def drive(request: Json, directory: Path, labels: Path, *, crash: str = "") -> Json:
    """Resume the last complete transaction and never apply acknowledged feedback twice."""
    checked_request(request)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / "state.json"
    state = initial(request)
    if path.exists():
        envelope = json.loads(path.read_bytes())
        state = envelope["state"]
        if envelope["sha256"] != canonical_hash(state) or state["request_hash"] != canonical_hash(
            request
        ):
            raise ValueError("resume_state")

    def hook(phase: str) -> None:
        """A real child exits at a frozen transaction boundary without Python cleanup."""
        if crash == phase and state["query"] == (3 if request.get("feature_count") == 7 else 2):
            print(f"[exp9099-child] hard_exit={phase}", flush=True)
            os._exit(73)

    while state["phase"] != "done":
        if state["phase"] == "predict":
            state["rows"].extend(predictions(state, request, "online", state["query"]))
            state["pending"].append(state["query"])
            state["query"] += 1
            state["phase"] = "release"
            commit(path, state, hook, prediction=True)
            print(
                f"[exp9099-child] sealed_groups={state['query']} pending={len(state['pending'])}",
                flush=True,
            )
        if state["phase"] == "release":
            group = state["query"] - 2
            if group in state["pending"]:
                ys = release(state, request, labels, group)
                hook("after_feedback_receipt")
                feedback(state, request, group, ys)
                commit(path, state, hook)
            state["phase"] = "predict" if state["query"] < 5 else "drain"
            commit(path, state, hook)
        if state["phase"] == "drain":
            if 4 in state["pending"]:
                feedback(state, request, 4, release(state, request, labels, 4))
            state["retention_rows"] = [
                r for group in range(100) for r in predictions(state, request, "retention", group)
            ]
            state["phase"] = "done"
            commit(path, state, hook)
    return state
