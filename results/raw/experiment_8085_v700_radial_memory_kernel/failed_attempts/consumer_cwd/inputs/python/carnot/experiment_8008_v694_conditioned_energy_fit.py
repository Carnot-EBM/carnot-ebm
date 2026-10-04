"""REQ-REPORT-8008: condition a small binary head without new generator calls.

The intercept starts at fitting prevalence. Centering and curvature address
optimizer scale; they do not establish a new architecture or decision benefit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
from scipy.special import expit  # type: ignore[import-untyped]
from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from carnot import experiment_8007_v694_conditioning_diagnosis as prior
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import sparse_energy_7996 as sparse
from carnot.verify import qwen_energy_calibration_7972 as scalar

Json = dict[str, Any]
Array = prior.Array
ROOT = prior.ROOT
NAME = "experiment_8008_v694_conditioned_energy_fit"
TASK = "exp8008-conditioned-energy-fit"
UPSTREAM_TASK = prior.TASK
OWNED = [f"python/carnot/{NAME}.py", f"scripts/experiments/{NAME}.py"]
TEST = "tests/python/test_conditioned_energy_8008.py"
CONFIG = dict(
    seeds=list(sparse.SEEDS),
    penalties=[0.0001, 0.001, 0.01],
    max_steps=2000,
    fit_budget_s=600,
    gradient_tolerance=1e-8,
    learning_rate=0.01,
    calibration_groups=62,
    calibration_bounds=[[-8.0, 8.0], [0.25, 4.0]],
    calibration_l2=0.0001,
    clip_q=[0.0001, 0.9999],
    knots=sparse.KNOTS,
    arms=[
        "conditioned_energy",
        "linear",
        "intercept_only",
        "equal_information_spline",
        "fixed_step",
        "frozen_v693",
        "isotonic",
    ],
    optimizer="damped_Newton_backtracking",
    initialization="fit_prevalence_intercept_then_converged_two_coefficient_linear_then_local_residuals",
    selection="tune_binary_log_loss",
    compute_matching="same maximum budget and paired full-batch steps; record Hessian overhead",
    acceptance="all converged owned heads, numerical audits, CLI replay and owned checks",
    decision_benefit="unassessed",
    intercept_novelty=False,
)
BEGAN = time.monotonic()
threadpool_limits(limits=1)


def progress(phase: str, units: int = 0, pending: int = 0) -> None:
    """Real elapsed time and counts let an operator distinguish work from a stall."""
    print(
        f"[exp8008] phase={phase} elapsed_s={time.monotonic() - BEGAN:.3f} "
        f"units={units} pending={pending}",
        flush=True,
    )


def geometry(x: Array) -> Json:
    """Fit-only centering prevents reserved examples from changing the predictor."""
    q = np.clip(x[:, 0], *CONFIG["clip_q"])
    z = np.log(q / (1 - q))
    return dict(
        scaler=sparse.fit_scaler(x),
        logit_center=float(z.mean()),
        logit_scale=float(max(z.std(), 1e-8)),
        knots=sparse.KNOTS,
    )


def design(arm: str, x: Array, geo: Json) -> Array:
    """Local residuals share the same nine public inputs as the scalar controls."""
    sx, _ = sparse.scale(x, geo["scaler"])
    q = np.clip(x[:, 0], *CONFIG["clip_q"])
    z = (np.log(q / (1 - q)) - geo["logit_center"]) / geo["logit_scale"]
    intercept = np.ones((len(x), 1))
    if arm == "intercept_only":
        return intercept
    linear = np.column_stack((intercept, z))
    return (
        np.column_stack((linear, sx[:, 1:]))
        if arm == "linear"
        else np.column_stack((linear, sparse.basis(sx)[:, :-1]))
    )


def objective(theta: Array, matrix: Array, y: Array, l2: float) -> tuple[float, Array, Array]:
    """Positive ridge curvature gives a unique optimum even with redundant bases."""
    z = matrix @ theta
    p = expit(z)
    loss = float(np.mean(np.logaddexp(0, z) - y * z) + l2 * (theta @ theta))
    gradient = matrix.T @ (p - y) / len(y) + 2 * l2 * theta
    hessian = matrix.T @ ((p * (1 - p))[:, None] * matrix) / len(y) + 2 * l2 * np.eye(len(theta))
    return loss, np.asarray(gradient), np.asarray(hessian)


def optimize(
    matrix: Array,
    y: Array,
    l2: float,
    seed: int,
    *,
    fixed_steps: int | None = None,
    deadline: float = float("inf"),
    bounds: Array | None = None,
    initial: Array | None = None,
) -> Json:
    """Backtracking uses the stated objective; fixed descent uses identical inputs."""
    began = time.monotonic()
    theta = np.zeros(matrix.shape[1]) if initial is None else initial.copy()
    if initial is None:
        theta[0] = np.log(
            np.clip(y.mean(), 1e-4, 1 - 1e-4) / (1 - np.clip(y.mean(), 1e-4, 1 - 1e-4))
        )
        theta[1:] = np.random.default_rng(seed).normal(0, 1e-5, len(theta) - 1)
    initial_parameters = theta.tolist()
    epochs: list[Json] = []
    evaluations = 0
    steps = CONFIG["max_steps"] if fixed_steps is None else fixed_steps
    for step in range(steps + 1):
        if time.monotonic() >= deadline:
            raise ValueError("fit_budget")
        loss, gradient, hessian = objective(theta, matrix, y, l2)
        evaluations += 1
        projected = (
            gradient
            if bounds is None
            else theta - np.clip(theta - gradient, bounds[:, 0], bounds[:, 1])
        )
        norm = float(np.max(np.abs(projected)))
        epochs.append(dict(epoch=step, loss=loss, gradient_norm=norm))
        print(
            f"[exp8008] epoch={step} seed={seed} loss={loss:.12g} gradient={norm:.6g} "
            f"elapsed_s={time.monotonic() - BEGAN:.3f} pending_steps={steps - step}",
            flush=True,
        )
        if step == steps or (fixed_steps is None and norm < CONFIG["gradient_tolerance"]):
            break
        direction = (
            -0.01 * gradient if fixed_steps is not None else -np.linalg.solve(hessian, gradient)
        )
        if bounds is not None:
            free = ~(
                ((theta <= bounds[:, 0]) & (gradient > 0))
                | ((theta >= bounds[:, 1]) & (gradient < 0))
            )
            direction = np.zeros(len(theta))
            direction[free] = -np.linalg.solve(hessian[np.ix_(free, free)], gradient[free])
        rate = 1.0
        for _ in range(32):
            candidate = theta + rate * direction
            if bounds is not None:
                candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            trial = objective(candidate, matrix, y, l2)[0]
            evaluations += 1
            if trial <= loss + 1e-4 * float(gradient @ (candidate - theta)) + 1e-14:
                break
            rate *= 0.5
        theta = candidate
    return dict(
        parameters=theta.tolist(),
        initial_parameters=initial_parameters,
        duration_s=time.monotonic() - began,
        seed=seed,
        l2=l2,
        epochs=epochs,
        initial_loss=epochs[0]["loss"],
        final_loss=loss,
        converged=norm < 1e-8,
        gradient_norm=norm,
        optimizer_steps=step,
        objective_evaluations=evaluations,
        hessian_evaluations=evaluations,
        optimizer_state=dict(last_gradient=gradient.tolist(), last_rate=rate if step else 0.0),
        parameter_count=len(theta),
    )


def probabilities(head: Json, x: Array) -> Array:
    """Stable two-label normalization has an exact sigmoid form without another fit."""
    if head["arm"] == "frozen_v693":
        return sparse.predict(head["historical"], x)
    if head["arm"] == "isotonic":
        return np.asarray(np.interp(x[:, 0], head["x"], head["p"]))
    z = design(head["arm"], x, head["geometry"]) @ np.asarray(head["parameters"])
    return np.asarray(expit(z))


def audit(theta: Array, x: Array, y: int, geo: Json, l2: float) -> Json:
    """Finite differences check the loss; local writes check complete ridge decay."""
    matrix = design("conditioned_energy", x[None, :], geo)
    _, gradient, hessian = objective(theta, matrix, np.array([y], dtype=float), l2)
    numeric = []
    for i in range(len(theta)):
        delta = np.eye(1, len(theta), i)[0] * 1e-6
        numeric.append(
            (
                objective(theta + delta, matrix, np.array([y]), l2)[0]
                - objective(theta - delta, matrix, np.array([y]), l2)[0]
            )
            / 2e-6
        )
    ids = np.flatnonzero(matrix[0])
    sparse_update = theta * (1 - 0.02 * l2)
    sparse_update[ids] -= 0.01 * matrix[0, ids] * (expit(matrix[0] @ theta) - y)
    dense_update = theta - 0.01 * gradient
    error = float(np.max(np.abs(np.asarray(numeric) - gradient)))
    update_error = float(np.max(np.abs(sparse_update - dense_update)))
    z = matrix @ theta
    energies = np.column_stack((np.zeros(len(z)), -z))
    weights = np.exp(-energies - np.max(-energies, axis=1, keepdims=True))
    parity = float(np.max(np.abs(weights[:, 1] / weights.sum(axis=1) - expit(z))))
    endpoints = np.repeat(x[None, :], 2, axis=0)
    endpoints[:, 0] = [0.0, 1.0]
    finite_endpoints = bool(
        np.isfinite(expit(design("conditioned_energy", endpoints, geo) @ theta)).all()
    )
    curvature = float(np.linalg.eigvalsh(hessian).min())
    return dict(
        finite_difference_error=error,
        dense_sparse_update_error=update_error,
        sigmoid_error=parity,
        active_coefficients=len(ids),
        global_decay_coefficients=len(theta),
        updated_parameters=sparse_update.tolist(),
        stable_q_endpoints=finite_endpoints,
        minimum_penalized_curvature=curvature,
        passed=error < 1e-7
        and update_error < 1e-10
        and parity < 1e-15
        and len(ids) <= 38
        and finite_endpoints
        and curvature > 0,
    )


def train(data: Json, historical: Json, raw: Path) -> Json:
    """Fit and tune roles alone select base coefficients, with a shared wall budget."""
    sparse.validate_data(data)
    f, t = sparse.usable(data["fit"]), sparse.usable(data["tune"])
    x, tx = sparse.inputs(f), sparse.inputs(t)
    y, ty = (np.asarray([r["y"] for r in rows], dtype=float) for rows in (f, t))
    geo = geometry(x)
    deadline = time.monotonic() + CONFIG["fit_budget_s"]
    heads, trials, checks, work = [], [], [], []
    for arm in CONFIG["arms"][:4]:
        for seed in CONFIG["seeds"]:
            progress("benchmark_before_" + arm, seed)
            candidates = []
            for l2 in CONFIG["penalties"]:
                progress(f"benchmark_before_{arm}_seed_{seed}_l2_{l2}")
                matrix = design(arm, x, geo)
                initial = None
                warm = None
                if arm in ("conditioned_energy", "equal_information_spline"):
                    warm = optimize(matrix[:, :2], y, l2, seed, deadline=deadline)
                    initial = np.random.default_rng(seed).normal(0, 1e-5, matrix.shape[1])
                    initial[:2] = warm["parameters"]
                    work.append(
                        dict(
                            arm=arm + "_linear_initialization",
                            seed=seed,
                            l2=l2,
                            steps=warm["optimizer_steps"],
                            objective_evaluations=warm["objective_evaluations"],
                            hessian_evaluations=warm["hessian_evaluations"],
                            duration_s=warm["duration_s"],
                        )
                    )
                head = optimize(matrix, y, l2, seed, deadline=deadline, initial=initial)
                if warm is not None:
                    head["linear_initialization"] = warm
                head.update(arm=arm, geometry=geo)
                head["tune_loss"] = scalar.bce(probabilities(head, tx), ty)
                trial_path = raw / "trials" / f"{arm}-{seed}-{l2}.json"
                atomic_json(trial_path, head)
                candidates.append(head)
                trials.append(
                    dict(
                        arm=arm,
                        seed=seed,
                        l2=l2,
                        converged=head["converged"],
                        tune_loss=head["tune_loss"],
                        final_loss=head["final_loss"],
                        gradient_norm=head["gradient_norm"],
                        checkpoint=reference(trial_path),
                    )
                )
                progress(f"benchmark_after_{arm}_seed_{seed}_l2_{l2}")
                work.append(
                    dict(
                        arm=arm,
                        seed=seed,
                        l2=l2,
                        steps=head["optimizer_steps"],
                        objective_evaluations=head["objective_evaluations"],
                        hessian_evaluations=head["hessian_evaluations"],
                        duration_s=head["duration_s"],
                    )
                )
            selected = min(candidates, key=lambda h: h["tune_loss"])
            heads.append(selected)
            if arm == "conditioned_energy":
                checks.append(
                    dict(
                        seed=seed,
                        **audit(np.asarray(selected["parameters"]), x[0], 1, geo, selected["l2"]),
                    )
                )
                fixed = optimize(
                    design(arm, x, geo),
                    y,
                    selected["l2"],
                    seed,
                    fixed_steps=selected["optimizer_steps"],
                    deadline=deadline,
                    initial=np.asarray(selected["initial_parameters"]),
                )
                fixed.update(arm="fixed_step", geometry=geo, paired_arm=arm)
                heads.append(fixed)
                work.append(
                    dict(
                        arm="fixed_step",
                        seed=seed,
                        l2=selected["l2"],
                        steps=fixed["optimizer_steps"],
                        objective_evaluations=fixed["objective_evaluations"],
                        hessian_evaluations=fixed["hessian_evaluations"],
                        duration_s=fixed["duration_s"],
                    )
                )
            progress("benchmark_after_" + arm, seed)
    progress("benchmark_before_isotonic")
    iso = IsotonicRegression(out_of_bounds="clip").fit(x[:, 0], y)
    for seed in CONFIG["seeds"]:
        heads += [
            dict(
                arm="isotonic",
                seed=seed,
                x=iso.X_thresholds_.tolist(),
                p=iso.y_thresholds_.tolist(),
                parameter_count=2 * len(iso.X_thresholds_),
            ),
            dict(
                arm="frozen_v693",
                seed=seed,
                historical=next(h for h in historical["spline"] if h["seed"] == seed)
                if len(historical["spline"]) > 1
                else historical["spline"][0],
                parameter_count=109,
            ),
        ]
    progress("benchmark_after_isotonic")
    checkpoints = []
    for head in heads:
        path = raw / "heads" / f"{head['arm']}-{head['seed']}.json"
        atomic_json(path, head)
        checkpoints.append(reference(path))
    return dict(
        heads=heads,
        checkpoints=checkpoints,
        geometry=geo,
        convergence_rows=trials,
        gradient_checks=checks,
        optimizer_work=work,
        role_hashes={role: canonical_hash(rows) for role, rows in data.items()},
        converged=all(r["converged"] for r in trials),
        fit_duration_s=CONFIG["fit_budget_s"] - (deadline - time.monotonic()),
    )


def calibrate(fitted: Json, rows: list[Json], raw: Path) -> Json:
    """A bounded post-hoc affine map uses calibration groups after base freezing."""
    eligible = sparse.usable(rows)
    if len({r["source_cluster_id"] for r in eligible}) != CONFIG["calibration_groups"]:
        raise ValueError("calibration_support")
    x = sparse.inputs(eligible)
    y = np.asarray([r["y"] for r in eligible], dtype=float)
    maps = {}
    deadline = time.monotonic() + CONFIG["fit_budget_s"] - fitted["fit_duration_s"]
    for head in fitted["heads"]:
        progress("calibration_before_" + head["arm"], head["seed"])
        p = np.clip(probabilities(head, x), 1e-12, 1 - 1e-12)
        matrix = np.column_stack((np.ones(len(x)), np.log(p / (1 - p))))
        fit = optimize(
            matrix,
            y,
            CONFIG["calibration_l2"],
            head["seed"],
            bounds=np.asarray(CONFIG["calibration_bounds"]),
            initial=np.array([0.0, 1.0]),
            deadline=deadline,
        )
        maps[f"{head['arm']}-{head['seed']}"] = fit
        progress("calibration_after_" + head["arm"], head["seed"])
    value = dict(
        maps=maps,
        independent=len(eligible),
        role_hash=canonical_hash(rows),
        bounds=CONFIG["calibration_bounds"],
        frozen=True,
    )
    atomic_json(raw / "calibration.json", value)
    return value


def load_bundle(root: Path, raw: Path) -> tuple[Json, list[Json]]:
    """Copy checked prior bytes so published references never depend on test paths."""
    path = root / "results" / "experiment_8007_v694_conditioning_diagnosis.json"
    value = json.loads(path.read_text()) if path.is_file() else {}
    if value and "methods_ready_score" not in value:
        raise ValueError("upstream_contract:methods_ready_score")
    receipt = reader_receipt(
        value.get("task_id", UPSTREAM_TASK), root / "results", field="methods_ready_score"
    )
    check = dict(
        upstream_id="exp8007",
        path=str(path),
        hash=sha256_file(path) if path.is_file() else None,
        artifact_field="methods_ready_score",
        expected=1,
        observed=value.get("methods_ready_score"),
        passed=value.get("methods_ready_score") == 1 and receipt["passed"],
    )
    if not check["passed"]:
        return dict(references=[]), [check]
    prior_ref = dict(
        prior.copy_evidence(reference(path), raw),
        imported_fields=["methods_ready_score", "checkpoints", "role_manifests"],
    )
    bundle = json.loads(checked(value["checkpoints"]["bundle"]).read_text())

    def relocate(node: Any) -> Any:
        if isinstance(node, dict):
            if "path" in node and "sha256" in node:
                return dict(node, **prior.copy_evidence(node, raw))
            return {k: relocate(v) for k, v in node.items()}
        if isinstance(node, list):
            return [relocate(v) for v in node]
        return node

    bundle = relocate(bundle)
    bundle["references"].append(prior_ref)
    bundle["upstream_checks"] = [check]
    return bundle, []


def development(bundle: Json, role: str) -> list[Json]:
    """Open each reserved role only after its owning checkpoint has been frozen."""
    if "development" in bundle:
        return list(bundle["development"][role])
    manifests = bundle["role_manifests"]
    public = json.loads(checked(manifests["public"][role]).read_text())["features"]
    labels = json.loads(checked(manifests["evaluator"][role]).read_text())["rows"]
    features = {r["family_id"]: r for r in public}
    targets = {r["family_id"]: r for r in labels}
    return [
        dict(
            family_id=s["family_id"],
            source_cluster_id=s["source_cluster_id"],
            q=s["q"],
            features=features[s["family_id"]]["values"],
            y=targets[s["family_id"]].get("y"),
            status=s["status"],
        )
        for s in bundle["slots"]
        if s["role"] == role
    ]


def reduce_rows(fitted: Json, calibration: Json, bundle: Json) -> Json:
    """Primitive probabilities and losses reduce without optimizing typed thresholds."""
    rows, extrapolation = [], {}
    roles = dict(
        bundle["data"],
        **{role: development(bundle, role) for role in ("calibration", "stream", "retention")},
    )
    for role, roster in roles.items():
        eligible = sparse.usable(roster)
        x = sparse.inputs(eligible)
        extrapolation[role] = sparse.scale(x, fitted["geometry"]["scaler"])[1]
        for head in fitted["heads"]:
            progress("benchmark_before_" + role + "_" + head["arm"], len(roster))
            ps = probabilities(head, x)
            logits = np.log(np.clip(ps, 1e-12, 1 - 1e-12) / (1 - np.clip(ps, 1e-12, 1 - 1e-12)))
            mapping = calibration["maps"].get(f"{head['arm']}-{head['seed']}")
            calibrated = (
                expit(mapping["parameters"][0] + mapping["parameters"][1] * logits)
                if mapping
                else np.full(len(ps), np.nan)
            )
            lookup = {
                r["family_id"]: (float(p), float(cp) if np.isfinite(cp) else None)
                for r, p, cp in zip(eligible, ps, calibrated, strict=True)
            }
            for r in roster:
                pair = lookup.get(r["family_id"])
                for condition in ("base", "posthoc"):
                    p = pair[int(condition == "posthoc")] if pair else None
                    bce = None if p is None else scalar.bce(np.asarray([p]), np.asarray([r["y"]]))
                    rows.append(
                        dict(
                            family_id=r["family_id"],
                            source_cluster_id=r["source_cluster_id"],
                            role=role,
                            arm=head["arm"],
                            seed=head["seed"],
                            condition=condition,
                            metric="binary_log_loss",
                            p=p,
                            y=r["y"],
                            numerator=bce if bce is not None else 0.0,
                            denominator=int(p is not None),
                            brier=None if p is None else (p - r["y"]) ** 2,
                            status="completed"
                            if p is not None
                            else (
                                "blocked" if condition == "posthoc" and not mapping else "excluded"
                            ),
                            exclusion_reason=None
                            if p is not None
                            else (
                                "calibration_support_gate"
                                if condition == "posthoc" and not mapping
                                else "missing_q_features_or_target"
                            ),
                            censor_reason=None,
                        )
                    )
            progress("benchmark_after_" + role + "_" + head["arm"], len(roster))
    original = [r for roster in roles.values() for r in roster]
    eligible = sparse.usable(original)
    summaries = []
    for role in roles:
        for arm in CONFIG["arms"]:
            for condition in ("base", "posthoc"):
                group = [
                    r
                    for r in rows
                    if r["role"] == role
                    and r["arm"] == arm
                    and r["condition"] == condition
                    and r["denominator"]
                ]
                summaries.append(
                    dict(
                        role=role,
                        arm=arm,
                        condition=condition,
                        log_loss=sum(r["numerator"] for r in group) / len(group) if group else None,
                        brier=sum(r["brier"] for r in group) / len(group) if group else None,
                        independent=len({r["source_cluster_id"] for r in group}),
                        descriptive=True,
                    )
                )
    return dict(
        rows=rows,
        arm_metrics=summaries,
        extrapolation_counts=extrapolation,
        decision_benefit="unassessed",
        sample_size_budget=dict(
            intended=len(original),
            eligible=len(eligible),
            started=len(original),
            completed=len(eligible),
            excluded=len(original) - len(eligible),
            failed=0,
            censored=0,
            independent=len({r["source_cluster_id"] for r in eligible}),
            seeds_are_independent=False,
        ),
    )


def replay(path: Path) -> Json:
    """Cold loading and primitive reduction detect changed checkpoints and aggregates."""
    value = json.loads(path.read_text())
    for ref in value["raw_shard_hashes"] + value["code_config_hashes"]:
        checked(ref)
    if value["checkpoints"]:
        bundle = json.loads(checked(value["checkpoints"]["bundle"]).read_text())
        fitted = json.loads(checked(value["checkpoints"]["fitted"]).read_text())
        for row in fitted["convergence_rows"]:
            checked(row["checkpoint"])
        for head, ref in zip(fitted["heads"], fitted["checkpoints"], strict=True):
            if head != json.loads(checked(ref).read_text()):
                raise ValueError("checkpoint_state")
        probe = sparse.inputs(sparse.usable(bundle["data"]["fit"]))[0]
        audits = [
            dict(
                seed=h["seed"],
                **audit(np.asarray(h["parameters"]), probe, 1, h["geometry"], h["l2"]),
            )
            for h in fitted["heads"]
            if h["arm"] == "conditioned_energy"
        ]
        if value["gradient_checks"] != audits:
            raise ValueError("roundtrip_audits")
        calibration = json.loads(checked(value["calibration_checkpoint"]).read_text())
        expected = reduce_rows(fitted, calibration, bundle)
        for key, result in expected.items():
            if value[key] != result:
                raise ValueError("reduction:" + key)
    if value["conditioned_fit_ready_score"] and (
        value["verdict_class"] in ("blocked", "disqualified")
        or not value["validation_receipts"]
        or not all(r["passed"] for r in value["validation_receipts"])
    ):
        raise ValueError("unsafe_readiness")
    return dict(passed=True, sha256=sha256_file(path))


def validation_plan(scratch: Path) -> list[CommandSpec]:
    """Existing owned checks stay separate from a single bounded repository diagnostic."""
    commands = prior.validation_plan(scratch)
    return [
        replace(
            c, argv=tuple(a.replace(prior.NAME, NAME).replace(prior.TEST, TEST) for a in c.argv)
        )
        for c in commands
    ]


def terminal(path: Path) -> Json:
    """Fresh CLI and existing terminal readers check the exact publication candidate."""
    commands = [
        CommandSpec(
            "cold_reduction",
            (
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / OWNED[-1]),
                "--cold-replay",
                str(path),
            ),
            "terminal",
            120,
        ),
        CommandSpec(
            "adversarial",
            (str(ROOT / ".venv/bin/python"), "scripts/adversarial_verify.py", str(path), "--json"),
            "terminal",
            120,
        ),
        CommandSpec(
            "strict_rows",
            (
                str(ROOT / ".venv/bin/python"),
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(path),
            ),
            "terminal",
            120,
        ),
    ]
    receipts = run_commands(
        ROOT,
        commands,
        log_dir=path.parent / "terminal_logs" / sha256_file(path).split(":")[-1],
        heartbeat_s=30,
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Fit once, retain real checks and publish only the validated final bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261002"], default="20261002")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    began = time.monotonic()
    progress("begin_no_pretrained_model_load_or_generation")
    try:
        if args.cold_replay:
            replay(args.cold_replay)
            progress("cold_reduction_passed")
            return 0
        output = args.output.absolute()
        raw = output.parent / "raw" / output.stem
        scratch = Path(tempfile.mkdtemp(prefix="carnot-8008-"))
        commands = validation_plan(scratch)
        code = [reference(ROOT / p) for p in OWNED + [TEST]]
        atomic_json(
            raw / "configuration.json",
            dict(config=CONFIG, code=code, commands=[asdict(c) for c in commands]),
        )
        bundle, failures = (
            (json.loads(args.fixture_input.read_text()), [])
            if args.fixture_input
            else load_bundle(args.root, raw)
        )
        atomic_json(
            raw / "role_freeze.json",
            dict(
                config=CONFIG,
                manifests=bundle.get("role_manifests"),
                historical_hashes={
                    role: canonical_hash(rows) for role, rows in bundle.get("data", {}).items()
                },
            ),
        )
        progress("roles_methods_budgets_gates_frozen")
        frozen = time.monotonic()
        value = prior.base(failures)
        value.update(
            experiment_id=8008,
            task_id=TASK,
            schema="carnot.v694.conditioned_energy_fit.v1",
            honest_verdict="complete_blocked_conditioned_energy_fit"
            if failures
            else "complete_null_conditioned_energy_fit",
            claim_scope="Optimizer conditioning on exposed cached development. Adding bias is established practice. Decision benefit remains unassessed.",
            conditioned_fit_ready_score=0,
            conditioned_head_checkpoints=[],
            baseline_checkpoints=[],
            convergence_rows=[],
            gradient_checks=[],
            sigmoid_equivalence={},
            calibration_checkpoint={},
            optimizer_work=[],
            active_coefficients=[],
            cited_upstream_artifacts=bundle["references"],
            statistical_plan=CONFIG,
            methods_ready_score=0,
        )
        if not failures:
            fitted = train(bundle["data"], bundle["heads"], raw)
            atomic_json(raw / "fitted.json", fitted)
            progress("base_heads_saved_before_calibration_labels")
            calibration_rows = development(bundle, "calibration")
            public_eligible = [
                r for r in calibration_rows if r["q"] is not None and r["features"] is not None
            ]
            known_groups = len({r["source_cluster_id"] for r in sparse.usable(calibration_rows)})
            value["calibration_support_summary"] = dict(
                required_groups=CONFIG["calibration_groups"],
                public_input_groups=len(public_eligible),
                known_target_groups=known_groups,
                unknown_target_groups=len(public_eligible) - known_groups,
            )
            atomic_json(
                raw / "calibration_input.json",
                dict(rows=calibration_rows, eligible_known_target_groups=known_groups),
            )
            if known_groups != CONFIG["calibration_groups"]:
                ref = (
                    bundle["role_manifests"]
                    .get("evaluator", {})
                    .get("calibration", reference(raw / "calibration_input.json"))
                )
                manifest = json.loads(checked(ref).read_text())["rows"]
                failures = [
                    dict(
                        upstream_id="exp7994",
                        path=ref["path"],
                        hash=ref["sha256"],
                        artifact_field=f"rows[{i}].y",
                        expected="binary target 0 or 1 for all 62 public-input calibration groups",
                        observed=r["y"],
                        passed=False,
                        family_id=r["family_id"],
                        exclusion_reason=r.get("exclusion_reason", "unknown_target"),
                    )
                    for i, r in enumerate(manifest)
                    if r["family_id"] in {p["family_id"] for p in public_eligible}
                    and r["y"] is None
                ]
                value.update(
                    gate_check_summary=failures,
                    verdict_class="blocked",
                    honest_verdict="complete_blocked_calibration_target_support",
                )
                calibration = dict(
                    maps={},
                    independent=known_groups,
                    frozen=True,
                    status="blocked",
                    required_groups=CONFIG["calibration_groups"],
                    role_hash=canonical_hash(calibration_rows),
                    bounds=CONFIG["calibration_bounds"],
                    reason="insufficient_known_calibration_targets",
                )
                atomic_json(raw / "calibration.json", calibration)
                progress(
                    "calibration_blocked_checkpoint_frozen",
                    known_groups,
                    CONFIG["calibration_groups"] - known_groups,
                )
            else:
                calibration = calibrate(fitted, calibration_rows, raw)
            progress("calibration_frozen_before_stream_retention_labels")
            atomic_json(raw / "bundle.json", bundle)
            measured = reduce_rows(fitted, calibration, bundle)
            atomic_json(raw / "measurement.json", measured)
            value.update(
                measured,
                inference_substrate="verifier_ensemble_against_cached_candidates",
                checkpoints=dict(
                    bundle=reference(raw / "bundle.json"),
                    fitted=reference(raw / "fitted.json"),
                    measurement=reference(raw / "measurement.json"),
                ),
                calibration_checkpoint=reference(raw / "calibration.json"),
                conditioned_head_checkpoints=[
                    r
                    for h, r in zip(fitted["heads"], fitted["checkpoints"], strict=True)
                    if h["arm"] == "conditioned_energy"
                ],
                baseline_checkpoints=[
                    r
                    for h, r in zip(fitted["heads"], fitted["checkpoints"], strict=True)
                    if h["arm"] != "conditioned_energy"
                ],
                convergence_rows=fitted["convergence_rows"],
                gradient_checks=fitted["gradient_checks"],
                sigmoid_equivalence=dict(
                    exact_reparameterization="p=expit(b+a*centered_logit(q)+sum_j spline_j(x_j))",
                    additional_fit_steps=0,
                    max_error=max(r["sigmoid_error"] for r in fitted["gradient_checks"]),
                ),
                active_coefficients=[r["active_coefficients"] for r in fitted["gradient_checks"]],
                optimizer_work=fitted["optimizer_work"]
                + [
                    dict(
                        arm="posthoc_calibration",
                        head=key,
                        steps=h["optimizer_steps"],
                        objective_evaluations=h["objective_evaluations"],
                        hessian_evaluations=h["hessian_evaluations"],
                        duration_s=h["duration_s"],
                    )
                    for key, h in calibration["maps"].items()
                ],
                trained_head_specs=[
                    dict(
                        arm=h["arm"], seed=h["seed"], parameters=h["parameter_count"], device="cpu"
                    )
                    for h in fitted["heads"]
                    if h["arm"] != "frozen_v693"
                ],
                role_manifests=bundle["role_manifests"],
                role_hashes=fitted["role_hashes"],
                genuine_headroom=dict(
                    optimization_gain=[
                        dict(
                            seed=h["seed"],
                            initial_loss=h["initial_loss"],
                            final_loss=h["final_loss"],
                            gain=h["initial_loss"] - h["final_loss"],
                        )
                        for h in fitted["heads"]
                        if h["arm"] == "conditioned_energy"
                    ],
                    decision_benefit="unassessed",
                ),
                positive_control_results=dict(
                    scope="circular_test_fixture_only", natural_benefit_credit=False
                ),
                acceptance_gate_results=dict(
                    measurement=fitted["converged"]
                    and all(r["passed"] for r in fitted["gradient_checks"])
                    and bool(calibration["maps"])
                    and all(h["converged"] for h in calibration["maps"].values()),
                    natural_benefit=False,
                ),
            )
            if args.fixture_input and not failures:
                value.update(
                    verdict_class="circular_positive",
                    honest_verdict="complete_circular_positive_conditioned_fit_fixture",
                )
        measured_at = time.monotonic()
        if not args.fixture_input and value["checkpoints"]:
            receipts = run_commands(
                ROOT,
                commands,
                log_dir=raw / "validation_logs",
                heartbeat_s=30,
                extra_env=dict(
                    PYTHONUNBUFFERED="1",
                    JAX_PLATFORMS="cpu",
                    OPENBLAS_NUM_THREADS="1",
                    CARNOT_8008_COVERAGE_FILE=str(scratch / ".coverage"),
                    COVERAGE_FILE=str(scratch / ".coverage-health"),
                ),
            )
            value["validation_receipts"] = [r for r in receipts if r["scope"] == "owned"]
            value["repository_health"] = [r for r in receipts if r["scope"] == "repository_health"]
            coverage = (
                json.loads((scratch / "coverage.json").read_text())
                if (scratch / "coverage.json").exists()
                else dict(files={})
            )
            value["coverage_statement_counts"] = {
                k: v["summary"] for k, v in coverage["files"].items()
            }
            passed = bool(value["coverage_statement_counts"]) and all(
                v["missing_lines"] == 0 and v["num_statements"] > 0
                for v in value["coverage_statement_counts"].values()
            )
            value["conditioned_fit_ready_score"] = int(
                passed
                and all(r["passed"] for r in value["validation_receipts"])
                and value["acceptance_gate_results"]["measurement"]
            )
            if not value["conditioned_fit_ready_score"] and not failures:
                value.update(
                    verdict_class="disqualified",
                    honest_verdict="complete_disqualified_conditioned_energy_fit",
                )
        ended = time.monotonic()
        value.update(
            duration_s=ended - began,
            phase_spans=[
                dict(phase="freeze", duration_s=frozen - began),
                dict(phase="fit_calibrate_measure", duration_s=measured_at - frozen),
                dict(phase="validation", duration_s=ended - measured_at),
            ],
            code_config_hashes=code,
            terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        )
        value["raw_shard_hashes"] = (
            list(value["checkpoints"].values())
            + value["conditioned_head_checkpoints"]
            + value["baseline_checkpoints"]
            + value["cited_upstream_artifacts"]
            + [reference(raw / "configuration.json"), reference(raw / "role_freeze.json")]
            + [reference(p) for p in sorted((raw / "failure_logs").glob("*.log"))]
        )
        if value["calibration_checkpoint"]:
            value["raw_shard_hashes"].append(value["calibration_checkpoint"])
            value["raw_shard_hashes"].append(reference(raw / "calibration_input.json"))
        value["reproducibility_checksum"] = canonical_hash(
            dict(config=CONFIG, code=code, raw=value["raw_shard_hashes"])
        )
        value["field_principles"] = {
            k: "Bind this invocation to durable checked bytes. Mechanism readiness and circular fixtures give no natural decision benefit credit."
            for k in value
        }
        progress("publish_before")
        receipt = publish_primary(output, value, replay if args.fixture_input else terminal)
        atomic_json(raw / "terminal_validation.json", receipt)
        reader = reader_receipt(
            TASK,
            output.parent,
            field="conditioned_fit_ready_score",
            expected=value["conditioned_fit_ready_score"],
        )
        atomic_json(raw / "reader_receipt.json", reader)
        if not reader["passed"] or reader["gate_sha256"] != sha256_file(output):
            raise ValueError("primary_reader")
        replay(output)
        progress("publish_after")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"[exp8008] failed={error}", flush=True)
        return 1
