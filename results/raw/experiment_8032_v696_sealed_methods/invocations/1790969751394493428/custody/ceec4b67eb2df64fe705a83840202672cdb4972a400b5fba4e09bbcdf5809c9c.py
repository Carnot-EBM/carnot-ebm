"""REQ-REPORT-8020: fit CPU heads on qualified labels before sealed calibration.

This experiment prepares replayable probabilities on exposed development data.
It measures numerical readiness; correctness and learning benefit stay untested.
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

from carnot import experiment_8008_v694_conditioned_energy_fit as old
from carnot import experiment_8019_v695_eligible_targets as eligible
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import sparse_energy_7996 as sparse
from carnot.verify import qwen_energy_calibration_7972 as scalar

Json = dict[str, Any]
Array = old.Array
ROOT = old.ROOT
NAME = "experiment_8020_v695_qualified_energy_fit"
TASK = "exp8020-qualified-energy-fit"
OWNED = [f"python/carnot/{NAME}.py", f"scripts/experiments/{NAME}.py"]
TEST = "tests/python/test_qualified_energy_8020.py"
ARMS = [
    "conditioned_energy",
    "sigmoid_identity",
    "linear",
    "scalar_affine",
    "intercept_only",
    "isotonic",
    "fixed_200_residual",
]
CONFIG = dict(
    old.CONFIG,
    arms=ARMS,
    calibration_groups_minimum=48,
    calibration_per_class=8,
    calibration_groups="Exp8019 sealed complete-label mask",
    fixed_control_steps=200,
    identity_independent=False,
    public_features=9,
    scalar_controls_use_only_q=True,
)
BEGAN = time.monotonic()


def progress(phase: str, units: int = 0, pending: int = 0) -> None:
    """Flush real phase counts so operators can distinguish work from a stall."""
    print(
        f"[exp8020] phase={phase} elapsed_s={time.monotonic() - BEGAN:.3f} completed={units} pending={pending}",
        flush=True,
    )


def load_sources(root: Path, raw: Path) -> tuple[Json, list[Json]]:
    """Gate exact contracts and copy role shards without opening reserved targets."""
    path = root / "results" / "experiment_8019_v695_eligible_targets.json"
    value = json.loads(path.read_text()) if path.is_file() else {}
    checks = [
        dict(
            upstream_id="exp8019",
            path=str(path),
            hash=sha256_file(path) if path.is_file() else None,
            artifact_field=field,
            expected=1,
            observed=value.get(field, "MISSING_CONTRACT_FIELD"),
            passed=value.get(field) == 1,
        )
        for field in ("fit_targets_ready_score", "calibration_targets_ready_score")
    ]
    if not all(r["passed"] for r in checks):
        return dict(references=[]), [r for r in checks if not r["passed"]]
    try:
        copy = old.prior.copy_evidence
        source = dict(
            public_manifests={r: copy(value["public_manifests"][r], raw) for r in eligible.SLOTS},
            evaluator_manifests={
                r: copy(value["evaluator_manifests"][r], raw)
                for r in ("fit", "tune", "calibration")
            },
            support_by_role=value["support_by_role"],
            exclusion_manifest=copy(value["exclusion_manifest"], raw),
            references=[copy(reference(path), raw)],
            gate_checks=checks,
        )
        seen: set[str] = set()
        for role in eligible.SLOTS:
            rows = json.loads(checked(source["public_manifests"][role]).read_text())["rows"]
            groups = {r["source_cluster_id"] for r in rows}
            if seen & groups or any(
                r["role"] != role or "y" in r or "eligible_y" in r for r in rows
            ):
                raise ValueError("public_role_isolation")
            seen.update(groups)
        return source, []
    except (ValueError, KeyError, OSError) as error:
        return dict(references=[]), [
            dict(
                checks[0],
                artifact_field="immutable_manifests",
                expected="complete byte-bound isolated roles",
                observed=str(error),
                passed=False,
            )
        ]


def read_role(source: Json, role: str, *, labels: bool = True) -> list[Json]:
    """Join only allowed label roles; public predictions never open evaluator rows."""
    public = json.loads(checked(source["public_manifests"][role]).read_text())["rows"]
    targets = (
        {
            r["family_id"]: r
            for r in json.loads(checked(source["evaluator_manifests"][role]).read_text())["rows"]
        }
        if labels
        else {}
    )
    return [
        dict(
            family_id=r["family_id"],
            source_cluster_id=r["source_cluster_id"],
            q=r["q"] if r["public_eligible"] else None,
            features=r["features"] if r["public_eligible"] else None,
            status=r["status"],
            **(dict(y=targets[r["family_id"]]["eligible_y"]) if labels else {}),
        )
        for r in public
    ]


def matrix(arm: str, x: Array, geo: Json) -> Array:
    """All full information arms share nine inputs; scalar controls restrict to q."""
    if arm == "scalar_affine":
        return old.design("linear", x, geo)[:, :2]
    return old.design(arm if arm in {"linear", "intercept_only"} else "conditioned_energy", x, geo)


def predict(head: Json, x: Array) -> Array:
    """The sigmoid duplicate uses exactly the energy coefficients with zero fitting."""
    if head["arm"] == "isotonic":
        return np.asarray(np.interp(x[:, 0], head["x"], head["p"]))
    if head["arm"] == "fixed_200_residual":
        return sparse.predict(dict(head, arm="spline"), x)
    z = matrix(head["arm"], x, head["geometry"]) @ np.asarray(head["parameters"])
    if head["arm"] == "conditioned_energy":
        weights = np.exp(np.column_stack((np.zeros(len(z)), z)) - np.maximum(0, z)[:, None])
        return np.asarray(weights[:, 1] / weights.sum(axis=1))
    return np.asarray(expit(z))


def solve(
    arm: str,
    x: Array,
    y: Array,
    geo: Json,
    l2: float,
    seed: int,
    deadline: float,
) -> Json:
    """Reuse damped Newton and retain actual objective traces and curvature bounds."""
    m = matrix(arm, x, geo)
    progress("benchmark_before_" + arm, seed)
    warm = None
    initial = None
    if arm == "conditioned_energy":
        warm = old.optimize(m[:, :2], y, l2, seed, deadline=deadline)
        initial = np.random.default_rng(seed).normal(0, 1e-5, m.shape[1])
        initial[:2] = warm["parameters"]
    h = old.optimize(m, y, l2, seed, deadline=deadline, initial=initial)
    h["linear_initialization"] = warm
    h.update(arm=arm, geometry=geo)
    bound = float(1 + np.linalg.norm(m, 2) ** 2 / (8 * len(y) * l2))
    active = int(np.max(np.count_nonzero(m, axis=1)))
    for row in h["epochs"]:
        row.update(condition_upper_bound=bound, active_coefficient_count=active)
        if row["epoch"] % 50 == 0 or row == h["epochs"][-1]:
            progress(
                f"trace_{arm}_objective={row['loss']:.8g}_gradient={row['gradient_norm']:.8g}_condition_bound={bound:.8g}_active={active}",
                row["epoch"],
                2000 - row["epoch"],
            )
    h["finite"] = bool(np.isfinite(h["parameters"]).all())
    theta = np.asarray(h["parameters"])
    gradient = old.objective(theta, m, y, l2)[1]
    numeric = []
    for i in range(len(theta)):
        delta = np.eye(1, len(theta), i)[0] * 1e-6
        numeric.append(
            (old.objective(theta + delta, m, y, l2)[0] - old.objective(theta - delta, m, y, l2)[0])
            / 2e-6
        )
    h["finite_difference_error"] = float(np.max(np.abs(np.asarray(numeric) - gradient)))
    h["condition_estimate"] = float(
        np.linalg.cond(old.objective(np.asarray(h["parameters"]), m, y, l2)[2])
    )
    progress("benchmark_after_" + arm, h["optimizer_steps"])
    return h


def historical_control(x: Array, y: Array, tx: Array, ty: Array, seed: int) -> Json:
    """Retain the original residual objective, scaling and 200 fixed updates."""
    progress("benchmark_before_historical_fixed_200", seed)
    head = dict(
        arm="spline",
        seed=seed,
        parameters=[0.0] * 109,
        decay_scale=1.0,
        scaler=sparse.fit_scaler(x),
        temperature=1.0,
        parameter_count=109,
    )
    theta = np.asarray(head["parameters"])
    _, initial_jac = sparse.logits_jacobian(head, x)
    bound = float(1 + np.linalg.norm(initial_jac, 2) ** 2 / (0.008 * len(x)))
    active = int(np.max(np.count_nonzero(initial_jac, axis=1)))
    epochs = []
    for step in range(201):
        z, jac = sparse.logits_jacobian(head, x)
        gradient = jac.T @ (expit(z) - y) / len(x) + 0.002 * theta
        row = dict(
            epoch=step,
            loss=sparse.objective(head, x, y),
            gradient_norm=float(np.max(np.abs(gradient))),
            condition_upper_bound=bound,
            active_coefficient_count=active,
        )
        epochs.append(row)
        if step % 50 == 0:
            print(
                f"[exp8020] historical seed={seed} elapsed_s={time.monotonic() - BEGAN:.3f} completed={step} pending={200 - step} objective={row['loss']} gradient={row['gradient_norm']} condition_bound={row['condition_upper_bound']} active={row['active_coefficient_count']}",
                flush=True,
            )
        if step < 200:
            theta -= 0.01 * gradient
            head["parameters"] = theta.tolist()
    losses = {
        str(t): scalar.bce(expit(sparse.logits_jacobian(head, tx)[0] / t), ty)
        for t in sparse.CONFIG["temperatures"]
    }
    head.update(
        arm="fixed_200_residual",
        epochs=epochs,
        optimizer_steps=200,
        temperature=min(sparse.CONFIG["temperatures"], key=lambda t: losses[str(t)]),
        tune_temperature_losses=losses,
        converged=row["gradient_norm"] < 1e-8,
        gradient_norm=row["gradient_norm"],
        finite=bool(np.isfinite(theta).all()),
        original_configuration=sparse.CONFIG,
    )
    progress("benchmark_after_historical_fixed_200", 200)
    return head


def train(source: Json, raw: Path) -> Json:
    """Fit coefficients only on fit; select penalties only by tune log loss."""
    data = {r: read_role(source, r) for r in ("fit", "tune")}
    sparse.validate_data(data)
    f, t = (sparse.usable(data[r]) for r in ("fit", "tune"))
    x, tx = sparse.inputs(f), sparse.inputs(t)
    y, ty = (np.asarray([r["y"] for r in rows], dtype=float) for rows in (f, t))
    geo = old.geometry(x)
    deadline = time.monotonic() + CONFIG["fit_budget_s"]
    heads, trials, audits = [], [], []
    for arm in ("conditioned_energy", "linear", "scalar_affine", "intercept_only"):
        for seed in CONFIG["seeds"]:
            candidates = []
            for l2 in CONFIG["penalties"]:
                h = solve(arm, x, y, geo, l2, seed, deadline)
                h["tune_loss"] = scalar.bce(predict(h, tx), ty)
                ref = eligible.shard(raw, "trials", h)
                trials.append(
                    dict(
                        arm=arm,
                        seed=seed,
                        l2=l2,
                        converged=h["converged"],
                        gradient_norm=h["gradient_norm"],
                        checkpoint=ref,
                    )
                )
                candidates.append(h)
            selected = min(candidates, key=lambda h: h["tune_loss"])
            heads.append(selected)
            if arm == "conditioned_energy":
                audits.append(
                    dict(
                        seed=seed,
                        **old.audit(
                            np.asarray(selected["parameters"]), x[0], int(y[0]), geo, selected["l2"]
                        ),
                    )
                )
                heads.append(
                    dict(
                        selected,
                        arm="sigmoid_identity",
                        duplicate_of=f"conditioned_energy-{seed}",
                        additional_fit_steps=0,
                    )
                )
                heads.append(historical_control(x, y, tx, ty, seed))
    progress("benchmark_before_isotonic")
    iso = IsotonicRegression(out_of_bounds="clip").fit(x[:, 0], y)
    for seed in CONFIG["seeds"]:
        heads.append(
            dict(
                arm="isotonic",
                seed=seed,
                x=iso.X_thresholds_.tolist(),
                p=iso.y_thresholds_.tolist(),
                parameter_count=2 * len(iso.X_thresholds_),
                converged=True,
                finite=True,
            )
        )
    progress("benchmark_after_isotonic")
    checkpoints = [eligible.shard(raw, "heads", h) for h in heads]
    ready = all(
        h["finite"] and h["converged"] and h.get("finite_difference_error", 0) < 1e-7
        for h in heads
        if h["arm"] != "fixed_200_residual"
    ) and all(r["passed"] for r in audits)
    return dict(
        heads=heads,
        checkpoints=checkpoints,
        convergence_rows=trials,
        gradient_checks=audits,
        fit_role_hashes={r: canonical_hash(v) for r, v in data.items()},
        ready=ready,
        deadline_remaining_s=max(0.0, deadline - time.monotonic()),
    )


def calibrate(source: Json, fitted: Json, raw: Path) -> Json:
    """Open complete calibration labels after checking that every base is frozen."""
    for head, ref in zip(fitted["heads"], fitted["checkpoints"], strict=True):
        if head != json.loads(checked(ref).read_text()):
            raise ValueError("unfrozen_head")
    rows = read_role(source, "calibration")
    valid = sparse.usable(rows)
    support = scalar.support(valid, 48, 8)
    if not support["passed"]:
        raise ValueError("calibration_support")
    x, y = sparse.inputs(valid), np.asarray([r["y"] for r in valid], dtype=float)
    deadline = time.monotonic() + fitted["deadline_remaining_s"]
    maps = {}
    for head in fitted["heads"]:
        key = f"{head['arm']}-{head['seed']}"
        if head["arm"] == "sigmoid_identity":
            maps[key] = maps[f"conditioned_energy-{head['seed']}"]
            continue
        progress("calibration_benchmark_before_" + key)
        p = np.clip(predict(head, x), 1e-12, 1 - 1e-12)
        m = np.column_stack((np.ones(len(x)), np.log(p / (1 - p))))
        maps[key] = old.optimize(
            m,
            y,
            CONFIG["calibration_l2"],
            head["seed"],
            deadline=deadline,
            bounds=np.asarray(CONFIG["calibration_bounds"]),
            initial=np.array([0.0, 1.0]),
        )
        progress("calibration_benchmark_after_" + key, maps[key]["optimizer_steps"])
    value = dict(
        maps=maps,
        independent=support["independent"],
        support=support,
        role_hash=canonical_hash(rows),
        ready=all(h["converged"] and np.isfinite(h["parameters"]).all() for h in maps.values()),
        excluded=[r for r in rows if r not in valid],
        frozen=True,
    )
    atomic_json(raw / "calibration.json", value)
    return value


def measure(source: Json, fitted: Json, calibration: Json) -> Json:
    """Reduce primitive probabilities; reserved roles carry no outcome metrics."""
    primitive, summary, originals = [], [], []
    for role in eligible.SLOTS:
        labeled = role in {"fit", "tune", "calibration"}
        roster = read_role(source, role, labels=labeled)
        original_public = json.loads(checked(source["public_manifests"][role]).read_text())["rows"]
        exclusion_reasons = {r["family_id"]: r.get("exclusion_reason") for r in original_public}
        if labeled:
            original_targets = json.loads(checked(source["evaluator_manifests"][role]).read_text())[
                "rows"
            ]
            exclusion_reasons.update(
                {
                    r["family_id"]: r.get("eligibility_reason") or r.get("exclusion_reason")
                    for r in original_targets
                    if r.get("eligible_y") is None
                }
            )
        originals += [dict(r, role=role) for r in roster]
        usable = [r for r in roster if r["q"] is not None and r["features"] is not None]
        x = sparse.inputs(usable)
        for head in fitted["heads"]:
            progress("prediction_benchmark_before_" + role + "_" + head["arm"], head["seed"])
            base = predict(head, x)
            mapping = calibration["maps"][f"{head['arm']}-{head['seed']}"]["parameters"]
            p = np.clip(base, 1e-12, 1 - 1e-12)
            calibrated = expit(mapping[0] + mapping[1] * np.log(p / (1 - p)))
            for condition, probs in (("base", base), ("posthoc", calibrated)):
                lookup = {r["family_id"]: float(p) for r, p in zip(usable, probs, strict=True)}
                group = []
                for r in roster:
                    p = lookup.get(r["family_id"])
                    y = r.get("y")
                    loss = (
                        scalar.bce(np.array([p]), np.array([y]))
                        if p is not None and y is not None
                        else None
                    )
                    row = dict(
                        role=role,
                        family_id=r["family_id"],
                        source_cluster_id=r["source_cluster_id"],
                        arm=head["arm"],
                        seed=head["seed"],
                        condition=condition,
                        p=p,
                        y=y,
                        log_loss=loss,
                        brier=(p - y) ** 2 if loss is not None else None,
                        exclusion_reason=(
                            exclusion_reasons.get(r["family_id"]) or "missing_public_inputs"
                        )
                        if p is None
                        else (exclusion_reasons.get(r["family_id"]) or "unknown_target")
                        if labeled and y is None
                        else None,
                        failure_reason=None,
                        censor_reason=None,
                        independent_method=head["arm"] != "sigmoid_identity",
                    )
                    primitive.append(row)
                    group.append(row)
                measured = [r for r in group if r["log_loss"] is not None]
                predicted = [r for r in group if r["p"] is not None]
                summary.append(
                    dict(
                        role=role,
                        arm=head["arm"],
                        seed=head["seed"],
                        condition=condition,
                        metric="log_loss" if labeled else "public_prediction_coverage",
                        numerator=sum(r["log_loss"] for r in measured)
                        if labeled
                        else len(predicted),
                        denominator=len(measured) if labeled else len(group),
                        intended=len(group),
                        eligible=len(measured) if labeled else len(predicted),
                        started=len(group),
                        completed=len(predicted),
                        excluded=len(group) - len(predicted),
                        failed=0,
                        censored=sum(r["p"] is not None and r["y"] is None for r in group)
                        if labeled
                        else 0,
                        independent=len(measured) if labeled else len(predicted),
                        descriptive=True,
                    )
                )
            progress("prediction_benchmark_after_" + role + "_" + head["arm"], len(roster))
    valid = [r for r in originals if r["q"] is not None and r["features"] is not None]
    return dict(
        primitive_rows=primitive,
        rows=summary,
        sample_size_budget=dict(
            intended=len(originals),
            eligible=len(valid),
            started=len(originals),
            completed=len(valid),
            excluded=len(originals) - len(valid),
            failed=0,
            censored=sum(
                r.get("y") is None and r["role"] in {"fit", "tune", "calibration"} for r in valid
            ),
            independent=len({r["source_cluster_id"] for r in valid}),
            seeds_are_independent=False,
            unit="public_predictor_source_group; outcomes measured only for fit/tune/calibration",
        ),
        eligible_target_counts=source["support_by_role"],
        public_prediction_seals={
            r: canonical_hash([v for v in primitive if v["role"] == r])
            for r in ("stream", "retention")
        },
    )


def replay(path: Path) -> Json:
    """Recompute primitive predictions and summaries from durable sealed states."""
    v = json.loads(path.read_text())
    for ref in v["raw_shard_hashes"] + v["code_config_hashes"]:
        checked(ref)
    if v["checkpoint_references"]:
        source, fitted, calibration = [
            json.loads(checked(r).read_text()) for r in v["checkpoint_references"]
        ]
        for h, ref in zip(fitted["heads"], fitted["checkpoints"], strict=True):
            if h != json.loads(checked(ref).read_text()):
                raise ValueError("checkpoint_state")
        measured = measure(source, fitted, calibration)
        actual = json.loads(checked(v["measurement_checkpoint"]).read_text())
        if measured != actual:
            raise ValueError("primitive_reduction")
        for key in (
            "rows",
            "sample_size_budget",
            "public_prediction_seals",
            "eligible_target_counts",
        ):
            if v[key] != measured[key]:
                raise ValueError("reduction:" + key)
    if v["energy_fit_ready_score"] and (
        v["verdict_class"] in {"blocked", "disqualified", "circular_positive"}
        or not v["acceptance_gate_results"]["measurement"]
        or not v["validation_receipts"]
        or not all(r["passed"] for r in v["validation_receipts"])
    ):
        raise ValueError("unsafe_readiness")
    return dict(passed=True, sha256=sha256_file(path))


def validation_plan(scratch: Path) -> list[CommandSpec]:
    """Reuse the existing bounded validation manifest with this task's file scope."""
    commands = eligible.validation_plan(scratch)
    config = scratch / "coverage.ini"
    config.write_text(config.read_text().replace(eligible.NAME, NAME))
    return [
        replace(
            c,
            argv=tuple(a.replace(eligible.NAME, NAME).replace(eligible.TEST, TEST) for a in c.argv),
        )
        for c in commands
    ]


def terminal(path: Path) -> Json:
    """Run fresh reduction and both existing terminal readers on exact bytes."""
    py = str(ROOT / ".venv/bin/python")
    commands = [
        CommandSpec(
            "cold_reduction",
            (py, "-u", str(ROOT / OWNED[-1]), "--cold-replay", str(path)),
            "terminal",
            120,
        ),
        CommandSpec(
            "adversarial",
            (py, "scripts/adversarial_verify.py", str(path), "--json"),
            "terminal",
            120,
        ),
        CommandSpec(
            "strict_rows",
            (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)),
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
    """Freeze inputs, fit once, retain checks and atomically publish checked bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261002"], default="20261002")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture", action="store_true")
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
        scratch = Path(tempfile.mkdtemp(prefix="carnot-8020-"))
        commands = validation_plan(scratch)
        dependencies = [old.__file__, sparse.__file__, scalar.__file__]
        code = [old.prior.copy_evidence(reference(ROOT / p), raw) for p in OWNED + [TEST]]
        code += [old.prior.copy_evidence(reference(Path(p)), raw) for p in dependencies]
        source, failures = load_sources(args.root, raw)
        history_path = ROOT / "results" / (old.NAME + ".json")
        history = json.loads(history_path.read_text())
        historical_custody = [
            old.prior.copy_evidence(reference(history_path), raw),
            old.prior.copy_evidence(
                reference(history_path.parent / "raw" / old.NAME / "configuration.json"), raw
            ),
        ]
        historical_custody += [
            old.prior.copy_evidence(reference(ROOT / r["log_path"]), raw)
            for r in history["validation_receipts"] + history.get("repository_health", [])
            if not r["passed"]
        ]
        config_ref = eligible.shard(
            raw,
            "configuration",
            dict(
                config=CONFIG,
                original_config=old.CONFIG,
                code=code,
                commands=[asdict(c) for c in commands],
                sources=source,
            ),
        )
        v = old.prior.base(failures)
        v.update(
            experiment_id=8020,
            task_id=TASK,
            milestone="2026.10.695",
            schema="carnot.v695.qualified_energy_fit.v1",
            honest_verdict="complete_blocked_qualified_energy_inputs"
            if failures
            else "complete_null_qualified_energy_fit",
            energy_fit_ready_score=0,
            generalized_learning_benefit_score=0,
            claim_scope="Qualified fitting and post-hoc calibration on historically exposed cached development. Public stream and retention outcomes are unopened. No correctness, learning, deployment or architecture novelty claim.",
            inference_substrate="verifier_ensemble_against_cached_candidates",
            head_checkpoints=[],
            calibration_checkpoint={},
            immutable_input_hashes=[],
            convergence_rows=[],
            gradient_checks=[],
            public_prediction_seals={},
            arm_controls=ARMS,
            eligible_target_counts={},
            checkpoint_references=[],
            measurement_checkpoint={},
            code_config_hashes=code,
            cited_upstream_artifacts=source["references"],
            positive_control_results=dict(
                scope="private_artificial_CPU_fixture", natural_benefit_credit=False
            ),
            genuine_headroom=dict(
                under_convergence="hypothesis tested by registered gradient criterion",
                correctness_win="unassessed",
            ),
        )
        progress("methods_roles_exclusions_budgets_gates_frozen")
        frozen = time.monotonic()
        preflight = []
        if not args.fixture and not failures:
            preflight = run_commands(
                ROOT,
                commands[:1],
                log_dir=raw / "preflight_logs",
                heartbeat_s=30,
                extra_env=dict(
                    PYTHONUNBUFFERED="1",
                    JAX_PLATFORMS="cpu",
                    OPENBLAS_NUM_THREADS="1",
                    CARNOT_8020_COVERAGE_CONFIG=str(scratch / "coverage.ini"),
                ),
            )
            progress("private_training_serialization_cli_fixture_checked")
        if not failures and all(r["passed"] for r in preflight):
            source_ref = eligible.shard(raw, "sources", source)
            fitted = train(source, raw)
            fitted_ref = eligible.shard(raw, "fitted", fitted)
            progress("base_checkpoints_frozen_before_calibration_labels")
            calibration = calibrate(source, fitted, raw)
            cal_ref = reference(raw / "calibration.json")
            progress("calibration_frozen_before_public_predictions")
            measured = measure(source, fitted, calibration)
            measure_ref = eligible.shard(raw, "measurement", measured)
            v.update(
                {
                    k: measured[k]
                    for k in (
                        "rows",
                        "sample_size_budget",
                        "public_prediction_seals",
                        "eligible_target_counts",
                    )
                }
            )
            v.update(
                head_checkpoints=fitted["checkpoints"],
                calibration_checkpoint=cal_ref,
                checkpoint_references=[source_ref, fitted_ref, cal_ref],
                measurement_checkpoint=measure_ref,
                convergence_rows=fitted["convergence_rows"],
                gradient_checks=fitted["gradient_checks"],
                immutable_input_hashes=list(source["public_manifests"].values())
                + list(source["evaluator_manifests"].values())
                + [source["exclusion_manifest"]],
                trained_head_specs=[
                    dict(
                        arm=h["arm"],
                        seed=h["seed"],
                        parameters=h["parameter_count"],
                        device="cpu",
                        independent_method=h["arm"] != "sigmoid_identity",
                        converged=h["converged"],
                    )
                    for h in fitted["heads"]
                ],
                acceptance_gate_results=dict(
                    measurement=fitted["ready"] and calibration["ready"],
                    natural_benefit=False,
                    owned_checks=False,
                ),
            )
            v["genuine_headroom"].update(
                converged_primary_heads=sum(
                    h["converged"] for h in fitted["heads"] if h["arm"] == "conditioned_energy"
                ),
                historical_fixed_200_gradient_norms=[
                    h["gradient_norm"] for h in fitted["heads"] if h["arm"] == "fixed_200_residual"
                ],
                correctness_benefit_measured=False,
            )
            v["arm_controls"] = dict(
                arms=ARMS,
                sigmoid_identity="same coefficients and calibration, zero extra fitting, zero independent credit",
                historical_fixed_200=sparse.CONFIG,
                scalar_controls="same source roster and labels; q-only restricted information",
                convergent_controls="identical nine public inputs for energy and linear; intercept uses prevalence only",
            )
            if args.fixture:
                v.update(
                    verdict_class="circular_positive",
                    honest_verdict="complete_circular_positive_qualified_fit_fixture",
                    verifier_is_oracle=True,
                )
        measured_at = time.monotonic()
        if not args.fixture:
            receipts = preflight + run_commands(
                ROOT,
                commands[1:] if preflight else commands,
                log_dir=raw / "validation_logs",
                heartbeat_s=30,
                extra_env=dict(
                    PYTHONUNBUFFERED="1",
                    JAX_PLATFORMS="cpu",
                    OPENBLAS_NUM_THREADS="1",
                    CARNOT_8020_COVERAGE_CONFIG=str(scratch / "coverage.ini"),
                    COVERAGE_FILE=str(scratch / ".coverage-health"),
                ),
            )
            v["validation_receipts"] = [r for r in receipts if r["scope"] == "owned"]
            v["repository_health"] = [r for r in receipts if r["scope"] == "repository_health"]
            report = (
                json.loads((scratch / "coverage.json").read_text())
                if (scratch / "coverage.json").exists()
                else dict(files={})
            )
            counts = {k: r["summary"] for k, r in report["files"].items()}
            v["coverage_statement_counts"] = counts
            passed = (
                bool(counts)
                and all(
                    r["missing_lines"] == 0 and r["num_statements"] > 0 for r in counts.values()
                )
                and all(r["passed"] for r in v["validation_receipts"])
            )
            v["acceptance_gate_results"]["owned_checks"] = passed
            v["energy_fit_ready_score"] = int(
                passed and not failures and v["acceptance_gate_results"]["measurement"]
            )
            if not passed and not failures:
                v.update(
                    verdict_class="disqualified",
                    honest_verdict="complete_disqualified_qualified_energy_checks",
                )
        ended = time.monotonic()
        v.update(
            duration_s=ended - began,
            phase_spans=[
                dict(phase="freeze", duration_s=frozen - began),
                dict(phase="fit_calibrate_predict", duration_s=measured_at - frozen),
                dict(phase="validation", duration_s=ended - measured_at),
            ],
            terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        )
        v["raw_shard_hashes"] = (
            [config_ref]
            + historical_custody
            + v["checkpoint_references"]
            + v["head_checkpoints"]
            + v["immutable_input_hashes"]
            + v["cited_upstream_artifacts"]
        )
        v["raw_shard_hashes"] += [r["checkpoint"] for r in v["convergence_rows"]]
        if v["measurement_checkpoint"]:
            v["raw_shard_hashes"].append(v["measurement_checkpoint"])
        v["reproducibility_checksum"] = canonical_hash(
            dict(config=CONFIG, code=code, raw=v["raw_shard_hashes"])
        )
        v["field_principles"] = {
            k: "Bind this invocation to exact durable evidence. Numerical readiness, duplicate seeds and artificial fixtures give no natural correctness or learning credit."
            for k in v
        }
        v["field_principles"].update(
            energy_fit_ready_score="Exact downstream integer: valid converged measurement and all owned checks; no scientific benefit credit.",
            sample_size_budget="Count source groups once; seeds, identity duplicates and prediction repetitions do not enlarge independent n.",
            rows="Descriptive fit/tune/calibration log loss and public prediction coverage; stream and retention targets remain unopened.",
            genuine_headroom="Compare registered convergence diagnostics; lower objective alone cannot qualify a correctness win.",
            eligible_target_counts="Import Exp8019 support counts from original checked bytes; no outcome-dependent cohort selection.",
            repository_health="One bounded full-suite diagnostic; original failures stay separate from task-owned qualification.",
        )
        progress("publication_before")
        receipt = publish_primary(output, v, replay if args.fixture else terminal)
        atomic_json(raw / "terminal_validation.json", receipt)
        reader = reader_receipt(
            TASK,
            output.parent,
            field="energy_fit_ready_score",
            expected=v["energy_fit_ready_score"],
        )
        atomic_json(raw / "reader_receipt.json", reader)
        if not reader["passed"] or reader["gate_sha256"] != sha256_file(output):
            raise ValueError("primary_reader")
        replay(output)
        progress("publication_after")
        return 0
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(f"[exp8020] failed={error}", flush=True)
        return 1
