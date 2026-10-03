"""REQ-REPORT-8073: train fixed source interactions on exposed cached evidence.

The representation changes by three registered products. Exact binary energies
and logistic probabilities describe the same fitted predictor, with no new truth
information. Reserved evaluation outcomes belong to the independent audit.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np
from scipy.special import expit  # type: ignore[import-untyped]

from carnot import experiment_8020_v695_qualified_energy_fit as qualified
from carnot import experiment_8072_v699_sealed_methods as methods
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar
from carnot.reporting.v686_contract_validation import run_check
from carnot.verify import evidence_features_7980 as features

Json = dict[str, Any]
Array = qualified.Array
old, sparse, scalar = qualified.old, qualified.sparse, qualified.scalar
ROOT = qualified.ROOT
NAME = "experiment_8073_v699_interaction_energy_fit"
TASK = "exp8073-interaction-energy-fit"
MODULE, CLI, TEST = (
    f"python/carnot/{NAME}.py",
    f"scripts/experiments/{NAME}.py",
    "tests/python/test_interaction_energy_8073.py",
)
OWNED = [MODULE, CLI]
CONFIG = methods.methods()["source"]
SEED = 6998073
PINS = {
    "experiment_8019_v695_eligible_targets": "sha256:310fbb2c5d0b254ceb2e00cd742c1d83c856ff4082075b7cd61e592c940ea518",
    "experiment_8020_v695_qualified_energy_fit": "sha256:54b02f0752b47f4c27930e8bec1d4f5d7ab165b1f258c344f8267f0d7cad4bf0",
    "experiment_8058_v698_sealed_evidence_methods": methods.PINS[8058],
    "experiment_8072_v699_sealed_methods": "sha256:159df4b8b393c5a4feaeb71292b81b7b1c7219de39bdfd34d47a8bd1ba026d9e",
}
START = time.monotonic()


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real work counts so operators can distinguish fitting from a stall."""
    print(
        f"[exp8073] {phase} elapsed_s={time.monotonic() - START:.3f} completed={completed} pending={pending}",
        flush=True,
    )


def load_sources(root: Path, raw: Path, *, mutate: bool = False) -> Json:
    """Bind original bytes before joining labels; reserved targets stay unopened."""
    plan: Json = dict(
        data={},
        failures=[],
        source_artifact_hashes=[],
        cited_upstream_artifacts=[],
        label_access_events=[],
        support={},
        role_hashes={},
    )
    operand: Json = {}

    def require(path: Path, field: str, expected: Any, observed: Any) -> None:
        nonlocal operand
        operand = dict(
            check=field,
            upstream=path.stem,
            path=str(path.absolute()),
            hash=sha256_file(path) if path.is_file() else None,
            field=field,
            op="==",
            expected=expected,
            observed=observed,
        )
        if observed != expected:
            raise ValueError(field)

    def bind(path: Path, digest: str | None = None) -> Json:
        require(path, "resource_exists", True, path.is_file())
        if digest:
            require(path, "sha256", digest, sha256_file(path))
        ref = qualified.old.prior.copy_evidence(reference(path), raw)
        plan["source_artifact_hashes"].append(ref)
        return json.loads(checked(ref).read_text())

    progress("preconditions_before")
    try:
        for relative in methods.prior.INPUTS + [
            "python/carnot/experiment_8020_v695_qualified_energy_fit.py",
            "python/carnot/experiment_8008_v694_conditioned_energy_fit.py",
            "python/carnot/verify/sparse_energy_7996.py",
        ]:
            p = root / relative
            require(p, "resource_exists", True, p.is_file())
        for tool in ("python", "pytest", "coverage", "ruff", "mypy"):
            require(
                ROOT / ".venv/bin" / tool,
                "resource_exists",
                True,
                (ROOT / ".venv/bin" / tool).is_file(),
            )
        require(Path(sys.executable), "python_version>=3.11", True, sys.version_info >= (3, 11))
        parents = {}
        for name, digest in PINS.items():
            path = root / "results" / (name + ".json")
            v = bind(path, digest)
            require(path, "flagged_adversarial", False, v["flagged_adversarial"])
            publication = bind(Path(v["terminal_validation_sidecar_path"]))
            pub = publication.get("publication", publication)
            bind(Path(pub["sidecar_path"]))
            try:
                report = read_bound_sidecar(path, Path(pub["sidecar_path"]))
            except ValueError as error:
                require(path, "authenticated_terminal", "current_primary_path_and_hash", str(error))
            require(path, "terminal_primary_sha256", digest, pub["primary_sha256"])
            require(path, "terminal_report.passed", True, report["report"]["passed"])
            require(path, "terminal_primary_path", str(path.absolute()), report["primary_path"])
            parents[name] = v
        targets, sealed = parents[next(iter(PINS))], parents[list(PINS)[-1]]
        require(
            root / "results",
            "source_protocol_ready_score",
            1,
            sealed["source_protocol_ready_score"],
        )
        require(
            root / "results",
            "sealed_source_methods",
            CONFIG,
            sealed["method_freeze"]["methods"]["source"],
        )
        capture_path = root / "results/experiment_7969_v691_qwen_calibration_capture.json"
        capture = bind(
            capture_path, "sha256:3f25b2e4b43d50536525e64ace321db55e9d6889aab97cc2581e4665014f2f51"
        )
        captures = {
            r["family_id"]: (r, capture["raw_response_shards"][i])
            for i, r in enumerate(capture["rows"])
        }
        plan["cited_upstream_artifacts"].append(
            dict(
                reference(capture_path),
                MODEL_SPECS=capture["MODEL_SPECS"],
                model_identity_receipt=capture["model_identity_receipt"],
                model_invocation_counts=capture["model_invocation_counts"],
                scope="historical_live_Qwen_provenance; zero current calls",
            )
        )
        seen: set[str] = set()
        for role in ("fit", "tune", "evaluation"):
            ref = sealed["role_manifests"][role]
            original = bind(Path(ref["path"]), ref["sha256"])["rows"]
            require(
                Path(ref["path"]),
                "original_slots",
                list(range(CONFIG["roles"][role])),
                [r["slot"] for r in original],
            )
            if mutate and role == "tune":
                original[0]["source_cluster_id"] = plan["data"]["fit"][0]["source"]
            groups = {r["source_cluster_id"] for r in original}
            require(Path(ref["path"]), "source_cluster_overlap", [], sorted(seen & groups))
            seen.update(groups)
            plan["role_hashes"][role] = canonical_hash(original)
            if role == "evaluation":
                continue
            public_ref, target_ref = (
                targets["public_manifests"][role],
                targets["evaluator_manifests"][role],
            )
            public = {
                r["family_id"]: r
                for r in bind(Path(public_ref["path"]), public_ref["sha256"])["rows"]
            }
            labels = {
                r["family_id"]: r
                for r in bind(Path(target_ref["path"]), target_ref["sha256"])["rows"]
            }
            plan["label_access_events"].append(
                dict(
                    order=len(plan["label_access_events"]),
                    role=role,
                    purpose="fit_only_CV" if role == "fit" else "affine_calibration_only",
                    path=target_ref["path"],
                    sha256=target_ref["sha256"],
                    selected_rows=len(original),
                )
            )
            joined = []
            for r in original:
                p, label = public[r["family_id"]], labels[r["family_id"]]
                call, call_ref = captures[r["family_id"]]
                saved = bind(Path(call_ref["path"]), call_ref["sha256"])
                require(Path(call_ref["path"]), "capture_bytes", call, saved)
                require(
                    Path(call_ref["path"]),
                    "qualified_Qwen_scalar",
                    call["parsed"].get("probability"),
                    p["q"],
                )
                require(
                    Path(call_ref["path"]),
                    "capture_identity",
                    call["capture_identity"],
                    p["capture_identity"],
                )
                for key in ("source_bytes", "answer_bytes", "source_cluster_id"):
                    require(Path(public_ref["path"]), key, r[key], p[key])
                require(
                    Path(public_ref["path"]),
                    "source_normalization",
                    features.normalized(bytes.fromhex(r["source_bytes"])),
                    r["source_cluster_id"],
                )
                fresh = features.extract(
                    {k: r[k] for k in ("source_bytes", "answer_bytes", "family_id")}
                )
                require(
                    Path(public_ref["path"]), "original_features", fresh["values"], p["features"]
                )
                require(
                    Path(target_ref["path"]),
                    "response_sha256",
                    "sha256:" + hashlib.sha256(bytes.fromhex(r["answer_bytes"])).hexdigest(),
                    label["response_sha256"],
                )
                eligible = p["public_eligible"] and label["eligible_y"] in (0, 1)
                if eligible:
                    require(
                        Path(target_ref["path"]),
                        "complete_response_target",
                        True,
                        label["completely_annotated"] and label["custody_passed"],
                    )
                joined.append(
                    dict(
                        unit=f"{role}/{r['slot']}",
                        source=r["source_cluster_id"],
                        family_id=r["family_id"],
                        role=role,
                        slot=r["slot"],
                        q=p["q"],
                        features=p["features"],
                        capture_identity=p["capture_identity"],
                        y=label["eligible_y"],
                        status="completed" if eligible else "excluded",
                        exclusion_reason=None
                        if eligible
                        else label["eligibility_reason"] or p["exclusion_reason"],
                    )
                )
            plan["data"][role] = joined
            plan["support"][role] = scalar.support(
                [
                    dict(r, source_cluster_id=r["source"])
                    for r in joined
                    if r["status"] == "completed"
                ],
                *CONFIG["floors"][role],
            )
            require(
                Path(target_ref["path"]), role + "_support", True, plan["support"][role]["passed"]
            )
            progress("joined_" + role, len(joined), 0)
        plan["cited_upstream_artifacts"] += deepcopy(plan["source_artifact_hashes"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        plan["failures"].append(dict(operand, error=str(error)))
    progress("preconditions_after", sum(map(len, plan["data"].values())), 0)
    return plan


def products(x: Array, geo: Json) -> Array:
    """Center each input before multiplying so the new terms isolate joint variation."""
    sx, _ = sparse.scale(x, geo["scaler"])
    z = (
        np.log(
            np.clip(x[:, 0], *old.CONFIG["clip_q"]) / (1 - np.clip(x[:, 0], *old.CONFIG["clip_q"]))
        )
        - geo["logit_center"]
    ) / geo["logit_scale"]
    centered = sx - np.asarray(geo["feature_centers"])
    return np.column_stack(
        (z * centered[:, 2], z * centered[:, 8], centered[:, 6] * centered[:, 4])
    )


def geometry(x: Array) -> Json:
    """Training rows alone fix bounds, centering and the inherited spline knots."""
    geo = old.geometry(x)
    sx, _ = sparse.scale(x, geo["scaler"])
    geo.update(feature_centers=sx.mean(axis=0).tolist(), feature_names=CONFIG["features"])
    geo["interaction_centers"] = products(x, geo).mean(axis=0).tolist()
    return geo


def design(arm: str, x: Array, geo: Json) -> Array:
    """Keep the matched additive columns identical while appending fixed products."""
    if geo["feature_names"] != CONFIG["features"] or x.shape[1] != 9:
        raise ValueError("feature_columns")
    if arm not in CONFIG["arms"]:
        raise ValueError("arm")
    mapped = {"intercept": "intercept_only", "scalar_affine": "linear", "linear": "linear"}.get(
        arm, "conditioned_energy"
    )
    base = old.design(mapped, x, geo)
    if arm == "scalar_affine":
        return base[:, :2]
    if arm == "interaction":
        return np.column_stack((base, products(x, geo) - np.asarray(geo["interaction_centers"])))
    return base


def predict(head: Json, x: Array, *, logistic: bool = False) -> Array:
    """Normalize both energies independently of the sigmoid equivalence control."""
    z = design(head["arm"], x, head["geometry"]) @ np.asarray(head["parameters"])
    b, a = head.get("calibration", [0.0, 1.0])
    z = b + a * z
    if logistic:
        return np.asarray(expit(z))
    weights = np.exp(np.column_stack((np.zeros(len(z)), z)) - np.maximum(0, z)[:, None])
    return np.asarray(weights[:, 1] / weights.sum(axis=1))


def choose_ridge(losses: dict[float, float]) -> float:
    """Break identical loss ties toward stronger ridge without consulting tune labels."""
    return min(losses, key=lambda r: (losses[r], -r))


def solve(arm: str, x: Array, y: Array, geo: Json, ridge: float, deadline: float) -> Json:
    """Retain genuine optimizer traces and verify analytical gradients numerically."""
    progress("benchmark_before_" + arm)
    matrix = design(arm, x, geo)
    h = old.optimize(matrix, y, ridge, SEED, deadline=deadline)
    theta = np.asarray(h["parameters"])
    numeric = []
    for i in range(len(theta)):
        d = np.eye(1, len(theta), i)[0] * 1e-6
        numeric.append(
            (
                old.objective(theta + d, matrix, y, ridge)[0]
                - old.objective(theta - d, matrix, y, ridge)[0]
            )
            / 2e-6
        )
    error = float(np.max(np.abs(np.asarray(numeric) - old.objective(theta, matrix, y, ridge)[1])))
    h.update(
        arm=arm,
        geometry=geo,
        finite=bool(np.isfinite(theta).all() and np.isfinite(h["final_loss"])),
        gradient_check_error=error,
    )
    progress("benchmark_after_" + arm, h["optimizer_steps"], 0)
    return h


def train(data: Json, raw: Path) -> Json:
    """Fit grouped CV on fit only, then freeze bases before tune affine calibration."""
    if set(data) != {"fit", "tune"}:
        raise ValueError("role_roster")
    if any(r["role"] != role for role, rows in data.items() for r in rows) or {
        r["source"] for r in data["fit"]
    } & {r["source"] for r in data["tune"]}:
        raise ValueError("held_out_role_exclusion")
    usable = {role: [r for r in rows if r["status"] == "completed"] for role, rows in data.items()}
    x, tx = (sparse.inputs(usable[role]) for role in ("fit", "tune"))
    y, ty = (np.asarray([r["y"] for r in usable[role]], dtype=float) for role in ("fit", "tune"))
    groups = sorted(
        {r["source"] for r in usable["fit"]}, key=lambda s: hashlib.sha256(s.encode()).hexdigest()
    )
    assignment = {s: i % 4 for i, s in enumerate(groups)}
    folds = np.array([assignment[r["source"]] for r in usable["fit"]])
    deadline = time.monotonic() + CONFIG["budget_s"]
    heads, trials, audits, bases, refs = [], [], [], [], []
    fold_geo = {fold: geometry(x[folds != fold]) for fold in range(4)}
    geo = geometry(x)
    for arm in CONFIG["arms"]:
        losses = {}
        for ridge in CONFIG["ridge_grid"]:
            held_losses = []
            for fold in range(4):
                mask = folds != fold
                h = solve(arm, x[mask], y[mask], fold_geo[fold], ridge, deadline)
                loss = scalar.bce(predict(h, x[~mask]), y[~mask])
                held_losses.append(loss)
                trials.append(
                    dict(
                        arm=arm,
                        ridge=ridge,
                        fold=fold,
                        loss=loss,
                        fit_roles=["fit"],
                        training_sources=[s for s in groups if assignment[s] != fold],
                        held_out_sources=[s for s in groups if assignment[s] == fold],
                        geometry_hash=canonical_hash(fold_geo[fold]),
                        checkpoint=qualified.eligible.shard(raw, "trials", h),
                        converged=h["converged"],
                        finite=h["finite"],
                        gradient_norm=h["gradient_norm"],
                        gradient_check_error=h["gradient_check_error"],
                        optimizer_steps=h["optimizer_steps"],
                        objective_evaluations=h["objective_evaluations"],
                    )
                )
                progress("cross_validation", len(trials), 105 - len(trials))
            losses[ridge] = float(np.mean(held_losses))
        ridge = choose_ridge(losses)
        h = solve(arm, x, y, geo, ridge, deadline)
        h.update(
            selected_ridge=ridge,
            cross_validation_losses={str(k): v for k, v in losses.items()},
            fit_source_role_hash=canonical_hash(data["fit"]),
        )
        bases.append(qualified.eligible.shard(raw, "base_heads", h))
        trials.append(
            dict(
                arm=arm,
                ridge=ridge,
                fold="final",
                fit_roles=["fit"],
                converged=h["converged"],
                finite=h["finite"],
                gradient_norm=h["gradient_norm"],
                gradient_check_error=h["gradient_check_error"],
                optimizer_steps=h["optimizer_steps"],
                objective_evaluations=h["objective_evaluations"],
                checkpoint=bases[-1],
            )
        )
        heads.append(h)
    progress("all_base_heads_sealed_before_tune_calibration", len(bases), 0)
    for h in heads:
        z = design(h["arm"], tx, geo) @ np.asarray(h["parameters"])
        progress("calibration_benchmark_before_" + h["arm"])
        cal = old.optimize(
            np.column_stack((np.ones(len(tx)), z)),
            ty,
            0.0001,
            SEED,
            deadline=deadline,
            initial=np.array([0.0, 1.0]),
        )
        h.update(
            calibration=cal["parameters"],
            calibration_receipt=cal,
            tune_source_role_hash=canonical_hash(data["tune"]),
        )
        refs.append(qualified.eligible.shard(raw, "heads", h))
        audits.append(
            dict(
                arm=h["arm"],
                maximum_error=h["gradient_check_error"],
                passed=h["gradient_check_error"] < 1e-7,
            )
        )
        progress("calibration_benchmark_after_" + h["arm"], cal["optimizer_steps"], 0)
    interaction = heads[-1]
    parity = float(np.max(np.abs(predict(interaction, x) - predict(interaction, x, logistic=True))))
    swap = x.copy()
    swap[:, [2, 8]] = swap[:, [8, 2]]
    mutation = dict(
        passed=not np.array_equal(design("interaction", x, geo), design("interaction", swap, geo)),
        maximum_prediction_change=float(
            np.max(np.abs(predict(interaction, x) - predict(interaction, swap)))
        ),
        columns=[CONFIG["features"][2], CONFIG["features"][8]],
    )
    reload_ok = all(
        np.array_equal(predict(h, x), predict(json.loads(checked(ref).read_text()), x))
        for h, ref in zip(heads, refs, strict=True)
    )
    return dict(
        heads=heads,
        head_checkpoints=refs,
        base_checkpoints=bases,
        convergence_rows=trials,
        gradient_checks=audits,
        fold_assignment=assignment,
        fold_geometry=fold_geo,
        normalization_hashes=[canonical_hash(geo), *[canonical_hash(g) for g in fold_geo.values()]],
        equivalent_logistic_parity=dict(
            maximum_error=parity, passed=parity < 1e-10, additional_fits=0, independent_method=False
        ),
        feature_permutation_check=mutation,
        save_load_prediction_parity=reload_ok,
        ready=all(
            r["finite"] and r["converged"] and r["gradient_check_error"] < 1e-7 for r in trials
        )
        and all(h["calibration_receipt"]["converged"] for h in heads)
        and parity < 1e-10
        and mutation["passed"]
        and reload_ok,
    )


def measure(data: Json, fitted: Json) -> list[Json]:
    """Export probabilities and loss operands without reading evaluation targets."""
    rows = []
    for role, originals in data.items():
        usable = [r for r in originals if r["status"] == "completed"]
        x = sparse.inputs(usable)
        for head in fitted["heads"]:
            predictions = dict(
                zip([r["unit"] for r in usable], predict(head, x).tolist(), strict=True)
            )
            for r in originals:
                p = predictions.get(r["unit"])
                loss = (
                    None
                    if p is None
                    else float(
                        -r["y"] * np.log(max(p, 1e-12)) - (1 - r["y"]) * np.log(max(1 - p, 1e-12))
                    )
                )
                rows.append(
                    dict(
                        r,
                        arm=head["arm"],
                        seed=SEED,
                        condition=role,
                        probability=p,
                        metric="binary_log_loss",
                        numerator=0 if loss is None else loss,
                        denominator=int(loss is not None),
                        binary_log_loss=loss,
                    )
                )
    return rows


def build(work: Json, raw: Path, receipts: list[Json], coverage: Json, fixture: bool) -> Json:
    """Numerical readiness is separate from improvement and external support blocks."""
    source, fitted = work["source"], work["fitted"]
    required = [r for r in receipts if r.get("classification", "required") == "required"]
    checks = (
        not fixture
        and [r["name"] for r in required] == work["validation_manifest"]
        and all(r["passed"] for r in required)
        and all(
            p in coverage
            and coverage[p]["summary"]["num_statements"] > 0
            and coverage[p]["summary"]["missing_lines"] == 0
            for p in OWNED
        )
    )
    numerical = bool(fitted and fitted["ready"])
    kind = (
        "disqualified"
        if not fixture and not checks
        else "blocked"
        if source["failures"]
        else "null"
        if fixture or (checks and numerical)
        else "disqualified"
    )
    rows = measure(source["data"], fitted) if fitted else []
    sizes = dict(
        intended_count=480,
        eligible_count=sum(r["status"] == "completed" for r in rows),
        independent_count=len({r["source"] for r in rows if r["status"] == "completed"}),
        completed_count=sum(r["status"] == "completed" for r in rows),
        censored_count=480 - len(rows),
        excluded_count=sum(r["status"] == "excluded" for r in rows),
        failed_count=0,
    )
    gates = deepcopy(source["failures"])
    if kind == "disqualified":
        gates.append(
            dict(
                check="owned_validation_and_numerics",
                upstream=TASK,
                path=str(raw / "work.json"),
                hash=sha256_file(raw / "work.json"),
                field="checks_and_ready",
                op="==",
                expected=True,
                observed=checks and numerical,
            )
        )
    v: Json = dict(
        experiment_id=8073,
        task_id=TASK,
        milestone="2026.10.699",
        run_date="20261003",
        schema="carnot.v699.interaction_energy_fit.v1",
        honest_verdict="complete_" + kind + "_interaction_energy_fit",
        verdict_class=kind,
        verifier_is_oracle=False,
        flagged_adversarial=False,
        required_checks_passed=checks,
        interaction_fit_ready_score=int(checks and numerical and not gates),
        generalized_learning_benefit_score=0,
        claim_scope="Numerical fitting readiness on historically exposed development source groups. Decision improvement is unmeasured; evaluation96 labels remain unopened. Energy and equivalent logistic forms share all coefficients and information.",
        rows=rows,
        **sizes,
        sample_size_budget=dict(
            **sizes,
            count_unit="head/source fitting rows; independent_count deduplicates original sources",
            source_groups_planned=96,
            roles=CONFIG["roles"],
            support_floors=CONFIG["floors"],
            independent_environments=0,
        ),
        gate_check_summary=gates,
        inference_substrate="verifier_ensemble_against_cached_candidates"
        if fitted
        else "aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=deepcopy(ZERO_INVOCATION_COUNTS),
        substrate_declaration=dict(
            reduction="verifier_ensemble_against_cached_candidates"
            if fitted
            else "aggregation_from_upstream_artifacts",
            mode="no_model_load",
            MODEL_SPECS=[],
        ),
        random_seed=SEED,
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        source_artifact_hashes=source["source_artifact_hashes"],
        cited_upstream_artifacts=source["cited_upstream_artifacts"],
        code_config_hashes=work["code_config_hashes"],
        head_checkpoints=fitted.get("head_checkpoints", []),
        normalization_hashes=fitted.get("normalization_hashes", []),
        convergence_rows=fitted.get("convergence_rows", []),
        gradient_checks=fitted.get("gradient_checks", []),
        equivalent_logistic_parity=fitted.get("equivalent_logistic_parity", {}),
        feature_permutation_check=fitted.get("feature_permutation_check", {}),
        save_load_prediction_parity=fitted.get("save_load_prediction_parity", False),
        interaction_names=CONFIG["interactions"],
        basis_equations=dict(
            additive="[1,z,108 inherited clamped cubic spline columns]",
            centering="z=(logit(clip(q,.0001,.9999))-fit_mean)/fit_std; u_j=fit_minmax(x_j)-fit_mean_j",
            interactions="[z*u_overlap_max,z*u_uncovered_max,u_numeric_max*u_negation_max] minus fit product means",
            energies="E(x,0)=0; E(x,1)=-(b+a*B(x)w); p(unsupported)=sigmoid(b+a*B(x)w)",
            knots=sparse.KNOTS,
        ),
        label_access_events=source["label_access_events"],
        fit_tune_support=source["support"],
        source_role_hashes=source["role_hashes"],
        trained_head_specs=[
            dict(
                arm=h["arm"],
                parameter_count=h["parameter_count"],
                calibration_parameter_count=2,
                selected_ridge=h["selected_ridge"],
                calibration=h["calibration"],
                device="cpu",
                independent_method=True,
            )
            for h in fitted.get("heads", [])
        ],
        optimizer_work=dict(
            fits=len(fitted.get("convergence_rows", [])),
            calibration_fits=len(fitted.get("heads", [])),
            steps=sum(r["optimizer_steps"] for r in fitted.get("convergence_rows", []))
            + sum(h["calibration_receipt"]["optimizer_steps"] for h in fitted.get("heads", [])),
            budget_s=600,
        ),
        validation_receipts=receipts,
        validation_command_manifest=work["commands"],
        coverage_statement_counts=coverage,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        repository_health=[r for r in receipts if r.get("classification") == "diagnostic"],
        methodology_note="Five fixed heads and three fixed interactions; fit-only grouped CV; tune-only affine calibration with ridge .0001. All current pretrained model counts are zero. Readiness does not test H1 or imply generalized learning.",
    )
    v["raw_shard_hashes"] = (
        [reference(raw / "work.json"), reference(raw / "configuration.json")]
        + fitted.get("head_checkpoints", [])
        + fitted.get("base_checkpoints", [])
        + [r["checkpoint"] for r in fitted.get("convergence_rows", [])]
    )
    v["reproducibility_checksum"] = canonical_hash(
        dict(
            config=CONFIG,
            code=work["code_config_hashes"],
            sources=source["source_artifact_hashes"],
            shards=v["raw_shard_hashes"],
        )
    )
    v["field_principles"] = {
        k: "Bind "
        + k
        + " to this invocation's exact evidence; fitting validity and exposed development observations cannot establish independent scientific benefit."
        for k in v
    }
    v["field_principles"].update(
        interaction_fit_ready_score="Finite fitting and all owned checks qualify readiness even without improvement.",
        equivalent_logistic_parity="Identical coefficients and basis prevent energy renaming from receiving independent method credit.",
        label_access_events="Only fit/tune targets are opened; evaluation outcomes cannot select representation or capacity.",
        independent_count="Count source groups once across heads, folds and repetitions.",
    )
    return v


def replay(path: Path) -> bool:
    """Rebuild predictions and reductions from authenticated frozen checkpoints."""
    try:
        v = json.loads(path.read_text())
        raw = Path(v["terminal_validation_sidecar_path"]).parent
        for ref in v["source_artifact_hashes"] + v["raw_shard_hashes"]:
            checked(ref)
        if any(sha256_file(ROOT / p) != digest for p, digest in v["code_config_hashes"].items()):
            return False
        for receipt in v["validation_receipts"]:
            if sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]:
                return False
        work = json.loads((raw / "work.json").read_text())
        if json.loads((raw / "configuration.json").read_text())["config"] != CONFIG:
            return False
        fitted = work["fitted"]
        if fitted:
            x = sparse.inputs(
                [r for r in work["source"]["data"]["fit"] if r["status"] == "completed"]
            )
            if fitted["heads"][0]["geometry"] != geometry(x):
                return False
            for h, ref in zip(fitted["heads"], fitted["head_checkpoints"], strict=True):
                if json.loads(checked(ref).read_text()) != h:
                    return False
                if not np.array_equal(
                    predict(h, x), predict(json.loads(checked(ref).read_text()), x)
                ):
                    return False
        validation = json.loads((raw / "validation.json").read_text())
        return (
            build(work, raw, validation["receipts"], validation["coverage"], validation["fixture"])
            == v
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def manifest(private: Path) -> list[Json]:
    """Reuse bounded validation routes while measuring statements only in new code."""
    specs = methods.manifest(private)
    replacements = {methods.MODULE: MODULE, methods.CLI: CLI, methods.TEST: TEST}
    for spec in specs:
        spec["argv"] = [
            replacements.get(a, a)
            .replace(str(ROOT / methods.MODULE), str(ROOT / MODULE))
            .replace(str(ROOT / methods.CLI), str(ROOT / CLI))
            for a in spec["argv"]
        ]
        if spec["name"] == "repository_full_suite":
            spec["argv"] = [str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"]
    (private / "coverage.ini").write_text(
        "[run]\nparallel = true\ndata_file = "
        + str(private / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
    )
    return specs


def terminal(path: Path) -> Json:
    """Bound fresh cold replay and artifact linters before reader-visible publication."""
    raw = Path(json.loads(path.read_text())["terminal_validation_sidecar_path"]).parent
    commands = [
        (
            "cold_replay",
            [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / CLI), "--cold-replay", str(path)],
        ),
        (
            "adversarial",
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "scripts/adversarial_verify.py"),
                "--json",
                str(path),
            ],
        ),
        (
            "strict_rows",
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "scripts/verdict_row_consistency_lint.py"),
                "--strict",
                str(path),
            ],
        ),
    ]
    with tempfile.TemporaryDirectory(prefix="carnot-8073-terminal-") as temp:
        receipts = [
            run_check(
                ROOT,
                dict(name=n, argv=a, deadline_s=60, expected_exit=0),
                Path(temp),
                raw / "terminal_logs",
            )
            for n, a in commands
        ]
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Run fitting in a bounded child and publish after normal exit and validation."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    progress("start_no_model_loads")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261003"], default="20261003")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--fit-child", type=Path)
    parser.add_argument("--mutate", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            return 0 if replay(args.cold_replay) else 1
        if args.fit_child:
            source = load_sources(args.root, args.fit_child, mutate=args.mutate)
            error = None
            try:
                fitted = {} if source["failures"] else train(source["data"], args.fit_child)
            except ValueError as failure:
                fitted, error = {}, str(failure)
            atomic_json(
                args.fit_child / "fitting.json",
                dict(source=source, fitted=fitted, owned_fitting_error=error),
            )
            progress("fitting_child_normal_exit", len(fitted.get("heads", [])), 0)
            return 0
        output = (args.fixture_output or args.output).absolute()
        raw = output.parent / "raw" / output.stem
        if (raw / "work.json").exists():
            raise ValueError("immutable_invocation_exists")
        raw.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="carnot-8073-") as temp:
            private = Path(temp)
            specs = manifest(private)
            child = dict(
                name="fitting_child_normal_exit",
                argv=[
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(ROOT / CLI),
                    "--root",
                    str(args.root),
                    "--fit-child",
                    str(raw),
                ]
                + (["--mutate"] if args.mutate else []),
                deadline_s=660,
                expected_exit=0,
                classification="required",
            )
            codes = {
                p: sha256_file(ROOT / p)
                for p in OWNED
                + [
                    TEST,
                    qualified.OWNED[0],
                    old.OWNED[0],
                    str(Path(sparse.__file__).relative_to(ROOT)),
                    str(Path(features.__file__).relative_to(ROOT)),
                ]
            }
            commands = [child, *specs]
            atomic_json(
                raw / "configuration.json",
                dict(
                    config=CONFIG,
                    commands=commands,
                    code_config_hashes=codes,
                    calibration_ridge=0.0001,
                    random_seed=SEED,
                ),
            )
            frozen = time.monotonic()
            progress("configuration_and_validation_manifest_frozen")
            child_receipt = run_check(ROOT, child, private, raw / "validation_logs")
            work = (
                json.loads((raw / "fitting.json").read_text())
                if child_receipt["passed"]
                else dict(
                    source=dict(
                        data={},
                        failures=[],
                        source_artifact_hashes=[],
                        cited_upstream_artifacts=[],
                        label_access_events=[],
                        support={},
                        role_hashes={},
                    ),
                    fitted={},
                    owned_fitting_error="fitting_child_failed",
                )
            )
            measured = time.monotonic()
            work.update(
                code_config_hashes=codes,
                commands=commands,
                validation_manifest=[
                    s["name"] for s in commands if s.get("classification", "required") == "required"
                ],
            )
            os.environ["CARNOT_8073_COVERAGE_CONFIG"] = str(private / "coverage.ini")
            progress("owned_validation_before", 1, len(specs))
            receipts = [child_receipt] + (
                []
                if args.fixture_output
                else [run_check(ROOT, s, private, raw / "validation_logs") for s in specs]
            )
            coverage = (
                json.loads((private / "coverage.json").read_text())["files"]
                if (private / "coverage.json").is_file()
                else {}
            )
            ended = time.monotonic()
            work.update(
                duration_s=ended - began,
                phase_spans=[
                    dict(phase="freeze", duration_s=frozen - began),
                    dict(phase="fit_calibrate_seal", duration_s=measured - frozen),
                    dict(phase="validation", duration_s=ended - measured),
                ],
            )
            atomic_json(raw / "work.json", work)
            atomic_json(
                raw / "validation.json",
                dict(receipts=receipts, coverage=coverage, fixture=bool(args.fixture_output)),
            )
            value = build(work, raw, receipts, coverage, bool(args.fixture_output))
            atomic_json(raw / "primitive_fitting_rows.json", dict(rows=value["rows"]))
            progress("owned_validation_after", len(receipts), 0)
            publication = publish_primary(output, value, terminal)
            atomic_json(
                raw / "terminal_validation.json",
                dict(publication=publication, owned_invocation_exit=0),
            )
            progress("complete", value["completed_count"], 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        progress("rejected_" + str(error))
        return 1
