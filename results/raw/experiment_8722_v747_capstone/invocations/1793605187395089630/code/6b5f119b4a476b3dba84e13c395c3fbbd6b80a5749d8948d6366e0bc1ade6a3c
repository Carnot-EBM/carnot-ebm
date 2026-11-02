"""REQ-REPORT-8023: preserve intervention evidence separately from human targets.

Likelihood measures answer plausibility. Human annotations alone supply labels;
all fitting is exposed development and earns no independent science credit.
"""

from __future__ import annotations

import math
import json
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

import numpy as np
from scipy.special import expit  # type: ignore[import-untyped]

from carnot import experiment_8008_v694_conditioned_energy_fit as solver
from carnot.experiment_8019_v695_eligible_targets import shard
from carnot.inference import fixed_answer_likelihood_8022 as s
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify.qwen_energy_calibration_7972 import bce, support

Json = dict[str, Any]
FEATURES = ["full_mean_nll", "no_source_minus_full", "removal_minus_full", "q"]
ARMS = dict(
    conditional_linear_energy=[0, 1, 2, 3],
    same_feature_logistic=[0, 1, 2, 3],
    full_nll_qwen=[0, 3],
    scalar_qwen=[3],
    intercept=[],
)
CONFIG = dict(
    ridge_grid=[0.001, 0.01, 0.1, 1],
    max_iterations=1000,
    cpu_training_seconds=120,
    seed=69523,
    feature_names=FEATURES,
    feature_signs="unconstrained_fit_only",
    costs=dict(accept_unsupported=5, reject_supported=1, escalate=0.5, correct=0),
    thresholds=dict(accept_below=0.1, reject_above=0.5),
    tune_selection="lowest_tune_BCE_then_grid_order",
    comparator_selection="lowest_tune_BCE_then_registered_control_order",
    exposure=s.METHODS["exposure"],
    eval_label_access_count=0,
)


def capture(panel: Json, runtime: Any, raw: Path, *, deadline: float) -> list[Json]:
    """Checkpoint starts and nonstarts so timeouts cannot shrink the denominator."""
    started = time.monotonic()
    rows: list[Json] = []
    stopped = False
    for item in panel["rows"]:
        for arm in s.ARMS:
            view = item["views"][arm]
            row = dict(
                id=item["family_id"] + ":" + arm,
                family_id=item["family_id"],
                role=item["role"],
                arm=arm,
                seed=69523,
                view_sha256=item["view_hashes"][arm],
                source_sha256=canonical_hash(item["source_bytes"]),
                answer_bytes=item["answer_bytes"],
                answer_sha256=canonical_hash(item["answer_bytes"]),
                status="censored",
                mean_nll=None,
                numerator=None,
                denominator=len(item["target_tokens"]),
                token_rows=[],
                generated_tokens=0,
                failure_reason=None,
                exclusion_reason=None,
                censor_reason="model_work_cap_or_prior_failure",
            )
            path = raw / f"pass-{len(rows):03d}.json"
            if not stopped and time.monotonic() < deadline:
                row.update(
                    status="running", censor_reason=None, started_monotonic_ns=time.monotonic_ns()
                )
                atomic_json(path, row)
                s.progress(
                    "8023_before_teacher_forcing",
                    started,
                    len(rows),
                    4 * len(panel["rows"]) - len(rows),
                )
                try:
                    runtime.reset()
                    runtime.eval(view["tokens"])
                    if runtime.n_tokens != len(view["tokens"]):
                        raise ValueError("runtime_token_alignment")
                    result = s.target_likelihood(
                        runtime.scores, view["tokens"], view["response_start"]
                    )
                    row["token_rows"] = [
                        dict(token_id=t, offset=o, log_probability=p)
                        for t, o, p in zip(
                            item["target_tokens"],
                            item["response_token_offsets"],
                            result["target_logprobs"],
                            strict=True,
                        )
                    ]
                    numerator = math.fsum(-r["log_probability"] for r in row["token_rows"])
                    row.update(
                        status="completed",
                        numerator=numerator,
                        mean_nll=numerator / row["denominator"],
                        normalization_max_error=result["normalization_max_error"],
                    )
                except (OSError, RuntimeError, TimeoutError, ValueError) as error:
                    row.update(status="failed", failure_reason=f"{type(error).__name__}:{error}")
                    stopped = True
                row["ended_monotonic_ns"] = time.monotonic_ns()
                s.progress(
                    "8023_after_teacher_forcing",
                    started,
                    len(rows) + 1,
                    4 * len(panel["rows"]) - len(rows) - 1,
                )
            atomic_json(path, row)
            rows.append(row)
    return rows


def reduce(panel: Json, rows: list[Json]) -> Json:
    """Rebuild means from token primitives without trusting the scorer's totals."""
    expected = [(r, a) for r in panel["rows"] for a in s.ARMS]
    if [r["id"] for r in rows] != [r["family_id"] + ":" + a for r, a in expected]:
        raise ValueError("slot_roster")
    means: Json = {}
    tokens = 0
    for row, (item, arm) in zip(rows, expected, strict=True):
        if (
            row["view_sha256"] != canonical_hash(item["views"][arm])
            or row["source_sha256"] != canonical_hash(item["source_bytes"])
            or row["answer_bytes"] != item["answer_bytes"]
            or row["generated_tokens"] != 0
        ):
            raise ValueError("fixed_view_identity")
        if row["status"] != "completed":
            continue
        primitive = row["token_rows"]
        if (
            [r["token_id"] for r in primitive] != item["target_tokens"]
            or [r["offset"] for r in primitive] != item["response_token_offsets"]
            or any(
                not math.isfinite(r["log_probability"]) or r["log_probability"] > 0
                for r in primitive
            )
        ):
            raise ValueError("token_identity")
        numerator = math.fsum(-r["log_probability"] for r in primitive)
        if (
            row["denominator"] != len(primitive)
            or not primitive
            or abs(row["numerator"] - numerator) > 1e-10
            or abs(row["mean_nll"] - numerator / len(primitive)) > 1e-10
        ):
            raise ValueError("token_aggregate")
        means[row["id"]] = numerator / len(primitive)
        tokens += len(primitive)
    features, drift = [], []
    for item in panel["rows"]:
        fid = item["family_id"]
        if any(fid + ":" + a not in means for a in s.ARMS):
            continue
        values = {a: means[fid + ":" + a] for a in s.ARMS}
        difference = abs(values["duplicate"] - values["full"])
        drift.append(dict(family_id=fid, drift=difference, passed=difference <= 1e-6))
        features.append(
            dict(
                family_id=fid,
                role=item["role"],
                source_cluster_id=item["source_normalized_hash"],
                full_mean_nll=values["full"],
                no_source_minus_full=values["no_source"] - values["full"],
                removal_minus_full=values["removed_chunk"] - values["full"],
                status="completed" if difference <= 1e-6 else "disqualified",
            )
        )
    return dict(
        source_feature_rows=features,
        duplicate_drift_rows=drift,
        scored_tokens=tokens,
        forward_pass_counts=sum(r["status"] in {"completed", "failed"} for r in rows),
        passed=len(features) == len(panel["rows"]) and all(r["passed"] for r in drift),
    )


def predict(head: Json, rows: list[Json]) -> Any:
    """Normalize two label energies; the identity arm uses the same frozen gap."""
    x = np.asarray([[r[k] for k in FEATURES] for r in rows], dtype=float)
    scaled = (x - np.asarray(head["mean"])) / np.asarray(head["scale"])
    matrix = np.column_stack((np.ones(len(rows)), scaled[:, head["columns"]]))
    z = matrix @ np.asarray(head["parameters"])
    if head["arm"] == "conditional_linear_energy":
        weights = np.exp(np.column_stack((np.zeros(len(z)), z)) - np.maximum(0, z)[:, None])
        return weights[:, 1] / weights.sum(axis=1)
    return expit(z)


def train(rows: list[Json], raw: Path) -> Json:
    """Only fit labels change coefficients; tune labels choose among frozen ridges."""
    usable = [
        r for r in rows if r["status"] == "completed" and r["y"] in (0, 1) and r["q"] is not None
    ]
    f, t = ([r for r in usable if r["role"] == role] for role in ("fit", "tune"))
    supports = {
        role: support(group, minimum, classes)
        for role, group, minimum, classes in [("fit", f, 48, 8), ("tune", t, 24, 4)]
    }
    result: Json = dict(
        heads=[],
        checkpoints=[],
        trials=[],
        role_support=supports,
        config=CONFIG,
        ready=False,
        selected_comparator=None,
    )
    if not all(v["passed"] for v in supports.values()):
        return result
    x = np.asarray([[r[k] for k in FEATURES] for r in f], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite_features")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1
    scaled = (x - mean) / scale
    y, ty = (np.asarray([r["y"] for r in group], dtype=float) for group in (f, t))
    deadline = time.monotonic() + 120
    for arm, columns in ARMS.items():
        candidates = []
        for ridge in CONFIG["ridge_grid"]:
            s.progress(
                "8023_before_CPU_benchmark_" + arm,
                deadline - 120,
                len(result["trials"]),
                20 - len(result["trials"]),
            )
            matrix = np.column_stack((np.ones(len(f)), scaled[:, columns]))
            with patch.dict(solver.CONFIG, max_steps=1000):
                h = solver.optimize(matrix, y, ridge, CONFIG["seed"], deadline=deadline)
            h.update(
                arm=arm,
                columns=columns,
                mean=mean.tolist(),
                scale=scale.tolist(),
                finite=bool(np.isfinite(h["parameters"]).all()),
                energy_definition="E(x,0)=0; E(x,1)=-z(x)",
                feature_names=FEATURES,
                thresholds=CONFIG["thresholds"],
            )
            h["tune_loss"] = bce(predict(h, t), ty)
            ref = shard(raw, "trials", h)
            result["trials"].append(ref)
            candidates.append(h)
            s.progress(
                "8023_after_CPU_benchmark_" + arm,
                deadline - 120,
                len(result["trials"]),
                20 - len(result["trials"]),
            )
        chosen = min(candidates, key=lambda h: h["tune_loss"])
        result["heads"].append(chosen)
        if arm == "conditional_linear_energy":
            identity = dict(
                chosen, arm="sigmoid_identity", additional_fit_steps=0, duplicate_of=arm
            )
            if not np.allclose(predict(chosen, t), predict(identity, t), atol=1e-15, rtol=0):
                raise ValueError("sigmoid_identity")
            result["heads"].append(identity)
    result["checkpoints"] = [shard(raw, "heads", h) for h in result["heads"]]
    controls = [h for h in result["heads"] if h["arm"] in list(ARMS)[1:]]
    result["selected_comparator"] = min(controls, key=lambda h: h["tune_loss"])["arm"]
    result["ready"] = all(h["converged"] and h["finite"] for h in result["heads"])
    result["descriptive_losses"] = [
        dict(
            arm=h["arm"],
            role=role,
            metric=bce(predict(h, group), labels),
            numerator=bce(predict(h, group), labels) * len(group),
            denominator=len(group),
        )
        for h in result["heads"]
        for role, group, labels in [("fit", f, y), ("tune", t, ty)]
    ]
    return result


def check_fit(rows: list[Json], fitted: Json) -> None:
    """Cold-check scaling, selected states and losses from original joined rows."""
    from carnot.reporting.evidence_features_custody_7980 import checked

    usable = [
        r for r in rows if r["status"] == "completed" and r["y"] in (0, 1) and r["q"] is not None
    ]
    f, t = ([r for r in usable if r["role"] == role] for role in ("fit", "tune"))
    supports = {
        role: support(group, minimum, classes)
        for role, group, minimum, classes in [("fit", f, 48, 8), ("tune", t, 24, 4)]
    }
    if fitted["role_support"] != supports or fitted["config"] != CONFIG:
        raise ValueError("fit_contract_drift")
    if not fitted["heads"]:
        return
    x = np.asarray([[r[k] for k in FEATURES] for r in f], dtype=float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1
    trials = [json.loads(checked(ref).read_text()) for ref in fitted["trials"]]
    for head in trials:
        matrix = np.column_stack((np.ones(len(f)), ((x - mean) / scale)[:, head["columns"]]))
        loss, gradient, _ = solver.objective(
            np.asarray(head["parameters"]), matrix, np.asarray([r["y"] for r in f]), head["l2"]
        )
        if (
            head["mean"] != mean.tolist()
            or head["scale"] != scale.tolist()
            or abs(loss - head["final_loss"]) > 1e-10
            or abs(float(np.max(np.abs(gradient))) - head["gradient_norm"]) > 1e-10
            or head["converged"] != (float(np.max(np.abs(gradient))) < 1e-8)
            or abs(head["tune_loss"] - bce(predict(head, t), np.asarray([r["y"] for r in t])))
            > 1e-10
        ):
            raise ValueError("fit_state_drift")
    for head in fitted["heads"]:
        arm = "conditional_linear_energy" if head["arm"] == "sigmoid_identity" else head["arm"]
        chosen = min((h for h in trials if h["arm"] == arm), key=lambda h: h["tune_loss"])
        if head["parameters"] != chosen["parameters"] or head["l2"] != chosen["l2"]:
            raise ValueError("tune_selection_drift")
    controls = [h for h in fitted["heads"] if h["arm"] in list(ARMS)[1:]]
    if fitted["selected_comparator"] != min(controls, key=lambda h: h["tune_loss"])["arm"]:
        raise ValueError("comparator_drift")
