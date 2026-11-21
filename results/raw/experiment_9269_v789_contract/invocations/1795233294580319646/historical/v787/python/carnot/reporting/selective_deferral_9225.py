"""REQ-REPORT-9225-V785: selective-deferral feedback masks and spline-support diagnostics.

Authenticate original Exp9097 human custody, Exp9126/9132 method receipts and
Exp9211 qualified target/ambiguity authority independently of contract gates.
Freeze deterministic delayed-feedback observation masks, inverse-propensity
estimators and label-free 57-column cubic B-spline support-density strata.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from carnot.reporting import human_label_custody_8389 as old
from carnot.reporting import logit_methods_runner_9126 as static_runner
from carnot.reporting import online_methods_9132 as online_methods
from carnot.reporting import online_methods_runner_9132 as online_runner
from carnot.reporting.logit_methods_9126 import ref as ref

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = (
    "experiment_9225_v785_selective_deferral_protocol",
    "exp9225-selective-deferral-protocol",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_selective_deferral_9225.py"
OWNED = [
    "python/carnot/reporting/selective_deferral_9225.py",
    "python/carnot/reporting/selective_deferral_runner_9225.py",
    CLI,
]
SEED = 785225
SEED_PREFIX = "v785-selective-deferral"
ALPHA_GRID = (0.10, 0.25)
ESCALATION_LOW = 0.1
ESCALATION_HIGH = 0.9
REGIMES = ("full_release", "escalation_only", "audited_deferral")
SPLINE_KNOTS = (-1.0, -1.0, -1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0)
SPLINE_STRATA = ("interior_dense", "boundary_sparse_or_clipped")
BOUNDARY_MASS_THRESHOLD = 0.10
ROLES = dict(TRAIN=150, CAL=50, TEST=100)
GENERATORS = ("llama3-8b", "gemma-2-9b", "mistral-7b")
MODEL_SPECS: list[Json] = []
PINS = {
    "results/experiment_9097_v776_human_authority.json": (
        "sha256:24c9e0983d9a4eb0aa5a488b5467386e57d6f13b79447f0f3a2f3d1c68e25532"
    ),
    "results/experiment_9126_v778_logit_methods.json": (
        "sha256:16cd3742f2ec4dea0d67b6503e3d9e4dd84f7a3898fdbe15087065208ed1a1fb"
    ),
    "results/experiment_9132_v778_online_methods.json": (
        "sha256:f59a4197962a8591757d094b480082a9abcad40e81ecd2e1bf6244218802bd55"
    ),
    "results/experiment_9137_v778_capstone.json": (
        "sha256:4cc5a692cfdefdbe72073ee03eed2a16bd9674a2ca56e3a7d0bf41e9e2bb842e"
    ),
    "results/experiment_9141_v779_human_contrast_audit.json": (
        "sha256:b2aa70be88fc641c93452909cfb8f754b894958aa8d4a10ed8b64634ec214d5e"
    ),
    "results/experiment_9180_v782_contract.json": (
        "sha256:42c35a172e548872dbbc82c737ce2c3ee3a624d4a0fec2495225c2f7f38e92f2"
    ),
    "results/experiment_9211_v784_label_ambiguity.json": (
        "sha256:00da04356c958a063793c21d85287653e38eda69c85c65ca1819426519852610"
    ),
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase boundaries so authentication and CPU method controls stay visible."""
    print(f"[exp9225] phase={phase} completed={completed} pending={pending}", flush=True)


def check(
    path: Path, field: str, expected: Any, observed: Any, name: str, upstream: str = TASK
) -> Json:
    """Keep strict operand types so missing_file, missing_field and observed zero differ."""
    return dict(
        check=name,
        upstream=upstream,
        **ref(path),
        artifact_field=field,
        op="==",
        expected=expected,
        observed=observed,
        passed=type(expected) is type(observed) and expected == observed,
    )


def selective_deferral_protocol() -> Json:
    """Freeze selective delayed-feedback observation rules before any model outcome."""
    return dict(
        alpha_grid=list(ALPHA_GRID),
        escalation_interval=[ESCALATION_LOW, ESCALATION_HIGH],
        primary_gate_rescue=False,
        primary_protocol="full_release",
        regimes=list(REGIMES),
        seed_prefix=SEED_PREFIX,
        spline_basis_columns=57,
        spline_strata=list(SPLINE_STRATA),
        spline_support_knots=list(SPLINE_KNOTS),
        unobserved_accepted_drift_at_alpha_zero="unidentifiable",
        version="v785-selective-deferral-v1",
    )


def spline_support_protocol() -> Json:
    """Freeze label-free 57-column cubic B-spline support-density rules before reading labels."""
    return dict(
        version="v785-spline-support-v1",
        paper="PostHoc FREE Calibrating on Kolmogorov Arnold Networks, arXiv:2503.01195 (2025-03)",
        knot_vector=list(SPLINE_KNOTS),
        degree=3,
        feature_count=7,
        basis_columns=57,
        spline_columns_per_feature=7,
        boundary_basis_indices=[0, 6],
        interior_basis_indices=[1, 2, 3, 4, 5],
        boundary_thresholds=dict(
            clip_abs_threshold=1.0,
            interior_coord_bound=0.5,
            boundary_basis_mass_fraction_threshold=BOUNDARY_MASS_THRESHOLD,
            clipped_coordinate_count_threshold=1,
        ),
        strata=list(SPLINE_STRATA),
        labels_read=False,
        primary_gate_rescue=False,
        selection="support-density strata are label-free descriptive slices and never alter knots, loss or updates",
    )


def audit_draw(stream_id: str, qid: str, sibling: str) -> Json:
    """Derive deterministic pre-outcome Bernoulli audit draws from SHA256('v785-selective-deferral' + stream_id + qid + sibling)."""
    if not (stream_id and qid and sibling):
        raise ValueError("invalid_seed_identifiers")
    seed_material = f"{SEED_PREFIX}{stream_id}{qid}{sibling}"
    digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    seed_int = int(digest[:16], 16)
    audit_u = seed_int / float(1 << 64)
    return dict(
        stream_id=stream_id,
        qid=qid,
        sibling=sibling,
        seed_material=seed_material,
        seed_sha256="sha256:" + digest,
        seed_int=seed_int,
        audit_u=audit_u,
        non_escalated_audit_draws={
            "0.1": bool(audit_u < 0.10),
            "0.25": bool(audit_u < 0.25),
        },
    )


def typed_decision(p: float) -> str:
    """Apply the frozen Exp9132 typed verifier thresholds: accept<=0.1, reject>=0.9, else escalate."""
    val = float(p)
    if not math.isfinite(val) or not (0.0 <= val <= 1.0):
        raise ValueError("invalid_probability")
    if val <= ESCALATION_LOW:
        return "accept"
    if val >= ESCALATION_HIGH:
        return "reject"
    return "escalate"


def observation_propensity(p: float, regime: str, alpha: float = 0.0) -> float:
    """Return label observation probability pi under full_release, escalation_only, or audited_deferral."""
    decision = typed_decision(p)
    if regime not in REGIMES:
        raise ValueError("unknown_regime")
    if regime == "full_release":
        return 1.0
    if regime == "escalation_only":
        if alpha != 0.0:
            raise ValueError("invalid_alpha")
        return 1.0 if decision == "escalate" else 0.0
    a = float(alpha)
    if not math.isfinite(a) or not (0.0 < a <= 1.0):
        raise ValueError("invalid_alpha")
    return 1.0 if decision == "escalate" else a


def observe_feedback(
    p: float,
    regime: str,
    *,
    alpha: float = 0.0,
    stream_id: str = "stream_00",
    qid: str = "q0",
    sibling: str = "llama3-8b",
    audit_u: float | None = None,
) -> Json:
    """Evaluate deterministic selective-feedback observation indicator and inverse-propensity weight."""
    decision = typed_decision(p)
    pi = observation_propensity(p, regime, alpha)
    if audit_u is None:
        draw = audit_draw(stream_id, qid, sibling)
        u = float(draw["audit_u"])
        seed_sha256: str | None = str(draw["seed_sha256"])
    else:
        u = float(audit_u)
        if not math.isfinite(u) or not (0.0 <= u < 1.0):
            raise ValueError("invalid_audit_u")
        seed_sha256 = None
    observed = (
        True
        if (regime == "full_release" or decision == "escalate")
        else False
        if regime == "escalation_only"
        else bool(u < float(alpha))
    )
    eff_alpha = (
        1.0 if regime == "full_release" else 0.0 if regime == "escalation_only" else float(alpha)
    )
    ipw_weight = (1.0 / pi if observed else 0.0) if pi > 0.0 else None
    return dict(
        decision=decision,
        regime=regime,
        alpha=eff_alpha,
        propensity=pi,
        audit_u=u,
        seed_sha256=seed_sha256,
        observed=observed,
        inverse_propensity_weight=ipw_weight,
    )


def estimate_selective_risk(observations: list[Json], regime: str, alpha: float = 0.0) -> Json:
    """Compute exact Horvitz-Thompson inverse-propensity and naive observed-only risk estimators."""
    if not observations:
        raise ValueError("empty_observations")
    default_weight = 1.0 / len(observations)
    records: list[Json] = []
    weights: list[float] = []
    for idx, obs in enumerate(observations):
        p = float(obs["p"])
        y = obs.get("y")
        w = float(obs.get("weight", default_weight))
        if type(y) is not int or y not in (0, 1) or not math.isfinite(w) or w <= 0.0:
            raise ValueError("invalid_observation_operands")
        fb = observe_feedback(
            p,
            regime,
            alpha=alpha,
            stream_id=str(obs.get("stream_id", f"stream_{idx // 15:02d}")),
            qid=str(obs.get("qid", obs.get("unit_id", f"q_{idx:03d}"))),
            sibling=str(obs.get("sibling", GENERATORS[idx % 3])),
            audit_u=obs.get("audit_u"),
        )
        records.append(dict(p=p, y=y, w=w, brier=(p - y) ** 2, accepted=p <= ESCALATION_LOW, **fb))
        weights.append(w)
    total_w = math.fsum(weights)
    norm_records = [dict(r, nw=r["w"] / total_w) for r in records]
    true_brier = math.fsum(r["nw"] * r["brier"] for r in norm_records)
    acc_w = math.fsum(r["nw"] for r in norm_records if r["accepted"])
    true_acc_err_joint = math.fsum(r["nw"] * r["y"] for r in norm_records if r["accepted"])
    true_acc_err_cond = (true_acc_err_joint / acc_w) if acc_w > 0.0 else None
    obs_w = math.fsum(r["nw"] for r in norm_records if r["observed"])
    naive_brier = (
        math.fsum(r["nw"] * r["brier"] for r in norm_records if r["observed"]) / obs_w
        if obs_w > 0.0
        else None
    )
    identifiable = all(r["propensity"] > 0.0 for r in norm_records)
    acc_identifiable = all(r["propensity"] > 0.0 for r in norm_records if r["accepted"])
    if identifiable:
        ipw_brier: float | None = math.fsum(
            r["nw"] * float(r["inverse_propensity_weight"]) * r["brier"] for r in norm_records
        )
        exact_ipw_brier: float | None = true_brier
    else:
        ipw_brier = None
        exact_ipw_brier = None
    if acc_w > 0.0 and acc_identifiable:
        ipw_acc_joint: float | None = math.fsum(
            r["nw"] * float(r["inverse_propensity_weight"]) * r["y"]
            for r in norm_records
            if r["accepted"]
        )
        ipw_acc_cond: float | None = ipw_acc_joint / acc_w
        exact_ipw_acc_cond: float | None = true_acc_err_cond
        acc_obs_w = math.fsum(r["nw"] for r in norm_records if r["accepted"] and r["observed"])
        naive_acc_cond: float | None = (
            math.fsum(r["nw"] * r["y"] for r in norm_records if r["accepted"] and r["observed"])
            / acc_obs_w
            if acc_obs_w > 0.0
            else None
        )
    else:
        ipw_acc_joint = None
        ipw_acc_cond = None
        exact_ipw_acc_cond = None
        naive_acc_cond = None
    eff_alpha = (
        1.0 if regime == "full_release" else 0.0 if regime == "escalation_only" else float(alpha)
    )
    return dict(
        regime=regime,
        alpha=eff_alpha,
        observation_count=len(norm_records),
        observed_count=sum(bool(r["observed"]) for r in norm_records),
        observed_weight_fraction=obs_w,
        accepted_weight_fraction=acc_w,
        naive_estimate=naive_brier,
        ipw_estimate=ipw_brier,
        true_fixture_risk=true_brier,
        exact_ipw_brier_expectation=exact_ipw_brier,
        true_accepted_error_joint=true_acc_err_joint,
        true_accepted_error_conditional=true_acc_err_cond,
        ipw_accepted_error_joint=ipw_acc_joint,
        ipw_accepted_error_conditional=ipw_acc_cond,
        exact_ipw_accepted_error_expectation=exact_ipw_acc_cond,
        naive_accepted_error_conditional=naive_acc_cond,
        accepted_drift_identifiable=bool(acc_identifiable),
        identifiability_status="identifiable" if identifiable else "unidentifiable",
        selection_bias=(naive_brier - true_brier) if naive_brier is not None else None,
    )


def monte_carlo_audit_summary(
    observations: list[Json], *, alpha: float, trials: int = 400, seed: int = SEED
) -> Json:
    """Evaluate Monte Carlo mean of Horvitz-Thompson IPW and naive estimators over independent audit draws."""
    if not observations or trials <= 0:
        raise ValueError("invalid_monte_carlo_operands")
    rng = np.random.default_rng(seed)
    draws = rng.random((trials, len(observations)))
    ipw_briers: list[float] = []
    ipw_acc_errs: list[float] = []
    naive_briers: list[float] = []
    for t in range(trials):
        trial_obs = [dict(obs, audit_u=float(draws[t, i])) for i, obs in enumerate(observations)]
        res = estimate_selective_risk(trial_obs, "audited_deferral", alpha=alpha)
        ipw_briers.append(float(res["ipw_estimate"]))
        if res["ipw_accepted_error_conditional"] is not None:
            ipw_acc_errs.append(float(res["ipw_accepted_error_conditional"]))
        if res["naive_estimate"] is not None:
            naive_briers.append(float(res["naive_estimate"]))
    return dict(
        alpha=float(alpha),
        trials=trials,
        seed=seed,
        mc_ipw_brier_mean=math.fsum(ipw_briers) / len(ipw_briers),
        mc_ipw_accepted_error_mean=math.fsum(ipw_acc_errs) / len(ipw_acc_errs),
        mc_naive_brier_mean=math.fsum(naive_briers) / len(naive_briers),
    )


def fit_feature_transform(x_init: Any) -> Json:
    """Fit initialization-only standardization and min/max bounds without reading labels."""
    a = np.asarray(x_init, dtype=float)
    if (
        a.ndim != 2
        or a.shape[1] != 7
        or len(a) == 0
        or np.isinf(a).any()
        or np.isnan(a).all(axis=0).any()
    ):
        raise ValueError("invalid_feature_initialization")
    median = np.nanmedian(a, axis=0)
    imputed = np.where(np.isnan(a), median, a)
    mean = imputed.mean(axis=0)
    raw_scale = imputed.std(axis=0)
    constant_coordinates = np.flatnonzero(raw_scale == 0).tolist()
    scale = raw_scale.copy()
    scale[scale == 0] = 1.0
    standardized = (imputed - mean) / scale
    return dict(
        median=median.tolist(),
        mean=mean.tolist(),
        scale=scale.tolist(),
        minimum=standardized.min(axis=0).tolist(),
        maximum=standardized.max(axis=0).tolist(),
        constant_coordinates=[int(i) for i in constant_coordinates],
        fit_rows=int(len(a)),
    )


def classify_spline_support(
    x: Any,
    fitted: Json,
    *,
    unit_records: list[Json] | None = None,
    monotone_transform: str | None = None,
) -> Json:
    """Classify 57-column cubic B-spline support-density strata using feature values only."""
    if unit_records is not None and any(
        any(k in rec for k in ("annotator1", "annotator2", "label", "y")) for rec in unit_records
    ):
        raise ValueError("label_leakage_refused")
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[1] != 7 or len(a) == 0 or np.isinf(a).any():
        raise ValueError("invalid_feature_matrix")
    if unit_records is not None and len(unit_records) != len(a):
        raise ValueError("unit_records_length")
    standardized = (np.where(np.isnan(a), fitted["median"], a) - fitted["mean"]) / fitted["scale"]
    low, high = (
        np.asarray(fitted["minimum"], dtype=float),
        np.asarray(fitted["maximum"], dtype=float),
    )
    span = high - low
    unclipped = np.where(
        span > 0, 2.0 * (standardized - low) / np.where(span > 0, span, 1.0) - 1.0, 0.0
    )
    scaled = online_methods.transform(a, fitted)
    if monotone_transform is not None:
        if monotone_transform != "cubic_preserving_endpoints":
            raise ValueError("unknown_monotone_transform")
        scaled = np.clip(0.95 * scaled + 0.05 * (scaled**3), -1.0, 1.0)
    phi = online_methods.design(scaled, True)
    spline_blocks = phi[:, 8:57].reshape(len(a), 7, 7)
    coord_sums = spline_blocks.sum(axis=2)
    partition_errors = np.max(np.abs(coord_sums - 1.0), axis=1)
    boundary_masses = spline_blocks[:, :, 0] + spline_blocks[:, :, 6]
    rows: list[Json] = []
    for idx in range(len(a)):
        rec_meta = unit_records[idx] if unit_records is not None else {}
        uid = str(rec_meta.get("unit_id", f"row_{idx:03d}"))
        clipped_count = int(np.sum(np.abs(scaled[idx]) >= 1.0 - 1e-12))
        out_of_range_count = int(np.sum(np.abs(unclipped[idx]) > 1.0 + 1e-12))
        total_basis_sum = float(np.sum(coord_sums[idx]))
        boundary_mass = float(np.sum(boundary_masses[idx]))
        boundary_fraction = boundary_mass / total_basis_sum
        interior_fraction = 1.0 - boundary_fraction
        stratum = (
            "boundary_sparse_or_clipped"
            if (clipped_count >= 1 or boundary_fraction >= BOUNDARY_MASS_THRESHOLD)
            else "interior_dense"
        )
        rows.append(
            dict(
                unit_id=uid,
                scaled_features=[float(v) for v in scaled[idx]],
                clipped_coordinate_count=clipped_count,
                out_of_range_coordinate_count=out_of_range_count,
                basis_sum=total_basis_sum,
                max_coordinate_partition_error=float(partition_errors[idx]),
                boundary_basis_mass=boundary_mass,
                boundary_basis_mass_fraction=boundary_fraction,
                interior_basis_mass_fraction=interior_fraction,
                stratum=stratum,
            )
        )
    return dict(
        basis_columns=int(phi.shape[1]),
        knot_vector=list(SPLINE_KNOTS),
        row_count=len(rows),
        interior_dense_count=sum(r["stratum"] == "interior_dense" for r in rows),
        boundary_sparse_or_clipped_count=sum(
            r["stratum"] == "boundary_sparse_or_clipped" for r in rows
        ),
        max_partition_of_unity_error=float(np.max(partition_errors)),
        labels_read=False,
        rows=rows,
    )


def build_stream_deferral_masks(
    stream_roster: list[list[str]],
    label_index: dict[str, Json],
    master_rows: dict[str, Json],
) -> Json:
    """Build deterministic pre-outcome delayed-feedback masks for the ten five-qid streams."""
    if len(stream_roster) != 10 or any(len(s) != 5 for s in stream_roster):
        raise ValueError("invalid_stream_roster")
    seen_qids: set[str] = set()
    rows: list[Json] = []
    mask_rows: list[Json] = []
    audit_seeds: list[Json] = []
    for s_idx, stream_qids in enumerate(stream_roster):
        stream_id = f"stream_{s_idx:02d}"
        for q_idx, qid in enumerate(stream_qids):
            if qid in seen_qids:
                raise ValueError("duplicate_stream_qid")
            seen_qids.add(qid)
            for sibling in GENERATORS:
                uid = f"{qid}__{sibling}"
                lrec = label_index.get(uid)
                mrec = master_rows.get(uid)
                if (
                    lrec is None
                    or mrec is None
                    or lrec.get("role") != "TRAIN"
                    or mrec.get("model") != sibling
                ):
                    raise ValueError("stream_member_invalid")
                draw = audit_draw(stream_id, qid, sibling)
                a1 = int(lrec["annotator1"])
                raw_a2 = lrec.get("annotator2")
                a2 = -1 if raw_a2 is None else int(raw_a2)
                a2_status = (
                    "missing_a2"
                    if a2 == -1
                    else "observed_agreement"
                    if a1 == a2
                    else "observed_disagreement"
                )
                audit_0_10 = int(bool(draw["non_escalated_audit_draws"]["0.1"]))
                audit_0_25 = int(bool(draw["non_escalated_audit_draws"]["0.25"]))
                entry = dict(
                    unit_id=uid,
                    stream_unit_id=f"{stream_id}:{uid}",
                    stream_id=stream_id,
                    stream_index=s_idx,
                    query_index=q_idx,
                    qid=qid,
                    group_id=lrec["group_id"],
                    sibling=sibling,
                    role="TRAIN",
                    seed_material=draw["seed_material"],
                    seed_sha256=draw["seed_sha256"],
                    audit_u=draw["audit_u"],
                    full_release_non_escalated_pi=1.0,
                    escalation_only_non_escalated_pi=0.0,
                    audit_alpha_0_10=audit_0_10,
                    audit_alpha_0_25=audit_0_25,
                    annotator1=a1,
                    annotator2=a2,
                    a2_status=a2_status,
                )
                mask_rows.append(entry)
                audit_seeds.append(
                    dict(
                        stream_id=stream_id,
                        qid=qid,
                        sibling=sibling,
                        seed_sha256=draw["seed_sha256"],
                        audit_u=draw["audit_u"],
                        audit_alpha_0_10=bool(audit_0_10),
                        audit_alpha_0_25=bool(audit_0_25),
                    )
                )
                rows.append(
                    dict(
                        **entry,
                        group=qid,
                        arm="selective_deferral_stream_mask",
                        condition=stream_id,
                        seed=SEED,
                        metric="audit_draw_u",
                        absolute_metric=float(draw["audit_u"]),
                        numerator=float(draw["audit_u"]),
                        denominator=1,
                        status="completed",
                        exclusion_reason=None,
                        censor_reason=None,
                    )
                )
    return dict(rows=rows, mask_rows=mask_rows, audit_seeds=audit_seeds)


def controls(mask_rows: list[Json], all_labels: list[Json], rosters: Json) -> Json:
    """Prove IPW unbiasedness, alpha=0 unidentifiability, and 57-column spline support controls."""
    progress("controls_before")
    fixture_obs: list[Json] = []
    for idx, m in enumerate(mask_rows):
        bucket = idx % 3
        k = idx // 3
        p = 0.05 if bucket == 0 else 0.50 if bucket == 1 else 0.95
        y = (
            int(k % 10 == 0)
            if bucket == 0
            else int(k % 2 == 0)
            if bucket == 1
            else int(k % 10 != 0)
        )
        fixture_obs.append(
            dict(
                unit_id=m["unit_id"],
                stream_id=m["stream_id"],
                qid=m["qid"],
                sibling=m["sibling"],
                p=p,
                y=y,
                audit_u=m["audit_u"],
                weight=1.0 / len(mask_rows),
            )
        )
    drifted_obs = [dict(o, y=1 if o["p"] <= ESCALATION_LOW else o["y"]) for o in fixture_obs]
    full_res = estimate_selective_risk(fixture_obs, "full_release")
    esc_res = estimate_selective_risk(fixture_obs, "escalation_only")
    esc_drift_res = estimate_selective_risk(drifted_obs, "escalation_only")
    estimator_rows: list[Json] = [
        dict(
            control_id="full_release_omniscient_reference",
            regime="full_release",
            alpha=1.0,
            naive_estimate=full_res["naive_estimate"],
            ipw_estimate=full_res["ipw_estimate"],
            true_fixture_risk=full_res["true_fixture_risk"],
            exact_ipw_brier_expectation=full_res["exact_ipw_brier_expectation"],
            true_accepted_error_conditional=full_res["true_accepted_error_conditional"],
            ipw_accepted_error_conditional=full_res["ipw_accepted_error_conditional"],
            naive_accepted_error_conditional=full_res["naive_accepted_error_conditional"],
            identifiability_status=full_res["identifiability_status"],
            accepted_drift_identifiable=full_res["accepted_drift_identifiable"],
            selection_bias=full_res["selection_bias"],
            passed=bool(
                full_res["identifiability_status"] == "identifiable"
                and math.isclose(
                    float(full_res["ipw_estimate"]),
                    float(full_res["true_fixture_risk"]),
                    abs_tol=1e-15,
                )
                and math.isclose(
                    float(full_res["naive_estimate"]),
                    float(full_res["true_fixture_risk"]),
                    abs_tol=1e-15,
                )
            ),
        ),
        dict(
            control_id="escalation_only_selection_bias_and_unidentifiability",
            regime="escalation_only",
            alpha=0.0,
            naive_estimate=esc_res["naive_estimate"],
            ipw_estimate=esc_res["ipw_estimate"],
            true_fixture_risk=esc_res["true_fixture_risk"],
            exact_ipw_brier_expectation=esc_res["exact_ipw_brier_expectation"],
            true_accepted_error_conditional=esc_res["true_accepted_error_conditional"],
            ipw_accepted_error_conditional=esc_res["ipw_accepted_error_conditional"],
            naive_accepted_error_conditional=esc_res["naive_accepted_error_conditional"],
            identifiability_status=esc_res["identifiability_status"],
            accepted_drift_identifiable=esc_res["accepted_drift_identifiable"],
            selection_bias=esc_res["selection_bias"],
            passed=bool(
                esc_res["identifiability_status"] == "unidentifiable"
                and esc_res["accepted_drift_identifiable"] is False
                and esc_res["ipw_estimate"] is None
                and esc_res["ipw_accepted_error_conditional"] is None
                and esc_res["naive_accepted_error_conditional"] is None
                and float(esc_res["selection_bias"]) > 0.08
            ),
        ),
        dict(
            control_id="escalation_only_catastrophic_accepted_drift_unobservable",
            regime="escalation_only",
            alpha=0.0,
            naive_estimate=esc_drift_res["naive_estimate"],
            ipw_estimate=esc_drift_res["ipw_estimate"],
            true_fixture_risk=esc_drift_res["true_fixture_risk"],
            exact_ipw_brier_expectation=esc_drift_res["exact_ipw_brier_expectation"],
            true_accepted_error_conditional=esc_drift_res["true_accepted_error_conditional"],
            ipw_accepted_error_conditional=esc_drift_res["ipw_accepted_error_conditional"],
            naive_accepted_error_conditional=esc_drift_res["naive_accepted_error_conditional"],
            identifiability_status=esc_drift_res["identifiability_status"],
            accepted_drift_identifiable=esc_drift_res["accepted_drift_identifiable"],
            selection_bias=esc_drift_res["selection_bias"],
            passed=bool(
                esc_drift_res["identifiability_status"] == "unidentifiable"
                and esc_drift_res["ipw_accepted_error_conditional"] is None
                and esc_drift_res["naive_accepted_error_conditional"] is None
                and math.isclose(
                    float(esc_drift_res["naive_estimate"]),
                    float(esc_res["naive_estimate"]),
                    abs_tol=1e-15,
                )
            ),
        ),
    ]
    for alpha in ALPHA_GRID:
        aud_res = estimate_selective_risk(fixture_obs, "audited_deferral", alpha=alpha)
        mc = monte_carlo_audit_summary(fixture_obs, alpha=alpha, trials=400, seed=SEED)
        estimator_rows.append(
            dict(
                control_id=f"audited_deferral_alpha_{alpha:.2f}_ipw_unbiased",
                regime="audited_deferral",
                alpha=float(alpha),
                naive_estimate=aud_res["naive_estimate"],
                ipw_estimate=aud_res["ipw_estimate"],
                true_fixture_risk=aud_res["true_fixture_risk"],
                exact_ipw_brier_expectation=aud_res["exact_ipw_brier_expectation"],
                true_accepted_error_conditional=aud_res["true_accepted_error_conditional"],
                ipw_accepted_error_conditional=aud_res["ipw_accepted_error_conditional"],
                naive_accepted_error_conditional=aud_res["naive_accepted_error_conditional"],
                mc_ipw_brier_mean=mc["mc_ipw_brier_mean"],
                mc_ipw_accepted_error_mean=mc["mc_ipw_accepted_error_mean"],
                mc_naive_brier_mean=mc["mc_naive_brier_mean"],
                identifiability_status=aud_res["identifiability_status"],
                accepted_drift_identifiable=aud_res["accepted_drift_identifiable"],
                selection_bias=aud_res["selection_bias"],
                passed=bool(
                    aud_res["identifiability_status"] == "identifiable"
                    and aud_res["accepted_drift_identifiable"] is True
                    and math.isclose(
                        float(aud_res["exact_ipw_brier_expectation"]),
                        float(aud_res["true_fixture_risk"]),
                        abs_tol=1e-15,
                    )
                    and math.isclose(
                        float(aud_res["exact_ipw_accepted_error_expectation"]),
                        float(aud_res["true_accepted_error_conditional"]),
                        abs_tol=1e-15,
                    )
                    and abs(mc["mc_ipw_brier_mean"] - float(aud_res["true_fixture_risk"])) < 0.006
                    and abs(
                        mc["mc_ipw_accepted_error_mean"]
                        - float(aud_res["true_accepted_error_conditional"])
                    )
                    < 0.015
                    and (mc["mc_naive_brier_mean"] - float(aud_res["true_fixture_risk"])) > 0.05
                ),
            )
        )
    base_init = np.linspace(-2.0, 2.0, 300)[:, None] * np.arange(1, 8, dtype=float)[None, :]
    fitted = fit_feature_transform(base_init)
    int_res = classify_spline_support(np.zeros((5, 7), dtype=float), fitted)
    bound_raw = np.vstack([base_init[0], base_init[-1], base_init[-1] * 2.5, base_init[-1] * 0.75])
    bound_res = classify_spline_support(bound_raw, fitted)
    const_init = base_init.copy()
    const_init[:, 3] = 7.0
    const_fitted = fit_feature_transform(const_init)
    const_test = np.zeros((3, 7), dtype=float)
    const_test[:, 3] = 42.0
    const_res = classify_spline_support(const_test, const_fitted)
    scales = np.array([2.0, 1.5, 0.5, 3.0, 2.5, 1.25, 4.0])
    shifts = np.array([1.0, -2.0, 0.5, 3.0, -1.0, 0.0, 2.0])
    aff_fitted = fit_feature_transform(base_init * scales + shifts)
    mixed_raw = np.vstack([np.zeros((2, 7), dtype=float), bound_raw])
    mixed_orig = classify_spline_support(mixed_raw, fitted)
    mixed_aff = classify_spline_support(mixed_raw * scales + shifts, aff_fitted)
    mixed_mono = classify_spline_support(
        mixed_raw, fitted, monotone_transform="cubic_preserving_endpoints"
    )
    spline_rows: list[Json] = [
        dict(
            control_id="interior_dense_partition_of_unity",
            basis_columns=int_res["basis_columns"],
            max_partition_of_unity_error=int_res["max_partition_of_unity_error"],
            interior_dense_count=int_res["interior_dense_count"],
            boundary_sparse_or_clipped_count=int_res["boundary_sparse_or_clipped_count"],
            labels_read=False,
            passed=bool(
                int_res["basis_columns"] == 57
                and int_res["max_partition_of_unity_error"] < 1e-14
                and int_res["interior_dense_count"] == 5
                and int_res["boundary_sparse_or_clipped_count"] == 0
            ),
        ),
        dict(
            control_id="boundary_clipping_and_out_of_range_test_inputs",
            basis_columns=bound_res["basis_columns"],
            max_partition_of_unity_error=bound_res["max_partition_of_unity_error"],
            interior_dense_count=bound_res["interior_dense_count"],
            boundary_sparse_or_clipped_count=bound_res["boundary_sparse_or_clipped_count"],
            labels_read=False,
            passed=bool(
                bound_res["max_partition_of_unity_error"] < 1e-14
                and bound_res["boundary_sparse_or_clipped_count"] == 4
                and bound_res["rows"][2]["out_of_range_coordinate_count"] == 7
                and bound_res["rows"][2]["clipped_coordinate_count"] == 7
            ),
        ),
        dict(
            control_id="constant_feature_zero_mapping",
            basis_columns=const_res["basis_columns"],
            max_partition_of_unity_error=const_res["max_partition_of_unity_error"],
            interior_dense_count=const_res["interior_dense_count"],
            boundary_sparse_or_clipped_count=const_res["boundary_sparse_or_clipped_count"],
            labels_read=False,
            passed=bool(
                const_fitted["constant_coordinates"] == [3]
                and const_res["max_partition_of_unity_error"] < 1e-14
                and const_res["interior_dense_count"] == 3
                and all(r["scaled_features"][3] == 0.0 for r in const_res["rows"])
            ),
        ),
        dict(
            control_id="monotone_affine_and_odd_bijection_invariance",
            basis_columns=mixed_mono["basis_columns"],
            max_partition_of_unity_error=max(
                mixed_aff["max_partition_of_unity_error"],
                mixed_mono["max_partition_of_unity_error"],
            ),
            interior_dense_count=mixed_mono["interior_dense_count"],
            boundary_sparse_or_clipped_count=mixed_mono["boundary_sparse_or_clipped_count"],
            labels_read=False,
            passed=bool(
                [r["stratum"] for r in mixed_orig["rows"]]
                == [r["stratum"] for r in mixed_aff["rows"]]
                == [r["stratum"] for r in mixed_mono["rows"]]
                and mixed_aff["max_partition_of_unity_error"] < 1e-14
                and mixed_mono["max_partition_of_unity_error"] < 1e-14
            ),
        ),
    ]
    stream_qids = {q for s in rosters["stream_roster"] for q in s}
    future_qids = {q for s in rosters["stream_roster"] for q in s[1:]}
    supports: Json = {}
    for name, subset, stream_factor in [
        ("ALL", all_labels, 1),
        ("TRAIN", [m for m in all_labels if m["role"] == "TRAIN"], 1),
        ("CAL", [m for m in all_labels if m["role"] == "CAL"], 1),
        ("TEST", [m for m in all_labels if m["role"] == "TEST"], 1),
        (
            "initial100",
            [m for m in all_labels if m["qid"] in set(rosters["initial_group_roster"])],
            1,
        ),
        ("stream50", [m for m in all_labels if m["qid"] in stream_qids], 1),
        ("future40", [m for m in all_labels if m["qid"] in future_qids], 1),
        ("guard50", [m for m in all_labels if m["qid"] in set(rosters["guard_roster"])], 1),
        (
            "retention100",
            [m for m in all_labels if m["qid"] in set(rosters["retention_roster"])],
            10,
        ),
    ]:
        supports[name] = dict(
            independent_qids=len({m["qid"] for m in subset}),
            unique_responses=len(subset),
            repeated_stream_appearances=len(subset) * stream_factor,
        )
    supports["stream50"].update(
        audited_alpha_0_10_non_escalated_hits=sum(m["audit_alpha_0_10"] for m in mask_rows),
        audited_alpha_0_25_non_escalated_hits=sum(m["audit_alpha_0_25"] for m in mask_rows),
    )
    progress("controls_after", len(estimator_rows) + len(spline_rows), 0)
    return dict(
        estimator_control_rows=estimator_rows,
        spline_control_rows=spline_rows,
        support_counts=supports,
    )


def inspect(root: Path) -> Json:
    """Independently authenticate Exp9097 custody, Exp9126/9132 methods and Exp9211 authority."""
    progress("authenticate_before")
    result: Json = dict(
        checks=[],
        source_references=[],
        protocol_references={},
        rows=[],
        mask_rows=[],
        audit_seeds=[],
        rosters={},
        controls={},
        reused_health_receipt={},
    )
    checks = result["checks"]
    values: dict[str, Json] = {}
    try:
        for rel_path, pin in PINS.items():
            path = root / rel_path
            upstream = "exp" + Path(rel_path).name.split("_")[1]
            gate = check(
                path,
                "sha256",
                pin,
                ref(path)["sha256"] or "missing_file",
                "immutable_bytes",
                upstream,
            )
            checks.append(gate)
            result["source_references"].append(
                dict(
                    path=str(path),
                    producer_path=str(path),
                    sha256=pin,
                    imported_fields=["original_authority", "readiness", "protocol_references"],
                )
            )
            if gate["passed"]:
                values[Path(rel_path).name.split("_")[1]] = json.loads(path.read_bytes())
        if len(values) != len(PINS):
            return result
        for num, expected_fields in [
            ("9097", dict(source_ready_score=1, required_checks_passed=True)),
            ("9126", dict(methods_ready_score=1, required_checks_passed=True)),
            ("9132", dict(online_methods_ready_score=1, required_checks_passed=True)),
            ("9137", dict(reconciliation_ready_score=1, required_checks_passed=True)),
            ("9141", dict(contrast_protocol_ready_score=1, required_checks_passed=True)),
            (
                "9180",
                dict(
                    source_ready_score=1,
                    static_methods_ready_score=1,
                    online_methods_ready_score=1,
                    required_checks_passed=True,
                ),
            ),
            (
                "9211",
                dict(
                    target_identifiability_ready_score=1,
                    label_ambiguity_ready_score=1,
                    required_checks_passed=True,
                ),
            ),
        ]:
            rel_path = next(p for p in PINS if p.startswith(f"results/experiment_{num}_"))
            path = root / rel_path
            for field, expected in expected_fields.items():
                checks.append(
                    check(
                        path,
                        field,
                        expected,
                        values[num].get(field, "missing_field"),
                        "upstream_gate",
                        f"exp{num}",
                    )
                )
        if not all(c["passed"] for c in checks):
            return result
        cap_work_ref = values["9137"]["work_reference"]
        cap_work_path = Path(cap_work_ref["path"])
        checks.append(
            check(
                cap_work_path,
                "sha256",
                cap_work_ref["sha256"],
                ref(cap_work_path)["sha256"] or "missing_file",
                "capstone_work",
                "exp9137",
            )
        )
        saved_operands = json.loads(cap_work_path.read_bytes())["operands"]

        def bind(reference: Json, upstream: str, fields: list[str]) -> Path:
            original = str(reference["path"])
            retained = saved_operands.get(original, reference)
            target = Path(retained["path"])
            checks.append(
                check(
                    target,
                    "sha256",
                    reference["sha256"],
                    ref(target)["sha256"] or "missing_file",
                    "retained_producer_binding",
                    upstream,
                )
            )
            result["source_references"].append(
                dict(
                    path=str(target),
                    producer_path=reference.get("producer_path", original),
                    sha256=reference["sha256"],
                    imported_fields=fields,
                )
            )
            return target

        a9097 = values["9097"]
        for r in [*a9097["source_references"], *a9097["current_work_receipts"]]:
            if "path" in r and "sha256" in r:
                bind(r, "exp9097", ["human_custody_chain"])
        manifest_path = bind(
            a9097["source_manifest_reference"], "exp9097", ["release_files", "target_mapping"]
        )
        manifest = json.loads(manifest_path.read_bytes())
        for name, file_ref in manifest["release_files"].items():
            bound_file = bind(file_ref, "exp8389", [name])
            old.authenticate(name, dict(file_ref, path=str(bound_file)))
        for key, ref_obj in sorted(values["9180"]["protocol_references"].items()):
            bound_p = bind(ref_obj, "exp9180", [key])
            result["protocol_references"][key] = dict(
                ref(bound_p), producer_path=ref_obj.get("producer_path", ref_obj["path"])
            )
        for key in (
            "ambiguity_masks_reference",
            "target_rubric_reference",
            "label_ambiguity_protocol_reference",
            "reducer_reference",
        ):
            ref_9211 = values["9211"][key]
            bound_9211 = bind(ref_9211, "exp9211", [key])
            result["protocol_references"][key] = dict(
                ref(bound_9211), producer_path=ref_9211["path"]
            )
        for num, runner, label in [
            ("9126", static_runner, "static"),
            ("9132", online_runner, "online"),
        ]:
            m_work_path = bind(
                values[num]["work_reference"], f"exp{num}", ["method_work_and_receipts"]
            )
            m_work = json.loads(m_work_path.read_bytes())
            m_val = dict(
                checks=values[num]["validation_receipts"],
                owned_statement_coverage=values[num]["owned_statement_coverage"],
            )
            m_inputs = {
                r["path"]: bind(r, f"exp{num}", ["method_code_or_test"])
                for r in m_work["scope"]["code_test_hashes"]
            }
            with patch.object(
                runner.d,
                "ref",
                lambda p, mapping=m_inputs: dict(
                    ref(p), sha256=ref(mapping.get(str(p), p))["sha256"]
                ),
            ):
                receipt_ok = bool(runner.receipts_pass(m_work, m_val))
            checks.append(
                check(
                    m_work_path,
                    f"{label}_method_receipts_pass",
                    True,
                    receipt_ok,
                    "method_receipt_replay",
                    f"exp{num}",
                )
            )
        master_ref_raw = manifest["release_files"][old.MASTER]
        master_path = Path(saved_operands.get(master_ref_raw["path"], master_ref_raw)["path"])
        with master_path.open(encoding="utf-8-sig", newline="") as stream:
            master_rows = {
                row["item_id"]: dict(row, original_row=idx + 2)
                for idx, row in enumerate(csv.DictReader(stream))
            }
        eval_labels = json.loads(
            Path(result["protocol_references"]["evaluator_label_reference"]["path"]).read_bytes()
        )["rows"]
        inf_view = json.loads(
            Path(result["protocol_references"]["inference_view_reference"]["path"]).read_bytes()
        )["rows"]
        rosters = json.loads(
            Path(result["protocol_references"]["initial_roster_reference"]["path"]).read_bytes()
        )
        view_clean = len(inf_view) == 900 and all(
            set(r) == {"unit_id", "qid", "sibling", "question", "answer_text"}
            and "sha256:" + hashlib.sha256(r["question"].encode()).hexdigest()
            == "sha256:"
            + hashlib.sha256(master_rows[r["unit_id"]]["question"].encode()).hexdigest()
            and "sha256:" + hashlib.sha256(r["answer_text"].encode()).hexdigest()
            == "sha256:"
            + hashlib.sha256(master_rows[r["unit_id"]]["answer_text"].encode()).hexdigest()
            for r in inf_view
        )
        checks.append(
            check(
                Path(result["protocol_references"]["inference_view_reference"]["path"]),
                "label_free_inference_view_900",
                True,
                view_clean,
                "inference_view_separation",
                "exp9124",
            )
        )
        label_index = {str(r["unit_id"]): r for r in eval_labels}
        built = build_stream_deferral_masks(rosters["stream_roster"], label_index, master_rows)
        ctrl = controls(built["mask_rows"], eval_labels, rosters)
        checks.append(
            check(
                master_path,
                "all_estimator_and_spline_controls_passed",
                True,
                all(r["passed"] for r in ctrl["estimator_control_rows"])
                and all(r["passed"] for r in ctrl["spline_control_rows"]),
                "method_controls",
                TASK,
            )
        )
        if all(c["passed"] for c in checks):
            result.update(
                rows=built["rows"],
                mask_rows=built["mask_rows"],
                audit_seeds=built["audit_seeds"],
                rosters=rosters,
                controls=ctrl,
                reused_health_receipt=values["9211"].get("repository_health", {}),
            )
    except (OSError, ValueError, KeyError, TypeError) as error:
        checks.append(
            check(
                root,
                "authenticated_source_custody",
                "complete_valid_source",
                str(error),
                "source_custody_exception",
                TASK,
            )
        )
    progress("authenticate_after", len(result["rows"]), 0)
    return result
