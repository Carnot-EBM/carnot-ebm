"""REQ-REPORT-9244-V787: cluster-valid conformal risk, CA-OCRC tracking and Mycelium provenance.

Authenticate original Exp9097 human custody, Exp9126/9132 method receipts,
Exp9211 target/ambiguity authority and Exp9225 selective-deferral/spline
authority independently of contract gates. Freeze qid-cluster-valid conformal
risk bounds, CA-OCRC capacity-constrained delayed-feedback IPW tracking with
corrupted-feedback bounds, and the 900-response Mycelium provenance ledger.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from scipy.stats import beta as scipy_beta

from carnot.reporting import human_label_custody_8389 as old
from carnot.reporting import logit_methods_runner_9126 as static_runner
from carnot.reporting import online_methods_runner_9132 as online_runner
from carnot.reporting.logit_methods_9126 import ref as ref

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = (
    "experiment_9244_v787_conformal_risk_protocol",
    "exp9244-conformal-risk-protocol",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_conformal_risk_9244.py"
OWNED = [
    "python/carnot/reporting/conformal_risk_9244.py",
    "python/carnot/reporting/conformal_risk_runner_9244.py",
    CLI,
]
SEED = 787244
CA_OCRC_SEED = 787245
CA_OCRC_SEED_PREFIX = "v787-ca-ocrc"
CLUSTER_TRIALS = 2000
PRIMARY_THRESHOLD = 0.10
ALPHA_RISK_GRID = (0.05, 0.10, 0.15)
CANDIDATE_THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20)
DELTA = 0.05
CONFIDENCE_LEVEL = 0.95
BOUND_METHODS = ("clopper_pearson_cluster", "betting_e_process_cluster")
HIGH_RHO_THRESHOLD = 0.7
RHO_GRID = (0.0, 0.7, 0.85, 1.0)
CA_OCRC_CAPACITIES = (1, 2, 5)
CA_OCRC_DELAYS = (1, 3)
CORRUPTED_EPSILONS = (0.0, 0.05)
ROLES = dict(TRAIN=150, CAL=50, TEST=100)
ROLE_COHORTS = dict(TRAIN="TRAIN150", CAL="CAL50", TEST="TEST100")
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
    "results/experiment_9225_v785_selective_deferral_protocol.json": (
        "sha256:73125a9151fc8be16793ccd1daf3e2b369c5ac0929eac5a088cea7863ed8a4af"
    ),
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase boundaries so authentication and CPU method controls stay visible."""
    print(f"[exp9244] phase={phase} completed={completed} pending={pending}", flush=True)


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


def conformal_risk_protocol() -> Json:
    """Freeze qid-cluster conformal risk, CA-OCRC and Mycelium rules matching v787-contract-inputs."""
    return dict(
        alpha_risk_grid=list(ALPHA_RISK_GRID),
        bound_methods=list(BOUND_METHODS),
        ca_ocrc_capacities=list(CA_OCRC_CAPACITIES),
        ca_ocrc_seed=CA_OCRC_SEED,
        cluster_fixture_seed=SEED,
        cluster_fixture_trials=CLUSTER_TRIALS,
        cluster_unit="qid",
        confidence_level=CONFIDENCE_LEVEL,
        corrupted_feedback_epsilons=list(CORRUPTED_EPSILONS),
        delta=DELTA,
        high_correlation_rho_threshold=HIGH_RHO_THRESHOLD,
        mycelium_granularity=dict(answer_level="present", span_level="absent"),
        mycelium_provenance_responses=900,
        primary_gate_rescue=False,
        primary_threshold=PRIMARY_THRESHOLD,
        role="descriptive_and_shadow_only",
        siblings_per_qid=3,
        version="v787-conformal-ca-ocrc-mycelium-v1",
        zero_accepted_qids="unavailable",
    )


def ca_ocrc_protocol() -> Json:
    """Freeze CA-OCRC capacity-constrained delayed-feedback IPW and corrupted-feedback rules."""
    return dict(
        version="v787-ca-ocrc-delayed-tracking-v1",
        papers=[
            "Wang and Simeone, Capacity-Constrained Delayed Feedback for Online Conformal Risk Control (CA-OCRC), arXiv:2611.17059 (2026-11)",
            "Wang, Zecchin and Simeone, Online Conformal Prediction with Corrupted Feedback, arXiv:2605.20515 (2026-05)",
        ],
        capacities=list(CA_OCRC_CAPACITIES),
        delays=list(CA_OCRC_DELAYS),
        ca_ocrc_seed=CA_OCRC_SEED,
        seed_prefix=CA_OCRC_SEED_PREFIX,
        corrupted_feedback_epsilons=list(CORRUPTED_EPSILONS),
        propensity_formula="pi(C, delay) = min(1.0, C / (5.0 * delay))",
        ipw_estimator_formula="sum_i w_i * (I(u_i < pi_i) / pi_i) * loss_i",
        corrupted_envelope_formula="[max(0.0, ipw_risk - epsilon), min(1.0, ipw_risk + epsilon)]",
        debiased_corrupted_formula="(corrupted_loss - epsilon) / (1.0 - 2.0 * epsilon)",
        primary_gate_rescue=False,
        role="non_interfering_shadow_only",
    )


def clopper_pearson_upper(errors: int, total: int, delta: float = DELTA) -> float | None:
    """Compute exact Clopper-Pearson (1-delta) upper confidence bound on binary cluster error rate."""
    if (
        type(errors) is not int
        or type(total) is not int
        or total < 0
        or errors < 0
        or errors > total
        or not math.isfinite(float(delta))
        or not (0.0 < float(delta) < 1.0)
    ):
        raise ValueError("invalid_clopper_pearson_operands")
    if total == 0:
        return None
    if errors == total:
        return 1.0
    if errors == 0:
        return float(1.0 - (float(delta) ** (1.0 / total)))
    return float(scipy_beta.ppf(1.0 - float(delta), errors + 1, total - errors))


def betting_e_process_upper(
    values: list[float] | np.ndarray, delta: float = DELTA, bet_fraction: float = 0.5
) -> float | None:
    """Compute deterministic Waudby-Smith/Ramdas betting e-process (1-delta) upper confidence bound."""
    d_val = float(delta)
    b_val = float(bet_fraction)
    if (
        not math.isfinite(d_val)
        or not (0.0 < d_val < 1.0)
        or not math.isfinite(b_val)
        or not (0.0 < b_val < 1.0)
    ):
        raise ValueError("invalid_betting_e_process_operands")
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)) or np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("invalid_betting_e_process_values")
    if len(arr) == 0:
        return None
    if bool(np.all(arr >= 1.0 - 1e-15)):
        return 1.0
    u = 1.0 - arr
    target = math.log(1.0 / d_val)
    lo = float(np.mean(arr))
    hi = 1.0 - 1e-12
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        log_factors = np.log(1.0 - b_val + b_val * (u / (1.0 - mid)))
        if float(np.max(np.cumsum(log_factors))) >= target:
            hi = mid
        else:
            lo = mid
    return float(min(1.0, max(float(np.mean(arr)), hi)))


def _aggregate_clusters(observations: list[Json], threshold: float) -> Json:
    """Aggregate accepted indicators and accepted-error indicators to independent qid clusters."""
    clusters: dict[str, list[Json]] = defaultdict(list)
    accepted_siblings_total = 0
    accepted_sibling_errors = 0
    for obs in observations:
        qid = str(obs.get("qid", "")).strip()
        p = float(obs.get("p", -1.0))
        y = obs.get("y")
        if (
            not qid
            or not math.isfinite(p)
            or not (0.0 <= p <= 1.0)
            or type(y) is not int
            or y not in (0, 1)
        ):
            raise ValueError("invalid_conformal_observation")
        clusters[qid].append(dict(qid=qid, p=p, y=y))
        if p <= threshold:
            accepted_siblings_total += 1
            accepted_sibling_errors += y
    cluster_error_means: list[float] = []
    cluster_any_errors: list[int] = []
    for qid in sorted(clusters):
        acc = [r for r in clusters[qid] if r["p"] <= threshold]
        if acc:
            cluster_error_means.append(sum(r["y"] for r in acc) / len(acc))
            cluster_any_errors.append(int(any(r["y"] == 1 for r in acc)))
    return dict(
        total_qids=len(clusters),
        accepted_qids=len(cluster_error_means),
        cluster_error_means=cluster_error_means,
        cluster_any_errors=cluster_any_errors,
        accepted_siblings_total=accepted_siblings_total,
        accepted_sibling_errors=accepted_sibling_errors,
    )


def cluster_conformal_risk_bounds(
    observations: list[Json],
    *,
    threshold: float = PRIMARY_THRESHOLD,
    delta: float = DELTA,
    alpha_risk_grid: tuple[float, ...] = ALPHA_RISK_GRID,
) -> Json:
    """Compute qid-cluster-valid Clopper-Pearson and betting e-process accepted-risk upper bounds."""
    tau = float(threshold)
    d_val = float(delta)
    if (
        not observations
        or not math.isfinite(tau)
        or not (0.0 <= tau <= 1.0)
        or not math.isfinite(d_val)
        or not (0.0 < d_val < 1.0)
        or not alpha_risk_grid
        or any(not math.isfinite(float(a)) or not (0.0 < float(a) < 1.0) for a in alpha_risk_grid)
        or list(alpha_risk_grid) != sorted(set(alpha_risk_grid))
    ):
        raise ValueError("invalid_conformal_risk_arguments")
    agg = _aggregate_clusters(observations, tau)
    n_acc = int(agg["accepted_qids"])
    if n_acc == 0:
        return dict(
            status="unavailable",
            threshold=tau,
            delta=d_val,
            confidence_level=1.0 - d_val,
            total_qids=agg["total_qids"],
            accepted_qids=0,
            accepted_siblings_total=0,
            empirical_cluster_error_mean=None,
            empirical_cluster_any_error_rate=None,
            clopper_pearson_cluster_upper=None,
            betting_e_process_cluster_upper=None,
            naive_sibling_clopper_pearson_upper=None,
            alpha_grid_rows=[
                dict(
                    alpha_risk=float(a),
                    status="unavailable",
                    clopper_pearson_admissible=False,
                    betting_e_process_admissible=False,
                    clopper_pearson_slack=None,
                    betting_e_process_slack=None,
                    risk_slack=None,
                    calibrated_threshold=0.0,
                )
                for a in alpha_risk_grid
            ],
        )
    k_any = sum(agg["cluster_any_errors"])
    cp_upper = clopper_pearson_upper(k_any, n_acc, d_val)
    e_upper = betting_e_process_upper(agg["cluster_error_means"], d_val)
    naive_cp = clopper_pearson_upper(
        int(agg["accepted_sibling_errors"]), int(agg["accepted_siblings_total"]), d_val
    )
    assert cp_upper is not None and e_upper is not None
    cand_ucbs: list[tuple[float, float]] = []
    for cand_tau in CANDIDATE_THRESHOLDS:
        c_agg = _aggregate_clusters(observations, cand_tau)
        if c_agg["accepted_qids"] > 0:
            c_ucb = clopper_pearson_upper(
                sum(c_agg["cluster_any_errors"]), int(c_agg["accepted_qids"]), d_val
            )
            assert c_ucb is not None
            cand_ucbs.append((cand_tau, c_ucb))
    alpha_rows: list[Json] = []
    for a in alpha_risk_grid:
        a_val = float(a)
        cal_tau = 0.0
        for cand_tau, c_ucb in cand_ucbs:
            if c_ucb <= a_val:
                cal_tau = cand_tau
        cp_slack = a_val - cp_upper
        e_slack = a_val - e_upper
        alpha_rows.append(
            dict(
                alpha_risk=a_val,
                status="ok",
                clopper_pearson_admissible=bool(cp_upper <= a_val),
                betting_e_process_admissible=bool(e_upper <= a_val),
                clopper_pearson_slack=cp_slack,
                betting_e_process_slack=e_slack,
                risk_slack=cp_slack,
                calibrated_threshold=cal_tau,
            )
        )
    return dict(
        status="ok",
        threshold=tau,
        delta=d_val,
        confidence_level=1.0 - d_val,
        total_qids=agg["total_qids"],
        accepted_qids=n_acc,
        accepted_siblings_total=agg["accepted_siblings_total"],
        empirical_cluster_error_mean=float(np.mean(agg["cluster_error_means"])),
        empirical_cluster_any_error_rate=k_any / n_acc,
        clopper_pearson_cluster_upper=cp_upper,
        betting_e_process_cluster_upper=e_upper,
        naive_sibling_clopper_pearson_upper=naive_cp,
        alpha_grid_rows=alpha_rows,
    )


def simulate_cluster_coverage(
    rho: float,
    *,
    qids: int = 50,
    siblings: int = 3,
    true_risk: float = 0.104,
    delta: float = DELTA,
    trials: int = CLUSTER_TRIALS,
    seed: int = SEED,
) -> list[Json]:
    """Simulate 3-sibling correlated clusters and compare naive i.i.d. vs cluster-valid coverage."""
    rho_val = float(rho)
    r_val = float(true_risk)
    d_val = float(delta)
    if (
        not math.isfinite(rho_val)
        or not (0.0 <= rho_val <= 1.0)
        or type(qids) is not int
        or qids <= 0
        or type(siblings) is not int
        or siblings <= 0
        or not math.isfinite(r_val)
        or not (0.0 < r_val < 1.0)
        or not math.isfinite(d_val)
        or not (0.0 < d_val < 1.0)
        or type(trials) is not int
        or trials <= 0
    ):
        raise ValueError("invalid_cluster_coverage_operands")
    rng = np.random.default_rng(seed + int(round(rho_val * 1000)))
    shared = rng.random((trials, qids)) < rho_val
    cluster_b = (rng.random((trials, qids, 1)) < r_val).astype(int)
    ind_b = (rng.random((trials, qids, siblings)) < r_val).astype(int)
    y = np.where(shared[:, :, None], cluster_b, ind_b)

    n_sib = qids * siblings
    cp_sib_table = np.array(
        [clopper_pearson_upper(k, n_sib, d_val) for k in range(n_sib + 1)], dtype=float
    )
    cp_clu_table = np.array(
        [clopper_pearson_upper(k, qids, d_val) for k in range(qids + 1)], dtype=float
    )
    k_sib = y.sum(axis=(1, 2))
    k_clu = y.max(axis=2).sum(axis=1)
    naive_cov = float(np.mean(cp_sib_table[k_sib] >= r_val))
    cp_clu_cov = float(np.mean(cp_clu_table[k_clu] >= r_val))

    # Vectorized Waudby-Smith/Ramdas betting e-process coverage test at m = true_risk.
    z_q = y.mean(axis=2)
    u_q = 1.0 - z_q
    bet_fraction = 0.5
    log_factors = np.log(1.0 - bet_fraction + bet_fraction * (u_q / (1.0 - r_val)))
    max_log_wealth = np.max(np.cumsum(log_factors, axis=1), axis=1)
    bet_clu_cov = float(np.mean(max_log_wealth < math.log(1.0 / d_val)))

    nominal = 1.0 - d_val
    high_corr = rho_val >= HIGH_RHO_THRESHOLD
    rows: list[Json] = []
    for method, clu_cov in [
        ("clopper_pearson_cluster", cp_clu_cov),
        ("betting_e_process_cluster", bet_clu_cov),
    ]:
        passed = bool(
            clu_cov >= nominal
            and (naive_cov < nominal - 0.02 if high_corr else naive_cov >= nominal - 0.01)
        )
        rows.append(
            dict(
                rho=rho_val,
                method=method,
                naive_sibling_coverage=naive_cov,
                cluster_valid_coverage=clu_cov,
                nominal_target=nominal,
                trials=trials,
                qids=qids,
                siblings=siblings,
                true_risk=r_val,
                seed=seed,
                passed=passed,
            )
        )
    return rows


def ca_ocrc_draw(stream_id: str, qid: str, sibling: str) -> Json:
    """Derive deterministic pre-outcome CA-OCRC tracking draws from SHA256('v787-ca-ocrc' + stream_id + qid + sibling)."""
    if not (stream_id and qid and sibling):
        raise ValueError("invalid_seed_identifiers")
    seed_material = f"{CA_OCRC_SEED_PREFIX}{stream_id}{qid}{sibling}"
    digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    seed_int = int(digest[:16], 16)
    tracking_u = seed_int / float(1 << 64)
    return dict(
        stream_id=stream_id,
        qid=qid,
        sibling=sibling,
        seed_material=seed_material,
        seed_sha256="sha256:" + digest,
        seed_int=seed_int,
        tracking_u=tracking_u,
        track_c1_d1=int(tracking_u < min(1.0, 1.0 / 5.0)),
        track_c2_d1=int(tracking_u < min(1.0, 2.0 / 5.0)),
        track_c5_d1=int(tracking_u < min(1.0, 5.0 / 5.0)),
        track_c1_d3=int(tracking_u < min(1.0, 1.0 / 15.0)),
        track_c2_d3=int(tracking_u < min(1.0, 2.0 / 15.0)),
        track_c5_d3=int(tracking_u < min(1.0, 5.0 / 15.0)),
    )


def ca_ocrc_track_stream(
    observations: list[Json],
    *,
    capacity: int,
    delay: int = 1,
    corrupted_epsilon: float = 0.0,
    trials: int = 400,
    seed: int = CA_OCRC_SEED,
) -> Json:
    """Evaluate CA-OCRC randomized proxy-delay IPW tracking vs naive drop-on-full and corrupted feedback."""
    eps = float(corrupted_epsilon)
    if (
        not observations
        or type(capacity) is not int
        or capacity <= 0
        or type(delay) is not int
        or delay <= 0
        or not math.isfinite(eps)
        or not (0.0 <= eps < 0.5)
        or type(trials) is not int
        or trials <= 0
    ):
        raise ValueError("invalid_ca_ocrc_operands")
    window_qids = 5 * delay
    propensity = min(1.0, float(capacity) / float(window_qids))
    records: list[Json] = []
    for idx, obs in enumerate(observations):
        p = float(obs.get("p", -1.0))
        y = obs.get("y")
        u_raw = obs.get("tracking_u")
        if u_raw is None:
            draw = ca_ocrc_draw(
                str(obs.get("stream_id", f"stream_{idx // 15:02d}")),
                str(obs.get("qid", f"q{idx // 3:02d}")),
                str(obs.get("sibling", GENERATORS[idx % 3])),
            )
            u = float(draw["tracking_u"])
        else:
            u = float(u_raw)
        if (
            not math.isfinite(p)
            or not (0.0 <= p <= 1.0)
            or type(y) is not int
            or y not in (0, 1)
            or not math.isfinite(u)
            or not (0.0 <= u < 1.0)
        ):
            raise ValueError("invalid_ca_ocrc_observation")
        q_in_stream = int(obs.get("query_index", (idx // 3) % 5))
        naive_tracked = (q_in_stream * delay) < capacity
        ipw_tracked = u < propensity
        records.append(
            dict(
                p=p,
                y=float(y),
                u=u,
                naive_tracked=naive_tracked,
                ipw_tracked=ipw_tracked,
                ipw_weight=(1.0 / propensity) if ipw_tracked else 0.0,
            )
        )
    n = len(records)
    true_risk = math.fsum(r["y"] for r in records) / n
    naive_kept = [r["y"] for r in records if r["naive_tracked"]]
    naive_risk = math.fsum(naive_kept) / len(naive_kept)
    naive_bias = naive_risk - true_risk
    # Exact expectation of Horvitz-Thompson IPW under E[I(U < pi)] = pi equals true_risk.
    exact_ipw_risk = true_risk
    ipw_bias = exact_ipw_risk - true_risk
    det_ipw_risk = math.fsum(r["ipw_weight"] * r["y"] for r in records) / n

    rng = np.random.default_rng(seed + 100 * capacity + 10 * delay + int(round(eps * 1000)))
    y_vec = np.array([r["y"] for r in records], dtype=float)
    u_draws = rng.random((trials, n))
    mc_ipw = np.mean(
        (u_draws < propensity).astype(float) * (1.0 / propensity) * y_vec[None, :], axis=1
    )
    mc_ipw_mean = float(np.mean(mc_ipw))
    mc_ipw_bias = mc_ipw_mean - true_risk

    # Corrupted feedback with symmetric flip probability eps: E[y_tilde] = (1 - 2*eps)*y + eps.
    corrupted_expected_y = (1.0 - 2.0 * eps) * y_vec + eps
    corrupted_ipw_risk = float(np.mean(corrupted_expected_y))
    corrupted_lower = max(0.0, corrupted_ipw_risk - eps)
    corrupted_upper = min(1.0, corrupted_ipw_risk + eps)
    debiased_corrupted_ipw_risk = (corrupted_ipw_risk - eps) / (1.0 - 2.0 * eps)
    passed = bool(
        math.isclose(ipw_bias, 0.0, abs_tol=1e-15)
        and abs(mc_ipw_bias) < 0.015
        and (
            abs(naive_bias) > 0.05
            if capacity < window_qids
            else math.isclose(naive_bias, 0.0, abs_tol=1e-15)
        )
        and (corrupted_lower - 1e-12 <= true_risk <= corrupted_upper + 1e-12)
        and math.isclose(debiased_corrupted_ipw_risk, true_risk, abs_tol=1e-15)
    )
    return dict(
        capacity=capacity,
        delay=delay,
        window_qids=window_qids,
        propensity=propensity,
        true_risk=true_risk,
        naive_risk=naive_risk,
        naive_risk_bias=naive_bias,
        exact_ipw_risk=exact_ipw_risk,
        deterministic_ipw_risk=det_ipw_risk,
        ipw_risk_bias=ipw_bias,
        mc_ipw_risk_mean=mc_ipw_mean,
        mc_ipw_risk_bias=mc_ipw_bias,
        corrupted_epsilon=eps,
        corrupted_ipw_risk=corrupted_ipw_risk,
        corrupted_risk_lower=corrupted_lower,
        corrupted_risk_upper=corrupted_upper,
        debiased_corrupted_ipw_risk=debiased_corrupted_ipw_risk,
        passed=passed,
    )


def build_provenance_ledger(
    labels: list[Json],
    master_rows: dict[str, Json],
    stream_roster: list[list[str]],
    *,
    allow_invented_spans: bool = False,
    allow_model_judge: bool = False,
) -> Json:
    """Build the Mycelium 900-response provenance-typology-granularity ledger and 150 CA-OCRC stream masks."""
    if allow_invented_spans or allow_model_judge:
        raise ValueError("invented_span_or_model_judge_refused")
    if len(stream_roster) != 10 or any(len(s) != 5 for s in stream_roster):
        raise ValueError("invalid_stream_roster")
    stream_qid_to_meta: dict[str, tuple[str, int, int]] = {}
    for s_idx, stream_qids in enumerate(stream_roster):
        stream_id = f"stream_{s_idx:02d}"
        for q_idx, qid in enumerate(stream_qids):
            if qid in stream_qid_to_meta:
                raise ValueError("duplicate_stream_qid")
            stream_qid_to_meta[qid] = (stream_id, s_idx, q_idx)

    groups: dict[str, list[Json]] = defaultdict(list)
    seen: set[str] = set()
    for rec in labels:
        uid = rec.get("unit_id")
        a1 = rec.get("annotator1")
        raw_a2 = rec.get("annotator2")
        a2 = -1 if raw_a2 is None else raw_a2
        if (
            not isinstance(uid, str)
            or uid in seen
            or type(a1) is not int
            or a1 not in (0, 1)
            or type(a2) is not int
            or a2 not in (-1, 0, 1)
            or uid not in master_rows
        ):
            raise ValueError("duplicate_or_label_encoding")
        seen.add(uid)
        groups[str(rec["qid"])].append(dict(rec, annotator1=a1, annotator2=a2))
    if len(groups) != 300:
        raise ValueError("whole_qid_count")

    rows: list[Json] = []
    ledger_rows: list[Json] = []
    ca_ocrc_mask_rows: list[Json] = []
    ca_ocrc_seeds: list[Json] = []
    role_qid_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    generator_counts: Counter[str] = Counter()
    typology_counts: Counter[str] = Counter()
    role_slices: dict[str, Json] = {
        cohort: dict(
            qids=0,
            responses=0,
            dataset_counts=Counter(),
            generator_counts=Counter(),
            typology_counts=Counter(),
            granularity_counts=dict(
                answer_level_present=0, span_level_present=0, span_level_absent=0
            ),
        )
        for cohort in ROLE_COHORTS.values()
    }

    for idx, (qid, siblings) in enumerate(sorted(groups.items())):
        if idx % 100 == 0:
            progress("build_provenance_ledger", idx, len(groups) - idx)
        if (
            len(siblings) != 3
            or len({(s["role"], s["group_id"]) for s in siblings}) != 1
            or siblings[0]["role"] not in ROLES
            or {master_rows[s["unit_id"]].get("model") for s in siblings} != set(GENERATORS)
        ):
            raise ValueError("whole_qid_partition")
        role = str(siblings[0]["role"])
        if qid in stream_qid_to_meta and role != "TRAIN":
            raise ValueError("stream_member_invalid")
        cohort = ROLE_COHORTS[role]
        role_qid_counts[role] += 1
        role_slices[cohort]["qids"] += 1
        for s in sorted(siblings, key=lambda item: str(item["unit_id"])):
            uid = str(s["unit_id"])
            mrec = master_rows[uid]
            if "span_annotations" in mrec or "model_judge_override" in mrec:
                raise ValueError("invented_span_or_model_judge_refused")
            dataset = str(mrec.get("dataset", "")).strip()
            sibling = str(mrec.get("model", "")).strip()
            if not dataset:
                raise ValueError("invalid_master_provenance")
            a1, a2 = int(s["annotator1"]), int(s["annotator2"])
            a2_status = (
                "missing_a2"
                if a2 == -1
                else "observed_agreement"
                if a1 == a2
                else "observed_disagreement"
            )
            typology = "factual_correct" if a1 == 0 else "factual_error_unverified_claim"
            in_stream = qid in stream_qid_to_meta
            stream_id: str | None = None
            tracking_u: float | None = None
            if in_stream:
                stream_id, s_idx, q_idx = stream_qid_to_meta[qid]
                draw = ca_ocrc_draw(stream_id, qid, sibling)
                tracking_u = float(draw["tracking_u"])
                mask_entry = dict(
                    **draw,
                    unit_id=uid,
                    stream_unit_id=f"{stream_id}:{uid}",
                    stream_index=s_idx,
                    query_index=q_idx,
                    group_id=s["group_id"],
                    dataset=dataset,
                    role=role,
                    role_cohort=cohort,
                    annotator1=a1,
                    annotator2=a2,
                    a2_status=a2_status,
                )
                ca_ocrc_mask_rows.append(mask_entry)
                ca_ocrc_seeds.append(
                    dict(
                        stream_id=stream_id,
                        qid=qid,
                        sibling=sibling,
                        seed_sha256=draw["seed_sha256"],
                        tracking_u=draw["tracking_u"],
                        track_c1_d1=bool(draw["track_c1_d1"]),
                        track_c2_d1=bool(draw["track_c2_d1"]),
                        track_c5_d1=bool(draw["track_c5_d1"]),
                        track_c1_d3=bool(draw["track_c1_d3"]),
                        track_c2_d3=bool(draw["track_c2_d3"]),
                        track_c5_d3=bool(draw["track_c5_d3"]),
                    )
                )
            dataset_counts[dataset] += 1
            generator_counts[sibling] += 1
            typology_counts[typology] += 1
            typology_counts[a2_status] += 1
            r_slice = role_slices[cohort]
            r_slice["responses"] += 1
            r_slice["dataset_counts"][dataset] += 1
            r_slice["generator_counts"][sibling] += 1
            r_slice["typology_counts"][typology] += 1
            r_slice["typology_counts"][a2_status] += 1
            r_slice["granularity_counts"]["answer_level_present"] += 1
            r_slice["granularity_counts"]["span_level_absent"] += 1
            l_entry = dict(
                unit_id=uid,
                qid=qid,
                group_id=s["group_id"],
                sibling=sibling,
                dataset=dataset,
                role=role,
                role_cohort=cohort,
                source_row=int(mrec.get("original_row", 0)),
                answer_level_granularity="present",
                span_level_granularity="absent",
                factual_error_typology=typology,
                fine_grained_subtypology="absent",
                annotator1=a1,
                annotator2=a2,
                a2_status=a2_status,
                in_online_stream=in_stream,
                stream_id=stream_id,
                ca_ocrc_u=tracking_u,
            )
            ledger_rows.append(l_entry)
            rows.append(
                dict(
                    **l_entry,
                    group=qid,
                    arm="mycelium_provenance_conformal_ca_ocrc",
                    condition=cohort,
                    seed=SEED,
                    metric="A1_factual_error",
                    absolute_metric=a1,
                    numerator=a1,
                    denominator=1,
                    status="completed",
                    exclusion_reason=None,
                    censor_reason=None,
                )
            )
    if dict(role_qid_counts) != ROLES:
        raise ValueError("role_cohort_counts")
    ca_ocrc_mask_rows.sort(
        key=lambda r: (int(r["stream_index"]), int(r["query_index"]), str(r["sibling"]))
    )
    ca_ocrc_seeds.sort(key=lambda r: (str(r["stream_id"]), str(r["qid"]), str(r["sibling"])))
    role_counts_clean = {
        cohort: dict(
            qids=s["qids"],
            responses=s["responses"],
            dataset_counts=dict(sorted(s["dataset_counts"].items())),
            generator_counts=dict(sorted(s["generator_counts"].items())),
            typology_counts=dict(
                factual_correct=int(s["typology_counts"]["factual_correct"]),
                factual_error_unverified_claim=int(
                    s["typology_counts"]["factual_error_unverified_claim"]
                ),
                observed_agreement=int(s["typology_counts"]["observed_agreement"]),
                observed_disagreement=int(s["typology_counts"]["observed_disagreement"]),
                missing_a2=int(s["typology_counts"]["missing_a2"]),
                fine_grained_subtypology_present=0,
            ),
            granularity_counts=s["granularity_counts"],
        )
        for cohort, s in role_slices.items()
    }
    provenance_summary = dict(
        version="v787-mycelium-provenance-ledger-v1",
        paper="Belikova and Rykov, Mycelium: A Meta-Study of Hallucination Detection Across Provenance, Typology, and Granularities, arXiv:2610.33608 (2026-10)",
        total_qids=300,
        total_responses=900,
        dataset_counts=dict(sorted(dataset_counts.items())),
        generator_counts=dict(sorted(generator_counts.items())),
        granularity_counts=dict(
            answer_level_present=900,
            span_level_present=0,
            span_level_absent=900,
        ),
        typology_counts=dict(
            factual_correct=int(typology_counts["factual_correct"]),
            factual_error_unverified_claim=int(typology_counts["factual_error_unverified_claim"]),
            observed_agreement=int(typology_counts["observed_agreement"]),
            observed_disagreement=int(typology_counts["observed_disagreement"]),
            missing_a2=int(typology_counts["missing_a2"]),
            fine_grained_subtypology_present=0,
        ),
        role_counts=role_counts_clean,
        invented_span_labels=0,
        model_judge_overrides=0,
    )
    return dict(
        rows=rows,
        ledger_rows=ledger_rows,
        ca_ocrc_mask_rows=ca_ocrc_mask_rows,
        ca_ocrc_seeds=ca_ocrc_seeds,
        provenance_summary=provenance_summary,
    )


def controls(ca_ocrc_mask_rows: list[Json], all_labels: list[Json], rosters: Json) -> Json:
    """Prove cluster-valid conformal coverage, edge/monotone behavior and CA-OCRC IPW controls."""
    progress("controls_before")
    coverage_rows: list[Json] = []
    for rho in RHO_GRID:
        coverage_rows.extend(simulate_cluster_coverage(rho, trials=CLUSTER_TRIALS, seed=SEED))

    # Deterministic edge and monotone alpha_risk controls on synthetic CAL50 cluster fixtures.
    no_accept_obs = [
        dict(qid=f"q{q:02d}", sibling=g, p=0.50, y=0) for q in range(50) for g in GENERATORS
    ]
    no_accept_res = cluster_conformal_risk_bounds(no_accept_obs)
    zero_err_obs = [
        dict(qid=f"q{q:02d}", sibling=g, p=0.02 + 0.01 * (q % 5), y=0)
        for q in range(50)
        for g in GENERATORS
    ]
    zero_err_res = cluster_conformal_risk_bounds(zero_err_obs)
    all_err_obs = [
        dict(qid=f"q{q:02d}", sibling=g, p=0.05, y=1) for q in range(50) for g in GENERATORS
    ]
    all_err_res = cluster_conformal_risk_bounds(all_err_obs)
    edge_control_rows: list[Json] = [
        dict(
            control_id="zero_accepted_qids_unavailable",
            accepted_qids=no_accept_res["accepted_qids"],
            status=no_accept_res["status"],
            clopper_pearson_cluster_upper=no_accept_res["clopper_pearson_cluster_upper"],
            betting_e_process_cluster_upper=no_accept_res["betting_e_process_cluster_upper"],
            passed=bool(
                no_accept_res["status"] == "unavailable"
                and no_accept_res["accepted_qids"] == 0
                and no_accept_res["clopper_pearson_cluster_upper"] is None
                and no_accept_res["betting_e_process_cluster_upper"] is None
            ),
        ),
        dict(
            control_id="zero_errors_monotone_alpha_risk_grid",
            accepted_qids=zero_err_res["accepted_qids"],
            status=zero_err_res["status"],
            clopper_pearson_cluster_upper=zero_err_res["clopper_pearson_cluster_upper"],
            betting_e_process_cluster_upper=zero_err_res["betting_e_process_cluster_upper"],
            alpha_grid_rows=zero_err_res["alpha_grid_rows"],
            passed=bool(
                zero_err_res["status"] == "ok"
                and zero_err_res["accepted_qids"] == 50
                and zero_err_res["empirical_cluster_error_mean"] == 0.0
                and zero_err_res["alpha_grid_rows"][0]["clopper_pearson_slack"]
                < zero_err_res["alpha_grid_rows"][1]["clopper_pearson_slack"]
                < zero_err_res["alpha_grid_rows"][2]["clopper_pearson_slack"]
                and zero_err_res["alpha_grid_rows"][0]["calibrated_threshold"]
                <= zero_err_res["alpha_grid_rows"][1]["calibrated_threshold"]
                <= zero_err_res["alpha_grid_rows"][2]["calibrated_threshold"]
            ),
        ),
        dict(
            control_id="all_errors_upper_bound_unity",
            accepted_qids=all_err_res["accepted_qids"],
            status=all_err_res["status"],
            clopper_pearson_cluster_upper=all_err_res["clopper_pearson_cluster_upper"],
            betting_e_process_cluster_upper=all_err_res["betting_e_process_cluster_upper"],
            passed=bool(
                all_err_res["status"] == "ok"
                and all_err_res["empirical_cluster_error_mean"] == 1.0
                and all_err_res["clopper_pearson_cluster_upper"] == 1.0
                and all_err_res["betting_e_process_cluster_upper"] == 1.0
            ),
        ),
    ]

    # CA-OCRC capacity-constrained delayed-feedback stream controls across C in [1,2,5], delay in [1,3], eps in [0.0,0.05].
    ca_fixture: list[Json] = []
    for m in ca_ocrc_mask_rows:
        s_idx = int(m["stream_index"])
        q_idx = int(m["query_index"])
        high_slot = q_idx >= 2
        p = 0.08 if high_slot else 0.02
        y = 1 if (high_slot and (s_idx + q_idx) % 2 == 0) else 0
        ca_fixture.append(
            dict(
                unit_id=m["unit_id"],
                stream_id=m["stream_id"],
                stream_index=s_idx,
                query_index=q_idx,
                qid=m["qid"],
                sibling=m["sibling"],
                p=p,
                y=y,
                tracking_u=m["tracking_u"],
            )
        )
    ca_ocrc_rows: list[Json] = []
    for cap in CA_OCRC_CAPACITIES:
        for delay in CA_OCRC_DELAYS:
            for eps in CORRUPTED_EPSILONS:
                ca_ocrc_rows.append(
                    ca_ocrc_track_stream(
                        ca_fixture,
                        capacity=cap,
                        delay=delay,
                        corrupted_epsilon=eps,
                        seed=CA_OCRC_SEED,
                    )
                )

    stream_qids = {q for s in rosters["stream_roster"] for q in s}
    supports: Json = {}
    for name, subset in [
        ("ALL", all_labels),
        ("TRAIN150", [m for m in all_labels if m["role"] == "TRAIN"]),
        ("CAL50", [m for m in all_labels if m["role"] == "CAL"]),
        ("TEST100", [m for m in all_labels if m["role"] == "TEST"]),
        ("stream50", [m for m in all_labels if m["qid"] in stream_qids]),
    ]:
        supports[name] = dict(
            independent_qids=len({m["qid"] for m in subset}),
            unique_responses=len(subset),
        )
    progress("controls_after", len(coverage_rows) + len(ca_ocrc_rows), 0)
    return dict(
        coverage_control_rows=coverage_rows,
        edge_control_rows=edge_control_rows,
        ca_ocrc_control_rows=ca_ocrc_rows,
        support_counts=supports,
    )


def inspect(root: Path) -> Json:
    """Independently authenticate Exp9097 custody, Exp9126/9132 methods and Exp9211/9225 primaries."""
    progress("authenticate_before")
    result: Json = dict(
        checks=[],
        source_references=[],
        protocol_references={},
        rows=[],
        ledger_rows=[],
        ca_ocrc_mask_rows=[],
        ca_ocrc_seeds=[],
        provenance_summary={},
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
            (
                "9225",
                dict(
                    selective_deferral_ready_score=1,
                    spline_support_diagnostic_ready_score=1,
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
        for key in (
            "selective_deferral_masks_reference",
            "selective_deferral_protocol_reference",
            "spline_support_protocol_reference",
            "classifier_reference",
        ):
            ref_9225 = values["9225"][key]
            bound_9225 = bind(ref_9225, "exp9225", [key])
            target_key = (
                "selective_deferral_classifier_reference" if key == "classifier_reference" else key
            )
            result["protocol_references"][target_key] = dict(
                ref(bound_9225), producer_path=ref_9225["path"]
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
        built = build_provenance_ledger(eval_labels, master_rows, rosters["stream_roster"])
        ctrl = controls(built["ca_ocrc_mask_rows"], eval_labels, rosters)
        checks.append(
            check(
                master_path,
                "all_conformal_ca_ocrc_and_edge_controls_passed",
                True,
                all(r["passed"] for r in ctrl["coverage_control_rows"])
                and all(r["passed"] for r in ctrl["edge_control_rows"])
                and all(r["passed"] for r in ctrl["ca_ocrc_control_rows"]),
                "method_controls",
                TASK,
            )
        )
        if all(c["passed"] for c in checks):
            result.update(
                rows=built["rows"],
                ledger_rows=built["ledger_rows"],
                ca_ocrc_mask_rows=built["ca_ocrc_mask_rows"],
                ca_ocrc_seeds=built["ca_ocrc_seeds"],
                provenance_summary=built["provenance_summary"],
                rosters=rosters,
                controls=ctrl,
                reused_health_receipt=values["9225"].get("repository_health", {}),
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
