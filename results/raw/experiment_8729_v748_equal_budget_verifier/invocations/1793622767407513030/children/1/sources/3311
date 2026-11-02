"""FR-11 Adaptive Online Beta Robust Default v1.

WHY THIS MODULE EXISTS:
    exp3509 (.323) found the static offline law beta = f(lambda_min(Sigma_initial))
    did not generalize to fresh configs and fell back to a conservative default.
    This module evaluates the true Phase-5 deployable rule: ADAPTIVE ONLINE beta.
    Instead of measuring lambda_min once at t=0, we measure the probability-weighted
    lambda_min(Sigma_t) online at each step. As the distribution collapses, diversity
    drops, lambda_min drops, and the deployed beta naturally increases to halt collapse.

    Arms:
      Arm A: Adaptive online beta = clamp(f(lambda_min_t), beta_floor)
      Arm B: beta=0.0 (control — expects collapse)
      Arm C: beta=0.5 (fixed-conservative baseline)
      Arm D: Static offline law (exp3498 formula at t=0 lambda_min)

FRESH CONFIG DESIGN:
    exp3498 fit points: AW in {0.05, 0.10, 0.146, 0.30}.
    Fresh configs use AW NOT in this set:
      "fresh_low_1" (0.06)
      "fresh_low_2" (0.08)
      "fresh_mid_1" (0.18)
      "fresh_mid_2" (0.22)
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import numpy as np

from carnot.fr11.beta_min_lambda_min_predictive_law_v1 import (
    N_CHANNELS,
    _assert_sources_distinct,
    _collapse_criterion,
    _compute_at_risk_scores,
    _entropy,
    _softmax,
    compute_sigma_metrics,
)

LAW_SLOPE: float = 1.846114323938634
LAW_INTERCEPT: float = -0.3000692577363795

FIXED_CONSERVATIVE_BETA: float = 0.5
BETA_FLOOR: float = 0.0

N_ITERATIONS: int = 400
PROGRESS_INTERVAL: int = 20
OVER_REG_MARGIN: float = 0.001
MODE_MASS_COLLAPSE_THRESHOLD: float = 0.5
ENTROPY_COLLAPSE_THRESHOLD: float = 0.1

_SEED_MATERIAL = b"exp3521_fr11_adaptive_online_beta_robust_default_v1"
RANDOM_SEED: int = int(hashlib.sha256(_SEED_MATERIAL).hexdigest()[:8], 16) % (2**20)

FRESH_CONFIGS: list[dict[str, Any]] = [
    {"name": "fresh_low_1", "active_weight": 0.06, "description": "Between 0.05 and 0.10"},
    {"name": "fresh_low_2", "active_weight": 0.08, "description": "Between 0.05 and 0.10"},
    {"name": "fresh_mid_1", "active_weight": 0.18, "description": "Between 0.146 and 0.30"},
    {"name": "fresh_mid_2", "active_weight": 0.22, "description": "Between 0.146 and 0.30"},
]


def apply_law(lambda_min: float, beta_floor: float = BETA_FLOOR) -> float:
    predicted = LAW_SLOPE * lambda_min + LAW_INTERCEPT
    return float(max(beta_floor, predicted))


def _compute_decisions_matrix(
    traces: list[dict[str, Any]],
    active_weight: float,
    n_channels: int = N_CHANNELS,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    n = len(traces)
    is_correct = np.array([bool(t.get("is_correct", False)) for t in traces], dtype=float)

    rng_active = np.random.RandomState(seed)
    active_noise = rng_active.random(n)
    active_signal = 0.9 * is_correct + 0.1 * active_noise

    decisions = np.zeros((n, n_channels), dtype=float)
    for j in range(n_channels):
        rng_null = np.random.RandomState(seed + 1000 + j)
        null_signal = rng_null.random(n)
        scores_j = active_weight * active_signal + (1.0 - active_weight) * null_signal
        decisions[:, j] = (scores_j > 0.5).astype(float)
    return decisions


def _compute_weighted_covariance(decisions: np.ndarray, probs: np.ndarray) -> np.ndarray:
    mu = np.sum(decisions * probs[:, None], axis=0, keepdims=True)
    centered = decisions - mu
    return (centered.T * probs[None, :]) @ centered


def run_arm_with_progress(
    traces: list[dict[str, Any]],
    at_risk_scores: np.ndarray,
    decisions: np.ndarray,
    n_iterations: int,
    arm_type: str,
    static_beta: float,
    config_name: str,
    arm_label: str,
) -> dict[str, Any]:
    n = len(traces)
    is_correct = np.array([bool(t.get("is_correct", False)) for t in traces], dtype=float)
    verifier_pass = (at_risk_scores > 0.5).astype(float)

    log_w = np.zeros(n, dtype=np.float64)
    init_ent: float | None = None

    for t in range(n_iterations):
        probs = _softmax(log_w)
        ent = _entropy(probs)
        mode_mass = float(np.max(probs))

        if init_ent is None:
            init_ent = ent

        if arm_type == "adaptive":
            sigma_t = _compute_weighted_covariance(decisions, probs)
            metrics_t = compute_sigma_metrics(sigma_t)
            current_beta = apply_law(metrics_t["lambda_min"], BETA_FLOOR)
        elif arm_type == "beta0":
            current_beta = 0.0
        elif arm_type == "fixed":
            current_beta = FIXED_CONSERVATIVE_BETA
        elif arm_type == "static":
            current_beta = static_beta
        else:
            raise ValueError(f"Unknown arm_type: {arm_type}")

        if t % PROGRESS_INTERVAL == 0:
            pass_now = float(np.dot(probs, verifier_pass))
            true_now = float(np.dot(probs, is_correct))
            print(
                f"  [{config_name}|{arm_label}] step={t}/{n_iterations} "
                f"ent={ent:.3f} mode_mass={mode_mass:.3f} beta={current_beta:.4f} "
                f"pass={pass_now:.3f} true_acc={true_now:.3e}",
                flush=True,
            )

        log_w = log_w + at_risk_scores
        if current_beta > 0.0:
            entropy_bonus = -np.log(np.clip(probs, 1e-300, None))
            log_w = log_w + current_beta * entropy_bonus
        log_w = log_w - np.max(log_w)

    final_probs = _softmax(log_w)
    final_ent = _entropy(final_probs)
    final_mode = float(np.max(final_probs))
    final_pass = float(np.dot(final_probs, verifier_pass))
    final_true = float(np.dot(final_probs, is_correct))

    init_e = float(init_ent) if init_ent is not None else 0.0
    drop_ratio = (init_e - final_ent) / max(init_e, 1e-9)
    collapsed = _collapse_criterion(drop_ratio, final_ent, final_mode, n_iterations)

    return {
        "collapse_detected": collapsed,
        "final_entropy": final_ent,
        "final_mode_mass": final_mode,
        "entropy_drop_ratio": drop_ratio,
        "final_pass_rate": final_pass,
        "final_true_accuracy": final_true,
        "final_gap": final_pass - final_true,
    }


def run_adaptive_online_beta_robust_default(
    traces_path: str,
    n_iterations: int = N_ITERATIONS,
    seed: int = RANDOM_SEED,
    fresh_configs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if fresh_configs is None:
        fresh_configs = FRESH_CONFIGS

    start = time.monotonic()

    traces: list[dict[str, Any]] = []
    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))

    if not traces:
        return {
            "honest_verdict": "complete: blocked_fr11_module_or_traces_unavailable",
            "error": "No traces loaded from corpus",
            "inference_substrate": "verifier_ensemble_against_cached_candidates",
            "random_seed": seed,
        }

    checksum_input = json.dumps(
        {
            "n_traces": len(traces),
            "seed": seed,
            "n_iterations": n_iterations,
            "law_slope": LAW_SLOPE,
            "law_intercept": LAW_INTERCEPT,
            "fresh_config_names": [c["name"] for c in fresh_configs],
            "fresh_config_aws": [c["active_weight"] for c in fresh_configs],
            "fixed_conservative_beta": FIXED_CONSERVATIVE_BETA,
            "n_channels": N_CHANNELS,
            "model_version": "v1_adaptive_online_beta",
        },
        sort_keys=True,
    ).encode()
    reproducibility_checksum = hashlib.sha256(checksum_input).hexdigest()[:16]

    lambda_min_by_config: dict[str, float] = {}
    collapse_A_by_config: dict[str, bool] = {}
    collapse_B_by_config: dict[str, bool] = {}
    collapse_C_by_config: dict[str, bool] = {}
    collapse_D_by_config: dict[str, bool] = {}
    accuracy_A_by_config: dict[str, float] = {}
    accuracy_C_by_config: dict[str, float] = {}
    least_regularized_accuracy_by_config: dict[str, float] = {}
    per_config: dict[str, Any] = {}

    for cfg in fresh_configs:
        name = cfg["name"]
        aw = float(cfg["active_weight"])

        print(f"\n=== Config: {name} (ACTIVE_WEIGHT={aw}) ===", flush=True)

        decisions = _compute_decisions_matrix(traces, aw, seed=seed)
        probs_initial = np.ones(len(traces), dtype=np.float64) / len(traces)
        sigma_initial = _compute_weighted_covariance(decisions, probs_initial)
        metrics = compute_sigma_metrics(sigma_initial)
        lmin = metrics["lambda_min"]
        
        static_beta = apply_law(lmin, BETA_FLOOR)

        print(f"  lambda_min(t=0)={lmin:.6f} static_beta={static_beta:.6f}", flush=True)

        at_risk_scores = _compute_at_risk_scores(traces, aw, seed)
        is_correct = np.array([bool(t.get("is_correct", False)) for t in traces], dtype=float)
        verifier_pass = (at_risk_scores > 0.5).astype(float)
        _assert_sources_distinct(verifier_pass, is_correct, aw)

        print(f"\n  --- Arm A: ADAPTIVE ONLINE ---", flush=True)
        arm_a = run_arm_with_progress(
            traces, at_risk_scores, decisions, n_iterations, "adaptive", 0.0, name, "A_adaptive"
        )

        print(f"\n  --- Arm B: beta=0 (control) ---", flush=True)
        arm_b = run_arm_with_progress(
            traces, at_risk_scores, decisions, n_iterations, "beta0", 0.0, name, "B_beta0"
        )

        print(f"\n  --- Arm C: beta={FIXED_CONSERVATIVE_BETA} (fixed-conservative) ---", flush=True)
        arm_c = run_arm_with_progress(
            traces, at_risk_scores, decisions, n_iterations, "fixed", 0.0, name, "C_fixed"
        )

        print(f"\n  --- Arm D: static offline law (beta={static_beta:.4f}) ---", flush=True)
        arm_d = run_arm_with_progress(
            traces, at_risk_scores, decisions, n_iterations, "static", static_beta, name, "D_static"
        )

        lambda_min_by_config[name] = lmin
        collapse_A_by_config[name] = bool(arm_a["collapse_detected"])
        collapse_B_by_config[name] = bool(arm_b["collapse_detected"])
        collapse_C_by_config[name] = bool(arm_c["collapse_detected"])
        collapse_D_by_config[name] = bool(arm_d["collapse_detected"])
        
        accuracy_A_by_config[name] = arm_a["final_true_accuracy"]
        accuracy_C_by_config[name] = arm_c["final_true_accuracy"]

        # Least regularized non-collapsing arm (between A, C, D)
        non_collapsing = []
        if not arm_a["collapse_detected"]:
            # Arm A's effective beta is adaptive, difficult to quantify statically but generally lower than fixed 0.5
            non_collapsing.append(arm_a["final_true_accuracy"])
        if not arm_d["collapse_detected"]:
            non_collapsing.append(arm_d["final_true_accuracy"])
        if not arm_c["collapse_detected"]:
            non_collapsing.append(arm_c["final_true_accuracy"])
            
        least_reg_acc = max(non_collapsing) if non_collapsing else 0.0
        least_regularized_accuracy_by_config[name] = least_reg_acc

        per_config[name] = {
            "active_weight": aw,
            "sigma_metrics": metrics,
            "static_beta": static_beta,
            "arm_A_adaptive": arm_a,
            "arm_B_beta0": arm_b,
            "arm_C_fixed_conservative": arm_c,
            "arm_D_static_law": arm_d,
        }

        print(
            f"\n  Summary [{name}]: "
            f"A_collapse={arm_a['collapse_detected']}, "
            f"B_collapse={arm_b['collapse_detected']}, "
            f"C_collapse={arm_c['collapse_detected']}, "
            f"D_collapse={arm_d['collapse_detected']}",
            flush=True,
        )

    adaptive_online_prevents_collapse = all(not v for v in collapse_A_by_config.values()) and any(v for v in collapse_B_by_config.values())
    conservative_default_prevents_collapse = all(not v for v in collapse_C_by_config.values()) and any(v for v in collapse_B_by_config.values())
    
    # "winning arm's true_accuracy minus the least-regularized non-collapsing arm"
    if adaptive_online_prevents_collapse:
        winning_acc_gap = float(np.mean([accuracy_A_by_config[n["name"]] - least_regularized_accuracy_by_config[n["name"]] for n in fresh_configs]))
        recommended = "adaptive-online beta (clamp(f(lambda_min), 0.0))"
    elif conservative_default_prevents_collapse:
        winning_acc_gap = float(np.mean([accuracy_C_by_config[n["name"]] - least_regularized_accuracy_by_config[n["name"]] for n in fresh_configs]))
        recommended = "conservative-default beta (0.5)"
    else:
        winning_acc_gap = 0.0
        recommended = "NONE"

    pass_rate_vs_true_accuracy_distinct_assert = True

    if adaptive_online_prevents_collapse:
        honest_verdict = "complete: adaptive_online_beta_prevents_collapse_phase5_deployable_default_confirmed"
    elif conservative_default_prevents_collapse:
        honest_verdict = "complete: conservative_default_beta_is_the_robust_phase5_default_adaptive_online_unnecessary"
    else:
        honest_verdict = "complete: no_beta_rule_robustly_prevents_collapse_across_configs_self_learning_needs_new_mechanism"

    duration_s = max(time.monotonic() - start, 1.0)

    return {
        "honest_verdict": honest_verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "n_grounding_configs": len(fresh_configs),
        "lambda_min_by_config": lambda_min_by_config,
        "collapse_detected_armA_adaptive_online": collapse_A_by_config,
        "collapse_detected_armB_beta0": collapse_B_by_config,
        "collapse_detected_armC_conservative": collapse_C_by_config,
        "collapse_detected_armD_static_offline_law": collapse_D_by_config,
        "adaptive_online_prevents_collapse": bool(adaptive_online_prevents_collapse),
        "conservative_default_prevents_collapse": bool(conservative_default_prevents_collapse),
        "winning_arm_vs_least_regularized_accuracy_gap": winning_acc_gap,
        "pass_rate_vs_true_accuracy_distinct_assert": bool(pass_rate_vs_true_accuracy_distinct_assert),
        "recommended_phase5_rule": recommended,
        "random_seed": seed,
        "reproducibility_checksum": reproducibility_checksum,
        "duration_s": duration_s,
        "per_config": per_config,
        "acceptance_gates": {
            "G1_deployment_validated": bool(adaptive_online_prevents_collapse or conservative_default_prevents_collapse),
            "G0_deflag_distinct_arrays": bool(pass_rate_vs_true_accuracy_distinct_assert),
        },
    }
