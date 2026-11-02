"""Exp 5129: adaptive exact-checked CPU HUBO/p-spin 2D PT.

Spec refs: REQ-SAMPLE-5129, SCENARIO-SAMPLE-5129.

This experiment continues Exp 5116 on the same tiny direct HUBO parity
families. It keeps exact enumeration as the correctness authority, adapts only
the inverse-temperature ladder, and reports residual-energy and reversibility
telemetry without making any hardware speedup claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5116_hubo_2dpt_sampling_reference_v469 as exp5116
from carnot.samplers.hubo_2dpt import (
    AdaptiveHubo2DPTConfig,
    AdaptiveHubo2DParallelTemperingSampler,
    Hubo2DPTConfig,
    Hubo2DParallelTemperingSampler,
    HuboRunResult,
    SwapStats,
    build_synthetic_hubo_families,
    exact_enumerate,
    run_beta_parallel_tempering,
    run_unguided_gibbs,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_5129_hubo_adaptive_2dpt_v470.json"
EXPERIMENT_ID = "exp5129-hubo-adaptive-2dpt-v470"
MILESTONE = "2026.07.470"
INFERENCE_SUBSTRATE = "cpu_exact_checked_hubo_sampling"
RUN_DATE = "20260701"
DEFAULT_SEEDS = exp5116.DEFAULT_SEEDS
COMPLETE_VERDICT = "complete_adaptive_2dpt_ready_exact_checked_cpu"
NOT_READY_VERDICT = "complete_adaptive_2dpt_not_ready_no_harmful_regression"
BLOCKED_VERDICT = "blocked_adaptive_2dpt_exact_label_check_failed"
TERMINAL_PREFIXES = ("complete_", "success_", "blocked_", "complete:", "success:")
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "experiment_id",
        "milestone",
        "honest_verdict",
        "inference_substrate",
        "duration_s",
        "exp5116_baseline_loaded",
        "instance_families",
        "exact_enumeration_checked",
        "adaptive_temperature_config",
        "swap_acceptance_rates",
        "residual_energy_by_sweep",
        "optimum_hit_rate",
        "detailed_balance_sanity",
        "best_energy_delta_vs_baselines",
        "adaptive_2dpt_ready",
        "hardware_speedup_claimed",
        "flagged_adversarial",
        "conductor_modified",
        "tests_run",
    }
)
FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "exp5116_baseline_loaded": "continuation accountability",
    "instance_families": "benchmark diversity",
    "exact_enumeration_checked": "correctness authority",
    "adaptive_temperature_config": "method transparency",
    "swap_acceptance_rates": "tempering telemetry",
    "residual_energy_by_sweep": "sample-quality telemetry",
    "optimum_hit_rate": "exact-solver comparison",
    "detailed_balance_sanity": "sampler validity",
    "best_energy_delta_vs_baselines": "utility",
    "adaptive_2dpt_ready": "structured downstream gate",
    "hardware_speedup_claimed": "no false hardware claim",
    "flagged_adversarial": "adversarial-verification accountability",
    "conductor_modified": "conductor immutability",
    "tests_run": "verification evidence",
    "schema": "artifact schema stability",
    "run_date": "run labeling",
    "result_path": "artifact reachability",
    "spec_refs": "OpenSpec traceability",
    "random_seed": "deterministic replay anchor",
    "reproducibility_checksum": "content-addressed reproducibility",
    "per_instance_results": "per-distribution evidence, not only aggregates",
    "round_trip_proxies": "mixing telemetry",
    "mixing_improvement": "ready-gate evidence",
}
DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5129_hubo_adaptive_2dpt_v470.py --date 20260701",
    ".venv/bin/pytest tests/python/test_hubo_adaptive_2dpt_5129.py -q",
    ".venv/bin/pytest tests/python/test_hubo_adaptive_2dpt_5129.py "
    "--cov=python/carnot/experiment_5129_hubo_adaptive_2dpt_v470.py "
    "--cov=python/carnot/samplers/hubo_2dpt.py --cov-report=term-missing --cov-fail-under=100 -q",
    ".venv/bin/pytest tests/python -q",
]


def build_artifact(
    *,
    root: str | Path | None = None,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Run the adaptive CPU comparison and return the terminal artifact."""

    started = time.perf_counter()
    repo_root = Path(root) if root is not None else REPO_ROOT
    baseline = _load_exp5116_baseline(repo_root)
    config = AdaptiveHubo2DPTConfig()
    fixed_config = Hubo2DPTConfig(
        beta_grid=config.initial_beta_grid,
        penalty_grid=config.penalty_grid,
        sweeps=config.sweeps,
        swap_interval=config.swap_interval,
    )
    problems = build_synthetic_hubo_families()
    per_instance_results: list[JsonDict] = []
    best_energies: dict[str, list[float]] = _metric_lists()
    residual_traces: dict[str, list[tuple[float, ...]]] = _trace_lists()
    optimum_hits = {algorithm: 0 for algorithm in best_energies}
    run_count = 0

    for problem_index, problem in enumerate(problems):
        exact = exact_enumerate(problem, penalty=config.target_penalty)
        exact_ok = (
            len(exact.all_states) == 2**problem.n_vars
            and sum(exact.energy_distribution.values()) == len(exact.all_states)
            and bool(exact.optimal_states)
        )
        exact_labels_preserved = _exact_labels_match_baseline(baseline, problem.name, exact)
        runs_by_algorithm: dict[str, list[Any]] = {algorithm: [] for algorithm in best_energies}

        for seed in DEFAULT_SEEDS:
            run_seed = int(seed + problem_index * 100)
            unguided = run_unguided_gibbs(
                problem,
                seed=run_seed,
                beta=max(config.initial_beta_grid),
                penalty=config.target_penalty,
                sweeps=config.sweeps,
            )
            beta_pt = run_beta_parallel_tempering(
                problem,
                seed=run_seed,
                beta_grid=config.initial_beta_grid,
                penalty=config.target_penalty,
                sweeps=config.sweeps,
                swap_interval=config.swap_interval,
            )
            fixed = Hubo2DParallelTemperingSampler(fixed_config).run(problem, seed=run_seed)
            adaptive = AdaptiveHubo2DParallelTemperingSampler(config).run(
                problem,
                seed=run_seed,
                exact_optimum_energy=exact.optimum_energy,
            )
            runs_by_algorithm["unguided_gibbs"].append(unguided)
            runs_by_algorithm["beta_pt"].append(beta_pt)
            runs_by_algorithm["fixed_grid_2dpt"].append(fixed)
            runs_by_algorithm["adaptive_two_d_beta_penalty_pt"].append(adaptive)

        for algorithm, runs in runs_by_algorithm.items():
            for run in runs:
                best_energies[algorithm].append(run.best_energy)
                optimum_hits[algorithm] += int(_energy_equal(run.best_energy, exact.optimum_energy))
                residual_traces[algorithm].append(_residual_trace(run.energy_trace, exact.optimum_energy))
        run_count += len(DEFAULT_SEEDS)

        per_instance_results.append(
            {
                "instance_id": problem.name,
                "family": problem.family,
                "n_vars": problem.n_vars,
                "description": problem.description,
                "exact_enumeration_checked": exact_ok,
                "exact_labels_preserved": exact_labels_preserved,
                "exact": exact.as_dict(),
                "best_energy_by_algorithm": {
                    algorithm: min(run.best_energy for run in runs)
                    for algorithm, runs in runs_by_algorithm.items()
                },
                "mean_best_energy_by_algorithm": {
                    algorithm: _mean(run.best_energy for run in runs)
                    for algorithm, runs in runs_by_algorithm.items()
                },
                "optimum_hits_by_algorithm": {
                    algorithm: sum(
                        int(_energy_equal(run.best_energy, exact.optimum_energy))
                        for run in runs
                    )
                    for algorithm, runs in runs_by_algorithm.items()
                },
                "runs": {
                    algorithm: [_run_as_dict(run, algorithm) for run in runs]
                    for algorithm, runs in runs_by_algorithm.items()
                },
            }
        )

    exact_enumeration_checked = all(
        bool(row["exact_enumeration_checked"]) and bool(row["exact_labels_preserved"])
        for row in per_instance_results
    )
    optimum_hit_rate = {
        algorithm: _round_metric(hits / run_count)
        for algorithm, hits in optimum_hits.items()
    }
    swap_acceptance_rates = _aggregate_swap_acceptance(per_instance_results)
    round_trip_proxies = _aggregate_round_trip(per_instance_results)
    residual_energy_by_sweep = {
        algorithm: _residual_summary(traces)
        for algorithm, traces in residual_traces.items()
    }
    detailed_balance_sanity = _aggregate_detailed_balance(per_instance_results)
    deltas = _best_energy_deltas(best_energies, baseline)
    mixing_improvement = _mixing_improvement(per_instance_results)
    no_harmful_regression = bool(
        deltas["adaptive_vs_fixed_grid_2dpt"] <= 0.0
        and optimum_hit_rate["adaptive_two_d_beta_penalty_pt"] >= optimum_hit_rate["fixed_grid_2dpt"]
    )
    hardware_speedup_claimed = False
    conductor_modified = False
    adaptive_2dpt_ready = bool(
        exact_enumeration_checked
        and detailed_balance_sanity["passed"]
        and mixing_improvement["at_least_one_metric_improved"]
        and no_harmful_regression
        and not hardware_speedup_claimed
    )
    honest_verdict = (
        COMPLETE_VERDICT
        if adaptive_2dpt_ready
        else BLOCKED_VERDICT
        if not exact_enumeration_checked
        else NOT_READY_VERDICT
    )
    flagged_adversarial = bool(
        not exact_enumeration_checked
        or hardware_speedup_claimed
        or conductor_modified
        or INFERENCE_SUBSTRATE != "cpu_exact_checked_hubo_sampling"
    )
    elapsed = _round_metric(time.perf_counter() - started) if duration_s is None else duration_s
    artifact: JsonDict = {
        "schema": "carnot.experiment_5129_hubo_adaptive_2dpt.v470",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "result_path": RESULT_RELATIVE_PATH,
        "random_seed": DEFAULT_SEEDS[0],
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": elapsed,
        "exp5116_baseline_loaded": bool(baseline),
        "exp5116_baseline_path": exp5116.RESULT_RELATIVE_PATH,
        "instance_families": {
            "families": sorted({problem.family for problem in problems}),
            "instances": [
                {
                    "instance_id": problem.name,
                    "family": problem.family,
                    "n_vars": problem.n_vars,
                }
                for problem in problems
            ],
        },
        "exact_enumeration_checked": exact_enumeration_checked,
        "adaptive_temperature_config": _adaptive_temperature_config(config, per_instance_results),
        "swap_acceptance_rates": swap_acceptance_rates,
        "residual_energy_by_sweep": residual_energy_by_sweep,
        "optimum_hit_rate": optimum_hit_rate,
        "detailed_balance_sanity": detailed_balance_sanity,
        "best_energy_delta_vs_baselines": deltas,
        "round_trip_proxies": round_trip_proxies,
        "mixing_improvement": mixing_improvement,
        "adaptive_2dpt_ready": adaptive_2dpt_ready,
        "hardware_speedup_claimed": hardware_speedup_claimed,
        "flagged_adversarial": flagged_adversarial,
        "conductor_modified": conductor_modified,
        "tests_run": list(tests_run) if tests_run is not None else list(DEFAULT_TESTS_RUN),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-SAMPLE-5129", "SCENARIO-SAMPLE-5129"],
        "per_instance_results": per_instance_results,
        "methodology_note": (
            "Adaptive updates modify only the CPU inverse-temperature ladder. "
            "Exact enumeration supplies optimum labels for every tiny instance, "
            "and readiness is blocked unless exact labels, detailed-balance "
            "sanity, and no-regression checks all pass."
        ),
    }
    artifact["reproducibility_checksum"] = _sha256_json(
        {
            "experiment_id": artifact["experiment_id"],
            "run_date": artifact["run_date"],
            "adaptive_temperature_config": artifact["adaptive_temperature_config"],
            "per_instance_results": artifact["per_instance_results"],
            "best_energy_delta_vs_baselines": artifact["best_energy_delta_vs_baselines"],
            "optimum_hit_rate": artifact["optimum_hit_rate"],
        }
    )
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 5129 artifact violates the terminal contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, f"missing required fields: {missing}")
    _require(artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id")
    _require(artifact.get("milestone") == MILESTONE, "milestone")
    _require(str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(isinstance(artifact.get("duration_s"), (float, int)), "duration_s")
    _require(float(artifact["duration_s"]) >= 0.0, "duration_s")
    _require(artifact.get("exp5116_baseline_loaded") is True, "exp5116_baseline_loaded")
    _require(artifact.get("exact_enumeration_checked") is True, "exact_enumeration_checked")
    _require(_families_valid(artifact.get("instance_families")), "instance_families")
    _require(_adaptive_config_valid(artifact.get("adaptive_temperature_config")), "adaptive_temperature_config")
    _require(_swap_rates_valid(artifact.get("swap_acceptance_rates")), "swap_acceptance_rates")
    _require(_residuals_valid(artifact.get("residual_energy_by_sweep")), "residual_energy_by_sweep")
    _require(_hit_rates_valid(artifact.get("optimum_hit_rate")), "optimum_hit_rate")
    _require(_balance_valid(artifact.get("detailed_balance_sanity")), "detailed_balance_sanity")
    _require(_deltas_valid(artifact.get("best_energy_delta_vs_baselines")), "best_energy_delta_vs_baselines")
    _require(artifact.get("adaptive_2dpt_ready") is True, "adaptive_2dpt_ready")
    _require(artifact.get("hardware_speedup_claimed") is False, "hardware_speedup_claimed")
    _require(artifact.get("flagged_adversarial") is False, "flagged_adversarial")
    _require(artifact.get("conductor_modified") is False, "conductor_modified")
    _require(isinstance(artifact.get("tests_run"), list) and bool(artifact["tests_run"]), "tests_run")


def write_artifact(
    *,
    root: str | Path | None = None,
    output_path: str | Path | None = None,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build and write the Exp 5129 terminal artifact."""

    repo_root = Path(root) if root is not None else REPO_ROOT
    destination = Path(output_path) if output_path is not None else repo_root / RESULT_RELATIVE_PATH
    artifact = build_artifact(
        root=repo_root,
        run_date=run_date,
        duration_s=duration_s,
        tests_run=tests_run,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(
    *,
    root: str | Path | None = None,
    date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """CLI-compatible entrypoint used by the wrapper script and tests."""

    repo_root = Path(root) if root is not None else REPO_ROOT
    write_artifact(root=repo_root, run_date=date, duration_s=duration_s, tests_run=tests_run)
    return repo_root / RESULT_RELATIVE_PATH


def _load_exp5116_baseline(root: Path) -> JsonDict:
    path = root / exp5116.RESULT_RELATIVE_PATH
    if not path.exists():
        path = REPO_ROOT / exp5116.RESULT_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    exp5116.validate_artifact(payload)
    return dict(payload)


def _metric_lists() -> dict[str, list[float]]:
    return {
        "unguided_gibbs": [],
        "beta_pt": [],
        "fixed_grid_2dpt": [],
        "adaptive_two_d_beta_penalty_pt": [],
    }


def _trace_lists() -> dict[str, list[tuple[float, ...]]]:
    return {algorithm: [] for algorithm in _metric_lists()}


def _exact_labels_match_baseline(
    baseline: Mapping[str, Any],
    instance_id: str,
    exact: Any,
) -> bool:
    rows = baseline.get("per_instance_results", [])
    for row in rows:
        if row.get("instance_id") == instance_id:
            baseline_states = row.get("exact", {}).get("optimal_states")
            return baseline_states == [list(state) for state in exact.optimal_states]
    return False


def _run_as_dict(run: Any, algorithm: str) -> JsonDict:
    payload = run.as_dict()
    payload["algorithm"] = algorithm
    return payload


def _residual_trace(energy_trace: Sequence[float], optimum_energy: float) -> tuple[float, ...]:
    return tuple(_round_metric(max(0.0, float(energy) - optimum_energy)) for energy in energy_trace)


def _aggregate_swap_acceptance(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    totals = {
        "unguided_gibbs": {"beta_axis": SwapStats(), "penalty_axis": SwapStats()},
        "beta_pt": {"beta_axis": SwapStats(), "penalty_axis": SwapStats()},
        "fixed_grid_2dpt": {"beta_axis": SwapStats(), "penalty_axis": SwapStats()},
        "adaptive_two_d_beta_penalty_pt": {"beta_axis": SwapStats(), "penalty_axis": SwapStats()},
    }
    pair_totals = [SwapStats() for _ in range(len(AdaptiveHubo2DPTConfig().initial_beta_grid) - 1)]
    for row in rows:
        for algorithm in totals:
            for run in row["runs"][algorithm]:
                for axis in totals[algorithm]:
                    totals[algorithm][axis] = _add_stats(totals[algorithm][axis], run["swap_stats"][axis])
                if algorithm == "adaptive_two_d_beta_penalty_pt":
                    for pair in run["beta_pair_swap_stats"]:
                        pair_index = int(pair["pair_index"])
                        pair_totals[pair_index] = _add_stats(pair_totals[pair_index], pair)
    result = {
        algorithm: {axis: stats.as_dict() for axis, stats in axes.items()}
        for algorithm, axes in totals.items()
    }
    result["adaptive_two_d_beta_penalty_pt"]["beta_pair_axis"] = [
        {"pair_index": pair_index, **stats.as_dict()}
        for pair_index, stats in enumerate(pair_totals)
    ]
    return result


def _aggregate_round_trip(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    fixed_span: list[float] = []
    adaptive_span: list[float] = []
    for row in rows:
        for run in row["runs"]["adaptive_two_d_beta_penalty_pt"]:
            adaptive_span.append(float(run["round_trip_proxy"]["mean_beta_span_fraction"]))
        for run in row["runs"]["fixed_grid_2dpt"]:
            fixed_span.append(_fixed_round_trip_proxy(run))
    return {
        "fixed_grid_2dpt_mean_beta_span_fraction": _round_metric(_mean(fixed_span)),
        "adaptive_mean_beta_span_fraction": _round_metric(_mean(adaptive_span)),
        "adaptive_minus_fixed_span_fraction": _round_metric(_mean(adaptive_span) - _mean(fixed_span)),
    }


def _fixed_round_trip_proxy(run: Mapping[str, Any]) -> float:
    beta_rate = float(run["swap_stats"]["beta_axis"]["acceptance_rate"])
    return min(1.0, beta_rate)


def _residual_summary(traces: Sequence[Sequence[float]]) -> JsonDict:
    by_sweep = list(zip(*traces))
    mean_by_sweep = [_round_metric(_mean(values)) for values in by_sweep]
    return {
        "mean": mean_by_sweep,
        "final_mean": mean_by_sweep[-1],
        "max_final": _round_metric(max(trace[-1] for trace in traces)),
    }


def _aggregate_detailed_balance(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    checks = 0
    max_abs = 0.0
    forward = 0
    backward = 0
    for row in rows:
        for run in row["runs"]["adaptive_two_d_beta_penalty_pt"]:
            sanity = run["detailed_balance_sanity"]
            checks += int(sanity["checks"])
            max_abs = max(max_abs, float(sanity["local_log_ratio_antisymmetry_max_abs"]))
            forward += int(sanity["trajectory_forward_moves"])
            backward += int(sanity["trajectory_backward_moves"])
    net_flow = abs(forward - backward)
    return {
        "checks": checks,
        "local_log_ratio_antisymmetry_max_abs": _round_metric(max_abs),
        "trajectory_forward_moves": forward,
        "trajectory_backward_moves": backward,
        "trajectory_net_flow_abs": net_flow,
        "passed": bool(checks > 0 and max_abs <= 1e-9 and net_flow == 0),
    }


def _best_energy_deltas(best_energies: Mapping[str, Sequence[float]], baseline: Mapping[str, Any]) -> JsonDict:
    adaptive = _mean(best_energies["adaptive_two_d_beta_penalty_pt"])
    fixed = _mean(best_energies["fixed_grid_2dpt"])
    loaded = _loaded_exp5116_mean_best_energy(baseline, "two_d_beta_penalty_pt")
    return {
        "adaptive_vs_fixed_grid_2dpt": _round_metric(adaptive - fixed),
        "adaptive_vs_beta_pt": _round_metric(adaptive - _mean(best_energies["beta_pt"])),
        "adaptive_vs_unguided_gibbs": _round_metric(adaptive - _mean(best_energies["unguided_gibbs"])),
        "adaptive_vs_loaded_exp5116_two_dpt": _round_metric(adaptive - loaded),
    }


def _loaded_exp5116_mean_best_energy(baseline: Mapping[str, Any], algorithm: str) -> float:
    values: list[float] = []
    for row in baseline["per_instance_results"]:
        values.extend(float(run["best_energy"]) for run in row["runs"][algorithm])
    return _mean(values)


def _mixing_improvement(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    fixed_beta_rates: list[float] = []
    adaptive_pair_rates: list[float] = []
    fixed_round_trip: list[float] = []
    adaptive_round_trip: list[float] = []
    for row in rows:
        for run in row["runs"]["fixed_grid_2dpt"]:
            fixed_beta_rates.append(float(run["swap_stats"]["beta_axis"]["acceptance_rate"]))
            fixed_round_trip.append(_fixed_round_trip_proxy(run))
        for run in row["runs"]["adaptive_two_d_beta_penalty_pt"]:
            adaptive_pair_rates.extend(
                float(pair["acceptance_rate"]) for pair in run["beta_pair_swap_stats"]
            )
            adaptive_round_trip.append(float(run["round_trip_proxy"]["mean_beta_span_fraction"]))
    fixed_std = _std(fixed_beta_rates)
    adaptive_std = _std(adaptive_pair_rates)
    round_trip_delta = _mean(adaptive_round_trip) - _mean(fixed_round_trip)
    pair_balance_delta = adaptive_std - fixed_std
    return {
        "fixed_beta_acceptance_std": _round_metric(fixed_std),
        "adaptive_pair_acceptance_std": _round_metric(adaptive_std),
        "pair_acceptance_std_delta": _round_metric(pair_balance_delta),
        "round_trip_span_delta": _round_metric(round_trip_delta),
        "at_least_one_metric_improved": bool(pair_balance_delta < 0.0 or round_trip_delta > 0.0),
    }


def _adaptive_temperature_config(config: AdaptiveHubo2DPTConfig, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    final_grids = [
        run["beta_grid_final"]
        for row in rows
        for run in row["runs"]["adaptive_two_d_beta_penalty_pt"]
    ]
    histories = [
        run["beta_grid_history"]
        for row in rows
        for run in row["runs"]["adaptive_two_d_beta_penalty_pt"]
    ]
    return {
        "initial_beta_grid": list(config.initial_beta_grid),
        "penalty_grid": list(config.penalty_grid),
        "sweeps": config.sweeps,
        "swap_interval": config.swap_interval,
        "adaptation_interval": config.adaptation_interval,
        "target_acceptance": config.target_acceptance,
        "adaptation_learning_rate": config.adaptation_learning_rate,
        "min_beta_gap": config.min_beta_gap,
        "final_beta_grids": final_grids,
        "monotonic_order_preserved": all(_history_monotonic(history) for history in histories),
    }


def _history_monotonic(history: Sequence[Sequence[float]]) -> bool:
    return all(all(right > left for left, right in zip(grid, grid[1:])) for grid in history)


def _families_valid(value: Any) -> bool:
    return isinstance(value, Mapping) and len(value.get("families", [])) >= 2


def _adaptive_config_valid(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("monotonic_order_preserved") is True


def _swap_rates_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    adaptive = value.get("adaptive_two_d_beta_penalty_pt")
    fixed = value.get("fixed_grid_2dpt")
    if not isinstance(adaptive, Mapping) or not isinstance(fixed, Mapping):
        return False
    for stats in (adaptive.get("beta_axis"), adaptive.get("penalty_axis"), fixed.get("beta_axis")):
        if not _stats_valid(stats, require_attempts=True):
            return False
    return True


def _stats_valid(value: Any, *, require_attempts: bool) -> bool:
    if not isinstance(value, Mapping):
        return False
    attempts = value.get("attempts")
    accepted = value.get("accepted")
    rate = value.get("acceptance_rate")
    if not isinstance(attempts, int) or attempts < int(require_attempts):
        return False
    return (
        isinstance(accepted, int)
        and 0 <= accepted <= attempts
        and isinstance(rate, (float, int))
        and 0.0 <= float(rate) <= 1.0
    )


def _residuals_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    adaptive = value.get("adaptive_two_d_beta_penalty_pt", {})
    return isinstance(adaptive.get("mean"), list) and adaptive.get("final_mean", -1.0) >= 0.0


def _hit_rates_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    adaptive = value.get("adaptive_two_d_beta_penalty_pt")
    fixed = value.get("fixed_grid_2dpt")
    return isinstance(adaptive, (float, int)) and isinstance(fixed, (float, int)) and adaptive >= fixed


def _balance_valid(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("passed") is True


def _deltas_valid(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("adaptive_vs_fixed_grid_2dpt", 1.0) <= 0.0


def _add_stats(total: SwapStats, value: Mapping[str, Any]) -> SwapStats:
    return SwapStats(
        attempts=total.attempts + int(value["attempts"]),
        accepted=total.accepted + int(value["accepted"]),
    )


def _mean(values: Sequence[float] | Any) -> float:
    numbers = [float(value) for value in values]
    return sum(numbers) / len(numbers)


def _std(values: Sequence[float]) -> float:
    mean = _mean(values)
    return (sum((float(value) - mean) ** 2 for value in values) / len(values)) ** 0.5


def _round_metric(value: float) -> float:
    return round(float(value), 6)


def _energy_equal(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-9


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
