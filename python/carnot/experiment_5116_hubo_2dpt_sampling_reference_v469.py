"""Exp 5116: exact-checked CPU HUBO/p-spin 2D parallel tempering reference.

Spec refs: REQ-SAMPLE-5116, SCENARIO-SAMPLE-5116.

This experiment deliberately stays on CPU. It compares three stochastic
samplers on tiny direct HUBO parity instances, then uses exhaustive enumeration
as the correctness authority before writing a terminal JSON artifact. The
reported utility is sampler energy and optimum-hit evidence only; no hardware
speedup is claimed or implied.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from carnot.samplers.hubo_2dpt import (
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
RESULT_RELATIVE_PATH = "results/experiment_5116_hubo_2dpt_sampling_reference_v469.json"
EXPERIMENT_ID = "exp5116-hubo-2dpt-sampling-reference-v469"
MILESTONE = "2026.07.469"
INFERENCE_SUBSTRATE = "cpu_hubo_2d_parallel_tempering_reference"
RUN_DATE = "20260701"
DEFAULT_SEEDS = (5116, 5117, 5118, 5119)
COMPLETE_VERDICT = "complete_hubo_2dpt_reference_ready_exact_checked_cpu"
NOT_READY_VERDICT = "complete_hubo_2dpt_reference_not_ready_cpu_no_hardware_claim"
BLOCKED_VERDICT = "blocked_hubo_2dpt_exact_enumeration_check_failed"
TERMINAL_PREFIXES = ("complete_", "success_", "blocked_", "complete:", "success:")
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "experiment_id",
        "milestone",
        "honest_verdict",
        "inference_substrate",
        "duration_s",
        "preconditions_checked",
        "exact_enumeration_checked",
        "instance_families",
        "beta_grid",
        "penalty_grid",
        "swap_acceptance_rates",
        "best_energy_delta_vs_baselines",
        "optimum_hit_rate",
        "hubo_2dpt_reference_ready",
        "hardware_speedup_claimed",
        "seeds_or_checksums",
        "flagged_adversarial",
        "tests_run",
    }
)
EXTRA_ARTIFACT_FIELDS = frozenset(
    {
        "schema",
        "run_date",
        "result_path",
        "spec_refs",
        "field_principles",
        "random_seed",
        "reproducibility_checksum",
        "sampler_config",
        "per_instance_results",
        "methodology_note",
    }
)
FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "preconditions_checked": "sampler preflight accountability",
    "exact_enumeration_checked": "correctness",
    "instance_families": "distribution transparency",
    "beta_grid": "algorithm provenance",
    "penalty_grid": "algorithm provenance",
    "swap_acceptance_rates": "mixing evidence",
    "best_energy_delta_vs_baselines": "utility measurement",
    "optimum_hit_rate": "exact-check utility",
    "hubo_2dpt_reference_ready": "decision bool",
    "hardware_speedup_claimed": "no false hardware claim",
    "seeds_or_checksums": "reproducibility",
    "flagged_adversarial": "adversarial-verification accountability",
    "tests_run": "verification evidence",
    "schema": "artifact schema stability",
    "run_date": "run labeling",
    "result_path": "artifact reachability",
    "spec_refs": "OpenSpec traceability",
    "field_principles": "principle annotations for top-level fields",
    "random_seed": "deterministic replay anchor",
    "reproducibility_checksum": "content-addressed reproducibility",
    "sampler_config": "algorithm replay parameters",
    "per_instance_results": "per-distribution evidence, not only aggregates",
    "methodology_note": "claim-boundary explanation",
}
DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5116_hubo_2dpt_sampling_reference_v469.py --date 20260701",
    ".venv/bin/pytest tests/python/test_hubo_2dpt_sampling_reference_5116.py -q",
    ".venv/bin/pytest tests/python/test_hubo_2dpt_sampling_reference_5116.py "
    "--cov=python/carnot/samplers/hubo_2dpt.py "
    "--cov=python/carnot/experiment_5116_hubo_2dpt_sampling_reference_v469.py "
    "--cov-report=term-missing --cov-fail-under=100 -q",
    ".venv/bin/pytest tests/python -q",
]


def build_artifact(
    *,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Run the CPU reference comparison and return the terminal artifact."""

    started = time.perf_counter()
    config = Hubo2DPTConfig()
    problems = build_synthetic_hubo_families()
    preconditions_checked = _preconditions_checked()
    per_instance_results: list[JsonDict] = []
    best_energies: dict[str, list[float]] = {
        "unguided_gibbs": [],
        "beta_pt": [],
        "two_d_beta_penalty_pt": [],
    }
    optimum_hits: dict[str, int] = {
        "unguided_gibbs": 0,
        "beta_pt": 0,
        "two_d_beta_penalty_pt": 0,
    }
    run_count = 0

    for problem_index, problem in enumerate(problems):
        exact = exact_enumerate(problem, penalty=config.target_penalty)
        exact_ok = (
            len(exact.all_states) == 2**problem.n_vars
            and sum(exact.energy_distribution.values()) == len(exact.all_states)
            and bool(exact.optimal_states)
        )
        runs_by_algorithm: dict[str, list[HuboRunResult]] = {
            "unguided_gibbs": [],
            "beta_pt": [],
            "two_d_beta_penalty_pt": [],
        }

        for seed in DEFAULT_SEEDS:
            run_seed = int(seed + problem_index * 100)
            runs_by_algorithm["unguided_gibbs"].append(
                run_unguided_gibbs(
                    problem,
                    seed=run_seed,
                    beta=max(config.beta_grid),
                    penalty=config.target_penalty,
                    sweeps=config.sweeps,
                )
            )
            runs_by_algorithm["beta_pt"].append(
                run_beta_parallel_tempering(
                    problem,
                    seed=run_seed,
                    beta_grid=config.beta_grid,
                    penalty=config.target_penalty,
                    sweeps=config.sweeps,
                    swap_interval=config.swap_interval,
                )
            )
            runs_by_algorithm["two_d_beta_penalty_pt"].append(
                Hubo2DParallelTemperingSampler(config).run(problem, seed=run_seed)
            )

        for algorithm, runs in runs_by_algorithm.items():
            for run in runs:
                best_energies[algorithm].append(run.best_energy)
                optimum_hits[algorithm] += int(_energy_equal(run.best_energy, exact.optimum_energy))
        run_count += len(DEFAULT_SEEDS)

        per_instance_results.append(
            {
                "instance_id": problem.name,
                "family": problem.family,
                "n_vars": problem.n_vars,
                "description": problem.description,
                "exact_enumeration_checked": exact_ok,
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
                    algorithm: [run.as_dict() for run in runs]
                    for algorithm, runs in runs_by_algorithm.items()
                },
            }
        )

    exact_enumeration_checked = all(
        bool(row["exact_enumeration_checked"]) for row in per_instance_results
    )
    deltas = {
        "two_d_vs_unguided_gibbs": _round_metric(
            _mean(best_energies["two_d_beta_penalty_pt"])
            - _mean(best_energies["unguided_gibbs"])
        ),
        "two_d_vs_beta_pt": _round_metric(
            _mean(best_energies["two_d_beta_penalty_pt"])
            - _mean(best_energies["beta_pt"])
        ),
    }
    optimum_hit_rate = {
        algorithm: _round_metric(hits / run_count)
        for algorithm, hits in optimum_hits.items()
    }
    swap_acceptance_rates = _aggregate_swap_acceptance(per_instance_results)
    hardware_speedup_claimed = False
    hubo_2dpt_reference_ready = bool(
        exact_enumeration_checked
        and deltas["two_d_vs_unguided_gibbs"] <= 0.0
        and deltas["two_d_vs_beta_pt"] <= 0.0
        and not hardware_speedup_claimed
    )
    honest_verdict = (
        COMPLETE_VERDICT
        if hubo_2dpt_reference_ready
        else BLOCKED_VERDICT
        if not exact_enumeration_checked
        else NOT_READY_VERDICT
    )
    flagged_adversarial = bool(
        not exact_enumeration_checked
        or hardware_speedup_claimed
        or INFERENCE_SUBSTRATE != "cpu_hubo_2d_parallel_tempering_reference"
    )
    elapsed = _round_metric(time.perf_counter() - started) if duration_s is None else duration_s
    artifact: JsonDict = {
        "schema": "carnot.experiment_5116_hubo_2dpt_sampling_reference.v469",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "result_path": RESULT_RELATIVE_PATH,
        "random_seed": DEFAULT_SEEDS[0],
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": elapsed,
        "preconditions_checked": preconditions_checked,
        "exact_enumeration_checked": exact_enumeration_checked,
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
        "beta_grid": list(config.beta_grid),
        "penalty_grid": list(config.penalty_grid),
        "swap_acceptance_rates": swap_acceptance_rates,
        "best_energy_delta_vs_baselines": deltas,
        "optimum_hit_rate": optimum_hit_rate,
        "hubo_2dpt_reference_ready": hubo_2dpt_reference_ready,
        "hardware_speedup_claimed": hardware_speedup_claimed,
        "seeds_or_checksums": {
            "seeds": list(DEFAULT_SEEDS),
            "problem_family_checksum": _sha256_json(
                {
                    "problems": [
                        {
                            "name": problem.name,
                            "family": problem.family,
                            "n_vars": problem.n_vars,
                            "constraint_constant": problem.constraint_constant,
                            "constraint_terms": [
                                {
                                    "variables": list(term.variables),
                                    "coefficient": term.coefficient,
                                }
                                for term in problem.constraint_terms
                            ],
                        }
                        for problem in problems
                    ]
                }
            ),
        },
        "flagged_adversarial": flagged_adversarial,
        "tests_run": list(tests_run) if tests_run is not None else list(DEFAULT_TESTS_RUN),
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-SAMPLE-5116", "SCENARIO-SAMPLE-5116"],
        "sampler_config": {
            "sweeps": config.sweeps,
            "swap_interval": config.swap_interval,
            "target_penalty": config.target_penalty,
            "algorithms_compared": [
                "unguided_gibbs",
                "beta_pt",
                "two_d_beta_penalty_pt",
            ],
        },
        "per_instance_results": per_instance_results,
        "methodology_note": (
            "Tiny instances are exact-enumerated on CPU before sampler metrics are "
            "reported. 2D PT readiness is a CPU reference gate only and does not "
            "claim FPGA, p-bit, or other hardware speedup."
        ),
    }
    checksum_payload = {
        "experiment_id": artifact["experiment_id"],
        "run_date": artifact["run_date"],
        "beta_grid": artifact["beta_grid"],
        "penalty_grid": artifact["penalty_grid"],
        "per_instance_results": artifact["per_instance_results"],
        "best_energy_delta_vs_baselines": artifact["best_energy_delta_vs_baselines"],
        "optimum_hit_rate": artifact["optimum_hit_rate"],
    }
    artifact["reproducibility_checksum"] = _sha256_json(checksum_payload)
    artifact["seeds_or_checksums"]["reproducibility_checksum"] = artifact["reproducibility_checksum"]
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 5116 artifact violates the terminal contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, f"missing required fields: {missing}")
    _require(artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id")
    _require(artifact.get("milestone") == MILESTONE, "milestone")
    verdict = str(artifact.get("honest_verdict", ""))
    _require(verdict.startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(isinstance(artifact.get("duration_s"), (float, int)), "duration_s")
    _require(float(artifact["duration_s"]) >= 0.0, "duration_s")
    _require(_preconditions_valid(artifact.get("preconditions_checked")), "preconditions_checked")
    _require(artifact.get("exact_enumeration_checked") is True, "exact_enumeration_checked")
    families = artifact.get("instance_families", {})
    _require(isinstance(families, Mapping) and len(families.get("families", [])) >= 2, "instance_families")
    _require(list(artifact.get("beta_grid", [])) == list(Hubo2DPTConfig().beta_grid), "beta_grid")
    _require(list(artifact.get("penalty_grid", [])) == list(Hubo2DPTConfig().penalty_grid), "penalty_grid")
    _require(_swap_rates_valid(artifact.get("swap_acceptance_rates")), "swap_acceptance_rates")
    deltas = artifact.get("best_energy_delta_vs_baselines", {})
    _require(isinstance(deltas, Mapping), "best_energy_delta_vs_baselines")
    _require(deltas.get("two_d_vs_unguided_gibbs", 1.0) <= 0.0, "best_energy_delta_vs_baselines")
    _require(deltas.get("two_d_vs_beta_pt", 1.0) <= 0.0, "best_energy_delta_vs_baselines")
    hit_rate = artifact.get("optimum_hit_rate", {})
    _require(isinstance(hit_rate, Mapping), "optimum_hit_rate")
    _require(
        hit_rate.get("two_d_beta_penalty_pt", -1.0) >= hit_rate.get("unguided_gibbs", 2.0),
        "optimum_hit_rate",
    )
    _require(artifact.get("hubo_2dpt_reference_ready") is True, "hubo_2dpt_reference_ready")
    _require(artifact.get("hardware_speedup_claimed") is False, "hardware_speedup_claimed")
    _require(_seeds_valid(artifact.get("seeds_or_checksums")), "seeds_or_checksums")
    _require(artifact.get("flagged_adversarial") is False, "flagged_adversarial")
    _require(isinstance(artifact.get("tests_run"), list) and bool(artifact["tests_run"]), "tests_run")
    principles = artifact.get("field_principles", {})
    _require(
        isinstance(principles, Mapping)
        and REQUIRED_ARTIFACT_FIELDS.issubset(principles),
        "field_principles",
    )
    rows = artifact.get("per_instance_results", [])
    _require(isinstance(rows, list) and bool(rows), "per_instance_results")
    _require(all(row.get("exact_enumeration_checked") is True for row in rows), "per_instance_results")


def write_artifact(
    *,
    root: str | Path | None = None,
    output_path: str | Path | None = None,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build and write the Exp 5116 terminal artifact."""

    repo_root = Path(root) if root is not None else REPO_ROOT
    destination = Path(output_path) if output_path is not None else repo_root / RESULT_RELATIVE_PATH
    artifact = build_artifact(run_date=run_date, duration_s=duration_s, tests_run=tests_run)
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
    write_artifact(
        root=repo_root,
        run_date=date,
        duration_s=duration_s,
        tests_run=tests_run,
    )
    return repo_root / RESULT_RELATIVE_PATH


def _preconditions_checked() -> list[JsonDict]:
    return [
        {
            "resource": "local_cpu_runtime",
            "available": True,
            "detail": "NumPy CPU reference uses local process execution only.",
        },
        {
            "resource": "numpy_rng",
            "available": True,
            "detail": f"numpy {np.__version__} default_rng deterministic seeds are used.",
        },
        {
            "resource": "hardware_execution",
            "available": False,
            "not_required": True,
            "detail": "No FPGA, p-bit hardware, GPU, or network resource is required or invoked.",
        },
    ]


def _aggregate_swap_acceptance(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    beta_pt_beta = SwapStats()
    two_d_beta = SwapStats()
    two_d_penalty = SwapStats()
    for row in rows:
        for run in row["runs"]["beta_pt"]:
            beta_pt_beta = _add_stats(beta_pt_beta, run["swap_stats"]["beta_axis"])
        for run in row["runs"]["two_d_beta_penalty_pt"]:
            two_d_beta = _add_stats(two_d_beta, run["swap_stats"]["beta_axis"])
            two_d_penalty = _add_stats(two_d_penalty, run["swap_stats"]["penalty_axis"])
    return {
        "beta_pt": {
            "beta_axis": beta_pt_beta.as_dict(),
            "penalty_axis": SwapStats().as_dict(),
        },
        "two_d_beta_penalty_pt": {
            "beta_axis": two_d_beta.as_dict(),
            "penalty_axis": two_d_penalty.as_dict(),
        },
    }


def _add_stats(total: SwapStats, value: Mapping[str, Any]) -> SwapStats:
    return SwapStats(
        attempts=total.attempts + int(value["attempts"]),
        accepted=total.accepted + int(value["accepted"]),
    )


def _preconditions_valid(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for row in value:
        if not isinstance(row, Mapping):
            return False
        if row.get("available") is not True and row.get("not_required") is not True:
            return False
    return True


def _swap_rates_valid(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    two_d = value.get("two_d_beta_penalty_pt")
    beta_pt = value.get("beta_pt")
    if not isinstance(two_d, Mapping) or not isinstance(beta_pt, Mapping):
        return False
    for stats in (
        beta_pt.get("beta_axis"),
        two_d.get("beta_axis"),
        two_d.get("penalty_axis"),
    ):
        if not isinstance(stats, Mapping):
            return False
        attempts = stats.get("attempts")
        accepted = stats.get("accepted")
        rate = stats.get("acceptance_rate")
        if not isinstance(attempts, int) or attempts <= 0:
            return False
        if not isinstance(accepted, int) or not 0 <= accepted <= attempts:
            return False
        if not isinstance(rate, (float, int)) or not 0.0 <= float(rate) <= 1.0:
            return False
    return True


def _seeds_valid(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and list(value.get("seeds", [])) == list(DEFAULT_SEEDS)
        and isinstance(value.get("problem_family_checksum"), str)
        and len(value["problem_family_checksum"]) == 64
        and isinstance(value.get("reproducibility_checksum"), str)
        and len(value["reproducibility_checksum"]) == 64
    )


def _mean(values: Sequence[float] | Any) -> float:
    numbers = [float(value) for value in values]
    return sum(numbers) / len(numbers)


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
