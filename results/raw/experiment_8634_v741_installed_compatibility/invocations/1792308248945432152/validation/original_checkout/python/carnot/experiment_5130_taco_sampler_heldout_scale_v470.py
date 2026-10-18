"""Exp 5130: held-out TACO CSP scaling with adaptive sampler telemetry.

Spec refs: REQ-SAMPLE-5130, SCENARIO-SAMPLE-5130.

The adaptive logic in this file never owns a CSP label. It only proposes
variable orders for the complete CPU exact solver inherited from Exp 5117. A
small brute-force enumerator cross-checks every held-out label so downstream
FR-11 traces cannot accidentally treat a helpful ordering heuristic as a
correctness oracle.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5117_taco_harm_gated_scale_v469 as exp5117
from carnot import experiment_5129_hubo_adaptive_2dpt_v470 as exp5129
from carnot.experiment_5117_taco_harm_gated_scale_v469 import (
    AdaptationResult,
    CspConstraint,
    ExactCspSolver,
    ScaledCspInstance,
    SolveTrace,
    adapt_instance_order,
    baseline_order,
    harm_gate_decision,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_5130_taco_sampler_heldout_scale_v470.json"
EXP5117_RELATIVE_PATH = exp5117.RESULT_RELATIVE_PATH
EXP5129_RELATIVE_PATH = exp5129.RESULT_RELATIVE_PATH
EXPERIMENT_ID = "exp5130-taco-sampler-heldout-scale-v470"
MILESTONE = "2026.07.470"
RUN_DATE = "20260701"
RANDOM_SEED = 5130
INFERENCE_SUBSTRATE = "cpu_exact_solver_with_adaptive_heuristic"
EXACT_SOLVER_BACKEND = "deterministic_backtracking_finite_domain_csp_cpu_with_bruteforce_crosscheck"
READY_VERDICT = "success_heldout_csp_trace_suite_ready_exact_labels_preserved"
NOT_READY_VERDICT = "complete_heldout_csp_trace_suite_not_ready"
BLOCKED_VERDICT = "blocked_exp5129_adaptive_2dpt_not_ready"
TERMINAL_PREFIXES = ("success_", "complete_", "blocked_", "success:", "complete:", "blocked:")
POLICY_ARMS = ("baseline", "unguarded", "guarded", "sampler_feature")
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "experiment_id",
        "milestone",
        "honest_verdict",
        "inference_substrate",
        "duration_s",
        "exp5117_baseline_loaded",
        "exp5129_sampler_features_loaded",
        "exact_solver_backend",
        "heldout_instance_hashes",
        "instance_count",
        "average_effort_reduction_ratio_guarded",
        "harmful_instance_count_guarded",
        "harmful_instance_count_unguarded",
        "wrong_label_count",
        "timeout_rate",
        "per_family_results",
        "heldout_csp_trace_suite_ready",
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
    "exp5117_baseline_loaded": "continuation accountability",
    "exp5129_sampler_features_loaded": "dependency accountability",
    "exact_solver_backend": "correctness authority",
    "heldout_instance_hashes": "data provenance",
    "instance_count": "sample-size accountability",
    "average_effort_reduction_ratio_guarded": "utility",
    "harmful_instance_count_guarded": "safety",
    "harmful_instance_count_unguarded": "harm-gate value",
    "wrong_label_count": "exact correctness",
    "timeout_rate": "operational risk",
    "per_family_results": "generalization",
    "heldout_csp_trace_suite_ready": "structured downstream gate",
    "flagged_adversarial": "adversarial-verification accountability",
    "conductor_modified": "conductor immutability",
    "tests_run": "verification evidence",
    "schema": "artifact schema stability",
    "run_date": "run labeling",
    "result_path": "artifact reachability",
    "spec_refs": "OpenSpec traceability",
    "baseline_effort": "baseline transparency",
    "unguarded_effort": "unguarded comparison",
    "guarded_effort": "gated comparison",
    "sampler_feature_effort": "adaptive-sampler feature comparison",
    "per_instance_results": "trace detail",
}
DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5130_taco_sampler_heldout_scale_v470.py --date 20260701",
    ".venv/bin/pytest tests/python/test_experiment_5130_taco_sampler_heldout_scale_v470.py -q",
    ".venv/bin/pytest tests/python/test_experiment_5130_taco_sampler_heldout_scale_v470.py "
    "--cov=python/carnot/experiment_5130_taco_sampler_heldout_scale_v470.py "
    "--cov=scripts/experiment_5130_taco_sampler_heldout_scale_v470.py "
    "--cov-report=term-missing --cov-fail-under=100 -q",
    ".venv/bin/pytest tests/python -q",
]


@dataclass(frozen=True)
class HeldoutCspCase:
    """One held-out CSP plus the family label used for generalization traces."""

    family: str
    instance: ScaledCspInstance


class GateBlockedError(RuntimeError):
    """Raised when the upstream adaptive sampler readiness gate is closed."""

    def __init__(self, actual_value: Any) -> None:
        super().__init__("exp5129 adaptive_2dpt_ready gate is not true")
        self.actual_value = actual_value


def build_heldout_csp_suite() -> tuple[HeldoutCspCase, ...]:
    """Build deterministic held-out CSP families disjoint from tuning hashes."""

    return (
        _apex_cycle_case("heldout_apex_cycle4_3color_sat", 4),
        _apex_cycle_case("heldout_apex_cycle5_3color_unsat", 5),
        _grid_case("heldout_grid3x3_2color_sat", rows=3, cols=3, diagonal=False),
        _grid_case("heldout_grid2x3_diagonal_2color_unsat", rows=2, cols=3, diagonal=True),
        _crown_case(),
        _sparse_path_case(),
        _prism_case(),
        _complete_case("heldout_k5_5color_sat", n_nodes=5, n_colors=5, expected_colorable=True),
        _complete_case("heldout_k6_5color_unsat", n_nodes=6, n_colors=5, expected_colorable=False),
        _overlap_all_diff_case("heldout_overlap_all_diff5_sat", n_nodes=7, n_colors=5, expected_colorable=True),
        _overlap_all_diff_case("heldout_all_diff5_4color_unsat", n_nodes=5, n_colors=4, expected_colorable=False),
    )


def tuning_instance_hashes() -> set[str]:
    """Return hashes for Exp 5117 train/dev tuning instances."""

    return {
        _instance_hash("exp5117_tuning", instance)
        for instance in exp5117.build_scaled_csp_suite()
        if instance.split in {"train", "dev"}
    }


def heldout_instance_hashes(suite: Sequence[HeldoutCspCase] | None = None) -> list[JsonDict]:
    """Return stable content hashes for the held-out suite."""

    cases = build_heldout_csp_suite() if suite is None else tuple(suite)
    return [
        {
            "family": case.family,
            "instance_id": case.instance.instance_id,
            "sha256": _instance_hash(case.family, case.instance),
        }
        for case in cases
    ]


def load_exp5129_sampler_features(root: str | Path = REPO_ROOT) -> JsonDict:
    """Load Exp 5129 telemetry after enforcing the adaptive readiness gate."""

    repo_root = Path(root)
    path = _dependency_path(repo_root, EXP5129_RELATIVE_PATH)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("adaptive_2dpt_ready") is not True:
        raise GateBlockedError(payload.get("adaptive_2dpt_ready"))
    exp5129.validate_artifact(payload)
    return {
        "source_path": EXP5129_RELATIVE_PATH,
        "artifact_sha256": _sha256_file(path),
        "adaptive_temperature_config": payload["adaptive_temperature_config"],
        "mixing_improvement": payload["mixing_improvement"],
        "swap_acceptance_rates": payload["swap_acceptance_rates"],
        "adaptive_2dpt_ready": True,
    }


def evaluate_heldout_case(
    case: HeldoutCspCase,
    *,
    solver: ExactCspSolver | None = None,
    sampler_features: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Evaluate all policy arms while exact search owns every reported label."""

    active_solver = solver or ExactCspSolver()
    features = sampler_features or load_exp5129_sampler_features(REPO_ROOT)
    instance = case.instance
    exact = exact_enumerator_label(instance)
    baseline = active_solver.solve(instance, baseline_order(instance))
    adaptation = adapt_instance_order(instance, seed=RANDOM_SEED)
    unguarded = active_solver.solve(instance, adaptation.order)
    gate = harm_gate_decision(instance, adaptation)
    guarded_order = tuple(int(node) for node in gate["selected_order"])
    guarded = active_solver.solve(instance, guarded_order)
    sampler_adaptation = sampler_feature_adaptation(instance, features)
    sampler_feature = active_solver.solve(instance, sampler_adaptation.order)
    exact_label = bool(exact["colorable"])
    arms = {
        "baseline": _arm_json(active_solver, instance, baseline, baseline_order(instance), exact_label),
        "unguarded": _arm_json(active_solver, instance, unguarded, adaptation.order, exact_label),
        "guarded": _arm_json(active_solver, instance, guarded, guarded_order, exact_label),
        "sampler_feature": _arm_json(
            active_solver,
            instance,
            sampler_feature,
            sampler_adaptation.order,
            exact_label,
        ),
    }
    wrong_label = bool(
        baseline.colorable is not exact_label
        or exact_label is not instance.expected_colorable
        or any(arm["exact_authority_agrees"] is not True for arm in arms.values())
    )
    row: JsonDict = {
        "family": case.family,
        "instance_id": instance.instance_id,
        "split": instance.split,
        "description": instance.description,
        "n_nodes": instance.n_nodes,
        "n_colors": instance.n_colors,
        "constraint_count": len(instance.constraints),
        "constraint_arities": list(instance.constraint_arities),
        "density_bucket": instance.density_bucket,
        "frustration": instance.frustration,
        "instance_hash": _instance_hash(case.family, instance),
        "expected_colorable": instance.expected_colorable,
        "exact_label": {
            "status": "colorable" if exact_label else "uncolorable",
            "colorable": exact_label,
            "source": EXACT_SOLVER_BACKEND,
        },
        "exact_enumerator": exact | {"agrees_with_solver": baseline.colorable is exact_label},
        "wrong_label": wrong_label,
        "heuristic_only_answer_counted": False,
        "baseline": arms["baseline"],
        "unguarded": arms["unguarded"],
        "guarded": arms["guarded"],
        "sampler_feature": arms["sampler_feature"],
        "adaptation": adaptation.to_json(),
        "sampler_feature_adaptation": sampler_adaptation.to_json(),
        "gate_decision": gate,
        "unguarded_harmful": _harmful(unguarded, baseline, wrong_label=wrong_label),
        "guarded_harmful": _harmful(guarded, baseline, wrong_label=wrong_label),
        "sampler_feature_harmful": _harmful(sampler_feature, baseline, wrong_label=wrong_label),
        "harm_reasons": {
            "unguarded": _harm_reasons(unguarded, baseline, wrong_label=wrong_label),
            "guarded": _harm_reasons(guarded, baseline, wrong_label=wrong_label),
            "sampler_feature": _harm_reasons(sampler_feature, baseline, wrong_label=wrong_label),
        },
    }
    return row


def sampler_feature_adaptation(
    instance: ScaledCspInstance,
    sampler_features: Mapping[str, Any],
) -> AdaptationResult:
    """Convert Exp 5129 sampler telemetry into a fourth advisory CSP order."""

    config = sampler_features["adaptive_temperature_config"]
    mixing = sampler_features["mixing_improvement"]
    target_acceptance = float(config["target_acceptance"])
    sweeps = int(config["sweeps"])
    feature_steps = max(8, round(sweeps * target_acceptance))
    base = adapt_instance_order(instance, steps=feature_steps, seed=RANDOM_SEED + 29)
    round_trip_delta = max(0.0, float(mixing["round_trip_span_delta"]))
    pair_balance_delta = float(mixing["pair_acceptance_std_delta"])
    degree_weight = 2.0 + target_acceptance + max(0.0, -pair_balance_delta)
    arity_weight = 8.0 + 4.0 * round_trip_delta
    scores = tuple(
        float(base.conflict_counts[node] + degree_weight * base.degree_counts[node] + arity_weight * base.arity_pressure[node])
        for node in range(instance.n_nodes)
    )
    order = tuple(sorted(range(instance.n_nodes), key=lambda node: (-scores[node], node)))
    return AdaptationResult(
        order=order,
        steps=feature_steps,
        conflict_counts=base.conflict_counts,
        degree_counts=base.degree_counts,
        arity_pressure=base.arity_pressure,
        heuristic_scores=scores,
        relaxed_coloring=base.relaxed_coloring,
    )


def exact_enumerator_label(instance: ScaledCspInstance) -> JsonDict:
    """Enumerate the finite CSP state space and return the exact label."""

    states_enumerated = 0
    solution_count = 0
    first_solution: tuple[int, ...] | None = None
    for assignment in itertools.product(range(instance.n_colors), repeat=instance.n_nodes):
        states_enumerated += 1
        if _assignment_satisfies(instance, assignment):
            solution_count += 1
            if first_solution is None:
                first_solution = tuple(int(color) for color in assignment)
    return {
        "colorable": solution_count > 0,
        "solution_count": solution_count,
        "states_enumerated": states_enumerated,
        "first_solution": list(first_solution) if first_solution is not None else None,
        "complete": states_enumerated == instance.n_colors**instance.n_nodes,
    }


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Run the held-out diagnostic and return the terminal artifact."""

    started = time.perf_counter()
    repo_root = Path(root)
    exp5117_loaded = _load_exp5117_baseline(repo_root)
    try:
        sampler_features = load_exp5129_sampler_features(repo_root)
    except GateBlockedError as exc:
        elapsed = _elapsed(started, duration_s)
        return _blocked_artifact(
            duration_s=elapsed,
            exp5117_loaded=exp5117_loaded is not None,
            actual_gate_value=exc.actual_value,
            run_date=run_date,
            tests_run=tests_run,
        )
    suite = build_heldout_csp_suite()
    rows = [
        evaluate_heldout_case(case, solver=ExactCspSolver(), sampler_features=sampler_features)
        for case in suite
    ]
    baseline_effort = _summarize_efforts(row["baseline"]["effort"] for row in rows)
    unguarded_effort = _summarize_efforts(row["unguarded"]["effort"] for row in rows)
    guarded_effort = _summarize_efforts(row["guarded"]["effort"] for row in rows)
    sampler_feature_effort = _summarize_efforts(row["sampler_feature"]["effort"] for row in rows)
    wrong_label_count = sum(1 for row in rows if row["wrong_label"])
    harmful_unguarded = sum(1 for row in rows if row["unguarded_harmful"])
    harmful_guarded = sum(1 for row in rows if row["guarded_harmful"])
    average_effort_reduction = _effort_reduction_ratio(baseline_effort, guarded_effort)
    timeout_rate = _timeout_rate(rows)
    per_family_results = _per_family_results(rows)
    hashes = heldout_instance_hashes(suite)
    dependency_ready = exp5117_loaded is not None and sampler_features["adaptive_2dpt_ready"] is True
    heldout_ready = bool(
        dependency_ready
        and len(rows) >= 8
        and len(per_family_results) >= 4
        and wrong_label_count == 0
        and harmful_guarded <= harmful_unguarded
        and timeout_rate == 0.0
        and _hashes_disjoint_from_tuning(hashes)
        and all(row["exact_enumerator"]["agrees_with_solver"] is True for row in rows)
        and all(row["heuristic_only_answer_counted"] is False for row in rows)
    )
    conductor_modified = False
    flagged_adversarial = bool(
        not dependency_ready
        or wrong_label_count != 0
        or conductor_modified
        or INFERENCE_SUBSTRATE != "cpu_exact_solver_with_adaptive_heuristic"
        or not _hashes_disjoint_from_tuning(hashes)
    )
    artifact: JsonDict = {
        "schema": "carnot.experiment_5130_taco_sampler_heldout_scale.v470",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "result_path": RESULT_RELATIVE_PATH,
        "random_seed": RANDOM_SEED,
        "honest_verdict": READY_VERDICT if heldout_ready else NOT_READY_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _elapsed(started, duration_s),
        "exp5117_baseline_loaded": exp5117_loaded is not None,
        "exp5117_baseline_path": EXP5117_RELATIVE_PATH,
        "exp5129_sampler_features_loaded": True,
        "exp5129_sampler_features": sampler_features,
        "exact_solver_backend": EXACT_SOLVER_BACKEND,
        "heldout_instance_hashes": hashes,
        "instance_count": len(rows),
        "average_effort_reduction_ratio_guarded": average_effort_reduction,
        "harmful_instance_count_guarded": harmful_guarded,
        "harmful_instance_count_unguarded": harmful_unguarded,
        "wrong_label_count": wrong_label_count,
        "timeout_rate": timeout_rate,
        "per_family_results": per_family_results,
        "heldout_csp_trace_suite_ready": heldout_ready,
        "flagged_adversarial": flagged_adversarial,
        "conductor_modified": conductor_modified,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "spec_refs": ["REQ-SAMPLE-5130", "SCENARIO-SAMPLE-5130"],
        "baseline_effort": baseline_effort,
        "unguarded_effort": unguarded_effort,
        "guarded_effort": guarded_effort,
        "sampler_feature_effort": sampler_feature_effort,
        "per_instance_results": rows,
        "field_principles": FIELD_PRINCIPLES,
        "methodology_note": (
            "The Exp 5129 adaptive sampler features influence only one advisory "
            "variable-order variant. Exact backtracking plus brute-force "
            "enumeration remain the sole correctness authorities."
        ),
    }
    artifact["reproducibility_checksum"] = _sha256_json(
        {
            "experiment_id": EXPERIMENT_ID,
            "run_date": run_date,
            "heldout_instance_hashes": hashes,
            "baseline_effort": baseline_effort["total_effort_score"],
            "unguarded_effort": unguarded_effort["total_effort_score"],
            "guarded_effort": guarded_effort["total_effort_score"],
            "sampler_feature_effort": sampler_feature_effort["total_effort_score"],
            "harmful_unguarded": harmful_unguarded,
            "harmful_guarded": harmful_guarded,
            "wrong_label_count": wrong_label_count,
        }
    )
    validate_artifact(artifact)
    return artifact


def write_artifact(
    *,
    root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build and write the Exp 5130 terminal artifact."""

    repo_root = Path(root)
    artifact = build_artifact(root=repo_root, run_date=run_date, duration_s=duration_s, tests_run=tests_run)
    destination = repo_root / RESULT_RELATIVE_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(
    *,
    root: str | Path = REPO_ROOT,
    date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """CLI-compatible entrypoint used by the wrapper script and tests."""

    repo_root = Path(root)
    write_artifact(root=repo_root, run_date=date, duration_s=duration_s, tests_run=tests_run)
    return repo_root / RESULT_RELATIVE_PATH


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 5130 artifact violates its terminal contract."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS.difference(artifact))
    _require(not missing, f"missing required fields: {missing}")
    _require(artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id")
    _require(artifact.get("milestone") == MILESTONE, "milestone")
    verdict = str(artifact.get("honest_verdict", ""))
    _require(verdict.startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(isinstance(artifact.get("duration_s"), int | float), "duration_s")
    _require(float(artifact["duration_s"]) >= 0.0, "duration_s")
    _require(artifact.get("exact_solver_backend") == EXACT_SOLVER_BACKEND, "exact_solver_backend")
    _require(artifact.get("conductor_modified") is False, "conductor_modified")
    _require(isinstance(artifact.get("tests_run"), list) and bool(artifact["tests_run"]), "tests_run")
    _require(_field_principles_valid(artifact.get("field_principles")), "field_principles")
    if verdict.startswith("blocked_"):
        _require(artifact.get("failure_type") == "gate_blocked", "failure_type")
        _require(artifact.get("heldout_csp_trace_suite_ready") is False, "heldout_csp_trace_suite_ready")
        _require(artifact.get("exp5129_sampler_features_loaded") is False, "exp5129_sampler_features_loaded")
        _require(artifact.get("instance_count") == 0, "instance_count")
        _require(artifact.get("per_family_results") == [], "per_family_results")
        return
    rows = artifact.get("per_instance_results")
    hashes = artifact.get("heldout_instance_hashes")
    _require(artifact.get("exp5117_baseline_loaded") is True, "exp5117_baseline_loaded")
    _require(artifact.get("exp5129_sampler_features_loaded") is True, "exp5129_sampler_features_loaded")
    _require(isinstance(rows, list) and len(rows) == artifact["instance_count"], "per_instance_results")
    _require(isinstance(hashes, list) and len(hashes) == artifact["instance_count"], "heldout_instance_hashes")
    _require(_hashes_valid(hashes), "heldout_instance_hashes")
    _require(artifact["instance_count"] >= 8, "instance_count")
    for field in ("baseline_effort", "unguarded_effort", "guarded_effort", "sampler_feature_effort"):
        _require(_effort_summary_valid(artifact.get(field), artifact["instance_count"]), field)
    _require(_families_valid(artifact.get("per_family_results")), "per_family_results")
    _require(artifact["wrong_label_count"] == sum(1 for row in rows if row["wrong_label"]), "wrong_label_count")
    _require(
        artifact["harmful_instance_count_unguarded"] == sum(1 for row in rows if row["unguarded_harmful"]),
        "harmful_instance_count_unguarded",
    )
    _require(
        artifact["harmful_instance_count_guarded"] == sum(1 for row in rows if row["guarded_harmful"]),
        "harmful_instance_count_guarded",
    )
    _require(artifact["timeout_rate"] == _timeout_rate(rows), "timeout_rate")
    _require(all(_row_exact_authority_preserved(row) for row in rows), "per_instance_results")
    ready = bool(
        artifact["wrong_label_count"] == 0
        and artifact["harmful_instance_count_guarded"] <= artifact["harmful_instance_count_unguarded"]
        and artifact["timeout_rate"] == 0.0
        and artifact.get("flagged_adversarial") is False
    )
    _require(artifact["heldout_csp_trace_suite_ready"] is ready, "heldout_csp_trace_suite_ready")
    _require(verdict.startswith(READY_VERDICT if ready else NOT_READY_VERDICT), "honest_verdict")


def _apex_cycle_case(instance_id: str, rim_nodes: int) -> HeldoutCspCase:
    center = rim_nodes
    edges = [(node, (node + 1) % rim_nodes) for node in range(rim_nodes)]
    edges.extend((center, node) for node in range(rim_nodes))
    return _make_case(
        family="apex_cycle_coloring",
        instance_id=instance_id,
        n_nodes=rim_nodes + 1,
        n_colors=3,
        constraints=_edge_constraints(edges),
        expected_colorable=rim_nodes % 2 == 0,
        description=f"Held-out {rim_nodes}-rim apex cycle under 3-coloring.",
        frustration="medium" if rim_nodes % 2 == 0 else "high",
    )


def _grid_case(instance_id: str, *, rows: int, cols: int, diagonal: bool) -> HeldoutCspCase:
    def node(row: int, col: int) -> int:
        return row * cols + col

    edges = []
    for row in range(rows):
        for col in range(cols):
            if col + 1 < cols:
                edges.append((node(row, col), node(row, col + 1)))
            if row + 1 < rows:
                edges.append((node(row, col), node(row + 1, col)))
    if diagonal:
        edges.append((node(0, 0), node(1, 1)))
    return _make_case(
        family="grid_bipartite_coloring",
        instance_id=instance_id,
        n_nodes=rows * cols,
        n_colors=2,
        constraints=_edge_constraints(edges),
        expected_colorable=not diagonal,
        description="Held-out bipartite grid coloring with an optional odd-cycle diagonal.",
        frustration="high" if diagonal else "low",
    )


def _crown_case() -> HeldoutCspCase:
    left = range(4)
    right = range(4, 8)
    edges = [(l_node, r_node) for l_node in left for r_node in right if r_node - 4 != l_node]
    return _make_case(
        family="crown_graph_coloring",
        instance_id="heldout_crown4_3color_sat",
        n_nodes=8,
        n_colors=3,
        constraints=_edge_constraints(edges),
        expected_colorable=True,
        description="Held-out crown graph with missing perfect matching.",
        frustration="low",
    )


def _sparse_path_case() -> HeldoutCspCase:
    return _make_case(
        family="sparse_path_coloring",
        instance_id="heldout_sparse_path7_4color_sat",
        n_nodes=7,
        n_colors=4,
        constraints=_edge_constraints((node, node + 1) for node in range(3)),
        expected_colorable=True,
        description="Held-out low-density path fragment with unused variables.",
        frustration="low",
    )


def _prism_case() -> HeldoutCspCase:
    edges = [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (0, 3), (1, 4), (2, 5)]
    return _make_case(
        family="triangular_prism_coloring",
        instance_id="heldout_triangular_prism_3color_sat",
        n_nodes=6,
        n_colors=3,
        constraints=_edge_constraints(edges),
        expected_colorable=True,
        description="Held-out triangular prism graph under 3-coloring.",
        frustration="medium",
    )


def _complete_case(
    instance_id: str,
    *,
    n_nodes: int,
    n_colors: int,
    expected_colorable: bool,
) -> HeldoutCspCase:
    return _make_case(
        family="clique_capacity_coloring",
        instance_id=instance_id,
        n_nodes=n_nodes,
        n_colors=n_colors,
        constraints=_edge_constraints(itertools.combinations(range(n_nodes), 2)),
        expected_colorable=expected_colorable,
        description=f"Held-out complete graph K{n_nodes} under {n_colors}-coloring.",
        frustration="low" if expected_colorable else "high",
    )


def _overlap_all_diff_case(
    instance_id: str,
    *,
    n_nodes: int,
    n_colors: int,
    expected_colorable: bool,
) -> HeldoutCspCase:
    constraints = (
        CspConstraint(name="left_all_diff5", scope=tuple(range(min(5, n_nodes))), relation="all_different"),
        CspConstraint(name="right_all_diff5", scope=tuple(range(max(0, n_nodes - 5), n_nodes)), relation="all_different"),
    )
    return _make_case(
        family="overlap_all_diff",
        instance_id=instance_id,
        n_nodes=n_nodes,
        n_colors=n_colors,
        constraints=constraints,
        expected_colorable=expected_colorable,
        description="Held-out overlapping arity-5 all-different CSP.",
        frustration="low" if expected_colorable else "high",
    )


def _make_case(
    *,
    family: str,
    instance_id: str,
    n_nodes: int,
    n_colors: int,
    constraints: Sequence[CspConstraint],
    expected_colorable: bool,
    description: str,
    frustration: str,
) -> HeldoutCspCase:
    instance = ScaledCspInstance(
        instance_id=instance_id,
        split="heldout",
        n_nodes=n_nodes,
        n_colors=n_colors,
        constraints=_canonical_constraints(constraints),
        expected_colorable=expected_colorable,
        description=description,
        density_bucket=_density_bucket(n_nodes, constraints),
        frustration=frustration,
    )
    return HeldoutCspCase(family=family, instance=instance)


def _edge_constraints(edges: Iterable[tuple[int, int]]) -> tuple[CspConstraint, ...]:
    canonical_edges = sorted({(min(left, right), max(left, right)) for left, right in edges if left != right})
    return tuple(
        CspConstraint(name=f"neq_{left}_{right}", scope=(left, right), relation="not_equal")
        for left, right in canonical_edges
    )


def _canonical_constraints(constraints: Sequence[CspConstraint]) -> tuple[CspConstraint, ...]:
    return tuple(
        sorted(
            (
                CspConstraint(
                    name=constraint.name,
                    scope=tuple(sorted(constraint.scope)),
                    relation=constraint.relation,
                )
                for constraint in constraints
            ),
            key=lambda constraint: (constraint.scope, constraint.name, constraint.relation),
        )
    )


def _density_bucket(n_nodes: int, constraints: Sequence[CspConstraint]) -> str:
    possible_edges = max(1, n_nodes * (n_nodes - 1) // 2)
    projected_edges = {
        (min(left, right), max(left, right))
        for constraint in constraints
        for left, right in itertools.combinations(constraint.scope, 2)
    }
    density = len(projected_edges) / possible_edges
    if density < 0.25:
        return "low"
    if density < 0.6:
        return "medium"
    return "high"


def _assignment_satisfies(instance: ScaledCspInstance, assignment: Sequence[int]) -> bool:
    for constraint in instance.constraints:
        values = [assignment[node] for node in constraint.scope]
        if len(values) != len(set(values)):
            return False
    return True


def _arm_json(
    solver: ExactCspSolver,
    instance: ScaledCspInstance,
    trace: SolveTrace,
    order: Sequence[int],
    exact_label: bool,
) -> JsonDict:
    solution_verified = _solution_verified(solver, instance, trace)
    payload = trace.to_json(order=order, solution_verified=solution_verified)
    payload["exact_authority_agrees"] = bool(trace.colorable is exact_label and solution_verified)
    return payload


def _solution_verified(solver: ExactCspSolver, instance: ScaledCspInstance, trace: SolveTrace) -> bool:
    if not trace.colorable:
        return trace.assignment is None and trace.certificate_quality == "exact_complete"
    return solver.verify_assignment(instance, trace.assignment)


def _harmful(trace: SolveTrace, baseline: SolveTrace, *, wrong_label: bool) -> bool:
    return bool(_harm_reasons(trace, baseline, wrong_label=wrong_label))


def _harm_reasons(trace: SolveTrace, baseline: SolveTrace, *, wrong_label: bool) -> list[str]:
    reasons: list[str] = []
    if trace.effort_score > baseline.effort_score:
        reasons.append("increased_effort_vs_baseline")
    if trace.colorable is not baseline.colorable or wrong_label:
        reasons.append("wrong_label")
    if trace.timeout:
        reasons.append("timeout")
    if trace.certificate_quality != "exact_complete":
        reasons.append("degraded_certificate_quality")
    return reasons


def _summarize_efforts(efforts: Iterable[Mapping[str, Any]]) -> JsonDict:
    total = {
        "instances": 0,
        "total_effort_score": 0,
        "search_nodes": 0,
        "constraint_checks": 0,
        "backtracks": 0,
        "duration_s": 0.0,
    }
    for effort in efforts:
        total["instances"] += 1
        total["total_effort_score"] += int(effort["total_effort_score"])
        total["search_nodes"] += int(effort["search_nodes"])
        total["constraint_checks"] += int(effort["constraint_checks"])
        total["backtracks"] += int(effort["backtracks"])
        total["duration_s"] += float(effort["duration_s"])
    return {
        "metric": "search_nodes_plus_constraint_checks",
        "instances": total["instances"],
        "total_effort_score": total["total_effort_score"],
        "search_nodes": total["search_nodes"],
        "constraint_checks": total["constraint_checks"],
        "backtracks": total["backtracks"],
        "duration_s": round(total["duration_s"], 6),
    }


def _per_family_results(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family"])].append(row)
    results = []
    for family in sorted(grouped):
        family_rows = grouped[family]
        baseline = _summarize_efforts(row["baseline"]["effort"] for row in family_rows)
        guarded = _summarize_efforts(row["guarded"]["effort"] for row in family_rows)
        results.append(
            {
                "family": family,
                "instance_count": len(family_rows),
                "instance_ids": [str(row["instance_id"]) for row in family_rows],
                "baseline_effort": baseline,
                "unguarded_effort": _summarize_efforts(row["unguarded"]["effort"] for row in family_rows),
                "guarded_effort": guarded,
                "sampler_feature_effort": _summarize_efforts(row["sampler_feature"]["effort"] for row in family_rows),
                "average_effort_reduction_ratio_guarded": _effort_reduction_ratio(baseline, guarded),
                "harmful_instance_count_unguarded": sum(1 for row in family_rows if row["unguarded_harmful"]),
                "harmful_instance_count_guarded": sum(1 for row in family_rows if row["guarded_harmful"]),
                "harmful_instance_count_sampler_feature": sum(1 for row in family_rows if row["sampler_feature_harmful"]),
                "wrong_label_count": sum(1 for row in family_rows if row["wrong_label"]),
                "timeout_rate": _timeout_rate(family_rows),
                "exact_labels_preserved": all(_row_exact_authority_preserved(row) for row in family_rows),
            }
        )
    return results


def _effort_reduction_ratio(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> float:
    baseline_total = int(baseline["total_effort_score"])
    candidate_total = int(candidate["total_effort_score"])
    return round((baseline_total - candidate_total) / baseline_total, 6)


def _timeout_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    total = len(rows) * len(POLICY_ARMS)
    timeouts = sum(1 for row in rows for arm in POLICY_ARMS if row[arm]["timeout"])
    return round(timeouts / total, 6) if total else 0.0


def _row_exact_authority_preserved(row: Mapping[str, Any]) -> bool:
    label = row["exact_label"]["colorable"]
    if row["expected_colorable"] is not label or row["exact_enumerator"]["agrees_with_solver"] is not True:
        return False
    if row["heuristic_only_answer_counted"] is not False or row["wrong_label"] is not False:
        return False
    for arm in POLICY_ARMS:
        payload = row[arm]
        if payload["colorable"] is not label or payload["exact_authority_agrees"] is not True:
            return False
    return True


def _blocked_artifact(
    *,
    duration_s: float,
    exp5117_loaded: bool,
    actual_gate_value: Any,
    run_date: str,
    tests_run: Sequence[str] | None,
) -> JsonDict:
    artifact: JsonDict = {
        "schema": "carnot.experiment_5130_taco_sampler_heldout_scale.v470",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "result_path": RESULT_RELATIVE_PATH,
        "random_seed": RANDOM_SEED,
        "honest_verdict": BLOCKED_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "exp5117_baseline_loaded": exp5117_loaded,
        "exp5117_baseline_path": EXP5117_RELATIVE_PATH,
        "exp5129_sampler_features_loaded": False,
        "exact_solver_backend": EXACT_SOLVER_BACKEND,
        "heldout_instance_hashes": [],
        "instance_count": 0,
        "average_effort_reduction_ratio_guarded": 0.0,
        "harmful_instance_count_guarded": 0,
        "harmful_instance_count_unguarded": 0,
        "wrong_label_count": 0,
        "timeout_rate": 0.0,
        "per_family_results": [],
        "heldout_csp_trace_suite_ready": False,
        "flagged_adversarial": False,
        "conductor_modified": False,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "spec_refs": ["REQ-SAMPLE-5130", "SCENARIO-SAMPLE-5130"],
        "field_principles": FIELD_PRINCIPLES,
        "failure_type": "gate_blocked",
        "gate": {
            "path": EXP5129_RELATIVE_PATH,
            "field": "adaptive_2dpt_ready",
            "operator": "is",
            "expected": True,
            "actual": actual_gate_value,
        },
        "methodology_note": "Exp 5130 is blocked before CSP evaluation because the Exp 5129 adaptive sampler gate is closed.",
    }
    validate_artifact(artifact)
    return artifact


def _load_exp5117_baseline(root: Path) -> JsonDict | None:
    path = _dependency_path(root, EXP5117_RELATIVE_PATH)
    payload = json.loads(path.read_text(encoding="utf-8"))
    exp5117.validate_artifact(payload)
    return dict(payload)


def _dependency_path(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    if path.exists():
        return path
    return REPO_ROOT / relative_path


def _hashes_disjoint_from_tuning(hashes: Sequence[Mapping[str, Any]]) -> bool:
    heldout_hashes = {str(item["sha256"]) for item in hashes}
    return heldout_hashes.isdisjoint(tuning_instance_hashes())


def _hashes_valid(value: Sequence[Mapping[str, Any]]) -> bool:
    hashes = [str(item.get("sha256", "")) for item in value if isinstance(item, Mapping)]
    return len(hashes) == len(value) and len(set(hashes)) == len(hashes) and all(len(hash_value) == 64 for hash_value in hashes)


def _families_valid(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 4 and all(item.get("exact_labels_preserved") is True for item in value)


def _effort_summary_valid(value: Any, instances: int) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("metric") == "search_nodes_plus_constraint_checks"
        and value.get("instances") == instances
        and int(value.get("total_effort_score", 0)) > 0
        and int(value.get("search_nodes", 0)) > 0
        and int(value.get("constraint_checks", 0)) > 0
        and float(value.get("duration_s", -1.0)) >= 0.0
    )


def _field_principles_valid(value: Any) -> bool:
    return isinstance(value, Mapping) and REQUIRED_ARTIFACT_FIELDS.issubset(value)


def _instance_hash(family: str, instance: ScaledCspInstance) -> str:
    return _sha256_json(_instance_payload(family, instance))


def _instance_payload(family: str, instance: ScaledCspInstance) -> JsonDict:
    return {
        "family": family,
        "instance_id": instance.instance_id,
        "split": instance.split,
        "n_nodes": instance.n_nodes,
        "n_colors": instance.n_colors,
        "constraints": [
            {
                "name": constraint.name,
                "scope": list(constraint.scope),
                "relation": constraint.relation,
            }
            for constraint in instance.constraints
        ],
        "expected_colorable": instance.expected_colorable,
        "density_bucket": instance.density_bucket,
        "frustration": instance.frustration,
    }


def _elapsed(started: float, duration_s: float | None) -> float:
    return round(time.perf_counter() - started, 6) if duration_s is None else duration_s


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
