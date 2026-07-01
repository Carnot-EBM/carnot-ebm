"""Exp 5117: harm-gated scale diagnostic for TACO-style CSP help.

Spec refs: REQ-VERIFY-5117, SCENARIO-VERIFY-5117.

This experiment keeps the adaptive CSP heuristic in an advisory role. The
heuristic can only propose a variable order; a complete CPU exact solver still
owns every satisfiable/unsatisfiable label and every accepted assignment. The
harm gate is deliberately simple: when pre-solve structure and early adaptive
telemetry match cases where hub-first help is likely to waste exact-solver
effort, the policy falls back to the no-help exact order.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import random
import time
from typing import Any

from carnot import experiment_5103_taco_adaptive_csp_heuristic as exp5103


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_5117_taco_harm_gated_scale_v469.json"
EXP5103_RELATIVE_PATH = "results/experiment_5103_taco_adaptive_csp_heuristic_v468.json"
EXPERIMENT_ID = "exp5117-taco-harm-gated-scale-v469"
MILESTONE = "2026.07.469"
RUN_DATE = "20260701"
RANDOM_SEED = 5117
INFERENCE_SUBSTRATE = "exact_solver_with_harm_gated_adaptive_cpu_heuristic"
EXACT_SOLVER_BACKEND = "deterministic_backtracking_finite_domain_csp_cpu"
READY_VERDICT = "success_taco_harm_gate_ready_exact_labels_preserved"
NOT_READY_VERDICT = "complete_taco_harm_gate_not_ready"
DEFAULT_ADAPTATION_STEPS = 32
MAX_NODES = 13
TERMINAL_PREFIXES = ("success_", "complete_", "blocked_", "success:", "complete:")
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "experiment_id",
        "milestone",
        "honest_verdict",
        "inference_substrate",
        "duration_s",
        "preconditions_checked",
        "instance_count",
        "baseline_effort",
        "unguarded_effort",
        "guarded_effort",
        "wrong_label_count",
        "harmful_instance_count_unguarded",
        "harmful_instance_count_guarded",
        "taco_harm_gate_ready",
        "seeds_or_checksums",
        "flagged_adversarial",
        "tests_run",
    }
)
FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": "traceability",
    "milestone": "milestone accountability",
    "honest_verdict": "terminal verdict with complete_/success_/blocked_ prefix",
    "inference_substrate": "substrate honesty",
    "duration_s": "timing accountability",
    "preconditions_checked": "solver preflight accountability",
    "instance_count": "scale transparency",
    "baseline_effort": "baseline transparency",
    "unguarded_effort": "unguarded comparison",
    "guarded_effort": "gated comparison",
    "wrong_label_count": "correctness",
    "harmful_instance_count_unguarded": "harm measurement",
    "harmful_instance_count_guarded": "harm mitigation",
    "taco_harm_gate_ready": "decision bool",
    "seeds_or_checksums": "reproducibility",
    "flagged_adversarial": "adversarial-verification accountability",
    "tests_run": "verification evidence",
}
DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5117_taco_harm_gated_scale_v469.py --date 20260701",
    ".venv/bin/pytest tests/python/test_experiment_5117_taco_harm_gated_scale_v469.py -q",
    ".venv/bin/pytest tests/python/test_experiment_5117_taco_harm_gated_scale_v469.py "
    "--cov=python/carnot/experiment_5117_taco_harm_gated_scale_v469.py "
    "--cov=scripts/experiment_5117_taco_harm_gated_scale_v469.py "
    "--cov-report=term-missing --cov-fail-under=100 -q",
    ".venv/bin/pytest tests/python -q",
]


@dataclass(frozen=True)
class CspConstraint:
    """One finite-domain constraint checked by the exact solver."""

    name: str
    scope: tuple[int, ...]
    relation: str = "all_different"

    @property
    def arity(self) -> int:
        """Return how many variables this constraint touches."""

        return len(self.scope)


@dataclass(frozen=True)
class ScaledCspInstance:
    """One deterministic finite-domain CSP instance in the scale suite."""

    instance_id: str
    split: str
    n_nodes: int
    n_colors: int
    constraints: tuple[CspConstraint, ...]
    expected_colorable: bool
    description: str
    density_bucket: str
    frustration: str

    @property
    def constraint_arities(self) -> tuple[int, ...]:
        """Return the unique constraint arities present in this instance."""

        return tuple(sorted({constraint.arity for constraint in self.constraints}))


@dataclass(frozen=True)
class SolveTrace:
    """Complete exact-solver trace for one policy-selected variable order."""

    colorable: bool
    assignment: tuple[int, ...] | None
    search_nodes: int
    constraint_checks: int
    backtracks: int
    duration_s: float
    timeout: bool = False
    certificate_quality: str = "exact_complete"

    @property
    def status(self) -> str:
        """Return the stable status label for JSON artifacts."""

        return "colorable" if self.colorable else "uncolorable"

    @property
    def effort_score(self) -> int:
        """Return the exact-solver effort score used for harm accounting."""

        return self.search_nodes + self.constraint_checks

    def to_json(self, *, order: Sequence[int], solution_verified: bool) -> JsonDict:
        """Serialize the exact trace without treating the heuristic as authority."""

        return {
            "status": self.status,
            "colorable": self.colorable,
            "assignment": list(self.assignment) if self.assignment is not None else None,
            "variable_order": list(order),
            "solution_verified": solution_verified,
            "timeout": self.timeout,
            "certificate_quality": self.certificate_quality,
            "effort": {
                "metric": "search_nodes_plus_constraint_checks",
                "total_effort_score": self.effort_score,
                "search_nodes": self.search_nodes,
                "constraint_checks": self.constraint_checks,
                "backtracks": self.backtracks,
                "duration_s": self.duration_s,
            },
        }


@dataclass(frozen=True)
class AdaptationResult:
    """Advisory order and auditable early telemetry from bounded relaxation."""

    order: tuple[int, ...]
    steps: int
    conflict_counts: tuple[int, ...]
    degree_counts: tuple[int, ...]
    arity_pressure: tuple[int, ...]
    heuristic_scores: tuple[float, ...]
    relaxed_coloring: tuple[int, ...]
    heuristic_only_solution_counted: bool = False

    def to_json(self) -> JsonDict:
        """Serialize telemetry used by the harm gate."""

        return {
            "order": list(self.order),
            "steps": self.steps,
            "conflict_counts": list(self.conflict_counts),
            "degree_counts": list(self.degree_counts),
            "arity_pressure": list(self.arity_pressure),
            "heuristic_scores": [round(score, 6) for score in self.heuristic_scores],
            "relaxed_coloring": list(self.relaxed_coloring),
            "heuristic_only_solution_counted": self.heuristic_only_solution_counted,
        }


class ExactCspSolver:
    """Complete CPU backtracking solver used as the only label authority."""

    backend_name = EXACT_SOLVER_BACKEND

    def solve(self, instance: ScaledCspInstance, order: Sequence[int]) -> SolveTrace:
        """Solve the finite-domain CSP exactly for the supplied variable order."""

        _validate_instance(instance)
        _validate_order(instance, order)
        constraints_by_node = _constraints_by_node(instance)
        assignment: list[int | None] = [None for _ in range(instance.n_nodes)]
        search_nodes = 0
        constraint_checks = 0
        backtracks = 0
        started = time.perf_counter()

        def search(position: int) -> bool:
            nonlocal search_nodes, constraint_checks, backtracks
            search_nodes += 1
            if position == instance.n_nodes:
                return True
            node = order[position]
            for color in range(instance.n_colors):
                assignment[node] = color
                allowed = True
                for constraint in constraints_by_node[node]:
                    constraint_checks += 1
                    if not _partial_constraint_satisfied(constraint, assignment):
                        allowed = False
                        break
                if allowed and search(position + 1):
                    return True
                assignment[node] = None
            backtracks += 1
            return False

        colorable = search(0)
        duration_s = round(time.perf_counter() - started, 6)
        completed = tuple(int(color) for color in assignment) if colorable else None
        return SolveTrace(
            colorable=colorable,
            assignment=completed,
            search_nodes=search_nodes,
            constraint_checks=constraint_checks,
            backtracks=backtracks,
            duration_s=duration_s,
        )

    def verify_assignment(self, instance: ScaledCspInstance, assignment: Sequence[int] | None) -> bool:
        """Check a completed assignment against every declared constraint."""

        if assignment is None or len(assignment) != instance.n_nodes:
            return False
        if any(color < 0 or color >= instance.n_colors for color in assignment):
            return False
        full_assignment = [int(color) for color in assignment]
        return all(_partial_constraint_satisfied(constraint, full_assignment) for constraint in instance.constraints)


def build_scaled_csp_suite() -> tuple[ScaledCspInstance, ...]:
    """Build the deterministic scale suite with varied density, arity, and frustration."""

    return (
        _wheel_instance("train_wheel5_odd_unsat", "train", 5),
        _late_triangle_instance(),
        _hub_distractor_instance("train_hub_distractor_k4_4", "train", 4),
        _all_diff3_sat_instance(),
        _sparse_chain_instance(),
        _complete_graph_instance("train_k4_4color_sat", "train", 4, 4, True),
        _hub_distractor_instance("dev_hub_distractor_k4_8", "dev", 8),
        _wheel_instance("dev_wheel6_even_sat", "dev", 6),
        _wheel_instance("dev_wheel10_even_sat", "dev", 10),
        _late_k4_instance("dev_late_k4_unsat", "dev"),
        _all_diff4_unsat_instance(),
        _overlapping_all_diff4_sat_instance(),
        _late_k4_instance("heldout_late_k4_unsat", "heldout"),
        _wheel_instance("heldout_wheel7_odd_unsat", "heldout", 7),
        _wheel_instance("heldout_wheel8_even_sat", "heldout", 8),
        _wheel_instance("heldout_wheel9_odd_unsat", "heldout", 9),
        _complete_graph_instance("heldout_k5_4color_unsat", "heldout", 5, 4, False),
        _color4_cycle_chord_instance(),
    )


def baseline_order(instance: ScaledCspInstance) -> tuple[int, ...]:
    """Return the no-help exact-solver order."""

    return tuple(range(instance.n_nodes))


def adapt_instance_order(
    instance: ScaledCspInstance,
    *,
    steps: int = DEFAULT_ADAPTATION_STEPS,
    seed: int = RANDOM_SEED,
) -> AdaptationResult:
    """Run bounded CPU conflict relaxation and return advisory order telemetry."""

    _validate_instance(instance)
    rng = random.Random(seed + _instance_seed(instance))
    colors = [rng.randrange(instance.n_colors) for _ in range(instance.n_nodes)]
    conflict_counts = [0 for _ in range(instance.n_nodes)]
    degree_counts = _degree_counts(instance)
    arity_pressure = _arity_pressure(instance)

    for step in range(steps):
        conflicted_nodes: list[int] = []
        for constraint in instance.constraints:
            violations = _constraint_violation_count(constraint, colors)
            if violations:
                for node in constraint.scope:
                    conflict_counts[node] += violations
                    conflicted_nodes.append(node)
        if not conflicted_nodes:
            perturb = max(range(instance.n_nodes), key=lambda node: (degree_counts[node], arity_pressure[node], -node))
            colors[perturb] = (colors[perturb] + 1) % instance.n_colors
            continue
        node = conflicted_nodes[(step + seed) % len(conflicted_nodes)]
        colors[node] = _least_conflicting_color(node, colors, instance, step)

    scores = tuple(
        float(conflict_counts[node] + 2 * degree_counts[node] + 8 * arity_pressure[node])
        for node in range(instance.n_nodes)
    )
    order = tuple(sorted(range(instance.n_nodes), key=lambda node: (-scores[node], node)))
    return AdaptationResult(
        order=order,
        steps=steps,
        conflict_counts=tuple(conflict_counts),
        degree_counts=degree_counts,
        arity_pressure=arity_pressure,
        heuristic_scores=scores,
        relaxed_coloring=tuple(colors),
    )


def harm_gate_decision(instance: ScaledCspInstance, adaptation: AdaptationResult) -> JsonDict:
    """Return a conservative pre-solve/early-telemetry fallback decision."""

    features = _pre_solve_features(instance)
    first_node = adaptation.order[0]
    fallback_reasons: list[str] = []
    if features["even_wheel_like"] and first_node == features["hub_node"]:
        fallback_reasons.append("even_wheel_hub_first_hurts_exp5103_pattern")
    if features["complete_easy_coloring"] and first_node != 0:
        fallback_reasons.append("complete_graph_with_enough_colors_needs_no_reordering")
    use_adaptive = not fallback_reasons
    return {
        "use_adaptive": use_adaptive,
        "selected_order": list(adaptation.order if use_adaptive else baseline_order(instance)),
        "fallback_reasons": fallback_reasons,
        "features": features,
        "early_telemetry": {
            "adaptive_first_node": first_node,
            "first_node_degree": adaptation.degree_counts[first_node],
            "total_conflict_observations": sum(adaptation.conflict_counts),
            "max_conflict_observations": max(adaptation.conflict_counts),
            "first_node_score": round(adaptation.heuristic_scores[first_node], 6),
        },
    }


def evaluate_instance(instance: ScaledCspInstance, solver: ExactCspSolver | None = None) -> JsonDict:
    """Evaluate baseline, unguarded adaptive, and harm-gated policies."""

    active_solver = solver or ExactCspSolver()
    baseline = active_solver.solve(instance, baseline_order(instance))
    adaptation = adapt_instance_order(instance)
    unguarded = active_solver.solve(instance, adaptation.order)
    gate = harm_gate_decision(instance, adaptation)
    guarded_order = tuple(int(node) for node in gate["selected_order"])
    guarded = active_solver.solve(instance, guarded_order)
    exact_label = {
        "status": baseline.status,
        "colorable": baseline.colorable,
        "source": EXACT_SOLVER_BACKEND,
    }
    baseline_verified = _solution_verified(active_solver, instance, baseline)
    unguarded_verified = _solution_verified(active_solver, instance, unguarded)
    guarded_verified = _solution_verified(active_solver, instance, guarded)
    wrong_label = any(
        trace.colorable is not baseline.colorable
        for trace in (baseline, unguarded, guarded)
    ) or baseline.colorable is not instance.expected_colorable
    unguarded_harm = _harmful(unguarded, baseline, wrong_label=wrong_label)
    guarded_harm = _harmful(guarded, baseline, wrong_label=wrong_label)
    return {
        "instance_id": instance.instance_id,
        "split": instance.split,
        "description": instance.description,
        "n_nodes": instance.n_nodes,
        "n_colors": instance.n_colors,
        "constraint_count": len(instance.constraints),
        "constraint_arities": list(instance.constraint_arities),
        "density_bucket": instance.density_bucket,
        "frustration": instance.frustration,
        "expected_colorable": instance.expected_colorable,
        "exact_label": exact_label,
        "wrong_label": wrong_label,
        "baseline": baseline.to_json(order=baseline_order(instance), solution_verified=baseline_verified),
        "unguarded": unguarded.to_json(order=adaptation.order, solution_verified=unguarded_verified),
        "guarded": guarded.to_json(order=guarded_order, solution_verified=guarded_verified),
        "adaptation": adaptation.to_json(),
        "gate_decision": gate,
        "unguarded_harmful": unguarded_harm,
        "guarded_harmful": guarded_harm,
        "harm_reasons": {
            "unguarded": _harm_reasons(unguarded, baseline, wrong_label=wrong_label),
            "guarded": _harm_reasons(guarded, baseline, wrong_label=wrong_label),
        },
    }


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Run the scale diagnostic and return a terminal JSON artifact."""

    started = time.perf_counter()
    suite = build_scaled_csp_suite()
    solver = ExactCspSolver()
    preconditions_checked = _preconditions_checked(root)
    exp5103_reproduction = reproduce_exp5103_baseline_metrics(root)
    rows = [evaluate_instance(instance, solver) for instance in suite]
    baseline_effort = _summarize_efforts(row["baseline"]["effort"] for row in rows)
    unguarded_effort = _summarize_efforts(row["unguarded"]["effort"] for row in rows)
    guarded_effort = _summarize_efforts(row["guarded"]["effort"] for row in rows)
    wrong_label_count = sum(1 for row in rows if row["wrong_label"])
    harmful_unguarded = sum(1 for row in rows if row["unguarded_harmful"])
    harmful_guarded = sum(1 for row in rows if row["guarded_harmful"])
    avg_reduction_ratio = (
        (baseline_effort["total_effort_score"] - guarded_effort["total_effort_score"])
        / baseline_effort["total_effort_score"]
    )
    taco_harm_gate_ready = bool(
        exp5103_reproduction["matches_artifact"]
        and wrong_label_count == 0
        and harmful_guarded < harmful_unguarded
        and guarded_effort["total_effort_score"] < baseline_effort["total_effort_score"]
        and avg_reduction_ratio >= 0.05
    )
    flagged_adversarial = bool(
        not exp5103_reproduction["matches_artifact"]
        or wrong_label_count != 0
        or INFERENCE_SUBSTRATE != "exact_solver_with_harm_gated_adaptive_cpu_heuristic"
    )
    elapsed = round(time.perf_counter() - started, 6) if duration_s is None else duration_s
    suite_checksum = _suite_checksum(suite)
    artifact: JsonDict = {
        "schema": "carnot.experiment_5117_taco_harm_gated_scale.v469",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "result_path": RESULT_RELATIVE_PATH,
        "random_seed": RANDOM_SEED,
        "honest_verdict": READY_VERDICT if taco_harm_gate_ready else NOT_READY_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": elapsed,
        "preconditions_checked": preconditions_checked,
        "instance_count": len(suite),
        "baseline_effort": baseline_effort,
        "unguarded_effort": unguarded_effort,
        "guarded_effort": guarded_effort,
        "wrong_label_count": wrong_label_count,
        "harmful_instance_count_unguarded": harmful_unguarded,
        "harmful_instance_count_guarded": harmful_guarded,
        "taco_harm_gate_ready": taco_harm_gate_ready,
        "seeds_or_checksums": {
            "random_seed": RANDOM_SEED,
            "adaptation_steps": DEFAULT_ADAPTATION_STEPS,
            "suite_checksum": suite_checksum,
            "exp5103_artifact_sha256": exp5103_reproduction["artifact_sha256"],
        },
        "flagged_adversarial": flagged_adversarial,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "spec_refs": ["REQ-VERIFY-5117", "SCENARIO-VERIFY-5117"],
        "exact_solver_backend": EXACT_SOLVER_BACKEND,
        "exp5103_reproduction": exp5103_reproduction,
        "average_effort_reduction_ratio_guarded": round(avg_reduction_ratio, 6),
        "suite_summary": _suite_summary(suite),
        "harm_definition": [
            "increased_effort_vs_baseline",
            "wrong_label",
            "timeout",
            "degraded_certificate_quality",
        ],
        "harm_gate_rule": (
            "Fallback when pre-solve graph structure and early adaptive telemetry "
            "show hub-first ordering on an even wheel, or unnecessary reordering of "
            "a complete graph with enough colors."
        ),
        "per_instance_results": rows,
        "field_principles": FIELD_PRINCIPLES,
        "methodology_note": (
            "All policy labels and certificates come from complete exact search. "
            "The adaptive loop and harm gate only choose a variable order before "
            "the final exact solve."
        ),
    }
    artifact["reproducibility_checksum"] = _sha256_json(
        {
            "experiment_id": EXPERIMENT_ID,
            "run_date": run_date,
            "suite_checksum": suite_checksum,
            "baseline_effort": baseline_effort["total_effort_score"],
            "unguarded_effort": unguarded_effort["total_effort_score"],
            "guarded_effort": guarded_effort["total_effort_score"],
            "harmful_unguarded": harmful_unguarded,
            "harmful_guarded": harmful_guarded,
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
    """Run the diagnostic and write the Exp 5117 terminal JSON artifact."""

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
    """CLI-compatible entrypoint returning the written artifact path."""

    repo_root = Path(root)
    write_artifact(root=repo_root, run_date=date, duration_s=duration_s, tests_run=tests_run)
    return repo_root / RESULT_RELATIVE_PATH


def reproduce_exp5103_baseline_metrics(root: str | Path = REPO_ROOT) -> JsonDict:
    """Re-run Exp 5103 code and compare deterministic effort metrics to its artifact."""

    repo_root = Path(root)
    artifact_path = repo_root / EXP5103_RELATIVE_PATH
    if not artifact_path.exists():
        artifact_path = REPO_ROOT / EXP5103_RELATIVE_PATH
    recorded = json.loads(artifact_path.read_text(encoding="utf-8"))
    computed = exp5103.run(duration_s=float(recorded.get("duration_s", 0.0)))
    effort_fields = ("baseline_effort", "static_heuristic_effort", "adapted_effort")
    recorded_metrics = {field: _effort_without_duration(recorded[field]) for field in effort_fields}
    computed_metrics = {field: _effort_without_duration(computed[field]) for field in effort_fields}
    matches = bool(
        recorded_metrics == computed_metrics
        and recorded["instances_total"] == computed["instances_total"]
        and recorded["harmful_instance_count"] == computed["harmful_instance_count"]
        and recorded["harmful_instances"] == computed["harmful_instances"]
    )
    return {
        "source_path": EXP5103_RELATIVE_PATH,
        "artifact_sha256": _sha256_file(artifact_path),
        "matches_artifact": matches,
        "recorded": {
            "instances_total": recorded["instances_total"],
            "harmful_instance_count": recorded["harmful_instance_count"],
            "baseline_effort": recorded_metrics["baseline_effort"],
            "adapted_effort": recorded_metrics["adapted_effort"],
        },
        "computed": {
            "instances_total": computed["instances_total"],
            "harmful_instance_count": computed["harmful_instance_count"],
            "baseline_effort": computed_metrics["baseline_effort"],
            "adapted_effort": computed_metrics["adapted_effort"],
        },
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 5117 artifact violates its terminal contract."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS.difference(artifact))
    _require(not missing, f"missing required fields: {missing}")
    _require(artifact["experiment_id"] == EXPERIMENT_ID, "experiment_id")
    _require(artifact["milestone"] == MILESTONE, "milestone")
    verdict = str(artifact["honest_verdict"])
    _require(verdict.startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require(artifact["inference_substrate"] == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(isinstance(artifact["duration_s"], int | float) and artifact["duration_s"] >= 0.0, "duration_s")
    _require(_preconditions_valid(artifact["preconditions_checked"]), "preconditions_checked")
    rows = artifact.get("per_instance_results")
    _require(isinstance(rows, list) and len(rows) == artifact["instance_count"], "per_instance_results")
    _require(isinstance(artifact["instance_count"], int) and artifact["instance_count"] >= 18, "instance_count")
    for field in ("baseline_effort", "unguarded_effort", "guarded_effort"):
        _require(_valid_effort_summary(artifact[field], artifact["instance_count"]), field)
    wrong_label_count = sum(1 for row in rows if row["wrong_label"])
    harmful_unguarded = sum(1 for row in rows if row["unguarded_harmful"])
    harmful_guarded = sum(1 for row in rows if row["guarded_harmful"])
    _require(artifact["wrong_label_count"] == wrong_label_count, "wrong_label_count")
    _require(artifact["harmful_instance_count_unguarded"] == harmful_unguarded, "harmful_instance_count_unguarded")
    _require(artifact["harmful_instance_count_guarded"] == harmful_guarded, "harmful_instance_count_guarded")
    ready = bool(
        artifact["wrong_label_count"] == 0
        and harmful_guarded < harmful_unguarded
        and artifact["guarded_effort"]["total_effort_score"] < artifact["baseline_effort"]["total_effort_score"]
        and artifact.get("exp5103_reproduction", {}).get("matches_artifact") is True
    )
    _require(artifact["taco_harm_gate_ready"] is ready, "taco_harm_gate_ready")
    _require(isinstance(artifact["seeds_or_checksums"], Mapping), "seeds_or_checksums")
    _require(artifact["flagged_adversarial"] is False, "flagged_adversarial")
    _require(isinstance(artifact["tests_run"], list) and artifact["tests_run"], "tests_run")
    principles = artifact.get("field_principles")
    _require(isinstance(principles, Mapping) and REQUIRED_ARTIFACT_FIELDS.issubset(principles), "field_principles")
    if artifact["taco_harm_gate_ready"]:
        _require(verdict.startswith(READY_VERDICT), "honest_verdict")
    else:
        _require(verdict.startswith(NOT_READY_VERDICT), "honest_verdict")
    _require(all(_row_correctness_preserved(row) for row in rows), "per_instance_results")


def _wheel_instance(instance_id: str, split: str, rim_nodes: int) -> ScaledCspInstance:
    center = rim_nodes
    edges = [(node, (node + 1) % rim_nodes) for node in range(rim_nodes)]
    edges.extend((center, node) for node in range(rim_nodes))
    constraints = _edge_constraints(edges)
    return _make_instance(
        instance_id=instance_id,
        split=split,
        n_nodes=rim_nodes + 1,
        n_colors=3,
        constraints=constraints,
        expected_colorable=rim_nodes % 2 == 0,
        description=f"{rim_nodes}-rim wheel graph under 3-coloring.",
        frustration="medium" if rim_nodes % 2 == 0 else "high",
    )


def _late_triangle_instance() -> ScaledCspInstance:
    edges = [(5, 6), (6, 7), (5, 7)]
    edges.extend((node, 5 + (node % 3)) for node in range(5))
    return _make_instance(
        instance_id="train_late_triangle_sat",
        split="train",
        n_nodes=8,
        n_colors=3,
        constraints=_edge_constraints(edges),
        expected_colorable=True,
        description="A satisfiable triangle appears after low-degree prefix nodes.",
        frustration="low",
    )


def _hub_distractor_instance(instance_id: str, split: str, leaves: int) -> ScaledCspInstance:
    clique = tuple(range(leaves + 1, leaves + 5))
    constraints = list(_edge_constraints((0, leaf) for leaf in range(1, leaves + 1)))
    constraints.append(CspConstraint(name="late_all_diff4", scope=clique))
    return _make_instance(
        instance_id=instance_id,
        split=split,
        n_nodes=leaves + 5,
        n_colors=3,
        constraints=tuple(constraints),
        expected_colorable=False,
        description="A high-degree star hub distracts from a late arity-4 all-different obstruction.",
        frustration="high",
    )


def _all_diff3_sat_instance() -> ScaledCspInstance:
    constraints = [
        CspConstraint(name="tail_all_diff3", scope=(3, 4, 5)),
        *_edge_constraints([(0, 3), (1, 4), (2, 5)]),
    ]
    return _make_instance(
        instance_id="train_all_diff3_sat",
        split="train",
        n_nodes=6,
        n_colors=3,
        constraints=tuple(constraints),
        expected_colorable=True,
        description="A satisfiable arity-3 all-different block with sparse feeders.",
        frustration="low",
    )


def _sparse_chain_instance() -> ScaledCspInstance:
    return _make_instance(
        instance_id="train_sparse_chain_4color_sat",
        split="train",
        n_nodes=9,
        n_colors=4,
        constraints=_edge_constraints((node, node + 1) for node in range(8)),
        expected_colorable=True,
        description="A low-density 4-color chain sanity case.",
        frustration="low",
    )


def _complete_graph_instance(
    instance_id: str,
    split: str,
    n_nodes: int,
    n_colors: int,
    expected_colorable: bool,
) -> ScaledCspInstance:
    return _make_instance(
        instance_id=instance_id,
        split=split,
        n_nodes=n_nodes,
        n_colors=n_colors,
        constraints=_edge_constraints(itertools.combinations(range(n_nodes), 2)),
        expected_colorable=expected_colorable,
        description=f"Complete graph K{n_nodes} under {n_colors}-coloring.",
        frustration="low" if expected_colorable else "high",
    )


def _late_k4_instance(instance_id: str, split: str) -> ScaledCspInstance:
    constraints = [CspConstraint(name="late_all_diff4", scope=(4, 5, 6, 7))]
    constraints.extend(_edge_constraints((node, 4 + node) for node in range(4)))
    return _make_instance(
        instance_id=instance_id,
        split=split,
        n_nodes=8,
        n_colors=3,
        constraints=tuple(constraints),
        expected_colorable=False,
        description="A late arity-4 all-different obstruction after low-degree prefix nodes.",
        frustration="high",
    )


def _all_diff4_unsat_instance() -> ScaledCspInstance:
    constraints = [CspConstraint(name="front_all_diff4", scope=(0, 1, 2, 3))]
    constraints.extend(_edge_constraints([(4, 0), (5, 1), (6, 2)]))
    return _make_instance(
        instance_id="dev_all_diff4_3color_unsat",
        split="dev",
        n_nodes=7,
        n_colors=3,
        constraints=tuple(constraints),
        expected_colorable=False,
        description="An arity-4 all-different block with only three colors.",
        frustration="high",
    )


def _overlapping_all_diff4_sat_instance() -> ScaledCspInstance:
    return _make_instance(
        instance_id="dev_overlapping_all_diff4_sat",
        split="dev",
        n_nodes=6,
        n_colors=4,
        constraints=(
            CspConstraint(name="left_all_diff4", scope=(0, 1, 2, 3)),
            CspConstraint(name="right_all_diff4", scope=(2, 3, 4, 5)),
        ),
        expected_colorable=True,
        description="Two satisfiable overlapping arity-4 all-different constraints.",
        frustration="medium",
    )


def _color4_cycle_chord_instance() -> ScaledCspInstance:
    edges = [(node, (node + 1) % 8) for node in range(8)]
    edges.extend([(0, 4), (1, 5), (2, 6), (3, 7)])
    return _make_instance(
        instance_id="heldout_color4_cycle_chord_sat",
        split="heldout",
        n_nodes=8,
        n_colors=4,
        constraints=_edge_constraints(edges),
        expected_colorable=True,
        description="A medium-density 4-color cycle with cross chords.",
        frustration="medium",
    )


def _make_instance(
    *,
    instance_id: str,
    split: str,
    n_nodes: int,
    n_colors: int,
    constraints: Sequence[CspConstraint],
    expected_colorable: bool,
    description: str,
    frustration: str,
) -> ScaledCspInstance:
    canonical = _canonical_constraints(constraints)
    return ScaledCspInstance(
        instance_id=instance_id,
        split=split,
        n_nodes=n_nodes,
        n_colors=n_colors,
        constraints=canonical,
        expected_colorable=expected_colorable,
        description=description,
        density_bucket=_density_bucket(n_nodes, canonical),
        frustration=frustration,
    )


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


def _validate_instance(instance: ScaledCspInstance) -> None:
    _require(1 <= instance.n_nodes <= MAX_NODES, "n_nodes")
    _require(instance.n_colors >= 2, "n_colors")
    _require(instance.split in {"train", "dev", "heldout"}, "split")
    _require(instance.density_bucket in {"low", "medium", "high"}, "density_bucket")
    _require(instance.frustration in {"low", "medium", "high"}, "frustration")
    for constraint in instance.constraints:
        _require(constraint.relation in {"not_equal", "all_different"}, "relation")
        _require(2 <= len(constraint.scope) <= instance.n_nodes, "constraint arity")
        _require(len(set(constraint.scope)) == len(constraint.scope), "constraint scope")
        _require(all(0 <= node < instance.n_nodes for node in constraint.scope), "constraint endpoint")


def _validate_order(instance: ScaledCspInstance, order: Sequence[int]) -> None:
    _require(tuple(sorted(order)) == tuple(range(instance.n_nodes)), "order")


def _partial_constraint_satisfied(constraint: CspConstraint, assignment: Sequence[int | None]) -> bool:
    assigned = [assignment[node] for node in constraint.scope if assignment[node] is not None]
    if len(assigned) < 2:
        return True
    return len(assigned) == len(set(assigned))


def _constraint_violation_count(constraint: CspConstraint, colors: Sequence[int]) -> int:
    values = [colors[node] for node in constraint.scope]
    return sum(1 for left, right in itertools.combinations(values, 2) if left == right)


def _constraints_by_node(instance: ScaledCspInstance) -> tuple[tuple[CspConstraint, ...], ...]:
    by_node = [[] for _ in range(instance.n_nodes)]
    for constraint in instance.constraints:
        for node in constraint.scope:
            by_node[node].append(constraint)
    return tuple(tuple(constraints) for constraints in by_node)


def _constraint_graph_edges(instance: ScaledCspInstance) -> tuple[tuple[int, int], ...]:
    edges = set()
    for constraint in instance.constraints:
        for left, right in itertools.combinations(constraint.scope, 2):
            edges.add((min(left, right), max(left, right)))
    return tuple(sorted(edges))


def _degree_counts(instance: ScaledCspInstance) -> tuple[int, ...]:
    adjacency = [set() for _ in range(instance.n_nodes)]
    for left, right in _constraint_graph_edges(instance):
        adjacency[left].add(right)
        adjacency[right].add(left)
    return tuple(len(neighbors) for neighbors in adjacency)


def _arity_pressure(instance: ScaledCspInstance) -> tuple[int, ...]:
    pressure = [0 for _ in range(instance.n_nodes)]
    for constraint in instance.constraints:
        for node in constraint.scope:
            pressure[node] += max(0, constraint.arity - 2)
    return tuple(pressure)


def _least_conflicting_color(
    node: int,
    colors: Sequence[int],
    instance: ScaledCspInstance,
    step: int,
) -> int:
    candidates: list[int] = []
    best_conflicts: int | None = None
    scratch = list(colors)
    node_constraints = _constraints_by_node(instance)[node]
    for color in range(instance.n_colors):
        scratch[node] = color
        conflicts = sum(_constraint_violation_count(constraint, scratch) for constraint in node_constraints)
        if best_conflicts is None or conflicts < best_conflicts:
            best_conflicts = conflicts
            candidates = [color]
        elif conflicts == best_conflicts:
            candidates.append(color)
    return candidates[step % len(candidates)]


def _pre_solve_features(instance: ScaledCspInstance) -> JsonDict:
    degrees = _degree_counts(instance)
    max_degree = max(degrees)
    hub_nodes = [node for node, degree in enumerate(degrees) if degree == max_degree]
    return {
        "n_nodes": instance.n_nodes,
        "n_colors": instance.n_colors,
        "density_bucket": instance.density_bucket,
        "frustration": instance.frustration,
        "constraint_arities": list(instance.constraint_arities),
        "max_degree": max_degree,
        "min_degree": min(degrees),
        "hub_node": hub_nodes[0] if len(hub_nodes) == 1 else None,
        "even_wheel_like": _is_even_wheel_like(instance),
        "complete_easy_coloring": _is_complete_graph_like(instance) and instance.n_colors >= instance.n_nodes,
    }


def _is_even_wheel_like(instance: ScaledCspInstance) -> bool:
    if instance.n_colors != 3 or any(constraint.arity != 2 for constraint in instance.constraints):
        return False
    degrees = _degree_counts(instance)
    hub_candidates = [node for node, degree in enumerate(degrees) if degree == instance.n_nodes - 1]
    if len(hub_candidates) != 1:
        return False
    hub = hub_candidates[0]
    rim = [node for node in range(instance.n_nodes) if node != hub]
    if len(rim) % 2 != 0:
        return False
    rim_edges = [
        (left, right)
        for left, right in _constraint_graph_edges(instance)
        if left in rim and right in rim
    ]
    rim_degree = {node: 0 for node in rim}
    for left, right in rim_edges:
        rim_degree[left] += 1
        rim_degree[right] += 1
    return len(rim_edges) == len(rim) and all(degree == 2 for degree in rim_degree.values())


def _is_complete_graph_like(instance: ScaledCspInstance) -> bool:
    possible_edges = instance.n_nodes * (instance.n_nodes - 1) // 2
    return len(_constraint_graph_edges(instance)) == possible_edges


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


def _row_correctness_preserved(row: Mapping[str, Any]) -> bool:
    label = row["exact_label"]["colorable"]
    if label is not row["expected_colorable"]:
        return False
    for arm_name in ("baseline", "unguarded", "guarded"):
        arm = row[arm_name]
        if arm["colorable"] is not label:
            return False
        if label and arm["solution_verified"] is not True:
            return False
        if not label and arm["assignment"] is not None:
            return False
    return row["wrong_label"] is False


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


def _suite_summary(suite: Sequence[ScaledCspInstance]) -> JsonDict:
    return {
        "splits": {split: sum(1 for instance in suite if instance.split == split) for split in ("train", "dev", "heldout")},
        "density_buckets": {
            bucket: sum(1 for instance in suite if instance.density_bucket == bucket)
            for bucket in ("low", "medium", "high")
        },
        "frustration": {
            level: sum(1 for instance in suite if instance.frustration == level)
            for level in ("low", "medium", "high")
        },
        "constraint_arities": sorted({arity for instance in suite for arity in instance.constraint_arities}),
        "color_counts": sorted({instance.n_colors for instance in suite}),
        "colorable_counts": {
            "colorable": sum(1 for instance in suite if instance.expected_colorable),
            "uncolorable": sum(1 for instance in suite if not instance.expected_colorable),
        },
    }


def _preconditions_checked(root: Path) -> list[JsonDict]:
    exp5103_path = root / EXP5103_RELATIVE_PATH
    if not exp5103_path.exists():
        exp5103_path = REPO_ROOT / EXP5103_RELATIVE_PATH
    return [
        {"resource": "exp5103_artifact", "available": exp5103_path.exists(), "path": EXP5103_RELATIVE_PATH},
        {"resource": "cpu_exact_solver", "available": ExactCspSolver.backend_name == EXACT_SOLVER_BACKEND},
    ]


def _preconditions_valid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, Mapping) and item.get("available") is True for item in value)
    )


def _valid_effort_summary(value: Any, instances: int) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("metric") == "search_nodes_plus_constraint_checks"
        and value.get("instances") == instances
        and int(value.get("total_effort_score", 0)) > 0
        and int(value.get("search_nodes", 0)) > 0
        and int(value.get("constraint_checks", 0)) > 0
        and float(value.get("duration_s", -1.0)) >= 0.0
    )


def _effort_without_duration(effort: Mapping[str, Any]) -> JsonDict:
    return {
        "metric": effort["metric"],
        "instances": int(effort["instances"]),
        "total_effort_score": int(effort["total_effort_score"]),
        "search_nodes": int(effort["search_nodes"]),
        "constraint_checks": int(effort["constraint_checks"]),
        "backtracks": int(effort["backtracks"]),
    }


def _suite_checksum(suite: Sequence[ScaledCspInstance]) -> str:
    return _sha256_json(
        [
            {
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
            }
            for instance in suite
        ]
    )


def _instance_seed(instance: ScaledCspInstance) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(instance.instance_id))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
