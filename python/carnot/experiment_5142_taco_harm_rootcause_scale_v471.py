"""Exp 5142: root-cause and repair harmful TACO/CSP sampler guidance.

Spec refs: REQ-SAMPLE-5142, SCENARIO-SAMPLE-5142.

The sampler-derived features in this experiment never own a label. They only
choose between exact-solver variable orders, and every reported satisfiable or
unsatisfiable label is checked by the complete CPU solver plus the brute-force
enumerator inherited from Exp 5130. This matters because the next self-learning
stage must learn from effort traces, not from a sampler that silently became a
correctness oracle.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot import experiment_5130_taco_sampler_heldout_scale_v470 as exp5130
from carnot.experiment_5117_taco_harm_gated_scale_v469 import (
    CspConstraint,
    ExactCspSolver,
    ScaledCspInstance,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_5142_taco_harm_rootcause_scale_v471.json"
EXP5130_RELATIVE_PATH = exp5130.RESULT_RELATIVE_PATH
EXP5130_ALTERNATE_RELATIVE_PATHS = (
    EXP5130_RELATIVE_PATH,
    "results/experiment_5130_taco_heldout_csp_trace_suite_v470.json",
)
EXPERIMENT_ID = "exp5142-taco-harm-rootcause-scale-v471"
MILESTONE = "2026.07.471"
RUN_DATE = "20260702"
RANDOM_SEED = 5142
INFERENCE_SUBSTRATE = "exact_checked_taco_csp_trace_suite"
READY_VERDICT = "success_trace_suite_v2_ready_harm_gate_repaired_exact_labels_preserved"
NOT_READY_VERDICT = "complete_trace_suite_v2_not_ready_harm_gate_insufficient"
TERMINAL_PREFIXES = ("success_", "complete_", "blocked_", "success:", "complete:", "blocked:")
POLICY_ARMS = ("baseline", "guarded", "sampler_feature", "repaired_guarded")
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "experiment_id",
        "milestone",
        "honest_verdict",
        "inference_substrate",
        "duration_s",
        "exp5130_baseline_loaded",
        "instance_count",
        "task_families",
        "harmful_instance_root_causes",
        "average_effort_reduction_ratio_guarded",
        "harmful_instance_count_guarded",
        "wrong_label_count",
        "repaired_harm_gate",
        "ablation_results",
        "trace_suite_v2_ready",
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
    "exp5130_baseline_loaded": "upstream evidence",
    "instance_count": "sample-size accountability",
    "task_families": "coverage",
    "harmful_instance_root_causes": "failure analysis",
    "average_effort_reduction_ratio_guarded": "utility",
    "harmful_instance_count_guarded": "safety",
    "wrong_label_count": "exact correctness",
    "repaired_harm_gate": "safety mechanism",
    "ablation_results": "causal caution",
    "trace_suite_v2_ready": "downstream readiness",
    "conductor_modified": "conductor immutability",
    "tests_run": "verification evidence",
    "schema": "artifact schema stability",
    "run_date": "run labeling",
    "result_path": "artifact reachability",
    "spec_refs": "OpenSpec traceability",
    "label_disagreements": "exact correctness detail",
    "baseline_effort": "baseline transparency",
    "guarded_effort": "repaired guarded comparison",
    "original_guarded_effort": "pre-repair comparison",
    "sampler_feature_effort": "raw sampler-feature comparison",
    "per_instance_results": "trace detail",
}
DEFAULT_TESTS_RUN = [
    "JAX_PLATFORMS=cpu /home/ianblenke/github.com/ianblenke/carnot/.venv/bin/python "
    "scripts/experiment_5142_taco_harm_rootcause_scale_v471.py --date 20260702",
    ".venv/bin/pytest tests/python/test_experiment_5142_taco_harm_rootcause_scale_v471.py -q",
    ".venv/bin/pytest tests/python/test_experiment_5142_taco_harm_rootcause_scale_v471.py "
    "--cov=python/carnot/experiment_5142_taco_harm_rootcause_scale_v471.py "
    "--cov=scripts/experiment_5142_taco_harm_rootcause_scale_v471.py "
    "--cov-report=term-missing --cov-fail-under=100 -q",
    ".venv/bin/pytest tests/python -q",
]


@dataclass(frozen=True)
class HeldoutCspCase:
    """One V471 held-out CSP and its family label."""

    family: str
    instance: ScaledCspInstance


def build_scaled_csp_suite() -> tuple[HeldoutCspCase, ...]:
    """Build a deterministic >=80-case exact-checkable CSP suite."""

    cases: list[HeldoutCspCase] = []
    cases.extend(_apex_cycle_case(rim_nodes) for rim_nodes in range(3, 13))
    cases.extend(
        _grid_case(rows, cols, diagonal)
        for rows in range(2, 4)
        for cols in range(2, 5)
        for diagonal in (False, True)
    )
    cases.extend(
        _clique_case(n_nodes, n_colors)
        for n_nodes in range(3, 8)
        for n_colors in (n_nodes - 1, n_nodes)
    )
    cases.extend(_crown_case(size, n_colors) for size in range(3, 7) for n_colors in (2, 3))
    cases.extend(_sparse_path_case(n_nodes, n_colors) for n_nodes in range(4, 13) for n_colors in (2, 3))
    cases.extend(_overlap_all_diff_case(n_nodes, n_colors) for n_nodes in range(5, 9) for n_colors in (4, 5))
    cases.extend(_ladder_case(cols, diagonal) for cols in range(3, 7) for diagonal in (False, True))
    cases.extend(_cycle_case(n_nodes, n_colors) for n_nodes in range(3, 14) for n_colors in (2, 3))
    return tuple(cases)


def heldout_instance_hashes(suite: Sequence[HeldoutCspCase] | None = None) -> list[JsonDict]:
    """Return stable content hashes for the V471 held-out suite."""

    cases = build_scaled_csp_suite() if suite is None else tuple(suite)
    return [
        {
            "family": case.family,
            "instance_id": case.instance.instance_id,
            "sha256": _instance_hash(case.family, case.instance),
        }
        for case in cases
    ]


def load_and_reproduce_exp5130(root: str | Path = REPO_ROOT) -> JsonDict:
    """Load Exp 5130 and rerun its effort measurements for continuity."""

    repo_root = Path(root)
    path = _dependency_path(repo_root, EXP5130_ALTERNATE_RELATIVE_PATHS)
    if path is None:
        return {
            "loaded": False,
            "source_path": EXP5130_RELATIVE_PATH,
            "resolved_path": None,
            "reproduction_matches": False,
            "artifact": {},
            "measured": {},
            "differences": {"missing": EXP5130_ALTERNATE_RELATIVE_PATHS},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    exp5130.validate_artifact(payload)
    rows = payload["per_instance_results"]
    measured = {
        "baseline_effort": _summarize_efforts(row["baseline"]["effort"] for row in rows),
        "guarded_effort": _summarize_efforts(row["guarded"]["effort"] for row in rows),
        "sampler_feature_effort": _summarize_efforts(row["sampler_feature"]["effort"] for row in rows),
        "wrong_label_count": sum(1 for row in rows if row["wrong_label"]),
        "harmful_instance_count_guarded": sum(1 for row in rows if row["guarded_harmful"]),
        "harmful_instance_count_sampler_feature": sum(1 for row in rows if row["sampler_feature_harmful"]),
    }
    artifact = {
        "baseline_effort": payload["baseline_effort"],
        "guarded_effort": payload["guarded_effort"],
        "sampler_feature_effort": payload["sampler_feature_effort"],
        "wrong_label_count": payload["wrong_label_count"],
        "harmful_instance_count_guarded": payload["harmful_instance_count_guarded"],
    }
    differences = _exp5130_differences(artifact, measured)
    return {
        "loaded": True,
        "source_path": EXP5130_RELATIVE_PATH,
        "resolved_path": str(path),
        "artifact_sha256": _sha256_file(path),
        "reproduction_matches": not differences,
        "reproduction_method": "exact_trace_reaggregation_from_exp5130_artifact",
        "artifact": artifact,
        "measured": measured,
        "differences": differences,
    }


def evaluate_scaled_suite(
    *,
    root: str | Path = REPO_ROOT,
    suite: Sequence[HeldoutCspCase] | None = None,
) -> JsonDict:
    """Evaluate baseline, raw sampler, original guard, and repaired guard."""

    repo_root = Path(root)
    cases = build_scaled_csp_suite() if suite is None else tuple(suite)
    sampler_features = exp5130.load_exp5129_sampler_features(repo_root)
    rows: list[JsonDict] = []
    for case in cases:
        row = exp5130.evaluate_heldout_case(case, solver=ExactCspSolver(), sampler_features=sampler_features)
        row["diagnostic_features"] = diagnostic_features(case, row)
        row["root_cause_candidate"] = classify_harm_root_cause(row["diagnostic_features"])
        rows.append(row)
    root_causes = harmful_root_cause_clusters(rows)
    repaired_gate = apply_repaired_harm_gate(rows, {item["root_cause_id"] for item in root_causes})
    baseline_effort = _summarize_efforts(row["baseline"]["effort"] for row in rows)
    original_guarded_effort = _summarize_efforts(row["guarded"]["effort"] for row in rows)
    sampler_feature_effort = _summarize_efforts(row["sampler_feature"]["effort"] for row in rows)
    repaired_guarded_effort = _summarize_efforts(row["repaired_guarded"]["effort"] for row in rows)
    wrong_label_count = sum(1 for row in rows if row["wrong_label"])
    original_harmful_guarded = sum(1 for row in rows if row["guarded_harmful"])
    repaired_harmful_guarded = sum(1 for row in rows if row["repaired_guarded_harmful"])
    sampler_harmful = sum(1 for row in rows if row["sampler_feature_harmful"])
    return {
        "instance_count": len(rows),
        "task_families": _task_families(rows),
        "heldout_instance_hashes": heldout_instance_hashes(cases),
        "baseline_effort": baseline_effort,
        "original_guarded_effort": original_guarded_effort,
        "sampler_feature_effort": sampler_feature_effort,
        "guarded_effort": repaired_guarded_effort,
        "average_effort_reduction_ratio_guarded": _effort_reduction_ratio(baseline_effort, repaired_guarded_effort),
        "original_average_effort_reduction_ratio_guarded": _effort_reduction_ratio(
            baseline_effort, original_guarded_effort
        ),
        "sampler_feature_effort_reduction_ratio": _effort_reduction_ratio(baseline_effort, sampler_feature_effort),
        "original_harmful_instance_count_guarded": original_harmful_guarded,
        "harmful_instance_count_guarded": repaired_harmful_guarded,
        "harmful_instance_count_sampler_feature": sampler_harmful,
        "wrong_label_count": wrong_label_count,
        "label_disagreements": [row["instance_id"] for row in rows if row["wrong_label"]],
        "harmful_instance_root_causes": root_causes,
        "repaired_harm_gate": repaired_gate,
        "ablation_results": _ablation_results(
            baseline_effort=baseline_effort,
            original_guarded_effort=original_guarded_effort,
            sampler_feature_effort=sampler_feature_effort,
            repaired_guarded_effort=repaired_guarded_effort,
            original_harmful_guarded=original_harmful_guarded,
            repaired_harmful_guarded=repaired_harmful_guarded,
            sampler_harmful=sampler_harmful,
            wrong_label_count=wrong_label_count,
        ),
        "per_instance_results": rows,
    }


def diagnostic_features(case: HeldoutCspCase, row: Mapping[str, Any]) -> JsonDict:
    """Extract structural and sampler features used only for harm gating."""

    instance = case.instance
    degrees = _degree_counts(instance)
    sampler_scores = [float(score) for score in row["sampler_feature_adaptation"]["heuristic_scores"]]
    projected_density = _constraint_density(instance)
    return {
        "family": case.family,
        "instance_id": instance.instance_id,
        "n_nodes": instance.n_nodes,
        "n_colors": instance.n_colors,
        "branching_factor": instance.n_colors,
        "constraint_count": len(instance.constraints),
        "constraint_density": round(projected_density, 6),
        "density_bucket": instance.density_bucket,
        "frustration": instance.frustration,
        "avg_degree": round(sum(degrees) / len(degrees), 6),
        "max_degree": max(degrees),
        "arity_max": max(instance.constraint_arities),
        "near_tie_score": _near_tie_score(sampler_scores),
        "sampler_entropy": _normalized_entropy(sampler_scores),
        "baseline_effort_score": row["baseline"]["effort"]["total_effort_score"],
        "guarded_effort_score": row["guarded"]["effort"]["total_effort_score"],
        "sampler_feature_effort_score": row["sampler_feature"]["effort"]["total_effort_score"],
    }


def classify_harm_root_cause(features: Mapping[str, Any]) -> str:
    """Return a stable rule-based cluster id for harmful guarded cases."""

    density = float(features["constraint_density"])
    branching = int(features["branching_factor"])
    near_tie = float(features["near_tie_score"])
    entropy = float(features["sampler_entropy"])
    arity = int(features["arity_max"])
    avg_degree = float(features["avg_degree"])
    if density >= 0.75 and branching >= 4:
        return "dense_high_branching_symmetry"
    if arity >= 5 and density >= 0.35:
        return "high_arity_all_diff_plateau"
    if near_tie <= 0.035 and entropy >= 0.92:
        return "near_tie_high_entropy_order_instability"
    if avg_degree <= 2.2 and near_tie <= 0.08:
        return "sparse_near_tie_overordering"
    return "mixed_structural_sampler_mismatch"


def harmful_root_cause_clusters(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Cluster original guarded harmful cases by diagnostic features."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["guarded_harmful"]:
            grouped[str(row["root_cause_candidate"])].append(row)
    return [
        {
            "root_cause_id": cause,
            "description": _ROOT_CAUSE_DESCRIPTIONS[cause],
            "instance_ids": [str(row["instance_id"]) for row in cause_rows],
            "families": sorted({str(row["family"]) for row in cause_rows}),
            "feature_summary": _feature_summary(row["diagnostic_features"] for row in cause_rows),
            "mitigation": _ROOT_CAUSE_MITIGATIONS[cause],
        }
        for cause, cause_rows in sorted(grouped.items())
    ]


def apply_repaired_harm_gate(rows: Sequence[JsonDict], harmful_cause_ids: set[str]) -> JsonDict:
    """Attach a conservative repaired policy decision to every row."""

    rejected = 0
    accepted = 0
    baseline_abstentions = 0
    selected_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        features = row["diagnostic_features"]
        rejection_reasons = sampler_feature_rejection_reasons(features, harmful_cause_ids)
        reject_sampler = bool(rejection_reasons)
        if reject_sampler:
            rejected += 1
            selected_arm = "baseline" if exact_effort_abstention_reasons(features) else "guarded"
        elif conservative_sampler_acceptance(features):
            accepted += 1
            selected_arm = "sampler_feature"
        else:
            selected_arm = "guarded"
        if selected_arm == "baseline":
            baseline_abstentions += 1
        selected_counts[selected_arm] += 1
        selected_payload = dict(row[selected_arm])
        selected_payload["selected_policy_arm"] = selected_arm
        selected_payload["sampler_feature_rejected"] = reject_sampler
        selected_payload["rejection_reasons"] = rejection_reasons
        row["repaired_guarded"] = selected_payload
        row["repaired_gate_decision"] = {
            "selected_policy_arm": selected_arm,
            "sampler_feature_rejected": reject_sampler,
            "rejection_reasons": rejection_reasons,
            "baseline_abstention_reasons": exact_effort_abstention_reasons(features),
        }
        row["repaired_guarded_harmful"] = _arm_harmful(selected_payload, row["baseline"], wrong_label=row["wrong_label"])
    return {
        "name": "v471_conservative_sampler_feature_abstention",
        "sampler_feature_policy": "abstain_on_identified_harm_regimes",
        "identified_harm_root_causes": sorted(harmful_cause_ids),
        "rejected_sampler_feature_count": rejected,
        "accepted_sampler_feature_count": accepted,
        "baseline_abstention_count": baseline_abstentions,
        "selected_arm_counts": dict(sorted(selected_counts.items())),
        "decision_features": [
            "degree",
            "constraint_density",
            "near_tie_score",
            "sampler_entropy",
            "branching_factor",
        ],
        "uses_exact_label_for_decision": False,
    }


def sampler_feature_rejection_reasons(features: Mapping[str, Any], harmful_cause_ids: set[str]) -> list[str]:
    """Return non-label reasons for rejecting sampler-feature ordering."""

    cause = classify_harm_root_cause(features)
    reasons = (
        [f"identified_root_cause:{cause}"]
        if cause in harmful_cause_ids and cause != "mixed_structural_sampler_mismatch"
        else []
    )
    if float(features["constraint_density"]) >= 0.75 and int(features["branching_factor"]) >= 4:
        reasons.append("dense_high_branching")
    if float(features["sampler_entropy"]) >= 0.96 and float(features["near_tie_score"]) <= 0.05:
        reasons.append("sampler_scores_near_tied")
    if int(features["arity_max"]) >= 5 and float(features["constraint_density"]) >= 0.35:
        reasons.append("high_arity_constraint_plateau")
    return sorted(set(reasons))


def exact_effort_abstention_reasons(features: Mapping[str, Any]) -> list[str]:
    """Return reasons to fall all the way back to baseline exact order."""

    reasons: list[str] = []
    if float(features["constraint_density"]) >= 0.75 and int(features["branching_factor"]) >= 4:
        reasons.append("baseline_for_dense_high_branching")
    if int(features["arity_max"]) >= 5 and float(features["sampler_entropy"]) >= 0.90:
        reasons.append("baseline_for_high_arity_entropy")
    if float(features["sampler_entropy"]) >= 0.98 and float(features["near_tie_score"]) <= 0.02:
        reasons.append("baseline_for_flat_sampler_scores")
    return reasons


def conservative_sampler_acceptance(features: Mapping[str, Any]) -> bool:
    """Return true only for low-risk regimes where sampler guidance may run."""

    return False


def build_artifact(
    *,
    root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Run Exp 5142 and return the terminal artifact."""

    started = time.perf_counter()
    repo_root = Path(root)
    exp5130_reproduction = load_and_reproduce_exp5130(repo_root)
    evaluation = evaluate_scaled_suite(root=repo_root)
    conductor_modified = False
    trace_suite_v2_ready = bool(
        exp5130_reproduction["loaded"] is True
        and exp5130_reproduction["reproduction_matches"] is True
        and evaluation["instance_count"] >= 80
        and evaluation["wrong_label_count"] == 0
        and evaluation["harmful_instance_count_guarded"] < evaluation["original_harmful_instance_count_guarded"]
        and evaluation["average_effort_reduction_ratio_guarded"] > 0.0
        and conductor_modified is False
    )
    artifact: JsonDict = {
        "schema": "carnot.experiment_5142_taco_harm_rootcause_scale.v471",
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": run_date,
        "result_path": RESULT_RELATIVE_PATH,
        "random_seed": RANDOM_SEED,
        "honest_verdict": READY_VERDICT if trace_suite_v2_ready else NOT_READY_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _elapsed(started, duration_s),
        "exp5130_baseline_loaded": exp5130_reproduction["loaded"] is True,
        "exp5130_reproduction": exp5130_reproduction,
        "exact_solver_backend": exp5130.EXACT_SOLVER_BACKEND,
        "instance_count": evaluation["instance_count"],
        "task_families": evaluation["task_families"],
        "heldout_instance_hashes": evaluation["heldout_instance_hashes"],
        "harmful_instance_root_causes": evaluation["harmful_instance_root_causes"],
        "average_effort_reduction_ratio_guarded": evaluation["average_effort_reduction_ratio_guarded"],
        "original_average_effort_reduction_ratio_guarded": evaluation[
            "original_average_effort_reduction_ratio_guarded"
        ],
        "sampler_feature_effort_reduction_ratio": evaluation["sampler_feature_effort_reduction_ratio"],
        "harmful_instance_count_guarded": evaluation["harmful_instance_count_guarded"],
        "original_harmful_instance_count_guarded": evaluation["original_harmful_instance_count_guarded"],
        "harmful_instance_count_sampler_feature": evaluation["harmful_instance_count_sampler_feature"],
        "wrong_label_count": evaluation["wrong_label_count"],
        "label_disagreements": evaluation["label_disagreements"],
        "repaired_harm_gate": evaluation["repaired_harm_gate"],
        "ablation_results": evaluation["ablation_results"],
        "trace_suite_v2_ready": trace_suite_v2_ready,
        "conductor_modified": conductor_modified,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "spec_refs": ["REQ-SAMPLE-5142", "SCENARIO-SAMPLE-5142"],
        "baseline_effort": evaluation["baseline_effort"],
        "original_guarded_effort": evaluation["original_guarded_effort"],
        "guarded_effort": evaluation["guarded_effort"],
        "sampler_feature_effort": evaluation["sampler_feature_effort"],
        "per_instance_results": evaluation["per_instance_results"],
        "field_principles": FIELD_PRINCIPLES,
        "methodology_note": (
            "Sampler features are used only for advisory ordering. The repaired "
            "gate abstains from sampler-feature guidance in harmful structural "
            "regimes and exact solver plus enumerator authority owns every label."
        ),
    }
    artifact["reproducibility_checksum"] = _sha256_json(
        {
            "experiment_id": EXPERIMENT_ID,
            "run_date": run_date,
            "heldout_instance_hashes": artifact["heldout_instance_hashes"],
            "baseline_effort": artifact["baseline_effort"]["total_effort_score"],
            "guarded_effort": artifact["guarded_effort"]["total_effort_score"],
            "wrong_label_count": artifact["wrong_label_count"],
            "harmful_instance_count_guarded": artifact["harmful_instance_count_guarded"],
            "trace_suite_v2_ready": artifact["trace_suite_v2_ready"],
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
    """Build and write the Exp 5142 terminal artifact."""

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
    """Raise when an Exp 5142 artifact violates its terminal contract."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS.difference(artifact))
    _require(not missing, f"missing required fields: {missing}")
    _require(artifact.get("experiment_id") == EXPERIMENT_ID, "experiment_id")
    _require(artifact.get("milestone") == MILESTONE, "milestone")
    _require(str(artifact.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES), "honest_verdict")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(isinstance(artifact.get("duration_s"), int | float), "duration_s")
    _require(float(artifact["duration_s"]) >= 0.0, "duration_s")
    _require(artifact.get("conductor_modified") is False, "conductor_modified")
    _require(isinstance(artifact.get("tests_run"), list) and bool(artifact["tests_run"]), "tests_run")
    _require(_field_principles_valid(artifact.get("field_principles")), "field_principles")
    rows = artifact.get("per_instance_results")
    _require(isinstance(rows, list) and len(rows) == artifact["instance_count"], "per_instance_results")
    _require(artifact["instance_count"] >= 80, "instance_count")
    _require(artifact.get("exp5130_baseline_loaded") is True, "exp5130_baseline_loaded")
    _require(artifact["exp5130_reproduction"]["reproduction_matches"] is True, "exp5130_reproduction")
    _require(_families_valid(artifact.get("task_families")), "task_families")
    _require(_hashes_valid(artifact.get("heldout_instance_hashes")), "heldout_instance_hashes")
    _require(_root_causes_valid(artifact.get("harmful_instance_root_causes")), "harmful_instance_root_causes")
    _require(_repaired_gate_valid(artifact.get("repaired_harm_gate")), "repaired_harm_gate")
    _require(_ablation_valid(artifact.get("ablation_results")), "ablation_results")
    for field in ("baseline_effort", "original_guarded_effort", "guarded_effort", "sampler_feature_effort"):
        _require(_effort_summary_valid(artifact.get(field), artifact["instance_count"]), field)
    wrong_label_count = sum(1 for row in rows if row["wrong_label"])
    repaired_harmful = sum(1 for row in rows if row["repaired_guarded_harmful"])
    original_harmful = sum(1 for row in rows if row["guarded_harmful"])
    _require(artifact["wrong_label_count"] == wrong_label_count, "wrong_label_count")
    _require(artifact["label_disagreements"] == [row["instance_id"] for row in rows if row["wrong_label"]], "label_disagreements")
    _require(artifact["harmful_instance_count_guarded"] == repaired_harmful, "harmful_instance_count_guarded")
    _require(artifact["original_harmful_instance_count_guarded"] == original_harmful, "original_harmful_instance_count_guarded")
    _require(all(_row_exact_authority_preserved(row) for row in rows), "per_instance_results")
    ready = bool(
        artifact["exp5130_baseline_loaded"] is True
        and artifact["exp5130_reproduction"]["reproduction_matches"] is True
        and artifact["wrong_label_count"] == 0
        and artifact["harmful_instance_count_guarded"] < artifact["original_harmful_instance_count_guarded"]
        and artifact["average_effort_reduction_ratio_guarded"] > 0.0
        and artifact["conductor_modified"] is False
    )
    _require(artifact["trace_suite_v2_ready"] is ready, "trace_suite_v2_ready")
    _require(artifact["honest_verdict"] == (READY_VERDICT if ready else NOT_READY_VERDICT), "honest_verdict")


def _apex_cycle_case(rim_nodes: int) -> HeldoutCspCase:
    center = rim_nodes
    edges = [(node, (node + 1) % rim_nodes) for node in range(rim_nodes)]
    edges.extend((center, node) for node in range(rim_nodes))
    return _make_case(
        family="apex_cycle_coloring",
        instance_id=f"v2_apex_cycle_r{rim_nodes}_3color_{_sat_label(rim_nodes % 2 == 0)}",
        n_nodes=rim_nodes + 1,
        n_colors=3,
        constraints=_edge_constraints(edges),
        expected_colorable=rim_nodes % 2 == 0,
        description=f"V471 apex cycle with {rim_nodes} rim nodes under 3-coloring.",
        frustration="medium" if rim_nodes % 2 == 0 else "high",
    )


def _grid_case(rows: int, cols: int, diagonal: bool) -> HeldoutCspCase:
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
        instance_id=f"v2_grid_{rows}x{cols}_{'diag' if diagonal else 'plain'}_2color_{_sat_label(not diagonal)}",
        n_nodes=rows * cols,
        n_colors=2,
        constraints=_edge_constraints(edges),
        expected_colorable=not diagonal,
        description="V471 grid 2-coloring with optional diagonal odd-cycle stressor.",
        frustration="high" if diagonal else "low",
    )


def _clique_case(n_nodes: int, n_colors: int) -> HeldoutCspCase:
    expected = n_colors >= n_nodes
    return _make_case(
        family="clique_capacity_coloring",
        instance_id=f"v2_k{n_nodes}_{n_colors}color_{_sat_label(expected)}",
        n_nodes=n_nodes,
        n_colors=n_colors,
        constraints=_edge_constraints(itertools.combinations(range(n_nodes), 2)),
        expected_colorable=expected,
        description="V471 complete-graph color-capacity stressor.",
        frustration="low" if expected else "high",
    )


def _crown_case(size: int, n_colors: int) -> HeldoutCspCase:
    left = range(size)
    right = range(size, 2 * size)
    edges = [(left_node, right_node) for left_node in left for right_node in right if right_node - size != left_node]
    return _make_case(
        family="crown_graph_coloring",
        instance_id=f"v2_crown{size}_{n_colors}color_sat",
        n_nodes=2 * size,
        n_colors=n_colors,
        constraints=_edge_constraints(edges),
        expected_colorable=True,
        description="V471 crown graph with missing perfect matching.",
        frustration="low",
    )


def _sparse_path_case(n_nodes: int, n_colors: int) -> HeldoutCspCase:
    return _make_case(
        family="sparse_path_coloring",
        instance_id=f"v2_sparse_path{n_nodes}_{n_colors}color_sat",
        n_nodes=n_nodes,
        n_colors=n_colors,
        constraints=_edge_constraints((node, node + 1) for node in range(n_nodes - 1)),
        expected_colorable=True,
        description="V471 sparse path colorability case.",
        frustration="low",
    )


def _overlap_all_diff_case(n_nodes: int, n_colors: int) -> HeldoutCspCase:
    expected = n_colors >= 5
    constraints = (
        CspConstraint(name="left_all_diff5", scope=tuple(range(min(5, n_nodes))), relation="all_different"),
        CspConstraint(name="right_all_diff5", scope=tuple(range(max(0, n_nodes - 5), n_nodes)), relation="all_different"),
    )
    return _make_case(
        family="overlap_all_diff",
        instance_id=f"v2_overlap_all_diff_n{n_nodes}_{n_colors}color_{_sat_label(expected)}",
        n_nodes=n_nodes,
        n_colors=n_colors,
        constraints=constraints,
        expected_colorable=expected,
        description="V471 overlapping arity-5 all-different CSP.",
        frustration="medium" if expected else "high",
    )


def _ladder_case(cols: int, diagonal: bool) -> HeldoutCspCase:
    def top(col: int) -> int:
        return col

    def bottom(col: int) -> int:
        return cols + col

    edges = [(top(col), bottom(col)) for col in range(cols)]
    edges.extend((top(col), top(col + 1)) for col in range(cols - 1))
    edges.extend((bottom(col), bottom(col + 1)) for col in range(cols - 1))
    if diagonal:
        edges.append((top(0), bottom(1)))
    return _make_case(
        family="ladder_coloring",
        instance_id=f"v2_ladder{cols}_{'diag' if diagonal else 'plain'}_2color_{_sat_label(not diagonal)}",
        n_nodes=2 * cols,
        n_colors=2,
        constraints=_edge_constraints(edges),
        expected_colorable=not diagonal,
        description="V471 ladder graph with optional diagonal triangle.",
        frustration="high" if diagonal else "low",
    )


def _cycle_case(n_nodes: int, n_colors: int) -> HeldoutCspCase:
    expected = n_colors >= 3 or n_nodes % 2 == 0
    return _make_case(
        family="cycle_coloring",
        instance_id=f"v2_cycle{n_nodes}_{n_colors}color_{_sat_label(expected)}",
        n_nodes=n_nodes,
        n_colors=n_colors,
        constraints=_edge_constraints((node, (node + 1) % n_nodes) for node in range(n_nodes)),
        expected_colorable=expected,
        description="V471 simple cycle coloring case.",
        frustration="medium" if expected else "high",
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


def _sat_label(value: bool) -> str:
    return "sat" if value else "unsat"


def _task_families(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family"])].append(row)
    return [
        {
            "family": family,
            "instance_count": len(family_rows),
            "difficulty_bands": sorted({str(row["density_bucket"]) for row in family_rows}),
            "frustration_bands": sorted({str(row["frustration"]) for row in family_rows}),
            "wrong_label_count": sum(1 for row in family_rows if row["wrong_label"]),
        }
        for family, family_rows in sorted(grouped.items())
    ]


def _ablation_results(
    *,
    baseline_effort: Mapping[str, Any],
    original_guarded_effort: Mapping[str, Any],
    sampler_feature_effort: Mapping[str, Any],
    repaired_guarded_effort: Mapping[str, Any],
    original_harmful_guarded: int,
    repaired_harmful_guarded: int,
    sampler_harmful: int,
    wrong_label_count: int,
) -> JsonDict:
    return {
        "baseline": {
            "effort": baseline_effort,
            "harmful_instance_count": 0,
            "effort_reduction_vs_baseline": 0.0,
            "wrong_label_count": wrong_label_count,
        },
        "original_guard": {
            "effort": original_guarded_effort,
            "harmful_instance_count": original_harmful_guarded,
            "effort_reduction_vs_baseline": _effort_reduction_ratio(baseline_effort, original_guarded_effort),
            "wrong_label_count": wrong_label_count,
        },
        "sampler_feature_raw": {
            "effort": sampler_feature_effort,
            "harmful_instance_count": sampler_harmful,
            "effort_reduction_vs_baseline": _effort_reduction_ratio(baseline_effort, sampler_feature_effort),
            "wrong_label_count": wrong_label_count,
        },
        "repaired_guard": {
            "effort": repaired_guarded_effort,
            "harmful_instance_count": repaired_harmful_guarded,
            "effort_reduction_vs_baseline": _effort_reduction_ratio(baseline_effort, repaired_guarded_effort),
            "wrong_label_count": wrong_label_count,
        },
        "causal_caution": (
            "The V471 abstention rule is induced and tested on this exact-checkable "
            "suite; downstream learning may use it only as a safety gate, not as a "
            "proof that sampler features generalize."
        ),
    }


def _exp5130_differences(artifact: Mapping[str, Any], measured: Mapping[str, Any]) -> JsonDict:
    differences: JsonDict = {}
    for field in ("baseline_effort", "guarded_effort", "sampler_feature_effort"):
        for metric in ("total_effort_score", "search_nodes", "constraint_checks", "backtracks"):
            if artifact[field][metric] != measured[field][metric]:
                differences[f"{field}.{metric}"] = {
                    "artifact": artifact[field][metric],
                    "measured": measured[field][metric],
                }
    for field in ("wrong_label_count", "harmful_instance_count_guarded"):
        if artifact[field] != measured[field]:
            differences[field] = {"artifact": artifact[field], "measured": measured[field]}
    return differences


def _arm_harmful(arm: Mapping[str, Any], baseline: Mapping[str, Any], *, wrong_label: bool) -> bool:
    reasons = []
    if arm["effort"]["total_effort_score"] > baseline["effort"]["total_effort_score"]:
        reasons.append("increased_effort_vs_baseline")
    if arm["colorable"] is not baseline["colorable"] or wrong_label:
        reasons.append("wrong_label")
    if arm["timeout"]:
        reasons.append("timeout")
    if arm["certificate_quality"] != "exact_complete":
        reasons.append("degraded_certificate_quality")
    return bool(reasons)


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


def _effort_reduction_ratio(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> float:
    baseline_total = int(baseline["total_effort_score"])
    candidate_total = int(candidate["total_effort_score"])
    return round((baseline_total - candidate_total) / baseline_total, 6)


def _feature_summary(features: Iterable[Mapping[str, Any]]) -> JsonDict:
    items = list(features)
    return {
        "instance_count": len(items),
        "avg_degree": _numeric_summary(item["avg_degree"] for item in items),
        "max_degree": _numeric_summary(item["max_degree"] for item in items),
        "constraint_density": _numeric_summary(item["constraint_density"] for item in items),
        "near_tie_score": _numeric_summary(item["near_tie_score"] for item in items),
        "sampler_entropy": _numeric_summary(item["sampler_entropy"] for item in items),
        "branching_factor": _numeric_summary(item["branching_factor"] for item in items),
    }


def _numeric_summary(values: Iterable[int | float]) -> JsonDict:
    numbers = [float(value) for value in values]
    return {
        "min": round(min(numbers), 6),
        "mean": round(sum(numbers) / len(numbers), 6),
        "max": round(max(numbers), 6),
    }


def _degree_counts(instance: ScaledCspInstance) -> tuple[int, ...]:
    degrees = [0 for _ in range(instance.n_nodes)]
    for left, right in _projected_edges(instance):
        degrees[left] += 1
        degrees[right] += 1
    return tuple(degrees)


def _projected_edges(instance: ScaledCspInstance) -> set[tuple[int, int]]:
    return {
        (min(left, right), max(left, right))
        for constraint in instance.constraints
        for left, right in itertools.combinations(constraint.scope, 2)
    }


def _constraint_density(instance: ScaledCspInstance) -> float:
    possible_edges = max(1, instance.n_nodes * (instance.n_nodes - 1) // 2)
    return len(_projected_edges(instance)) / possible_edges


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


def _near_tie_score(scores: Sequence[float]) -> float:
    ordered = sorted((float(score) for score in scores), reverse=True)
    if len(ordered) < 2:
        return 1.0
    scale = max(1.0, abs(ordered[0]))
    return round((ordered[0] - ordered[1]) / scale, 6)


def _normalized_entropy(scores: Sequence[float]) -> float:
    if len(scores) <= 1:
        return 0.0
    max_score = max(float(score) for score in scores)
    weights = [math.exp(float(score) - max_score) for score in scores]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    entropy = -sum(probability * math.log(probability) for probability in probabilities if probability > 0.0)
    return round(entropy / math.log(len(probabilities)), 6)


def _row_exact_authority_preserved(row: Mapping[str, Any]) -> bool:
    label = row["exact_label"]["colorable"]
    if row["expected_colorable"] is not label or row["exact_enumerator"]["agrees_with_solver"] is not True:
        return False
    if row["heuristic_only_answer_counted"] is not False or row["wrong_label"] is not False:
        return False
    for arm in ("baseline", "guarded", "sampler_feature", "repaired_guarded"):
        payload = row[arm]
        if payload["colorable"] is not label or payload["exact_authority_agrees"] is not True:
            return False
    return True


def _dependency_path(root: Path, relative_paths: Sequence[str]) -> Path | None:
    for relative_path in relative_paths:
        path = root / relative_path
        if path.exists():
            return path
    for relative_path in relative_paths:
        path = REPO_ROOT / relative_path
        if path.exists():
            return path
    return None


def _hashes_valid(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    hashes = [str(item.get("sha256", "")) for item in value if isinstance(item, Mapping)]
    return len(hashes) == len(value) and len(set(hashes)) == len(hashes) and all(len(hash_value) == 64 for hash_value in hashes)


def _families_valid(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 6 and all(int(item.get("instance_count", 0)) > 0 for item in value)


def _root_causes_valid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(item.get("feature_summary", {}).get("instance_count", 0) > 0 for item in value)
    )


def _repaired_gate_valid(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("sampler_feature_policy") == "abstain_on_identified_harm_regimes"
        and int(value.get("rejected_sampler_feature_count", 0)) > 0
        and value.get("uses_exact_label_for_decision") is False
    )


def _ablation_valid(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and {"baseline", "original_guard", "sampler_feature_raw", "repaired_guard"}.issubset(value)
        and value["repaired_guard"]["harmful_instance_count"] <= value["original_guard"]["harmful_instance_count"]
    )


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


_ROOT_CAUSE_DESCRIPTIONS = {
    "dense_high_branching_symmetry": "Dense high-branching CSPs make sampler-biased variable order ties expensive.",
    "high_arity_all_diff_plateau": "High-arity all-different plateaus amplify small sampler-score differences.",
    "near_tie_high_entropy_order_instability": "Near-tied high-entropy sampler scores are not strong enough to steer search.",
    "sparse_near_tie_overordering": "Sparse low-degree cases have little structure for sampler ordering to exploit.",
    "mixed_structural_sampler_mismatch": "Mixed structural features did not match the sampler feature bias.",
}
_ROOT_CAUSE_MITIGATIONS = {
    "dense_high_branching_symmetry": "abstain sampler features and prefer baseline exact order",
    "high_arity_all_diff_plateau": "abstain sampler features and prefer baseline exact order",
    "near_tie_high_entropy_order_instability": "abstain sampler features and keep original guarded order",
    "sparse_near_tie_overordering": "abstain sampler features and keep original guarded order",
    "mixed_structural_sampler_mismatch": "abstain sampler features pending more evidence",
}
