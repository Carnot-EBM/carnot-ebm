"""Experiment 4725: audit ARC generation/exploration nulls for silent no-ops.

Spec refs: REQ-ARC-WMTE-4725, SCENARIO-ARC-WMTE-4725-AUDIT-CLASSIFICATION.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))

EXPERIMENT = "experiment_4725_silent_bug_audit"
SCHEMA = "carnot.arc.silent_bug_audit_4725.v1"
RESULT_RELATIVE_PATH = "results/experiment_4725_silent_bug_audit.json"
REPORT_RELATIVE_PATH = "ops/arc_null_silent_bug_audit.md"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts -- reads upstream nulls + module code, no model load "
    "(100us floor)."
)
RANDOM_SEED = 4725
TERMINAL_PREFIXES = ("complete:", "blocked_")

JsonDict = dict[str, Any]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; complete: silent_bug_audit_<N>_nulls_<K>_must_reopen."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts -- reads upstream nulls + module code, no "
            "model load (100us floor)."
        )
    },
    "nulls_audited": {
        "principle": (
            "the count of .428-.434 generation/exploration-lever nulls inspected -- the "
            "audit scope."
        )
    },
    "silent_bug_nulls": {
        "principle": (
            "the list classified silent_bug_must_reopen with evidence (degenerate shape / "
            "empty pool / dead archive / dropped candidate / byte-identical arms) -- the "
            "load-bearing finding."
        )
    },
    "a4_tautology_verdict": {
        "principle": (
            "the explicit adjudication of the .434 A4 all-arms-0.04 TAUTOLOGY: "
            "online_driver_arms_degenerate (no-op, must reopen) | trustworthy_null -- this "
            "GROUNDS whether .435 A1 is a valid reopen."
        )
    },
    "trustworthy_nulls": {
        "principle": (
            "the list classified trustworthy_null (the mechanism genuinely ran on "
            "non-degenerate data) -- so the loop does NOT re-run a real null."
        )
    },
    "reopen_recommendations": {
        "principle": (
            "the prioritized re-run list for the planner (which closed levers reopen) -- "
            "the audit's actionable output."
        )
    },
    "go_explore_fix_confirmed": {
        "principle": (
            "true if the Go-Explore _frame_grid (1,64,64) fix (2026-06-25) is confirmed "
            "landed."
        )
    },
    "audit_report_path": {
        "principle": "ops/arc_null_silent_bug_audit.md -- the human-readable per-null table."
    },
    "verifier_is_oracle": {"principle": "false -- an audit invokes no oracle."},
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent drift in the audited artifact set."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (null artifacts present, arc_go_explore importable); "
            "pre-empts missing-resource fabrication."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "spec_refs",
    "field_principles",
    "audited_artifacts",
    "audited_artifact_checksums",
    "per_null_verdicts",
    "duration_s",
)

AUDIT_TARGETS: tuple[dict[str, str], ...] = (
    {
        "null_id": "experiment_4628_dense_curiosity_progress_loop",
        "artifact_path": "results/experiment_4628_dense_curiosity_progress_loop.json",
        "module_path": "python/carnot/experiment_4628_dense_curiosity_progress_loop.py",
    },
    {
        "null_id": "experiment_4640_goal_energy_generation_live",
        "artifact_path": "results/experiment_4640_goal_energy_generation_live.json",
        "module_path": "python/carnot/experiment_4640_goal_energy_generation_live.py",
    },
    {
        "null_id": "experiment_4653_energy_fitness_qd_generation_live",
        "artifact_path": "results/experiment_4653_energy_fitness_qd_generation_live.json",
        "module_path": "python/carnot/experiment_4653_energy_fitness_qd_generation_live.py",
    },
    {
        "null_id": "experiment_4664_l2_goal_predicate_induction_live",
        "artifact_path": "results/experiment_4664_l2_goal_predicate_induction_live.json",
        "module_path": "python/carnot/experiment_4664_l2_goal_predicate_induction_live.py",
    },
    {
        "null_id": "experiment_4676_hierarchical_subgoal_search_live",
        "artifact_path": "results/experiment_4676_hierarchical_subgoal_search_live.json",
        "module_path": "python/carnot/experiment_4676_hierarchical_subgoal_search_live.py",
    },
    {
        "null_id": "experiment_4688_controllable_novelty_proposal_policy_live",
        "artifact_path": "results/experiment_4688_controllable_novelty_proposal_policy_live.json",
        "module_path": "python/carnot/experiment_4688_controllable_novelty_proposal_policy_live.py",
    },
    {
        "null_id": "experiment_4700_object_centric_perception_proposal_live",
        "artifact_path": "results/experiment_4700_object_centric_perception_proposal_live.json",
        "module_path": "python/carnot/experiment_4700_object_centric_perception_proposal_live.py",
    },
    {
        "null_id": "experiment_4701_amortized_exploration_prior_go_explore_live",
        "artifact_path": "results/experiment_4701_amortized_exploration_prior_go_explore_live.json",
        "module_path": "python/carnot/experiment_4701_amortized_exploration_prior_go_explore_live.py",
    },
    {
        "null_id": "experiment_4710_arms_summary",
        "artifact_path": "results/experiment_4710_arms_summary.json",
        "module_path": "python/carnot/experiment_4710_online_action_learning_arms.py",
    },
    {
        "null_id": "experiment_4712_perception_grounded_l2_goal_lp85",
        "artifact_path": "results/experiment_4712_perception_grounded_l2_goal_lp85.json",
        "module_path": "python/carnot/experiment_4712_perception_grounded_l2_goal_lp85.py",
    },
    {
        "null_id": "experiment_4713_surface_present_winner_verifier_ranker",
        "artifact_path": "results/experiment_4713_surface_present_winner_verifier_ranker.json",
        "module_path": "python/carnot/experiment_4713_surface_present_winner_verifier_ranker.py",
    },
    {
        "null_id": "experiment_4715_online_action_learning_driver_corrected",
        "artifact_path": "results/experiment_4715_online_action_learning_driver_corrected.json",
        "module_path": "python/carnot/experiment_4715_online_action_learning_driver_corrected.py",
    },
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_stable_json(value).encode("utf-8"))


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    payload["duration_s"] = 0.0
    return _sha256_json(payload)


def _read_json(path: Path) -> JsonDict:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return loaded


def _file_checksum(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        parsed = float(value)
        return parsed if np.isfinite(parsed) else None
    return None


def _int_value(value: Any) -> int:
    parsed = _finite_float(value)
    return int(parsed) if parsed is not None else 0


def _round_float(value: Any, digits: int = 10) -> float:
    parsed = _finite_float(value)
    return round(parsed if parsed is not None else 0.0, digits)


def _short(value: Any, limit: int = 220) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _append_unique(rows: list[str], text: str) -> None:
    if text and text not in rows:
        rows.append(text)


def _dict_without(row: Mapping[str, Any], excluded: set[str]) -> dict[str, Any]:
    return {str(k): v for k, v in row.items() if str(k) not in excluded}


def _normalised_rows_equal(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], *, excluded: set[str]
) -> bool:
    return [_dict_without(row, excluded) for row in left] == [
        _dict_without(row, excluded) for row in right
    ]


def _attempts(measurement: Mapping[str, Any], key: str = "variant_attempts") -> list[Mapping[str, Any]]:
    rows = measurement.get(key)
    return [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def _archive_diagnostics(artifact: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            diag = value.get("go_explore_archive_diagnostics")
            if isinstance(diag, Mapping):
                rows.append(diag)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(artifact)
    return rows


def _max_diag_value(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    return max((_int_value(row.get(key)) for row in rows), default=0)


def _classify_4628(artifact: Mapping[str, Any], evidence: list[str], signatures: list[str]) -> None:
    loop = artifact.get("loop_measurement") if isinstance(artifact.get("loop_measurement"), Mapping) else {}
    attempts = _attempts(loop)
    enabled = 0
    events = 0
    edge_values = 0
    for row in attempts:
        diag = row.get("curiosity_diagnostics")
        if isinstance(diag, Mapping) and diag.get("enabled") is True:
            enabled += 1
            events += _int_value(diag.get("prediction_error_events"))
            edge_values += _int_value(diag.get("edge_values"))
    state_delta = _round_float(artifact.get("state_coverage_delta"))
    if enabled and (events > 0 or edge_values > 0 or state_delta != 0.0):
        _append_unique(
            evidence,
            f"dense_curiosity_enabled_attempts={enabled}, prediction_error_events={events}, "
            f"edge_values={edge_values}, state_coverage_delta={state_delta}",
        )
    else:
        _append_unique(signatures, "curiosity_loop_no_exercise_evidence")


def _classify_4640(artifact: Mapping[str, Any], evidence: list[str], signatures: list[str]) -> None:
    baseline = artifact.get("baseline_measurement")
    goal = artifact.get("goal_energy_measurement")
    uniform = artifact.get("uniform_measurement")
    if not all(isinstance(row, Mapping) for row in (baseline, goal, uniform)):
        _append_unique(signatures, "goal_energy_measurements_missing")
        return
    baseline_rows = _attempts(baseline)  # type: ignore[arg-type]
    goal_rows = _attempts(goal)  # type: ignore[arg-type]
    uniform_rows = _attempts(uniform)  # type: ignore[arg-type]
    cloned_goal = _normalised_rows_equal(
        baseline_rows,
        goal_rows,
        excluded={"goal_energy_neutral_on_cached_frame", "uniform_energy_ablation"},
    )
    cloned_uniform = _normalised_rows_equal(
        baseline_rows,
        uniform_rows,
        excluded={"goal_energy_neutral_on_cached_frame", "uniform_energy_ablation"},
    )
    neutral = sum(1 for row in goal_rows if row.get("goal_energy_neutral_on_cached_frame") is True)
    if baseline_rows and cloned_goal and cloned_uniform and neutral == len(goal_rows):
        _append_unique(signatures, "no_op_goal_energy_cached_frame")
        _append_unique(signatures, "byte_identical_goal_energy_and_baseline")
        _append_unique(
            evidence,
            f"goal_energy_measurement cloned {len(goal_rows)} baseline attempts with "
            "goal_energy_neutral_on_cached_frame=True; uniform arm also cloned baseline",
        )
    else:
        _append_unique(
            evidence,
            f"goal_energy rows={len(goal_rows)}, baseline rows={len(baseline_rows)}, "
            f"uniform rows={len(uniform_rows)}, cloned_goal={cloned_goal}",
        )


def _classify_4653(artifact: Mapping[str, Any], evidence: list[str], signatures: list[str]) -> None:
    search = artifact.get("search_measurement")
    random_qd = artifact.get("random_mutation_measurement")
    qd = artifact.get("qd_measurement")
    if not all(isinstance(row, Mapping) for row in (search, random_qd, qd)):
        _append_unique(signatures, "qd_measurements_missing")
        return
    search_rows = _attempts(search, key="attempts")  # type: ignore[arg-type]
    random_rows = _attempts(random_qd, key="attempts")  # type: ignore[arg-type]
    qd_rows = _attempts(qd, key="attempts")  # type: ignore[arg-type]
    qd_cloned = _normalised_rows_equal(search_rows, qd_rows, excluded={"arm"})
    random_cloned = _normalised_rows_equal(search_rows, random_rows, excluded={"arm"})
    if search_rows and qd_cloned and random_cloned:
        _append_unique(signatures, "byte_identical_qd_search_random_arms")
        _append_unique(signatures, "unchanged_candidate_pool")
        _append_unique(
            evidence,
            f"energy_qd/search/random arms share {len(search_rows)} identical attempts except arm label; "
            "winner_generated_count=0 with no distinct QD pool evidence",
        )
    else:
        _append_unique(
            evidence,
            f"search_attempts={len(search_rows)}, qd_attempts={len(qd_rows)}, "
            f"random_attempts={len(random_rows)}, qd_cloned={qd_cloned}",
        )


def _classify_4664(artifact: Mapping[str, Any], evidence: list[str], signatures: list[str]) -> None:
    del signatures
    per_game = artifact.get("per_game")
    lp85 = per_game.get("lp85") if isinstance(per_game, Mapping) else {}
    goal_sat = artifact.get("goal_predicate_satisfiable")
    l2_plan = artifact.get("l2_plan_len")
    _append_unique(
        evidence,
        "single-exemplar goal predicate evaluated: "
        f"goal_predicate_satisfiable={_short(goal_sat)}, l2_plan_len={_short(l2_plan)}, "
        f"lp85_bare_control={bool(isinstance(lp85, Mapping) and lp85.get('bare_control_passed'))}",
    )


def _classify_4676(artifact: Mapping[str, Any], evidence: list[str], signatures: list[str]) -> None:
    subgoals = artifact.get("subgoal_decomposition")
    reachable = artifact.get("per_subgoal_reachable")
    target = artifact.get("target_arm_results")
    if not isinstance(subgoals, list) or not isinstance(reachable, list):
        _append_unique(signatures, "subgoal_evidence_missing")
        return
    levels: list[int] = []
    if isinstance(target, Mapping):
        for key in ("hierarchical_subgoal", "no_subgoal", "random_subgoal"):
            row = target.get(key)
            if isinstance(row, Mapping):
                levels.append(_int_value(row.get("reached_level")))
    if not subgoals and not reachable and levels and len(set(levels)) == 1:
        _append_unique(signatures, "empty_subgoal_decomposition")
        _append_unique(signatures, "subgoal_arm_noop")
        _append_unique(
            evidence,
            f"subgoal_decomposition=[] and per_subgoal_reachable=[]; target arm levels={levels}",
        )
    else:
        _append_unique(
            evidence,
            f"subgoals={len(subgoals)}, per_subgoal_reachable={len(reachable)}, target_levels={levels}",
        )


def _classify_4688(artifact: Mapping[str, Any], evidence: list[str], signatures: list[str]) -> None:
    del signatures
    target = artifact.get("target_arm_results")
    novelty = target.get("controllable_novelty") if isinstance(target, Mapping) else {}
    diag = novelty.get("controllable_novelty_diagnostics") if isinstance(novelty, Mapping) else {}
    candidate_scores = _int_value(diag.get("candidate_scores")) if isinstance(diag, Mapping) else 0
    observed = _int_value(diag.get("observed_effects")) if isinstance(diag, Mapping) else 0
    rnd_updates = _int_value(diag.get("rnd_updates")) if isinstance(diag, Mapping) else 0
    _append_unique(
        evidence,
        f"controllable_novelty candidate_scores={candidate_scores}, "
        f"observed_effects={observed}, rnd_updates={rnd_updates}",
    )


def _classify_4700(artifact: Mapping[str, Any], evidence: list[str], signatures: list[str]) -> None:
    coverage = artifact.get("proposal_coverage_by_representation")
    obj = coverage.get("object_centric") if isinstance(coverage, Mapping) else {}
    raw_hits = obj.get("step_hits") if isinstance(obj, Mapping) else []
    hits = raw_hits if isinstance(raw_hits, list) else []
    candidate_counts = [
        _int_value(row.get("candidate_count"))
        for row in hits
        if isinstance(row, Mapping) and _int_value(row.get("candidate_count")) > 0
    ]
    target = artifact.get("target_arm_results")
    obj_arm = target.get("object_centric") if isinstance(target, Mapping) else {}
    diag = (
        obj_arm.get("object_centric_proposal_diagnostics") if isinstance(obj_arm, Mapping) else {}
    )
    augmented = _int_value(diag.get("augmented_candidates")) if isinstance(diag, Mapping) else 0
    scores = _int_value(diag.get("candidate_scores")) if isinstance(diag, Mapping) else 0
    if candidate_counts and augmented > 0 and scores > 0:
        _append_unique(evidence, "object_centric_pool_nonempty")
        _append_unique(
            evidence,
            f"coverage={obj.get('coverage') if isinstance(obj, Mapping) else None}, "
            f"candidate_counts={candidate_counts}, augmented_candidates={augmented}, "
            f"candidate_scores={scores}",
        )
    else:
        _append_unique(signatures, "empty_object_centric_candidate_pool")


def _classify_4701(artifact: Mapping[str, Any], evidence: list[str], signatures: list[str]) -> None:
    archives = _archive_diagnostics(artifact)
    observations = _max_diag_value(archives, "observations")
    cells = _max_diag_value(archives, "stored_cells")
    actions_injected = _max_diag_value(archives, "actions_injected")
    prefixes = _max_diag_value(archives, "prefixes_injected")
    coverage = artifact.get("target_arm_results")
    cov = coverage.get("coverage") if isinstance(coverage, Mapping) else {}
    with_prior = cov.get("with_prior") if isinstance(cov, Mapping) else {}
    total_steps = _int_value(with_prior.get("total_steps")) if isinstance(with_prior, Mapping) else 0
    if archives and observations == 0 and cells == 0 and actions_injected == 0 and prefixes == 0:
        _append_unique(signatures, "dead_go_explore_archive")
    if _round_float(artifact.get("candidate_generation_coverage_with_prior")) == 0.0 and total_steps == 0:
        _append_unique(signatures, "empty_candidate_generation_pool")
    _append_unique(
        evidence,
        f"go_explore_archive observations={observations}, stored_cells={cells}, "
        f"actions_injected={actions_injected}, prefixes_injected={prefixes}, "
        f"with_prior_total_steps={total_steps}",
    )


def _classify_4710(artifact: Mapping[str, Any], evidence: list[str], signatures: list[str]) -> None:
    arms = artifact.get("arms")
    if not isinstance(arms, list):
        _append_unique(signatures, "online_arm_summary_missing")
        return
    online_rows = [
        row
        for row in arms
        if isinstance(row, Mapping) and str(row.get("arm") or "").startswith("online-")
    ]
    max_observed = 0
    max_fits = 0
    min_errors: int | None = None
    for row in online_rows:
        diag = row.get("scorer_diagnostics")
        if not isinstance(diag, Mapping):
            continue
        max_observed = max(max_observed, _int_value(diag.get("observed")))
        max_fits = max(max_fits, _int_value(diag.get("fits")))
        errors = _int_value(diag.get("errors"))
        min_errors = errors if min_errors is None else min(min_errors, errors)
    all_rates = {_round_float(row.get("first_win_rate")) for row in arms if isinstance(row, Mapping)}
    if max_observed > 0 and max_fits > 0 and (min_errors or 0) == 0:
        _append_unique(
            evidence,
            f"online_cnn_observed_{max_observed}_fits_{max_fits}_errors_{min_errors or 0}",
        )
        if len(all_rates) == 1:
            _append_unique(
                evidence,
                f"first_win rates flat at {next(iter(all_rates))} but CNN exercise counters are positive",
            )
    else:
        _append_unique(signatures, "dropped_dict_candidate")
        _append_unique(
            evidence,
            f"online arms observed={max_observed}, fits={max_fits}, min_errors={min_errors}",
        )


def _classify_4712(artifact: Mapping[str, Any], evidence: list[str], signatures: list[str]) -> None:
    control = artifact.get("detector_positive_control")
    diag = control.get("diagnostics") if isinstance(control, Mapping) else {}
    detected = bool(diag.get("detected")) if isinstance(diag, Mapping) else False
    slots = _int_value(diag.get("object_centric_slot_count")) if isinstance(diag, Mapping) else 0
    pieces = _int_value(diag.get("piece_count")) if isinstance(diag, Mapping) else 0
    goals = _int_value(diag.get("goal_count")) if isinstance(diag, Mapping) else 0
    if detected and slots > 0 and pieces > 0 and goals > 0:
        _append_unique(
            evidence,
            f"structural detector ran: slots={slots}, piece_count={pieces}, goal_count={goals}, "
            f"goal_predicate_satisfiable={artifact.get('goal_predicate_satisfiable')}",
        )
    else:
        _append_unique(signatures, "structural_goal_detector_degenerate")


def _classify_4713(artifact: Mapping[str, Any], evidence: list[str], signatures: list[str]) -> None:
    diag = artifact.get("surfacing_ranker_diagnostics")
    ranker = diag.get("surfacing_ranker") if isinstance(diag, Mapping) else {}
    samples = _int_value(ranker.get("samples")) if isinstance(ranker, Mapping) else 0
    fits = _int_value(ranker.get("fit_count")) if isinstance(ranker, Mapping) else 0
    positives = _int_value(ranker.get("positive_samples")) if isinstance(ranker, Mapping) else 0
    scores = _int_value(diag.get("candidate_scores")) if isinstance(diag, Mapping) else 0
    if _round_float(artifact.get("winner_present_coverage")) >= 1.0 and samples > 0 and fits > 0:
        _append_unique(
            evidence,
            f"winner_present_coverage=1.0, surfacing_samples={samples}, "
            f"fit_count={fits}, positive_samples={positives}, candidate_scores={scores}",
        )
    else:
        _append_unique(signatures, "surfacing_ranker_no_exercise_evidence")


def _classify_4715(artifact: Mapping[str, Any], evidence: list[str], signatures: list[str]) -> None:
    frozen = _round_float(artifact.get("frozen_first_win"))
    scratch = _round_float(artifact.get("online_scratch_first_win"))
    warm = _round_float(artifact.get("online_warm_first_win"))
    delta = _round_float(artifact.get("online_warm_vs_frozen_delta"))
    if frozen == scratch == warm == 0.04 and delta == 0.0:
        _append_unique(signatures, "byte_identical_online_driver_arms")
        _append_unique(signatures, "a4_all_arms_0_04_tautology")
        _append_unique(
            evidence,
            f"frozen_first_win=online_scratch_first_win=online_warm_first_win={frozen}, "
            f"delta={delta}, flagged_adversarial={artifact.get('flagged_adversarial')}",
        )
    else:
        _append_unique(
            evidence,
            f"driver arm rates frozen={frozen}, scratch={scratch}, warm={warm}, delta={delta}",
        )
    probe = artifact.get("goal_free_probe")
    if isinstance(probe, Mapping):
        cells = [
            _int_value(row.get("archive_cells"))
            for row in probe.get("games", [])
            if isinstance(row, Mapping)
        ]
        if cells:
            _append_unique(evidence, f"goal_free_probe archive_cells={cells}")


CLASSIFIERS = {
    "experiment_4628_dense_curiosity_progress_loop": _classify_4628,
    "experiment_4640_goal_energy_generation_live": _classify_4640,
    "experiment_4653_energy_fitness_qd_generation_live": _classify_4653,
    "experiment_4664_l2_goal_predicate_induction_live": _classify_4664,
    "experiment_4676_hierarchical_subgoal_search_live": _classify_4676,
    "experiment_4688_controllable_novelty_proposal_policy_live": _classify_4688,
    "experiment_4700_object_centric_perception_proposal_live": _classify_4700,
    "experiment_4701_amortized_exploration_prior_go_explore_live": _classify_4701,
    "experiment_4710_arms_summary": _classify_4710,
    "experiment_4712_perception_grounded_l2_goal_lp85": _classify_4712,
    "experiment_4713_surface_present_winner_verifier_ranker": _classify_4713,
    "experiment_4715_online_action_learning_driver_corrected": _classify_4715,
}


def classify_null(null_id: str, artifact: Mapping[str, Any]) -> JsonDict:
    """REQ-ARC-WMTE-4725: classify one null artifact from exercise evidence."""

    evidence: list[str] = []
    signatures: list[str] = []
    classifier = CLASSIFIERS.get(null_id)
    if classifier is None:
        _append_unique(signatures, "unknown_null_scope")
    else:
        classifier(artifact, evidence, signatures)
    verdict = "silent_bug_must_reopen" if signatures else "trustworthy_null"
    if verdict == "trustworthy_null" and not evidence:
        _append_unique(evidence, "no silent representation no-op signature found")
    return {
        "null_id": null_id,
        "verdict": verdict,
        "evidence": evidence,
        "exercise_evidence": list(evidence),
        "silent_bug_signatures": signatures,
    }


def go_explore_fix_confirmed() -> bool:
    """Confirm _frame_grid squeezes a leading singleton channel to a 2-D grid."""

    try:
        from carnot.agentic import arc_go_explore

        frame = SimpleNamespace(frame=np.zeros((1, 64, 64), dtype=np.int16), levels_completed=0)
        grid = np.asarray(arc_go_explore._frame_grid(frame))
        return bool(grid.shape == (64, 64) and grid.ndim == 2)
    except Exception:
        return False


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    """REQ-ARC-WMTE-4725: verify audited resources before producing the artifact."""

    root_path = Path(root)
    missing = [
        target["artifact_path"]
        for target in AUDIT_TARGETS
        if not (root_path / target["artifact_path"]).exists()
    ]
    missing_modules = [
        target["module_path"] for target in AUDIT_TARGETS if not (root_path / target["module_path"]).exists()
    ]
    try:
        from carnot.agentic import arc_go_explore  # noqa: F401

        arc_go_explore_importable = True
        arc_go_explore_error = ""
    except Exception as exc:
        arc_go_explore_importable = False
        arc_go_explore_error = repr(exc)[:300]
    fix_confirmed = go_explore_fix_confirmed() if arc_go_explore_importable else False
    ok = bool(not missing and arc_go_explore_importable and fix_confirmed)
    return {
        "ok": ok,
        "null_artifacts_present": not missing,
        "missing_artifacts": missing,
        "module_files_present": not missing_modules,
        "missing_modules": missing_modules,
        "arc_go_explore_importable": arc_go_explore_importable,
        "arc_go_explore_error": arc_go_explore_error,
        "go_explore_frame_grid_fix_confirmed": fix_confirmed,
        "resolved_aliases": {
            "results/experiment_4710_online_action_learning_arms_summary.json": (
                "results/experiment_4710_arms_summary.json"
            )
        },
    }


def _blocked_artifact(checks: Mapping[str, Any], duration_s: float) -> JsonDict:
    blocked = "blocked_null_artifacts_missing"
    if not checks.get("arc_go_explore_importable"):
        blocked = "blocked_arc_go_explore_import"
    elif not checks.get("go_explore_frame_grid_fix_confirmed"):
        blocked = "blocked_arc_go_explore_frame_grid_fix_unconfirmed"
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": ["REQ-ARC-WMTE-4725", "SCENARIO-ARC-WMTE-4725-AUDIT-CLASSIFICATION"],
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": blocked,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "nulls_audited": 0,
        "silent_bug_nulls": [],
        "a4_tautology_verdict": "blocked_preconditions",
        "trustworthy_nulls": [],
        "reopen_recommendations": [],
        "go_explore_fix_confirmed": bool(checks.get("go_explore_frame_grid_fix_confirmed")),
        "audit_report_path": REPORT_RELATIVE_PATH,
        "verifier_is_oracle": False,
        "random_seed": RANDOM_SEED,
        "preconditions_checked": dict(checks),
        "audited_artifacts": [],
        "audited_artifact_checksums": {},
        "per_null_verdicts": [],
        "duration_s": round(max(0.001, duration_s), 6),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _recommendations(silent_bug_nulls: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    ids = {str(row.get("null_id") or "") for row in silent_bug_nulls}
    rows: list[JsonDict] = []
    if "experiment_4715_online_action_learning_driver_corrected" in ids:
        rows.append(
            {
                "priority": "P0",
                "lever": "online_action_learning_driver",
                "recommendation": "reopen_as_435_A1",
                "source_nulls": ["experiment_4715_online_action_learning_driver_corrected"],
                "reason": (
                    "The .434 A4 artifact has frozen/scratch/warm first-win all equal to 0.04 "
                    "with zero delta and flagged_adversarial=true; the faithful driver was not "
                    "validly distinguished."
                ),
            }
        )
    if "experiment_4701_amortized_exploration_prior_go_explore_live" in ids:
        rows.append(
            {
                "priority": "P1",
                "lever": "amortized_prior_go_explore_archive",
                "recommendation": "rerun_after_frame_grid_fix_with_positive_archive_cells",
                "source_nulls": ["experiment_4701_amortized_exploration_prior_go_explore_live"],
                "reason": (
                    "The archive arm reported zero observations, zero cells, and zero prefix "
                    "injections, matching the confirmed (1,64,64) frame-grid no-op."
                ),
            }
        )
    if "experiment_4640_goal_energy_generation_live" in ids:
        rows.append(
            {
                "priority": "P2",
                "lever": "goal_energy_generation_live",
                "recommendation": "rerun_only_if_goal_energy_scores_real_candidate_states",
                "source_nulls": ["experiment_4640_goal_energy_generation_live"],
                "reason": (
                    "The goal-energy arm cloned cached baseline attempts and marked them "
                    "goal_energy_neutral_on_cached_frame, so generation was not exercised."
                ),
            }
        )
    if "experiment_4653_energy_fitness_qd_generation_live" in ids:
        rows.append(
            {
                "priority": "P3",
                "lever": "energy_fitness_qd_generation",
                "recommendation": "rerun_with_distinct_qd_and_random_mutation_candidate_pools",
                "source_nulls": ["experiment_4653_energy_fitness_qd_generation_live"],
                "reason": (
                    "The QD, random-QD, and search arms are byte-identical except for arm labels."
                ),
            }
        )
    if "experiment_4676_hierarchical_subgoal_search_live" in ids:
        rows.append(
            {
                "priority": "P4",
                "lever": "hierarchical_subgoal_search",
                "recommendation": "rerun_only_after_nonempty_subgoal_decomposition_gate",
                "source_nulls": ["experiment_4676_hierarchical_subgoal_search_live"],
                "reason": (
                    "The subgoal-search arm emitted an empty subgoal decomposition and no reachable "
                    "subgoal evidence."
                ),
            }
        )
    if "experiment_4710_arms_summary" in ids:
        rows.append(
            {
                "priority": "P5",
                "lever": "online_action_learning_graft",
                "recommendation": "rerun_after_dict_candidate_normalization",
                "source_nulls": ["experiment_4710_arms_summary"],
                "reason": "The online CNN arm lacks positive observe/fit/no-error evidence.",
            }
        )
    return rows


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    per_null_verdicts: Sequence[Mapping[str, Any]],
    audited_artifact_checksums: Mapping[str, str],
    duration_s: float,
) -> JsonDict:
    silent = [dict(row) for row in per_null_verdicts if row.get("verdict") == "silent_bug_must_reopen"]
    trustworthy = [dict(row) for row in per_null_verdicts if row.get("verdict") == "trustworthy_null"]
    a4_silent = any(
        row.get("null_id") == "experiment_4715_online_action_learning_driver_corrected"
        for row in silent
    )
    a4_verdict = (
        "online_driver_arms_degenerate (no-op, must reopen)"
        if a4_silent
        else "trustworthy_null"
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": ["REQ-ARC-WMTE-4725", "SCENARIO-ARC-WMTE-4725-AUDIT-CLASSIFICATION"],
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": (
            f"complete: silent_bug_audit_{len(per_null_verdicts)}_nulls_"
            f"{len(silent)}_must_reopen"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "nulls_audited": int(len(per_null_verdicts)),
        "silent_bug_nulls": silent,
        "a4_tautology_verdict": a4_verdict,
        "trustworthy_nulls": trustworthy,
        "reopen_recommendations": _recommendations(silent),
        "go_explore_fix_confirmed": bool(
            preconditions_checked.get("go_explore_frame_grid_fix_confirmed")
        ),
        "audit_report_path": REPORT_RELATIVE_PATH,
        "verifier_is_oracle": False,
        "random_seed": RANDOM_SEED,
        "preconditions_checked": dict(preconditions_checked),
        "audited_artifacts": [target["artifact_path"] for target in AUDIT_TARGETS],
        "audited_artifact_checksums": dict(audited_artifact_checksums),
        "per_null_verdicts": [dict(row) for row in per_null_verdicts],
        "duration_s": round(max(0.001, float(duration_s)), 6),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [f"missing:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")
    if artifact.get("audit_report_path") != REPORT_RELATIVE_PATH:
        errors.append("audit_report_path_mismatch")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_mismatch")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if not isinstance(artifact.get("silent_bug_nulls"), list):
        errors.append("silent_bug_nulls_must_be_list")
    if not isinstance(artifact.get("trustworthy_nulls"), list):
        errors.append("trustworthy_nulls_must_be_list")
    if not isinstance(artifact.get("reopen_recommendations"), list):
        errors.append("reopen_recommendations_must_be_list")
    if artifact.get("honest_verdict", "").startswith("complete:"):
        expected = len(artifact.get("silent_bug_nulls") or []) + len(
            artifact.get("trustworthy_nulls") or []
        )
        if artifact.get("nulls_audited") != expected:
            errors.append("nulls_audited_does_not_match_verdict_lists")
    return errors


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _report_row(row: Mapping[str, Any]) -> str:
    evidence = "; ".join(str(item) for item in row.get("evidence", [])[:2])
    signatures = ", ".join(str(item) for item in row.get("silent_bug_signatures", [])) or "-"
    return (
        f"| `{row.get('null_id')}` | `{row.get('verdict')}` | "
        f"{_short(signatures, 160)} | {_short(evidence, 260)} |"
    )


def render_report(artifact: Mapping[str, Any]) -> str:
    lines = [
        "# ARC Null Silent-Bug Audit",
        "",
        f"- Verdict: `{artifact.get('honest_verdict')}`",
        f"- Nulls audited: `{artifact.get('nulls_audited')}`",
        f"- Must reopen: `{len(artifact.get('silent_bug_nulls') or [])}`",
        f"- A4 tautology verdict: `{artifact.get('a4_tautology_verdict')}`",
        f"- Go-Explore `_frame_grid` fix confirmed: `{artifact.get('go_explore_fix_confirmed')}`",
        "",
        "## Per-Null Verdicts",
        "",
        "| Null | Verdict | Silent signatures | Evidence |",
        "|---|---|---|---|",
    ]
    for row in artifact.get("per_null_verdicts", []):
        if isinstance(row, Mapping):
            lines.append(_report_row(row))
    lines.extend(["", "## Prioritized Reopen Recommendations", ""])
    recommendations = artifact.get("reopen_recommendations") or []
    if not recommendations:
        lines.append("No reopen recommendations.")
    else:
        for row in recommendations:
            if isinstance(row, Mapping):
                lines.append(
                    f"- `{row.get('priority')}` `{row.get('lever')}`: "
                    f"{row.get('recommendation')} - {row.get('reason')}"
                )
    lines.extend(["", "## Preconditions", "", "```json"])
    lines.append(json.dumps(artifact.get("preconditions_checked"), indent=2, sort_keys=True))
    lines.extend(["```", ""])
    return "\n".join(lines)


def write_report(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    path = Path(root) / REPORT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(artifact), encoding="utf-8")
    return path


def run(root: Path | str = REPO_ROOT, *, write: bool = True) -> JsonDict:
    started = time.monotonic()
    root_path = Path(root)
    checks = check_preconditions(root_path)
    if not checks.get("ok"):
        artifact = _blocked_artifact(checks, time.monotonic() - started)
        if write:
            write_report(artifact, root=root_path)
            write_artifact(artifact, root=root_path)
        return artifact

    verdicts: list[JsonDict] = []
    checksums: dict[str, str] = {}
    for target in AUDIT_TARGETS:
        rel = target["artifact_path"]
        path = root_path / rel
        data = _read_json(path)
        checksums[rel] = _file_checksum(path)
        module_path = root_path / target["module_path"]
        if module_path.exists():
            checksums[target["module_path"]] = _file_checksum(module_path)
        verdict = classify_null(target["null_id"], data)
        verdict["artifact_path"] = rel
        verdict["module_path"] = target["module_path"]
        verdicts.append(verdict)

    artifact = build_artifact(
        preconditions_checked=checks,
        per_null_verdicts=verdicts,
        audited_artifact_checksums=checksums,
        duration_s=time.monotonic() - started,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_report(artifact, root=root_path)
        write_artifact(artifact, root=root_path)
    return artifact


def main() -> int:  # pragma: no cover - CLI shim.
    artifact = run(REPO_ROOT, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if not artifact_schema_errors(artifact) else 1


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
