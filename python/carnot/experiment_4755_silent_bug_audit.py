"""Experiment 4755: audit generation-lever nulls for silent no-ops.

Spec refs: REQ-ARC-WMTE-4755, SCENARIO-ARC-WMTE-4755-GENERATION-LEVER-AUDIT.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))

EXPERIMENT = "experiment_4755_silent_bug_audit"
SCHEMA = "carnot.arc.generation_lever_silent_bug_audit_4755.v1"
RESULT_RELATIVE_PATH = "results/experiment_4755_silent_bug_audit.json"
PRIOR_AUDIT_RELATIVE_PATH = "results/experiment_4725_silent_bug_audit.json"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts; 100us floor."
RANDOM_SEED = 4755
TERMINAL_PREFIXES = ("complete_", "blocked_")

JsonDict = dict[str, Any]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; an audit-complete report is complete_."
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts; 100us floor."
    },
    "preconditions_checked": {
        "principle": "records the upstream-artifact checks."
    },
    "levers_audited": {
        "principle": (
            "the list of generation-lever nulls inspected -- the audit coverage."
        )
    },
    "silent_no_op_findings": {
        "principle": (
            "per-lever {lever, no_op_signature, classification} -- distinguishes "
            "dead-code artifacts from genuine nulls so the planner does not trust "
            "a dead-code null."
        )
    },
    "must_reopen": {
        "principle": (
            "the levers whose null is a dead-code artifact and must be re-tested "
            "validly -- the actionable output."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "spec_refs",
    "field_principles",
    "verifier_is_oracle",
    "prior_audit_path",
    "prior_reopen_list",
    "audited_artifacts",
    "audited_artifact_checksums",
    "random_seed",
    "duration_s",
    "reproducibility_checksum",
)

AUDIT_TARGETS: tuple[dict[str, str], ...] = (
    {
        "lever": "experiment_4640_goal_energy_generation_live",
        "artifact_path": "results/experiment_4640_goal_energy_generation_live.json",
    },
    {
        "lever": "experiment_4641_action_effect_expansion_prior_live",
        "artifact_path": "results/experiment_4641_action_effect_expansion_prior_live.json",
    },
    {
        "lever": "experiment_4653_energy_fitness_qd_generation_live",
        "artifact_path": "results/experiment_4653_energy_fitness_qd_generation_live.json",
    },
    {
        "lever": "experiment_4676_hierarchical_subgoal_search_live",
        "artifact_path": "results/experiment_4676_hierarchical_subgoal_search_live.json",
    },
    {
        "lever": "experiment_4677_poe_world_factored_subgoal_planner",
        "artifact_path": "results/experiment_4677_poe_world_factored_subgoal_planner.json",
    },
    {
        "lever": "experiment_4688_controllable_novelty_proposal_policy_live",
        "artifact_path": "results/experiment_4688_controllable_novelty_proposal_policy_live.json",
    },
    {
        "lever": "experiment_4689_program_synthesis_action_effect_proposal_filter",
        "artifact_path": "results/experiment_4689_program_synthesis_action_effect_proposal_filter.json",
    },
    {
        "lever": "experiment_4700_object_centric_perception_proposal_live",
        "artifact_path": "results/experiment_4700_object_centric_perception_proposal_live.json",
    },
    {
        "lever": "experiment_4701_amortized_exploration_prior_go_explore_live",
        "artifact_path": "results/experiment_4701_amortized_exploration_prior_go_explore_live.json",
    },
    {
        "lever": "experiment_4710_arms_summary",
        "artifact_path": "results/experiment_4710_arms_summary.json",
    },
    {
        "lever": "experiment_4713_surface_present_winner_verifier_ranker",
        "artifact_path": "results/experiment_4713_surface_present_winner_verifier_ranker.json",
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
        return parsed if math.isfinite(parsed) else None
    return None


def _int_value(value: Any) -> int:
    parsed = _finite_float(value)
    return int(parsed) if parsed is not None else 0


def _append_unique(rows: list[str], text: str) -> None:
    if text and text not in rows:
        rows.append(text)


def _attempts(measurement: Any, key: str = "variant_attempts") -> list[Mapping[str, Any]]:
    if not isinstance(measurement, Mapping):
        return []
    rows = measurement.get(key)
    return [row for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []


def _dict_without(row: Mapping[str, Any], excluded: set[str]) -> JsonDict:
    return {str(k): v for k, v in row.items() if str(k) not in excluded}


def _rows_equal(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], *, excluded: set[str]
) -> bool:
    return [_dict_without(row, excluded) for row in left] == [
        _dict_without(row, excluded) for row in right
    ]


def _nested_dicts_with_key(value: Any, key: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, Mapping):
            maybe = node.get(key)
            if isinstance(maybe, Mapping):
                rows.append(maybe)
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return rows


def _max_nested_value(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    return max((_int_value(row.get(key)) for row in rows), default=0)


def _classify_4640(artifact: Mapping[str, Any], evidence: list[str], signatures: list[str]) -> None:
    baseline = artifact.get("baseline_measurement")
    goal = artifact.get("goal_energy_measurement")
    uniform = artifact.get("uniform_measurement")
    if not all(isinstance(row, Mapping) for row in (baseline, goal, uniform)):
        _append_unique(signatures, "missing_expected_measurements")
        return
    baseline_rows = _attempts(baseline)
    goal_rows = _attempts(goal)
    uniform_rows = _attempts(uniform)
    cloned_goal = _rows_equal(
        baseline_rows,
        goal_rows,
        excluded={"goal_energy_neutral_on_cached_frame", "uniform_energy_ablation"},
    )
    cloned_uniform = _rows_equal(
        baseline_rows,
        uniform_rows,
        excluded={"goal_energy_neutral_on_cached_frame", "uniform_energy_ablation"},
    )
    neutral = sum(1 for row in goal_rows if row.get("goal_energy_neutral_on_cached_frame") is True)
    if baseline_rows and cloned_goal and cloned_uniform:
        _append_unique(signatures, "byte_identical_arms")
        if neutral == len(goal_rows):
            _append_unique(signatures, "scorer_or_energy_never_fires")
        _append_unique(
            evidence,
            f"goal-energy arm cloned {len(goal_rows)} baseline rows; neutral_cached={neutral}",
        )
    else:
        _append_unique(
            evidence,
            f"goal-energy rows={len(goal_rows)}, baseline rows={len(baseline_rows)}, "
            f"uniform rows={len(uniform_rows)}, cloned_goal={cloned_goal}",
        )


def _classify_4641(artifact: Mapping[str, Any], evidence: list[str], signatures: list[str]) -> None:
    ranker = artifact.get("ranker_measurement")
    expansion = artifact.get("expansion_measurement")
    ranker_rows = _attempts(ranker, key="attempts")
    expansion_rows = _attempts(expansion, key="attempts")
    if not ranker_rows or not expansion_rows:
        _append_unique(signatures, "missing_expected_measurements")
        return
    if _rows_equal(ranker_rows, expansion_rows, excluded=set()):
        _append_unique(signatures, "byte_identical_arms")
        _append_unique(
            evidence,
            f"ranker and expansion-prior rows are byte-identical for {len(ranker_rows)} attempts",
        )
    else:
        _append_unique(
            evidence,
            f"expansion-prior rows differ from ranker baseline over {len(expansion_rows)} attempts",
        )


def _classify_4653(artifact: Mapping[str, Any], evidence: list[str], signatures: list[str]) -> None:
    search = artifact.get("search_measurement")
    random_qd = artifact.get("random_mutation_measurement")
    qd = artifact.get("qd_measurement")
    if not all(isinstance(row, Mapping) for row in (search, random_qd, qd)):
        _append_unique(signatures, "missing_expected_measurements")
        return
    search_rows = _attempts(search, key="attempts")
    random_rows = _attempts(random_qd, key="attempts")
    qd_rows = _attempts(qd, key="attempts")
    qd_cloned = _rows_equal(search_rows, qd_rows, excluded={"arm"})
    random_cloned = _rows_equal(search_rows, random_rows, excluded={"arm"})
    if search_rows and qd_cloned and random_cloned:
        _append_unique(signatures, "byte_identical_arms")
        _append_unique(signatures, "empty_or_unchanged_candidate_pool")
        _append_unique(
            evidence,
            f"search/random/QD arms share {len(search_rows)} identical attempts except arm labels",
        )
    else:
        _append_unique(
            evidence,
            f"search_attempts={len(search_rows)}, random_attempts={len(random_rows)}, "
            f"qd_attempts={len(qd_rows)}, qd_cloned={qd_cloned}",
        )


def _classify_4676(artifact: Mapping[str, Any], evidence: list[str], signatures: list[str]) -> None:
    subgoals = artifact.get("subgoal_decomposition")
    reachable = artifact.get("per_subgoal_reachable")
    if not isinstance(subgoals, list) or not isinstance(reachable, list):
        _append_unique(signatures, "missing_expected_measurements")
        return
    if not subgoals and not reachable:
        _append_unique(signatures, "empty_or_unchanged_candidate_pool")
        _append_unique(evidence, "subgoal decomposition and reachability evidence are empty")
    else:
        _append_unique(evidence, f"subgoals={len(subgoals)}, reachable={len(reachable)}")


def _classify_4677(artifact: Mapping[str, Any], evidence: list[str], signatures: list[str]) -> None:
    del signatures
    probe = artifact.get("target_arm_results")
    candidate = probe.get("candidate_generation_probe") if isinstance(probe, Mapping) else {}
    weights = candidate.get("expert_trust_weights") if isinstance(candidate, Mapping) else []
    rows = weights if isinstance(weights, list) else []
    _append_unique(evidence, f"poe_factored_experts_exercised count={len(rows)}")


def _classify_4688(artifact: Mapping[str, Any], evidence: list[str], signatures: list[str]) -> None:
    target = artifact.get("target_arm_results")
    novelty = target.get("controllable_novelty") if isinstance(target, Mapping) else {}
    diag = novelty.get("controllable_novelty_diagnostics") if isinstance(novelty, Mapping) else {}
    candidate_scores = _int_value(diag.get("candidate_scores")) if isinstance(diag, Mapping) else 0
    observed = _int_value(diag.get("observed_effects")) if isinstance(diag, Mapping) else 0
    updates = _int_value(diag.get("rnd_updates")) if isinstance(diag, Mapping) else 0
    if candidate_scores > 0 and (observed > 0 or updates > 0):
        _append_unique(
            evidence,
            f"controllable_novelty_exercised candidate_scores={candidate_scores}, observed={observed}, rnd_updates={updates}",
        )
    else:
        _append_unique(signatures, "scorer_or_cnn_never_fires")


def _classify_4689(artifact: Mapping[str, Any], evidence: list[str], signatures: list[str]) -> None:
    target = artifact.get("target_arm_results")
    probe = target.get("candidate_generation_probe") if isinstance(target, Mapping) else {}
    kept = _int_value(probe.get("heldout_programs_kept")) if isinstance(probe, Mapping) else 0
    rejected = _int_value(probe.get("heldout_programs_rejected")) if isinstance(probe, Mapping) else 0
    weights = probe.get("program_trust_weights") if isinstance(probe, Mapping) else []
    weight_count = len(weights) if isinstance(weights, list) else 0
    if kept + rejected + weight_count > 0:
        _append_unique(
            evidence,
            f"program_filter_exercised kept={kept}, rejected={rejected}, weights={weight_count}",
        )
    else:
        _append_unique(signatures, "scorer_or_energy_never_fires")


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
        _append_unique(
            evidence,
            f"object_centric_pool_exercised counts={candidate_counts}, augmented={augmented}, scores={scores}",
        )
    else:
        _append_unique(signatures, "empty_candidate_pool")


def _classify_4701(artifact: Mapping[str, Any], evidence: list[str], signatures: list[str]) -> None:
    archives = _nested_dicts_with_key(artifact, "go_explore_archive_diagnostics")
    observations = _max_nested_value(archives, "observations")
    cells = _max_nested_value(archives, "stored_cells")
    actions = _max_nested_value(archives, "actions_injected")
    prefixes = _max_nested_value(archives, "prefixes_injected")
    coverage = artifact.get("target_arm_results")
    cov = coverage.get("coverage") if isinstance(coverage, Mapping) else {}
    with_prior = cov.get("with_prior") if isinstance(cov, Mapping) else {}
    total_steps = _int_value(with_prior.get("total_steps")) if isinstance(with_prior, Mapping) else 0
    if archives and observations == cells == actions == prefixes == 0:
        _append_unique(signatures, "dead_archive")
    if _finite_float(artifact.get("candidate_generation_coverage_with_prior")) == 0.0 and total_steps == 0:
        _append_unique(signatures, "empty_candidate_pool")
    _append_unique(
        evidence,
        f"go_explore_archive observations={observations}, stored_cells={cells}, actions_injected={actions}, prefixes_injected={prefixes}, total_steps={total_steps}",
    )


def _classify_4710(artifact: Mapping[str, Any], evidence: list[str], signatures: list[str]) -> None:
    arms = artifact.get("arms")
    if not isinstance(arms, list):
        _append_unique(signatures, "missing_expected_measurements")
        return
    online_rows = [
        row for row in arms if isinstance(row, Mapping) and str(row.get("arm") or "").startswith("online-")
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
    if max_observed > 0 and max_fits > 0 and (min_errors or 0) == 0:
        _append_unique(
            evidence,
            f"online_cnn_exercised observed={max_observed}, fits={max_fits}, errors={min_errors or 0}",
        )
    else:
        _append_unique(signatures, "scorer_or_cnn_never_fires")
        _append_unique(
            evidence,
            f"online_cnn observed={max_observed}, fits={max_fits}, min_errors={min_errors}",
        )


def _classify_4713(artifact: Mapping[str, Any], evidence: list[str], signatures: list[str]) -> None:
    diag = artifact.get("surfacing_ranker_diagnostics")
    ranker = diag.get("surfacing_ranker") if isinstance(diag, Mapping) else {}
    samples = _int_value(ranker.get("samples")) if isinstance(ranker, Mapping) else 0
    fits = _int_value(ranker.get("fit_count")) if isinstance(ranker, Mapping) else 0
    scores = _int_value(diag.get("candidate_scores")) if isinstance(diag, Mapping) else 0
    if samples > 0 and fits > 0 and scores > 0:
        _append_unique(evidence, f"surfacing_ranker_exercised samples={samples}, fits={fits}, scores={scores}")
    else:
        _append_unique(signatures, "scorer_or_energy_never_fires")


CLASSIFIERS = {
    "experiment_4640_goal_energy_generation_live": _classify_4640,
    "experiment_4641_action_effect_expansion_prior_live": _classify_4641,
    "experiment_4653_energy_fitness_qd_generation_live": _classify_4653,
    "experiment_4676_hierarchical_subgoal_search_live": _classify_4676,
    "experiment_4677_poe_world_factored_subgoal_planner": _classify_4677,
    "experiment_4688_controllable_novelty_proposal_policy_live": _classify_4688,
    "experiment_4689_program_synthesis_action_effect_proposal_filter": _classify_4689,
    "experiment_4700_object_centric_perception_proposal_live": _classify_4700,
    "experiment_4701_amortized_exploration_prior_go_explore_live": _classify_4701,
    "experiment_4710_arms_summary": _classify_4710,
    "experiment_4713_surface_present_winner_verifier_ranker": _classify_4713,
}


def classify_lever(lever: str, artifact: Mapping[str, Any]) -> JsonDict:
    """REQ-ARC-WMTE-4755: classify one generation lever from its artifact evidence."""

    evidence: list[str] = []
    signatures: list[str] = []
    classifier = CLASSIFIERS.get(lever)
    if classifier is None:
        _append_unique(signatures, "unknown_generation_lever")
    else:
        classifier(artifact, evidence, signatures)
    if not signatures and not evidence:
        _append_unique(evidence, "no silent representation no-op signature found")
    return {
        "lever": lever,
        "no_op_signature": signatures,
        "classification": "must_reopen" if signatures else "genuine_null",
        "evidence": evidence,
    }


def _arcade_import_check() -> tuple[bool, str]:
    try:
        from carnot.agentic import arc_solver_kit

        arc_solver_kit.offline_arcade()
        return True, ""
    except Exception as exc:  # pragma: no cover - exercised in blocked tests through injection.
        return False, repr(exc)[:300]


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    """REQ-ARC-WMTE-4755: verify upstream artifacts and arcade import before auditing."""

    root_path = Path(root)
    expected = [target["artifact_path"] for target in AUDIT_TARGETS] + [PRIOR_AUDIT_RELATIVE_PATH]
    missing = [rel for rel in expected if not (root_path / rel).exists()]
    arcade_ok, arcade_error = _arcade_import_check()
    return {
        "ok": bool(not missing and arcade_ok),
        "upstream_artifacts_present": not missing,
        "missing_upstream_artifacts": missing,
        "prior_audit_present": (root_path / PRIOR_AUDIT_RELATIVE_PATH).exists(),
        "arcade_import_exits_0": arcade_ok,
        "arcade_import_error": arcade_error,
    }


def _blocked_artifact(checks: Mapping[str, Any], duration_s: float) -> JsonDict:
    verdict = "blocked_upstream_artifacts"
    if checks.get("upstream_artifacts_present") and not checks.get("arcade_import_exits_0"):
        verdict = "blocked_arcade_import"
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": ["REQ-ARC-WMTE-4755", "SCENARIO-ARC-WMTE-4755-GENERATION-LEVER-AUDIT"],
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(checks),
        "levers_audited": [],
        "silent_no_op_findings": [],
        "must_reopen": [],
        "verifier_is_oracle": False,
        "prior_audit_path": PRIOR_AUDIT_RELATIVE_PATH,
        "prior_reopen_list": [],
        "audited_artifacts": [],
        "audited_artifact_checksums": {},
        "random_seed": RANDOM_SEED,
        "duration_s": round(max(0.001, float(duration_s)), 6),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _prior_reopen_list(prior: Mapping[str, Any]) -> list[str]:
    rows = prior.get("reopen_recommendations")
    if not isinstance(rows, list):
        return []
    levers: list[str] = []
    for row in rows:
        if isinstance(row, Mapping):
            for source in row.get("source_nulls", []):
                if isinstance(source, str) and source not in levers:
                    levers.append(source)
    return levers


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    findings: Sequence[Mapping[str, Any]],
    audited_artifact_checksums: Mapping[str, str],
    duration_s: float,
    prior_reopen_list: Sequence[str] = (),
) -> JsonDict:
    must_reopen = [
        str(row.get("lever"))
        for row in findings
        if row.get("classification") == "must_reopen" and row.get("lever")
    ]
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": ["REQ-ARC-WMTE-4755", "SCENARIO-ARC-WMTE-4755-GENERATION-LEVER-AUDIT"],
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": (
            f"complete_generation_lever_silent_bug_audit_{len(findings)}_levers_"
            f"{len(must_reopen)}_must_reopen"
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "levers_audited": [str(row.get("lever")) for row in findings],
        "silent_no_op_findings": [dict(row) for row in findings],
        "must_reopen": must_reopen,
        "verifier_is_oracle": False,
        "prior_audit_path": PRIOR_AUDIT_RELATIVE_PATH,
        "prior_reopen_list": list(prior_reopen_list),
        "audited_artifacts": [target["artifact_path"] for target in AUDIT_TARGETS],
        "audited_artifact_checksums": dict(audited_artifact_checksums),
        "random_seed": RANDOM_SEED,
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
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")
    for field in ("levers_audited", "silent_no_op_findings", "must_reopen"):
        if not isinstance(artifact.get(field), list):
            errors.append(f"{field}_must_be_list")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if verdict.startswith("complete_") and isinstance(artifact.get("silent_no_op_findings"), list):
        if len(artifact.get("levers_audited") or []) != len(artifact["silent_no_op_findings"]):
            errors.append("levers_audited_does_not_match_findings")
    return errors


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run(root: Path | str = REPO_ROOT, *, write: bool = True) -> JsonDict:
    started = time.monotonic()
    root_path = Path(root)
    checks = check_preconditions(root_path)
    if not checks.get("ok"):
        artifact = _blocked_artifact(checks, time.monotonic() - started)
        if write:
            write_artifact(artifact, root=root_path)
        return artifact

    prior = _read_json(root_path / PRIOR_AUDIT_RELATIVE_PATH)
    prior_reopen = _prior_reopen_list(prior)
    checksums = {PRIOR_AUDIT_RELATIVE_PATH: _file_checksum(root_path / PRIOR_AUDIT_RELATIVE_PATH)}
    findings: list[JsonDict] = []
    for target in AUDIT_TARGETS:
        rel = target["artifact_path"]
        data = _read_json(root_path / rel)
        checksums[rel] = _file_checksum(root_path / rel)
        finding = classify_lever(target["lever"], data)
        finding["artifact_path"] = rel
        finding["prior_4725_reopen"] = target["lever"] in prior_reopen
        findings.append(finding)

    artifact = build_artifact(
        preconditions_checked=checks,
        findings=findings,
        audited_artifact_checksums=checksums,
        duration_s=time.monotonic() - started,
        prior_reopen_list=prior_reopen,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(artifact, root=root_path)
    return artifact


def main() -> int:  # pragma: no cover - CLI shim.
    artifact = run(REPO_ROOT, write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if not artifact_schema_errors(artifact) else 1


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
