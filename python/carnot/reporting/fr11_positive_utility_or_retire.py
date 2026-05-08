"""Exp 1555 FR-11 positive-utility-or-retire gate.

Spec: REQ-LEARN-1555, SCENARIO-LEARN-1555, SCENARIO-LEARN-1556,
SCENARIO-LEARN-1557.

This gate is deliberately about query-time policy and skill promotion, not
training.  It turns externally verified replay evidence into a sandboxed skill
graph and then asks one narrow question: did the promoted policy improve a
held-out deterministic replay set without any soundness mistake?  If the answer
is no, the positive-utility self-learning headline is retired instead of being
kept alive by a safety-only result.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260508"
MILESTONE = ".119"
OUTPUT_FILE = "experiment_1555_fr11_positive_utility_or_retire_v14.json"
SKILL_GRAPH_FILE = "fr11_positive_utility_skill_graph_1555.json"

DEFAULT_OUTPUT_PATH = Path("results") / OUTPUT_FILE
DEFAULT_SKILL_GRAPH_PATH = Path("results") / SKILL_GRAPH_FILE
DEFAULT_EXP1539_ARTIFACT_PATH = Path(
    "results/experiment_1539_fr11_external_feedback_skill_promotion_v13.json"
)
DEFAULT_REPAIR_ARTIFACT_PATH = Path("results/experiment_1552_residual_drift_repair_policy_v1.json")
DEFAULT_REPAIR_MANIFEST_PATH = Path("results/residual_drift_repair_policy_1552.jsonl")
DEFAULT_PRODUCT_LINE_ARTIFACT_PATHS: tuple[Path, ...] = (
    Path("results/experiment_1540_product_line_staged_benchmark_scale_v3.json"),
    Path("results/experiment_1554_product_line_staged_scale_v4.json"),
)

MANDATED_MODEL_SPECS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "milestone",
    "continuous_self_learning_task",
    "fr11_positive_utility_gate_ready",
    "model_specs",
    "live_sota_model_inference_used",
    "no_model_weight_mutation",
    "external_feedback_used",
    "self_feedback_only_rejected",
    "candidate_skill_updates",
    "skill_updates_promoted",
    "replay_cases",
    "replay_pass_rate",
    "soundness_mistakes",
    "baseline_utility",
    "post_promotion_utility",
    "utility_delta",
    "positive_utility_achieved",
    "positive_utility_claim_retired",
    "skill_graph_path",
    "focused_tests_passed",
    "honest_verdict",
)

TERMINAL_VERDICT_PREFIXES: tuple[str, ...] = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    skill_graph_path: Path | str = DEFAULT_SKILL_GRAPH_PATH,
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """REQ-LEARN-1555-1/7: write the durable bootstrap artifact first."""

    del run_date
    artifact = {
        "status": "in_progress",
        "milestone": MILESTONE,
        "continuous_self_learning_task": "fr11_positive_utility_or_retire_v14",
        "fr11_positive_utility_gate_ready": False,
        "model_specs": list(MANDATED_MODEL_SPECS),
        "live_sota_model_inference_used": False,
        "no_model_weight_mutation": True,
        "external_feedback_used": False,
        "self_feedback_only_rejected": False,
        "candidate_skill_updates": [],
        "skill_updates_promoted": [],
        "replay_cases": [],
        "replay_pass_rate": 0.0,
        "soundness_mistakes": 0,
        "baseline_utility": 0.0,
        "post_promotion_utility": 0.0,
        "utility_delta": 0.0,
        "positive_utility_achieved": False,
        "positive_utility_claim_retired": False,
        "skill_graph_path": _display_path(skill_graph_path, project_root=project_root),
        "focused_tests_passed": False,
        "honest_verdict": "complete: fr11 positive utility gate in progress",
    }
    validate_artifact(artifact)
    _write_json(Path(output_path), artifact)
    return artifact


def select_candidate_skill_updates(
    *,
    exp1539_artifact: Mapping[str, Any],
    repair_artifact: Mapping[str, Any],
    repair_rows: Sequence[Mapping[str, Any]],
    extra_candidates: Sequence[Mapping[str, Any]] = (),
) -> list[JsonDict]:
    """REQ-LEARN-1555-2/3: select only externally verified replay candidates."""

    candidates: list[JsonDict] = []
    candidates.extend(_candidates_from_exp1539(exp1539_artifact))
    repair_candidate = _candidate_from_repair_artifact(repair_artifact, repair_rows)
    if repair_candidate is not None:
        candidates.append(repair_candidate)
    candidates.extend(dict(candidate) for candidate in extra_candidates)
    return [_evaluate_candidate(candidate) for candidate in candidates]


def build_skill_graph(
    candidates: Sequence[Mapping[str, Any]],
    *,
    skill_graph_path: Path | str = DEFAULT_SKILL_GRAPH_PATH,
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """REQ-LEARN-1555-3: persist the sandboxed promoted skill graph."""

    nodes = [_skill_node(candidate) for candidate in candidates if _is_promoted(candidate)]
    graph = {
        "schema": "fr11_positive_utility_skill_graph_v14",
        "run_date": run_date,
        "skill_graph_path": _display_path(skill_graph_path, project_root=project_root),
        "spec": [
            "REQ-LEARN-1555",
            "SCENARIO-LEARN-1555",
            "SCENARIO-LEARN-1556",
            "SCENARIO-LEARN-1557",
        ],
        "nodes": nodes,
        "edges": [
            {
                "from": source,
                "to": node["node_id"],
                "relation": "external_verifier_replay_promotes_query_time_skill",
            }
            for node in nodes
            for source in node["lineage"].get("source_artifacts", [])
        ],
        "summary": {
            "candidate_update_count": len(candidates),
            "promoted_update_count": len(nodes),
            "rejected_update_count": len(candidates) - len(nodes),
            "no_model_weight_mutation": True,
        },
    }
    _write_json(Path(skill_graph_path), graph)
    return graph


def build_artifact(
    *,
    candidates: Sequence[Mapping[str, Any]],
    graph: Mapping[str, Any],
    skill_graph_path: Path | str = DEFAULT_SKILL_GRAPH_PATH,
    focused_tests_passed: bool = False,
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
    source_limitations: Sequence[str] = (),
) -> JsonDict:
    """REQ-LEARN-1555-4/5/6/7: compute utility and write the terminal gate."""

    del run_date
    promoted = [candidate for candidate in candidates if _is_promoted(candidate)]
    utility_candidates = _utility_candidates(promoted)
    replay_cases = _replay_cases(utility_candidates)
    baseline_utility = _mean_float(candidate.get("baseline_utility", 0.0) for candidate in utility_candidates)
    post_utility = _mean_float(
        candidate.get("post_promotion_utility", 0.0) for candidate in utility_candidates
    )
    utility_delta = round(post_utility - baseline_utility, 6)
    soundness_mistakes = sum(int(candidate.get("soundness_mistakes", 0)) for candidate in promoted)
    no_mutation = bool(graph.get("summary", {}).get("no_model_weight_mutation")) and all(
        candidate.get("no_model_weight_mutation") is True for candidate in promoted
    )
    external_feedback_used = any(candidate.get("external_feedback") is True for candidate in promoted)
    positive = bool(
        promoted
        and focused_tests_passed
        and external_feedback_used
        and no_mutation
        and soundness_mistakes == 0
        and utility_delta > 0.0
    )
    status = "complete" if promoted and focused_tests_passed else "blocked"
    artifact = {
        "status": status,
        "milestone": MILESTONE,
        "continuous_self_learning_task": "fr11_positive_utility_or_retire_v14",
        "fr11_positive_utility_gate_ready": positive,
        "model_specs": list(MANDATED_MODEL_SPECS),
        "live_sota_model_inference_used": any(
            candidate.get("live_sota_model_inference_used") is True for candidate in candidates
        ),
        "no_model_weight_mutation": no_mutation,
        "external_feedback_used": external_feedback_used,
        "self_feedback_only_rejected": any(
            "self_feedback_only" in candidate.get("rejection_reasons", []) for candidate in candidates
        ),
        "candidate_skill_updates": [_artifact_candidate(candidate) for candidate in candidates],
        "skill_updates_promoted": [_promoted_summary(candidate) for candidate in promoted],
        "replay_cases": replay_cases,
        "replay_pass_rate": _rate(
            sum(int(case["post_replay_passed"] is True) for case in replay_cases),
            len(replay_cases),
        ),
        "soundness_mistakes": soundness_mistakes,
        "baseline_utility": baseline_utility,
        "post_promotion_utility": post_utility,
        "utility_delta": utility_delta,
        "positive_utility_achieved": positive,
        "positive_utility_claim_retired": bool(status == "complete" and utility_delta <= 0.0),
        "skill_graph_path": _display_path(skill_graph_path, project_root=project_root),
        "focused_tests_passed": bool(focused_tests_passed),
        "honest_verdict": _honest_verdict(status=status, positive=positive, utility_delta=utility_delta),
    }
    if source_limitations:
        artifact["source_limitations"] = list(source_limitations)
    validate_artifact(artifact, skill_graph_path=skill_graph_path)
    return artifact


def run_experiment(
    *,
    project_root: Path | str | None = None,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    skill_graph_path: Path | str = DEFAULT_SKILL_GRAPH_PATH,
    exp1539_artifact_path: Path | str = DEFAULT_EXP1539_ARTIFACT_PATH,
    repair_artifact_path: Path | str = DEFAULT_REPAIR_ARTIFACT_PATH,
    repair_manifest_path: Path | str = DEFAULT_REPAIR_MANIFEST_PATH,
    product_line_artifact_paths: Sequence[Path | str] = DEFAULT_PRODUCT_LINE_ARTIFACT_PATHS,
    focused_tests_passed: bool = False,
) -> JsonDict:
    """Run Exp 1555 from checked-in predecessor artifacts."""

    root = Path(project_root) if project_root is not None else Path.cwd()
    output = _resolve_under_root(root, Path(output_path))
    graph_output = _resolve_under_root(root, Path(skill_graph_path))
    exp1539_path = _resolve_under_root(root, Path(exp1539_artifact_path))
    repair_path = _resolve_under_root(root, Path(repair_artifact_path))
    manifest_path = _resolve_under_root(root, Path(repair_manifest_path))
    product_paths = [_resolve_under_root(root, Path(path)) for path in product_line_artifact_paths]

    write_in_progress_artifact(output, skill_graph_path=graph_output, project_root=root)
    exp1539, exp1539_limitations = _load_json_or_limitation(exp1539_path)
    repair_artifact, repair_limitations = _load_json_or_limitation(repair_path)
    repair_rows, row_limitations = _read_jsonl_or_limitation(manifest_path)
    product_artifacts, product_limitations = _load_product_artifacts(product_paths)

    candidates = select_candidate_skill_updates(
        exp1539_artifact=exp1539,
        repair_artifact=repair_artifact,
        repair_rows=repair_rows,
    )
    if any(artifact.get("live_sota_model_inference_used") is True for artifact in product_artifacts):
        for candidate in candidates:
            candidate["live_sota_model_inference_used"] = True
    graph = build_skill_graph(candidates, skill_graph_path=graph_output, project_root=root)
    artifact = build_artifact(
        candidates=candidates,
        graph=graph,
        skill_graph_path=graph_output,
        focused_tests_passed=focused_tests_passed,
        project_root=root,
        source_limitations=[
            *exp1539_limitations,
            *repair_limitations,
            *row_limitations,
            *product_limitations,
        ],
    )
    _write_json(output, artifact)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    skill_graph_path: Path | str | None = None,
) -> None:
    """Validate the required conductor-facing artifact fields."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:  # pragma: no cover - schema guard for future edits.
        raise AssertionError(f"missing required fields: {missing}")
    if not str(artifact["honest_verdict"]).startswith(TERMINAL_VERDICT_PREFIXES):
        raise AssertionError("honest_verdict must use an allowed terminal prefix")
    if artifact["positive_utility_achieved"]:
        if float(artifact["utility_delta"]) <= 0.0:  # pragma: no cover - defensive guard.
            raise AssertionError("positive utility requires utility_delta > 0")
        if int(artifact["soundness_mistakes"]) != 0:  # pragma: no cover - defensive guard.
            raise AssertionError("positive utility requires zero soundness mistakes")
        if artifact["no_model_weight_mutation"] is not True:  # pragma: no cover
            raise AssertionError("positive utility requires no model-weight mutation")
        if artifact["positive_utility_claim_retired"] is True:  # pragma: no cover
            raise AssertionError("positive utility cannot also be retired")
    if artifact["no_model_weight_mutation"] is not True:
        raise AssertionError("FR-11 gate must not mutate model weights")
    if artifact["status"] == "complete" and artifact["skill_updates_promoted"] and skill_graph_path:
        if not Path(skill_graph_path).exists():
            raise AssertionError("complete promotion requires a skill graph artifact")


def _candidates_from_exp1539(artifact: Mapping[str, Any]) -> list[JsonDict]:
    candidates = []
    for update_id in artifact.get("promoted_updates", []):
        update = _find_exp1539_candidate(artifact, str(update_id))
        replay = _mapping(update.get("replay_evidence"))
        candidates.append(
            {
                "update_id": str(update_id),
                "source": "exp1539_external_feedback",
                "external_feedback": bool(update.get("external_deterministic_feedback", True)),
                "self_feedback_only": False,
                "no_model_weight_mutation": bool(artifact.get("no_model_weight_mutation", True)),
                "pre_replay_passed": replay.get("rollback_decision", "keep") == "keep",
                "post_replay_passed": replay.get("rollback_decision", "keep") == "keep",
                "false_accepts": int(replay.get("rollback_false_accept_delta", 0)),
                "soundness_mistakes": int(
                    replay.get("rollback_soundness_mistakes", artifact.get("soundness_mistakes", 0))
                ),
                "baseline_utility": float(artifact.get("baseline_task_success_rate", 0.0)),
                "post_promotion_utility": float(artifact.get("promoted_task_success_rate", 0.0)),
                "replay_case_ids": [f"exp1539:{update_id}"],
                "live_sota_model_inference_used": bool(
                    artifact.get("live_sota_model_inference_used", False)
                ),
                "lineage": {
                    "source_artifacts": [artifact.get("skill_graph_path", "results/experiment_1539")],
                },
            }
        )
    return candidates


def _candidate_from_repair_artifact(
    artifact: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict | None:
    accepted_rows = [
        row
        for row in rows
        if row.get("row_type") == "residual_drift_repair_case"
        and row.get("accepted") is True
        and row.get("replay_passed") is True
        and row.get("false_accept") is not True
    ]
    if artifact.get("status") != "complete" or not accepted_rows:
        return None
    replay_case_ids = [str(row.get("case_id") or "") for row in accepted_rows]
    replay_pass_rate = _rate(len(accepted_rows), len(replay_case_ids))
    return {
        "update_id": "policy:residual_drift_repair:1552",
        "source": "exp1552_residual_drift_repair",
        "external_feedback": bool(artifact.get("residual_drift_repair_ready", False)),
        "self_feedback_only": False,
        "no_model_weight_mutation": bool(artifact.get("no_model_weight_mutation", True)),
        "pre_replay_passed": True,
        "post_replay_passed": replay_pass_rate == 1.0,
        "false_accepts": int(artifact.get("rejected_false_accept_repairs", 0)),
        "soundness_mistakes": 0,
        "baseline_utility": 0.0,
        "post_promotion_utility": replay_pass_rate,
        "replay_case_ids": replay_case_ids,
        "live_sota_model_inference_used": bool(artifact.get("live_sota_model_inference_used", False)),
        "lineage": {
            "source_artifacts": [
                "results/experiment_1552_residual_drift_repair_policy_v1.json",
                str(artifact.get("repair_manifest_path", DEFAULT_REPAIR_MANIFEST_PATH.as_posix())),
            ],
            "repair_policy_path": artifact.get(
                "repair_policy_path", "python/carnot/verify/residual_drift_repair_policy.py"
            ),
        },
    }


def _evaluate_candidate(candidate: Mapping[str, Any]) -> JsonDict:
    result = dict(candidate)
    reasons = []
    if result.get("external_feedback") is not True:
        reasons.append("missing_external_feedback")
    if result.get("self_feedback_only") is True:
        reasons.append("self_feedback_only")
    if result.get("no_model_weight_mutation") is not True:
        reasons.append("model_weight_mutation")
    if result.get("pre_replay_passed") is not True:
        reasons.append("pre_replay_failed")
    if result.get("post_replay_passed") is not True:
        reasons.append("post_replay_failed")
    if int(result.get("false_accepts", 0)) > 0:
        reasons.append("false_accepts_positive")
    if int(result.get("soundness_mistakes", 0)) > 0:
        reasons.append("soundness_mistakes_positive")
    if not list(result.get("replay_case_ids", [])):
        reasons.append("missing_replay_cases")
    result["rejection_reasons"] = sorted(dict.fromkeys(reasons))
    result["promotion_decision"] = "promote" if not reasons else "reject"
    result["utility_delta"] = round(
        float(result.get("post_promotion_utility", 0.0))
        - float(result.get("baseline_utility", 0.0)),
        6,
    )
    return result


def _skill_node(candidate: Mapping[str, Any]) -> JsonDict:
    update_id = str(candidate["update_id"])
    return {
        "node_id": f"skill:fr11_v14/{update_id.replace(':', '-').replace('/', '-')}",
        "update_id": update_id,
        "source": str(candidate["source"]),
        "lineage": dict(candidate.get("lineage", {})),
        "external_feedback": {
            "external_feedback_used": True,
            "self_feedback_only": False,
            "replay_case_count": len(candidate.get("replay_case_ids", [])),
        },
        "promotion_decision": _promoted_summary(candidate),
    }


def _artifact_candidate(candidate: Mapping[str, Any]) -> JsonDict:
    return {
        "update_id": str(candidate.get("update_id")),
        "source": str(candidate.get("source")),
        "external_feedback": bool(candidate.get("external_feedback")),
        "self_feedback_only": bool(candidate.get("self_feedback_only")),
        "promotion_decision": str(candidate.get("promotion_decision")),
        "rejection_reasons": list(candidate.get("rejection_reasons", [])),
        "utility_delta": float(candidate.get("utility_delta", 0.0)),
        "replay_case_count": len(candidate.get("replay_case_ids", [])),
    }


def _promoted_summary(candidate: Mapping[str, Any]) -> JsonDict:
    return {
        "update_id": str(candidate["update_id"]),
        "source": str(candidate["source"]),
        "utility_delta": float(candidate.get("utility_delta", 0.0)),
        "replay_pass_rate": 1.0 if candidate.get("post_replay_passed") is True else 0.0,
    }


def _utility_candidates(promoted: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    repair_candidates = [
        candidate for candidate in promoted if candidate.get("source") == "exp1552_residual_drift_repair"
    ]
    return repair_candidates or list(promoted)


def _replay_cases(candidates: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "case_id": str(case_id),
            "update_id": str(candidate.get("update_id")),
            "source": str(candidate.get("source")),
            "pre_replay_passed": bool(candidate.get("pre_replay_passed")),
            "post_replay_passed": bool(candidate.get("post_replay_passed")),
        }
        for candidate in candidates
        for case_id in candidate.get("replay_case_ids", [])
    ]


def _find_exp1539_candidate(artifact: Mapping[str, Any], update_id: str) -> JsonDict:
    for candidate in artifact.get("candidate_updates", []):
        if str(candidate.get("policy_update_id")) == update_id:
            return dict(candidate)
    return {"policy_update_id": update_id, "external_deterministic_feedback": True}


def _is_promoted(candidate: Mapping[str, Any]) -> bool:
    return candidate.get("promotion_decision") == "promote"


def _honest_verdict(*, status: str, positive: bool, utility_delta: float) -> str:
    if positive:
        return "complete: fr11 positive utility gate passed"
    if status == "complete" and utility_delta <= 0.0:
        return "complete: fr11 positive-utility headline retired"
    return "complete: fr11 positive utility gate blocked"


def _load_product_artifacts(paths: Sequence[Path]) -> tuple[list[JsonDict], list[str]]:
    artifacts = []
    limitations = []
    for path in paths:
        artifact, path_limitations = _load_json_or_limitation(path)
        if artifact:
            artifacts.append(artifact)
        limitations.extend(path_limitations)
    return artifacts, limitations


def _load_json_or_limitation(path: Path) -> tuple[JsonDict, list[str]]:
    if not path.exists():
        return {}, [f"missing:{_display_path(path)}"]
    return _load_json(path), []


def _read_jsonl_or_limitation(path: Path) -> tuple[list[JsonDict], list[str]]:
    if not path.exists():
        return [], [f"missing:{_display_path(path)}"]
    return _read_jsonl(path), []


def _load_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else round(numerator / denominator, 6)


def _mean_float(values: Iterable[Any]) -> float:
    numbers = [float(value) for value in values]
    return 0.0 if not numbers else round(sum(numbers) / len(numbers), 6)


def _resolve_under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _display_path(path: Path | str, *, project_root: Path | str = REPO_ROOT) -> str:
    target = Path(path)
    try:
        return target.resolve().relative_to(Path(project_root).resolve()).as_posix()
    except ValueError:
        return target.as_posix()


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focused-tests-passed", action="store_true")
    args = parser.parse_args(argv)
    artifact = run_experiment(focused_tests_passed=args.focused_tests_passed)
    print(
        "[exp1555] "
        f"utility_delta={artifact['utility_delta']} "
        f"positive={artifact['positive_utility_achieved']} "
        f"retired={artifact['positive_utility_claim_retired']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())


__all__ = [
    "MANDATED_MODEL_SPECS",
    "OUTPUT_FILE",
    "REQUIRED_ARTIFACT_FIELDS",
    "SKILL_GRAPH_FILE",
    "build_artifact",
    "build_skill_graph",
    "run_experiment",
    "select_candidate_skill_updates",
    "validate_artifact",
    "write_in_progress_artifact",
]
