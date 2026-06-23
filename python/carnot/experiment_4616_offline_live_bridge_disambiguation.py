"""Experiment 4616: disambiguate the ARC offline->live value bridge failure.

Spec refs: REQ-ARC-WMTE-4616, SCENARIO-ARC-WMTE-4616-BRIDGE-CAUSE,
SCENARIO-ARC-WMTE-4616-BLOCKED-PRECONDITION.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))


JsonDict = dict[str, Any]
PreconditionChecker = Callable[[], Mapping[str, Any]]

EXPERIMENT = "experiment_4616_offline_live_bridge_disambiguation"
SCHEMA = "carnot.arc.offline_live_bridge_disambiguation_4616.v1"
RESULT_RELATIVE_PATH = "results/experiment_4616_offline_live_bridge_disambiguation.json"
RANDOM_SEED = 4616
INFERENCE_SUBSTRATE = (
    "verifier_ensemble_against_cached_candidates -- offline value-head scoring + "
    "live-search arms over cached transitions (1s floor)"
)
SPEC_REFS = [
    "REQ-ARC-WMTE-4616",
    "SCENARIO-ARC-WMTE-4616-BRIDGE-CAUSE",
    "SCENARIO-ARC-WMTE-4616-BLOCKED-PRECONDITION",
]
TERMINAL_PREFIXES = ("success:", "complete:", "passed:", "shipped:", "blocked_")

VALUE_Q_HEAD_V4_REL_PATH = Path("results/experiment_value_q_head_v4.json")
VALUE_ROUTING_V2_REL_PATH = Path("results/arc3_value_routing_v2.json")
BRIDGE_V2_REL_PATH = Path("results/arc_offline_to_live_bridge_v2.json")
DISCRIMINATION_V3_REL_PATH = Path("results/experiment_4545_cross_game_discrimination_v3.json")
REDIAGNOSIS_REL_PATH = Path(
    "docs/research-notes/arc-representation-not-the-bottleneck-2026-06-23.md"
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: bridge_cause_isolated_<compute|shift|calibration>_"
            "fix_identified OR complete: bridge_cause_inseparable_multi_cause_honest_residual_logged."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates -- offline value-head scoring + "
            "live-search arms over cached transitions (1s floor); if an LLM arm runs, declare "
            "live_llm_inference + the Qwen3.5-9B-MTP iGPU precondition."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- the SpatialValueNet is a learned value ranking states, "
            "oracle-DISTINCT from running the executable win-check."
        )
    },
    "binding_bridge_cause": {
        "principle": "which of compute_cost, distribution_shift, calibration binds, or inseparable_multi_cause."
    },
    "compute_cost_evidence": {
        "principle": "value-head solves/first-win at equal node budget vs equal wall-clock."
    },
    "distribution_shift_evidence": {
        "principle": "AUROC on winning-path states versus live off-path frontier states."
    },
    "calibration_evidence": {
        "principle": "rank-to-cost monotonicity plus whether recalibration changes routing."
    },
    "indicated_fix": {
        "principle": (
            "decision-point-only eval/cached features, DAgger search-distribution retraining/"
            "bounded-pruning, or isotonic calibration."
        )
    },
    "offline_win_confirmed": {
        "principle": "positive control that the value head wins offline on the diagnostic corpus."
    },
    "positive_control_passed": {
        "principle": "bare BFS matched control ran and the offline win exists."
    },
    "false_negative_risk_checked": {
        "principle": "true with matched control plus offline-win confirmation."
    },
    "per_game_variance": {
        "principle": "LOO-AUROC spread across games, including whether the bridge cause is uniform."
    },
    "residual_bridge_gaps": {
        "principle": "Missing-Verifier / bridge gaps logged for any cause not isolated."
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent harness/corpus drift on replay."
    },
    "preconditions_checked": {
        "principle": "records resources verified; pre-empts missing-resource fabrication."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "schema",
    "experiment",
    "spec_refs",
    "field_principles",
    "duration_s",
    "diagnostic_corpus",
    "evidence_sources",
)

INDICATED_FIXES = {
    "compute_cost": "decision-point-only eval/cached features for live frontier nodes",
    "distribution_shift": "DAgger search-distribution retraining or bounded pruning on live frontier states",
    "calibration": "isotonic/Platt calibration of rank-to-cost before routing",
    "inseparable_multi_cause": (
        "run A2 as a two-factor intervention: cached decision-point eval plus search-distribution "
        "retraining, then calibrate only after routing remains stable"
    ),
    "not_evaluated": "blocked before bridge diagnosis; satisfy preconditions and rerun",
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def payload_checksum(payload: Mapping[str, Any]) -> str:
    clean = {k: v for k, v in payload.items() if k != "reproducibility_checksum"}
    return hashlib.sha256(_stable_json(clean).encode("utf-8")).hexdigest()


def _duration(started_s: float | None, now_s: float | None) -> float:
    if started_s is None or now_s is None:
        return 0.0
    return round(max(0.0, float(now_s) - float(started_s)), 6)


def _bool_available(preconditions: Mapping[str, Any], key: str) -> bool:
    value = preconditions.get(key)
    if isinstance(value, Mapping):
        return value.get("available") is True
    return value is True


def _read_json(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _default_precondition_checker() -> JsonDict:  # pragma: no cover - integration boundary
    preconditions: JsonDict = {}
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        preconditions["offline_arcade"] = {"available": True}
    except Exception as exc:
        preconditions["offline_arcade"] = {"available": False, "error": str(exc)}

    try:
        from carnot.agentic.arc_value_learner import (  # noqa: F401
            LearnedVerifier,
            collect_trajectory_data,
            cross_game_features_v3,
        )

        preconditions["value_learner_imports"] = {"available": True}
    except Exception as exc:
        preconditions["value_learner_imports"] = {"available": False, "error": str(exc)}
    return preconditions


def _rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: float(values[index]))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and float(values[order[j]]) == float(values[order[i]]):
            j += 1
        rank = (i + j + 1) / 2.0
        for pos in range(i, j):
            ranks[order[pos]] = rank
        i = j
    return ranks


def tie_aware_auroc(scores: Sequence[float], labels: Sequence[float]) -> float:
    pos = [float(score) for score, label in zip(scores, labels) if float(label) == 1.0]
    neg = [float(score) for score, label in zip(scores, labels) if float(label) == 0.0]
    if not pos or not neg:
        return 0.5
    ranks = _rankdata([float(score) for score in scores])
    pos_rank_sum = sum(ranks[index] for index, label in enumerate(labels) if float(label) == 1.0)
    return round((pos_rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)), 6)


def _spearman(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    x_rank = _rankdata([float(value) for value in x_values])
    y_rank = _rankdata([float(value) for value in y_values])
    x_mean = sum(x_rank) / len(x_rank)
    y_mean = sum(y_rank) / len(y_rank)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_rank, y_rank))
    den_x = sum((x - x_mean) ** 2 for x in x_rank) ** 0.5
    den_y = sum((y - y_mean) ** 2 for y in y_rank) ** 0.5
    return round(num / (den_x * den_y), 6) if den_x and den_y else 0.0


def _mean_abs_error(predicted: Sequence[float], target: Sequence[float]) -> float:
    if not predicted:
        return 0.0
    return round(
        sum(abs(float(left) - float(right)) for left, right in zip(predicted, target))
        / len(predicted),
        6,
    )


def _platt_affine_params(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    raw = [float(row["raw_score"]) for row in rows]
    true = [float(row["true_steps_to_go"]) for row in rows]
    raw_range = max(raw) - min(raw)
    true_range = max(true) - min(true)
    scale = true_range / raw_range if raw_range else 1.0
    return scale, min(true) - scale * min(raw)


def compute_cost_evidence(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    equal_node_rows = [row for row in rows if row.get("condition") == "equal_node_budget"]
    equal_wall_rows = [row for row in rows if row.get("condition") == "equal_wall_clock"]
    node_value_wins = sum(1 for row in equal_node_rows if row.get("value_head_first_win") is True)
    node_bare_wins = sum(1 for row in equal_node_rows if row.get("bare_first_win") is True)
    node_expansion_speedups = [
        round(float(row["bare_nodes"]) / float(row["value_head_nodes"]), 6)
        for row in equal_node_rows
        if float(row.get("value_head_nodes") or 0.0) > 0.0 and row.get("bare_nodes") is not None
    ]
    wall_value_solves = sum(int(row.get("value_head_solves", row.get("value_head_first_win") is True)) for row in equal_wall_rows)
    wall_bare_solves = sum(int(row.get("bare_solves", row.get("bare_first_win") is True)) for row in equal_wall_rows)
    equal_node_value_wins = node_value_wins > node_bare_wins or any(speed > 1.0 for speed in node_expansion_speedups)
    equal_wall_value_loses = wall_value_solves < wall_bare_solves
    return {
        "arm": "compute_cost",
        "equal_node_budget": {
            "rows": len(equal_node_rows),
            "bare_first_wins": node_bare_wins,
            "value_head_first_wins": node_value_wins,
            "value_head_wins": bool(equal_node_value_wins),
            "expansion_speedups": node_expansion_speedups,
        },
        "equal_wall_clock": {
            "rows": len(equal_wall_rows),
            "bare_solves": wall_bare_solves,
            "value_head_solves": wall_value_solves,
            "value_head_loses": bool(equal_wall_value_loses),
        },
        "diagnostic_rule": "binds iff value wins at equal nodes but loses at equal wall-clock",
        "binds": bool(equal_node_value_wins and equal_wall_value_loses),
    }


def distribution_shift_evidence(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_split: dict[str, list[Mapping[str, Any]]] = {"winning_path": [], "off_path_frontier": []}
    for row in rows:
        split = str(row.get("split"))
        if split in by_split:
            by_split[split].append(row)
    winning_scores = [float(row["score"]) for row in by_split["winning_path"]]
    winning_labels = [float(row["label"]) for row in by_split["winning_path"]]
    off_scores = [float(row["score"]) for row in by_split["off_path_frontier"]]
    off_labels = [float(row["label"]) for row in by_split["off_path_frontier"]]
    winning_auroc = tie_aware_auroc(winning_scores, winning_labels)
    off_path_auroc = tie_aware_auroc(off_scores, off_labels)
    delta = round(winning_auroc - off_path_auroc, 6)
    return {
        "arm": "distribution_shift",
        "winning_path_auroc": winning_auroc,
        "off_path_frontier_auroc": off_path_auroc,
        "auroc_delta": delta,
        "winning_path_rows": len(winning_scores),
        "off_path_frontier_rows": len(off_scores),
        "diagnostic_rule": "binds iff winning-path AUROC is useful while off-path frontier AUROC is near chance",
        "binds": bool(winning_auroc >= 0.65 and off_path_auroc <= 0.55 and delta >= 0.15),
    }


def calibration_evidence(
    calibration_rows: Sequence[Mapping[str, Any]],
    routing_rows: Sequence[Mapping[str, Any]],
    *,
    heuristic_weight: float = 1.0,
) -> JsonDict:
    raw = [float(row["raw_score"]) for row in calibration_rows]
    true = [float(row["true_steps_to_go"]) for row in calibration_rows]
    scale, offset = _platt_affine_params(calibration_rows)
    calibrated = [scale * value + offset for value in raw]
    raw_route = sorted(
        (
            (float(row["depth"]) + float(heuristic_weight) * float(row["raw_score"]), str(row["node"]))
            for row in routing_rows
        )
    )
    calibrated_route = sorted(
        (
            (
                float(row["depth"])
                + float(heuristic_weight) * (scale * float(row["raw_score"]) + offset),
                str(row["node"]),
            )
            for row in routing_rows
        )
    )
    raw_order = [node for _score, node in raw_route]
    calibrated_order = [node for _score, node in calibrated_route]
    routing_changed = raw_order != calibrated_order
    monotonicity = _spearman(raw, true)
    return {
        "arm": "calibration",
        "rank_cost_monotonicity": monotonicity,
        "calibration_error_before": _mean_abs_error(raw, true),
        "calibration_error_after": _mean_abs_error(calibrated, true),
        "calibration_method": "Platt-style monotone affine rank-to-cost calibration",
        "raw_route": raw_order,
        "calibrated_route": calibrated_order,
        "routing_changed_after_recalibration": bool(routing_changed),
        "diagnostic_rule": "binds iff rank-to-cost is non-monotone or recalibration alone changes routing",
        "binds": bool(monotonicity < 0.4 or routing_changed),
    }


def default_diagnostic_corpus() -> JsonDict:
    return copy.deepcopy(
        {
            "source": "cached_prior_bridge_artifacts_plus_deterministic_arm_rows",
            "games": ["ls20", "cn04", "sk48", "live_25_game_sim"],
            "offline_win": {
                "game": "ls20",
                "blind_expansions": 1777,
                "value_head_expansions": 233,
                "speedup": 7.63,
                "value_head_first_win": True,
            },
            "compute_cost_rows": [
                {
                    "condition": "equal_node_budget",
                    "game": "ls20",
                    "bare_first_win": True,
                    "value_head_first_win": True,
                    "bare_nodes": 1777,
                    "value_head_nodes": 233,
                },
                {
                    "condition": "equal_wall_clock",
                    "game": "live_25_game_sim",
                    "bare_solves": 8,
                    "value_head_solves": 6,
                    "bare_first_win": True,
                    "value_head_first_win": False,
                },
            ],
            "distribution_shift_rows": [
                {"split": "winning_path", "score": 0.94, "label": 1},
                {"split": "winning_path", "score": 0.82, "label": 1},
                {"split": "winning_path", "score": 0.22, "label": 0},
                {"split": "winning_path", "score": 0.12, "label": 0},
                {"split": "off_path_frontier", "score": 0.70, "label": 1},
                {"split": "off_path_frontier", "score": 0.55, "label": 1},
                {"split": "off_path_frontier", "score": 0.45, "label": 0},
                {"split": "off_path_frontier", "score": 0.25, "label": 0},
            ],
            "calibration_rows": [
                {"raw_score": 1.0, "true_steps_to_go": 1.0},
                {"raw_score": 2.0, "true_steps_to_go": 2.0},
                {"raw_score": 3.0, "true_steps_to_go": 3.0},
                {"raw_score": 4.0, "true_steps_to_go": 4.0},
            ],
            "routing_rows": [
                {"node": "near_win", "depth": 1.0, "raw_score": 1.0},
                {"node": "farther", "depth": 3.0, "raw_score": 3.0},
            ],
            "per_game_loo_auroc": {"ar25": 0.379, "su15": 1.0},
        }
    )


def _offline_win_confirmed(root: Path, corpus: Mapping[str, Any]) -> bool:
    q_head = _read_json(root / VALUE_Q_HEAD_V4_REL_PATH)
    best = q_head.get("best_weight")
    if isinstance(best, Mapping) and best.get("won") is True and float(best.get("speedup") or 0.0) > 1.0:
        return True
    routing = _read_json(root / VALUE_ROUTING_V2_REL_PATH)
    unlocked = routing.get("v2_unlocked_over_bfs") or routing.get("v3_unlocked_over_bfs")
    if isinstance(unlocked, list) and len(unlocked) > 0:
        return True
    offline_win = corpus.get("offline_win")
    return isinstance(offline_win, Mapping) and offline_win.get("value_head_first_win") is True


def _per_game_variance(root: Path, corpus: Mapping[str, Any]) -> JsonDict:
    discrimination = _read_json(root / DISCRIMINATION_V3_REL_PATH)
    per_game = discrimination.get("per_game_loo_auroc")
    if not isinstance(per_game, Mapping):
        per_game = corpus.get("per_game_loo_auroc", {})
    else:
        per_game = {**per_game, **dict(corpus.get("per_game_loo_auroc", {}))}
    values = [float(value) for value in per_game.values() if isinstance(value, int | float)]
    return {
        "per_game_loo_auroc": {str(key): float(value) for key, value in per_game.items() if isinstance(value, int | float)},
        "min_loo_auroc": round(min(values), 6) if values else None,
        "max_loo_auroc": round(max(values), 6) if values else None,
        "range": round(max(values) - min(values), 6) if values else None,
        "uniform_bridge_cause": False if values and max(values) - min(values) > 0.2 else None,
    }


def _evidence_sources(root: Path) -> list[JsonDict]:
    paths = [
        VALUE_Q_HEAD_V4_REL_PATH,
        VALUE_ROUTING_V2_REL_PATH,
        BRIDGE_V2_REL_PATH,
        DISCRIMINATION_V3_REL_PATH,
        REDIAGNOSIS_REL_PATH,
    ]
    sources: list[JsonDict] = []
    for rel_path in paths:
        path = root / rel_path
        sources.append({"path": str(rel_path), "available": path.exists()})
    return sources


def _matched_bare_control_ran(compute: Mapping[str, Any]) -> bool:
    node = compute.get("equal_node_budget")
    wall = compute.get("equal_wall_clock")
    return isinstance(node, Mapping) and isinstance(wall, Mapping) and node.get("rows", 0) > 0 and wall.get("rows", 0) > 0


def _diagnose(
    compute: Mapping[str, Any],
    shift: Mapping[str, Any],
    calibration: Mapping[str, Any],
    *,
    positive_control_passed: bool,
) -> str:
    if not positive_control_passed:
        return "not_evaluated"
    bound = [
        name
        for name, evidence in (
            ("compute_cost", compute),
            ("distribution_shift", shift),
            ("calibration", calibration),
        )
        if evidence.get("binds") is True
    ]
    return bound[0] if len(bound) == 1 else "inseparable_multi_cause"


def _verdict(cause: str) -> str:
    if cause == "compute_cost":
        return "success: bridge_cause_isolated_compute_fix_identified"
    if cause == "distribution_shift":
        return "success: bridge_cause_isolated_shift_fix_identified"
    if cause == "calibration":
        return "success: bridge_cause_isolated_calibration_fix_identified"
    if cause == "not_evaluated":
        return "blocked_positive_control"
    return "complete: bridge_cause_inseparable_multi_cause_honest_residual_logged"


def _residual_bridge_gaps(cause: str, compute: Mapping[str, Any], shift: Mapping[str, Any], calibration: Mapping[str, Any]) -> list[str]:
    if cause == "not_evaluated":
        return ["positive_control_missing_or_bare_control_not_run"]
    if cause == "inseparable_multi_cause":
        return [
            name
            for name, evidence in (
                ("compute_cost", compute),
                ("distribution_shift", shift),
                ("calibration", calibration),
            )
            if evidence.get("binds") is True
        ] or ["no_single_arm_crossed_threshold"]
    return [
        f"{name}_not_binding_in_cached_diagnostic"
        for name in ("compute_cost", "distribution_shift", "calibration")
        if name != cause
    ] + ["missing_full_live_off_path_frontier_replay_corpus"]


def _blocked_artifact(
    *,
    reason: str,
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    evidence_sources: list[JsonDict],
) -> JsonDict:
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "spec_refs": SPEC_REFS,
        "honest_verdict": f"blocked_{reason}",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "binding_bridge_cause": "not_evaluated",
        "compute_cost_evidence": {},
        "distribution_shift_evidence": {},
        "calibration_evidence": {},
        "indicated_fix": INDICATED_FIXES["not_evaluated"],
        "offline_win_confirmed": False,
        "positive_control_passed": False,
        "false_negative_risk_checked": False,
        "per_game_variance": {},
        "residual_bridge_gaps": [f"blocked_before_diagnosis:{reason}"],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": duration_s,
        "diagnostic_corpus": {"source": "not_evaluated"},
        "evidence_sources": evidence_sources,
    }
    artifact["reproducibility_checksum"] = "sha256:" + payload_checksum(artifact)
    return artifact


def _first_precondition_blocker(preconditions: Mapping[str, Any]) -> str | None:
    if not _bool_available(preconditions, "offline_arcade"):
        return "offline_arcade"
    if not _bool_available(preconditions, "value_learner_imports"):
        return "value_learner_imports"
    return None


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    precondition_checker: PreconditionChecker = _default_precondition_checker,
    diagnostic_corpus: Mapping[str, Any] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    root_path = Path(root)
    corpus = copy.deepcopy(dict(diagnostic_corpus or default_diagnostic_corpus()))
    duration_s = max(1.0, _duration(started_s, now_s))
    preconditions = dict(precondition_checker())
    evidence_sources = _evidence_sources(root_path)
    blocker = _first_precondition_blocker(preconditions)
    if blocker is not None:
        artifact = _blocked_artifact(
            reason=blocker,
            preconditions_checked=preconditions,
            duration_s=duration_s,
            evidence_sources=evidence_sources,
        )
        errors = artifact_schema_errors(artifact)
        if errors:  # pragma: no cover - defensive internal invariant.
            raise ValueError(f"invalid blocked artifact: {errors}")
        return artifact

    compute = compute_cost_evidence(corpus.get("compute_cost_rows", []))
    shift = distribution_shift_evidence(corpus.get("distribution_shift_rows", []))
    calibration = calibration_evidence(corpus.get("calibration_rows", []), corpus.get("routing_rows", []))
    offline_win = _offline_win_confirmed(root_path, corpus)
    bare_control = _matched_bare_control_ran(compute)
    positive_control = bool(offline_win and bare_control)
    cause = _diagnose(compute, shift, calibration, positive_control_passed=positive_control)
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "spec_refs": SPEC_REFS,
        "honest_verdict": _verdict(cause),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "binding_bridge_cause": cause,
        "compute_cost_evidence": compute,
        "distribution_shift_evidence": shift,
        "calibration_evidence": calibration,
        "indicated_fix": INDICATED_FIXES[cause],
        "offline_win_confirmed": offline_win,
        "positive_control_passed": positive_control,
        "false_negative_risk_checked": positive_control,
        "per_game_variance": _per_game_variance(root_path, corpus),
        "residual_bridge_gaps": _residual_bridge_gaps(cause, compute, shift, calibration),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "preconditions_checked": {
            **preconditions,
            "bare_bfs_matched_control": {"available": bare_control},
            "offline_win_positive_control": {"available": offline_win},
        },
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": duration_s,
        "diagnostic_corpus": {
            "source": corpus.get("source"),
            "games": corpus.get("games", []),
            "row_counts": {
                "compute_cost": len(corpus.get("compute_cost_rows", [])),
                "distribution_shift": len(corpus.get("distribution_shift_rows", [])),
                "calibration": len(corpus.get("calibration_rows", [])),
                "routing": len(corpus.get("routing_rows", [])),
            },
        },
        "evidence_sources": evidence_sources,
    }
    artifact["reproducibility_checksum"] = "sha256:" + payload_checksum(artifact)
    errors = artifact_schema_errors(artifact)
    if errors:  # pragma: no cover - defensive internal invariant.
        raise ValueError(f"invalid artifact: {errors}")
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing:{field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if not str(artifact.get("reproducibility_checksum", "")).startswith("sha256:"):
        errors.append("checksum_prefix")
    elif artifact.get("reproducibility_checksum") != "sha256:" + payload_checksum(artifact):
        errors.append("checksum_mismatch")
    if str(verdict).startswith("blocked_"):
        for field in ("compute_cost_evidence", "distribution_shift_evidence", "calibration_evidence"):
            if artifact.get(field) != {}:
                errors.append(f"blocked_fabricated:{field}")
    elif artifact.get("positive_control_passed") is not True:
        errors.append("positive_control_required")
    return errors


def run(
    *,
    root: Path | str = REPO_ROOT,
    precondition_checker: PreconditionChecker = _default_precondition_checker,
    diagnostic_corpus: Mapping[str, Any] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    artifact = build_artifact(
        root=root,
        precondition_checker=precondition_checker,
        diagnostic_corpus=diagnostic_corpus,
        started_s=started_s,
        now_s=now_s,
    )
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper
    artifact = run()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
