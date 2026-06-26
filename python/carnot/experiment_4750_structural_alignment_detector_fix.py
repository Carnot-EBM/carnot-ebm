"""Experiment 4750: structural-alignment detector segmentation fix.

Spec refs: REQ-ARC-WMTE-4712,
SCENARIO-ARC-WMTE-4712-STRUCTURAL-ALIGNMENT-GOAL,
SCENARIO-ARC-WMTE-4712-LIVE-REINDUCTION-WIRING.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))

EXPERIMENT = "experiment_4750_structural_alignment_detector_fix"
EXPERIMENT_ID = 4750
SCHEMA = "carnot.arc.structural_alignment_detector_fix_4750.v1"
RESULT_RELATIVE_PATH = "results/experiment_4750_structural_alignment_detector_fix.json"
TARGET_GAME = "lp85"
SPEC_REFS = [
    "REQ-ARC-WMTE-4712",
    "SCENARIO-ARC-WMTE-4712-STRUCTURAL-ALIGNMENT-GOAL",
    "SCENARIO-ARC-WMTE-4712-LIVE-REINDUCTION-WIRING",
]
DEFAULT_BUDGET = 3000

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; an L2 bank is success_, an honest detector-fixed-but-no-bank "
            "null is complete_."
        )
    },
    "inference_substrate": {
        "principle": "live_llm_inference (runs the live agent loop); 60s floor."
    },
    "preconditions_checked": {"principle": "records the arcade/value-learner import checks."},
    "detector_goal_count": {
        "principle": (
            "the over-segmentation metric (exp4712=42); FALSIFIABLE THRESHOLD: "
            "the fix must drive goal_count down to <= detector_piece_count."
        )
    },
    "detector_piece_count": {
        "principle": "the count of detected moveable pieces -- the explicit comparator."
    },
    "detector_aligned_piece_count": {
        "principle": (
            "must be able to reach detector_piece_count on the win frame -- direct evidence "
            "the goal is satisfiable."
        )
    },
    "goal_predicate_satisfiable": {
        "principle": "_goal_satisfiability_check finds the goal True on >=1 reachable grid."
    },
    "l2_plan_reaches_goal": {"principle": "plan_in_model reaches the goal."},
    "offline_reproduced": {
        "principle": "true only if arc_solver_kit.reproduce independently re-derives the L2 bank."
    },
    "solve_provenance": {
        "principle": "live_agent_self_discovery for an own-attempt advance; development_proxy for the offline twin."
    },
    "verifier_is_oracle": {
        "principle": "false -- the alignment predicate is over detected objects, not a learned-verifier moat."
    },
    "multi_exemplar_fallback": {
        "principle": (
            "when the detector-fixed live run still does not bank L2, replay at least two "
            "independent L1-completion frames and report whether the fixed structural "
            "candidate fits all of them without using the environment level-up oracle."
        )
    },
}


def _payload_checksum(payload: Mapping[str, Any]) -> str:
    clean = dict(payload)
    clean["reproducibility_checksum"] = ""
    encoded = json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _detector_metrics(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    piece_count = int(diagnostics.get("piece_count") or 0)
    goal_count = int(diagnostics.get("goal_count") or 0)
    aligned_count = int(diagnostics.get("aligned_piece_count") or 0)
    return {
        "detector_goal_count": goal_count,
        "detector_piece_count": piece_count,
        "detector_aligned_piece_count": aligned_count,
        "detector_raw_goal_count": int(diagnostics.get("raw_goal_count") or goal_count),
        "detector_pairing_gate": bool(piece_count > 0 and goal_count <= piece_count),
    }


def _diagnostics_from_reinduction(upstream: Mapping[str, Any]) -> dict[str, Any]:
    control = upstream.get("detector_positive_control")
    if isinstance(control, Mapping) and isinstance(control.get("diagnostics"), Mapping):
        return dict(control["diagnostics"])
    per_game = upstream.get("per_game")
    if isinstance(per_game, Mapping):
        lp85 = per_game.get("lp85")
        if isinstance(lp85, Mapping) and isinstance(
            lp85.get("structural_goal_diagnostics"), Mapping
        ):
            return dict(lp85["structural_goal_diagnostics"])
    return {}


def detector_positive_control_from_l1_trace(root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """REQ-ARC-WMTE-4712: run the lp85 post-L1 detector control on the real frame."""

    _ = Path(root)
    from carnot.agentic import arc_solver_kit as kit
    from carnot.experiment_4712_perception_grounded_l2_goal_lp85 import _detector_positive_control

    arc = kit.offline_arcade()
    control = dict(_detector_positive_control(arc))
    diagnostics = dict(control.get("diagnostics") or {})
    control.update(_detector_metrics(diagnostics))
    return control


def _labels_from_exp4664(root: Path) -> list[str]:
    path = root / "results" / "experiment_4664_l2_goal_predicate_induction_live.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = data["per_game"][TARGET_GAME]["solution_labels"]
    return [str(label) for label in labels]


def _labels_from_arc3_lp85_offline_resolve(root: Path) -> list[str]:
    path = root / "results" / "arc3_lp85_offline_resolve.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    labels: list[str] = []
    for move in data.get("solution") or []:
        if not isinstance(move, Mapping):
            continue
        labels.append(
            json.dumps(
                {
                    "action": int(move.get("action") or 6),
                    "data": {"x": int(move.get("x") or 0), "y": int(move.get("y") or 0)},
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return labels


def _replay_l1_exemplar(
    arc: Any,
    *,
    source: str,
    labels: list[str],
) -> dict[str, Any]:
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_competition_agent import _level_of
    from carnot.agentic.arc_executable_world_model import detect_cell, to_logical
    from carnot.agentic.arc_value_learner import structural_alignment_goal_candidate
    from carnot.experiment_4712_perception_grounded_l2_goal_lp85 import _apply_action_label, _gid

    env = arc.make(_gid(arc, TARGET_GAME), scorecard_id=arc.open_scorecard())
    frame = env.reset()
    start = int(_level_of(frame))
    for index, label in enumerate(labels, 1):
        frame = _apply_action_label(env, label, frame)
        if int(_level_of(frame)) <= start:
            continue
        grid = to_logical(grid_of(frame), detect_cell(grid_of(frame)))
        candidate = structural_alignment_goal_candidate(grid)
        diagnostics = dict(candidate.get("diagnostics") or {}) if candidate else {}
        metrics = _detector_metrics(diagnostics)
        fit = bool(candidate and metrics.get("detector_pairing_gate"))
        return {
            "source": source,
            "available": True,
            "l1_completion_reaches_level": int(_level_of(frame)),
            "l1_completion_steps": int(index),
            "structural_goal_detected": candidate is not None,
            "fit": fit,
            "diagnostics": diagnostics,
            **metrics,
        }
    return {
        "source": source,
        "available": False,
        "reason": "trace_did_not_reach_l1_completion",
        "fit": False,
    }


def multi_exemplar_fallback_from_l1_traces(root: Path | str = REPO_ROOT) -> dict[str, Any]:
    """SCENARIO-ARC-WMTE-4712: replay independent L1 completions through the fixed detector."""

    root_path = Path(root)
    from carnot.agentic import arc_solver_kit as kit

    arc = kit.offline_arcade()
    sources = [
        ("experiment_4664_l1_trace", _labels_from_exp4664),
        ("arc3_lp85_offline_resolve", _labels_from_arc3_lp85_offline_resolve),
    ]
    exemplars: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for source, loader in sources:
        try:
            labels = loader(root_path)
            exemplars.append(_replay_l1_exemplar(arc, source=source, labels=labels))
        except (KeyError, OSError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
            missing.append({"source": source, "reason": repr(exc)[:240]})
    fitted = [row for row in exemplars if bool(row.get("available"))]
    fit_all = bool(len(fitted) >= 2 and all(bool(row.get("fit")) for row in fitted))
    return {
        "available": bool(len(fitted) >= 2),
        "source_count": len(sources),
        "exemplar_count": len(fitted),
        "fit_all": fit_all,
        "exemplars": fitted,
        "missing_sources": missing,
    }


def _fallback_used(fallback: Mapping[str, Any]) -> bool:
    return bool(
        fallback
        and fallback.get("available")
        and int(fallback.get("exemplar_count") or 0) >= 2
        and fallback.get("fit_all")
    )


def _residual(upstream: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    if not bool(metrics.get("detector_pairing_gate")):
        return "detector_pairing_gate_failed"
    if not bool(upstream.get("goal_predicate_satisfiable")):
        return str(upstream.get("residual_cause_hypothesis") or "goal_predicate_not_satisfiable")
    if not bool(upstream.get("l2_plan_reaches_goal")):
        return "no_reachable_plan"
    if not bool(upstream.get("offline_reproduced")):
        return "offline_reproduction_missing"
    return "none"


def artifact_from_reinduction(
    upstream: Mapping[str, Any],
    *,
    duration_s: float | None = None,
    multi_exemplar_fallback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """REQ-ARC-WMTE-4712: add Exp4750 detector gates to a live reinduction result."""

    diagnostics = _diagnostics_from_reinduction(upstream)
    metrics = _detector_metrics(diagnostics)
    fallback = dict(multi_exemplar_fallback or upstream.get("multi_exemplar_fallback") or {})
    fallback_used = _fallback_used(fallback)
    blocked = str(upstream.get("honest_verdict") or "").startswith("blocked_")
    success = bool(
        upstream.get("goal_predicate_satisfiable")
        and upstream.get("l2_plan_reaches_goal")
        and upstream.get("offline_reproduced")
        and metrics.get("detector_pairing_gate")
    )
    residual = _residual(upstream, metrics)
    if blocked:
        verdict = str(upstream.get("honest_verdict"))
    elif success:
        verdict = "success_detector_fixed_l2_bank"
    elif fallback_used:
        verdict = f"complete_detector_fixed_multi_exemplar_but_no_bank_{residual}"
    else:
        verdict = f"complete_detector_fixed_but_no_bank_{residual}"
    artifact = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": "live_llm_inference",
        "preconditions_checked": dict(upstream.get("preconditions_checked") or {}),
        "goal_expression": str(upstream.get("goal_expression") or ""),
        "goal_predicate_satisfiable": bool(upstream.get("goal_predicate_satisfiable")),
        "l2_plan_reaches_goal": bool(upstream.get("l2_plan_reaches_goal")),
        "offline_reproduced": bool(upstream.get("offline_reproduced")),
        "reproduced_levels": int(upstream.get("reproduced_levels") or 0),
        "solve_provenance": str(upstream.get("solve_provenance") or "live_agent_self_discovery"),
        "verifier_is_oracle": False,
        "residual_cause_hypothesis": residual,
        "detector_positive_control": dict(upstream.get("detector_positive_control") or {}),
        "multi_exemplar_fallback": fallback,
        "multi_exemplar_fallback_used": fallback_used,
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(
            float(duration_s if duration_s is not None else upstream.get("duration_s") or 0.0), 6
        ),
        "upstream_experiment": str(upstream.get("experiment") or ""),
        "reproducibility_checksum": "",
        **metrics,
    }
    artifact["reproducibility_checksum"] = _payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    required = [
        "honest_verdict",
        "inference_substrate",
        "preconditions_checked",
        "detector_goal_count",
        "detector_piece_count",
        "detector_aligned_piece_count",
        "multi_exemplar_fallback",
        "multi_exemplar_fallback_used",
        "goal_predicate_satisfiable",
        "l2_plan_reaches_goal",
        "offline_reproduced",
        "solve_provenance",
        "verifier_is_oracle",
    ]
    missing = [field for field in required if field not in artifact]
    if missing:  # pragma: no cover - defensive schema guard
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact.get("honest_verdict") or "")
    if not (
        verdict.startswith("success_")
        or verdict.startswith("complete_")
        or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must use a terminal prefix")  # pragma: no cover
    if artifact.get("inference_substrate") != "live_llm_inference":
        raise ValueError("inference_substrate must be live_llm_inference")  # pragma: no cover
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle must be false")  # pragma: no cover
    if not verdict.startswith("blocked_") and int(artifact.get("detector_goal_count") or 0) > int(
        artifact.get("detector_piece_count") or 0
    ):
        raise ValueError("detector_goal_count must be <= detector_piece_count")  # pragma: no cover
    if artifact.get("multi_exemplar_fallback_used") and not bool(
        (artifact.get("multi_exemplar_fallback") or {}).get("fit_all")
    ):
        raise ValueError("used fallback must fit all exemplars")  # pragma: no cover


def build_artifact(
    root: Path | str = REPO_ROOT, *, budget: int = DEFAULT_BUDGET
) -> dict[str, Any]:  # pragma: no cover
    started = time.time()
    from carnot import experiment_4712_perception_grounded_l2_goal_lp85 as exp4712

    upstream = exp4712.build_artifact(Path(root), budget=int(budget), started_s=started)
    fallback = None
    if not bool(
        upstream.get("goal_predicate_satisfiable")
        and upstream.get("l2_plan_reaches_goal")
        and upstream.get("offline_reproduced")
    ):
        fallback = multi_exemplar_fallback_from_l1_traces(root)
    artifact = artifact_from_reinduction(
        upstream,
        duration_s=time.time() - started,
        multi_exemplar_fallback=fallback,
    )
    validate_artifact(artifact)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper
    artifact = build_artifact(
        REPO_ROOT,
        budget=int(os.environ.get("CARNOT_4750_BUDGET", DEFAULT_BUDGET)),
    )
    out = REPO_ROOT / RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: artifact[key]
                for key in (
                    "honest_verdict",
                    "detector_goal_count",
                    "detector_piece_count",
                    "detector_aligned_piece_count",
                    "goal_predicate_satisfiable",
                    "l2_plan_reaches_goal",
                    "offline_reproduced",
                )
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    raise SystemExit(main())
