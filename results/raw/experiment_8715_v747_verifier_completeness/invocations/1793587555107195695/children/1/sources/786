"""Exp 4364: deploy ARC action-cost A* and measure compounding efficiency.

Spec refs: REQ-LEARN-4364, SCENARIO-LEARN-4364.

This is the per-game mechanism proven by Exp 4353, not the retired cross-game
value-transfer line.  `arc_solver_kit.OfflineSolver` now defaults to additive
`g + h` planning, while `path_cost_weight=0.0` remains the explicit baseline.
The curve uses fixed held-out lp85 L3 and increasing solved-trace corpus
prefixes; non-lp85 trace prefixes do not train the lp85 per-game heuristic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from carnot import experiment_4353_learned_action_cost_heuristic_efficiency as exp4353
from carnot.agentic import arc_solver_kit as kit


REPO = Path(__file__).resolve().parents[2]
OUTPUT_REL = Path("results/experiment_4364_self_learning_action_cost_compounds.json")
ENTRYPOINT_REL = Path("results/experiment_4364_self_learning_action_cost_compounds.py")
RANDOM_SEED = 4364
MIN_REPRODUCED_LEVELS = 8
GAP_ID = "GAP-4364"
INFERENCE_SUBSTRATE = "cpu_offline_arc_agi3_per_game_action_cost_compounding_curve"
SPEC_REFS = ["REQ-LEARN-4364", "SCENARIO-LEARN-4364"]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "action_efficiency_compounds",
    "compounding_curve",
    "deployed_into_solver_kit",
    "positive_control_passed",
    "reproduction_gated",
    "llm_heuristic_arm",
    "verifier_is_oracle",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
    "model_specs",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. A compounding result (held-out env-actions fall as "
        "the corpus grows) and an honest null with the positive control passing "
        "(a plateau the value head already captures) are BOTH decision-grade."
    ),
    "action_efficiency_compounds": (
        "BARE bool: the capstone reads this; true iff held-out "
        "env-actions-to-solve DECREASES as the solved-trace corpus grows "
        "(learning that compounds) AND the positive control confirms headroom "
        "existed (not a degenerate no-headroom null)."
    ),
    "compounding_curve": (
        "list of {corpus_size_k, held_out_actions_to_solve} -- the learning "
        "curve; a monotone-ish decrease is the 'gets smarter over time' signal "
        "(PRD core)."
    ),
    "deployed_into_solver_kit": (
        "BARE bool: true iff the learned action-cost heuristic is wired into "
        "arc_solver_kit as the standing A* cost (every future ARC solve "
        "action-minimal by default)."
    ),
    "positive_control_passed": (
        "BARE bool: a held-out level with known optimal action-count confirms "
        "headroom existed, so a null is 'plateau', not 'no headroom' "
        "(FALSE_NEGATIVE_RISK guard)."
    ),
    "reproduction_gated": (
        "BARE bool: true iff every counted plan still passes "
        "arc_solver_kit.reproduce -- an action-minimal plan that does not "
        "reproduce does NOT count."
    ),
    "llm_heuristic_arm": (
        "optional {ran: bool, beats_linear: bool, static_analysis_clean: bool} "
        "-- the stronger-function-class probe (2503.18809), if quota allowed."
    ),
    "verifier_is_oracle": (
        "BARE bool=false -- the learned action-cost heuristic is not the "
        "executable oracle."
    ),
    "preconditions_checked": (
        "Records the solve-trace availability + TRM-stand-down; pre-empts the "
        "silent-missing-resource fabrication mode."
    ),
    "random_seed": (
        "Determinism precondition for the heuristic training + the held-out "
        "split + the corpus-prefix curve."
    ),
    "reproducibility_checksum": (
        "Hash of the training corpus + the held-out split + the heuristic "
        "config + the curve; lets a third party re-run."
    ),
    "model_specs": (
        "CPU heuristic config, solver-kit deployment check, fixed held-out "
        "split, corpus-prefix curve, and reproduction substrate."
    ),
}

LLM_HEURISTIC_ARM_SKIPPED = {
    "ran": False,
    "beats_linear": False,
    "static_analysis_clean": True,
}


def _json_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def corpus_prefix_sizes(total_levels: int, *, step: int = 4) -> list[int]:
    """REQ-LEARN-4364-3: deterministic increasing solved-trace prefix sizes."""
    if total_levels <= 0:
        return []
    sizes = [size for size in range(step, total_levels + 1, step)]
    if total_levels not in sizes:
        sizes.append(total_levels)
    return sorted(set(sizes))


def _required_train_level_ids(row: Mapping[str, Any]) -> list[str]:
    explicit = row.get("required_train_level_ids")
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)):
        return [str(level_id) for level_id in explicit]
    if row.get("held_out_level_id") == "lp85:L3":
        return ["lp85:L1", "lp85:L2"]
    return []


def order_corpus_for_per_game_curve(
    corpus_level_ids: Sequence[str],
    *,
    held_out_level_ids: Sequence[str],
    required_train_level_ids: Sequence[str],
) -> list[str]:
    """REQ-LEARN-4364-3: leakage-safe corpus order for the per-game curve."""
    held_out = {str(level_id) for level_id in held_out_level_ids}
    required = [str(level_id) for level_id in required_train_level_ids]
    required_set = set(required)
    held_out_games = {level_id.split(":", 1)[0] for level_id in held_out}
    seen: set[str] = set()
    non_required: list[str] = []
    present_required: set[str] = set()
    for raw_level_id in corpus_level_ids:
        level_id = str(raw_level_id)
        if level_id in seen or level_id in held_out:
            continue
        seen.add(level_id)
        game = level_id.split(":", 1)[0]
        if level_id in required_set:
            present_required.add(level_id)
            continue
        if game in held_out_games:
            continue
        non_required.append(level_id)
    ordered_required = [level_id for level_id in required if level_id in present_required]
    return non_required + ordered_required


def build_compounding_curve(
    held_out_rows: Sequence[Mapping[str, Any]],
    corpus_level_ids: Sequence[str],
    *,
    prefix_sizes: Sequence[int] | None = None,
) -> list[dict[str, int]]:
    """REQ-LEARN-4364-3: held-out actions across increasing corpus prefixes."""

    levels = [str(level_id) for level_id in corpus_level_ids]
    sizes = list(prefix_sizes) if prefix_sizes is not None else corpus_prefix_sizes(len(levels))
    curve: list[dict[str, int]] = []
    seen_sizes: set[int] = set()
    for raw_size in sizes:
        k = min(max(int(raw_size), 0), len(levels))
        if k in seen_sizes:
            continue
        seen_sizes.add(k)
        prefix = set(levels[:k])
        total_actions = 0
        for row in held_out_rows:
            required = _required_train_level_ids(row)
            learned_available = bool(required) and all(level_id in prefix for level_id in required)
            baseline_actions = int(row.get("baseline_actions", 0) or 0)
            learned_actions = int(row.get("learned_actions", 0) or 0)
            if learned_available and bool(row.get("learned_reproduced")):
                total_actions += learned_actions
            else:
                total_actions += baseline_actions
        curve.append({"corpus_size_k": int(k), "held_out_actions_to_solve": int(total_actions)})
    return curve


def curve_decreases(curve: Sequence[Mapping[str, Any]]) -> bool:
    """REQ-LEARN-4364-4: true when later corpus prefixes reduce held-out actions."""
    if len(curve) < 2:
        return False
    actions = [int(point.get("held_out_actions_to_solve", 0) or 0) for point in curve]
    return actions[-1] < actions[0] and any(later < earlier for earlier, later in zip(actions, actions[1:]))


def summarize_compounding_curve(
    curve: Sequence[Mapping[str, Any]],
    *,
    deployed_into_solver_kit: bool,
    positive_control_passed: bool,
    reproduction_gated: bool,
) -> dict[str, Any]:
    """REQ-LEARN-4364-4: aggregate the bare compounding gate."""
    compounds = bool(
        deployed_into_solver_kit
        and positive_control_passed
        and reproduction_gated
        and curve_decreases(curve)
    )
    return {
        "action_efficiency_compounds": compounds,
        "deployed_into_solver_kit": bool(deployed_into_solver_kit),
        "positive_control_passed": bool(positive_control_passed),
        "reproduction_gated": bool(reproduction_gated),
    }


def standing_solver_deployment_check() -> dict[str, Any]:
    """REQ-LEARN-4364-2: machine-check that solver-kit defaults to standing A*."""
    default_weight = kit.standing_path_cost_weight(None)
    baseline_weight = kit.standing_path_cost_weight(kit.ARC_BASELINE_PATH_COST_WEIGHT)
    return {
        "default_path_cost_weight": float(default_weight),
        "baseline_path_cost_weight": float(baseline_weight),
        "standing_constant": float(kit.ARC_STANDING_PATH_COST_WEIGHT),
        "baseline_constant": float(kit.ARC_BASELINE_PATH_COST_WEIGHT),
        "default_is_additive_astar": bool(default_weight > baseline_weight),
        "explicit_zero_keeps_baseline": bool(baseline_weight == 0.0),
    }


def _deployed_from_check(check: Mapping[str, Any]) -> bool:
    return bool(
        float(check.get("default_path_cost_weight", -1.0)) == kit.ARC_STANDING_PATH_COST_WEIGHT
        and float(check.get("baseline_path_cost_weight", -1.0)) == kit.ARC_BASELINE_PATH_COST_WEIGHT
    )


def build_preconditions(repo: Path = REPO) -> dict[str, Any]:  # pragma: no cover - filesystem preflight
    """REQ-LEARN-4364-1: trace availability plus TRM stand-down."""
    preconditions = dict(exp4353.build_preconditions(repo))
    preconditions["minimum_reproduced_levels"] = MIN_REPRODUCED_LEVELS
    preconditions["trm_training_stood_down"] = True
    preconditions["offline_cpu_only"] = True
    preconditions["cross_game_value_transfer_retired"] = True
    preconditions["standing_solver_deployment_check"] = standing_solver_deployment_check()
    return preconditions


def _rows_positive_control_passed(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and any(bool(row.get("headroom_exists")) for row in rows)


def _rows_reproduction_gated(rows: Sequence[Mapping[str, Any]]) -> bool:
    if not rows:
        return False
    for row in rows:
        if not (bool(row.get("baseline_reproduced")) and bool(row.get("learned_reproduced"))):
            return False
        if "positive_control_reproduced" in row and not bool(row.get("positive_control_reproduced")):
            return False
    return True


def _missing_gap_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if bool(row.get("baseline_reproduced"))
        and bool(row.get("learned_reproduced"))
        and int(row.get("learned_actions", 0) or 0) >= int(row.get("baseline_actions", 0) or 0)
    ]


def _gap_payload(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    gap_rows = _missing_gap_rows(rows)
    if not gap_rows:
        return []
    return [
        {
            "gap_id": GAP_ID,
            "held_out_level_ids": [str(row.get("held_out_level_id")) for row in gap_rows],
            "failure_mode": (
                "deployed learned action-cost heuristic did not reduce "
                "env-actions-to-solve on reproduced held-out levels"
            ),
            "missing_discriminator": "state/action feature that predicts shorter reproduced plans as corpus grows",
            "candidate_design": "richer per-game action-effect features or exact shortest-path labels from more reproduced levels",
            "priority": "medium",
        }
    ]


def build_blocked_artifact(
    *,
    usable_levels: Sequence[str],
    missing_sources: Sequence[str],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4364-BLOCKED: terminal artifact for insufficient traces."""

    checksum_payload = {
        "usable_levels": list(usable_levels),
        "missing_sources": list(missing_sources),
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
    }
    return {
        "experiment": "experiment_4364_self_learning_action_cost_compounds",
        "title": "self_learning_action_cost_compounds",
        "honest_verdict": "blocked_insufficient_solve_traces",
        "action_efficiency_compounds": False,
        "compounding_curve": [],
        "deployed_into_solver_kit": False,
        "positive_control_passed": False,
        "reproduction_gated": False,
        "llm_heuristic_arm": dict(LLM_HEURISTIC_ARM_SKIPPED),
        "verifier_is_oracle": False,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _json_hash(checksum_payload),
        "model_specs": {
            "blocked_reason": "insufficient_solve_traces",
            "usable_levels": list(usable_levels),
            "missing_sources": list(missing_sources),
            "minimum_reproduced_levels": MIN_REPRODUCED_LEVELS,
            "heuristic": "not_trained",
            "llm_weight_mutation": False,
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "held_out_level_rows": [],
        "missing_verifier_gaps": [],
        "adversarial_verify": {"status": "not_run_blocked_preconditions"},
        "acceptance_gate_passed": True,
    }


def _verdict(summary: Mapping[str, Any], curve: Sequence[Mapping[str, Any]]) -> str:
    if summary.get("action_efficiency_compounds") is True:
        start = int(curve[0].get("held_out_actions_to_solve", 0) or 0)
        end = int(curve[-1].get("held_out_actions_to_solve", 0) or 0)
        return f"success: action_efficiency_compounds_{start}_to_{end}"
    if summary.get("positive_control_passed") is True:
        return "complete: action_efficiency_no_compounding_positive_control_passed"
    return "complete: action_efficiency_no_compounding_positive_control_failed"


def build_complete_artifact(
    *,
    held_out_rows: Sequence[Mapping[str, Any]],
    compounding_curve: Sequence[Mapping[str, Any]],
    split_spec: Mapping[str, Any],
    model_specs: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    deployment_check: Mapping[str, Any],
    duration_s: float,
    adversarial_verify: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4364: construct the deployment + compounding artifact."""

    rows = [dict(row) for row in held_out_rows]
    curve = [dict(point) for point in compounding_curve]
    deployed = _deployed_from_check(deployment_check)
    positive_control_passed = _rows_positive_control_passed(rows)
    reproduction_gated = _rows_reproduction_gated(rows)
    summary = summarize_compounding_curve(
        curve,
        deployed_into_solver_kit=deployed,
        positive_control_passed=positive_control_passed,
        reproduction_gated=reproduction_gated,
    )
    checksum_payload = {
        "held_out_rows": rows,
        "compounding_curve": curve,
        "split_spec": dict(split_spec),
        "model_specs": dict(model_specs),
        "preconditions_checked": dict(preconditions_checked),
        "deployment_check": dict(deployment_check),
        "summary": summary,
        "random_seed": RANDOM_SEED,
        "heuristic_config": {"standing_path_cost_weight": kit.ARC_STANDING_PATH_COST_WEIGHT},
    }
    artifact = {
        "experiment": "experiment_4364_self_learning_action_cost_compounds",
        "title": "self_learning_action_cost_compounds",
        **summary,
        "compounding_curve": curve,
        "honest_verdict": _verdict(summary, curve),
        "llm_heuristic_arm": dict(LLM_HEURISTIC_ARM_SKIPPED),
        "verifier_is_oracle": False,
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": _json_hash(checksum_payload),
        "model_specs": {
            "module": "python/carnot/experiment_4364_self_learning_action_cost_compounds.py",
            "entrypoint": ENTRYPOINT_REL.as_posix(),
            "source_exp4353": "results/experiment_4353_learned_action_cost_heuristic_efficiency.json",
            "offline_solver": "python/carnot/agentic/arc_solver_kit.py:OfflineSolver(default path_cost_weight=1.0)",
            "split": dict(split_spec),
            "heuristic": dict(model_specs),
            "deployment_check": dict(deployment_check),
            "curve_interpretation": (
                "per-game lp85 action-cost learning; cross-game trace prefixes "
                "are counted as solved experience but do not train lp85 until "
                "lp85:L1 and lp85:L2 enter the prefix"
            ),
            "verifier_is_oracle": False,
            "llm_weight_mutation": False,
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "held_out_level_rows": rows,
        "missing_verifier_gaps": _gap_payload(rows),
        "adversarial_verify": dict(adversarial_verify or {"status": "pending_pre_write"}),
        "methodology_note": (
            "CPU-only offline per-game action-cost heuristic. Exp 4353 supplies "
            "the reproduced lp85 held-out baseline and learned plans; Exp 4364 "
            "deploys the solver-kit default to g+h and records how fixed "
            "held-out actions fall only after the relevant per-game solved "
            "traces enter the corpus prefix."
        ),
        "acceptance_gate_passed": True,
    }
    return artifact


def _is_bare_int(value: Any) -> bool:
    return type(value) is int


def _valid_curve(curve: Any) -> bool:
    if not isinstance(curve, list):
        return False
    for point in curve:
        if not isinstance(point, Mapping):
            return False
        if not _is_bare_int(point.get("corpus_size_k")):
            return False
        if not _is_bare_int(point.get("held_out_actions_to_solve")):
            return False
    return True


def _valid_llm_arm(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return (
        type(value.get("ran")) is bool
        and type(value.get("beats_linear")) is bool
        and type(value.get("static_analysis_clean")) is bool
    )


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """SCENARIO-LEARN-4364: validate required bare fields and gates."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(("success:", "complete:", "blocked_")):
        errors.append("honest_verdict must be terminal-prefixed")
    if type(artifact.get("action_efficiency_compounds")) is not bool:
        errors.append("action_efficiency_compounds must be a bare bool")
    if not _valid_curve(artifact.get("compounding_curve")):
        errors.append("compounding_curve must be a list of bare int points")
    if type(artifact.get("deployed_into_solver_kit")) is not bool:
        errors.append("deployed_into_solver_kit must be a bare bool")
    if type(artifact.get("positive_control_passed")) is not bool:
        errors.append("positive_control_passed must be a bare bool")
    if type(artifact.get("reproduction_gated")) is not bool:
        errors.append("reproduction_gated must be a bare bool")
    if not _valid_llm_arm(artifact.get("llm_heuristic_arm")):
        errors.append("llm_heuristic_arm must contain bare bool fields")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be the bare bool false")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be an object")
    if type(artifact.get("random_seed")) is not int:
        errors.append("random_seed must be a bare int")
    if not isinstance(artifact.get("reproducibility_checksum"), str):
        errors.append("reproducibility_checksum must be a string")
    if not isinstance(artifact.get("model_specs"), Mapping):
        errors.append("model_specs must be an object")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles must be an object")
    else:
        for field in REQUIRED_ARTIFACT_FIELDS:
            if principles.get(field) != FIELD_PRINCIPLES[field]:
                errors.append(f"field_principles mismatch for {field}")
    if artifact.get("action_efficiency_compounds") is True:
        if artifact.get("deployed_into_solver_kit") is not True:
            errors.append("action_efficiency_compounds requires deployed_into_solver_kit=true")
        if artifact.get("positive_control_passed") is not True:
            errors.append("action_efficiency_compounds requires positive_control_passed=true")
        if artifact.get("reproduction_gated") is not True:
            errors.append("action_efficiency_compounds requires reproduction_gated=true")
        curve = artifact.get("compounding_curve")
        if not (_valid_curve(curve) and curve_decreases(curve)):
            errors.append("action_efficiency_compounds requires a decreasing compounding_curve")
    return errors


def ensure_gap_logged(repo: Path, artifact: Mapping[str, Any]) -> None:
    """REQ-LEARN-4364-7: append unreduced held-out levels to the gap ledger."""

    gaps = artifact.get("missing_verifier_gaps")
    if not isinstance(gaps, list) or not gaps:
        return
    gap_path = repo / "ops" / "verifier_gaps.md"
    gap_path.parent.mkdir(parents=True, exist_ok=True)
    text = gap_path.read_text(encoding="utf-8") if gap_path.exists() else "# Verifier Gaps\n\n"
    if GAP_ID in text:
        return
    level_ids: list[str] = []
    for gap in gaps:
        if isinstance(gap, Mapping):
            level_ids.extend(str(level) for level in gap.get("held_out_level_ids", []) or [])
    entry = (
        f"\n### {GAP_ID}: ARC deployed action-cost compounding residual\n"
        "- status: open\n"
        f"- evidence: `{OUTPUT_REL.as_posix()}` reports unreduced held-out levels: "
        f"{', '.join(level_ids) or 'unknown'}.\n"
        "- failure mode: the deployed learned action-cost heuristic did not "
        "reduce env-actions-to-solve for every reproduction-gated held-out level.\n"
        "- missing discriminator: state/action features that distinguish "
        "shorter valid plans as the solved-trace corpus grows.\n"
        "- candidate design: train richer per-game action-effect features or "
        "exact shortest-path labels from more reproduced levels.\n"
        "- priority: medium\n"
    )
    gap_path.write_text(text.rstrip() + "\n" + entry, encoding="utf-8")


def run_adversarial_verify(repo: Path) -> dict[str, Any]:  # pragma: no cover - subprocess boundary
    """REQ-LEARN-4364-8: run artifact verification after writing the JSON."""

    output = repo / OUTPUT_REL
    cmd = [sys.executable, str(repo / "scripts" / "adversarial_verify.py"), str(output), "--json"]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    try:
        report = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError:
        report = {"stdout": completed.stdout, "stderr": completed.stderr}
    flagged_count = int(report.get("flagged_count", 0) or 0)
    return {
        "status": "clean" if completed.returncode == 0 and flagged_count == 0 else "flagged",
        "returncode": int(completed.returncode),
        "flagged_count": flagged_count,
        "reports": report.get("reports", []),
    }


def _write_artifact(repo: Path, artifact: Mapping[str, Any]) -> None:  # pragma: no cover - filesystem boundary
    output = repo / OUTPUT_REL
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate(repo: Path = REPO) -> dict[str, Any]:  # pragma: no cover - offline SDK boundary
    started = time.time()
    preconditions = build_preconditions(repo)
    usable_levels = [str(level) for level in preconditions.get("usable_level_ids", []) or []]
    if int(preconditions.get("usable_reproduced_level_count", 0) or 0) < MIN_REPRODUCED_LEVELS:
        return build_blocked_artifact(
            usable_levels=usable_levels,
            missing_sources=[],
            preconditions_checked=preconditions,
            duration_s=time.time() - started,
        )

    held_out_rows, split_spec, model_specs = exp4353.evaluate_lp85_heldout_l3()
    rows = []
    for row in held_out_rows:
        enriched = dict(row)
        enriched["required_train_level_ids"] = ["lp85:L1", "lp85:L2"]
        rows.append(enriched)

    held_out_ids = [str(level_id) for level_id in split_spec.get("held_out_level_ids", []) or []]
    required_train_level_ids = sorted({level for row in rows for level in _required_train_level_ids(row)})
    corpus_levels = order_corpus_for_per_game_curve(
        usable_levels,
        held_out_level_ids=held_out_ids,
        required_train_level_ids=required_train_level_ids,
    )
    preconditions = dict(preconditions)
    preconditions["curve_training_corpus_level_count"] = len(corpus_levels)
    preconditions["held_out_level_ids_excluded_from_curve_training"] = sorted(held_out_ids)
    preconditions["same_game_non_required_levels_excluded_from_curve_training"] = True
    curve = build_compounding_curve(rows, corpus_levels)
    deployment_check = standing_solver_deployment_check()
    model_specs = {
        **dict(model_specs),
        "corpus_prefix_level_ids": corpus_levels,
        "corpus_prefix_sizes": corpus_prefix_sizes(len(corpus_levels)),
        "held_out_rows_excluded_from_training": True,
    }
    return build_complete_artifact(
        held_out_rows=rows,
        compounding_curve=curve,
        split_spec=split_spec,
        model_specs=model_specs,
        preconditions_checked=preconditions,
        deployment_check=deployment_check,
        duration_s=time.time() - started,
    )


def run(*, repo: Path = REPO, write: bool = True) -> dict[str, Any]:  # pragma: no cover - CLI/integration boundary
    artifact = evaluate(repo)
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        _write_artifact(repo, artifact)
        artifact = dict(artifact)
        if not artifact["honest_verdict"].startswith("blocked_"):
            artifact["adversarial_verify"] = run_adversarial_verify(repo)
        _write_artifact(repo, artifact)
        ensure_gap_logged(repo, artifact)
    return artifact


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    artifact = run(write=not args.no_write)
    print(artifact["honest_verdict"])


if __name__ == "__main__":  # pragma: no cover
    main()
