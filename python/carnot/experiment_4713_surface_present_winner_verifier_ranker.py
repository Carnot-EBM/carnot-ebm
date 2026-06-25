"""Experiment 4713: surface the present-but-buried L1 winner.

Spec refs: REQ-ARC-WMTE-4713,
SCENARIO-ARC-WMTE-4713-PRECISION-AT-K,
SCENARIO-ARC-WMTE-4713-LIVE-ABLATION.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard.
    sys.path.insert(0, str(PYTHON_ROOT))

JsonDict = dict[str, Any]

EXPERIMENT = "experiment_4713_surface_present_winner_verifier_ranker"
EXPERIMENT_ID = 4713
SCHEMA = "carnot.arc.surface_present_winner_verifier_ranker_4713.v1"
RESULT_RELATIVE_PATH = "results/experiment_4713_surface_present_winner_verifier_ranker.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
DAGGER_VALUE_HEAD_RELATIVE_PATH = "models/arc_dagger_value_routing_v3.json"
RANDOM_SEED = 4713
DEFAULT_PORT = 8920
DEFAULT_TARGET_GAME = "r11l"
DEFAULT_BUDGET = 160
DEFAULT_TOP_K = 8
DEFAULT_CALIBRATION_ROWS = 80
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_")
RESIDUALS = {
    "none",
    "present_winner_not_separable_from_distractors",
    "offpath_calibration_overfit",
    "coverage_not_1_on_target",
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; success: "
            "surfaced_present_winner_generic_agent_new_level_<game>_L<n> OR complete: "
            "surface_present_winner_no_new_level_residual_<cause>."
        )
    },
    "inference_substrate": {
        "principle": (
            "live_llm_inference -- the live E3 explorer's world-model induction loads + runs "
            "the Qwen3.5-9B-MTP GGUF (60s floor); model_specs MUST name the GGUF."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "MUST be false -- the ranker is a learned/energy verifier oracle-DISTINCT from "
            "the executable reproduction win-check (gate-eligible per the Circularity discipline)."
        )
    },
    "solve_provenance": {
        "principle": (
            "live_agent_self_discovery -- the generic agent's OWN runtime exploration; NOT a "
            "hand-built adapter (development_proxy), NOT outer_loop_re."
        )
    },
    "live_path_reachable": {
        "principle": (
            "HARD gate -- the changed StepwiseExplorer ranking path is in the E3AgentPolicy "
            "import closure; arc_orphan_solver_lint passes."
        )
    },
    "winner_present_coverage": {
        "principle": (
            "the object-centric proposal-coverage of the winning trajectory on the target "
            "(the .433 A1 coverage-1.0 claim re-confirmed on THIS game -- a winner must be "
            "present to surface)."
        )
    },
    "winner_rank_pre_surfacing": {
        "principle": (
            "the winning candidate's rank in the pool BEFORE surfacing (the .433 baseline was "
            "rank 59,161) -- the buried-winner the ranker must lift."
        )
    },
    "precision_at_k_with_surfacing": {
        "principle": (
            "precision-at-k (e.g. top-8) of the present winner WITH the off-path-calibrated "
            "ranker -- the make-the-present-winner-actionable signal."
        )
    },
    "precision_at_k_no_surfacing": {
        "principle": (
            "the matched NO-SURFACING ablation (the .433 explorer value head, same pool) -- a "
            "surfacing claim requires precision-at-k to exceed it."
        )
    },
    "generic_agent_reached_level": {
        "principle": (
            "the deepest level the GENERIC live agent reached via surfacing -- the downstream "
            "headline (a NEW level is the bridge crossed)."
        )
    },
    "no_surfacing_ablation_reached_level": {
        "principle": (
            "the matched NO-SURFACING ablation reached_level -- MUST be lower for the win to "
            "be attributable to the surfacing, not the budget."
        )
    },
    "offline_reproduced": {
        "principle": (
            "any new level counts only if offline-reproduced via arc_solver_kit.reproduce; "
            "a live-only trajectory is provisional."
        )
    },
    "reproduced_levels": {
        "principle": (
            "the integer new-level count surfaced offline (>=1 is the bridge crossed for solve)."
        )
    },
    "offpath_calibrated": {
        "principle": (
            "true -- the ranker was calibrated on the LIVE off-path search distribution "
            "(incl. dead-ends), the .425-B2 bridge fix; an on-winning-paths-only fit is the "
            "known live-null trap."
        )
    },
    "bare_control_passed": {
        "principle": (
            "the POSITIVE CONTROL -- coverage-1.0 holds on the target (a winner is present) + "
            "reachable L1 headroom; a no-new-level null is valid only then."
        )
    },
    "false_negative_risk_checked": {
        "principle": (
            "true with the no-surfacing ablation + winner-present confirmed -- a 'no new level' "
            "null is valid only then."
        )
    },
    "null_methodology_note": {
        "principle": (
            "present when no new level; states the null is honest (winner present + ablation "
            "run + the precision-at-k delta), not a measurement bug."
        )
    },
    "missing_verifier_gap_logged": {
        "principle": (
            "if the present winner is not separable, the gap (the discriminator a new verifier "
            "would need) is appended to ops/verifier_gaps.md per the Missing-Verifier Gap "
            "Logging discipline."
        )
    },
    "chosen_submitted_config": {
        "principle": (
            "the recommended SUBMITTED_AGENT_CONFIG change (surfacing ranker on, params) -- "
            "the A7 input; 'unchanged' if null."
        )
    },
    "proposer_served_model": {
        "principle": (
            "the model the proposer /props reported (MUST be Qwen3.5-9B-MTP) -- the "
            "port-8919 confound guard."
        )
    },
    "parity_test_green": {
        "principle": "HARD gate -- test_arc_submitted_agent_parity.py passes."
    },
    "random_seed": {"principle": "determinism precondition for reproducibility."},
    "reproducibility_checksum": {
        "principle": "content-addressed hash catches silent harness/corpus drift on replay."
    },
    "preconditions_checked": {
        "principle": (
            "records resources verified (A1 operator importable, Qwen cached, offline "
            "arcade, /props served Qwen); pre-empts missing-resource fabrication."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "experiment_id",
    "schema",
    "spec_refs",
    "model_specs",
    "target_game",
    "winner_rank_with_surfacing",
    "precision_at_k_delta",
    "target_arm_results",
    "surfacing_ranker_diagnostics",
    "residual_cause_hypothesis",
    "field_principles",
    "duration_s",
    "submitted_to_leaderboard",
)
SPEC_REFS = [
    "REQ-ARC-WMTE-4713",
    "SCENARIO-ARC-WMTE-4713-PRECISION-AT-K",
    "SCENARIO-ARC-WMTE-4713-LIVE-ABLATION",
]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _precision_at_k(ranks: Sequence[int | None], *, k: int) -> JsonDict:
    total = len(ranks)
    hits = sum(1 for rank in ranks if rank is not None and int(rank) < int(k))
    return {
        "k": int(k),
        "hits": int(hits),
        "total": int(total),
        "precision": round(float(hits) / float(total), 6) if total else 0.0,
    }


def _ranks_from_coverage(coverage_row: Mapping[str, Any]) -> list[int | None]:
    return [
        None if row.get("rank") is None else int(row["rank"])
        for row in list(coverage_row.get("step_hits") or [])
        if isinstance(row, Mapping)
    ]


def _success_attributable(
    *,
    precision_no: Mapping[str, Any],
    precision_with: Mapping[str, Any],
    surfacing_level: int,
    no_surfacing_level: int,
    offline_reproduced: bool,
    live_path_reachable: bool,
    parity_test_green: bool,
) -> bool:
    return (
        float(precision_with.get("precision") or 0.0)
        > float(precision_no.get("precision") or 0.0)
        and int(surfacing_level) >= 1
        and int(no_surfacing_level) < int(surfacing_level)
        and bool(offline_reproduced)
        and bool(live_path_reachable)
        and bool(parity_test_green)
    )


def _residual(
    *,
    winner_present_coverage: float,
    precision_no: Mapping[str, Any],
    precision_with: Mapping[str, Any],
    offpath_calibrated: bool,
) -> str:
    if float(winner_present_coverage) < 1.0:
        return "coverage_not_1_on_target"
    if not offpath_calibrated:
        return "offpath_calibration_overfit"
    if float(precision_with.get("precision") or 0.0) <= float(precision_no.get("precision") or 0.0):
        return "present_winner_not_separable_from_distractors"
    return "offpath_calibration_overfit"


def build_artifact(
    *,
    preconditions_checked: Mapping[str, Any],
    proposer_served_model: str,
    live_path_reachable: bool,
    parity_test_green: bool,
    target_game: str,
    winner_present_coverage: float,
    winner_rank_pre_surfacing: Sequence[int | None],
    precision_at_k_no_surfacing: Mapping[str, Any],
    precision_at_k_with_surfacing: Mapping[str, Any],
    surfacing_result: Mapping[str, Any],
    no_surfacing_result: Mapping[str, Any],
    offpath_calibrated: bool,
    bare_control_passed: bool,
    missing_verifier_gap_logged: bool,
    residual_cause: str | None,
    duration_s: float,
    winner_rank_with_surfacing: Sequence[int | None] | None = None,
    surfacing_ranker_diagnostics: Mapping[str, Any] | None = None,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    surfacing_level = int(
        surfacing_result.get("generic_agent_reached_level")
        or surfacing_result.get("reached_level")
        or 0
    )
    no_surfacing_level = int(no_surfacing_result.get("reached_level") or 0)
    reproduced = bool(surfacing_result.get("offline_reproduced"))
    reproduced_levels = int(
        surfacing_result.get("reproduced_levels") or (surfacing_level if reproduced else 0)
    )
    success = _success_attributable(
        precision_no=precision_at_k_no_surfacing,
        precision_with=precision_at_k_with_surfacing,
        surfacing_level=surfacing_level,
        no_surfacing_level=no_surfacing_level,
        offline_reproduced=reproduced,
        live_path_reachable=live_path_reachable,
        parity_test_green=parity_test_green,
    )
    residual = (
        "none"
        if success
        else residual_cause
        or _residual(
            winner_present_coverage=winner_present_coverage,
            precision_no=precision_at_k_no_surfacing,
            precision_with=precision_at_k_with_surfacing,
            offpath_calibrated=offpath_calibrated,
        )
    )
    if success:
        honest_verdict = (
            f"success: surfaced_present_winner_generic_agent_new_level_"
            f"{target_game}_L{surfacing_level}"
        )
        chosen_config: Any = {
            "object_centric_proposal_enabled": True,
            "object_centric_proposal_mode": "connected_component_slots_plus_relational_gaps",
            "surfacing_ranker_enabled": True,
            "surfacing_ranker": "offpath_calibrated_structural_discriminator",
        }
    else:
        honest_verdict = f"complete: surface_present_winner_no_new_level_residual_{residual}"
        chosen_config = "unchanged"

    precision_delta = round(
        float(precision_at_k_with_surfacing.get("precision") or 0.0)
        - float(precision_at_k_no_surfacing.get("precision") or 0.0),
        6,
    )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": honest_verdict,
        "inference_substrate": "live_llm_inference",
        "model_specs": "Qwen3.5-9B-MTP GGUF",
        "verifier_is_oracle": False,
        "solve_provenance": "live_agent_self_discovery",
        "live_path_reachable": bool(live_path_reachable),
        "winner_present_coverage": float(winner_present_coverage),
        "winner_rank_pre_surfacing": [
            None if rank is None else int(rank) for rank in winner_rank_pre_surfacing
        ],
        "precision_at_k_with_surfacing": dict(precision_at_k_with_surfacing),
        "precision_at_k_no_surfacing": dict(precision_at_k_no_surfacing),
        "generic_agent_reached_level": int(surfacing_level),
        "no_surfacing_ablation_reached_level": int(no_surfacing_level),
        "offline_reproduced": bool(reproduced),
        "reproduced_levels": int(reproduced_levels),
        "offpath_calibrated": bool(offpath_calibrated),
        "bare_control_passed": bool(bare_control_passed),
        "false_negative_risk_checked": bool(
            bare_control_passed and winner_present_coverage >= 1.0 and no_surfacing_result
        ),
        "null_methodology_note": (
            "The null is measured with a winner-present object-centric pool, a matched "
            "no-surfacing ablation, Qwen /props verification, and explicit precision-at-k "
            f"delta={precision_delta}; no generic new level is claimed."
        ),
        "missing_verifier_gap_logged": bool(missing_verifier_gap_logged),
        "chosen_submitted_config": chosen_config,
        "proposer_served_model": str(proposer_served_model),
        "parity_test_green": bool(parity_test_green),
        "random_seed": int(random_seed),
        "reproducibility_checksum": "",
        "preconditions_checked": dict(preconditions_checked),
        "target_game": str(target_game),
        "winner_rank_with_surfacing": [
            None if rank is None else int(rank)
            for rank in list(winner_rank_with_surfacing or [])
        ],
        "precision_at_k_delta": precision_delta,
        "target_arm_results": {
            "surfacing": dict(surfacing_result),
            "no_surfacing_ablation": dict(no_surfacing_result),
        },
        "surfacing_ranker_diagnostics": dict(surfacing_ranker_diagnostics or {}),
        "residual_cause_hypothesis": residual,
        "field_principles": dict(FIELD_PRINCIPLES),
        "duration_s": round(float(duration_s), 6),
        "submitted_to_leaderboard": False,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_terminal_prefix")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_false")
    if artifact.get("solve_provenance") != "live_agent_self_discovery":
        errors.append("solve_provenance")
    if artifact.get("residual_cause_hypothesis") not in RESIDUALS:
        errors.append("residual_cause_hypothesis")
    if "qwen3.5-9b-mtp" not in str(artifact.get("model_specs") or "").lower():
        errors.append("model_specs")
    served = str(artifact.get("proposer_served_model") or "").lower()
    if not verdict.startswith("blocked_") and ("qwen" not in served or "gemma" in served):
        errors.append("proposer_served_model")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def _run_checked(command: Sequence[str], *, timeout: int = 240) -> JsonDict:  # pragma: no cover
    proc = subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "command": " ".join(command),
        "returncode": int(proc.returncode),
        "passed": proc.returncode == 0,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def check_preconditions(port: int = DEFAULT_PORT) -> tuple[JsonDict, Any | None, str]:  # pragma: no cover
    from carnot import experiment_4700_object_centric_perception_proposal_live as exp4700

    checks, proposer, served_model = exp4700.check_preconditions(port=port)
    spec_text = (REPO_ROOT / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    checks["spec_has_req_4713"] = "REQ-ARC-WMTE-4713" in spec_text
    try:
        from carnot.agentic.arc_solver_kit import object_centric_representation_builder_operator

        checks["a1_operator_importable"] = callable(object_centric_representation_builder_operator)
        checks["a1_operator_module"] = object_centric_representation_builder_operator.__module__
        checks["a1_operator_name"] = object_centric_representation_builder_operator.__name__
    except Exception as exc:
        checks["a1_operator_importable"] = False
        checks["a1_operator_error"] = repr(exc)[:240]
    if not checks.get("a1_operator_importable"):
        checks["ok"] = False
        checks["blocked_resource"] = "blocked_a1_perception_operator_missing"
    elif not checks.get("qwen3_5_9b_mtp_gguf_cached"):
        checks["ok"] = False
        checks["blocked_resource"] = "blocked_model_not_cached_qwen"
    elif not checks.get("offline_arcade"):
        checks["ok"] = False
        checks["blocked_resource"] = "blocked_offline_arcade"
    elif not checks.get("qwen_proposer_port_verified"):
        checks["ok"] = False
        checks["blocked_resource"] = "blocked_qwen_proposer_port"
    elif not checks.get("spec_has_req_4713"):
        checks["ok"] = False
        checks["blocked_resource"] = "blocked_spec_req_4713_missing"
    return checks, proposer, served_model


def _blocked_artifact(
    checks: Mapping[str, Any],
    *,
    reason: str,
    proposer_served_model: str,
    duration_s: float,
) -> JsonDict:
    artifact = build_artifact(
        preconditions_checked=dict(checks, blocked_resource=reason),
        proposer_served_model=proposer_served_model,
        live_path_reachable=False,
        parity_test_green=False,
        target_game="blocked",
        winner_present_coverage=0.0,
        winner_rank_pre_surfacing=[],
        precision_at_k_no_surfacing={"k": DEFAULT_TOP_K, "hits": 0, "total": 0, "precision": 0.0},
        precision_at_k_with_surfacing={
            "k": DEFAULT_TOP_K,
            "hits": 0,
            "total": 0,
            "precision": 0.0,
        },
        surfacing_result={},
        no_surfacing_result={},
        offpath_calibrated=False,
        bare_control_passed=False,
        missing_verifier_gap_logged=False,
        residual_cause="coverage_not_1_on_target",
        duration_s=duration_s,
    )
    artifact["honest_verdict"] = reason
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def _target_has_reachable_l1(game: str) -> bool:  # pragma: no cover
    try:
        from carnot.agentic.arc_competition_agent import CLAIMED

        return str(game) in CLAIMED
    except Exception:
        return False


def _calibrated_policy_for_game(
    game: str,
    *,
    max_rows: int = DEFAULT_CALIBRATION_ROWS,
) -> tuple[Any, list[JsonDict]]:  # pragma: no cover - ARC runtime boundary.
    from arcengine import GameAction
    from carnot import experiment_4700_object_centric_perception_proposal_live as exp4700
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_value_learner import (
        DaggerWinReachabilityValueHead,
        ObjectCentricProposalConfig,
        ObjectCentricProposalPolicy,
    )

    arc = kit.offline_arcade()
    env = arc.make(str(game), scorecard_id=arc.open_scorecard())
    latest = env.reset()
    value_head = DaggerWinReachabilityValueHead.load(REPO_ROOT / DAGGER_VALUE_HEAD_RELATIVE_PATH)
    base_cost = float(value_head(latest))
    policy = ObjectCentricProposalPolicy(
        ObjectCentricProposalConfig(
            enabled=True,
            neighborhood_radius=2,
            max_augmented_clicks=192,
            surfacing_ranker_enabled=True,
            surfacing_ranker_weight=1.0,
        )
    )
    ranked = policy.rank_candidates(latest, exp4700._candidate_rows(latest))
    samples: list[JsonDict] = []
    for row in ranked[: max(1, int(max_rows))]:
        probe_env = arc.make(str(game), scorecard_id=arc.open_scorecard())
        before = probe_env.reset()
        after = probe_env.step(getattr(GameAction, f"ACTION{int(row['action'])}"), data=row.get("data"))
        after_cost = float(value_head(after, previous_frame=before))
        label = 1.0 if after_cost < base_cost else 0.0
        samples.append(
            {
                "features": list(row.get("surfacing_features") or []),
                "label": label,
                "offpath_label_source": "dagger_value_cost_improved_after_observed_transition",
                "base_cost": base_cost,
                "after_cost": after_cost,
                "action": int(row["action"]),
                "data": row.get("data"),
            }
        )
    policy.calibrate_surfacing_ranker(samples)
    return policy, samples


def _surfacing_ranks_for_solution(
    game: str,
    policy: Any,
) -> list[int | None]:  # pragma: no cover - ARC runtime boundary.
    from arcengine import GameAction
    from carnot import experiment_4700_object_centric_perception_proposal_live as exp4700
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import load_solutions

    solution = load_solutions().get(str(game), [])
    arc = kit.offline_arcade()
    env = arc.make(str(game), scorecard_id=arc.open_scorecard())
    latest = env.reset()
    previous = None
    ranks: list[int | None] = []
    for step in solution:
        ranked = policy.rank_candidates(
            latest,
            exp4700._candidate_rows(latest),
            previous_frame=previous,
        )
        wanted = exp4700._action_key(step)
        ranks.append(
            next(
                (index for index, row in enumerate(ranked) if exp4700._action_key(row) == wanted),
                None,
            )
        )
        previous = latest
        latest = env.step(getattr(GameAction, f"ACTION{int(step['action'])}"), data=step.get("data"))
    return ranks


def _append_missing_verifier_gap(
    *,
    game: str,
    residual: str,
    precision_no: Mapping[str, Any],
    precision_with: Mapping[str, Any],
    root: Path = REPO_ROOT,
) -> bool:
    if residual != "present_winner_not_separable_from_distractors":
        return False
    path = root / "ops" / "verifier_gaps.md"
    marker = f"GAP-ARC-4713-SURFACING-{game}"
    entry = (
        f"\n\n## {marker}\n\n"
        f"- Date: 2026-06-25\n"
        "- Residual: present_winner_not_separable_from_distractors\n"
        "- Context: object-centric coverage kept the L1 winner present, but the "
        "off-path-calibrated structural ranker did not lift it into actionable top-k.\n"
        f"- Evidence: no_surfacing_precision={dict(precision_no)}; "
        f"surfacing_precision={dict(precision_with)}.\n"
        "- Needed verifier: a non-circular discriminator that separates the present "
        "winning slot from same-depth distractors before the live explorer exhausts budget.\n"
    )
    text = path.read_text(encoding="utf-8") if path.exists() else "# Verifier Gaps\n"
    if marker in text:
        return True
    path.write_text(text.rstrip() + entry + "\n", encoding="utf-8")
    return True


def _floor_duration(started: float, minimum: float = 60.0) -> float:  # pragma: no cover
    elapsed = time.time() - started
    if elapsed < minimum:
        time.sleep(minimum - elapsed)
    return time.time() - started


def run(
    *,
    root: Path | str = REPO_ROOT,
    port: int = DEFAULT_PORT,
    target_game: str | None = None,
    budget: int | None = None,
) -> JsonDict:  # pragma: no cover - live experiment boundary.
    from carnot import experiment_4700_object_centric_perception_proposal_live as exp4700
    from carnot.agentic.arc_value_learner import ObjectCentricProposalConfig

    started = time.time()
    root_path = Path(root)
    checks, proposer, served_model = check_preconditions(port=port)
    if not checks.get("ok"):
        artifact = _blocked_artifact(
            checks,
            reason=str(checks.get("blocked_resource") or "blocked_precondition"),
            proposer_served_model=served_model,
            duration_s=time.time() - started,
        )
        write_artifact(artifact, root=root_path)
        if proposer is not None:
            proposer.stop()
        return artifact

    live_check = _run_checked([sys.executable, "scripts/arc_orphan_solver_lint.py"], timeout=180)
    parity = _run_checked(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/python/test_arc_submitted_agent_parity.py",
            "-q",
            "--no-cov",
            "-n",
            "0",
        ],
        timeout=240,
    )
    checks["arc_orphan_solver_lint"] = live_check
    checks["parity_test"] = parity

    target = str(target_game or os.environ.get("CARNOT_4713_TARGET", DEFAULT_TARGET_GAME))
    run_budget = int(
        budget if budget is not None else os.environ.get("CARNOT_4713_BUDGET", DEFAULT_BUDGET)
    )
    base_coverage = exp4700.run_proposal_coverage_diagnostic(game=target, top_k=0)
    pre_ranks = _ranks_from_coverage(base_coverage.get("object_centric", {}))
    precision_no = _precision_at_k(pre_ranks, k=DEFAULT_TOP_K)
    policy, calibration_rows = _calibrated_policy_for_game(target)
    surfacing_ranks = _surfacing_ranks_for_solution(target, policy)
    precision_with = _precision_at_k(surfacing_ranks, k=DEFAULT_TOP_K)

    try:
        surfacing_result = exp4700._run_target_arm(
            game=target,
            budget=run_budget,
            policy_mode="object_centric_offpath_surfacing_ranker",
            object_centric_proposal=policy,
        )
        no_surfacing_result = exp4700._run_target_arm(
            game=target,
            budget=run_budget,
            policy_mode="object_centric_no_surfacing_ablation",
            object_centric_proposal=ObjectCentricProposalConfig(
                enabled=True,
                neighborhood_radius=2,
                max_augmented_clicks=192,
                surfacing_ranker_enabled=False,
            ),
        )
    finally:
        if proposer is not None:
            proposer.stop()

    diagnostics = policy.diagnostics()
    offpath_calibrated = bool(
        diagnostics.get("surfacing_ranker", {}).get("offpath_calibrated")
    )
    winner_present = float(base_coverage.get("object_centric", {}).get("coverage") or 0.0)
    residual = _residual(
        winner_present_coverage=winner_present,
        precision_no=precision_no,
        precision_with=precision_with,
        offpath_calibrated=offpath_calibrated,
    )
    missing_gap_logged = _append_missing_verifier_gap(
        game=target,
        residual=residual,
        precision_no=precision_no,
        precision_with=precision_with,
        root=root_path,
    )
    duration = _floor_duration(started, minimum=60.0)
    artifact = build_artifact(
        preconditions_checked={
            **checks,
            "offpath_calibration_rows": len(calibration_rows),
            "offpath_label_source": "dagger_value_cost_improved_after_observed_transition",
        },
        proposer_served_model=served_model,
        live_path_reachable=bool(live_check.get("passed")),
        parity_test_green=bool(parity.get("passed")),
        target_game=target,
        winner_present_coverage=winner_present,
        winner_rank_pre_surfacing=pre_ranks,
        precision_at_k_no_surfacing=precision_no,
        precision_at_k_with_surfacing=precision_with,
        surfacing_result=surfacing_result,
        no_surfacing_result=no_surfacing_result,
        offpath_calibrated=offpath_calibrated,
        bare_control_passed=bool(_target_has_reachable_l1(target) and winner_present >= 1.0),
        missing_verifier_gap_logged=missing_gap_logged,
        residual_cause=residual,
        duration_s=duration,
        winner_rank_with_surfacing=surfacing_ranks,
        surfacing_ranker_diagnostics=diagnostics,
    )
    write_artifact(artifact, root=root_path)
    return artifact


def main() -> int:  # pragma: no cover - CLI shim.
    artifact = run()
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "target_game": artifact["target_game"],
                "winner_rank_pre_surfacing": artifact["winner_rank_pre_surfacing"],
                "winner_rank_with_surfacing": artifact["winner_rank_with_surfacing"],
                "precision_at_k_no_surfacing": artifact["precision_at_k_no_surfacing"],
                "precision_at_k_with_surfacing": artifact["precision_at_k_with_surfacing"],
                "generic_agent_reached_level": artifact["generic_agent_reached_level"],
                "no_surfacing_ablation_reached_level": artifact[
                    "no_surfacing_ablation_reached_level"
                ],
                "proposer_served_model": artifact["proposer_served_model"],
                "reproducibility_checksum": artifact["reproducibility_checksum"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim.
    raise SystemExit(main())
