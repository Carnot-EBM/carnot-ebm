"""Exp 4908: adversarial audit of Exp 4903 A1 and Exp 4904 A1b.

Spec refs: REQ-ARC-WMTE-4908,
SCENARIO-ARC-WMTE-4908-A1-AUDIT,
SCENARIO-ARC-WMTE-4908-A1B-LIVE-OR-GATE-SKIPPED,
SCENARIO-ARC-WMTE-4908-BLOCKED-UPSTREAM.
"""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
for import_root in (REPO_ROOT, PYTHON_ROOT):  # pragma: no cover - direct script guard.
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from carnot import experiment_4903_env_grounded_location_pruned_search as exp4903  # noqa: E402


EXPERIMENT = "experiment_4908_env_grounded_search_audit"
EXPERIMENT_ID = 4908
SCHEMA = "carnot.arc.env_grounded_search_audit_4908.v1"
A1_ARTIFACT_RELATIVE_PATH = "results/experiment_4903_env_grounded_location_pruned_search.json"
A1_SCRIPT_RELATIVE_PATH = "python/carnot/experiment_4903_env_grounded_location_pruned_search.py"
A1B_ARTIFACT_RELATIVE_PATH = "results/experiment_4904_latent_action_interface.json"
RESULT_RELATIVE_PATH = "results/experiment_4908_env_grounded_search_audit.json"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
DURATION_FLOOR_S = 0.0001
LIVE_LLM_DURATION_FLOOR_S = 60.0
RANDOM_SEED = 4908
TERMINAL_PREFIXES = ("complete_", "blocked_", "success_")

SPEC_REFS = [
    "REQ-ARC-WMTE-4908",
    "SCENARIO-ARC-WMTE-4908-A1-AUDIT",
    "SCENARIO-ARC-WMTE-4908-A1B-LIVE-OR-GATE-SKIPPED",
    "SCENARIO-ARC-WMTE-4908-BLOCKED-UPSTREAM",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {"principle": "terminal prefix; complete_a1_a1b_audited."},
    "a1_value_from_real_env": {
        "principle": (
            "true iff the search read change-VALUE from executed env transitions, not the "
            "model's prediction -- the whole .452 premise."
        )
    },
    "a1_planner_blind": {
        "principle": (
            "true iff the banked winning prefix was NOT injected into ranking/search/scoring "
            "(else the first-win is a tautology)."
        )
    },
    "a1_positive_control_non_degenerate": {
        "principle": (
            "true iff tu93's change-LOCATION prior is non-degenerate (ranks the truly-changing "
            "action highly)."
        )
    },
    "a1_numbers_match_fork": {
        "principle": (
            "true iff the per-game (first-win x action-cost x migration) numbers support the "
            "named fork verdict."
        )
    },
    "a1_trustworthy": {
        "principle": (
            "real-env AND planner-blind AND positive-control AND numbers-match -- the capstone "
            "trusts A1 only if true."
        )
    },
    "a1b_ran_genuinely_live": {
        "principle": (
            "true iff A1b duration_s>60 and genuinely ran (the .450 A1b 13.7s non-test fix); "
            "or a1b_gate_skipped=true."
        )
    },
    "a1b_gate_skipped": {
        "principle": (
            "true iff A1 lifted first-win >= 0.1 so the gate correctly skipped the last "
            "representation swing."
        )
    },
    "adversarial_flags_found": {
        "principle": "any adversarial_verify flag on the A1/A1b artifacts (fabrication gate)."
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (reads upstream artifacts + the A1 script; no LLM)."
    },
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "experiment_id",
    "schema",
    "spec_refs",
    "result_path",
    "a1_artifact_path",
    "a1_script_path",
    "a1b_artifact_path",
    "a1_artifact_checksum",
    "a1_script_checksum",
    "a1b_artifact_checksum",
    "a1_source_honest_verdict",
    "a1_source_fork_verdict",
    "a1b_source_honest_verdict",
    "a1b_source_fork_verdict",
    "a1_live_path_reachable",
    "claim_audit_table",
    "checks",
    "a1_failure_reasons",
    "a1b_failure_reasons",
    "a1_summarizer_result",
    "a1_adversarial_result",
    "a1b_summarizer_result",
    "a1b_adversarial_result",
    "live_lint_result",
    "preconditions_checked",
    "field_principles",
    "random_seed",
    "duration_s",
    "reproducibility_checksum",
)

JsonDict = dict[str, Any]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _read_json(path: Path) -> JsonDict:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return loaded


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def file_checksum(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _tail(text: str, limit: int = 2000) -> str:
    return text[-limit:] if len(text) > limit else text


def run_summarizer(path: Path) -> JsonDict:
    from scripts import summarize_artifact

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        returncode = summarize_artifact.summarize(path)
    return {"returncode": int(returncode), "stdout": buffer.getvalue(), "stderr": ""}


def run_adversarial_verify(path: Path) -> JsonDict:
    from scripts import adversarial_verify

    try:
        return dict(adversarial_verify.verify_artifact(path))
    except Exception as exc:  # pragma: no cover - defensive CLI fallback.
        proc = subprocess.run(
            [sys.executable, "scripts/adversarial_verify.py", str(path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        return {
            "loaded": True,
            "flag_count": 0 if proc.returncode == 0 else 1,
            "flags": [],
            "returncode": int(proc.returncode),
            "stdout_tail": _tail(proc.stdout),
            "stderr_tail": _tail(proc.stderr),
            "fallback_error": repr(exc),
        }


def run_arc_orphan_solver_lint(root: Path | str) -> JsonDict:
    proc = subprocess.run(
        [sys.executable, "scripts/arc_orphan_solver_lint.py"],
        cwd=Path(root),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    return {
        "command": f"{sys.executable} scripts/arc_orphan_solver_lint.py",
        "passed": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "stdout_tail": _tail(proc.stdout),
        "stderr_tail": _tail(proc.stderr),
    }


def _full_call_name(callable_node: ast.AST) -> str:
    if isinstance(callable_node, ast.Name):
        return callable_node.id
    if isinstance(callable_node, ast.Attribute):
        prefix = _full_call_name(callable_node.value)
        return f"{prefix}.{callable_node.attr}" if prefix else callable_node.attr
    return ""


def _call_name(callable_node: ast.AST | None) -> str:
    if callable_node is None:
        return ""
    if isinstance(callable_node, ast.Name):
        return callable_node.id
    if isinstance(callable_node, ast.Attribute):
        return callable_node.attr
    return ""


def _attach_ast_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.__dict__["_parent"] = parent


def _find_function(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _find_class_method(tree: ast.AST, class_name: str, method_name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    return None


def _first_parent_call(node: ast.AST) -> ast.Call | None:
    parent = getattr(node, "_parent", None)
    while parent is not None:
        if isinstance(parent, ast.Call):
            return parent
        parent = getattr(parent, "_parent", None)
    return None


def _source_value_checks(source_text: str) -> tuple[JsonDict, list[str]]:
    reasons: list[str] = []
    try:
        tree = ast.parse(source_text)
    except SyntaxError as exc:
        return {"passed": False, "parse_error": str(exc)}, ["a1_source_not_parseable"]

    search = _find_function(tree, "interleaved_env_grounded_search")
    score = _find_class_method(tree, "ChangeLocationActionPrior", "score")
    if search is None:
        reasons.append("interleaved_env_grounded_search_missing")
    if score is None:
        reasons.append("change_location_prior_score_missing")

    search_calls: list[str] = []
    direct_engine_calls: list[str] = []
    real_transition_called = False
    real_reads_incremented = False
    prior_rank_called = False
    if search is not None:
        for node in ast.walk(search):
            if isinstance(node, ast.Call):
                call = _full_call_name(node.func)
                search_calls.append(call)
                if call == "real_transition":
                    real_transition_called = True
                if call == "engine":
                    direct_engine_calls.append(call)
                if call.endswith(".rank"):
                    prior_rank_called = True
            if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                if node.target.id == "real_env_value_reads":
                    real_reads_incremented = True
    if not real_transition_called:
        reasons.append("a1_source_real_transition_not_called")
    if direct_engine_calls:
        reasons.append("a1_source_engine_called_for_next_value")
    if not real_reads_incremented:
        reasons.append("a1_source_real_env_reads_not_counted")
    if not prior_rank_called:
        reasons.append("a1_source_prior_rank_not_called")

    score_calls: list[str] = []
    if score is not None:
        score_calls = [
            _full_call_name(node.func) for node in ast.walk(score) if isinstance(node, ast.Call)
        ]
    if not any(call.endswith("count_nonzero") or call == "count_nonzero" for call in score_calls):
        reasons.append("a1_source_prior_not_location_only")

    return (
        {
            "passed": not reasons,
            "real_transition_called": real_transition_called,
            "real_env_reads_incremented": real_reads_incremented,
            "prior_rank_called": prior_rank_called,
            "direct_engine_calls_in_search": direct_engine_calls,
            "score_calls": score_calls,
            "search_calls": search_calls,
        },
        list(dict.fromkeys(reasons)),
    )


def _source_planner_blind_checks(source_text: str) -> tuple[JsonDict, list[str]]:
    reasons: list[str] = []
    try:
        tree = ast.parse(source_text)
    except SyntaxError as exc:
        return {"passed": False, "parse_error": str(exc)}, ["a1_source_not_parseable"]

    _attach_ast_parents(tree)
    function = _find_function(tree, "measure_game_with_env_grounded_search")
    if function is None:
        return (
            {"passed": False, "function_present": False, "disallowed_refs": []},
            ["measure_game_with_env_grounded_search_missing"],
        )

    refs: list[JsonDict] = []
    disallowed: list[JsonDict] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and node.id == "winning_prefix":
            call = _first_parent_call(node)
            call_name = _call_name(call.func if call is not None else None)
            record = {"line": int(getattr(node, "lineno", 0)), "call": call_name or None}
            refs.append(record)
            if call_name != "_classify_after_search":
                disallowed.append(record)

    if not refs:
        reasons.append("winning_prefix_not_used_for_classification")
    if disallowed:
        reasons.append("banked_prefix_used_before_classification")
    return (
        {
            "passed": not reasons,
            "function_present": True,
            "winning_prefix_refs": refs,
            "disallowed_refs": disallowed,
        },
        list(dict.fromkeys(reasons)),
    )


def _row_map(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(artifact.get("per_game_first_win"))


def _a1_value_from_real_env_check(
    artifact: Mapping[str, Any], source_text: str
) -> tuple[JsonDict, list[str]]:
    source_check, source_reasons = _source_value_checks(source_text)
    reasons = list(source_reasons)
    rows = _row_map(artifact)
    config = _mapping(artifact.get("env_grounded_search_config"))
    bad_prediction_games: list[str] = []
    missing_read_games: list[str] = []
    if artifact.get("change_location_prior_used_not_value") is not True:
        reasons.append("a1_artifact_location_prior_flag_false")
    if config.get("change_location_prior_only") is not True:
        reasons.append("a1_config_not_location_prior_only")
    if config.get("env_supplies_change_value") is not True:
        reasons.append("a1_config_env_supplies_value_missing")
    if not rows:
        reasons.append("a1_per_game_first_win_missing")
    for game, row_value in rows.items():
        row = _mapping(row_value)
        if row.get("change_value_predictions_used") not in (0, None):
            bad_prediction_games.append(str(game))
        if int(row.get("real_env_value_reads") or 0) <= 0:
            missing_read_games.append(str(game))
    if bad_prediction_games:
        reasons.append("a1_model_change_value_predictions_used")
    if missing_read_games:
        reasons.append("a1_real_env_value_reads_missing")
    control = _mapping(artifact.get("positive_control_result"))
    if control and control.get("change_value_predictions_used") not in (0, None):
        reasons.append("positive_control_model_change_value_predictions_used")
    if control and int(control.get("real_env_value_reads") or 0) <= 0:
        reasons.append("positive_control_real_env_value_reads_missing")
    return (
        {
            "passed": not reasons,
            "evidence": (
                f"{len(rows)}/{len(rows)} A1 rows have change_value_predictions_used=0 and "
                f"real_env_value_reads>0; source real_transition_called="
                f"{source_check.get('real_transition_called')}"
            )
            if not reasons
            else "model-predicted change values or missing real env reads detected",
            "source": source_check,
            "bad_prediction_games": bad_prediction_games,
            "missing_read_games": missing_read_games,
            "config": dict(config),
        },
        list(dict.fromkeys(reasons)),
    )


def _a1_planner_blind_check(
    artifact: Mapping[str, Any], source_text: str
) -> tuple[JsonDict, list[str]]:
    source_check, source_reasons = _source_planner_blind_checks(source_text)
    reasons = list(source_reasons)
    config = _mapping(artifact.get("env_grounded_search_config"))
    preconditions = _mapping(artifact.get("preconditions_checked"))
    if artifact.get("planner_blind_to_banked_answer") is not True:
        reasons.append("a1_planner_blind_artifact_flag_false")
    if config.get("planner_blind_to_banked_answer") is not True:
        reasons.append("a1_planner_blind_config_flag_false")
    if preconditions.get("planner_blind_to_banked_answer") is not True:
        reasons.append("a1_planner_blind_precondition_flag_false")
    return (
        {
            "passed": not reasons,
            "evidence": (
                "winning_prefix appears only in _classify_after_search after search"
                if not reasons
                else "banked prefix entered ranking/search/scoring or planner-blind flags disagree"
            ),
            "source": source_check,
            "artifact_flag": artifact.get("planner_blind_to_banked_answer"),
            "config_flag": config.get("planner_blind_to_banked_answer"),
            "precondition_flag": preconditions.get("planner_blind_to_banked_answer"),
        },
        list(dict.fromkeys(reasons)),
    )


def _a1_positive_control_check(artifact: Mapping[str, Any]) -> tuple[JsonDict, list[str]]:
    row = _mapping(artifact.get("positive_control_result"))
    rank = _finite_float(row.get("true_changing_action_rank"))
    threshold = _finite_float(row.get("non_degenerate_rank_threshold")) or 5.0
    reasons: list[str] = []
    if artifact.get("positive_control_game") != "tu93" or row.get("game") != "tu93":
        reasons.append("tu93_positive_control_missing")
    if artifact.get("positive_control_non_degenerate") is not True:
        reasons.append("tu93_change_location_prior_degenerate")
    if row.get("location_ranker_non_degenerate") is not True:
        reasons.append("tu93_change_location_prior_degenerate")
    if rank is None or rank > threshold:
        reasons.append("tu93_true_changing_action_not_ranked_highly")
    if int(row.get("actual_changing_actions_seen") or 0) <= 0:
        reasons.append("tu93_no_actual_changing_actions_seen")
    if row.get("change_value_predictions_used") not in (0, None):
        reasons.append("tu93_model_change_value_predictions_used")
    if int(row.get("real_env_value_reads") or 0) <= 0:
        reasons.append("tu93_real_env_reads_missing")
    return (
        {
            "passed": not reasons,
            "evidence": (
                f"tu93 true_changing_action_rank={row.get('true_changing_action_rank')} "
                f"<= {row.get('non_degenerate_rank_threshold')}"
                if not reasons
                else "tu93 positive-control ranker is degenerate or unaudited"
            ),
            "positive_control_game": artifact.get("positive_control_game"),
            "row": dict(row),
        },
        list(dict.fromkeys(reasons)),
    )


def _bootstrap_iterations(artifact: Mapping[str, Any]) -> int:
    try:
        return int(_mapping(artifact.get("env_grounded_search_config")).get("bootstrap_iterations"))
    except (TypeError, ValueError):
        return exp4903.DEFAULT_BOOTSTRAP_ITERATIONS


def _bounded_action_cost(artifact: Mapping[str, Any]) -> int:
    try:
        return int(_mapping(artifact.get("env_grounded_search_config")).get("bounded_action_cost"))
    except (TypeError, ValueError):
        return exp4903.DEFAULT_BOUNDED_ACTION_COST


def _a1_numbers_match_fork_check(artifact: Mapping[str, Any]) -> tuple[JsonDict, list[str]]:
    rows = {str(game): _mapping(row) for game, row in _row_map(artifact).items()}
    reasons: list[str] = []
    bad_games: list[str] = []
    for game, row in rows.items():
        baseline = exp4903._unit(row.get("first_win_baseline"))
        env_grounded = exp4903._unit(row.get("first_win_env_grounded"))
        delta = exp4903._row_delta(row)
        if baseline is None or env_grounded is None or delta is None:
            reasons.append("invalid_first_win_metric")
            bad_games.append(game)
        elif round(env_grounded - baseline, 6) != round(delta, 6):
            reasons.append("first_win_delta_mismatch")
            bad_games.append(game)
        if row.get("bucket") not in exp4903.BUCKETS:
            reasons.append("invalid_bucket")
            bad_games.append(game)
        if not isinstance(row.get("migrated"), bool):
            reasons.append("invalid_migrated_flag")
            bad_games.append(game)
        try:
            if int(row.get("states_expanded")) < 0:
                reasons.append("invalid_states_expanded")
                bad_games.append(game)
        except (TypeError, ValueError):
            reasons.append("invalid_states_expanded")
            bad_games.append(game)
        actions = row.get("actions_to_first_win")
        if actions is not None:
            try:
                if int(actions) < 0:
                    reasons.append("invalid_actions_to_first_win")
                    bad_games.append(game)
            except (TypeError, ValueError):
                reasons.append("invalid_actions_to_first_win")
                bad_games.append(game)
    if len(rows) < 3:
        reasons.append("n_games_measured_below_3")
    try:
        reported_n = int(artifact.get("n_games_measured"))
    except (TypeError, ValueError):
        reported_n = -1
        reasons.append("n_games_measured_not_integer")
    if reported_n != len(rows):
        reasons.append("n_games_measured_mismatch")

    seed = int(artifact.get("random_seed") or exp4903.RANDOM_SEED)
    iterations = _bootstrap_iterations(artifact)
    bounded_cost = _bounded_action_cost(artifact)
    expected_median = exp4903._median_delta(rows)
    expected_ci95 = exp4903.bootstrap_ci95(
        exp4903._delta_values(rows), iterations=iterations, seed=seed
    )
    expected_actions = exp4903._median_actions_to_first_win(rows)
    expected_migration = exp4903._coverage_migration_count(rows, bounded_action_cost=bounded_cost)
    expected_fork = exp4903.compute_fork_verdict(
        rows,
        positive_control_row=_mapping(artifact.get("positive_control_result")),
        ci95=expected_ci95,
        bounded_action_cost=bounded_cost,
    )
    if artifact.get("value_grounded_first_win_delta_median") != expected_median:
        reasons.append("first_win_delta_median_mismatch")
    if artifact.get("value_grounded_first_win_delta_ci95") != expected_ci95:
        reasons.append("first_win_delta_ci95_mismatch")
    if artifact.get("median_actions_to_first_win") != expected_actions:
        reasons.append("median_actions_to_first_win_mismatch")
    if artifact.get("coverage_migration_count") != expected_migration:
        reasons.append("coverage_migration_count_mismatch")
    if artifact.get("fork_verdict") != expected_fork:
        reasons.append("fork_verdict_mismatch")
    return (
        {
            "passed": not reasons,
            "evidence": (
                f"median_delta={expected_median}, ci95={expected_ci95}, "
                f"coverage_migration_count={expected_migration}, "
                f"median_actions_to_first_win={expected_actions}, fork={expected_fork}"
                if not reasons
                else "reported per-game first-win/action-cost/migration numbers do not recompute the fork"
            ),
            "reported_fork_verdict": artifact.get("fork_verdict"),
            "computed_fork_verdict": expected_fork,
            "reported_median_delta": artifact.get("value_grounded_first_win_delta_median"),
            "computed_median_delta": expected_median,
            "reported_ci95": artifact.get("value_grounded_first_win_delta_ci95"),
            "computed_ci95": expected_ci95,
            "reported_coverage_migration_count": artifact.get("coverage_migration_count"),
            "computed_coverage_migration_count": expected_migration,
            "reported_median_actions_to_first_win": artifact.get("median_actions_to_first_win"),
            "computed_median_actions_to_first_win": expected_actions,
            "bad_games": sorted(set(bad_games)),
        },
        list(dict.fromkeys(reasons)),
    )


def _a1_live_path_check(
    artifact: Mapping[str, Any], live_lint_result: Mapping[str, Any]
) -> tuple[JsonDict, list[str]]:
    rows = {str(game): _mapping(row) for game, row in _row_map(artifact).items()}
    required = {
        "StepwiseExplorer.action_prior",
        "arc_executable_world_model.load_engine",
        "arc_executable_world_model.plan_in_model",
    }
    missing_methods = [
        game
        for game, row in rows.items()
        if not required.issubset({str(item) for item in row.get("live_path_methods_called") or []})
    ]
    reasons: list[str] = []
    if artifact.get("live_path_reachable") is not True:
        reasons.append("a1_artifact_live_path_false")
    if live_lint_result.get("passed") is not True:
        reasons.append("arc_orphan_solver_lint_failed")
    if missing_methods:
        reasons.append("a1_live_path_methods_missing")
    if artifact.get("verifier_is_oracle") is True:
        reasons.append("a1_verifier_is_oracle")
    if artifact.get("solve_provenance") != "development_proxy":
        reasons.append("a1_solve_provenance_not_development_proxy")
    return (
        {
            "passed": not reasons,
            "evidence": (
                "arc_orphan_solver_lint passed and all A1 rows cite StepwiseExplorer.action_prior, "
                "load_engine, and plan_in_model"
                if not reasons
                else "A1 live path is not reachable or row methods are missing"
            ),
            "artifact_live_path_reachable": artifact.get("live_path_reachable"),
            "arc_orphan_solver_lint_passed": live_lint_result.get("passed"),
            "missing_method_games": missing_methods,
            "verifier_is_oracle": artifact.get("verifier_is_oracle"),
            "solve_provenance": artifact.get("solve_provenance"),
        },
        list(dict.fromkeys(reasons)),
    )


def _flag_kinds(adversarial_result: Mapping[str, Any] | None) -> list[str]:
    flags = _mapping(adversarial_result).get("flags", [])
    return [str(flag.get("kind")) for flag in flags if isinstance(flag, Mapping)]


def _has_flags(adversarial_result: Mapping[str, Any] | None) -> bool:
    result = _mapping(adversarial_result)
    return int(result.get("flag_count") or 0) > 0 or bool(_flag_kinds(result))


def _a1b_gate_skipped(a1_artifact: Mapping[str, Any]) -> bool:
    delta = _finite_float(a1_artifact.get("value_grounded_first_win_delta_median"))
    return delta is not None and delta >= 0.1


def _heldout_games_from_a1(a1_artifact: Mapping[str, Any]) -> list[str]:
    config_games = _mapping(a1_artifact.get("env_grounded_search_config")).get("heldout_games")
    if isinstance(config_games, Sequence) and not isinstance(config_games, (str, bytes)):
        return [str(game) for game in config_games]
    return sorted(str(game) for game in _row_map(a1_artifact))


def _heldout_games_from_a1b(a1b_artifact: Mapping[str, Any]) -> list[str]:
    config_games = _mapping(a1b_artifact.get("latent_action_config")).get("heldout_games")
    if isinstance(config_games, Sequence) and not isinstance(config_games, (str, bytes)):
        return [str(game) for game in config_games]
    return sorted(str(game) for game in _mapping(a1b_artifact.get("per_game_value_gap")))


def _a1b_live_check(
    *,
    a1_artifact: Mapping[str, Any],
    a1b_artifact: Mapping[str, Any] | None,
    a1b_summarizer_result: Mapping[str, Any] | None,
    a1b_adversarial_result: Mapping[str, Any] | None,
) -> tuple[JsonDict, list[str], bool, bool]:
    gate_skipped = _a1b_gate_skipped(a1_artifact)
    if a1b_artifact is None:
        if gate_skipped:
            return (
                {
                    "passed": True,
                    "evidence": "A1 value_grounded_first_win_delta_median >= 0.1, so A1b gate skipped",
                    "status": "gate_skipped",
                    "a1_delta": a1_artifact.get("value_grounded_first_win_delta_median"),
                },
                [],
                True,
                True,
            )
        return (
            {
                "passed": False,
                "evidence": "A1b artifact missing while A1 first-win delta is below 0.1",
                "status": "missing",
                "a1_delta": a1_artifact.get("value_grounded_first_win_delta_median"),
            },
            ["a1b_missing_after_low_first_win_a1"],
            False,
            False,
        )

    a1_games = _heldout_games_from_a1(a1_artifact)
    a1b_games = _heldout_games_from_a1b(a1b_artifact)
    a1b_rows = _mapping(a1b_artifact.get("per_game_value_gap"))
    same_split = set(a1_games) == set(a1b_games) == set(str(game) for game in a1b_rows)
    duration = _finite_float(a1b_artifact.get("duration_s"))
    flag_kinds = _flag_kinds(a1b_adversarial_result)
    missing_live_methods = [
        str(game)
        for game, row_value in a1b_rows.items()
        if "arc_executable_world_model.load_engine"
        not in {str(item) for item in _mapping(row_value).get("live_path_methods_called") or []}
    ]
    reasons: list[str] = []
    if not same_split or a1b_artifact.get("delta_on_truly_heldout_split") is not True:
        reasons.append("a1b_not_same_heldout_split_as_a1")
    if duration is None or duration <= LIVE_LLM_DURATION_FLOOR_S:
        reasons.append("a1b_duration_below_live_floor")
    if a1b_artifact.get("ran_genuinely_live") is not True:
        reasons.append("a1b_ran_genuinely_live_flag_false")
    if "DURATION_TOO_SHORT" in flag_kinds:
        reasons.append("a1b_duration_too_short_flagged")
    if a1b_artifact.get("inference_substrate") != "live_llm_inference":
        reasons.append("a1b_inference_substrate_not_live_llm")
    if a1b_artifact.get("verifier_is_oracle") is True:
        reasons.append("a1b_verifier_is_oracle")
    if "CIRCULAR_MOAT_OVERCLAIM" in flag_kinds:
        reasons.append("a1b_circular_moat_overclaim")
    if a1b_artifact.get("live_path_reachable") is not True or missing_live_methods:
        reasons.append("a1b_live_path_unreachable")
    if a1b_artifact.get("solve_provenance") != "development_proxy":
        reasons.append("a1b_solve_provenance_not_development_proxy")
    if _mapping(a1b_summarizer_result).get("returncode") not in (0, None):
        reasons.append("a1b_summarizer_failed")
    return (
        {
            "passed": not reasons,
            "evidence": (
                f"A1b duration_s={a1b_artifact.get('duration_s')} > 60 and heldout split matches A1"
                if not reasons
                else "A1b did not prove a genuine live same-split oracle-distinct run"
            ),
            "status": "ran",
            "a1_games": sorted(a1_games),
            "a1b_games": sorted(a1b_games),
            "same_heldout_split": same_split,
            "duration_s": a1b_artifact.get("duration_s"),
            "duration_floor_s": LIVE_LLM_DURATION_FLOOR_S,
            "ran_genuinely_live": a1b_artifact.get("ran_genuinely_live"),
            "duration_too_short_flagged": "DURATION_TOO_SHORT" in flag_kinds,
            "oracle_distinct": a1b_artifact.get("verifier_is_oracle") is not True
            and "CIRCULAR_MOAT_OVERCLAIM" not in flag_kinds,
            "missing_live_method_games": missing_live_methods,
            "adversarial_flag_kinds": flag_kinds,
        },
        list(dict.fromkeys(reasons)),
        not reasons,
        gate_skipped,
    )


def _claim(name: str, passed: bool, evidence: str) -> JsonDict:
    return {"claim": name, "passed": bool(passed), "evidence": str(evidence)}


def audit_sources(
    *,
    a1_artifact: Mapping[str, Any],
    a1_source_text: str,
    a1_summarizer_result: Mapping[str, Any],
    a1_adversarial_result: Mapping[str, Any],
    a1b_artifact: Mapping[str, Any] | None,
    a1b_summarizer_result: Mapping[str, Any] | None,
    a1b_adversarial_result: Mapping[str, Any] | None,
    live_lint_result: Mapping[str, Any],
) -> JsonDict:
    value_check, value_reasons = _a1_value_from_real_env_check(a1_artifact, a1_source_text)
    blind_check, blind_reasons = _a1_planner_blind_check(a1_artifact, a1_source_text)
    positive_check, positive_reasons = _a1_positive_control_check(a1_artifact)
    numbers_check, number_reasons = _a1_numbers_match_fork_check(a1_artifact)
    live_check, live_reasons = _a1_live_path_check(a1_artifact, live_lint_result)
    a1b_check, a1b_reasons, a1b_live_or_skipped, a1b_gate_skipped = _a1b_live_check(
        a1_artifact=a1_artifact,
        a1b_artifact=a1b_artifact,
        a1b_summarizer_result=a1b_summarizer_result,
        a1b_adversarial_result=a1b_adversarial_result,
    )
    a1_tool_reasons: list[str] = []
    if a1_summarizer_result.get("returncode") not in (0, None):
        a1_tool_reasons.append("a1_summarizer_failed")
    if _has_flags(a1_adversarial_result):
        a1_tool_reasons.append("a1_adversarial_verify_flagged")

    a1_value = not value_reasons
    a1_blind = not blind_reasons
    a1_positive = not positive_reasons
    a1_numbers = not number_reasons
    a1_live = not live_reasons
    a1_trustworthy = a1_value and a1_blind and a1_positive and a1_numbers
    adversarial_flags = _has_flags(a1_adversarial_result) or _has_flags(a1b_adversarial_result)
    claim_table = [
        _claim("A1 real-env value grounding", a1_value, str(value_check.get("evidence"))),
        _claim("A1 planner-blind search", a1_blind, str(blind_check.get("evidence"))),
        _claim("A1 tu93 positive control", a1_positive, str(positive_check.get("evidence"))),
        _claim("A1 numbers match fork", a1_numbers, str(numbers_check.get("evidence"))),
        _claim("A1 live-path reachable", a1_live, str(live_check.get("evidence"))),
        _claim(
            "A1b genuinely live or gate-skipped",
            a1b_live_or_skipped,
            str(a1b_check.get("evidence")),
        ),
    ]
    return {
        "honest_verdict": "complete_a1_a1b_audited",
        "a1_value_from_real_env": a1_value,
        "a1_planner_blind": a1_blind,
        "a1_positive_control_non_degenerate": a1_positive,
        "a1_numbers_match_fork": a1_numbers,
        "a1_live_path_reachable": a1_live,
        "a1_trustworthy": a1_trustworthy,
        "a1b_ran_genuinely_live": a1b_live_or_skipped,
        "a1b_gate_skipped": a1b_gate_skipped,
        "adversarial_flags_found": adversarial_flags,
        "claim_audit_table": claim_table,
        "a1_failure_reasons": list(
            dict.fromkeys(
                value_reasons
                + blind_reasons
                + positive_reasons
                + number_reasons
                + live_reasons
                + a1_tool_reasons
            )
        ),
        "a1b_failure_reasons": a1b_reasons,
        "checks": {
            "a1_value_from_real_env": value_check,
            "a1_planner_blind": blind_check,
            "a1_positive_control_non_degenerate": positive_check,
            "a1_numbers_match_fork": numbers_check,
            "a1_live_path_reachable": live_check,
            "a1b_live_or_gate_skipped": a1b_check,
            "a1_summarizer": dict(a1_summarizer_result),
            "a1_adversarial_verify": dict(a1_adversarial_result),
            "a1b_summarizer": dict(a1b_summarizer_result or {}),
            "a1b_adversarial_verify": dict(a1b_adversarial_result or {}),
        },
    }


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    repo = Path(root)
    spec_text = (
        (repo / SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
        if (repo / SPEC_RELATIVE_PATH).exists()
        else ""
    )
    a1_present = (repo / A1_ARTIFACT_RELATIVE_PATH).exists()
    a1_script_present = (repo / A1_SCRIPT_RELATIVE_PATH).exists()
    a1b_present = (repo / A1B_ARTIFACT_RELATIVE_PATH).exists()
    a1b_gate_skipped = False
    a1_delta = None
    if a1_present:
        try:
            a1_delta = _read_json(repo / A1_ARTIFACT_RELATIVE_PATH).get(
                "value_grounded_first_win_delta_median"
            )
            a1b_gate_skipped = (_finite_float(a1_delta) or -math.inf) >= 0.1
        except (
            OSError,
            ValueError,
            json.JSONDecodeError,
        ):  # pragma: no cover - corrupt precondition.
            a1_delta = None
    a1b_required_present = a1b_present or a1b_gate_skipped
    ok = (
        a1_present
        and a1_script_present
        and a1b_required_present
        and "REQ-ARC-WMTE-4908" in spec_text
        and (repo / "scripts/summarize_artifact.py").exists()
        and (repo / "scripts/adversarial_verify.py").exists()
        and (repo / "scripts/arc_orphan_solver_lint.py").exists()
    )
    return {
        "ok": ok,
        "a1_artifact_present": a1_present,
        "a1_script_present": a1_script_present,
        "a1b_artifact_present": a1b_present,
        "a1b_gate_skipped": a1b_gate_skipped,
        "a1_value_grounded_first_win_delta_median": a1_delta,
        "a1b_artifact_required_present": a1b_required_present,
        "spec_has_req_4908": "REQ-ARC-WMTE-4908" in spec_text,
        "summarizer_script_present": (repo / "scripts/summarize_artifact.py").exists(),
        "adversarial_verify_script_present": (repo / "scripts/adversarial_verify.py").exists(),
        "arc_orphan_solver_lint_present": (repo / "scripts/arc_orphan_solver_lint.py").exists(),
    }


def _base_artifact(*, preconditions_checked: Mapping[str, Any], duration_s: float) -> JsonDict:
    gate_skipped = preconditions_checked.get("a1b_gate_skipped") is True
    return {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "a1_artifact_path": A1_ARTIFACT_RELATIVE_PATH,
        "a1_script_path": A1_SCRIPT_RELATIVE_PATH,
        "a1b_artifact_path": A1B_ARTIFACT_RELATIVE_PATH,
        "a1_artifact_checksum": None,
        "a1_script_checksum": None,
        "a1b_artifact_checksum": None,
        "a1_source_honest_verdict": None,
        "a1_source_fork_verdict": None,
        "a1b_source_honest_verdict": None,
        "a1b_source_fork_verdict": None,
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": "blocked_upstream_artifact_missing",
        "a1_value_from_real_env": False,
        "a1_planner_blind": False,
        "a1_positive_control_non_degenerate": False,
        "a1_numbers_match_fork": False,
        "a1_live_path_reachable": False,
        "a1_trustworthy": False,
        "a1b_ran_genuinely_live": gate_skipped,
        "a1b_gate_skipped": gate_skipped,
        "adversarial_flags_found": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "claim_audit_table": [],
        "checks": {},
        "a1_failure_reasons": ["missing_required_a1_artifact_or_script"],
        "a1b_failure_reasons": [] if gate_skipped else ["missing_required_a1b_artifact"],
        "a1_summarizer_result": {},
        "a1_adversarial_result": {},
        "a1b_summarizer_result": {},
        "a1b_adversarial_result": {},
        "live_lint_result": {},
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "duration_s": round(max(DURATION_FLOOR_S, duration_s), 6),
        "reproducibility_checksum": "",
    }


def blocked_artifact(preconditions_checked: Mapping[str, Any], *, duration_s: float) -> JsonDict:
    artifact = _base_artifact(preconditions_checked=preconditions_checked, duration_s=duration_s)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def build_artifact(
    *,
    root: Path | str,
    a1_artifact: Mapping[str, Any],
    a1b_artifact: Mapping[str, Any] | None,
    audit: Mapping[str, Any],
    a1_summarizer_result: Mapping[str, Any],
    a1_adversarial_result: Mapping[str, Any],
    a1b_summarizer_result: Mapping[str, Any] | None,
    a1b_adversarial_result: Mapping[str, Any] | None,
    live_lint_result: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    repo = Path(root)
    a1_path = repo / A1_ARTIFACT_RELATIVE_PATH
    a1_script = repo / A1_SCRIPT_RELATIVE_PATH
    a1b_path = repo / A1B_ARTIFACT_RELATIVE_PATH
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "a1_artifact_path": A1_ARTIFACT_RELATIVE_PATH,
        "a1_script_path": A1_SCRIPT_RELATIVE_PATH,
        "a1b_artifact_path": A1B_ARTIFACT_RELATIVE_PATH,
        "a1_artifact_checksum": file_checksum(a1_path),
        "a1_script_checksum": file_checksum(a1_script),
        "a1b_artifact_checksum": file_checksum(a1b_path) if a1b_path.exists() else None,
        "a1_source_honest_verdict": a1_artifact.get("honest_verdict"),
        "a1_source_fork_verdict": a1_artifact.get("fork_verdict"),
        "a1b_source_honest_verdict": None
        if a1b_artifact is None
        else a1b_artifact.get("honest_verdict"),
        "a1b_source_fork_verdict": None
        if a1b_artifact is None
        else a1b_artifact.get("fork_verdict"),
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": audit.get("honest_verdict"),
        "a1_value_from_real_env": audit.get("a1_value_from_real_env"),
        "a1_planner_blind": audit.get("a1_planner_blind"),
        "a1_positive_control_non_degenerate": audit.get("a1_positive_control_non_degenerate"),
        "a1_numbers_match_fork": audit.get("a1_numbers_match_fork"),
        "a1_live_path_reachable": audit.get("a1_live_path_reachable"),
        "a1_trustworthy": audit.get("a1_trustworthy"),
        "a1b_ran_genuinely_live": audit.get("a1b_ran_genuinely_live"),
        "a1b_gate_skipped": audit.get("a1b_gate_skipped"),
        "adversarial_flags_found": audit.get("adversarial_flags_found"),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "claim_audit_table": list(audit.get("claim_audit_table") or []),
        "checks": dict(_mapping(audit.get("checks"))),
        "a1_failure_reasons": list(audit.get("a1_failure_reasons") or []),
        "a1b_failure_reasons": list(audit.get("a1b_failure_reasons") or []),
        "a1_summarizer_result": dict(a1_summarizer_result),
        "a1_adversarial_result": dict(a1_adversarial_result),
        "a1b_summarizer_result": dict(a1b_summarizer_result or {}),
        "a1b_adversarial_result": dict(a1b_adversarial_result or {}),
        "live_lint_result": dict(live_lint_result),
        "preconditions_checked": dict(preconditions_checked),
        "random_seed": RANDOM_SEED,
        "duration_s": round(max(DURATION_FLOOR_S, duration_s), 6),
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors = [f"missing:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_mismatch")
    bool_fields = (
        "a1_value_from_real_env",
        "a1_planner_blind",
        "a1_positive_control_non_degenerate",
        "a1_numbers_match_fork",
        "a1_live_path_reachable",
        "a1_trustworthy",
        "a1b_ran_genuinely_live",
        "a1b_gate_skipped",
        "adversarial_flags_found",
    )
    for field in bool_fields:
        if not isinstance(artifact.get(field), bool):
            errors.append(f"{field}_must_be_bool")
    if all(isinstance(artifact.get(field), bool) for field in bool_fields[:5]):
        expected_trust = (
            artifact.get("a1_value_from_real_env") is True
            and artifact.get("a1_planner_blind") is True
            and artifact.get("a1_positive_control_non_degenerate") is True
            and artifact.get("a1_numbers_match_fork") is True
        )
        if artifact.get("a1_trustworthy") != expected_trust:
            errors.append("a1_trustworthy_formula_mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if not isinstance(artifact.get("claim_audit_table"), list):
        errors.append("claim_audit_table_must_be_list")
    else:
        for index, row in enumerate(artifact.get("claim_audit_table") or []):
            if not isinstance(row, Mapping) or not isinstance(row.get("passed"), bool):
                errors.append(f"claim_audit_table.{index}")
    if not isinstance(artifact.get("checks"), dict):
        errors.append("checks_must_be_dict")
    for field in ("a1_failure_reasons", "a1b_failure_reasons"):
        if not isinstance(artifact.get(field), list):
            errors.append(f"{field}_must_be_list")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_mismatch")
    duration = _finite_float(artifact.get("duration_s"))
    if duration is None or duration < DURATION_FLOOR_S:
        errors.append("duration_below_aggregation_floor")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    now: Callable[[], float] | None = None,
) -> JsonDict:
    repo = Path(root)
    clock = now or time.monotonic
    start = clock()
    preconditions = check_preconditions(repo)
    if not preconditions["ok"]:
        artifact = blocked_artifact(preconditions, duration_s=clock() - start)
        if write:
            write_artifact(artifact, root=repo)
        return artifact

    a1_path = repo / A1_ARTIFACT_RELATIVE_PATH
    a1_script_path = repo / A1_SCRIPT_RELATIVE_PATH
    a1b_path = repo / A1B_ARTIFACT_RELATIVE_PATH
    a1_artifact = _read_json(a1_path)
    a1_source_text = a1_script_path.read_text(encoding="utf-8")
    a1b_artifact = _read_json(a1b_path) if a1b_path.exists() else None
    a1_summary = run_summarizer(a1_path)
    a1_adversarial = run_adversarial_verify(a1_path)
    a1b_summary = run_summarizer(a1b_path) if a1b_artifact is not None else None
    a1b_adversarial = run_adversarial_verify(a1b_path) if a1b_artifact is not None else None
    live_lint = run_arc_orphan_solver_lint(repo)
    audit = audit_sources(
        a1_artifact=a1_artifact,
        a1_source_text=a1_source_text,
        a1_summarizer_result=a1_summary,
        a1_adversarial_result=a1_adversarial,
        a1b_artifact=a1b_artifact,
        a1b_summarizer_result=a1b_summary,
        a1b_adversarial_result=a1b_adversarial,
        live_lint_result=live_lint,
    )
    artifact = build_artifact(
        root=repo,
        a1_artifact=a1_artifact,
        a1b_artifact=a1b_artifact,
        audit=audit,
        a1_summarizer_result=a1_summary,
        a1_adversarial_result=a1_adversarial,
        a1b_summarizer_result=a1b_summary,
        a1b_adversarial_result=a1b_adversarial,
        live_lint_result=live_lint,
        preconditions_checked=preconditions,
        duration_s=clock() - start,
    )
    if write:
        write_artifact(artifact, root=repo)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    _ = argv
    artifact = run(write=True)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if not artifact_schema_errors(artifact) else 1


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
