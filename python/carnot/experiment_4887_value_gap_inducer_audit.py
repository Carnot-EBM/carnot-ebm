"""Experiment 4887: adversarial audit of Exp 4882 A1 and Exp 4883 A1b.

Spec refs: REQ-ARC-WMTE-4887,
SCENARIO-ARC-WMTE-4887-A1-A1B-AUDIT,
SCENARIO-ARC-WMTE-4887-BLOCKED-A1-ARTIFACT.
"""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from statistics import median
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
for import_root in (REPO_ROOT, PYTHON_ROOT):  # pragma: no cover - direct script guard.
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


EXPERIMENT = "experiment_4887_value_gap_inducer_audit"
EXPERIMENT_ID = 4887
SCHEMA = "carnot.arc.value_gap_inducer_audit_4887.v1"
A1_ARTIFACT_RELATIVE_PATH = "results/experiment_4882_ttt_dynamics_value_gap.json"
A1_SCRIPT_RELATIVE_PATH = "python/carnot/experiment_4882_ttt_dynamics_value_gap.py"
A1B_ARTIFACT_RELATIVE_PATH = "results/experiment_4883_inducer_ceiling_ab.json"
RESULT_RELATIVE_PATH = "results/experiment_4887_value_gap_inducer_audit.json"
AUDIT_REPORT_RELATIVE_PATH = "ops/arc_null_silent_bug_audit.md"
SPEC_RELATIVE_PATH = "openspec/capabilities/arc-world-model-trust-energy/spec.md"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
DURATION_FLOOR_S = 0.0001
LIVE_LLM_DURATION_FLOOR_S = 60.0
RANDOM_SEED = 4887
BOOTSTRAP_SEED = 20260627
GENERATOR_BACKENDS = ("gpu0_cuda", "igpu_hip")
BUCKETS = ("COVERED", "ENUMERATED_BUT_LOST", "NEVER_ENUMERATED")
FORK_VERDICTS = ("INDUCER_CEILING_BEATABLE", "PLANNER_GAP", "INDUCER_CEILING_HARD")
TERMINAL_PREFIXES = ("complete_", "blocked_", "success_")

SPEC_REFS = [
    "REQ-ARC-WMTE-4887",
    "SCENARIO-ARC-WMTE-4887-A1-A1B-AUDIT",
    "SCENARIO-ARC-WMTE-4887-BLOCKED-A1-ARTIFACT",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; audit complete is complete_a1_a1b_audited."
    },
    "a1_genuinely_diagnostic": {
        "principle": (
            "the load-bearing check -- non-degenerate-graded-control AND "
            "held-out-disjoint-delta AND planner-blind AND numbers-match-fork AND "
            "live-path-reachable AND ran-live-on-GPU0; else A1's fork verdict is void."
        )
    },
    "a1_positive_control_non_degenerate_confirmed": {
        "principle": (
            "true iff the GRADED tu93 positive control really had cell_recall > 0 -- "
            "proves the .449 degenerate-metric failure is FIXED."
        )
    },
    "a1_delta_on_heldout_disjoint_confirmed": {
        "principle": (
            "true iff A1's value-accuracy delta was re-measured on a split DISJOINT "
            "from the TTA adapter's fit set (not a tautology)."
        )
    },
    "planner_blind_confirmed": {
        "principle": (
            "true iff the banked winner was used only to classify, never to seed "
            "induction/adaptation/planning."
        )
    },
    "numbers_match_fork": {
        "principle": (
            "true iff the per-game (cell_recall, value-accuracy, delta) numbers support "
            "the claimed fork_verdict."
        )
    },
    "a1b_ab_trustworthy": {
        "principle": (
            "true iff both A1b lanes scored the SAME held-out split as A1 and "
            "oracle-distinct (or A1b was gate-skipped)."
        )
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (0.0001s floor)."
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
    "a1_source_n_games_measured",
    "a1b_source_honest_verdict",
    "a1b_source_attribution",
    "a1_ran_live_on_gpu0",
    "a1_live_path_reachable_confirmed",
    "field_principles",
    "checks",
    "a1_failure_reasons",
    "a1b_failure_reasons",
    "a1_summarizer_result",
    "a1_adversarial_result",
    "a1b_adversarial_result",
    "live_lint_result",
    "preconditions_checked",
    "audit_report_path",
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


def _unit(value: Any) -> float | None:
    parsed = _finite_float(value)
    return parsed if parsed is not None and 0.0 <= parsed <= 1.0 else None


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
    except Exception as exc:
        command = [sys.executable, "scripts/adversarial_verify.py", str(path)]
        proc = subprocess.run(
            command,
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
    command = [sys.executable, "scripts/arc_orphan_solver_lint.py"]
    proc = subprocess.run(
        command,
        cwd=Path(root),
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    return {
        "command": " ".join(command),
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


def _first_parent_call(node: ast.AST) -> ast.Call | None:
    parent = getattr(node, "_parent", None)
    while parent is not None:
        if isinstance(parent, ast.Call):
            return parent
        parent = getattr(parent, "_parent", None)
    return None


def _source_checks(source_text: str, artifact: Mapping[str, Any]) -> tuple[JsonDict, list[str], JsonDict, list[str]]:
    planner_reasons: list[str] = []
    live_reasons: list[str] = []
    if artifact.get("planner_blind_to_banked_answer") is not True:
        planner_reasons.append("artifact_planner_blind_flag_false")
    try:
        tree = ast.parse(source_text)
    except SyntaxError as exc:
        return (
            {"passed": False, "parse_error": str(exc), "disallowed_refs": []},
            planner_reasons + ["a1_source_not_parseable"],
            {"passed": False, "parse_error": str(exc), "calls": []},
            ["source_live_path_uninspectable"],
        )

    _attach_ast_parents(tree)
    function = _find_function(tree, "measure_game_with_ttt_dynamics_adaptation")
    if function is None:
        return (
            {"passed": False, "function_present": False, "disallowed_refs": []},
            planner_reasons + ["measure_game_with_ttt_dynamics_adaptation_missing"],
            {"passed": False, "function_present": False, "calls": []},
            ["measure_game_with_ttt_dynamics_adaptation_missing"],
        )

    winning_prefix_refs: list[JsonDict] = []
    disallowed_refs: list[JsonDict] = []
    calls: list[str] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            calls.append(_full_call_name(node.func))
        if isinstance(node, ast.Name) and node.id == "winning_prefix":
            call = _first_parent_call(node)
            call_name = _call_name(call.func if call is not None else None)
            record = {"line": int(getattr(node, "lineno", 0)), "call": call_name or None}
            winning_prefix_refs.append(record)
            if call_name != "classify_planned_pool":
                disallowed_refs.append(record)

    if not winning_prefix_refs:
        planner_reasons.append("winning_prefix_not_used_for_classification")
    if disallowed_refs:
        planner_reasons.append("banked_answer_used_before_classification")

    has_adapter = any(name.endswith("DynamicsValueAdapter.fit") for name in calls)
    has_plan = any(name.endswith("_plan_with_adapted_engine") or name.endswith(".plan_in_model") for name in calls)
    has_classify = any(name.endswith("classify_planned_pool") for name in calls)
    if not has_adapter:
        live_reasons.append("dynamics_value_adapter_not_called")
    if not has_plan:
        live_reasons.append("plan_in_model_path_not_called")
    if not has_classify:
        live_reasons.append("classification_path_not_called")

    return (
        {
            "passed": not planner_reasons,
            "artifact_flag": artifact.get("planner_blind_to_banked_answer"),
            "function_present": True,
            "winning_prefix_refs": len(winning_prefix_refs),
            "disallowed_refs": disallowed_refs,
            "allowed_only_in_classify_planned_pool": not disallowed_refs and bool(winning_prefix_refs),
        },
        list(dict.fromkeys(planner_reasons)),
        {
            "passed": not live_reasons,
            "function_present": True,
            "calls": calls,
            "dynamics_value_adapter_called": has_adapter,
            "plan_path_called": has_plan,
            "classification_path_called": has_classify,
        },
        list(dict.fromkeys(live_reasons)),
    )


def _generator_backend(artifact: Mapping[str, Any]) -> str | None:
    if artifact.get("generator_backend") in GENERATOR_BACKENDS:
        return str(artifact.get("generator_backend"))
    model_specs = _mapping(artifact.get("model_specs"))
    if model_specs.get("backend") in GENERATOR_BACKENDS:
        return str(model_specs.get("backend"))
    generator = _mapping(_mapping(artifact.get("preconditions_checked")).get("generator"))
    backend = generator.get("generator_backend") or generator.get("backend")
    return str(backend) if backend in GENERATOR_BACKENDS else None


def _a1_live_gpu_check(artifact: Mapping[str, Any]) -> tuple[JsonDict, list[str], bool]:
    backend = _generator_backend(artifact)
    duration = _finite_float(artifact.get("duration_s"))
    flagged = artifact.get("flagged_adversarial") is True
    reasons: list[str] = []
    if backend not in GENERATOR_BACKENDS or duration is None or duration < LIVE_LLM_DURATION_FLOOR_S or flagged:
        reasons.append("a1_not_live_on_gpu0")
    if backend not in GENERATOR_BACKENDS:
        reasons.append("a1_generator_backend_not_gpu0_or_igpu")
    if duration is None or duration < LIVE_LLM_DURATION_FLOOR_S:
        reasons.append("a1_duration_below_live_floor")
    if flagged:
        reasons.append("a1_flagged_adversarial")
    return (
        {
            "passed": not reasons,
            "generator_backend": backend,
            "duration_s": artifact.get("duration_s"),
            "duration_floor_s": LIVE_LLM_DURATION_FLOOR_S,
            "flagged_adversarial": flagged,
        },
        list(dict.fromkeys(reasons)),
        not reasons,
    )


def _id_set(row: Mapping[str, Any], key: str) -> set[str]:
    return {str(item) for item in row.get(key) or []}


def _split_is_disjoint(row: Mapping[str, Any], *, fit_key: str = "fit_transition_ids", heldout_key: str = "remeasure_transition_ids") -> bool:
    fit = _id_set(row, fit_key)
    heldout = _id_set(row, heldout_key)
    return bool(fit) and bool(heldout) and fit.isdisjoint(heldout)


def _positive_control_recall(row: Mapping[str, Any]) -> float | None:
    direct = _unit(row.get("cell_recall_adapted"))
    if direct is not None:
        return direct
    direct = _unit(row.get("cell_recall_baseline"))
    if direct is not None:
        return direct
    for key in ("adapted_score", "baseline_score"):
        value = _unit(_mapping(row.get(key)).get("cell_recall"))
        if value is not None:
            return value
    return None


def _positive_control_check(artifact: Mapping[str, Any]) -> tuple[JsonDict, list[str]]:
    row = _mapping(artifact.get("positive_control_value_gap"))
    recall = _positive_control_recall(row)
    reasons: list[str] = []
    if artifact.get("positive_control_game") != "tu93" or row.get("game") != "tu93":
        reasons.append("a1_positive_control_not_tu93")
    if artifact.get("positive_control_non_degenerate") is not True:
        reasons.append("a1_positive_control_degenerate")
    if not row:
        reasons.append("a1_positive_control_row_missing")
    if recall is None or recall <= 0.0:
        reasons.append("a1_positive_control_degenerate")
    return (
        {
            "passed": not reasons,
            "positive_control_game": artifact.get("positive_control_game"),
            "artifact_positive_control_non_degenerate": artifact.get(
                "positive_control_non_degenerate"
            ),
            "row_game": row.get("game"),
            "cell_recall": recall,
        },
        list(dict.fromkeys(reasons)),
    )


def _delta_values(per_game: Mapping[str, Any]) -> list[float]:
    values: list[float] = []
    for row_value in per_game.values():
        row = _mapping(row_value)
        value = _finite_float(row.get("value_delta"))
        if value is not None:
            values.append(round(float(value), 6))
    return values


def _bootstrap_ci95(values: Sequence[float], *, iterations: int, seed: int) -> list[float | None]:
    vals = [float(value) for value in values]
    if not vals:
        return [None, None]
    if len(set(vals)) == 1:
        value = round(vals[0], 6)
        return [value, value]
    rng = random.Random(seed)
    samples: list[float] = []
    count = max(1, int(iterations))
    for _ in range(count):
        samples.append(float(median([rng.choice(vals) for _ in vals])))
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return [round(float(lo), 6), round(float(hi), 6)]


def _coverage_migration_count(per_game: Mapping[str, Any]) -> int:
    return sum(1 for row in per_game.values() if _mapping(row).get("migrated") is True)


def _median_delta(per_game: Mapping[str, Any]) -> float | None:
    values = _delta_values(per_game)
    return round(float(median(values)), 6) if values else None


def _median_cell_recall(per_game: Mapping[str, Any]) -> float | None:
    values: list[float] = []
    for row_value in per_game.values():
        row = _mapping(row_value)
        value = _unit(row.get("cell_recall_baseline"))
        if value is not None:
            values.append(value)
    return round(float(median(values)), 6) if values else None


def _computed_fork_verdict(per_game: Mapping[str, Any], positive_control: Mapping[str, Any], ci95: Sequence[Any]) -> str | None:
    if len(per_game) < 3 or (_positive_control_recall(positive_control) or 0.0) <= 0.0:
        return None
    med = _median_delta(per_game)
    lo = _finite_float(ci95[0]) if len(ci95) >= 1 else None
    hi = _finite_float(ci95[1]) if len(ci95) >= 2 else None
    real_lift = med is not None and med > 0.0 and lo is not None and hi is not None and lo > 0.0
    if not real_lift:
        return "INDUCER_CEILING_HARD"
    if _coverage_migration_count(per_game) >= 1:
        return "INDUCER_CEILING_BEATABLE"
    return "PLANNER_GAP"


def _delta_split_check(artifact: Mapping[str, Any]) -> tuple[JsonDict, list[str]]:
    per_game = _mapping(artifact.get("per_game_value_gap"))
    bad_games = sorted(
        str(game)
        for game, row in per_game.items()
        if not isinstance(row, Mapping) or not _split_is_disjoint(row)
    )
    artifact_flag = artifact.get("delta_on_truly_heldout_split") is True
    reasons: list[str] = []
    if bad_games or not artifact_flag:
        reasons.append("a1_delta_split_not_disjoint")
    return (
        {
            "passed": not reasons,
            "artifact_delta_on_truly_heldout_split": artifact_flag,
            "bad_games": bad_games,
            "checked_games": sorted(str(game) for game in per_game),
        },
        reasons,
    )


def _numbers_match_fork_check(artifact: Mapping[str, Any]) -> tuple[JsonDict, list[str]]:
    per_game = _mapping(artifact.get("per_game_value_gap"))
    reasons: list[str] = []
    invalid_games: list[str] = []
    never_enumerated_games = 0
    for game, row_value in per_game.items():
        row = _mapping(row_value)
        bucket = row.get("planned_bucket")
        baseline = _unit(row.get("value_acc_baseline"))
        adapted = _unit(row.get("value_acc_adapted"))
        delta = _finite_float(row.get("value_delta"))
        recall_base = _unit(row.get("cell_recall_baseline"))
        recall_adapted = _unit(row.get("cell_recall_adapted"))
        if bucket not in BUCKETS:
            reasons.append("invalid_planned_bucket")
            invalid_games.append(str(game))
        if bucket == "NEVER_ENUMERATED":
            never_enumerated_games += 1
        if baseline is None or adapted is None or delta is None:
            reasons.append("invalid_value_metric")
            invalid_games.append(str(game))
        elif round(adapted - baseline, 6) != round(delta, 6):
            reasons.append("value_delta_mismatch")
            invalid_games.append(str(game))
        if recall_base is None or recall_adapted is None:
            reasons.append("invalid_cell_recall")
            invalid_games.append(str(game))
        migrated = row.get("migrated")
        if not isinstance(migrated, bool):
            reasons.append("invalid_migrated_flag")
            invalid_games.append(str(game))
        elif bucket in BUCKETS and migrated != (bucket == "COVERED"):
            reasons.append("row_migrated_mismatch")
            invalid_games.append(str(game))

    if len(per_game) < 3:
        reasons.append("n_games_measured_below_3")
    if never_enumerated_games < 3:
        reasons.append("never_enumerated_games_below_3")
    try:
        reported_n = int(artifact.get("n_games_measured"))
    except (TypeError, ValueError):
        reported_n = -1
        reasons.append("n_games_measured_not_integer")
    if reported_n != len(per_game):
        reasons.append("n_games_measured_mismatch")

    computed_median = _median_delta(per_game)
    reported_median = _finite_float(artifact.get("tta_changed_cell_value_accuracy_delta_median"))
    if computed_median is None or reported_median is None or abs(reported_median - computed_median) > 1e-9:
        reasons.append("tta_delta_median_mismatch")

    iterations = int(_mapping(artifact.get("tta_config")).get("bootstrap_iterations") or 1000)
    seed = int(artifact.get("random_seed") or BOOTSTRAP_SEED)
    computed_ci95 = _bootstrap_ci95(_delta_values(per_game), iterations=iterations, seed=seed)
    reported_ci95 = artifact.get("tta_value_accuracy_delta_ci95")
    if not isinstance(reported_ci95, list) or list(reported_ci95) != computed_ci95:
        reasons.append("tta_delta_ci95_mismatch")

    computed_recall = _median_cell_recall(per_game)
    reported_recall = _finite_float(artifact.get("engine_cell_recall_median"))
    if computed_recall is None or reported_recall is None or abs(reported_recall - computed_recall) > 1e-9:
        reasons.append("engine_cell_recall_median_mismatch")

    computed_migrations = _coverage_migration_count(per_game)
    if artifact.get("coverage_migration_count") != computed_migrations:
        reasons.append("coverage_migration_count_mismatch")

    computed_fork = _computed_fork_verdict(
        per_game, _mapping(artifact.get("positive_control_value_gap")), computed_ci95
    )
    reported_fork = artifact.get("fork_verdict")
    if reported_fork not in FORK_VERDICTS:
        reasons.append("invalid_fork_verdict")
    elif computed_fork != reported_fork:
        reasons.append("fork_verdict_mismatch")

    unique = list(dict.fromkeys(reasons))
    return (
        {
            "passed": not unique,
            "reported_fork_verdict": reported_fork,
            "computed_fork_verdict": computed_fork,
            "reported_n_games_measured": artifact.get("n_games_measured"),
            "per_game_count": len(per_game),
            "never_enumerated_games": never_enumerated_games,
            "reported_median_delta": artifact.get("tta_changed_cell_value_accuracy_delta_median"),
            "computed_median_delta": computed_median,
            "reported_ci95": reported_ci95,
            "computed_ci95": computed_ci95,
            "reported_engine_cell_recall_median": artifact.get("engine_cell_recall_median"),
            "computed_engine_cell_recall_median": computed_recall,
            "reported_coverage_migration_count": artifact.get("coverage_migration_count"),
            "computed_coverage_migration_count": computed_migrations,
            "invalid_games": sorted(set(invalid_games)),
        },
        unique,
    )


def _tool_cleanliness_check(
    summarizer_result: Mapping[str, Any], adversarial_result: Mapping[str, Any]
) -> tuple[JsonDict, list[str]]:
    summarizer_clean = summarizer_result.get("returncode") == 0
    adversarial_clean = (
        adversarial_result.get("loaded") is not False
        and int(adversarial_result.get("flag_count") or 0) == 0
    )
    reasons: list[str] = []
    if not summarizer_clean:
        reasons.append("a1_summarizer_failed")
    if not adversarial_clean:
        reasons.append("a1_adversarial_verify_flagged")
    return (
        {
            "passed": summarizer_clean and adversarial_clean,
            "summarizer_returncode": summarizer_result.get("returncode"),
            "adversarial_flag_count": adversarial_result.get("flag_count"),
            "adversarial_loaded": adversarial_result.get("loaded"),
        },
        reasons,
    )


def _a1_oracle_check(artifact: Mapping[str, Any], adversarial_result: Mapping[str, Any]) -> tuple[JsonDict, list[str]]:
    circular_flags = [
        flag
        for flag in adversarial_result.get("flags", [])
        if isinstance(flag, Mapping) and flag.get("kind") == "CIRCULAR_MOAT_OVERCLAIM"
    ]
    reasons: list[str] = []
    if artifact.get("verifier_is_oracle") is True:
        reasons.append("a1_verifier_is_oracle")
    if circular_flags:
        reasons.append("a1_circular_moat_overclaim")
    return (
        {
            "passed": not reasons,
            "verifier_is_oracle": artifact.get("verifier_is_oracle"),
            "circular_moat_flag_count": len(circular_flags),
        },
        reasons,
    )


def _live_path_check(
    artifact: Mapping[str, Any],
    live_lint_result: Mapping[str, Any],
    source_live_check: Mapping[str, Any],
    source_live_reasons: list[str],
) -> tuple[JsonDict, list[str], bool]:
    lint_passed = live_lint_result.get("passed") is True
    artifact_live = artifact.get("live_path_reachable") is True
    source_live = source_live_check.get("passed") is True
    reasons: list[str] = []
    if not (lint_passed and artifact_live and source_live):
        reasons.append("live_path_unreachable")
    if artifact.get("solve_provenance") != "development_proxy":
        reasons.append("solve_provenance_not_development_proxy")
    reasons.extend(source_live_reasons)
    return (
        {
            "passed": not reasons,
            "arc_orphan_solver_lint_passed": lint_passed,
            "artifact_live_path_reachable": artifact_live,
            "source_live_path_called": source_live,
            "source_live_path": dict(source_live_check),
            "solve_provenance": artifact.get("solve_provenance"),
        },
        list(dict.fromkeys(reasons)),
        lint_passed and artifact_live and source_live,
    )


def _a1b_gate_skipped(a1_artifact: Mapping[str, Any]) -> bool:
    verdict = str(a1_artifact.get("honest_verdict") or "")
    fork = str(a1_artifact.get("fork_verdict") or "")
    median_delta = _finite_float(a1_artifact.get("tta_changed_cell_value_accuracy_delta_median"))
    return verdict.startswith("blocked_") or fork in {"INDUCER_CEILING_BEATABLE", "PLANNER_GAP"} or (
        median_delta is not None and median_delta >= 0.1
    )


def _same_ids(left: Any, right: Any) -> bool:
    return [str(item) for item in (left or [])] == [str(item) for item in (right or [])]


def _a1b_ab_check(
    *,
    a1_artifact: Mapping[str, Any],
    a1b_artifact: Mapping[str, Any] | None,
    a1b_adversarial_result: Mapping[str, Any] | None,
    live_lint_result: Mapping[str, Any],
) -> tuple[JsonDict, list[str], bool]:
    if a1b_artifact is None:
        skipped = _a1b_gate_skipped(a1_artifact)
        return (
            {
                "passed": skipped,
                "status": "gate_skipped" if skipped else "missing",
                "reason": "a1_closed_value_gap_or_blocked" if skipped else "a1b_missing_after_low_value_a1",
            },
            [] if skipped else ["a1b_artifact_missing_after_low_value_a1"],
            skipped,
        )

    a1_rows = _mapping(a1_artifact.get("per_game_value_gap"))
    lanes = _mapping(a1b_artifact.get("per_lane_per_game"))
    reference = _mapping(lanes.get("reference"))
    local = _mapping(lanes.get("local"))
    a1_games = set(str(game) for game in a1_rows)
    reference_games = set(str(game) for game in reference)
    local_games = set(str(game) for game in local)
    same_games = bool(a1_games) and reference_games == a1_games and local_games == a1_games
    bad_split_games: list[str] = []
    bad_delta_games: list[str] = []
    for game in sorted(a1_games):
        a1_row = _mapping(a1_rows.get(game))
        expected_ids = list(a1_row.get("remeasure_transition_ids") or [])
        baseline = _unit(a1_row.get("value_acc_baseline"))
        for lane_name, lane_rows in (("reference", reference), ("local", local)):
            row = _mapping(lane_rows.get(game))
            if not row or not _same_ids(row.get("heldout_transition_ids"), expected_ids):
                bad_split_games.append(f"{lane_name}:{game}:heldout")
            if not _same_ids(row.get("a1_heldout_transition_ids"), expected_ids):
                bad_split_games.append(f"{lane_name}:{game}:a1_heldout")
            if not _split_is_disjoint(row, heldout_key="heldout_transition_ids"):
                bad_split_games.append(f"{lane_name}:{game}:fit_overlap")
            value_acc = _unit(row.get("value_acc"))
            delta = _finite_float(row.get("delta_vs_baseline"))
            row_baseline = _unit(row.get("a1_baseline_value_acc"))
            if (
                baseline is None
                or row_baseline is None
                or value_acc is None
                or delta is None
                or abs(row_baseline - baseline) > 1e-9
                or round(value_acc - baseline, 6) != round(delta, 6)
            ):
                bad_delta_games.append(f"{lane_name}:{game}")

    circular_flags = [
        flag
        for flag in (a1b_adversarial_result or {}).get("flags", [])
        if isinstance(flag, Mapping) and flag.get("kind") == "CIRCULAR_MOAT_OVERCLAIM"
    ]
    adversarial_flag_count = int((a1b_adversarial_result or {}).get("flag_count") or 0)
    duration = _finite_float(a1b_artifact.get("duration_s"))
    reasons: list[str] = []
    if not same_games or bad_split_games or a1b_artifact.get("delta_on_truly_heldout_split") is not True:
        reasons.append("a1b_not_same_heldout_split_as_a1")
    if bad_delta_games:
        reasons.append("a1b_delta_vs_baseline_mismatch")
    if a1b_artifact.get("verifier_is_oracle") is True:
        reasons.append("a1b_verifier_is_oracle")
    if circular_flags:
        reasons.append("a1b_circular_moat_overclaim")
    if a1b_artifact.get("live_path_reachable") is not True or live_lint_result.get("passed") is not True:
        reasons.append("a1b_live_path_unreachable")
    if a1b_artifact.get("reference_lane_is_ceiling_only") is not True:
        reasons.append("a1b_reference_lane_not_ceiling_only")
    if a1b_artifact.get("solve_provenance") != "development_proxy":
        reasons.append("a1b_solve_provenance_not_development_proxy")
    if a1b_artifact.get("flagged_adversarial") is True:
        reasons.append("a1b_flagged_adversarial_stamp")
    if adversarial_flag_count != 0:
        reasons.append("a1b_adversarial_verify_flagged")
    if duration is not None and duration < LIVE_LLM_DURATION_FLOOR_S:
        reasons.append("a1b_duration_below_live_floor")

    unique = list(dict.fromkeys(reasons))
    return (
        {
            "passed": not unique,
            "status": "ran",
            "same_games_as_a1": same_games,
            "same_split_as_a1": not bad_split_games
            and a1b_artifact.get("delta_on_truly_heldout_split") is True,
            "bad_split_games": bad_split_games,
            "bad_delta_games": bad_delta_games,
            "a1_games": sorted(a1_games),
            "reference_games": sorted(reference_games),
            "local_games": sorted(local_games),
            "oracle_distinct": a1b_artifact.get("verifier_is_oracle") is not True
            and not circular_flags,
            "verifier_is_oracle": a1b_artifact.get("verifier_is_oracle"),
            "live_path_reachable": a1b_artifact.get("live_path_reachable"),
            "reference_lane_is_ceiling_only": a1b_artifact.get("reference_lane_is_ceiling_only"),
            "adversarial_flag_count": adversarial_flag_count,
            "duration_s": a1b_artifact.get("duration_s"),
            "duration_floor_s": LIVE_LLM_DURATION_FLOOR_S,
        },
        unique,
        not unique,
    )


def audit_sources(
    *,
    a1_artifact: Mapping[str, Any],
    a1_source_text: str,
    a1_summarizer_result: Mapping[str, Any],
    a1_adversarial_result: Mapping[str, Any],
    a1b_artifact: Mapping[str, Any] | None,
    a1b_adversarial_result: Mapping[str, Any] | None,
    live_lint_result: Mapping[str, Any],
) -> JsonDict:
    source_check, planner_reasons, source_live_check, source_live_reasons = _source_checks(
        a1_source_text, a1_artifact
    )
    live_gpu_check, live_gpu_reasons, ran_live = _a1_live_gpu_check(a1_artifact)
    positive_check, positive_reasons = _positive_control_check(a1_artifact)
    split_check, split_reasons = _delta_split_check(a1_artifact)
    numbers_check, number_reasons = _numbers_match_fork_check(a1_artifact)
    tool_check, tool_reasons = _tool_cleanliness_check(a1_summarizer_result, a1_adversarial_result)
    oracle_check, oracle_reasons = _a1_oracle_check(a1_artifact, a1_adversarial_result)
    live_path_check, live_path_reasons, live_path_confirmed = _live_path_check(
        a1_artifact, live_lint_result, source_live_check, source_live_reasons
    )
    a1b_check, a1b_reasons, a1b_trustworthy = _a1b_ab_check(
        a1_artifact=a1_artifact,
        a1b_artifact=dict(a1b_artifact) if a1b_artifact is not None else None,
        a1b_adversarial_result=a1b_adversarial_result,
        live_lint_result=live_lint_result,
    )

    a1_reasons = list(
        dict.fromkeys(
            live_gpu_reasons
            + positive_reasons
            + split_reasons
            + planner_reasons
            + number_reasons
            + tool_reasons
            + oracle_reasons
            + live_path_reasons
        )
    )
    positive_control = not positive_reasons
    heldout_disjoint = not split_reasons
    planner_blind = not planner_reasons
    numbers_match = not number_reasons
    diagnostic = (
        ran_live
        and positive_control
        and heldout_disjoint
        and planner_blind
        and numbers_match
        and live_path_confirmed
        and not tool_reasons
        and not oracle_reasons
    )
    return {
        "honest_verdict": "complete_a1_a1b_audited",
        "a1_genuinely_diagnostic": diagnostic,
        "a1_ran_live_on_gpu0": ran_live,
        "a1_live_path_reachable_confirmed": live_path_confirmed,
        "a1_positive_control_non_degenerate_confirmed": positive_control,
        "a1_delta_on_heldout_disjoint_confirmed": heldout_disjoint,
        "planner_blind_confirmed": planner_blind,
        "numbers_match_fork": numbers_match,
        "a1b_ab_trustworthy": a1b_trustworthy,
        "a1_failure_reasons": a1_reasons,
        "a1b_failure_reasons": a1b_reasons,
        "checks": {
            "a1_live_gpu": live_gpu_check,
            "a1_positive_control": positive_check,
            "a1_heldout_disjoint_delta": split_check,
            "a1_planner_blind_to_banked_answer": source_check,
            "a1_numbers_match_fork": numbers_check,
            "a1_summarizer_and_adversarial_verify": tool_check,
            "a1_oracle_distinct": oracle_check,
            "a1_live_path": live_path_check,
            "a1b_ab_fairness": a1b_check,
        },
    }


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    repo = Path(root)
    spec_text = (repo / SPEC_RELATIVE_PATH).read_text(encoding="utf-8") if (repo / SPEC_RELATIVE_PATH).exists() else ""
    return {
        "ok": (
            (repo / A1_ARTIFACT_RELATIVE_PATH).exists()
            and (repo / A1_SCRIPT_RELATIVE_PATH).exists()
            and "REQ-ARC-WMTE-4887" in spec_text
            and (repo / "scripts/summarize_artifact.py").exists()
            and (repo / "scripts/adversarial_verify.py").exists()
            and (repo / "scripts/arc_orphan_solver_lint.py").exists()
        ),
        "a1_artifact_present": (repo / A1_ARTIFACT_RELATIVE_PATH).exists(),
        "a1_script_present": (repo / A1_SCRIPT_RELATIVE_PATH).exists(),
        "a1b_artifact_present": (repo / A1B_ARTIFACT_RELATIVE_PATH).exists(),
        "spec_has_req_4887": "REQ-ARC-WMTE-4887" in spec_text,
        "summarizer_script_present": (repo / "scripts/summarize_artifact.py").exists(),
        "adversarial_verify_script_present": (repo / "scripts/adversarial_verify.py").exists(),
        "arc_orphan_solver_lint_present": (repo / "scripts/arc_orphan_solver_lint.py").exists(),
    }


def blocked_artifact(preconditions_checked: Mapping[str, Any], *, duration_s: float) -> JsonDict:
    artifact: JsonDict = {
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
        "a1_source_n_games_measured": None,
        "a1b_source_honest_verdict": None,
        "a1b_source_attribution": None,
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": "blocked_a1_artifact_missing",
        "a1_genuinely_diagnostic": False,
        "a1_ran_live_on_gpu0": False,
        "a1_live_path_reachable_confirmed": False,
        "a1_positive_control_non_degenerate_confirmed": False,
        "a1_delta_on_heldout_disjoint_confirmed": False,
        "planner_blind_confirmed": False,
        "numbers_match_fork": False,
        "a1b_ab_trustworthy": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "checks": {},
        "a1_failure_reasons": ["missing_a1_artifact_or_script"],
        "a1b_failure_reasons": ["not_audited_without_a1"],
        "a1_summarizer_result": {},
        "a1_adversarial_result": {},
        "a1b_adversarial_result": {},
        "live_lint_result": {},
        "preconditions_checked": dict(preconditions_checked),
        "audit_report_path": AUDIT_REPORT_RELATIVE_PATH,
        "random_seed": RANDOM_SEED,
        "duration_s": round(max(DURATION_FLOOR_S, duration_s), 6),
        "reproducibility_checksum": "",
    }
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
        "a1_source_n_games_measured": a1_artifact.get("n_games_measured"),
        "a1b_source_honest_verdict": None if a1b_artifact is None else a1b_artifact.get("honest_verdict"),
        "a1b_source_attribution": None if a1b_artifact is None else a1b_artifact.get("inducer_ceiling_attribution"),
        "field_principles": FIELD_PRINCIPLES,
        "honest_verdict": audit.get("honest_verdict"),
        "a1_genuinely_diagnostic": audit.get("a1_genuinely_diagnostic"),
        "a1_ran_live_on_gpu0": audit.get("a1_ran_live_on_gpu0"),
        "a1_live_path_reachable_confirmed": audit.get("a1_live_path_reachable_confirmed"),
        "a1_positive_control_non_degenerate_confirmed": audit.get(
            "a1_positive_control_non_degenerate_confirmed"
        ),
        "a1_delta_on_heldout_disjoint_confirmed": audit.get(
            "a1_delta_on_heldout_disjoint_confirmed"
        ),
        "planner_blind_confirmed": audit.get("planner_blind_confirmed"),
        "numbers_match_fork": audit.get("numbers_match_fork"),
        "a1b_ab_trustworthy": audit.get("a1b_ab_trustworthy"),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "checks": dict(_mapping(audit.get("checks"))),
        "a1_failure_reasons": list(audit.get("a1_failure_reasons") or []),
        "a1b_failure_reasons": list(audit.get("a1b_failure_reasons") or []),
        "a1_summarizer_result": dict(a1_summarizer_result),
        "a1_adversarial_result": dict(a1_adversarial_result),
        "a1b_adversarial_result": dict(a1b_adversarial_result or {}),
        "live_lint_result": dict(live_lint_result),
        "preconditions_checked": dict(preconditions_checked),
        "audit_report_path": AUDIT_REPORT_RELATIVE_PATH,
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
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    for field in (
        "a1_genuinely_diagnostic",
        "a1_positive_control_non_degenerate_confirmed",
        "a1_delta_on_heldout_disjoint_confirmed",
        "planner_blind_confirmed",
        "numbers_match_fork",
        "a1b_ab_trustworthy",
        "a1_ran_live_on_gpu0",
        "a1_live_path_reachable_confirmed",
    ):
        if not isinstance(artifact.get(field), bool):
            errors.append(f"{field}_must_be_bool")
    if not isinstance(artifact.get("checks"), dict):
        errors.append("checks_must_be_dict")
    for field in ("a1_failure_reasons", "a1b_failure_reasons"):
        if not isinstance(artifact.get(field), list):
            errors.append(f"{field}_must_be_list")
    if artifact.get("a1_genuinely_diagnostic") is True and artifact.get("a1_failure_reasons"):
        errors.append("diagnostic_artifact_has_a1_failure_reasons")
    if artifact.get("a1b_ab_trustworthy") is True and artifact.get("a1b_failure_reasons"):
        errors.append("trustworthy_a1b_has_failure_reasons")
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


def render_markdown_section(artifact: Mapping[str, Any]) -> str:
    rows = [
        "",
        "## Experiment 4887 A1/A1b Audit",
        "",
        f"- Verdict: `{artifact.get('honest_verdict')}`",
        f"- a1_genuinely_diagnostic: `{artifact.get('a1_genuinely_diagnostic')}`",
        f"- a1_positive_control_non_degenerate_confirmed: `{artifact.get('a1_positive_control_non_degenerate_confirmed')}`",
        f"- a1_delta_on_heldout_disjoint_confirmed: `{artifact.get('a1_delta_on_heldout_disjoint_confirmed')}`",
        f"- planner_blind_confirmed: `{artifact.get('planner_blind_confirmed')}`",
        f"- numbers_match_fork: `{artifact.get('numbers_match_fork')}`",
        f"- a1b_ab_trustworthy: `{artifact.get('a1b_ab_trustworthy')}`",
        f"- A1 reasons: `{', '.join(artifact.get('a1_failure_reasons') or []) or '-'}`",
        f"- A1b reasons: `{', '.join(artifact.get('a1b_failure_reasons') or []) or '-'}`",
        f"- Inference substrate: `{artifact.get('inference_substrate')}`",
        "",
        "| Check | Passed | Detail |",
        "|---|---:|---|",
    ]
    for name, check in _mapping(artifact.get("checks")).items():
        if isinstance(check, Mapping):
            detail = {key: value for key, value in check.items() if key != "passed"}
            rows.append(
                f"| `{name}` | `{check.get('passed')}` | `{json.dumps(detail, sort_keys=True)}` |"
            )
    rows.extend(
        [
            "",
            f"- A1 artifact checksum: `{artifact.get('a1_artifact_checksum')}`",
            f"- A1 script checksum: `{artifact.get('a1_script_checksum')}`",
            f"- A1b artifact checksum: `{artifact.get('a1b_artifact_checksum')}`",
            "",
        ]
    )
    return "\n".join(rows)


def append_markdown_report(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    report_path = Path(root) / AUDIT_REPORT_RELATIVE_PATH
    marker = "## Experiment 4887 A1/A1b Audit"
    if report_path.exists():
        current = report_path.read_text(encoding="utf-8")
        if marker in current:
            return report_path
    else:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        current = "# ARC Null Silent-Bug Audit\n"
    report_path.write_text(current.rstrip() + render_markdown_section(artifact), encoding="utf-8")
    return report_path


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
            append_markdown_report(artifact, root=repo)
        return artifact

    a1_path = repo / A1_ARTIFACT_RELATIVE_PATH
    a1_script = repo / A1_SCRIPT_RELATIVE_PATH
    a1b_path = repo / A1B_ARTIFACT_RELATIVE_PATH
    a1_artifact = _read_json(a1_path)
    a1_source = a1_script.read_text(encoding="utf-8")
    a1b_artifact = _read_json(a1b_path) if a1b_path.exists() else None
    a1_summary = run_summarizer(a1_path)
    a1_adversarial = run_adversarial_verify(a1_path)
    a1b_adversarial = run_adversarial_verify(a1b_path) if a1b_artifact is not None else None
    live_lint = run_arc_orphan_solver_lint(repo)
    audit = audit_sources(
        a1_artifact=a1_artifact,
        a1_source_text=a1_source,
        a1_summarizer_result=a1_summary,
        a1_adversarial_result=a1_adversarial,
        a1b_artifact=a1b_artifact,
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
        a1b_adversarial_result=a1b_adversarial,
        live_lint_result=live_lint,
        preconditions_checked=preconditions,
        duration_s=clock() - start,
    )
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    if write:
        write_artifact(artifact, root=repo)
        append_markdown_report(artifact, root=repo)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary.
    _ = argv
    artifact = run()
    print(
        json.dumps(
            {
                "artifact": RESULT_RELATIVE_PATH,
                "honest_verdict": artifact["honest_verdict"],
                "a1_genuinely_diagnostic": artifact["a1_genuinely_diagnostic"],
                "a1b_ab_trustworthy": artifact["a1b_ab_trustworthy"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary.
    raise SystemExit(main())
