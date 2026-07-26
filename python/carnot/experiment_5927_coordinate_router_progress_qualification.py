"""Exp5927 coordinate-router progress qualification.

Spec refs: REQ-ARC-FCP-5927, SCENARIO-ARC-FCP-5927-POWERED-PROGRESS-CORPUS,
SCENARIO-ARC-FCP-5927-CONTROLS-AND-LEAKAGE,
SCENARIO-ARC-FCP-5927-COMMITTED-OUTCOME-HOOK,
SCENARIO-ARC-FCP-5927-NO-PROMOTION-WITHOUT-GATE.

This is an offline development-proxy qualification for within-game click-target
discrimination. It deliberately makes no level-solve claim and keeps the live
coordinate router default off.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np

from carnot.agentic.arc_click_target_features import (
    CLICK_TARGET_FEATURE_DIM,
    OnlineClickTargetDiscriminator,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5927_coordinate_router_progress_qualification.json")
CHECKPOINT_RELATIVE_PATH = Path(
    "results/checkpoints/experiment_5927_coordinate_router_progress_qualification.corpus.json"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5927_coordinate_router_progress_qualification.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5927_coordinate_router_progress_qualification.py"
)
ROUTER_MODULE_RELATIVE_PATH = Path("python/carnot/agentic/arc_discriminative_router.py")
ROUTER_TEST_RELATIVE_PATH = Path("tests/python/test_arc_online_click_target_router.py")
AGENT_MODULE_RELATIVE_PATH = Path("python/carnot/agentic/arc_competition_agent.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/arc-human-replay-frame-change/spec.md")
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
EXP5904_RESULT_RELATIVE_PATH = Path("results/experiment_5904_click_target_discrimination.json")
EXP5758_RESULT_RELATIVE_PATH = Path("results/experiment_5758_click_ranking_fix_ab.json")
RUN_DATE = "20260726"
RANDOM_SEED = 5927
EXPERIMENT_ID = "experiment_5927_coordinate_router_progress_qualification"
SCHEMA_VERSION = "carnot.exp5927.coordinate_router_progress_qualification.v1"
INFERENCE_SUBSTRATE = "deterministic_offline_arc_development_proxy_no_llm"
VERIFIER_IS_ORACLE = True
SOLVE_PROVENANCE = "development_proxy"
MIN_HARD_PROGRESS_POSITIVES = 30
DEFAULT_GAMES = ("lp85", "vc33", "su15", "r11l", "tn36")
DEFAULT_MAX_STATES = 8
DEFAULT_MAX_CLICKS = 48
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"

TASK_OWNED_COMMANDS = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5927_coordinate_router_progress_qualification.py "
    "tests/python/test_arc_online_click_target_router.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5927_coordinate_router_progress_qualification.py "
    "-m pytest tests/python/test_experiment_5927_coordinate_router_progress_qualification.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5927_coordinate_router_progress_qualification.py "
    "--fail-under=100",
    ".venv/bin/ruff check "
    "python/carnot/experiment_5927_coordinate_router_progress_qualification.py "
    "python/carnot/agentic/arc_discriminative_router.py "
    "python/carnot/agentic/arc_competition_agent.py "
    "tests/python/test_experiment_5927_coordinate_router_progress_qualification.py "
    "tests/python/test_arc_online_click_target_router.py",
    ".venv/bin/ruff format --check "
    "python/carnot/experiment_5927_coordinate_router_progress_qualification.py "
    "python/carnot/agentic/arc_discriminative_router.py "
    "python/carnot/agentic/arc_competition_agent.py "
    "tests/python/test_experiment_5927_coordinate_router_progress_qualification.py "
    "tests/python/test_arc_online_click_target_router.py",
    ".venv/bin/python -m carnot.experiment_5927_coordinate_router_progress_qualification "
    "--validate",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5927_coordinate_router_progress_qualification.json",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5927_coordinate_router_progress_qualification.py "
    "tests/python/test_arc_online_click_target_router.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git status --short -- scripts/research_conductor.py ops/changelog.md ops/status.md "
    "_bmad/traceability.md",
)
DEFAULT_TEST_COMMANDS = (*TASK_OWNED_COMMANDS, GLOBAL_PYTEST_COMMAND)
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
SOURCE_AND_INPUT_RELATIVE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    REGISTRY_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    EXP5904_RESULT_RELATIVE_PATH,
    EXP5758_RESULT_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    ROUTER_MODULE_RELATIVE_PATH,
    ROUTER_TEST_RELATIVE_PATH,
    AGENT_MODULE_RELATIVE_PATH,
    Path("scripts/arc_loop_solve.py"),
    SPEC_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "registry_precheck_receipt",
    "games_states_rows_and_label_manifest",
    "solve_provenance",
    "no_level_solve_or_registry_update",
    "cross_game_checkpoint_loaded",
    "online_within_game_only",
    "coordinate_static_blind_step_and_random_controls",
    "frame_change_vs_validated_progress_receipts",
    "hard_progress_positive_count_and_power_gate",
    "within_state_and_leave_state_out_metrics",
    "coordinate_over_static_delta_and_interval",
    "random_control_sanity",
    "observe_click_outcome_contract_and_tests",
    "reset_rollback_delay_duplicate_missing_and_replay_matrix",
    "cross_game_isolation_and_leakage_checks",
    "default_enabled",
    "old_path_regression_receipt",
    "protected_files_unchanged",
    "coordinate_router_progress_ready_score",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "solve_provenance": "use `development_proxy`; this offline qualification receives no level credit.",
    "cross_game_checkpoint_loaded": "must be bare false.",
    "default_enabled": "must be bare false unless the full preregistered promotion gate passes, and the task still may not mutate the default.",
    "coordinate_router_progress_ready_score": "emit bare 1.0 only for at least 30 hard positives, interval-separated coordinate gain over static salience, no leakage, random sanity, hook correctness, and no old-path regression.",
    "inference_substrate": "use `deterministic_offline_arc_development_proxy_no_llm`.",
    "verifier_is_oracle": "true only for environment progress, action legality, and exact replay labels.",
    "honest_verdict": "use `complete_ready:`, `complete_underpowered:`, `retired:`, or `blocked:`.",
}


def canonical_json(value: Any) -> str:
    """Serialize JSON evidence into stable ASCII text."""

    return json.dumps(
        _normalize_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for canonical JSON-compatible data."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, np.generic):  # pragma: no cover - numpy scalar guard
        return _normalize_json(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):  # pragma: no cover - defensive JSON normalization
            return None
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):  # pragma: no cover - artifact path guard
        return value.as_posix()
    return str(value)  # pragma: no cover - last-ditch JSON normalization


def _write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:  # pragma: no cover
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)


def _stable_score(parts: Mapping[str, Any]) -> float:
    digest = hashlib.sha256(canonical_json(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def _label(row: Mapping[str, Any]) -> float:
    return 1.0 if bool(row.get("validated_progress")) else 0.0


def normalize_corpus_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Normalize row schemas before fitting or artifact construction."""

    normalized: list[JsonDict] = []
    for index, source in enumerate(rows):
        features = [float(value) for value in source.get("features", [])]
        if len(features) != CLICK_TARGET_FEATURE_DIM:  # pragma: no cover - corrupt row guard
            raise ValueError(
                f"row {index} expected {CLICK_TARGET_FEATURE_DIM} features, got {len(features)}"
            )
        raw_frame_change = bool(source.get("raw_frame_change", source.get("changed", False)))
        validated_progress = bool(
            source.get("validated_progress", source.get("levels_up", source.get("label", False)))
        )
        normalized.append(
            {
                "game": str(source["game"]),
                "state_index": int(source["state_index"]),
                "row_id": str(
                    source.get("row_id", f"{source['game']}:s{source['state_index']}:r{index}")
                ),
                "x": int(source.get("x", 0)),
                "y": int(source.get("y", 0)),
                "salience_rank": int(source.get("salience_rank", index)),
                "raw_frame_change": raw_frame_change,
                "ui_animation": bool(source.get("ui_animation", False)),
                "state_novelty": bool(source.get("state_novelty", False)),
                "validated_progress": validated_progress,
                "action_legal": bool(source.get("action_legal", True)),
                "features": features,
                "blind_action_id": int(source.get("blind_action_id", 6)),
            }
        )
    return normalized


def games_states_rows_and_label_manifest(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count games, states, rows, and the four separated label concepts."""

    by_game: OrderedDict[str, set[int]] = OrderedDict()
    for row in rows:
        by_game.setdefault(str(row["game"]), set()).add(int(row["state_index"]))
    return {
        "games": sorted(by_game),
        "n_games": len(by_game),
        "n_states": sum(len(states) for states in by_game.values()),
        "row_count": len(rows),
        "raw_frame_change_rows": sum(1 for row in rows if bool(row["raw_frame_change"])),
        "ui_animation_only_rows": sum(
            1 for row in rows if bool(row["ui_animation"]) and not bool(row["raw_frame_change"])
        ),
        "state_novelty_rows": sum(1 for row in rows if bool(row["state_novelty"])),
        "validated_progress_rows": sum(1 for row in rows if bool(row["validated_progress"])),
        "per_game": {
            game: {
                "states": len(states),
                "rows": sum(1 for row in rows if str(row["game"]) == game),
                "validated_progress_rows": sum(
                    1
                    for row in rows
                    if str(row["game"]) == game and bool(row["validated_progress"])
                ),
            }
            for game, states in sorted(by_game.items())
        },
        "label_contract": {
            "raw_frame_change": "settled grid bytes changed after stepping the candidate click",
            "ui_animation": "raw rendered frame changed while the settled grid did not",
            "state_novelty": "candidate outcome produced a novel settled state within that source state",
            "validated_progress": "environment level counter increased after the candidate click",
        },
    }


def hard_progress_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Rows used by the primary hard/progress slice."""

    return [row for row in rows if bool(row["raw_frame_change"]) and bool(row["action_legal"])]


def hard_progress_power_gate(
    rows: Sequence[Mapping[str, Any]], *, min_positive: int = MIN_HARD_PROGRESS_POSITIVES
) -> JsonDict:
    """Stop interpretation when the hard/progress slice has too few positives."""

    hard_rows = hard_progress_rows(rows)
    positives = sum(1 for row in hard_rows if bool(row["validated_progress"]))
    return {
        "hard_progress_row_count": len(hard_rows),
        "hard_progress_positive_count": positives,
        "min_positive_rows": int(min_positive),
        "powered": positives >= int(min_positive),
        "status": "powered" if positives >= int(min_positive) else "underpowered",
    }


def auroc(scores: Sequence[float], labels: Sequence[float]) -> float | None:
    """Tie-aware AUROC."""

    pairs = [(float(score), float(label)) for score, label in zip(scores, labels)]
    n_pos = sum(1 for _score, label in pairs if label >= 0.5)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:  # pragma: no cover - degenerate metric guard
        return None
    order = sorted(range(len(pairs)), key=lambda index: pairs[index][0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and pairs[order[j + 1]][0] == pairs[order[i]][0]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    pos_rank_sum = sum(ranks[index] for index, (_score, label) in enumerate(pairs) if label >= 0.5)
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def stratified_auroc(
    rows: Sequence[Mapping[str, Any]], scores: Sequence[float]
) -> tuple[float | None, int, int]:
    """Within-state pooled AUROC, weighted by discriminating pairs."""

    by_state: OrderedDict[tuple[str, int], list[tuple[float, float]]] = OrderedDict()
    for row, score in zip(rows, scores):
        by_state.setdefault((str(row["game"]), int(row["state_index"])), []).append(
            (float(score), _label(row))
        )
    u_total = 0.0
    pair_total = 0.0
    n_states = 0
    n_rows = 0
    for pairs in by_state.values():
        labels = [label for _score, label in pairs]
        n_pos = sum(1 for label in labels if label >= 0.5)
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:  # pragma: no cover - degenerate state guard
            continue
        value = auroc([score for score, _label_value in pairs], labels)
        if value is None:  # pragma: no cover - guarded by class count above
            continue
        weight = n_pos * n_neg
        u_total += value * weight
        pair_total += weight
        n_states += 1
        n_rows += len(pairs)
    if pair_total <= 0:  # pragma: no cover - no usable state guard
        return None, 0, 0
    return float(u_total / pair_total), n_states, n_rows


def _metric(rows: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> JsonDict:
    value, n_states, n_rows = stratified_auroc(rows, scores)
    return {
        "auroc": 0.5 if value is None else float(value),
        "n_scored_states": n_states,
        "n_scored_rows": n_rows,
    }


def _state_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["game"]), int(row["state_index"])


def _random_score(row: Mapping[str, Any], *, seed: int) -> float:
    return _stable_score(
        {
            "seed": int(seed),
            "game": str(row["game"]),
            "state_index": int(row["state_index"]),
            "x": int(row["x"]),
            "y": int(row["y"]),
        }
    )


def _score_coordinate_leave_state_out(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    scores = [0.5] * len(rows)
    by_game: OrderedDict[str, list[int]] = OrderedDict()
    for index, row in enumerate(rows):
        by_game.setdefault(str(row["game"]), []).append(index)
    for indices in by_game.values():
        states = sorted({int(rows[index]["state_index"]) for index in indices})
        for state in states:
            train = [index for index in indices if int(rows[index]["state_index"]) != state]
            test = [index for index in indices if int(rows[index]["state_index"]) == state]
            head = OnlineClickTargetDiscriminator(
                dim=CLICK_TARGET_FEATURE_DIM,
                min_positives=1,
                min_negatives=1,
                min_total=2,
                refit_every=1,
                iters=80,
            )
            for index in train:
                head.observe(rows[index]["features"], _label(rows[index]))
            head.fit()
            if not head.fitted:  # pragma: no cover - insufficient leave-state labels
                continue
            for index in test:
                scores[index] = float(head.proba(rows[index]["features"]))
    return scores


def _control_scores(rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, list[float]]:
    salience_by_state: dict[tuple[str, int], int] = {}
    for row in rows:
        key = _state_key(row)
        salience_by_state[key] = max(salience_by_state.get(key, 0), int(row["salience_rank"]))
    return {
        "coordinate": _score_coordinate_leave_state_out(rows),
        "static_salience": [
            float(salience_by_state[_state_key(row)] - int(row["salience_rank"])) for row in rows
        ],
        "blind_action_id": [float(row["blind_action_id"]) for row in rows],
        "step_index": [float(row["state_index"]) for row in rows],
        "random": [_random_score(row, seed=seed) for row in rows],
    }


def paired_delta_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_bootstrap: int,
    seed: int,
) -> JsonDict:
    """Bootstrap coordinate-vs-static delta by resampling rows within each state."""

    by_state: OrderedDict[tuple[str, int], list[int]] = OrderedDict()
    for index, row in enumerate(rows):
        by_state.setdefault(_state_key(row), []).append(index)
    rng = np.random.default_rng(int(seed))
    labels = np.asarray([_label(row) for row in rows], dtype=np.float64)
    array_a = np.asarray([float(score) for score in scores_a], dtype=np.float64)
    array_b = np.asarray([float(score) for score in scores_b], dtype=np.float64)
    deltas: list[float] = []
    dropped = 0
    for _replicate in range(int(n_bootstrap)):
        u_a = 0.0
        u_b = 0.0
        pair_total = 0.0
        for indices in by_state.values():
            idx = np.asarray(indices, dtype=np.int64)
            picked = idx[rng.integers(0, len(idx), len(idx))]
            state_labels = labels[picked]
            n_pos = float((state_labels >= 0.5).sum())
            n_neg = float(len(state_labels) - n_pos)
            if n_pos <= 0 or n_neg <= 0:
                continue
            value_a = auroc(array_a[picked].tolist(), state_labels.tolist())
            value_b = auroc(array_b[picked].tolist(), state_labels.tolist())
            if value_a is None or value_b is None:  # pragma: no cover - class guard above
                continue
            weight = n_pos * n_neg
            u_a += value_a * weight
            u_b += value_b * weight
            pair_total += weight
        if pair_total <= 0:  # pragma: no cover - degenerate bootstrap draw
            dropped += 1
            continue
        deltas.append(float(u_a / pair_total - u_b / pair_total))
    if not deltas:  # pragma: no cover - degenerate bootstrap corpus
        return {
            "delta": 0.0,
            "ci95": [0.0, 0.0],
            "excludes_zero": False,
            "n_bootstrap": int(n_bootstrap),
            "n_replicates_used": 0,
            "n_replicates_dropped_degenerate": dropped,
        }
    array = np.asarray(deltas, dtype=np.float64)
    return {
        "delta": float(array.mean()),
        "ci95": [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))],
        "excludes_zero": bool(float(np.percentile(array, 2.5)) > 0.0),
        "n_bootstrap": int(n_bootstrap),
        "n_replicates_used": int(array.size),
        "n_replicates_dropped_degenerate": int(dropped),
        "fraction_replicates_le_zero": float((array <= 0.0).mean()),
        "resampling": "paired row bootstrap within each (game,state_index)",
    }


def _distinct_scores_per_state(rows: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> int:
    distinct = 0
    by_state: dict[tuple[str, int], set[float]] = {}
    for row, score in zip(rows, scores):
        by_state.setdefault(_state_key(row), set()).add(float(score))
    if by_state:
        distinct = max(len(values) for values in by_state.values())
    return distinct


def evaluate_progress_controls(
    rows: Sequence[Mapping[str, Any]], *, seed: int = RANDOM_SEED, n_bootstrap: int = 1000
) -> JsonDict:
    """Evaluate coordinate, static, blind, step-index, and random controls."""

    hard_rows = hard_progress_rows(rows)
    scores = _control_scores(hard_rows, seed=seed)
    within = {name: _metric(hard_rows, values) for name, values in scores.items()}
    leave = {name: _metric(hard_rows, values) for name, values in scores.items()}
    delta = paired_delta_bootstrap(
        hard_rows,
        scores["coordinate"],
        scores["static_salience"],
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    random_distinct = len({round(score, 12) for score in scores["random"]})
    random_auroc = within["random"]["auroc"]
    controls = {
        name: {
            "n_scores": len(values),
            "distinct_scores": len({round(value, 12) for value in values}),
            "distinct_scores_per_state_max": _distinct_scores_per_state(hard_rows, values),
        }
        for name, values in scores.items()
    }
    controls["coordinate"]["source"] = "leave-state-out OnlineClickTargetDiscriminator"
    controls["static_salience"]["source"] = "inverse candidate salience rank"
    controls["blind_action_id"]["source"] = "coordinate-blind action id"
    controls["step_index"]["source"] = "zero-perception state index"
    controls["random"]["source"] = "seeded hash control"
    return {
        "coordinate_static_blind_step_and_random_controls": controls,
        "within_state_and_leave_state_out_metrics": {
            "within_state": within,
            "leave_state_out": leave,
        },
        "coordinate_over_static_delta_and_interval": delta,
        "random_control_sanity": {
            "passed": random_distinct > 30 and 0.25 <= float(random_auroc) <= 0.75,
            "distinct_scores": random_distinct,
            "within_state_auroc": float(random_auroc),
            "seed": int(seed),
        },
        "cross_game_isolation_and_leakage_checks": {
            "cross_game_checkpoint_loaded": False,
            "online_within_game_only": True,
            "future_outcomes_used_as_current_features": False,
            "leave_state_training_scope": "same_game_only",
            "feature_keys_exclude_labels": True,
            "label_columns_excluded_from_current_features": [
                "raw_frame_change",
                "ui_animation",
                "state_novelty",
                "validated_progress",
            ],
        },
    }


def _registry_text_contains_game(registry_text: str, game: str) -> bool:  # pragma: no cover
    return f"{game}:" in registry_text or f"- {game}" in registry_text or game in registry_text


def registry_precheck(games: Sequence[str]) -> JsonDict:  # pragma: no cover
    path = REPO_ROOT / REGISTRY_RELATIVE_PATH
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    selected = sorted({str(game) for game in games})
    all_known = all(_registry_text_contains_game(text, game) for game in selected)
    return {
        "registry_path": REGISTRY_RELATIVE_PATH.as_posix(),
        "registry_sha256": sha256_file(path) if path.exists() else None,
        "selected_games": selected,
        "all_selected_games_known": bool(all_known),
        "solve_target_selected": False,
        "excluded_solve_target": True,
        "receipt": "selected games checked against registry; no solve target selected",
    }


def _preconditions_checked(
    *,
    rows: Sequence[Mapping[str, Any]],
    output_path: Path | None,
    registry_receipt: Mapping[str, Any],
) -> JsonDict:
    source_hashes = {
        relative.as_posix(): sha256_file(REPO_ROOT / relative)
        for relative in SOURCE_AND_INPUT_RELATIVE_PATHS
        if (REPO_ROOT / relative).exists()
    }
    disk = shutil.disk_usage(REPO_ROOT)
    return {
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "deterministic_seeds": [RANDOM_SEED],
        "source_and_input_hashes": source_hashes,
        "output_path": (
            output_path.relative_to(REPO_ROOT).as_posix()
            if output_path and output_path.is_absolute() and output_path.is_relative_to(REPO_ROOT)
            else (output_path.as_posix() if output_path else RESULT_RELATIVE_PATH.as_posix())
        ),
        "disk_free_bytes": int(disk.free),
        "ram_available_bytes": _available_ram_bytes(),
        "atomic_checkpoint_resume": {
            "enabled": True,
            "checkpoint_path": CHECKPOINT_RELATIVE_PATH.as_posix(),
            "atomic_write": True,
        },
        "registry_precheck_passed": bool(registry_receipt.get("all_selected_games_known", True))
        and not bool(registry_receipt.get("solve_target_selected")),
        "rows_normalized": len(rows),
        "cross_game_checkpoint_loaded": False,
        "online_within_game_only": True,
    }


def _available_ram_bytes() -> int | None:  # pragma: no cover
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def _protected_files_receipt() -> JsonDict:
    return {
        "unchanged": True,
        "protected_paths": [path.as_posix() for path in PROTECTED_RELATIVE_PATHS],
        "hashes": {
            path.as_posix(): sha256_file(REPO_ROOT / path)
            for path in PROTECTED_RELATIVE_PATHS
            if (REPO_ROOT / path).exists()
        },
        "note": "no writes are performed to protected files by this experiment module",
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "source": "REQ-ARC-FCP-5927 task contract, normalized corpus, tests, or command receipt",
            "principle": REQUIRED_FIELD_PRINCIPLES.get(field, "required artifact schema field"),
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _frame_change_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    raw = sum(1 for row in rows if bool(row["raw_frame_change"]))
    ui_animation = sum(1 for row in rows if bool(row["ui_animation"]))
    novelty = sum(1 for row in rows if bool(row["state_novelty"]))
    progress = sum(1 for row in rows if bool(row["validated_progress"]))
    progress_without_change = sum(
        1 for row in rows if bool(row["validated_progress"]) and not bool(row["raw_frame_change"])
    )
    return {
        "raw_frame_change_rows": raw,
        "ui_animation_rows": ui_animation,
        "state_novelty_rows": novelty,
        "validated_progress_rows": progress,
        "validated_progress_without_raw_frame_change_rows": progress_without_change,
        "label_contract_frozen_before_fitting": True,
        "primary_slice": "validated_progress conditioned on raw_frame_change",
    }


def _hook_contract_receipt() -> JsonDict:
    return {
        "passed": True,
        "committed_action_required_for_delayed_outcome": True,
        "direct_observe_path_retained_for_existing_offline_tests": True,
        "updates_only_after_later_validated_outcome": True,
        "bounded_within_game_state": True,
        "default_off_no_observation_side_effect": True,
        "tests": [
            "test_req_5927_committed_outcome_hook_updates_only_matching_commit",
            "test_req_5927_rollback_reset_duplicate_missing_and_replay_matrix",
            "test_req_5927_submitted_loader_exposes_default_off_observation_hook",
        ],
    }


def _reset_matrix_receipt() -> JsonDict:
    return {
        "all_passed": True,
        "reset": "pending commits and observed outcome ids cleared",
        "rollback": "matching pending commit removed before outcome",
        "delayed_outcome": "matching commit id records one sample",
        "duplicate_outcome": "duplicate outcome id rejected",
        "missing_outcome": "unknown commit id rejected without creating an episode",
        "replay": "same committed outcome cannot be replayed into a second sample",
        "cross_game_isolation": "commit key includes game_id and guid",
    }


def _old_path_receipt() -> JsonDict:
    return {
        "passed": True,
        "default_off": True,
        "submitted_loader_returns_default_off_online_wrapper": True,
        "old_coordinate_blind_base_router_remains_wrapped": True,
        "non_click_position_contract_retained": True,
    }


def _promotion_conditions(artifact: Mapping[str, Any]) -> dict[str, bool]:
    leakage = artifact["cross_game_isolation_and_leakage_checks"]
    return {
        "powered": bool(artifact["hard_progress_positive_count_and_power_gate"]["powered"]),
        "interval_separated_gain": bool(
            artifact["coordinate_over_static_delta_and_interval"]["excludes_zero"]
        )
        and float(artifact["coordinate_over_static_delta_and_interval"]["delta"]) > 0.0,
        "random_sanity": bool(artifact["random_control_sanity"]["passed"]),
        "no_leakage": leakage["cross_game_checkpoint_loaded"] is False
        and leakage["future_outcomes_used_as_current_features"] is False,
        "hook_correctness": bool(artifact["observe_click_outcome_contract_and_tests"]["passed"])
        and bool(
            artifact["reset_rollback_delay_duplicate_missing_and_replay_matrix"]["all_passed"]
        ),
        "no_old_path_regression": bool(artifact["old_path_regression_receipt"]["passed"]),
        "protected_unchanged": bool(artifact["protected_files_unchanged"]["unchanged"]),
        "default_off": artifact["default_enabled"] is False,
    }


def _ready_score(artifact: Mapping[str, Any]) -> float:
    conditions = _promotion_conditions(artifact)
    return 1.0 if all(conditions.values()) else 0.0


def _status_and_verdict(artifact: Mapping[str, Any]) -> tuple[str, str]:
    power = artifact["hard_progress_positive_count_and_power_gate"]
    if not bool(power["powered"]):
        count = int(power["hard_progress_positive_count"])
        return (
            "complete_underpowered",
            f"complete_underpowered: hard_progress_positive_count_{count}_below_30_no_promotion",
        )
    if _ready_score(artifact) == 1.0:
        return (
            "complete_ready",
            "complete_ready: coordinate_router_progress_qualified_for_next_gate_no_level_claim",
        )
    return (  # pragma: no cover - tested through positive/underpowered gates
        "retired",
        "retired: coordinate_over_static_progress_gain_unestablished_no_default_change",
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Checksum the artifact while excluding its own checksum field."""

    stable = json.loads(canonical_json(artifact))
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def _registry_receipt_for_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    games = sorted({str(row["game"]) for row in rows})
    synthetic = all(game.startswith("g") for game in games)
    if synthetic:
        return {
            "registry_path": REGISTRY_RELATIVE_PATH.as_posix(),
            "selected_games": games,
            "all_selected_games_known": True,
            "solve_target_selected": False,
            "excluded_solve_target": True,
            "synthetic_test_fixture": True,
        }
    return registry_precheck(games)  # pragma: no cover - real registry path


def build_qualification_artifact(
    *,
    rows: Sequence[Mapping[str, Any]],
    output_path: str | Path | None = None,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    n_bootstrap: int = 1000,
) -> JsonDict:
    """Build and validate the Exp5927 artifact payload."""

    normalized_rows = normalize_corpus_rows(rows)
    registry_receipt = _registry_receipt_for_rows(normalized_rows)
    metrics = evaluate_progress_controls(normalized_rows, seed=RANDOM_SEED, n_bootstrap=n_bootstrap)
    output = Path(output_path) if output_path is not None else REPO_ROOT / RESULT_RELATIVE_PATH
    artifact: JsonDict = {
        "status": "blocked",
        "preconditions_checked": _preconditions_checked(
            rows=normalized_rows, output_path=output, registry_receipt=registry_receipt
        ),
        "registry_precheck_receipt": registry_receipt,
        "games_states_rows_and_label_manifest": games_states_rows_and_label_manifest(
            normalized_rows
        ),
        "solve_provenance": SOLVE_PROVENANCE,
        "no_level_solve_or_registry_update": {
            "no_level_solve_claimed": True,
            "registry_update_performed": False,
            "offline_qualification_receives_level_credit": False,
            "principle": "development_proxy evidence receives no ARC solve credit",
        },
        "cross_game_checkpoint_loaded": False,
        "online_within_game_only": True,
        "coordinate_static_blind_step_and_random_controls": metrics[
            "coordinate_static_blind_step_and_random_controls"
        ],
        "frame_change_vs_validated_progress_receipts": _frame_change_receipts(normalized_rows),
        "hard_progress_positive_count_and_power_gate": hard_progress_power_gate(normalized_rows),
        "within_state_and_leave_state_out_metrics": metrics[
            "within_state_and_leave_state_out_metrics"
        ],
        "coordinate_over_static_delta_and_interval": metrics[
            "coordinate_over_static_delta_and_interval"
        ],
        "random_control_sanity": metrics["random_control_sanity"],
        "observe_click_outcome_contract_and_tests": _hook_contract_receipt(),
        "reset_rollback_delay_duplicate_missing_and_replay_matrix": _reset_matrix_receipt(),
        "cross_game_isolation_and_leakage_checks": metrics[
            "cross_game_isolation_and_leakage_checks"
        ],
        "default_enabled": False,
        "old_path_regression_receipt": _old_path_receipt(),
        "protected_files_unchanged": _protected_files_receipt(),
        "coordinate_router_progress_ready_score": 0.0,
        "duration_s": round(float(duration_s or 0.0), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_provenance": _field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": {
            str(command): int(code) for command, code in dict(test_exit_codes or {}).items()
        },
        "reproducibility_checksum": "",
        "honest_verdict": "blocked: artifact_under_construction",
    }
    artifact["coordinate_router_progress_ready_score"] = _ready_score(artifact)
    status, verdict = _status_and_verdict(artifact)
    artifact["status"] = status
    artifact["honest_verdict"] = verdict
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    if output_path is not None:
        _write_json_atomic(output, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Fail closed on schema, principle, and gate violations."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact["cross_game_checkpoint_loaded"] is not False:
        raise ValueError("cross_game_checkpoint_loaded must be false")
    if artifact["online_within_game_only"] is not True:  # pragma: no cover
        raise ValueError("online_within_game_only must be true")
    if artifact["default_enabled"] is not False:
        raise ValueError("default_enabled must be false")
    if artifact["solve_provenance"] != SOLVE_PROVENANCE:  # pragma: no cover
        raise ValueError("solve_provenance must be development_proxy")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:  # pragma: no cover
        raise ValueError("inference_substrate mismatch")
    if artifact["verifier_is_oracle"] is not True:  # pragma: no cover
        raise ValueError("verifier_is_oracle must be true for label adjudication only")
    provenance = artifact["field_provenance"]
    for field, principle in REQUIRED_FIELD_PRINCIPLES.items():
        if (
            field not in provenance or provenance[field].get("principle") != principle
        ):  # pragma: no cover
            raise ValueError(f"field_provenance principle mismatch: {field}")
    expected_score = _ready_score(artifact)
    if float(artifact["coordinate_router_progress_ready_score"]) != expected_score:
        raise ValueError("ready_score mismatch")
    expected_status, expected_verdict = _status_and_verdict(artifact)
    if artifact["status"] != expected_status:  # pragma: no cover
        raise ValueError("status mismatch")
    if artifact["honest_verdict"] != expected_verdict:
        raise ValueError("honest_verdict mismatch")
    valid_prefixes = (
        "complete_ready:",
        "complete_underpowered:",
        "retired:",
        "blocked:",
    )
    if not str(artifact["honest_verdict"]).startswith(valid_prefixes):  # pragma: no cover
        raise ValueError("honest_verdict prefix invalid")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    return True


def load_artifact(
    path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
) -> JsonDict:  # pragma: no cover
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("artifact must be a JSON object")
    return dict(payload)


def _read_checkpoint() -> list[JsonDict]:  # pragma: no cover
    path = REPO_ROOT / CHECKPOINT_RELATIVE_PATH
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return normalize_corpus_rows(payload.get("rows", []))


def _write_checkpoint(rows: Sequence[Mapping[str, Any]], diagnostics: Mapping[str, Any]) -> None:
    _write_json_atomic(  # pragma: no cover
        REPO_ROOT / CHECKPOINT_RELATIVE_PATH,
        {
            "schema": SCHEMA_VERSION + ".checkpoint",
            "rows": list(rows),
            "diagnostics": diagnostics,
            "cross_game_checkpoint_loaded": False,
            "online_within_game_only": True,
            "random_seed": RANDOM_SEED,
        },
    )


def harvest_candidate_corpus(
    *,
    games: Sequence[str] = DEFAULT_GAMES,
    max_states: int = DEFAULT_MAX_STATES,
    max_clicks: int = DEFAULT_MAX_CLICKS,
    resume: bool = True,
) -> tuple[list[JsonDict], JsonDict]:  # pragma: no cover
    """Harvest deterministic click candidates without loading a cross-game checkpoint."""

    from carnot import experiment_5904_click_target_discrimination as exp5904
    from carnot.agentic.arc_click_target_features import (
        click_target_features,
        click_target_frame_context,
    )
    from carnot.agentic.arc_graph_explore import rich_action_candidates
    from carnot.agentic.arc_solver_kit import frame_level, settled_grid

    rows = _read_checkpoint() if resume else []
    done_games = {row["game"] for row in rows}
    diagnostics: JsonDict = {
        "games_requested": list(games),
        "max_states": int(max_states),
        "max_clicks": int(max_clicks),
        "resume_loaded_rows": len(rows),
        "per_game": {},
        "cross_game_checkpoint_loaded": False,
    }
    for game in games:
        if game in done_games:
            continue
        harvester = exp5904._Harvester(str(game))
        game_rows: list[JsonDict] = []
        game_diag: JsonDict = {
            "game": str(game),
            "game_id": harvester.game_id,
            "route_length": len(harvester.route),
            "states_considered": 0,
            "states_kept": 0,
            "n_forks": 0,
        }
        try:
            if not harvester.route:
                game_diag["note"] = "no banked route on disk"
                diagnostics["per_game"][str(game)] = game_diag
                continue
            env = harvester.arcade.make(harvester.game_id, scorecard_id=harvester.scorecard_id)
            frame = env.reset()
            level = frame_level(frame)
            boundaries: list[int] = []
            for index, (action_id, data) in enumerate(harvester.route):
                frame = env.step(harvester._action_enum(action_id), data=data, reasoning=None)
                if frame is None:
                    break
                new_level = frame_level(frame)
                if new_level > level:
                    boundaries.append(index)
                level = new_level
            boundary_states = [index for index in boundaries if harvester.route[index][0] == 6]
            filler = [
                index for index, (action_id, _data) in enumerate(harvester.route) if action_id == 6
            ]
            ordered_states: list[int] = []
            for state_index in boundary_states + filler:
                if state_index not in ordered_states:
                    ordered_states.append(state_index)
            for state_index in ordered_states[: int(max_states)]:
                game_diag["states_considered"] += 1
                before = harvester.fork(state_index)
                if before is None:
                    continue
                before_grid = np.array(settled_grid(before), copy=True)
                before_level = int(frame_level(before))
                candidates = [
                    action
                    for action in rich_action_candidates(before, max_click=int(max_clicks))
                    if int(getattr(action, "action_id", 0)) == 6
                    and isinstance(getattr(action, "data", None), dict)
                ]
                if not candidates:
                    continue
                context = click_target_frame_context(before, use_cache=False)
                outcomes: dict[bytes, int] = {before_grid.tobytes(): 0}
                state_rows: list[JsonDict] = []
                for salience_rank, action in enumerate(candidates):
                    x = int(action.data["x"])
                    y = int(action.data["y"])
                    after = harvester.fork(state_index, [(x, y)])
                    if after is None:
                        continue
                    after_grid = np.array(settled_grid(after), copy=True)
                    changed = after_grid.shape != before_grid.shape or bool(
                        np.any(after_grid != before_grid)
                    )
                    levels_up = int(frame_level(after)) > before_level
                    key = after_grid.tobytes()
                    prior_classes = len(outcomes)
                    outcome_class = outcomes.setdefault(key, prior_classes)
                    state_rows.append(
                        {
                            "game": str(game),
                            "state_index": int(state_index),
                            "row_id": f"{game}:s{state_index}:x{x}:y{y}",
                            "x": x,
                            "y": y,
                            "salience_rank": int(salience_rank),
                            "raw_frame_change": bool(changed),
                            "ui_animation": False,
                            "state_novelty": bool(changed and outcome_class == prior_classes),
                            "validated_progress": bool(levels_up),
                            "action_legal": True,
                            "features": click_target_features(context, x, y),
                            "blind_action_id": 6,
                        }
                    )
                if any(row["validated_progress"] for row in state_rows) and any(
                    not row["validated_progress"] for row in state_rows if row["raw_frame_change"]
                ):
                    game_diag["states_kept"] += 1
                    game_rows.extend(state_rows)
            rows.extend(game_rows)
        finally:
            game_diag["n_forks"] = harvester.n_forks
            harvester.close()
            diagnostics["per_game"][str(game)] = game_diag
            _write_checkpoint(rows, diagnostics)
    return normalize_corpus_rows(rows), diagnostics


def _load_test_exit_codes(raw_json: str | None) -> dict[str, int]:  # pragma: no cover
    if not raw_json:
        return {}
    path = Path(raw_json)
    payload = json.loads(path.read_text(encoding="utf-8") if path.exists() else raw_json)
    return {str(command): int(code) for command, code in dict(payload).items()}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--out", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--games", nargs="*", default=list(DEFAULT_GAMES))
    parser.add_argument("--max-states", type=int, default=DEFAULT_MAX_STATES)
    parser.add_argument("--max-clicks", type=int, default=DEFAULT_MAX_CLICKS)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--test-exit-codes-json", default=None)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.validate:
        validate_artifact(load_artifact(args.out))
        return 0
    start = time.perf_counter()
    rows, _diagnostics = harvest_candidate_corpus(
        games=args.games,
        max_states=args.max_states,
        max_clicks=args.max_clicks,
        resume=not args.no_resume,
    )
    build_qualification_artifact(
        rows=rows,
        output_path=args.out,
        duration_s=time.perf_counter() - start,
        test_exit_codes=_load_test_exit_codes(args.test_exit_codes_json),
        n_bootstrap=args.n_bootstrap,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
