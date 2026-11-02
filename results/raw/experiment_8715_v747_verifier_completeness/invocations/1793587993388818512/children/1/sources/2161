"""Exp6181 ARC logo shortcut audit.

Spec refs: REQ-ARC-WMTE-6181,
SCENARIO-ARC-WMTE-6181-SINGLE-SLOT-FIXED-POLICY-PRECONDITIONS,
SCENARIO-ARC-WMTE-6181-LABEL-CONTROLS-AND-SHORTCUT-AUDIT,
SCENARIO-ARC-WMTE-6181-NO-SOLVE-PATH-AND-REGISTRY-DELTA.

This is a robustness audit of the already-frozen Exp6167 ARC policy. It does
not collect new ARC solves, train a new inducer, inspect game source, run BFS,
or call the solver reproduction gate.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

from carnot import experiment_6167_arc_task_aware_multiseed_replication as exp6167
from carnot.agentic import arc_task_aware_energy as energy


JsonDict = dict[str, Any]
LabelTransform = Callable[[Mapping[str, Any], Sequence[str]], str]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6181_arc_logo_shortcut_audit.json")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6181_arc_logo_shortcut_audit.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6181_arc_logo_shortcut_audit.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
EXP6167_RESULT_RELATIVE_PATH = exp6167.RESULT_RELATIVE_PATH
EXP6167_MODULE_RELATIVE_PATH = exp6167.MODULE_RELATIVE_PATH
REGISTRY_RELATIVE_PATH = exp6167.REGISTRY_RELATIVE_PATH
LIVE_ENTRYPOINT_RELATIVE_PATH = exp6167.LIVE_ENTRYPOINT_RELATIVE_PATH
ADAPTER_RELATIVE_PATH = exp6167.ADAPTER_RELATIVE_PATH
SOLVER_KIT_RELATIVE_PATH = exp6167.SOLVER_KIT_RELATIVE_PATH
CALIBRATION_RELATIVE_PATH = exp6167.CALIBRATION_RELATIVE_PATH

SCHEMA = "carnot.experiment_6181.arc_logo_shortcut_audit.v1"
RUN_DATE = "20260807"
MILESTONE = "2026.08.535"
SINGLE_ARC_SLOT_ID = "exp6181-arc-logo-shortcut-audit"
INFERENCE_SUBSTRATE = exp6167.INFERENCE_SUBSTRATE
DEFAULT_GAMES = exp6167.DEFAULT_GAMES
DEFAULT_SEEDS = exp6167.DEFAULT_SEEDS
CONTROL_NAMES = (
    "known_label",
    "held_out_label",
    "shuffled_label",
    "alias",
    "unknown_label",
)
PROTECTED_FILES = exp6167.PROTECTED_FILES

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6181_arc_logo_shortcut_audit.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6181_arc_logo_shortcut_audit.py "
    "-m pytest tests/python/test_experiment_6181_arc_logo_shortcut_audit.py "
    "-q --no-cov -n 0 && "
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6181_arc_logo_shortcut_audit.py --fail-under=100"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6181_arc_logo_shortcut_audit.py"
)
RUFF_COMMAND = (
    ".venv/bin/ruff check python/carnot/experiment_6181_arc_logo_shortcut_audit.py "
    "tests/python/test_experiment_6181_arc_logo_shortcut_audit.py"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6181_arc_logo_shortcut_audit --validate"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6181_arc_logo_shortcut_audit.json"
)
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py ops/changelog.md "
    "ops/status.md _bmad/traceability.md"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    SPEC_COMMAND,
    RUFF_COMMAND,
    VALIDATE_COMMAND,
    ADVERSARIAL_COMMAND,
    PROTECTED_FILE_COMMAND,
    ROOT_CLUTTER_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "single_arc_slot_receipt",
    "fixed_exp6167_policy_freeze",
    "adapter_disabled_live_path_receipt",
    "live_attempt_label_dataset",
    "leave_one_game_out_controls",
    "label_control_results",
    "shortcut_audit_summary",
    "no_source_bfs_solver_kit_path_receipt",
    "solve_claimed",
    "level_credit_delta",
    "registry_delta",
    "registry_levels_unchanged",
    "provenance",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "terminal audit state; complete_no_shortcut_detected or blocked names whether the fixed-policy shortcut audit is valid.",
    "preconditions_checked": "registry, Exp6167 hashes, task labels, live runtime paths, protected files, root clutter, and protected git status are snapshotted before output.",
    "single_arc_slot_receipt": "proves Exp6181 is the only `.535` ARC slot being claimed by this artifact.",
    "fixed_exp6167_policy_freeze": "content-addresses the inherited Exp6167 policy and proves no held-control refit occurred.",
    "adapter_disabled_live_path_receipt": "carries the Exp6167 adapter-disabled live E3 path receipts.",
    "live_attempt_label_dataset": "exposes the game labels and abstracted label rows derived only from Exp6167 live agent attempts.",
    "leave_one_game_out_controls": "each game is held out exactly once and scored with the same fixed policy.",
    "label_control_results": "known-label, held-out-label, shuffled-label, alias, and unknown-label relabelings must leave decisions invariant.",
    "shortcut_audit_summary": "summarizes whether any label or logo shortcut changed the fixed-policy decisions.",
    "no_source_bfs_solver_kit_path_receipt": "proves source reads, offline BFS, solver-kit reproduction, and adapter routes were not used.",
    "solve_claimed": "bare false; robustness audits never claim public solve credit.",
    "level_credit_delta": "bare 0; known public level totals do not move.",
    "registry_delta": "bare false; no registry update is proposed.",
    "registry_levels_unchanged": "bare true; level fingerprint before and after the audit matches.",
    "provenance": "names Exp6167 as the frozen upstream and live self-discovery evidence source.",
    "protected_files_unchanged": "conductor, ops status/changelog, and traceability files remain byte-identical.",
    "duration_s": "wall-clock duration of the deterministic no-LLM audit.",
    "inference_substrate": "live_e3_adapter_disabled_runtime_transitions declares adapter-disabled no-LLM ARC live-path evidence.",
    "field_provenance": "every required field traces to spec, Exp6167 hashes, live attempts, controls, path guards, or command receipts.",
    "test_commands": "records focused, coverage, spec, validation, adversarial, protected-file, root-clutter, and full-suite checks.",
    "test_exit_codes": "records verification exit codes without implying unrun checks passed.",
    "reproducibility_checksum": "content-addressed checksum detects later audit drift.",
    "honest_verdict": "complete_no_shortcut_detected: or blocked: verdict with no solve or registry delta.",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    return exp6167.sha256_file(path)


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = dict(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def load_exp6167_artifact(root: Path = REPO_ROOT) -> JsonDict:
    return json.loads((root / EXP6167_RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))


def _load_registry(root: Path) -> JsonDict:
    return exp6167._load_yaml(root / REGISTRY_RELATIVE_PATH)


def _registry_level_fingerprint(root: Path) -> JsonDict:
    return exp6167._registry_level_fingerprint(_load_registry(root))


def _file_receipt(root: Path, relative: Path) -> JsonDict:
    path = root / relative
    return {
        "path": relative.as_posix(),
        "exists": path.exists(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
    }


def _protected_hashes(root: Path) -> dict[str, str]:
    return exp6167._protected_hashes(root)


def _protected_git_status_short(root: Path) -> list[str]:
    args = ["git", "status", "--short", "--", *(path.as_posix() for path in PROTECTED_FILES)]
    try:
        result = subprocess.run(
            args,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - git fallback.
        return ["git_status_unavailable"]
    return [line for line in result.stdout.splitlines() if line.strip()]


def _root_clutter_state(root: Path) -> JsonDict:
    return exp6167._root_clutter_state(root)


def fixed_task_aware_manifest(exp6167_artifact: Mapping[str, Any]) -> JsonDict:
    return dict(
        dict(exp6167_artifact.get("global_and_task_aware_freeze_manifests") or {}).get(
            "task_aware_fixed"
        )
        or {}
    )


def preconditions_checked(
    *,
    root: Path,
    result_path: Path,
    exp6167_artifact: Mapping[str, Any],
) -> JsonDict:
    registry = _registry_level_fingerprint(root)
    task_labels = {
        "games": list(DEFAULT_GAMES),
        "seeds": [int(seed) for seed in DEFAULT_SEEDS],
        "aliases": {game: f"arc-logo-alias::{game}" for game in DEFAULT_GAMES},
        "unknown_label": "unknown_arc_game",
        "label_set_sha256": sha256_json(list(DEFAULT_GAMES)),
    }
    exp6167_inputs = [
        EXP6167_RESULT_RELATIVE_PATH,
        EXP6167_MODULE_RELATIVE_PATH,
        exp6167.TEST_RELATIVE_PATH,
        CALIBRATION_RELATIVE_PATH,
        SPEC_RELATIVE_PATH,
    ]
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "registry_snapshot": {
            "path": REGISTRY_RELATIVE_PATH.as_posix(),
            "sha256": sha256_file(root / REGISTRY_RELATIVE_PATH),
            "level_fingerprint": registry,
            "level_fingerprint_sha256": sha256_json(registry),
        },
        "exp6167_hashes": [_file_receipt(root, relative) for relative in exp6167_inputs],
        "exp6167_policy_freeze_hash": dict(
            exp6167_artifact.get("global_and_task_aware_freeze_manifests") or {}
        ).get("freeze_hash"),
        "task_labels": task_labels,
        "live_runtime_paths": {
            "live_entrypoint": LIVE_ENTRYPOINT_RELATIVE_PATH.as_posix(),
            "calibration": CALIBRATION_RELATIVE_PATH.as_posix(),
            "adapter": ADAPTER_RELATIVE_PATH.as_posix(),
            "solver_kit": SOLVER_KIT_RELATIVE_PATH.as_posix(),
        },
        "output_path": {
            "path": str(result_path),
            "parent_exists": result_path.parent.exists(),
            "existed_before": result_path.exists(),
            "sha256_before": sha256_file(result_path),
        },
        "protected_file_hashes_before": _protected_hashes(root),
        "protected_git_status_short": _protected_git_status_short(root),
        "root_clutter": _root_clutter_state(root),
        "principle": FIELD_PRINCIPLES["preconditions_checked"],
    }


def single_arc_slot_receipt() -> JsonDict:
    return {
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "slot_id": SINGLE_ARC_SLOT_ID,
        "track": "arc",
        "slot_count_claimed": 1,
        "only_arc_slot_for_v535": True,
        "no_other_arc_slot_claimed_by_this_artifact": True,
        "principle": FIELD_PRINCIPLES["single_arc_slot_receipt"],
    }


def fixed_exp6167_policy_freeze(root: Path, exp6167_artifact: Mapping[str, Any]) -> JsonDict:
    manifests = dict(exp6167_artifact.get("global_and_task_aware_freeze_manifests") or {})
    task_manifest = dict(manifests.get("task_aware_fixed") or {})
    return {
        "exp6167_result_path": EXP6167_RESULT_RELATIVE_PATH.as_posix(),
        "exp6167_result_sha256": sha256_file(root / EXP6167_RESULT_RELATIVE_PATH),
        "exp6167_module_path": EXP6167_MODULE_RELATIVE_PATH.as_posix(),
        "exp6167_module_sha256": sha256_file(root / EXP6167_MODULE_RELATIVE_PATH),
        "policy_freeze_hash": manifests.get("freeze_hash"),
        "task_aware_manifest_hash": task_manifest.get("manifest_hash"),
        "task_aware_min_changed_cells": int(task_manifest.get("min_changed_cells") or 1),
        "fixed_before_controls": bool(manifests.get("frozen_before_episode_collection")),
        "held_control_refit_count": 0,
        "current_held_rows_used_for_fit": int(
            task_manifest.get("current_held_row_count_used_for_fit") or 0
        ),
        "uses_game_source": bool(task_manifest.get("uses_game_source")),
        "uses_offline_bfs": bool(task_manifest.get("uses_offline_bfs")),
        "principle": FIELD_PRINCIPLES["fixed_exp6167_policy_freeze"],
    }


def adapter_disabled_live_path_receipt(exp6167_artifact: Mapping[str, Any]) -> JsonDict:
    disable = dict(
        exp6167_artifact.get(
            "adapter_per_game_lookup_solver_gotcha_and_hand_calibration_disable_receipts"
        )
        or {}
    )
    reachability = dict(exp6167_artifact.get("live_entrypoint_and_import_reachability") or {})
    provenance = dict(exp6167_artifact.get("own_attempt_transition_provenance") or {})
    return {
        "adapter_disabled": disable.get("adapter_disabled") is True,
        "per_game_lookup_routes_disabled": disable.get("per_game_lookup_routes_disabled") is True,
        "solver_routes_disabled": disable.get("solver_routes_disabled") is True,
        "registry_gotcha_calibration_disabled": (
            disable.get("registry_gotcha_calibration_disabled") is True
        ),
        "hand_calibration_disabled": disable.get("hand_calibration_disabled") is True,
        "game_source_read_count": int(disable.get("game_source_read_count") or 0),
        "offline_ground_truth_bfs_run_count": int(
            disable.get("offline_ground_truth_bfs_run_count") or 0
        ),
        "llm_induction_disabled": disable.get("llm_induction_disabled") is True,
        "llm_invocation_count": int(exp6167_artifact.get("llm_invocation_count") or 0),
        "make_carnot_agent_constructed": reachability.get("make_carnot_agent_constructed") is True,
        "e3_policy_seen": reachability.get("e3_policy_seen") is True,
        "all_rows_live_agent_owned": provenance.get("all_rows_live_agent_owned") is True,
        "principle": FIELD_PRINCIPLES["adapter_disabled_live_path_receipt"],
    }


def abstract_live_attempt_rows(exp6167_artifact: Mapping[str, Any]) -> list[JsonDict]:
    metrics = dict(
        exp6167_artifact.get(
            "per_game_seed_transition_change_recall_safety_action_and_latency_metrics"
        )
        or {}
    )
    rows: list[JsonDict] = []
    for game in DEFAULT_GAMES:
        for seed in DEFAULT_SEEDS:
            arms = dict(dict(metrics.get(game) or {}).get(str(seed)) or {})
            task_metrics = dict(arms.get("task_aware") or {})
            transition_count = int(task_metrics.get("transition_count") or 0)
            changed_count = int(task_metrics.get("changed_row_count") or 0)
            for action_index in range(transition_count):
                changed = action_index < changed_count
                rows.append(
                    {
                        "row_id": f"{game}|{seed}|abstract|{action_index}",
                        "original_game": game,
                        "game": game,
                        "seed": int(seed),
                        "action_index": int(action_index),
                        "changed_cell_count": 1 if changed else 0,
                        "frame_changed": bool(changed),
                        "source": "exp6167_live_agent_self_discovery_aggregate",
                        "live_agent_self_discovery_attempt": True,
                    }
                )
    return rows


def live_attempt_label_dataset(exp6167_artifact: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_game = Counter(str(row.get("original_game")) for row in rows)
    return {
        "source": "exp6167_live_agent_self_discovery_aggregate",
        "upstream_row_ids_sha256": dict(
            exp6167_artifact.get("own_attempt_transition_provenance") or {}
        ).get("row_ids_sha256"),
        "row_count": len(rows),
        "games": list(DEFAULT_GAMES),
        "seeds": [int(seed) for seed in DEFAULT_SEEDS],
        "rows_by_game": {game: int(by_game[game]) for game in DEFAULT_GAMES},
        "all_rows_live_agent_self_discovery": all(
            row.get("live_agent_self_discovery_attempt") is True for row in rows
        ),
        "raw_game_source_materialized": False,
        "abstracted_from_metric_denominators_only": True,
        "principle": FIELD_PRINCIPLES["live_attempt_label_dataset"],
    }


def known_label(row: Mapping[str, Any], _games: Sequence[str]) -> str:
    return str(row.get("original_game") or row.get("game"))


def held_out_label(row: Mapping[str, Any], _games: Sequence[str]) -> str:
    return f"held_out::{row.get('original_game') or row.get('game')}"


def shuffled_label(row: Mapping[str, Any], games: Sequence[str]) -> str:
    original = str(row.get("original_game") or row.get("game"))
    index = list(games).index(original)
    return str(games[(index + 1) % len(games)])


def alias_label(row: Mapping[str, Any], _games: Sequence[str]) -> str:
    return f"arc-logo-alias::{row.get('original_game') or row.get('game')}"


def unknown_label(_row: Mapping[str, Any], _games: Sequence[str]) -> str:
    return "unknown_arc_game"


def _decision_outcomes(decisions: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [
        {
            "row_id": row.get("row_id"),
            "arm": row.get("arm"),
            "admitted": row.get("admitted"),
            "abstained": row.get("abstained"),
            "frame_changed": row.get("frame_changed"),
            "false_confident_admission": row.get("false_confident_admission"),
            "safe_abstention": row.get("safe_abstention"),
        }
        for row in decisions
    ]


def score_control(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    label_transform: LabelTransform,
) -> JsonDict:
    decisions: list[JsonDict] = []
    games = list(DEFAULT_GAMES)
    for row in rows:
        relabeled = dict(row)
        relabeled["game"] = label_transform(row, games)
        decisions.append(energy.score_transition(relabeled, energy.global_freeze_manifest(), arm="global"))
        decisions.append(energy.score_transition(relabeled, manifest, arm="task_aware"))
    outcomes = _decision_outcomes(decisions)
    by_arm = Counter(str(row.get("arm")) for row in decisions if row.get("admitted"))
    return {
        "control": label_transform.__name__.replace("_label", "_label_control"),
        "decision_count": len(decisions),
        "decision_signature_sha256": sha256_json(outcomes),
        "admitted_by_arm": {arm: int(by_arm[arm]) for arm in exp6167.DECISION_ARMS},
        "decision_outcomes": outcomes,
    }


def _changed_decision_count(
    baseline: Mapping[str, Any],
    control: Mapping[str, Any],
) -> int:
    base = list(baseline.get("decision_outcomes") or [])
    other = list(control.get("decision_outcomes") or [])
    return sum(1 for left, right in zip(base, other, strict=True) if left != right)


def _public_control_result(
    *,
    name: str,
    baseline: Mapping[str, Any],
    control: Mapping[str, Any],
    label_feature_visible: bool,
) -> JsonDict:
    changed_count = _changed_decision_count(baseline, control)
    return {
        "control": name,
        "label_feature_visible_to_row_metadata": label_feature_visible,
        "decision_count": int(control.get("decision_count") or 0),
        "baseline_decision_signature_sha256": baseline.get("decision_signature_sha256"),
        "decision_signature_sha256": control.get("decision_signature_sha256"),
        "changed_decision_count": changed_count,
        "passed": changed_count == 0
        and control.get("decision_signature_sha256") == baseline.get("decision_signature_sha256"),
    }


def label_control_results(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
) -> JsonDict:
    baseline = score_control(rows, manifest=manifest, label_transform=known_label)
    transforms: dict[str, tuple[LabelTransform, bool]] = {
        "known_label": (known_label, True),
        "held_out_label": (held_out_label, True),
        "shuffled_label": (shuffled_label, True),
        "alias": (alias_label, True),
        "unknown_label": (unknown_label, False),
    }
    return {
        name: _public_control_result(
            name=name,
            baseline=baseline,
            control=score_control(rows, manifest=manifest, label_transform=transform),
            label_feature_visible=visible,
        )
        for name, (transform, visible) in transforms.items()
    }


def leave_one_game_out_controls(
    rows: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
) -> JsonDict:
    folds: JsonDict = {}
    for held_game in DEFAULT_GAMES:
        held_rows = [row for row in rows if row.get("original_game") == held_game]
        known = score_control(held_rows, manifest=manifest, label_transform=known_label)
        held = score_control(held_rows, manifest=manifest, label_transform=held_out_label)
        folds[held_game] = {
            "held_game": held_game,
            "training_games": [game for game in DEFAULT_GAMES if game != held_game],
            "held_row_count": len(held_rows),
            "policy_refit_count": 0,
            "known_label_signature_sha256": known.get("decision_signature_sha256"),
            "held_out_label_signature_sha256": held.get("decision_signature_sha256"),
            "changed_decision_count": _changed_decision_count(known, held),
            "invariant": known.get("decision_signature_sha256")
            == held.get("decision_signature_sha256"),
        }
    return {
        "fold_count": len(folds),
        "folds": folds,
        "all_games_held_once": sorted(folds) == sorted(DEFAULT_GAMES),
        "policy_refit_count_total": sum(int(row["policy_refit_count"]) for row in folds.values()),
        "all_folds_invariant": all(row["invariant"] for row in folds.values()),
        "principle": FIELD_PRINCIPLES["leave_one_game_out_controls"],
    }


def _module_imports(root: Path, relative: Path) -> list[str]:
    tree = ast.parse((root / relative).read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return sorted(names)


def no_source_bfs_solver_kit_path_receipt(
    *,
    root: Path,
    exp6167_artifact: Mapping[str, Any],
) -> JsonDict:
    disable = adapter_disabled_live_path_receipt(exp6167_artifact)
    imports = _module_imports(root, MODULE_RELATIVE_PATH)
    source_read_used = int(disable["game_source_read_count"]) != 0
    bfs_used = int(disable["offline_ground_truth_bfs_run_count"]) != 0
    solver_imported = any(name.endswith("arc_solver_kit") for name in imports)
    receipt = {
        "source_read_used": source_read_used,
        "offline_ground_truth_bfs_run": bfs_used,
        "solver_kit_reproduce_called": False,
        "solver_kit_imported_by_exp6181_module": solver_imported,
        "adapter_route_used": not bool(disable["adapter_disabled"]),
        "exp6167_solver_routes_disabled": bool(disable["solver_routes_disabled"]),
        "exp6167_per_game_lookup_routes_disabled": bool(disable["per_game_lookup_routes_disabled"]),
        "exp6167_adapter_disabled": bool(disable["adapter_disabled"]),
        "inspected_module_imports": imports,
        "principle": FIELD_PRINCIPLES["no_source_bfs_solver_kit_path_receipt"],
    }
    receipt["proves_no_source_bfs_solver_path"] = (
        receipt["source_read_used"] is False
        and receipt["offline_ground_truth_bfs_run"] is False
        and receipt["solver_kit_reproduce_called"] is False
        and receipt["solver_kit_imported_by_exp6181_module"] is False
        and receipt["adapter_route_used"] is False
        and receipt["exp6167_solver_routes_disabled"] is True
        and receipt["exp6167_per_game_lookup_routes_disabled"] is True
        and receipt["exp6167_adapter_disabled"] is True
    )
    return receipt


def shortcut_audit_summary(artifact: Mapping[str, Any]) -> JsonDict:
    label_controls = dict(artifact.get("label_control_results") or {})
    baseline = None
    if label_controls:
        baseline = dict(next(iter(label_controls.values()))).get(
            "baseline_decision_signature_sha256"
        )
    label_passed = set(label_controls) == set(CONTROL_NAMES) and all(
        dict(row).get("passed") is True for row in label_controls.values()
    )
    loo = dict(artifact.get("leave_one_game_out_controls") or {})
    path = dict(artifact.get("no_source_bfs_solver_kit_path_receipt") or {})
    slot = dict(artifact.get("single_arc_slot_receipt") or {})
    all_controls = (
        label_passed
        and loo.get("all_folds_invariant") is True
        and int(loo.get("policy_refit_count_total") or 0) == 0
        and path.get("proves_no_source_bfs_solver_path") is True
        and slot.get("only_arc_slot_for_v535") is True
        and int(slot.get("slot_count_claimed") or 0) == 1
    )
    return {
        "baseline_decision_signature_sha256": baseline,
        "label_controls_passed": label_passed,
        "leave_one_game_out_passed": loo.get("all_folds_invariant") is True,
        "no_path_shortcut_passed": path.get("proves_no_source_bfs_solver_path") is True,
        "single_slot_passed": slot.get("only_arc_slot_for_v535") is True
        and int(slot.get("slot_count_claimed") or 0) == 1,
        "all_controls_passed": all_controls,
        "shortcut_detected": not all_controls,
        "solve_or_registry_credit_claimed": bool(artifact.get("solve_claimed"))
        or int(artifact.get("level_credit_delta") or 0) != 0
        or artifact.get("registry_delta") not in (0, False),
        "principle": FIELD_PRINCIPLES["shortcut_audit_summary"],
    }


def protected_files_unchanged(root: Path, before: Mapping[str, str]) -> JsonDict:
    after = _protected_hashes(root)
    return {
        "before": dict(before),
        "after": after,
        "unchanged": dict(before) == after,
        "principle": FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def provenance(exp6167_artifact: Mapping[str, Any]) -> JsonDict:
    upstream = dict(exp6167_artifact.get("own_attempt_transition_provenance") or {})
    return {
        "upstream_experiment": "experiment_6167_arc_task_aware_multiseed_replication",
        "upstream_result": EXP6167_RESULT_RELATIVE_PATH.as_posix(),
        "upstream_live_row_count": int(upstream.get("scored_row_count") or 0),
        "upstream_row_ids_sha256": upstream.get("row_ids_sha256"),
        "evidence_source": "Exp6167 live agent self-discovery attempts",
        "new_induction_mechanism_added": False,
        "public_level_solve_attempted": False,
        "principle": FIELD_PRINCIPLES["provenance"],
    }


def field_provenance() -> dict[str, dict[str, str]]:
    return {
        field: {
            "source": "experiment_6181_arc_logo_shortcut_audit",
            "principle": FIELD_PRINCIPLES[field],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not dict(artifact.get("preconditions_checked") or {}).get("root_clutter", {}).get("ok"):
        reasons.append("root_clutter")
    slot = dict(artifact.get("single_arc_slot_receipt") or {})
    if int(slot.get("slot_count_claimed") or 0) != 1 or slot.get("only_arc_slot_for_v535") is not True:
        reasons.append("single_arc_slot_receipt")
    fixed = dict(artifact.get("fixed_exp6167_policy_freeze") or {})
    if int(fixed.get("held_control_refit_count") or 0) != 0:
        reasons.append("fixed_exp6167_policy_freeze")
    live = dict(artifact.get("adapter_disabled_live_path_receipt") or {})
    if live.get("adapter_disabled") is not True or live.get("all_rows_live_agent_owned") is not True:
        reasons.append("adapter_disabled_live_path_receipt")
    dataset = dict(artifact.get("live_attempt_label_dataset") or {})
    if dataset.get("all_rows_live_agent_self_discovery") is not True:
        reasons.append("live_attempt_label_dataset")
    loo = dict(artifact.get("leave_one_game_out_controls") or {})
    if loo.get("all_folds_invariant") is not True or int(loo.get("policy_refit_count_total") or 0) != 0:
        reasons.append("leave_one_game_out_controls")
    controls = dict(artifact.get("label_control_results") or {})
    if set(controls) != set(CONTROL_NAMES) or any(
        dict(control).get("passed") is not True for control in controls.values()
    ):
        reasons.append("label_control_results")
    path = dict(artifact.get("no_source_bfs_solver_kit_path_receipt") or {})
    if path.get("proves_no_source_bfs_solver_path") is not True:
        reasons.append("no_source_bfs_solver_kit_path_receipt")
    if artifact.get("solve_claimed") is not False:
        reasons.append("solve_claimed")
    if int(artifact.get("level_credit_delta") or 0) != 0:
        reasons.append("level_credit_delta")
    if artifact.get("registry_delta") not in (0, False):
        reasons.append("registry_delta")
    if artifact.get("registry_levels_unchanged") is not True:
        reasons.append("registry_levels_unchanged")
    if not dict(artifact.get("protected_files_unchanged") or {}).get("unchanged"):
        reasons.append("protected_files_unchanged")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        reasons.append("inference_substrate")
    return reasons


def status(artifact: Mapping[str, Any]) -> str:
    return "blocked" if _blocked_reasons(artifact) else "complete_no_shortcut_detected"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    if status(artifact) == "complete_no_shortcut_detected":
        return (
            "complete_no_shortcut_detected: fixed_exp6167_policy_label_controls_"
            "invariant_no_solve_no_registry_delta"
        )
    reasons = "_".join(_blocked_reasons(artifact)[:4]) or "audit_blocked"
    return f"blocked: {reasons}_no_solve_no_registry_delta"


def run(
    *,
    result_path: Path | None = None,
    root: Path = REPO_ROOT,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    started = time.perf_counter()
    out_path = result_path or (root / RESULT_RELATIVE_PATH)
    exp6167_artifact = load_exp6167_artifact(root)
    registry_before = _registry_level_fingerprint(root)
    pre = preconditions_checked(root=root, result_path=out_path, exp6167_artifact=exp6167_artifact)
    rows = abstract_live_attempt_rows(exp6167_artifact)
    manifest = fixed_task_aware_manifest(exp6167_artifact)
    protected = protected_files_unchanged(root, dict(pre.get("protected_file_hashes_before") or {}))
    registry_after = _registry_level_fingerprint(root)
    artifact: JsonDict = {
        "status": "",
        "preconditions_checked": pre,
        "single_arc_slot_receipt": single_arc_slot_receipt(),
        "fixed_exp6167_policy_freeze": fixed_exp6167_policy_freeze(root, exp6167_artifact),
        "adapter_disabled_live_path_receipt": adapter_disabled_live_path_receipt(exp6167_artifact),
        "live_attempt_label_dataset": live_attempt_label_dataset(exp6167_artifact, rows),
        "leave_one_game_out_controls": leave_one_game_out_controls(rows, manifest=manifest),
        "label_control_results": label_control_results(rows, manifest=manifest),
        "shortcut_audit_summary": {},
        "no_source_bfs_solver_kit_path_receipt": no_source_bfs_solver_kit_path_receipt(
            root=root, exp6167_artifact=exp6167_artifact
        ),
        "solve_claimed": False,
        "level_credit_delta": 0,
        "registry_delta": False,
        "registry_levels_unchanged": registry_before == registry_after,
        "offline_reproduced": False,
        "provenance": provenance(exp6167_artifact),
        "protected_files_unchanged": protected,
        "duration_s": round(
            float(duration_s if duration_s is not None else time.perf_counter() - started),
            6,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": {
            str(command): int(code) for command, code in dict(test_exit_codes or {}).items()
        },
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["shortcut_audit_summary"] = shortcut_audit_summary(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    if write:
        _write_atomic_json(out_path, artifact)
    return artifact


def _write_atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:  # pragma: no cover - schema guard.
        raise ValueError(f"missing required fields: {missing}")
    if set(artifact.get("field_provenance", {})) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_provenance")  # pragma: no cover - schema guard.
    if artifact.get("solve_claimed") is not False:
        raise ValueError("solve_claimed")
    if int(artifact.get("level_credit_delta") or 0) != 0:
        raise ValueError("level_credit_delta")
    if artifact.get("registry_delta") not in (0, False):
        raise ValueError("registry_delta")
    if artifact.get("offline_reproduced", False) is not False:
        raise ValueError("offline_reproduced")  # pragma: no cover - no-solve guard.
    if artifact.get("registry_levels_unchanged") is not True:
        raise ValueError("registry_levels_unchanged")
    slot = dict(artifact.get("single_arc_slot_receipt") or {})
    if int(slot.get("slot_count_claimed") or 0) != 1 or slot.get("only_arc_slot_for_v535") is not True:
        raise ValueError("single_arc_slot_receipt")
    if any(
        dict(control).get("passed") is not True
        for control in dict(artifact.get("label_control_results") or {}).values()
    ):
        raise ValueError("label_control_results")
    path = dict(artifact.get("no_source_bfs_solver_kit_path_receipt") or {})
    if path.get("solver_kit_reproduce_called") is not False:
        raise ValueError("solver_kit_reproduce_called")
    for field in (
        "source_read_used",
        "offline_ground_truth_bfs_run",
        "solver_kit_imported_by_exp6181_module",
        "adapter_route_used",
    ):
        if path.get(field) is not False:
            raise ValueError(field)
    if path.get("proves_no_source_bfs_solver_path") is not True:
        raise ValueError("no_source_bfs_solver_kit_path_receipt")
    if artifact.get("shortcut_audit_summary") != shortcut_audit_summary(artifact):
        raise ValueError("shortcut_audit_summary")  # pragma: no cover - recomputed in run.
    if artifact.get("status") != status(artifact):
        raise ValueError("status")  # pragma: no cover - recomputed in run.
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")  # pragma: no cover - recomputed in run.
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        validate_artifact(json.loads((REPO_ROOT / RESULT_RELATIVE_PATH).read_text(encoding="utf-8")))
        print(RESULT_RELATIVE_PATH.as_posix())
        return 0
    run(write=True)
    print(RESULT_RELATIVE_PATH.as_posix())
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
